"""On-device image classification with model compression — pipeline CLI.

Stages (run individually or via `all`):
    teacher    train the ResNet teacher
    student    train the MobileNetV3-Small student from scratch (baseline)
    distill    train the student with knowledge distillation from the teacher
    prune      magnitude-prune the distilled student and fine-tune
    export     export a checkpoint to ONNX / TFLite
    benchmark  evaluate every stage (incl. INT8 quantization) -> table + plot
    all        teacher -> student -> distill -> prune -> benchmark -> export

Examples:
    python main.py all --teacher-epochs 40 --epochs 40 --finetune-epochs 10
    python main.py benchmark
    python main.py export --ckpt checkpoints/student_pruned.pt --int8
"""
from __future__ import annotations

import argparse
import os

import torch

from src import benchmark as bench
from src.data import get_calibration_loader, get_loaders
from src.distill import train_distill
from src.engine import fit
from src.export import export_onnx, export_tflite
from src.models import build_student, build_teacher
from src.prune import finalize_pruning, prune_model
from src.quantize import quantize_model
from src.utils import get_device, load_checkpoint, set_seed

CKPT = {
    "teacher": "checkpoints/teacher.pt",
    "baseline": "checkpoints/student_baseline.pt",
    "distilled": "checkpoints/student_distilled.pt",
    "pruned": "checkpoints/student_pruned.pt",
}


# --------------------------------------------------------------------------- helpers
def _loaders(args):
    return get_loaders(args.data_dir, args.batch_size, args.num_workers)


def _load_into(model, path, device):
    load_checkpoint(model, path, map_location=device)
    return model.to(device)


# --------------------------------------------------------------------------- stages
def cmd_teacher(args, device):
    train_loader, test_loader = _loaders(args)
    model = build_teacher(args.teacher_arch)
    res = fit(model, train_loader, test_loader, device, epochs=args.teacher_epochs,
              lr=args.lr, label=f"teacher:{args.teacher_arch}", checkpoint_path=CKPT["teacher"])
    print(f"Teacher best top-1: {res['best_top1']:.2f}%  -> {CKPT['teacher']}")


def cmd_student(args, device):
    train_loader, test_loader = _loaders(args)
    model = build_student(args.student_arch)
    res = fit(model, train_loader, test_loader, device, epochs=args.epochs,
              lr=args.lr, label=f"student:{args.student_arch}", checkpoint_path=CKPT["baseline"])
    print(f"Student baseline best top-1: {res['best_top1']:.2f}%  -> {CKPT['baseline']}")


def cmd_distill(args, device):
    train_loader, test_loader = _loaders(args)
    teacher = _load_into(build_teacher(args.teacher_arch), CKPT["teacher"], device)
    student = build_student(args.student_arch)
    res = train_distill(student, teacher, train_loader, test_loader, device,
                        epochs=args.epochs, lr=args.distill_lr, temperature=args.temperature,
                        alpha=args.alpha, checkpoint_path=CKPT["distilled"])
    print(f"Distilled student best top-1: {res['best_top1']:.2f}%  -> {CKPT['distilled']}")


def cmd_prune(args, device):
    train_loader, test_loader = _loaders(args)
    student = _load_into(build_student(args.student_arch),
                         args.ckpt or CKPT["distilled"], device)
    teacher = _load_into(build_teacher(args.teacher_arch), CKPT["teacher"], device)

    prune_model(student, amount=args.amount, structured=args.structured)
    print(f"Pruned {args.amount:.0%} of Conv/Linear weights "
          f"({'structured' if args.structured else 'unstructured'}); fine-tuning...")
    train_distill(student, teacher, train_loader, test_loader, device,
                  epochs=args.finetune_epochs, lr=args.finetune_lr,
                  temperature=args.temperature, alpha=args.alpha, label="prune-ft")
    finalize_pruning(student)
    from src.utils import save_checkpoint, global_sparsity
    save_checkpoint(student, CKPT["pruned"], sparsity=global_sparsity(student))
    print(f"Pruned student sparsity={global_sparsity(student):.1%}  -> {CKPT['pruned']}")


def cmd_export(args, device):
    student = build_student(args.student_arch)
    load_checkpoint(student, args.ckpt or CKPT["pruned"], map_location="cpu")
    stem = os.path.join("exports", os.path.splitext(os.path.basename(args.ckpt or CKPT["pruned"]))[0])
    export_onnx(student, stem + ".onnx")
    calib = get_calibration_loader(args.data_dir) if args.int8 else None
    export_tflite(student, stem + ".tflite", int8=args.int8, calib_loader=calib)


def cmd_benchmark(args, device):
    _, test_loader = _loaders(args)
    calib = get_calibration_loader(args.data_dir)
    rows = []

    specs = [
        ("Teacher (ResNet)", build_teacher, args.teacher_arch, CKPT["teacher"]),
        ("Student baseline", build_student, args.student_arch, CKPT["baseline"]),
        ("Student + KD", build_student, args.student_arch, CKPT["distilled"]),
        ("Student + KD + Prune", build_student, args.student_arch, CKPT["pruned"]),
    ]
    pruned_student = None
    for label, builder, arch, path in specs:
        if not os.path.exists(path):
            print(f"[benchmark] skipping '{label}' (missing {path})")
            continue
        model = builder(arch)
        load_checkpoint(model, path, map_location="cpu")
        rows.append(bench.benchmark_model(model, label, test_loader,
                                          acc_device=device, lat_device="cpu"))
        if path == CKPT["pruned"]:
            pruned_student = model

    # INT8 quantization of the most-compressed available student.
    base = pruned_student
    if base is None:
        for path in (CKPT["distilled"], CKPT["baseline"]):
            if os.path.exists(path):
                base = build_student(args.student_arch)
                load_checkpoint(base, path, map_location="cpu")
                break
    if base is not None:
        qmodel, method = quantize_model(base, calib)
        rows.append(bench.benchmark_model(qmodel, f"Student + KD + Prune + INT8 ({method})",
                                          test_loader, acc_device="cpu", lat_device="cpu"))

    if not rows:
        print("No checkpoints found. Train something first (e.g. `python main.py all`).")
        return
    print()
    bench.print_table(rows)
    bench.save_results(rows)
    print("\nSaved results/benchmark.json and results/tradeoff.png")


def cmd_all(args, device):
    cmd_teacher(args, device)
    cmd_student(args, device)
    cmd_distill(args, device)
    cmd_prune(args, device)
    cmd_benchmark(args, device)
    args.ckpt = CKPT["pruned"]
    cmd_export(args, device)


# --------------------------------------------------------------------------- CLI
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["teacher", "student", "distill", "prune",
                                      "export", "benchmark", "all"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    # architectures
    p.add_argument("--teacher-arch", default="resnet18")
    p.add_argument("--student-arch", default="mobilenet_v3_small")
    # training
    p.add_argument("--epochs", type=int, default=40, help="student / distill epochs")
    p.add_argument("--teacher-epochs", type=int, default=40)
    p.add_argument("--finetune-epochs", type=int, default=10, help="post-prune fine-tune")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--distill-lr", type=float, default=0.05)
    p.add_argument("--finetune-lr", type=float, default=0.01)
    # distillation
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.7)
    # pruning
    p.add_argument("--amount", type=float, default=0.5, help="fraction of weights to prune")
    p.add_argument("--structured", action="store_true", help="channel pruning instead of unstructured")
    # export
    p.add_argument("--ckpt", default=None, help="checkpoint path for export/prune")
    p.add_argument("--int8", action="store_true", help="INT8 quantize on export")
    return p


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")
    {
        "teacher": cmd_teacher, "student": cmd_student, "distill": cmd_distill,
        "prune": cmd_prune, "export": cmd_export, "benchmark": cmd_benchmark, "all": cmd_all,
    }[args.stage](args, device)


if __name__ == "__main__":
    main()
