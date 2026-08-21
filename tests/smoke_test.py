"""Fast end-to-end smoke test: exercises every pipeline stage on a tiny CIFAR subset.

Not a full training run — just verifies the code paths execute and shapes/sparsity/quant
behave. Run: python tests/smoke_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.benchmark import benchmark_model, print_table
from src.data import CIFAR10_MEAN, CIFAR10_STD
from src.distill import train_distill
from src.engine import evaluate, fit
from src.export import export_onnx
from src.models import build_student, build_teacher
from src.prune import finalize_pruning, prune_model
from src.quantize import quantize_model
from src.utils import get_device, global_sparsity

device = get_device()
print("device:", device)

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
train_full = datasets.CIFAR10("data", train=True, download=True, transform=tf)
test_full = datasets.CIFAR10("data", train=False, download=True, transform=tf)
train_loader = DataLoader(Subset(train_full, range(512)), batch_size=64, shuffle=True)
test_loader = DataLoader(Subset(test_full, range(256)), batch_size=64)
calib_loader = DataLoader(Subset(train_full, range(128)), batch_size=32)

print("\n[1/7] teacher 1-epoch fit")
teacher = build_teacher("resnet18")
fit(teacher, train_loader, test_loader, device, epochs=1, label="teacher")

print("\n[2/7] student 1-epoch fit")
student = build_student("mobilenet_v3_small")
fit(student, train_loader, test_loader, device, epochs=1, label="student")

print("\n[3/7] distillation 1-epoch")
train_distill(student, teacher, train_loader, test_loader, device, epochs=1)

print("\n[4/7] prune 50% + finalize")
prune_model(student, amount=0.5)
finalize_pruning(student)
sp = global_sparsity(student)
print(f"  sparsity after prune+finalize: {sp:.2%}")
assert sp > 0.4, f"expected ~50% sparsity, got {sp:.2%}"

print("\n[5/7] INT8 quantization")
qmodel, method = quantize_model(student, calib_loader)
print("  method:", method)
qacc = evaluate(qmodel, test_loader, "cpu")
print(f"  quantized eval top-1 (tiny set): {qacc['top1']:.1f}%")

print("\n[6/7] ONNX export")
export_onnx(student, "exports/_smoke.onnx")
assert os.path.exists("exports/_smoke.onnx")

print("\n[7/7] benchmark table")
rows = [
    benchmark_model(teacher, "Teacher", test_loader, acc_device=device, lat_device="cpu"),
    benchmark_model(student, "Student+Prune", test_loader, acc_device=device, lat_device="cpu"),
    benchmark_model(qmodel, f"Student+INT8 ({method})", test_loader, acc_device="cpu", lat_device="cpu"),
]
print_table(rows)

os.remove("exports/_smoke.onnx")
print("\nSMOKE TEST PASSED")
