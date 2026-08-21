"""Knowledge distillation (Hinton et al., 2015): train the student to match a teacher's
softened logits plus the ground-truth labels."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .engine import evaluate
from .utils import AverageMeter, accuracy, save_checkpoint


def distillation_loss(student_logits, teacher_logits, targets, T: float = 4.0, alpha: float = 0.7):
    """Weighted sum of soft-target KL (temperature T) and hard-label cross-entropy.

    The KL term is scaled by T^2 so its gradient magnitude is comparable to the CE term.
    """
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    kd = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T * T)
    ce = F.cross_entropy(student_logits, targets)
    return alpha * kd + (1.0 - alpha) * ce


def train_distill(
    student,
    teacher,
    train_loader,
    test_loader,
    device,
    epochs: int = 50,
    lr: float = 0.05,
    weight_decay: float = 5e-4,
    momentum: float = 0.9,
    temperature: float = 4.0,
    alpha: float = 0.7,
    label: str = "distill",
    checkpoint_path: str | None = None,
) -> dict:
    student.to(device)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.SGD(
        student.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_top1, history = 0.0, []
    for epoch in range(epochs):
        student.train()
        loss_m, top1_m = AverageMeter(), AverageMeter()
        pbar = tqdm(train_loader, desc=f"{label} {epoch + 1}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                t_logits = teacher(x)
            optimizer.zero_grad()
            s_logits = student(x)
            loss = distillation_loss(s_logits, t_logits, y, temperature, alpha)
            loss.backward()
            optimizer.step()
            (acc1,) = accuracy(s_logits, y, topk=(1,))
            loss_m.update(loss.item(), x.size(0))
            top1_m.update(acc1, x.size(0))
            pbar.set_postfix(loss=f"{loss_m.avg:.3f}", acc=f"{top1_m.avg:.1f}")
        scheduler.step()

        ev = evaluate(student, test_loader, device)
        history.append({"epoch": epoch + 1, "train_acc": top1_m.avg, "test": ev})
        print(f"[{label}] epoch {epoch + 1}/{epochs}  "
              f"train_acc={top1_m.avg:.2f}  test_acc={ev['top1']:.2f}")
        if ev["top1"] > best_top1:
            best_top1 = ev["top1"]
            if checkpoint_path:
                save_checkpoint(student, checkpoint_path, top1=best_top1, epoch=epoch + 1)
    return {"best_top1": best_top1, "history": history}
