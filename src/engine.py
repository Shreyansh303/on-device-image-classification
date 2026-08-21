"""Training / evaluation loops shared across baseline, distillation, and fine-tuning."""
from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import AverageMeter, accuracy


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, criterion=None) -> dict:
    model.to(device).eval()
    criterion = criterion or nn.CrossEntropyLoss()
    loss_m, top1_m = AverageMeter(), AverageMeter()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        (acc1,) = accuracy(out, y, topk=(1,))
        loss_m.update(loss.item(), x.size(0))
        top1_m.update(acc1, x.size(0))
    return {"loss": loss_m.avg, "top1": top1_m.avg}


def train_one_epoch(model, loader, optimizer, device, criterion=None, scheduler=None, desc="train") -> dict:
    model.train()
    criterion = criterion or nn.CrossEntropyLoss()
    loss_m, top1_m = AverageMeter(), AverageMeter()
    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        (acc1,) = accuracy(out, y, topk=(1,))
        loss_m.update(loss.item(), x.size(0))
        top1_m.update(acc1, x.size(0))
        pbar.set_postfix(loss=f"{loss_m.avg:.3f}", acc=f"{top1_m.avg:.1f}")
    if scheduler is not None:
        scheduler.step()
    return {"loss": loss_m.avg, "top1": top1_m.avg}


def fit(
    model,
    train_loader,
    test_loader,
    device,
    epochs: int = 50,
    lr: float = 0.1,
    weight_decay: float = 5e-4,
    momentum: float = 0.9,
    label: str = "model",
    checkpoint_path: str | None = None,
) -> dict:
    """Standard SGD + cosine schedule training. Returns best metrics + history."""
    from .utils import save_checkpoint

    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_top1, history = 0.0, []
    for epoch in range(epochs):
        tr = train_one_epoch(
            model, train_loader, optimizer, device, criterion, scheduler,
            desc=f"{label} {epoch + 1}/{epochs}",
        )
        ev = evaluate(model, test_loader, device, criterion)
        history.append({"epoch": epoch + 1, "train": tr, "test": ev})
        print(f"[{label}] epoch {epoch + 1}/{epochs}  "
              f"train_acc={tr['top1']:.2f}  test_acc={ev['top1']:.2f}  lr={scheduler.get_last_lr()[0]:.4f}")
        if ev["top1"] > best_top1:
            best_top1 = ev["top1"]
            if checkpoint_path:
                save_checkpoint(model, checkpoint_path, top1=best_top1, epoch=epoch + 1)
    return {"best_top1": best_top1, "history": history}
