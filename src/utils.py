"""Shared utilities: device selection, seeding, metrics, checkpointing, model stats."""
from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Pick the best available device. Quantized models must run on CPU."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """Tracks a running average of a scalar (e.g. loss, accuracy)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Top-k accuracy (%) for the given output logits and targets."""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k * 100.0 / batch_size).item())
    return res


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_nonzero_parameters(model: torch.nn.Module) -> int:
    return sum(int(torch.count_nonzero(p)) for p in model.parameters())


def global_sparsity(model: torch.nn.Module) -> float:
    """Fraction of weight entries that are exactly zero (0.0 = dense)."""
    total = count_parameters(model)
    if total == 0:
        return 0.0
    nonzero = count_nonzero_parameters(model)
    return 1.0 - nonzero / total


def model_size_mb(model: torch.nn.Module, path: str | None = None) -> float:
    """Serialized size of the model's state_dict on disk, in MB."""
    tmp = path or "_tmp_size.pt"
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp) / 1e6
    if path is None:
        os.remove(tmp)
    return size


def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / 1e6


@contextmanager
def timer():
    """Context manager yielding a callable that returns elapsed seconds."""
    start = time.perf_counter()
    elapsed = {"value": 0.0}
    try:
        yield lambda: time.perf_counter() - start
    finally:
        elapsed["value"] = time.perf_counter() - start


def save_checkpoint(model: torch.nn.Module, path: str, **extra) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"state_dict": model.state_dict(), **extra}
    torch.save(payload, path)


def load_checkpoint(model: torch.nn.Module, path: str, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    return ckpt
