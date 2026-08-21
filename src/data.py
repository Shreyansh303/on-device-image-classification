"""CIFAR-10 data pipeline: train/test loaders + a calibration subset for PTQ."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Standard CIFAR-10 channel statistics.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10
CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def _train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def _eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_loaders(
    data_dir: str = "data",
    batch_size: int = 128,
    num_workers: int = 2,
    augment: bool = True,
    download: bool = True,
):
    """Return (train_loader, test_loader) for CIFAR-10."""
    train_tf = _train_transform() if augment else _eval_transform()
    train_set = datasets.CIFAR10(data_dir, train=True, download=download, transform=train_tf)
    test_set = datasets.CIFAR10(data_dir, train=False, download=download, transform=_eval_transform())

    pin = torch.cuda.is_available()  # pinning is a no-op / warns on MPS
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    return train_loader, test_loader


def get_calibration_loader(
    data_dir: str = "data",
    num_batches: int = 16,
    batch_size: int = 32,
    num_workers: int = 2,
):
    """A small unshuffled slice of the *training* set for post-training quant calibration."""
    cal_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=_eval_transform())
    n = min(num_batches * batch_size, len(cal_set))
    subset = Subset(cal_set, list(range(n)))
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
