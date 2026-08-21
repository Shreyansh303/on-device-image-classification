"""Model factory: a ResNet teacher and a MobileNetV3-Small student, adapted for CIFAR-10.

torchvision's ImageNet stems aggressively downsample (7x7 stride-2 conv + maxpool, or a
stride-2 stem), which destroys 32x32 CIFAR images. We swap in CIFAR-friendly stems so the
spatial resolution survives the early layers.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

from .data import NUM_CLASSES


def build_teacher(name: str = "resnet18", num_classes: int = NUM_CLASSES) -> nn.Module:
    """ResNet teacher with a CIFAR stem (3x3 stride-1 conv, no early maxpool)."""
    factory = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
    }
    if name not in factory:
        raise ValueError(f"Unknown teacher '{name}'. Choose from {list(factory)}.")
    model = factory[name](weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_student(name: str = "mobilenet_v3_small", num_classes: int = NUM_CLASSES) -> nn.Module:
    """Lightweight student. MobileNetV3-Small / EfficientNet flavours, CIFAR-adapted stem."""
    if name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None, num_classes=num_classes)
        # Keep more spatial resolution: stem stride 2 -> 1 for 32x32 inputs.
        model.features[0][0].stride = (1, 1)
        return model
    if name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None, num_classes=num_classes)
        model.features[0][0].stride = (1, 1)
        return model
    if name == "efficientnet_b0":
        # The closest torchvision stand-in for "EfficientNet-Lite".
        model = models.efficientnet_b0(weights=None, num_classes=num_classes)
        model.features[0][0].stride = (1, 1)
        return model
    raise ValueError(f"Unknown student '{name}'.")


def build_model(role: str, name: str | None = None, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Convenience dispatcher used by the CLI."""
    if role == "teacher":
        return build_teacher(name or "resnet18", num_classes)
    if role == "student":
        return build_student(name or "mobilenet_v3_small", num_classes)
    raise ValueError(f"role must be 'teacher' or 'student', got {role!r}")
