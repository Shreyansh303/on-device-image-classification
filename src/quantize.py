"""INT8 quantization. Primary path is FX-graph post-training *static* quantization
(weights + activations -> int8, calibrated on real data). Falls back to dynamic
quantization if a model isn't symbolically traceable.

Quantized models execute on CPU via the qnnpack (ARM) / x86 backends.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


def _pick_backend() -> str:
    # qnnpack is the mobile/ARM backend (Apple Silicon); fbgemm/x86 otherwise.
    return "qnnpack" if "qnnpack" in torch.backends.quantized.supported_engines else "x86"


def quantize_static_fx(model: nn.Module, calib_loader, backend: str | None = None) -> nn.Module:
    """Post-training static quantization via FX graph mode, calibrated on `calib_loader`."""
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    backend = backend or _pick_backend()
    torch.backends.quantized.engine = backend

    model = copy.deepcopy(model).eval().cpu()
    qconfig_mapping = get_default_qconfig_mapping(backend)
    example_inputs = (torch.randn(1, 3, 32, 32),)

    prepared = prepare_fx(model, qconfig_mapping, example_inputs)
    with torch.no_grad():
        for x, _ in calib_loader:
            prepared(x.cpu())
    return convert_fx(prepared)


def quantize_dynamic(model: nn.Module) -> nn.Module:
    """Dynamic quantization of Linear layers — robust fallback, no calibration needed."""
    return torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(model).eval().cpu(), {nn.Linear}, dtype=torch.qint8
    )


def quantize_model(model: nn.Module, calib_loader=None):
    """Try static FX quantization; fall back to dynamic. Returns (qmodel, method_str)."""
    if calib_loader is not None:
        try:
            return quantize_static_fx(model, calib_loader), "static-fx-int8"
        except Exception as e:  # noqa: BLE001 - trace failures vary by model
            print(f"[quantize] static FX quantization failed ({type(e).__name__}: {e}); "
                  f"falling back to dynamic.")
    return quantize_dynamic(model), "dynamic-int8"
