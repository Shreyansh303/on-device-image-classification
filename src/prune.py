"""Magnitude-based weight pruning with torch.nn.utils.prune.

Default is *global* unstructured L1 pruning: the smallest-magnitude weights across the whole
network are zeroed, which best preserves accuracy at a given sparsity. We then fine-tune the
surviving weights (the mask keeps pruned weights at zero) and bake the mask in permanently.

Note: unstructured sparsity shrinks the model on disk (and with sparse kernels, at runtime)
but dense inference cost is unchanged. `ln_structured` channel pruning is offered for the
case where you want genuine dense-inference speedups.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.utils.prune as prune


def _prunable_parameters(model: nn.Module):
    """All Conv2d/Linear weight tensors — the layers worth pruning."""
    params = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            params.append((module, "weight"))
    return params


def prune_model(model: nn.Module, amount: float = 0.5, structured: bool = False) -> nn.Module:
    """Zero out `amount` fraction of weights. Leaves reparametrization in place so the mask
    survives subsequent fine-tuning; call `finalize_pruning` afterwards."""
    params = _prunable_parameters(model)
    if structured:
        # Per-layer channel (output-filter) pruning by L2 norm.
        for module, name in params:
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name=name, amount=amount, n=2, dim=0)
            else:
                prune.l1_unstructured(module, name=name, amount=amount)
    else:
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    return model


def finalize_pruning(model: nn.Module) -> nn.Module:
    """Remove the reparametrization, folding the mask into the weights permanently."""
    for module, name in _prunable_parameters(model):
        try:
            prune.remove(module, name)
        except ValueError:
            pass  # module was not pruned
    return model
