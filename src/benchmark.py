"""Benchmark accuracy / size / latency and render the comparison table + plot."""
from __future__ import annotations

import json
import os
import time

import torch

from .engine import evaluate
from .utils import count_parameters, global_sparsity, model_size_mb


@torch.no_grad()
def latency_ms(model, device="cpu", input_shape=(1, 3, 32, 32), warmup=10, runs=50) -> dict:
    """Mean/std single-inference latency in milliseconds at the given batch size."""
    model.eval().to(device)
    x = torch.randn(*input_shape, device=device)
    for _ in range(warmup):
        model(x)
    if device == "mps":
        torch.mps.synchronize()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        model(x)
        if device == "mps":
            torch.mps.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    t = torch.tensor(times)
    return {"mean_ms": t.mean().item(), "std_ms": t.std().item()}


def benchmark_model(model, name, test_loader, acc_device="cpu", lat_device="cpu") -> dict:
    """Full record for one model: accuracy, params, on-disk size, sparsity, latency."""
    ev = evaluate(model, test_loader, acc_device)
    lat = latency_ms(model, device=lat_device)
    return {
        "name": name,
        "top1": round(ev["top1"], 2),
        "params_M": round(count_parameters(model) / 1e6, 3),
        "size_MB": round(model_size_mb(model), 3),
        "sparsity": round(global_sparsity(model), 3),
        "latency_ms": round(lat["mean_ms"], 3),
        "latency_std_ms": round(lat["std_ms"], 3),
    }


def print_table(rows: list[dict]) -> str:
    cols = ["name", "top1", "params_M", "size_MB", "sparsity", "latency_ms"]
    headers = ["Model", "Top-1 %", "Params (M)", "Size (MB)", "Sparsity", "Latency (ms)"]
    widths = [max(len(h), max((len(str(r.get(c, ""))) for r in rows), default=0)) for c, h in zip(cols, headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "  ".join("-" * w for w in widths)
    out = [line, sep]
    for r in rows:
        out.append("  ".join(str(r.get(c, "")).ljust(w) for c, w in zip(cols, widths)))
    table = "\n".join(out)
    print(table)
    return table


def save_results(rows: list[dict], out_dir: str = "results") -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "benchmark.json"), "w") as f:
        json.dump(rows, f, indent=2)
    _plot(rows, os.path.join(out_dir, "tradeoff.png"))


def _plot(rows: list[dict], path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    sizes = [r["size_MB"] for r in rows]
    accs = [r["top1"] for r in rows]
    lats = [r["latency_ms"] for r in rows]
    sc = ax.scatter(sizes, accs, s=[max(40, l * 8) for l in lats], c=lats,
                    cmap="viridis", alpha=0.8, edgecolors="black")
    for r in rows:
        ax.annotate(r["name"], (r["size_MB"], r["top1"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Model size on disk (MB)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Compression trade-off: accuracy vs size (bubble/colour = latency)")
    fig.colorbar(sc, label="Latency (ms, CPU batch=1)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
