# On-Device Image Classification with Model Compression

A lightweight CNN image classifier built for **edge / mobile deployment**, compressed with
the three techniques that actually matter for shipping models on-device:

1. **Knowledge distillation** — a small MobileNetV3 student learns from a larger ResNet teacher.
2. **Pruning** — magnitude-based weight pruning + fine-tuning to remove redundant parameters.
3. **Quantization** — INT8 post-training quantization for ~4× smaller, faster inference.

The project optimizes for the full deployment story — **accuracy *and* model size *and* latency** —
and exports to the formats that run on real devices: **ONNX** and **TFLite** (Android-native).

> Two parallel pipelines are included:
> - **`src/` + `main.py`** — PyTorch training & compression, exporting to ONNX and TFLite (via Google's `ai-edge-torch`).
> - **`tf/`** — a TensorFlow → TFLite-native variant using `TFLiteConverter` (and TF-MOT) for side-by-side comparison.

---

## Results

CIFAR-10. Latency is single-image (batch=1) CPU inference, the realistic on-device condition.
Run `python main.py all` then `python main.py benchmark` to (re)generate `results/benchmark.json`
and `results/tradeoff.png`.

| Model                          | Top-1 % | Params (M) | Size (MB) | Latency (ms) |
|--------------------------------|--------:|-----------:|----------:|-------------:|
| Teacher (ResNet-18)            |   82.61 |     11.17  |    44.8   |         5.3  |
| Student (MobileNetV3-S) baseline | 77.42 |      1.53  |     6.2   |        32.0  |
| Student + Knowledge Distillation | 78.13 |      1.53  |     6.2   |        32.9  |
| Student + KD + Pruning (49.5%) |   78.83 |      1.53  |     6.2*  |        32.5  |
| Student + KD + Pruning + INT8  |   78.32 |      —     |     1.65  |         2.3  |

\* Unstructured pruning zeroes weights (sparsity shows in the table) but keeps the dense tensor
shape, so on-disk size only shrinks once combined with quantization or sparse storage.

_Numbers from a fast **8/8/3-epoch** demo run (teacher/student·distill/fine-tune) — enough to show
the compression trade-offs; train longer (`--teacher-epochs 40 --epochs 40 --finetune-epochs 10`)
for higher absolute accuracy. The compressed INT8 student keeps **~95% of the teacher's accuracy at
1/27th the size (1.65 MB) and ~14× lower latency than the fp32 student.**_

![Trade-off plot](results/tradeoff.png)

---

## Why this design

Shipping a model to a phone is not "train for max accuracy." You trade accuracy against a
**memory budget** (app size, RAM) and a **latency budget** (real-time inference, battery). This
repo demonstrates the standard toolkit an ML-infra / mobile team uses to hit those budgets:

- **Distillation** recovers most of a big model's accuracy in a model small enough to ship.
- **Pruning** exposes and removes redundant capacity.
- **Quantization** is the single biggest practical win: INT8 cuts size ~4× and speeds up CPU/NPU inference.

The benchmark deliberately reports **all three axes** so the trade-offs are explicit.

---


## Project structure

```
.
├── main.py               # pipeline CLI (teacher/student/distill/prune/export/benchmark/all)
├── src/
│   ├── data.py           # CIFAR-10 loaders + calibration subset
│   ├── models.py         # ResNet teacher & MobileNetV3 student (CIFAR-adapted stems)
│   ├── engine.py         # train / evaluate loops
│   ├── distill.py        # Hinton knowledge-distillation loss + trainer
│   ├── prune.py          # global magnitude pruning (+ structured option)
│   ├── quantize.py       # INT8 PTQ (FX static, dynamic fallback)
│   ├── export.py         # ONNX + TFLite export
│   ├── benchmark.py      # accuracy / size / latency table + trade-off plot
│   └── utils.py          # device, seeding, metrics, checkpoints, model stats
├── tf/                   # TensorFlow -> TFLite-native variant
├── tests/smoke_test.py   # fast end-to-end check on a tiny subset
└── requirements.txt
```

---

## Techniques in detail

**Knowledge distillation** (`src/distill.py`): loss = α·T²·KL(softₛ ‖ softₜ) + (1−α)·CE, with
temperature `T=4`, `α=0.7`. The teacher's softened probabilities carry "dark knowledge" (relative
class similarities) the student can't get from one-hot labels alone.

**Pruning** (`src/prune.py`): global unstructured L1 pruning ranks weights across the *whole*
network and zeroes the smallest, then fine-tunes (the mask keeps pruned weights at zero) and bakes
the mask in. A `--structured` channel-pruning mode is included for genuine dense-inference speedups.

**Quantization** (`src/quantize.py`): FX-graph post-training **static** quantization — weights and
activations both go to INT8, calibrated on real images via the `qnnpack` (ARM) backend. Falls back
to dynamic quantization for any model that isn't symbolically traceable.

**Export** (`src/export.py`): ONNX (verified for numerical parity against PyTorch) and TFLite via
`ai-edge-torch`, with optional INT8.
