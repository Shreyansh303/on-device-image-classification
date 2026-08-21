"""Convert the trained Keras SavedModel to TFLite in three flavours and benchmark each:

    float       - baseline FP32 .tflite
    dynamic     - dynamic-range INT8 (weights INT8, activations float at runtime)
    int8        - full-integer INT8 (weights + activations), calibrated on real images

Reports size / accuracy / latency per flavour using the TFLite interpreter — the same runtime
that executes on an Android device.

Run: python tf/export_tflite.py
"""
import json
import os
import time

import keras
import numpy as np
import tensorflow as tf

ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
SAVED_MODEL = os.path.join(ARTIFACTS, "saved_model")


def _calibration_data(x, n=200):
    def gen():
        for i in range(min(n, len(x))):
            yield [x[i:i + 1].astype("float32")]
    return gen


def convert(kind: str, x_calib) -> str:
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
    if kind == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif kind == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _calibration_data(x_calib)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
    elif kind != "float":
        raise ValueError(kind)
    tflite_model = converter.convert()
    path = os.path.join(ARTIFACTS, f"model_{kind}.tflite")
    with open(path, "wb") as f:
        f.write(tflite_model)
    return path


def benchmark_tflite(path: str, x_test, y_test, n_acc=2000, n_lat=200) -> dict:
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    # Accuracy
    correct = 0
    N = min(n_acc, len(x_test))
    for i in range(N):
        x = x_test[i:i + 1].astype(in_d["dtype"])
        interp.set_tensor(in_d["index"], x)
        interp.invoke()
        pred = int(np.argmax(interp.get_tensor(out_d["index"])[0]))
        correct += int(pred == y_test[i])
    acc = 100.0 * correct / N

    # Latency (single image)
    x = x_test[0:1].astype(in_d["dtype"])
    for _ in range(10):
        interp.set_tensor(in_d["index"], x); interp.invoke()
    times = []
    for _ in range(n_lat):
        t0 = time.perf_counter()
        interp.set_tensor(in_d["index"], x); interp.invoke()
        times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "name": os.path.basename(path),
        "top1": round(acc, 2),
        "size_MB": round(os.path.getsize(path) / 1e6, 3),
        "latency_ms": round(float(np.mean(times)), 3),
    }


def main():
    if not os.path.isdir(SAVED_MODEL):
        raise SystemExit(f"No SavedModel at {SAVED_MODEL}. Run tf/train_tf.py first.")
    (_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_test = x_test.astype("float32")
    y_test = y_test.flatten()

    rows = []
    for kind in ("float", "dynamic", "int8"):
        path = convert(kind, x_test)
        row = benchmark_tflite(path, x_test, y_test)
        print(f"  {kind:8s} -> {row['size_MB']:.2f} MB  acc={row['top1']:.2f}%  "
              f"lat={row['latency_ms']:.2f} ms")
        rows.append(row)

    os.makedirs("results", exist_ok=True)
    with open("results/benchmark_tflite.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nSaved results/benchmark_tflite.json")


if __name__ == "__main__":
    main()
