"""TensorFlow/Keras variant: train a MobileNetV3-Small on CIFAR-10, export a SavedModel.

This is the front half of the Android-native deployment path. The trained SavedModel is then
converted to .tflite (float / INT8) by tf/export_tflite.py.

Run: python tf/train_tf.py [--epochs 30]
"""
import argparse
import os

import keras
import tensorflow as tf

ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def build_model(num_classes: int = 10) -> keras.Model:
    # include_preprocessing=True adds a Rescaling layer; feed raw [0,255] images.
    return keras.applications.MobileNetV3Small(
        input_shape=(32, 32, 3),
        alpha=1.0,
        include_top=True,
        weights=None,
        classes=num_classes,
        include_preprocessing=True,
        classifier_activation="softmax",
    )


def get_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    return (x_train.astype("float32"), y_train.flatten()), \
           (x_test.astype("float32"), y_test.flatten())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    (x_train, y_train), (x_test, y_test) = get_data()
    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=args.epochs, batch_size=args.batch_size,
    )
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nKeras MobileNetV3-Small test accuracy: {acc * 100:.2f}%")

    os.makedirs(ARTIFACTS, exist_ok=True)
    model.save(os.path.join(ARTIFACTS, "mobilenetv3_cifar.keras"))
    model.export(os.path.join(ARTIFACTS, "saved_model"))  # TF SavedModel for TFLite conversion
    print(f"Saved Keras model + SavedModel to {ARTIFACTS}/")


if __name__ == "__main__":
    main()
