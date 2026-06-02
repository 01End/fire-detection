"""Train the KerasHub RetinaNet fire/smoke detector (TensorFlow backend).

Same dataset layout as the torch trainer::

    <data>/train/images/*.jpg   <data>/train/labels/*.txt
    <data>/val/images/*.jpg     <data>/val/labels/*.txt

Saves best weights to ``--out`` (must end in ``.weights.h5``), loadable by
``TFFireDetector.from_checkpoint``.

UNVERIFIED until tensorflow + keras-hub are installed (see docs/TF_PORT.md).
On native Windows TensorFlow runs CPU-only; for GPU use WSL2 or Google Colab.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from firewatch.detection.tf_model import build_model  # noqa: E402

from .tf_dataset import build_dataset, load_class_map  # noqa: E402


def run_training(
    data_dir: str,
    epochs: int = 8,
    batch_size: int = 4,
    lr: float = 0.001,
    out: str = "models/fire_retinanet.weights.h5",
    image_size: int = 512,
    pretrained_backbone: bool = True,
) -> str:
    import keras

    if not out.endswith(".weights.h5"):
        raise ValueError("--out must end with '.weights.h5' (Keras save_weights_only)")

    class_map = load_class_map(data_dir)
    print(f"class_map (dataset-id -> tf-index): {class_map}")

    train_ds = build_dataset(
        os.path.join(data_dir, "train", "images"),
        os.path.join(data_dir, "train", "labels"),
        class_map=class_map, image_size=image_size, batch_size=batch_size, shuffle=True,
    )
    val_ds = build_dataset(
        os.path.join(data_dir, "val", "images"),
        os.path.join(data_dir, "val", "labels"),
        class_map=class_map, image_size=image_size, batch_size=batch_size, shuffle=False,
    )

    model = build_model(arch="retinanet", pretrained_backbone=pretrained_backbone)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        box_loss=keras.losses.MeanAbsoluteError(reduction="sum"),
    )

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                filepath=out, save_best_only=True, save_weights_only=True,
                monitor="val_loss",
            )
        ],
    )
    print(f"best weights -> {out}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train KerasHub RetinaNet fire/smoke detector")
    p.add_argument("--data", required=True, help="dataset root with train/ and val/")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--out", default="models/fire_retinanet.weights.h5")
    p.add_argument("--image-size", type=int, default=512)
    a = p.parse_args(argv)
    run_training(a.data, a.epochs, a.batch_size, a.lr, a.out, a.image_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
