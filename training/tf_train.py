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
    lr: float = 0.0001,
    out: str = "models/fire_retinanet.weights.h5",
    image_size: int = 512,
    pretrained_backbone: bool = True,
    train_split: str = "train",
    val_split: str = "val",
    augment: bool = False,
    exposure: str = "none",
    init_weights: "str | None" = None,
) -> str:
    import keras

    if not out.endswith(".weights.h5"):
        raise ValueError("--out must end with '.weights.h5' (Keras save_weights_only)")

    class_map = load_class_map(data_dir)
    print(f"class_map (dataset-id -> tf-index): {class_map}")
    print(f"augment={augment}  exposure={exposure}")

    # train_split/val_split let us point straight at e.g. D-Fire's train/ and test/
    # folders without copying the dataset into a train//val// layout.
    train_ds = build_dataset(
        os.path.join(data_dir, train_split, "images"),
        os.path.join(data_dir, train_split, "labels"),
        class_map=class_map, image_size=image_size, batch_size=batch_size, shuffle=True,
        augment=augment, exposure=exposure,
    )
    val_ds = build_dataset(
        os.path.join(data_dir, val_split, "images"),
        os.path.join(data_dir, val_split, "labels"),
        class_map=class_map, image_size=image_size, batch_size=batch_size, shuffle=False,
        augment=False, exposure=exposure,
    )

    model = build_model(arch="retinanet", pretrained_backbone=pretrained_backbone)
    if init_weights:
        # Warm-start: continue from a previously trained model instead of the COCO backbone,
        # so fine-tuning on the merged set converges in far fewer epochs (fast retrain).
        model.load_weights(init_weights)
        print(f"warm-started from {init_weights}")
    # compile() with just an optimizer uses RetinaNet's built-in box + focal losses.
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # Stabilizers: a high LR made an earlier run diverge after a few epochs, so we
    # halve the LR whenever val_loss stalls and stop early (keeping the best weights)
    # rather than overshooting. ModelCheckpoint still mirrors the best weights to disk.
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=out, save_best_only=True, save_weights_only=True,
            monitor="val_loss",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True, verbose=1,
        ),
    ]
    model.fit(
        train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks,
    )
    print(f"best weights -> {out}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train KerasHub RetinaNet fire/smoke detector")
    p.add_argument("--data", required=True, help="dataset root with train/ and val/")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--out", default="models/fire_retinanet.weights.h5")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--train-split", default="train", help="subdir name for training data")
    p.add_argument("--val-split", default="val", help="subdir name for validation data")
    p.add_argument("--augment", action="store_true",
                   help="random brightness/exposure jitter on the training split")
    p.add_argument("--exposure", default="none", choices=("none", "clahe", "gamma"),
                   help="deterministic exposure normalization (match this at inference/eval)")
    p.add_argument("--init-weights", default=None,
                   help="warm-start from this .weights.h5 (fast fine-tune instead of from COCO)")
    a = p.parse_args(argv)
    run_training(a.data, a.epochs, a.batch_size, a.lr, a.out, a.image_size,
                 train_split=a.train_split, val_split=a.val_split,
                 augment=a.augment, exposure=a.exposure, init_weights=a.init_weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
