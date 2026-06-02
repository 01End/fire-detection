"""Build a tf.data pipeline from YOLO-format labels for KerasHub RetinaNet.

Produces batches of ``(images, {"boxes": yxyx, "labels": 0-indexed})`` as the KerasHub
object-detection guide expects. Class ids are remapped from the dataset's convention to
ours (fire=0, smoke=1).

UNVERIFIED until tensorflow + keras-hub are installed (see docs/TF_PORT.md).
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Our TF label indices (0-based, no background).
OUR_FIRE, OUR_SMOKE = 0, 1
# D-Fire convention is 0=smoke, 1=fire -> map to ours.
DEFAULT_CLASS_MAP: Mapping[int, int] = {1: OUR_FIRE, 0: OUR_SMOKE}


def load_class_map(data_dir: str) -> Mapping[int, int]:
    """Load class_map.json if present (written by prepare_dfire, torch 1-based) and
    convert to TF 0-based by subtracting one; else use the D-Fire default."""
    path = os.path.join(data_dir, "class_map.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            torch_map = {int(k): int(v) for k, v in json.load(fh).items()}
        return {k: v - 1 for k, v in torch_map.items()}  # torch 1-based -> tf 0-based
    return dict(DEFAULT_CLASS_MAP)


def _label_for(image_path: str, labels_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(labels_dir, stem + ".txt")


def _read_record(image_path: str, labels_dir: str, class_map: Mapping[int, int]):
    import cv2

    bgr = cv2.imread(image_path)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    rgb = bgr[:, :, ::-1].astype("float32")

    boxes: List[List[float]] = []
    labels: List[int] = []
    lbl = _label_for(image_path, labels_dir)
    if os.path.exists(lbl):
        with open(lbl, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, bw, bh = (float(p) for p in parts)
                mapped = class_map.get(int(cls))
                if mapped is None:
                    continue
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                boxes.append([y1, x1, y2, x2])  # yxyx
                labels.append(mapped)

    return (
        rgb,
        np.asarray(boxes, dtype="float32").reshape(-1, 4),
        np.asarray(labels, dtype="int32").reshape(-1),
    )


def _list_images(images_dir: str) -> List[str]:
    return sorted(
        p for p in glob.glob(os.path.join(images_dir, "*"))
        if os.path.splitext(p)[1].lower() in IMAGE_EXTS
    )


def build_dataset(
    images_dir: str,
    labels_dir: str,
    class_map: Optional[Mapping[int, int]] = None,
    image_size: int = 512,
    batch_size: int = 4,
    shuffle: bool = True,
    max_boxes: int = 100,
):
    """Return a batched tf.data.Dataset of (images, {"boxes","labels"})."""
    import keras
    import tensorflow as tf

    class_map = dict(class_map or DEFAULT_CLASS_MAP)
    paths = _list_images(images_dir)
    if not paths:
        raise FileNotFoundError(f"no images in {images_dir}")

    bbox_format = "yxyx"

    def gen():
        order = list(range(len(paths)))
        if shuffle:
            np.random.shuffle(order)
        for i in order:
            rec = _read_record(paths[i], labels_dir, class_map)
            if rec is None:
                continue
            img, boxes, labels = rec
            yield {
                "images": img,
                "bounding_boxes": {"boxes": boxes, "labels": labels},
            }

    output_signature = {
        "images": tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
        "bounding_boxes": {
            "boxes": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            "labels": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        },
    }

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    resizing = keras.layers.Resizing(
        height=image_size, width=image_size, interpolation="bilinear",
        pad_to_aspect_ratio=True, bounding_box_format=bbox_format,
    )
    max_box_layer = keras.layers.MaxNumBoundingBoxes(
        max_number=max_boxes, bounding_box_format=bbox_format
    )

    def to_tuple(record):
        return record["images"], {
            "boxes": record["bounding_boxes"]["boxes"],
            "labels": record["bounding_boxes"]["labels"],
        }

    ds = ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(max_box_layer, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
