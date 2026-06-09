"""Build a tf.data pipeline from YOLO-format labels for KerasHub RetinaNet.

Produces batches of ``(images, {"boxes": yxyx, "labels": 0-indexed})``. All image work
(resize, ImageNet-normalize, box rescaling, padding) is done in NumPy so we avoid the
KerasHub preprocessing layers that require tensorflow-text (unavailable on Windows).

Class ids are remapped from the dataset's convention to ours (fire=0, smoke=1).
VERIFIED on TF 2.20 / keras-hub 0.29 (Windows CPU).
"""
from __future__ import annotations

import glob
import json
import os
from typing import List, Mapping, Optional, Tuple

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from firewatch.detection.tf_model import normalize_imagenet  # noqa: E402
from firewatch.detection.exposure import auto_exposure, jitter_exposure  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Our TF label indices (0-based, no background).
OUR_FIRE, OUR_SMOKE = 0, 1
# D-Fire convention is 0=smoke, 1=fire -> map to ours.
DEFAULT_CLASS_MAP: Mapping[int, int] = {1: OUR_FIRE, 0: OUR_SMOKE}


def load_class_map(data_dir: str) -> Mapping[int, int]:
    """Load class_map.json (written by prepare_dfire, torch 1-based) and convert to TF
    0-based by subtracting one; else use the D-Fire default."""
    path = os.path.join(data_dir, "class_map.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            torch_map = {int(k): int(v) for k, v in json.load(fh).items()}
        return {k: v - 1 for k, v in torch_map.items()}  # torch 1-based -> tf 0-based
    return dict(DEFAULT_CLASS_MAP)


def _label_for(image_path: str, labels_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(labels_dir, stem + ".txt")


def _list_images(images_dir: str) -> List[str]:
    return sorted(
        p for p in glob.glob(os.path.join(images_dir, "*"))
        if os.path.splitext(p)[1].lower() in IMAGE_EXTS
    )


def _read_sample(
    image_path: str, labels_dir: str, class_map: Mapping[int, int],
    image_size: int, max_boxes: int, augment: bool = False, exposure: str = "none",
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    import cv2

    bgr = cv2.imread(image_path)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    # Exposure handling mirrors inference (TFFireDetector._prepare): optional random jitter
    # (train only) then the same deterministic normalization, before resize + ImageNet-norm.
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    if augment:
        rgb = jitter_exposure(rgb)
    rgb = auto_exposure(rgb, exposure)
    rgb = cv2.resize(rgb, (image_size, image_size))
    image = normalize_imagenet(rgb)

    sy, sx = image_size / h, image_size / w
    boxes = np.zeros((max_boxes, 4), dtype="float32")
    labels = np.full((max_boxes,), -1, dtype="int32")  # -1 = padding/ignore

    lbl = _label_for(image_path, labels_dir)
    k = 0
    if os.path.exists(lbl):
        with open(lbl, encoding="utf-8") as fh:
            for line in fh:
                if k >= max_boxes:
                    break
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, bw, bh = (float(p) for p in parts)
                mapped = class_map.get(int(cls))
                if mapped is None:
                    continue
                # YOLO normalized cxcywh -> absolute yxyx in the resized image.
                x1 = (cx - bw / 2) * w * sx
                y1 = (cy - bh / 2) * h * sy
                x2 = (cx + bw / 2) * w * sx
                y2 = (cy + bh / 2) * h * sy
                boxes[k] = [y1, x1, y2, x2]
                labels[k] = mapped
                k += 1
    return image, boxes, labels


def build_dataset(
    images_dir: str,
    labels_dir: str,
    class_map: Optional[Mapping[int, int]] = None,
    image_size: int = 512,
    batch_size: int = 4,
    shuffle: bool = True,
    max_boxes: int = 100,
    augment: bool = False,
    exposure: str = "none",
):
    """Return a batched tf.data.Dataset of (images, {"boxes","labels"})."""
    import tensorflow as tf

    class_map = dict(class_map or DEFAULT_CLASS_MAP)
    paths = _list_images(images_dir)
    if not paths:
        raise FileNotFoundError(f"no images in {images_dir}")

    def gen():
        order = list(range(len(paths)))
        if shuffle:
            np.random.shuffle(order)
        for i in order:
            sample = _read_sample(paths[i], labels_dir, class_map, image_size, max_boxes,
                                  augment=augment, exposure=exposure)
            if sample is None:
                continue
            image, boxes, labels = sample
            yield image, {"boxes": boxes, "labels": labels}

    output_signature = (
        tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
        {
            "boxes": tf.TensorSpec(shape=(max_boxes, 4), dtype=tf.float32),
            "labels": tf.TensorSpec(shape=(max_boxes,), dtype=tf.int32),
        },
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(tf.data.AUTOTUNE)
