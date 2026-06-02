"""TensorFlow / KerasHub detector model builder + pure geometry helpers.

Uses ``keras_hub.models.RetinaNetObjectDetector`` (RetinaNet + ResNet50-FPN, Apache-2.0).
Keras is imported lazily inside ``build_model`` so the pure helpers below (box-format and
label mapping) are importable and testable without TensorFlow installed.

VERIFIED on TF 2.20 / keras-hub 0.29 (native Windows, CPU). KerasHub specifics learned
during verification:
  * The KerasHub *preprocessor* hard-requires ``tensorflow-text``, which has no Windows
    wheel — so we build with ``preprocessor=None`` and do ImageNet normalization ourselves
    (see IMAGENET_MEAN/STD).
  * bounding boxes use ``yxyx`` order (y first), absolute pixels.
  * class indices are 0-based (0=fire, 1=smoke) — no background class.
  * ``compile()`` with just an optimizer uses RetinaNet's built-in box/focal losses.
  * ``predict()`` returns dict keys: boxes, confidence, labels, num_detections.
"""
from __future__ import annotations

from typing import Sequence, Tuple

from .types import CLASS_NAMES

BBox = Tuple[float, float, float, float]

# KerasHub default input size; multiple of 128 to satisfy the FPN strides.
DEFAULT_IMAGE_SIZE = 512
COCO_PRESET = "retinanet_resnet50_fpn_coco"

# ImageNet normalization (RetinaNet expects it; we apply it ourselves since the KerasHub
# preprocessor can't load on Windows). norm = (rgb/255 - mean) / std.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def num_classes_for(class_names: Sequence[str] = CLASS_NAMES) -> int:
    """Foreground class count (KerasHub is 0-indexed, no background class)."""
    return len(class_names)


def label_from_index(index: int, class_names: Sequence[str] = CLASS_NAMES) -> str:
    """Map a 0-based KerasHub class index to a label (0=fire, 1=smoke)."""
    if index < 0 or index >= len(class_names):
        raise ValueError(f"class index {index} out of range for {class_names}")
    return class_names[index]


def yxyx_to_xyxy(box: Sequence[float]) -> BBox:
    """Convert a KerasHub ``yxyx`` box to our internal ``xyxy``."""
    y1, x1, y2, x2 = box
    return (float(x1), float(y1), float(x2), float(y2))


def xyxy_to_yxyx(box: Sequence[float]) -> BBox:
    x1, y1, x2, y2 = box
    return (float(y1), float(x1), float(y2), float(x2))


def scale_box_xyxy(
    box: BBox, from_size: Tuple[int, int], to_size: Tuple[int, int]
) -> BBox:
    """Scale an xyxy box from one (width, height) to another (width, height)."""
    fw, fh = from_size
    tw, th = to_size
    sx, sy = tw / fw, th / fh
    x1, y1, x2, y2 = box
    return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)


def normalize_imagenet(rgb):
    """ImageNet-normalize an HxWx3 RGB array (0-255) -> float32. Pure numpy."""
    import numpy as np

    mean = np.array(IMAGENET_MEAN, dtype="float32")
    std = np.array(IMAGENET_STD, dtype="float32")
    return ((rgb.astype("float32") / 255.0) - mean) / std


def build_model(
    arch: str = "retinanet",
    class_names: Sequence[str] = CLASS_NAMES,
    pretrained_backbone: bool = True,
):
    """Construct a RetinaNetObjectDetector with a head sized for ``class_names``.

    The ResNet50-FPN backbone is loaded from the COCO preset (cached after first
    download) for a good fine-tuning start. ``preprocessor=None`` because the KerasHub
    preprocessor needs tensorflow-text (unavailable on Windows); callers must feed
    ImageNet-normalized images (use ``normalize_imagenet``).
    """
    import keras_hub

    if arch != "retinanet":
        raise ValueError(f"TF backend supports arch='retinanet', got {arch!r}")

    backbone = keras_hub.models.Backbone.from_preset(COCO_PRESET)
    return keras_hub.models.RetinaNetObjectDetector(
        backbone=backbone,
        num_classes=num_classes_for(class_names),
        preprocessor=None,
    )
