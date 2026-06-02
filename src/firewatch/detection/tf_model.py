"""TensorFlow / KerasHub detector model builder + pure geometry helpers.

Uses ``keras_hub.models.RetinaNetObjectDetector`` (RetinaNet + ResNet50-FPN, Apache-2.0).
Keras is imported lazily inside ``build_model`` so the pure helpers below (box-format and
label mapping) are importable and testable without TensorFlow installed.

NOTE: This backend is UNVERIFIED until ``tensorflow`` + ``keras-hub`` are installed (see
docs/TF_PORT.md). KerasHub specifics:
  * bounding boxes use ``yxyx`` order (y first), absolute pixels
  * class indices are 0-based (0=fire, 1=smoke) — no background class
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from .types import CLASS_NAMES

BBox = Tuple[float, float, float, float]

# KerasHub default input size; multiple of 128 to satisfy the FPN strides.
DEFAULT_IMAGE_SIZE = 512
COCO_PRESET = "retinanet_resnet50_fpn_coco"
IMAGENET_BACKBONE = "resnet_50_imagenet"


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


def build_model(
    arch: str = "retinanet",
    class_names: Sequence[str] = CLASS_NAMES,
    pretrained_backbone: bool = True,
):
    """Construct a RetinaNetObjectDetector with a head sized for ``class_names``.

    ``pretrained_backbone=True`` fine-tunes from the COCO preset (recommended);
    False builds on a from-scratch ImageNet backbone. Both download weights on first
    use, so this requires network access the first time.
    """
    import keras_hub  # lazy: only needed when actually building a model

    if arch != "retinanet":
        raise ValueError(f"TF backend supports arch='retinanet', got {arch!r}")

    num_classes = num_classes_for(class_names)

    if pretrained_backbone:
        backbone = keras_hub.models.Backbone.from_preset(COCO_PRESET)
        preprocessor = keras_hub.models.RetinaNetObjectDetectorPreprocessor.from_preset(
            COCO_PRESET
        )
    else:
        image_encoder = keras_hub.models.Backbone.from_preset(IMAGENET_BACKBONE)
        backbone = keras_hub.models.RetinaNetBackbone(
            image_encoder=image_encoder, min_level=3, max_level=5, use_p5=True
        )
        image_converter = keras_hub.layers.RetinaNetImageConverter(scale=1 / 255)
        preprocessor = keras_hub.models.RetinaNetObjectDetectorPreprocessor(
            image_converter=image_converter
        )

    return keras_hub.models.RetinaNetObjectDetector(
        backbone=backbone,
        num_classes=num_classes,
        preprocessor=preprocessor,
    )
