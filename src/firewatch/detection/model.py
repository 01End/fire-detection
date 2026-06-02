"""Build a permissively-licensed torchvision detection model for fire/smoke.

Two architectures are offered:
  * ``ssdlite`` — SSDlite320 + MobileNetV3 (light, CPU-friendly; the default).
  * ``retinanet`` — RetinaNet + ResNet50-FPN (heavier, more accurate on GPU).

Both are BSD-3 licensed (torchvision). Class index 0 is background; foreground classes
follow ``CLASS_NAMES`` order (1=fire, 2=smoke).
"""
from __future__ import annotations

from typing import Sequence

import torch.nn as nn

from .types import CLASS_NAMES

ARCHITECTURES = ("ssdlite", "retinanet")


def num_classes_for(class_names: Sequence[str] = CLASS_NAMES) -> int:
    """Total class count including the background class at index 0."""
    return len(class_names) + 1


def label_from_index(index: int, class_names: Sequence[str] = CLASS_NAMES) -> str:
    """Map a model class index (1-based foreground) to a label string."""
    if index < 1 or index > len(class_names):
        raise ValueError(f"class index {index} out of range for {class_names}")
    return class_names[index - 1]


def build_model(
    arch: str = "ssdlite",
    class_names: Sequence[str] = CLASS_NAMES,
    pretrained_backbone: bool = True,
) -> nn.Module:
    """Construct a detection model with a head sized for ``class_names`` (+ background).

    ``pretrained_backbone`` loads ImageNet backbone weights for transfer learning when
    True; set it False for fast offline construction (tests) or fully-random init.
    """
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHITECTURES}")

    num_classes = num_classes_for(class_names)

    if arch == "ssdlite":
        from torchvision.models import MobileNet_V3_Large_Weights
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large

        weights_backbone = (
            MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
        )
        return ssdlite320_mobilenet_v3_large(
            weights=None, weights_backbone=weights_backbone, num_classes=num_classes
        )

    from torchvision.models import ResNet50_Weights
    from torchvision.models.detection import retinanet_resnet50_fpn

    weights_backbone = ResNet50_Weights.DEFAULT if pretrained_backbone else None
    return retinanet_resnet50_fpn(
        weights=None, weights_backbone=weights_backbone, num_classes=num_classes
    )
