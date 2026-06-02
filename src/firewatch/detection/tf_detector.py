"""TensorFlow/KerasHub detector wrapper exposing the same interface as the torch one.

Mirrors ``FireDetector``: ``detect(image_bgr) -> list[Detection]`` and
``from_checkpoint(...)`` so the pipeline/config are backend-agnostic.

UNVERIFIED until tensorflow + keras-hub are installed (see docs/TF_PORT.md).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .tf_model import (
    DEFAULT_IMAGE_SIZE,
    build_model,
    label_from_index,
    scale_box_xyxy,
    yxyx_to_xyxy,
)
from .types import CLASS_NAMES, Detection


class TFFireDetector:
    """Runs a KerasHub RetinaNet model on BGR frames and returns detections.

    TensorFlow chooses GPU automatically when available (note: not on native Windows —
    see docs). The ``device`` argument is accepted for interface parity and ignored.
    """

    def __init__(
        self,
        model,
        device: Optional[str] = None,  # accepted for parity; TF manages devices
        score_threshold: float = 0.5,
        class_names: Sequence[str] = CLASS_NAMES,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ):
        self.model = model
        self.score_threshold = score_threshold
        self.class_names = tuple(class_names)
        self.image_size = image_size

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        arch: str = "retinanet",
        device: Optional[str] = None,
        score_threshold: float = 0.5,
        class_names: Sequence[str] = CLASS_NAMES,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> "TFFireDetector":
        model = build_model(arch=arch, class_names=class_names, pretrained_backbone=False)
        model.load_weights(path)
        return cls(model, device=device, score_threshold=score_threshold,
                   class_names=class_names, image_size=image_size)

    def _prepare(self, image_bgr: np.ndarray) -> np.ndarray:
        import cv2

        rgb = image_bgr[:, :, ::-1]
        resized = cv2.resize(rgb, (self.image_size, self.image_size))
        # Model's preprocessor applies the 1/255 scaling, so pass raw 0-255 floats.
        return resized.astype("float32")

    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        h, w = image_bgr.shape[:2]
        batch = self._prepare(image_bgr)[None, ...]  # (1, S, S, 3)

        preds = self.model.predict(batch, verbose=0)
        boxes = np.asarray(preds["boxes"])[0]            # (N,4) yxyx, abs px in SxS
        confidence = np.asarray(preds["confidence"])[0]  # (N,)
        classes = np.asarray(preds["classes"])[0]        # (N,) 0-indexed; -1 = padding

        detections: List[Detection] = []
        for box, conf, cls in zip(boxes, confidence, classes):
            cls = int(cls)
            if cls < 0 or float(conf) < self.score_threshold:
                continue
            xyxy = yxyx_to_xyxy(box)
            xyxy = scale_box_xyxy(
                xyxy, from_size=(self.image_size, self.image_size), to_size=(w, h)
            )
            detections.append(
                Detection(
                    bbox=xyxy,
                    label=label_from_index(cls, self.class_names),
                    confidence=float(conf),
                )
            )
        return detections
