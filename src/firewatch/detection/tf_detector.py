"""TensorFlow/KerasHub detector wrapper exposing the same interface as the torch one.

Mirrors ``FireDetector``: ``detect(image_bgr) -> list[Detection]`` and
``from_checkpoint(...)`` so the pipeline/config are backend-agnostic.

VERIFIED on TF 2.20 / keras-hub 0.29 (Windows CPU). TF runs CPU-only on native Windows.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .tf_model import (
    DEFAULT_IMAGE_SIZE,
    build_model,
    label_from_index,
    normalize_imagenet,
    scale_box_xyxy,
    yxyx_to_xyxy,
)
from .types import CLASS_NAMES, Detection


class TFFireDetector:
    """Runs a KerasHub RetinaNet model on BGR frames and returns detections.

    ``device`` is accepted for interface parity with the torch detector and ignored
    (TF manages devices; native Windows is CPU-only).
    """

    def __init__(
        self,
        model,
        device: Optional[str] = None,
        score_threshold: float = 0.5,
        class_names: Sequence[str] = CLASS_NAMES,
        image_size: int = DEFAULT_IMAGE_SIZE,
        exposure: str = "none",
    ):
        self.model = model
        self.score_threshold = score_threshold
        self.class_names = tuple(class_names)
        self.image_size = image_size
        self.exposure = exposure

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        arch: str = "retinanet",
        device: Optional[str] = None,
        score_threshold: float = 0.5,
        class_names: Sequence[str] = CLASS_NAMES,
        image_size: int = DEFAULT_IMAGE_SIZE,
        exposure: str = "none",
    ) -> "TFFireDetector":
        model = build_model(arch=arch, class_names=class_names)
        model.load_weights(path)
        # KerasHub's NMS decoder defaults to confidence_threshold=0.5, which silently
        # drops legitimate but under-confident detections before our own score gate ever
        # sees them. Lower it to our threshold so --score-thr is the single source of truth.
        try:
            model.prediction_decoder.confidence_threshold = min(score_threshold, 0.5)
        except AttributeError:  # pragma: no cover - decoder shape may vary by version
            pass
        return cls(model, device=device, score_threshold=score_threshold,
                   class_names=class_names, image_size=image_size, exposure=exposure)

    def _prepare(self, image_bgr: np.ndarray) -> np.ndarray:
        import cv2

        from .exposure import auto_exposure

        rgb = image_bgr[:, :, ::-1]
        rgb = auto_exposure(rgb, self.exposure)  # "none" by default = unchanged
        resized = cv2.resize(rgb, (self.image_size, self.image_size))
        return normalize_imagenet(resized)  # model has no preprocessor

    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        h, w = image_bgr.shape[:2]
        batch = self._prepare(image_bgr)[None, ...]  # (1, S, S, 3)

        preds = self.model.predict(batch, verbose=0)
        boxes = np.asarray(preds["boxes"])[0]            # (N,4) yxyx, abs px in SxS
        confidence = np.asarray(preds["confidence"])[0]  # (N,)
        labels = np.asarray(preds["labels"])[0]          # (N,) 0-indexed; -1 = padding
        n = int(np.asarray(preds["num_detections"])[0])  # valid count

        detections: List[Detection] = []
        for i in range(min(n, len(labels))):
            cls = int(labels[i])
            conf = float(confidence[i])
            if cls < 0 or conf < self.score_threshold:
                continue
            xyxy = yxyx_to_xyxy(boxes[i])
            xyxy = scale_box_xyxy(
                xyxy, from_size=(self.image_size, self.image_size), to_size=(w, h)
            )
            detections.append(
                Detection(
                    bbox=xyxy,
                    label=label_from_index(cls, self.class_names),
                    confidence=conf,
                )
            )
        return detections
