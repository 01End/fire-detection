"""Draw detection boxes and floor labels onto a frame for evidence/dashboard."""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import cv2
import numpy as np

from .detection.types import Detection

_FIRE_COLOR = (0, 0, 255)    # red (BGR)
_SMOKE_COLOR = (160, 160, 160)  # grey


def _color(label: str) -> Tuple[int, int, int]:
    return _FIRE_COLOR if label == "fire" else _SMOKE_COLOR


def annotate(
    image: np.ndarray,
    detections: Iterable[Tuple[Detection, Optional[int]]],
) -> np.ndarray:
    """Return a copy of ``image`` with each (detection, floor) drawn on it."""
    out = image.copy()
    for det, floor in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        color = _color(det.label)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        floor_txt = f"floor {floor}" if floor is not None else "UNMAPPED"
        caption = f"{det.label} {det.confidence:.2f} | {floor_txt}"
        y_text = max(y1 - 6, 12)
        cv2.putText(
            out, caption, (x1, y_text),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
        )
    return out
