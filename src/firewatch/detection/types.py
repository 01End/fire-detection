"""Detector output types. Intentionally free of torch imports so the rest of the
pipeline (and its tests) can use these without pulling in heavy ML dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)

# Class labels the detector produces (index 0 is reserved for background).
FIRE = "fire"
SMOKE = "smoke"
CLASS_NAMES = (FIRE, SMOKE)


@dataclass
class Detection:
    """A single detected fire/smoke region in one frame."""

    bbox: BBox
    label: str
    confidence: float
