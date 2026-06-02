"""The structured event emitted when a fire/smoke detection is confirmed."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]


@dataclass
class DetectionEvent:
    """A confirmed detection, ready to be logged, stored, or forwarded.

    ``annotated_image`` is the optional rendered frame (boxes + floor label). It is
    carried in-memory for sinks that persist evidence, but is deliberately excluded
    from ``to_dict()`` so the metadata stays JSON-serializable.
    """

    camera_id: str
    floor: Optional[int]
    label: str
    confidence: float
    timestamp: float
    bbox: BBox
    snapshot_path: Optional[str] = None
    annotated_image: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def unmapped(self) -> bool:
        """True when the detection fell outside every configured floor zone."""
        return self.floor is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "floor": self.floor,
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "timestamp": self.timestamp,
            "bbox": [float(v) for v in self.bbox],
            "snapshot_path": self.snapshot_path,
            "unmapped": self.unmapped,
        }
