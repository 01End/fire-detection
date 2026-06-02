"""Map a detection bounding box to a floor number using per-camera zones."""
from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

from .zones import FloorZone

BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)


def bottom_center(bbox: BBox) -> Tuple[float, float]:
    """The point where an object meets the ground/floor in the image.

    Using the bottom-center (rather than the box centroid) makes floor assignment
    robust to tall plumes of smoke/flame that extend upward into the floor above.
    """
    x1, _y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


class FloorMapper:
    """Resolves which floor a detection belongs to for one camera."""

    def __init__(self, zones: Sequence[FloorZone]):
        if not zones:
            raise ValueError("FloorMapper requires at least one floor zone")
        self.zones: List[FloorZone] = list(zones)

    def locate(self, bbox: BBox) -> Optional[int]:
        """Return the floor whose zone contains the bbox's bottom-center, else None."""
        point = bottom_center(bbox)
        for zone in self.zones:
            if zone.contains(point):
                return zone.floor
        return None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "FloorMapper":
        raw = cfg.get("floors") or []
        if not raw:
            raise ValueError(
                "camera config has no 'floors' zones — run 'firewatch setup-zones' first"
            )
        zones = [
            FloorZone(floor=int(z["floor"]), polygon=[tuple(p) for p in z["polygon"]])
            for z in raw
        ]
        return cls(zones)
