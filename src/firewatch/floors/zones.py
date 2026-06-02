"""Polygon floor zones and point-in-polygon geometry.

Implemented with the standard ray-casting algorithm so there is no dependency on
shapely or other geometry libraries — keeping the install light and license-clean.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

Point = Tuple[float, float]


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Return True if ``point`` lies inside ``polygon`` (ray-casting).

    Points exactly on an edge are treated as implementation-defined; zones should be
    drawn with a small margin so detections never land precisely on a boundary.
    """
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


@dataclass
class FloorZone:
    """A floor number paired with the image polygon that represents it."""

    floor: int
    polygon: List[Point]

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError(
                f"floor {self.floor}: a zone polygon needs at least 3 points, "
                f"got {len(self.polygon)}"
            )
        # Normalise to tuples of floats for consistent geometry.
        self.polygon = [(float(x), float(y)) for x, y in self.polygon]

    def contains(self, point: Point) -> bool:
        return point_in_polygon(point, self.polygon)
