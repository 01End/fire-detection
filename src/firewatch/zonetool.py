"""Author per-camera floor zones.

Two ways to produce the ``floors`` section of a camera config:
  * ``auto_bands`` — split the frame into N equal horizontal bands (headless, great as a
    starting point when floors stack evenly in view).
  * ``collect_polygons_interactive`` — draw a polygon per floor on a snapshot (needs a
    display). Both feed ``save_zones`` which writes the config back.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import yaml

from .sources import factory


def snapshot_from_config(cfg: Dict[str, Any]):
    """Grab the first frame from the camera/source described by ``cfg``."""
    source = factory.build_source(cfg)
    try:
        for frame in source.frames():
            return frame.image
    finally:
        source.release()
    raise RuntimeError("could not read a frame from the configured source")


def auto_bands(
    width: int, height: int, n_floors: int, top_floor: Optional[int] = None
) -> List[Dict[str, Any]]:
    """N equal horizontal bands. Top band = highest floor number, descending downward."""
    if n_floors < 1:
        raise ValueError("n_floors must be >= 1")
    top = n_floors if top_floor is None else top_floor
    band = height / n_floors
    zones: List[Dict[str, Any]] = []
    for i in range(n_floors):
        y0 = int(round(i * band))
        y1 = int(round((i + 1) * band))
        zones.append(
            {
                "floor": top - i,
                "polygon": [[0, y0], [width, y0], [width, y1], [0, y1]],
            }
        )
    return zones


def save_zones(config_path: str, cfg: Dict[str, Any], zones: List[Dict[str, Any]]) -> None:
    """Write ``zones`` into the config's ``floors`` key and persist as YAML."""
    cfg = dict(cfg)
    cfg["floors"] = zones
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)


def collect_polygons_interactive(image) -> List[Dict[str, Any]]:  # pragma: no cover
    """Open an OpenCV window to draw one polygon per floor (requires a display).

    Controls: left-click to add a vertex; 'n' to finish the current floor (you'll be
    prompted for its floor number in the console); 'u' to undo a vertex; 's' to save and
    exit; 'q' to cancel.
    """
    import cv2

    zones: List[Dict[str, Any]] = []
    current: List[List[int]] = []
    window = "FireWatch: draw floor zones"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append([x, y])

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        canvas = image.copy()
        for z in zones:
            pts = z["polygon"]
            for a, b in zip(pts, pts[1:] + pts[:1]):
                cv2.line(canvas, tuple(a), tuple(b), (0, 255, 0), 2)
        for p in current:
            cv2.circle(canvas, tuple(p), 3, (0, 0, 255), -1)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("u") and current:
            current.pop()
        elif key == ord("n") and len(current) >= 3:
            floor = int(input("floor number for this zone: "))
            zones.append({"floor": floor, "polygon": [list(p) for p in current]})
            current = []
        elif key == ord("s"):
            break
        elif key == ord("q"):
            zones = []
            break

    cv2.destroyWindow(window)
    return zones
