"""Live input from a local USB/built-in webcam."""
from __future__ import annotations

from .stream import CaptureSource


class WebcamSource(CaptureSource):
    """A USB/built-in camera addressed by device index (0, 1, ...)."""

    def __init__(self, index: int, camera_id: str):
        super().__init__(target=int(index), camera_id=camera_id, reconnect=False)
        self.index = int(index)
