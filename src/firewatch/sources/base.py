"""Common frame-source interface.

A `FrameSource` yields `Frame`s. Recorded files and live streams implement the same
contract so the rest of the pipeline never needs to know where frames come from.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class Frame:
    """A single captured frame.

    image is a BGR ``HxWx3`` uint8 array (OpenCV convention).
    timestamp is epoch seconds; index is a 0-based per-source counter.
    """

    camera_id: str
    image: np.ndarray
    timestamp: float
    index: int


class FrameSource(ABC):
    """Abstract source of frames for one camera."""

    def __init__(self, camera_id: str):
        self.camera_id = camera_id

    @abstractmethod
    def frames(self) -> Iterator[Frame]:
        """Yield frames until the source is exhausted (files) or stopped (streams)."""
        raise NotImplementedError

    def release(self) -> None:
        """Release any underlying resources. Safe to call multiple times."""

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> bool:
        self.release()
        return False
