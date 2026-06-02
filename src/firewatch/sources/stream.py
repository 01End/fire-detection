"""Shared base for live cv2.VideoCapture sources (webcam, RTSP)."""
from __future__ import annotations

import time
from typing import Iterator, Union

import cv2

from .base import Frame, FrameSource


class CaptureSource(FrameSource):
    """A live source backed by ``cv2.VideoCapture``.

    ``target`` is whatever VideoCapture accepts: an int device index (webcam) or a
    stream URL (RTSP/HTTP). When ``reconnect`` is set, transient read failures trigger
    a backoff-and-reopen loop instead of ending the stream — appropriate for network
    cameras. The capture object is created lazily in ``frames()`` so constructing a
    source never touches hardware (keeps the factory and tests cheap).
    """

    def __init__(
        self,
        target: Union[int, str],
        camera_id: str,
        reconnect: bool = False,
        backoff_seconds: float = 2.0,
        max_consecutive_failures: int = 30,
    ):
        super().__init__(camera_id)
        self.target = target
        self.reconnect = reconnect
        self.backoff_seconds = backoff_seconds
        self.max_consecutive_failures = max_consecutive_failures
        self._cap = None

    def _open(self):
        cap = cv2.VideoCapture(self.target)
        if not cap.isOpened():
            raise OSError(f"Cannot open capture source: {self.target!r}")
        return cap

    def frames(self) -> Iterator[Frame]:
        self._cap = self._open()
        idx = 0
        failures = 0
        try:
            while True:
                ok, img = self._cap.read()
                if ok:
                    failures = 0
                    yield Frame(self.camera_id, img, time.time(), idx)
                    idx += 1
                    continue

                # Read failed.
                if not self.reconnect:
                    break
                failures += 1
                if failures >= self.max_consecutive_failures:
                    raise OSError(
                        f"Capture source {self.target!r} failed "
                        f"{failures} times; giving up."
                    )
                time.sleep(self.backoff_seconds)
                self._cap.release()
                self._cap = self._open()
        finally:
            self.release()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
