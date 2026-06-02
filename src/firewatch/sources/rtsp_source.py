"""Live input from a networked IP camera over RTSP/HTTP."""
from __future__ import annotations

from .stream import CaptureSource


class RTSPSource(CaptureSource):
    """A network camera stream. Reconnects on transient read failures by default."""

    def __init__(
        self,
        url: str,
        camera_id: str,
        reconnect: bool = True,
        backoff_seconds: float = 2.0,
    ):
        super().__init__(
            target=url,
            camera_id=camera_id,
            reconnect=reconnect,
            backoff_seconds=backoff_seconds,
        )
        self.url = url
