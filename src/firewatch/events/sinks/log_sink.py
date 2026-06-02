"""Sink that writes a human-readable log line per event."""
from __future__ import annotations

import logging

from ..event import DetectionEvent
from .base import EventSink

log = logging.getLogger("firewatch.alert")


class LogSink(EventSink):
    def handle(self, event: DetectionEvent) -> None:
        floor = f"floor {event.floor}" if event.floor is not None else "UNMAPPED location"
        log.warning(
            "FIRE ALERT: %s detected on %s (camera %s, confidence %.2f)",
            event.label,
            floor,
            event.camera_id,
            event.confidence,
        )
