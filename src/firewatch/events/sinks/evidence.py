"""Sink that persists evidence: an annotated snapshot image + a metadata record.

This doubles as the dashboard's data source — the Streamlit app reads the same
``EventStore`` (events.jsonl + snapshots/) that this sink writes.
"""
from __future__ import annotations

import os
import time

import cv2

from ..event import DetectionEvent
from ..store import EventStore
from .base import EventSink


class EvidenceSink(EventSink):
    def __init__(self, store: EventStore):
        self.store = store

    def handle(self, event: DetectionEvent) -> None:
        if event.annotated_image is not None and event.snapshot_path is None:
            fname = f"{event.camera_id}_{int(event.timestamp * 1000)}_{int(time.time()*1000)%1000}.jpg"
            path = os.path.join(self.store.snapshots_dir, fname)
            cv2.imwrite(path, event.annotated_image)
            event.snapshot_path = path
        self.store.append(event.to_dict())
