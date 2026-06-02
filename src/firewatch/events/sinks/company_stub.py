"""Placeholder sink for the future hand-off to the company's central system.

The integration contract (transport, auth, payload schema) is deliberately deferred.
When it's defined, implement ``forward()`` — the rest of the pipeline already routes
confirmed events here, so wiring it up is a localized change.
"""
from __future__ import annotations

import logging

from ..event import DetectionEvent
from .base import EventSink

log = logging.getLogger("firewatch.company")


class CompanySystemSink(EventSink):
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def handle(self, event: DetectionEvent) -> None:
        if not self.enabled:
            return
        self.forward(event)

    def forward(self, event: DetectionEvent) -> None:
        # TODO(company-integration): replace with the real transport once the
        # central system's API/contract is finalized (see spec "Open items").
        log.info(
            "would forward to company system: %s on floor %s (camera %s)",
            event.label, event.floor, event.camera_id,
        )
