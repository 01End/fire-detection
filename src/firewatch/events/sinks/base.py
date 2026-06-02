"""Event sink interface and a fault-isolating dispatcher."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Iterable

from ..event import DetectionEvent

log = logging.getLogger("firewatch.events")


class EventSink(ABC):
    """Consumes confirmed detection events (log, store, forward, ...)."""

    @abstractmethod
    def handle(self, event: DetectionEvent) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Flush/close resources. Optional; default is a no-op."""


def dispatch(event: DetectionEvent, sinks: Iterable[EventSink]) -> None:
    """Deliver an event to every sink, isolating per-sink failures.

    One sink raising must never prevent the others from receiving the event or crash
    the pipeline — a logging failure can't be allowed to drop a fire alert.
    """
    for sink in sinks:
        try:
            sink.handle(event)
        except Exception:  # noqa: BLE001 - intentional isolation boundary
            log.exception("event sink %s failed", type(sink).__name__)
