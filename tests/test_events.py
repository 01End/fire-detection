"""Tests for the detection event model, debouncer, and sinks."""
import numpy as np
import pytest

from firewatch.events.event import DetectionEvent
from firewatch.events.debounce import FloorDebouncer
from firewatch.events.sinks.base import EventSink, dispatch
from firewatch.events.sinks.evidence import EvidenceSink
from firewatch.events.store import EventStore


# --- DetectionEvent ---------------------------------------------------------

def test_event_to_dict_is_serializable_and_omits_image():
    ev = DetectionEvent(
        camera_id="cam1", floor=3, label="fire", confidence=0.91,
        timestamp=1000.0, bbox=(1, 2, 3, 4),
        annotated_image=np.zeros((4, 4, 3), dtype=np.uint8),
    )
    d = ev.to_dict()
    assert d["floor"] == 3 and d["label"] == "fire"
    assert "annotated_image" not in d
    import json
    json.dumps(d)  # must not raise


def test_event_unmapped_when_floor_is_none():
    ev = DetectionEvent("cam1", None, "smoke", 0.7, 1.0, (0, 0, 1, 1))
    assert ev.unmapped is True


# --- FloorDebouncer ---------------------------------------------------------

def test_debouncer_confirms_after_n_consecutive():
    db = FloorDebouncer(confirm_n=3, window_m=5, cooldown_seconds=100)
    key = (3, "fire")
    assert db.update({key}, timestamp=1.0) == []          # 1 hit
    assert db.update({key}, timestamp=2.0) == []          # 2 hits
    assert db.update({key}, timestamp=3.0) == [key]       # 3rd hit -> confirmed


def test_debouncer_confirms_on_n_of_m_not_only_consecutive():
    db = FloorDebouncer(confirm_n=3, window_m=5, cooldown_seconds=100)
    key = (2, "fire")
    seq = [True, False, True, False, True]   # 3 hits within a 5-frame window
    confirmed = None
    for i, present in enumerate(seq):
        active = {key} if present else set()
        out = db.update(active, timestamp=float(i))
        if out:
            confirmed = i
    assert confirmed == 4


def test_debouncer_respects_cooldown_then_re_alerts():
    db = FloorDebouncer(confirm_n=2, window_m=3, cooldown_seconds=10)
    key = (1, "fire")
    db.update({key}, timestamp=0.0)
    assert db.update({key}, timestamp=1.0) == [key]       # confirmed
    # During cooldown: no repeat even though still firing.
    assert db.update({key}, timestamp=2.0) == []
    assert db.update({key}, timestamp=5.0) == []
    # Once the cooldown elapses, a still-active fire re-alerts on the next frame.
    assert db.update({key}, timestamp=20.0) == [key]


def test_debouncer_tracks_floors_independently():
    db = FloorDebouncer(confirm_n=2, window_m=3, cooldown_seconds=100)
    db.update({(1, "fire")}, 0.0)
    out = db.update({(1, "fire"), (2, "fire")}, 1.0)
    assert out == [(1, "fire")]            # only floor 1 reached 2 hits
    assert db.update({(2, "fire")}, 2.0) == [(2, "fire")]


# --- Sinks ------------------------------------------------------------------

class _RecordingSink(EventSink):
    def __init__(self):
        self.events = []

    def handle(self, event):
        self.events.append(event)


class _ExplodingSink(EventSink):
    def handle(self, event):
        raise RuntimeError("boom")


def _event(**kw):
    base = dict(camera_id="cam1", floor=2, label="fire", confidence=0.8,
                timestamp=1.0, bbox=(0, 0, 10, 10))
    base.update(kw)
    return DetectionEvent(**base)


def test_dispatch_delivers_to_all_sinks():
    a, b = _RecordingSink(), _RecordingSink()
    dispatch(_event(), [a, b])
    assert len(a.events) == 1 and len(b.events) == 1


def test_dispatch_isolates_failing_sink():
    good = _RecordingSink()
    # A failing sink must not stop the others.
    dispatch(_event(), [_ExplodingSink(), good])
    assert len(good.events) == 1


def test_evidence_sink_writes_snapshot_and_store(tmp_path):
    store = EventStore(str(tmp_path))
    sink = EvidenceSink(store)
    ev = _event(annotated_image=np.zeros((8, 8, 3), dtype=np.uint8))
    sink.handle(ev)

    assert ev.snapshot_path is not None
    import os
    assert os.path.exists(ev.snapshot_path)
    recent = store.recent()
    assert len(recent) == 1
    assert recent[0]["floor"] == 2
    assert recent[0]["snapshot_path"] == ev.snapshot_path
