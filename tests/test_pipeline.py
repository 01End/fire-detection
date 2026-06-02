"""End-to-end pipeline integration test with a deterministic stub detector."""
import cv2
import numpy as np

from firewatch.detection.types import Detection
from firewatch.events.debounce import FloorDebouncer
from firewatch.events.sinks.base import EventSink
from firewatch.events.sinks.evidence import EvidenceSink
from firewatch.events.store import EventStore
from firewatch.floors.mapper import FloorMapper
from firewatch.floors.zones import FloorZone
from firewatch.pipeline import Pipeline
from firewatch.sources.file_source import FileSource


class StubDetector:
    """Returns a fixed fire detection whose box sits on a known floor."""

    def __init__(self, bbox, label="fire", confidence=0.9):
        self._det = Detection(bbox=bbox, label=label, confidence=confidence)

    def detect(self, image):
        return [self._det]


class _Recorder(EventSink):
    def __init__(self):
        self.events = []

    def handle(self, event):
        self.events.append(event)


def _make_clip(folder, n=5, size=200):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.imwrite(str(folder / f"f{i:03d}.png"), img)


# Two stacked floors: top band = floor 2, bottom band = floor 1.
ZONES = [
    FloorZone(floor=2, polygon=[(0, 0), (200, 0), (200, 100), (0, 100)]),
    FloorZone(floor=1, polygon=[(0, 100), (200, 100), (200, 200), (0, 200)]),
]


def test_pipeline_emits_event_with_correct_floor_and_evidence(tmp_path):
    _make_clip(tmp_path / "clip", n=5)

    # A box whose bottom-center is at (100, 150) -> floor 1.
    detector = StubDetector(bbox=(90, 130, 110, 150))
    source = FileSource(str(tmp_path / "clip"), camera_id="cam1")
    mapper = FloorMapper(ZONES)
    debouncer = FloorDebouncer(confirm_n=3, window_m=5, cooldown_seconds=1000)
    store = EventStore(str(tmp_path / "out"))
    recorder = _Recorder()

    pipeline = Pipeline(source, detector, mapper, debouncer,
                        sinks=[EvidenceSink(store), recorder])
    pipeline.run()

    # 5 frames, confirm_n=3, long cooldown -> exactly one confirmed event.
    assert len(recorder.events) == 1
    ev = recorder.events[0]
    assert ev.floor == 1
    assert ev.label == "fire"

    # Evidence persisted: snapshot file on disk + one record in the store.
    import os
    assert ev.snapshot_path and os.path.exists(ev.snapshot_path)
    recent = store.recent()
    assert len(recent) == 1 and recent[0]["floor"] == 1


def test_pipeline_marks_detection_outside_zones_as_unmapped(tmp_path):
    _make_clip(tmp_path / "clip", n=3)
    # Bottom-center at (100, 199) is inside floor 1 zone... move it outside all zones.
    detector = StubDetector(bbox=(90, 300, 110, 320))
    source = FileSource(str(tmp_path / "clip"), camera_id="cam9")
    mapper = FloorMapper(ZONES)
    debouncer = FloorDebouncer(confirm_n=2, window_m=3, cooldown_seconds=1000)
    recorder = _Recorder()

    Pipeline(source, detector, mapper, debouncer, sinks=[recorder]).run()

    assert len(recorder.events) == 1
    assert recorder.events[0].unmapped is True
    assert recorder.events[0].floor is None
