"""Per-camera orchestrator: source -> detector -> floor mapper -> debounce -> sinks."""
from __future__ import annotations

from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple

from .annotate import annotate
from .detection.types import Detection
from .events.debounce import FloorDebouncer
from .events.event import DetectionEvent
from .events.sinks.base import EventSink, dispatch
from .floors.mapper import FloorMapper
from .sources.base import Frame, FrameSource

# key = (floor, label); floor may be None for unmapped detections.
Key = Tuple[Optional[int], str]


class Pipeline:
    """Wires the stages for one camera and drives them frame by frame.

    The detector is injected (any object with ``detect(image) -> list[Detection]``),
    which keeps this module free of heavy ML imports and makes it fully testable with a
    stub detector.
    """

    def __init__(
        self,
        source: FrameSource,
        detector,
        mapper: FloorMapper,
        debouncer: FloorDebouncer,
        sinks: Sequence[EventSink],
        frame_hook: Optional[Callable[[Frame, List[Tuple[Detection, Optional[int]]]], None]] = None,
    ):
        self.source = source
        self.detector = detector
        self.mapper = mapper
        self.debouncer = debouncer
        self.sinks = list(sinks)
        # Optional callback invoked every frame with all (detection, floor) pairs —
        # used for live display; left None for headless runs.
        self.frame_hook = frame_hook
        # Most recent qualifying detection + frame per key, used to build evidence
        # when a key is confirmed (which may lag the last frame it was seen in).
        self._last: Dict[Key, Tuple[Detection, Frame]] = {}

    def process(self, frame: Frame) -> List[DetectionEvent]:
        """Process one frame; return any events confirmed on it."""
        detections = self.detector.detect(frame.image)

        mapped: List[Tuple[Detection, Optional[int]]] = [
            (det, self.mapper.locate(det.bbox)) for det in detections
        ]
        if self.frame_hook is not None:
            self.frame_hook(frame, mapped)

        active: Dict[Key, Detection] = {}
        for det, floor in mapped:
            key: Key = (floor, det.label)
            # Keep the highest-confidence detection per key for this frame.
            if key not in active or det.confidence > active[key].confidence:
                active[key] = det

        for key, det in active.items():
            self._last[key] = (det, frame)

        confirmed_keys: List[Hashable] = self.debouncer.update(
            set(active.keys()), frame.timestamp
        )

        events: List[DetectionEvent] = []
        for key in confirmed_keys:
            det, src_frame = self._last[key]
            floor, label = key
            image = annotate(src_frame.image, [(det, floor)])
            event = DetectionEvent(
                camera_id=frame.camera_id,
                floor=floor,
                label=label,
                confidence=det.confidence,
                timestamp=frame.timestamp,
                bbox=det.bbox,
                annotated_image=image,
            )
            dispatch(event, self.sinks)
            events.append(event)
        return events

    def run(self, max_frames: Optional[int] = None) -> None:
        """Consume the source until exhausted (files) or ``max_frames`` is reached."""
        try:
            for n, frame in enumerate(self.source.frames()):
                if max_frames is not None and n >= max_frames:
                    break
                self.process(frame)
        finally:
            self.source.release()
            for sink in self.sinks:
                sink.close()
