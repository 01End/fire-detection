"""Load a camera config and assemble a Pipeline from it."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import yaml

from .events.debounce import FloorDebouncer
from .events.sinks.company_stub import CompanySystemSink
from .events.sinks.evidence import EvidenceSink
from .events.sinks.log_sink import LogSink
from .events.store import EventStore
from .floors.mapper import FloorMapper
from .pipeline import Pipeline
from .sources import factory


def load_camera_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"camera config {path} must be a mapping")
    return cfg


def build_pipeline(
    cfg: Dict[str, Any],
    model_path: str,
    output_dir: Optional[str] = None,
    frame_hook=None,
) -> Pipeline:
    """Construct a fully-wired Pipeline from a parsed camera config."""
    from .detection.detector import FireDetector  # local import: keeps torch optional

    source = factory.build_source(cfg)
    mapper = FloorMapper.from_config(cfg)

    det_cfg = cfg.get("detection", {}) or {}
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"model checkpoint not found: {model_path}\n"
            "Train one with the scripts in training/, or pass --model."
        )
    detector = FireDetector.from_checkpoint(
        model_path,
        arch=det_cfg.get("arch", "ssdlite"),
        score_threshold=float(det_cfg.get("score_threshold", 0.5)),
    )

    db_cfg = cfg.get("debounce", {}) or {}
    debouncer = FloorDebouncer(
        confirm_n=int(db_cfg.get("confirm_n", 3)),
        window_m=int(db_cfg.get("window_m", 5)),
        cooldown_seconds=float(db_cfg.get("cooldown_seconds", 30)),
    )

    output_dir = output_dir or cfg.get("output_dir", "output")
    store = EventStore(output_dir)
    sinks = [
        LogSink(),
        EvidenceSink(store),
        CompanySystemSink(enabled=bool(cfg.get("company_system", {}).get("enabled", False))),
    ]

    return Pipeline(source, detector, mapper, debouncer, sinks, frame_hook=frame_hook)
