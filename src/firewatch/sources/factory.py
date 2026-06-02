"""Build a FrameSource from a camera config dict.

Config shape::

    camera_id: cam1
    source:
      type: file | webcam | rtsp
      path: ./data/clip.mp4   # file: video file or image folder
      index: 0                # webcam: device index
      url: rtsp://host/stream # rtsp: stream URL
"""
from __future__ import annotations

from typing import Any, Mapping

from .base import FrameSource
from .file_source import FileSource
from .rtsp_source import RTSPSource
from .webcam_source import WebcamSource


def build_source(cfg: Mapping[str, Any]) -> FrameSource:
    camera_id = cfg.get("camera_id")
    if not camera_id:
        raise ValueError("camera config is missing required 'camera_id'")

    source = cfg.get("source") or {}
    stype = source.get("type")

    if stype == "file":
        path = source.get("path")
        if not path:
            raise ValueError("file source requires 'path'")
        return FileSource(path, camera_id=camera_id)

    if stype == "webcam":
        return WebcamSource(index=source.get("index", 0), camera_id=camera_id)

    if stype == "rtsp":
        url = source.get("url")
        if not url:
            raise ValueError("rtsp source requires 'url'")
        return RTSPSource(url, camera_id=camera_id)

    raise ValueError(f"unknown source type: {stype!r} (expected file|webcam|rtsp)")
