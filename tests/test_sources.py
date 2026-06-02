"""Tests for the frame-source layer.

These avoid real cameras by driving FileSource over a folder of generated images,
which exercises the same Frame contract the live sources produce.
"""
import numpy as np
import pytest

from firewatch.sources.base import Frame, FrameSource
from firewatch.sources.file_source import FileSource
from firewatch.sources import factory


def _write_images(folder, n, color=(0, 0, 255)):
    import cv2
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = np.full((8, 8, 3), color, dtype=np.uint8)
        cv2.imwrite(str(folder / f"frame_{i:03d}.png"), img)


def test_file_source_iterates_image_folder_in_order(tmp_path):
    _write_images(tmp_path / "imgs", 3)
    src = FileSource(str(tmp_path / "imgs"), camera_id="cam1")

    frames = list(src.frames())

    assert len(frames) == 3
    assert all(isinstance(f, Frame) for f in frames)
    assert [f.index for f in frames] == [0, 1, 2]
    assert all(f.camera_id == "cam1" for f in frames)
    assert frames[0].image.shape == (8, 8, 3)


def test_file_source_skips_non_images(tmp_path):
    d = tmp_path / "imgs"
    _write_images(d, 2)
    (d / "notes.txt").write_text("ignore me")
    src = FileSource(str(d), camera_id="cam1")
    assert len(list(src.frames())) == 2


def test_file_source_is_a_context_manager(tmp_path):
    _write_images(tmp_path / "imgs", 1)
    with FileSource(str(tmp_path / "imgs"), camera_id="cam1") as src:
        assert isinstance(src, FrameSource)
        assert len(list(src.frames())) == 1


def test_factory_builds_file_source(tmp_path):
    cfg = {"camera_id": "lobby", "source": {"type": "file", "path": str(tmp_path)}}
    src = factory.build_source(cfg)
    assert isinstance(src, FileSource)
    assert src.camera_id == "lobby"


def test_factory_builds_webcam_source():
    cfg = {"camera_id": "cam2", "source": {"type": "webcam", "index": 0}}
    from firewatch.sources.webcam_source import WebcamSource

    src = factory.build_source(cfg)
    assert isinstance(src, WebcamSource)
    assert src.camera_id == "cam2"


def test_factory_builds_rtsp_source():
    cfg = {"camera_id": "cam3", "source": {"type": "rtsp", "url": "rtsp://x/y"}}
    from firewatch.sources.rtsp_source import RTSPSource

    src = factory.build_source(cfg)
    assert isinstance(src, RTSPSource)
    assert src.camera_id == "cam3"


def test_factory_rejects_unknown_type():
    with pytest.raises(ValueError):
        factory.build_source({"camera_id": "c", "source": {"type": "magic"}})


def test_factory_requires_camera_id():
    with pytest.raises(ValueError):
        factory.build_source({"source": {"type": "webcam", "index": 0}})
