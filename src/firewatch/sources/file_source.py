"""Recorded input: a video file or a folder of images."""
from __future__ import annotations

import glob
import os
import time
from typing import Iterator

import cv2

from .base import Frame, FrameSource

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class FileSource(FrameSource):
    """Reads frames from a video file or a directory of images.

    A directory is iterated in sorted filename order; a single file is decoded with
    OpenCV's VideoCapture (also works for most video containers).
    """

    def __init__(self, path: str, camera_id: str):
        super().__init__(camera_id)
        self.path = path

    def frames(self) -> Iterator[Frame]:
        if os.path.isdir(self.path):
            yield from self._image_folder()
        else:
            yield from self._video()

    def _image_folder(self) -> Iterator[Frame]:
        files = sorted(
            p
            for p in glob.glob(os.path.join(self.path, "*"))
            if os.path.splitext(p)[1].lower() in IMAGE_EXTS
        )
        idx = 0
        for f in files:
            img = cv2.imread(f)
            if img is None:
                continue
            yield Frame(self.camera_id, img, time.time(), idx)
            idx += 1

    def _video(self) -> Iterator[Frame]:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Source file not found: {self.path}")
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {self.path}")
        idx = 0
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                yield Frame(self.camera_id, img, time.time(), idx)
                idx += 1
        finally:
            cap.release()
