"""YOLO-format detection dataset adapter for torchvision detection models.

Public fire/smoke datasets (D-Fire, FASDD) ship YOLO-style annotations: one ``.txt``
per image with lines ``class cx cy w h`` (all normalized to [0,1]). This adapter
converts them to the ``{boxes(xyxy, abs), labels}`` target format torchvision expects,
remapping dataset class ids to our 1-based indices (1=fire, 2=smoke; 0=background).
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Mapping, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# D-Fire convention: class 0 = smoke, 1 = fire. Map to our labels (fire=1, smoke=2).
DEFAULT_CLASS_MAP: Mapping[int, int] = {1: 1, 0: 2}


def _label_path_for(image_path: str, labels_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(labels_dir, stem + ".txt")


class YoloDetectionDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        class_map: Mapping[int, int] = DEFAULT_CLASS_MAP,
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.class_map = dict(class_map)
        self.images: List[str] = sorted(
            p
            for p in glob.glob(os.path.join(images_dir, "*"))
            if os.path.splitext(p)[1].lower() in IMAGE_EXTS
        )
        if not self.images:
            raise FileNotFoundError(f"no images found in {images_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def _read_target(self, label_path: str, w: int, h: int):
        boxes: List[List[float]] = []
        labels: List[int] = []
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    cls, cx, cy, bw, bh = (float(p) for p in parts)
                    mapped = self.class_map.get(int(cls))
                    if mapped is None:
                        continue
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w
                    y2 = (cy + bh / 2) * h
                    boxes.append([x1, y1, x2, y2])
                    labels.append(mapped)

        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        return boxes_t, labels_t, area

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_path = self.images[idx]
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise OSError(f"could not read image {image_path}")
        h, w = bgr.shape[:2]
        rgb = np.ascontiguousarray(bgr[:, :, ::-1].transpose(2, 0, 1))
        image = torch.from_numpy(rgb).float().div(255.0)

        boxes, labels, area = self._read_target(
            _label_path_for(image_path, self.labels_dir), w, h
        )
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((labels.shape[0],), dtype=torch.int64),
        }
        return image, target


def collate_fn(batch):
    """Detection batches are ragged: keep images and targets as parallel lists."""
    return tuple(zip(*batch))
