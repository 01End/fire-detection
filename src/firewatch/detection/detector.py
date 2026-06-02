"""Device-agnostic detector wrapper around a torchvision detection model."""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from .model import build_model, label_from_index
from .types import CLASS_NAMES, Detection


def _select_device(prefer: Optional[str]) -> torch.device:
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FireDetector:
    """Runs a trained model on frames and returns fire/smoke detections.

    Accepts BGR uint8 images (OpenCV convention). Automatically uses CUDA when
    available, falling back to CPU, so the same code runs on the GPU laptop now and on
    CPU-only machines later.
    """

    def __init__(
        self,
        model,
        device: Optional[str] = None,
        score_threshold: float = 0.5,
        class_names: Sequence[str] = CLASS_NAMES,
    ):
        self.device = _select_device(device)
        self.model = model.to(self.device).eval()
        self.score_threshold = score_threshold
        self.class_names = tuple(class_names)

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        arch: str = "ssdlite",
        device: Optional[str] = None,
        score_threshold: float = 0.5,
        class_names: Sequence[str] = CLASS_NAMES,
    ) -> "FireDetector":
        """Build the architecture and load trained weights from ``path``."""
        model = build_model(arch=arch, class_names=class_names, pretrained_backbone=False)
        state = torch.load(path, map_location="cpu")
        # Accept either a bare state_dict or a checkpoint dict containing one.
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        return cls(model, device=device, score_threshold=score_threshold,
                   class_names=class_names)

    def _to_tensor(self, image_bgr: np.ndarray) -> torch.Tensor:
        rgb = image_bgr[:, :, ::-1]  # BGR -> RGB
        chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
        return torch.from_numpy(chw).float().div(255.0).to(self.device)

    @torch.no_grad()
    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        tensor = self._to_tensor(image_bgr)
        output = self.model([tensor])[0]

        boxes = output["boxes"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        scores = output["scores"].cpu().numpy()

        detections: List[Detection] = []
        for box, label_idx, score in zip(boxes, labels, scores):
            if score < self.score_threshold:
                continue
            detections.append(
                Detection(
                    bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    label=label_from_index(int(label_idx), self.class_names),
                    confidence=float(score),
                )
            )
        return detections
