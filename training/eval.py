"""Lightweight detection evaluation: precision/recall at IoU>=0.5.

Avoids extra metric dependencies (no torchmetrics/pycocotools) by computing greedy
IoU-matched precision and recall over a dataset — enough to sanity-check a trained model
and compare runs.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from firewatch.detection.model import build_model  # noqa: E402

from .dataset import YoloDetectionDataset, collate_fn  # noqa: E402


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between two [N,4] / [M,4] xyxy box sets -> [N,M]."""
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]))
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-9)


@torch.no_grad()
def evaluate(model, loader, device, iou_thr=0.5, score_thr=0.5) -> Dict[str, float]:
    model.eval()
    tp = fp = fn = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for out, tgt in zip(outputs, targets):
            keep = out["scores"].cpu() >= score_thr
            pred = out["boxes"].cpu()[keep]
            gt = tgt["boxes"]
            matched = set()
            ious = box_iou(pred, gt)
            for pi in range(pred.shape[0]):
                if gt.shape[0] == 0:
                    fp += 1
                    continue
                best = torch.argmax(ious[pi]).item()
                if ious[pi, best] >= iou_thr and best not in matched:
                    tp += 1
                    matched.add(best)
                else:
                    fp += 1
            fn += gt.shape[0] - len(matched)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a trained detector (P/R @ IoU0.5)")
    p.add_argument("--data", required=True, help="dataset root with val/")
    p.add_argument("--model", required=True, help="checkpoint to evaluate")
    p.add_argument("--arch", default="ssdlite", choices=("ssdlite", "retinanet"))
    p.add_argument("--score-thr", type=float, default=0.5)
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)

    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(arch=a.arch, pretrained_backbone=False).to(device)
    model.load_state_dict(torch.load(a.model, map_location="cpu"))
    ds = YoloDetectionDataset(
        os.path.join(a.data, "val", "images"), os.path.join(a.data, "val", "labels")
    )
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    metrics = evaluate(model, loader, device, score_thr=a.score_thr)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
