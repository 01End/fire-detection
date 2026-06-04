"""TF-backend detection evaluation: precision / recall / F1 at IoU>=0.5.

Mirrors ``training/eval.py`` but runs the KerasHub RetinaNet via ``TFFireDetector`` so it
can evaluate a ``.weights.h5`` model (the torch ``eval.py`` only loads a ``state_dict``).
Pure-NumPy IoU matching — no torch, no pycocotools.

Usage::

    PYTHONPATH=src python -m training.tf_eval \
        --model models/fire_retinanet.weights.h5 \
        --data D-Fire/test --image-size 512 --score-thr 0.3 --limit 300
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from firewatch.detection.tf_model import label_from_index  # noqa: E402
from firewatch.detection.types import CLASS_NAMES  # noqa: E402

from .tf_dataset import _label_for, _list_images, load_class_map  # noqa: E402


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two xyxy box sets, [N,4] vs [M,4] -> [N,M]."""
    a = np.asarray(a, dtype="float32").reshape(-1, 4)
    b = np.asarray(b, dtype="float32").reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype="float32")
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def greedy_match(
    pred_boxes: Sequence, gt_boxes: Sequence, iou_thr: float = 0.5
) -> Tuple[int, int, int]:
    """Greedy IoU matching for one image -> (tp, fp, fn), class-agnostic.

    Each predicted box claims the highest-IoU unmatched ground-truth box; a claim above
    ``iou_thr`` is a true positive, otherwise a false positive. Unclaimed ground truths
    are false negatives.
    """
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)
    if n_gt == 0:
        return 0, n_pred, 0
    ious = box_iou(np.asarray(pred_boxes), np.asarray(gt_boxes))
    matched: set = set()
    tp = fp = 0
    for pi in range(n_pred):
        best = int(np.argmax(ious[pi]))
        if ious[pi, best] >= iou_thr and best not in matched:
            tp += 1
            matched.add(best)
        else:
            fp += 1
    fn = n_gt - len(matched)
    return tp, fp, fn


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _load_gt(label_path: str, w: int, h: int, class_map) -> Tuple[List[list], List[str]]:
    """Parse a YOLO label file -> (xyxy boxes in original px, label strings)."""
    boxes: List[list] = []
    labels: List[str] = []
    if not os.path.exists(label_path):
        return boxes, labels
    with open(label_path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, cx, cy, bw, bh = (float(p) for p in parts)
            our = class_map.get(int(cls))
            if our is None:
                continue
            boxes.append([
                (cx - bw / 2) * w, (cy - bh / 2) * h,
                (cx + bw / 2) * w, (cy + bh / 2) * h,
            ])
            labels.append(label_from_index(our))
    return boxes, labels


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a TF RetinaNet (P/R/F1 @ IoU0.5)")
    p.add_argument("--model", required=True, help="TF .weights.h5 checkpoint")
    p.add_argument("--data", required=True, help="split dir containing images/ and labels/")
    p.add_argument("--image-size", type=int, default=512, help="match the training size")
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=300,
                   help="evaluate at most N images (CPU is slow); <=0 means all")
    p.add_argument("--seed", type=int, default=0,
                   help="random seed for sampling when --limit is set")
    a = p.parse_args(argv)

    images_dir = os.path.join(a.data, "images")
    labels_dir = os.path.join(a.data, "labels")
    if not os.path.isdir(images_dir):
        raise SystemExit(f"no images/ dir under {a.data!r}")
    class_map = load_class_map(a.data)

    # Imported here so a plain --help / unit tests don't spin up TensorFlow.
    import cv2
    from firewatch.detection.tf_detector import TFFireDetector

    detector = TFFireDetector.from_checkpoint(
        a.model, arch="retinanet", score_threshold=a.score_thr, image_size=a.image_size
    )

    paths = _list_images(images_dir)
    if not paths:
        raise SystemExit(f"no images found in {images_dir!r}")
    # Sample RANDOMLY across the whole split, not the first N — D-Fire groups many
    # negative (empty-label) images together, so a head slice would miss all positives.
    if a.limit and 0 < a.limit < len(paths):
        rng = np.random.default_rng(a.seed)
        pick = rng.choice(len(paths), size=a.limit, replace=False)
        paths = [paths[i] for i in sorted(pick.tolist())]

    agg = {"tp": 0, "fp": 0, "fn": 0}
    per: Dict[str, Dict[str, int]] = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASS_NAMES}

    print(f"Evaluating {len(paths)} images from {a.data} ...")
    for i, path in enumerate(paths):
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        dets = detector.detect(img)
        pred_boxes = [d.bbox for d in dets]
        pred_labels = [d.label for d in dets]
        gt_boxes, gt_labels = _load_gt(_label_for(path, labels_dir), w, h, class_map)

        tp, fp, fn = greedy_match(pred_boxes, gt_boxes, a.iou_thr)
        agg["tp"] += tp; agg["fp"] += fp; agg["fn"] += fn

        for c in CLASS_NAMES:
            pb = [b for b, l in zip(pred_boxes, pred_labels) if l == c]
            gb = [b for b, l in zip(gt_boxes, gt_labels) if l == c]
            tpc, fpc, fnc = greedy_match(pb, gb, a.iou_thr)
            per[c]["tp"] += tpc; per[c]["fp"] += fpc; per[c]["fn"] += fnc

        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(paths)}")

    pr, rc, f1 = _prf(agg["tp"], agg["fp"], agg["fn"])
    print(f"\n=== Results @ IoU>={a.iou_thr}, score>={a.score_thr} "
          f"({len(paths)} images) ===")
    print(f"Overall (any class): precision={pr:.3f}  recall={rc:.3f}  f1={f1:.3f}  "
          f"(TP={agg['tp']} FP={agg['fp']} FN={agg['fn']})")
    for c in CLASS_NAMES:
        pc, rcc, fc = _prf(per[c]["tp"], per[c]["fp"], per[c]["fn"])
        print(f"  {c:<5}: precision={pc:.3f}  recall={rcc:.3f}  f1={fc:.3f}  "
              f"(TP={per[c]['tp']} FP={per[c]['fp']} FN={per[c]['fn']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
