"""Find the best PER-CLASS score threshold for a TF RetinaNet, with ONE inference pass.

Why: ``tf_eval`` applies a single global ``--score-thr`` to both classes, but fire and smoke
peak at different confidences (fire is the harder class). Tuning each class independently is a
free F1 gain — no retraining — and it directly lifts fire recall, which matters for an alarm.

How: run the detector once at a low floor (``--floor 0.05``) so we keep almost every box, record
for each prediction its ``(class, score, is_tp)`` via greedy IoU matching against ground truth
(matching is done per image, predictions sorted by score — the standard PR-curve construction),
then sweep thresholds offline. The sweep math (``sweep_thresholds`` / ``best_threshold``) is pure
Python so it is unit-testable without TensorFlow.

Usage::

    PYTHONPATH=src python -m training.tf_tune_thresholds \
        --model models/fire_retinanet_indoor.weights.h5 \
        --data data/D-Fire-merged/indoor_test --image-size 384 --limit 500
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from firewatch.detection.types import CLASS_NAMES  # noqa: E402

from .tf_eval import _load_gt, box_iou  # noqa: E402
from .tf_dataset import _label_for, _list_images, load_class_map  # noqa: E402


def match_with_scores(
    pred_boxes: Sequence, pred_scores: Sequence, gt_boxes: Sequence, iou_thr: float = 0.5
) -> List[Tuple[float, int]]:
    """Greedy match one image's predictions (single class) to its GT, score-ordered.

    Returns a list of ``(score, is_tp)`` — one entry per prediction. A prediction is a TP if it
    claims an as-yet-unclaimed GT box with IoU >= ``iou_thr`` (highest score claims first).
    """
    order = sorted(range(len(pred_boxes)), key=lambda i: -pred_scores[i])
    out: List[Tuple[float, int]] = []
    if len(gt_boxes) == 0:
        return [(float(pred_scores[i]), 0) for i in order]
    ious = box_iou(np.asarray(pred_boxes), np.asarray(gt_boxes))
    claimed: set = set()
    for i in order:
        row = ious[i]
        best = int(np.argmax(row)) if row.size else -1
        if best >= 0 and row[best] >= iou_thr and best not in claimed:
            claimed.add(best)
            out.append((float(pred_scores[i]), 1))
        else:
            out.append((float(pred_scores[i]), 0))
    return out


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def sweep_thresholds(
    scored: Sequence[Tuple[float, int]], total_gt: int, grid: Sequence[float]
) -> List[Tuple[float, float, float, float, int, int, int]]:
    """For each threshold in ``grid`` -> (thr, precision, recall, f1, tp, fp, fn).

    ``scored`` is the full set of ``(score, is_tp)`` predictions for one class (any score). At a
    threshold ``t`` we keep predictions with score >= t; FN = total_gt - TP(kept). Pure function.
    """
    rows = []
    for t in grid:
        tp = sum(1 for s, hit in scored if s >= t and hit)
        fp = sum(1 for s, hit in scored if s >= t and not hit)
        fn = total_gt - tp
        p, r, f1 = _prf(tp, fp, fn)
        rows.append((float(t), p, r, f1, tp, fp, fn))
    return rows


def best_threshold(rows: Sequence[Tuple[float, float, float, float, int, int, int]]):
    """Pick the row with the highest F1 (ties -> lower threshold = higher recall)."""
    return max(rows, key=lambda row: (row[3], -row[0]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Per-class score-threshold tuner (one inference pass)")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, help="split dir with images/ and labels/")
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--floor", type=float, default=0.05, help="low score floor for the single pass")
    p.add_argument("--limit", type=int, default=500, help="<=0 means all images")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exposure", default="none", choices=("none", "clahe", "gamma"))
    p.add_argument("--baseline", type=float, default=0.6,
                   help="current global threshold to compare against")
    a = p.parse_args(argv)

    images_dir = os.path.join(a.data, "images")
    labels_dir = os.path.join(a.data, "labels")
    if not os.path.isdir(images_dir):
        raise SystemExit(f"no images/ dir under {a.data!r}")
    class_map = load_class_map(a.data)

    import cv2
    from firewatch.detection.tf_detector import TFFireDetector

    detector = TFFireDetector.from_checkpoint(
        a.model, arch="retinanet", score_threshold=a.floor, image_size=a.image_size,
        exposure=a.exposure,
    )

    paths = _list_images(images_dir)
    if a.limit and 0 < a.limit < len(paths):
        rng = np.random.default_rng(a.seed)
        pick = rng.choice(len(paths), size=a.limit, replace=False)
        paths = [paths[i] for i in sorted(pick.tolist())]

    scored: Dict[str, List[Tuple[float, int]]] = {c: [] for c in CLASS_NAMES}
    total_gt: Dict[str, int] = {c: 0 for c in CLASS_NAMES}

    print(f"Tuning on {len(paths)} images from {a.data} (floor={a.floor}) ...")
    for i, path in enumerate(paths):
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        dets = detector.detect(img)
        gt_boxes, gt_labels = _load_gt(_label_for(path, labels_dir), w, h, class_map)
        for c in CLASS_NAMES:
            pb = [d.bbox for d in dets if d.label == c]
            ps = [d.confidence for d in dets if d.label == c]
            gb = [b for b, l in zip(gt_boxes, gt_labels) if l == c]
            total_gt[c] += len(gb)
            scored[c].extend(match_with_scores(pb, ps, gb, a.iou_thr))
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(paths)}")

    grid = [round(x, 2) for x in np.arange(0.10, 0.91, 0.05)]
    print(f"\n=== Per-class threshold sweep (IoU>={a.iou_thr}, {len(paths)} images) ===")
    best: Dict[str, tuple] = {}
    for c in CLASS_NAMES:
        rows = sweep_thresholds(scored[c], total_gt[c], grid)
        best[c] = best_threshold(rows)
        bt, bp, br, bf, *_ = best[c]
        print(f"\n[{c}] GT={total_gt[c]}  best thr={bt:.2f}  P={bp:.3f} R={br:.3f} F1={bf:.3f}")
        for t, pp, rr, ff, tp, fp, fn in rows:
            mark = "  <= best" if t == bt else ""
            print(f"   thr {t:.2f}: P={pp:.3f} R={rr:.3f} F1={ff:.3f} (TP={tp} FP={fp} FN={fn}){mark}")

    # Combined (micro-avg) F1 at the tuned per-class thresholds vs the global baseline.
    def combined(thr_of):
        tp = fp = fn = 0
        for c in CLASS_NAMES:
            t = thr_of(c)
            tpc = sum(1 for s, hit in scored[c] if s >= t and hit)
            fpc = sum(1 for s, hit in scored[c] if s >= t and not hit)
            tp += tpc; fp += fpc; fn += total_gt[c] - tpc
        return _prf(tp, fp, fn)

    tuned_p, tuned_r, tuned_f = combined(lambda c: best[c][0])
    base_p, base_r, base_f = combined(lambda c: a.baseline)
    print("\n=== Combined (both classes) ===")
    print(f"  baseline  (global {a.baseline:.2f}): P={base_p:.3f} R={base_r:.3f} F1={base_f:.3f}")
    tuned_str = ", ".join(f"{c}={best[c][0]:.2f}" for c in CLASS_NAMES)
    print(f"  tuned     ({tuned_str}): P={tuned_p:.3f} R={tuned_r:.3f} F1={tuned_f:.3f}")
    delta = tuned_f - base_f
    print(f"  F1 delta: {delta:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
