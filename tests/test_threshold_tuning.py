"""Unit tests for the pure threshold-sweep math in training/tf_tune_thresholds.

These exercise the sweep/argmax logic without TensorFlow (the heavy import is lazy in main()).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from training.tf_tune_thresholds import (  # noqa: E402
    best_threshold,
    match_with_scores,
    sweep_thresholds,
)


def test_sweep_counts_tp_fp_fn_at_each_threshold():
    # two TPs (0.9, 0.4) and one FP (0.7); 3 GT total -> one GT always missed.
    scored = [(0.9, 1), (0.7, 0), (0.4, 1)]
    rows = {r[0]: r for r in sweep_thresholds(scored, total_gt=3, grid=[0.5, 0.8])}
    # thr 0.5: keep 0.9(tp),0.7(fp) -> tp1 fp1 fn2
    _, p, r, f1, tp, fp, fn = rows[0.5]
    assert (tp, fp, fn) == (1, 1, 2)
    assert abs(p - 0.5) < 1e-9 and abs(r - 1 / 3) < 1e-9
    # thr 0.8: keep only 0.9(tp) -> tp1 fp0 fn2
    _, p, r, f1, tp, fp, fn = rows[0.8]
    assert (tp, fp, fn) == (1, 0, 2)
    assert p == 1.0


def test_best_threshold_prefers_higher_f1_then_lower_threshold():
    # craft rows where two thresholds tie on F1 -> lower threshold wins (more recall).
    rows = [
        (0.30, 0.5, 0.5, 0.5, 1, 1, 1),
        (0.60, 0.5, 0.5, 0.5, 1, 1, 1),
        (0.40, 0.9, 0.1, 0.18, 1, 0, 9),
    ]
    assert best_threshold(rows)[0] == 0.30


def test_match_with_scores_greedy_one_gt_one_tp():
    # two overlapping predictions, one GT -> highest score is the TP, the other an FP.
    gt = [[0, 0, 10, 10]]
    pred = [[0, 0, 10, 10], [0, 0, 9, 9]]
    scores = [0.6, 0.9]  # the 0.9 box should claim the GT
    res = dict(match_with_scores(pred, scores, gt, iou_thr=0.5))
    assert res[0.9] == 1
    assert res[0.6] == 0


def test_match_with_scores_no_gt_all_fp():
    res = match_with_scores([[0, 0, 5, 5]], [0.8], [], iou_thr=0.5)
    assert res == [(0.8, 0)]
