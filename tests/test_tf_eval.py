"""Unit tests for the pure-NumPy IoU + greedy-matching logic in training/tf_eval.

These don't load TensorFlow or any model weights — they exercise the metric math only.
"""
import numpy as np
import pytest

from training.tf_eval import box_iou, greedy_match


def test_box_iou_identical_is_one():
    a = np.array([[0, 0, 10, 10]], dtype="float32")
    assert box_iou(a, a)[0, 0] == pytest.approx(1.0)


def test_box_iou_disjoint_is_zero():
    a = np.array([[0, 0, 10, 10]], dtype="float32")
    b = np.array([[20, 20, 30, 30]], dtype="float32")
    assert box_iou(a, b)[0, 0] == pytest.approx(0.0)


def test_box_iou_partial_overlap():
    # [0,0,2,2] vs [1,1,3,3]: inter=1, union=4+4-1=7
    a = np.array([[0, 0, 2, 2]], dtype="float32")
    b = np.array([[1, 1, 3, 3]], dtype="float32")
    assert box_iou(a, b)[0, 0] == pytest.approx(1.0 / 7.0, rel=1e-5)


def test_box_iou_empty():
    a = np.zeros((0, 4), dtype="float32")
    b = np.array([[0, 0, 10, 10]], dtype="float32")
    assert box_iou(a, b).shape == (0, 1)


def test_greedy_match_perfect():
    box = [[0, 0, 10, 10]]
    assert greedy_match(box, box, iou_thr=0.5) == (1, 0, 0)


def test_greedy_match_no_ground_truth_is_false_positive():
    assert greedy_match([[0, 0, 10, 10]], [], iou_thr=0.5) == (0, 1, 0)


def test_greedy_match_missed_ground_truth_is_false_negative():
    assert greedy_match([], [[0, 0, 10, 10]], iou_thr=0.5) == (0, 0, 1)


def test_greedy_match_duplicate_predictions():
    # two preds on one GT: one TP, the duplicate is an FP.
    preds = [[0, 0, 10, 10], [0, 0, 10, 10]]
    gt = [[0, 0, 10, 10]]
    assert greedy_match(preds, gt, iou_thr=0.5) == (1, 1, 0)


def test_greedy_match_below_threshold_is_false_positive():
    # IoU 1/7 ~ 0.14 < 0.5 -> not a match: FP for the pred, FN for the GT.
    assert greedy_match([[0, 0, 2, 2]], [[1, 1, 3, 3]], iou_thr=0.5) == (0, 1, 1)
