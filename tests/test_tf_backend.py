"""Tests for the TensorFlow/KerasHub backend.

The pure helpers and the detector's post-processing (yxyx->xyxy, rescaling, filtering)
are tested here WITHOUT TensorFlow, using a fake model. Anything that builds a real
KerasHub model is skipped unless keras_hub is installed.
"""
import numpy as np
import pytest

from firewatch.detection.tf_model import (
    label_from_index,
    num_classes_for,
    scale_box_xyxy,
    xyxy_to_yxyx,
    yxyx_to_xyxy,
)
from firewatch.detection.tf_detector import TFFireDetector
from firewatch.detection.types import CLASS_NAMES, Detection


# --- pure helpers (no TF needed) -------------------------------------------

def test_num_classes_is_zero_indexed():
    assert num_classes_for(CLASS_NAMES) == 2  # no background class, unlike torch


def test_label_mapping_zero_indexed():
    assert label_from_index(0, CLASS_NAMES) == "fire"
    assert label_from_index(1, CLASS_NAMES) == "smoke"
    with pytest.raises(ValueError):
        label_from_index(2, CLASS_NAMES)


def test_box_format_roundtrip():
    assert yxyx_to_xyxy((1, 2, 3, 4)) == (2, 1, 4, 3)
    assert xyxy_to_yxyx((2, 1, 4, 3)) == (1, 2, 3, 4)


def test_scale_box_xyxy():
    # from 512x512 down to 200x100 (w,h)
    out = scale_box_xyxy((256, 256, 384, 384), from_size=(512, 512), to_size=(200, 100))
    assert out == pytest.approx((100.0, 50.0, 150.0, 75.0))


# --- detector post-processing with a fake model ----------------------------

class _FakeModel:
    """Stands in for a KerasHub detector: returns canned predict() output."""

    def __init__(self, preds):
        self._preds = preds

    def predict(self, batch, verbose=0):
        assert batch.shape[1:] == (512, 512, 3)  # detector resized to image_size
        return self._preds


def test_detect_maps_scales_and_filters():
    preds = {
        # yxyx boxes in the 512x512 model space: one real, one padding.
        "boxes": np.array([[[256, 256, 384, 384], [-1, -1, -1, -1]]], dtype="float32"),
        "confidence": np.array([[0.9, 0.0]], dtype="float32"),
        "classes": np.array([[0, -1]], dtype="int32"),  # 0=fire, -1=padding
    }
    det = TFFireDetector(_FakeModel(preds), score_threshold=0.5, image_size=512)
    image = np.zeros((100, 200, 3), dtype=np.uint8)  # h=100, w=200

    results = det.detect(image)

    assert len(results) == 1  # padding row dropped
    r = results[0]
    assert isinstance(r, Detection)
    assert r.label == "fire"
    assert r.confidence == pytest.approx(0.9)
    # box rescaled from 512-space yxyx to original xyxy (200x100)
    assert r.bbox == pytest.approx((100.0, 50.0, 150.0, 75.0))


def test_detect_threshold_filters_low_scores():
    preds = {
        "boxes": np.array([[[0, 0, 10, 10]]], dtype="float32"),
        "confidence": np.array([[0.3]], dtype="float32"),
        "classes": np.array([[1]], dtype="int32"),
    }
    det = TFFireDetector(_FakeModel(preds), score_threshold=0.5, image_size=512)
    assert det.detect(np.zeros((64, 64, 3), dtype=np.uint8)) == []


# --- real model build (needs keras_hub) ------------------------------------

def test_build_model_smoke():
    pytest.importorskip("keras_hub")
    from firewatch.detection.tf_model import build_model

    model = build_model(arch="retinanet", pretrained_backbone=True)
    assert model is not None
