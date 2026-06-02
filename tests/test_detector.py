"""Schema-level tests for the detector.

These build an untrained model with random weights (no network download) and assert the
output *contract* — shape, types, label mapping — rather than detection accuracy, which
is covered by the training/eval milestone.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from firewatch.detection.model import build_model, label_from_index, num_classes_for
from firewatch.detection.detector import FireDetector
from firewatch.detection.types import Detection, CLASS_NAMES


def test_num_classes_includes_background():
    # 2 foreground classes (fire, smoke) + background.
    assert num_classes_for(CLASS_NAMES) == 3


def test_label_mapping_skips_background():
    assert label_from_index(1, CLASS_NAMES) == "fire"
    assert label_from_index(2, CLASS_NAMES) == "smoke"


def test_build_model_ssdlite_runs_forward():
    model = build_model(arch="ssdlite", pretrained_backbone=False)
    model.eval()
    with torch.no_grad():
        out = model([torch.rand(3, 64, 64)])
    assert "boxes" in out[0] and "labels" in out[0] and "scores" in out[0]


def test_detector_returns_detection_list():
    model = build_model(arch="ssdlite", pretrained_backbone=False)
    det = FireDetector(model, score_threshold=0.5)
    image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)  # BGR

    results = det.detect(image)

    assert isinstance(results, list)
    for r in results:
        assert isinstance(r, Detection)
        assert r.label in CLASS_NAMES
        assert len(r.bbox) == 4
        assert 0.0 <= r.confidence <= 1.0


def test_detector_threshold_filters_low_scores():
    model = build_model(arch="ssdlite", pretrained_backbone=False)
    det = FireDetector(model, score_threshold=0.99)
    image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    # With a very high threshold and random weights, nothing should pass.
    assert det.detect(image) == []
