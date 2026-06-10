"""Unit tests for the fire/indoor oversampling path-expansion in training/tf_dataset."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from training.tf_dataset import (  # noqa: E402
    DEFAULT_CLASS_MAP,
    OUR_FIRE,
    _is_indoor,
    _label_has_fire,
    _oversampled_paths,
)


def _write_label(tmp_path, name, cls_ids):
    p = tmp_path / (name + ".txt")
    p.write_text("\n".join(f"{c} 0.5 0.5 0.2 0.2" for c in cls_ids))
    return str(p)


def test_is_indoor_prefixes():
    assert _is_indoor("/x/indoor_001.jpg")
    assert _is_indoor("/x/extra_zz.png")
    assert not _is_indoor("/x/dfire_123.jpg")


def test_label_has_fire_uses_class_map(tmp_path):
    # D-Fire default map: 1 -> fire, 0 -> smoke
    _write_label(tmp_path, "f", [1])      # contains fire
    _write_label(tmp_path, "s", [0])      # smoke only
    assert _label_has_fire(str(tmp_path / "f.txt"), DEFAULT_CLASS_MAP) is True
    assert _label_has_fire(str(tmp_path / "s.txt"), DEFAULT_CLASS_MAP) is False
    assert _label_has_fire(str(tmp_path / "missing.txt"), DEFAULT_CLASS_MAP) is False
    assert OUR_FIRE == 0


def test_no_oversample_returns_original_list(tmp_path):
    paths = ["/imgs/a.jpg", "/imgs/b.jpg"]
    assert _oversampled_paths(paths, str(tmp_path), DEFAULT_CLASS_MAP, 1, 1) is paths


def test_additive_repeat_indoor_fire_seen_most(tmp_path):
    # build a labels dir; image basenames must match label stems
    labels = tmp_path
    _write_label(labels, "dfire_fire", [1])      # outdoor fire
    _write_label(labels, "dfire_smoke", [0])     # outdoor smoke (background-ish)
    _write_label(labels, "indoor_fire", [1])     # indoor fire (rarest)
    _write_label(labels, "indoor_smoke", [0])    # indoor smoke
    paths = [
        "/imgs/dfire_fire.jpg",
        "/imgs/dfire_smoke.jpg",
        "/imgs/indoor_fire.jpg",
        "/imgs/indoor_smoke.jpg",
    ]
    out = _oversampled_paths(paths, str(labels), DEFAULT_CLASS_MAP,
                             oversample_fire=2, oversample_indoor=2)
    counts = {p: out.count(p) for p in paths}
    assert counts["/imgs/dfire_smoke.jpg"] == 1   # 1 (neither)
    assert counts["/imgs/dfire_fire.jpg"] == 2    # 1 + (2-1) fire
    assert counts["/imgs/indoor_smoke.jpg"] == 2  # 1 + (2-1) indoor
    assert counts["/imgs/indoor_fire.jpg"] == 3   # 1 + fire + indoor
