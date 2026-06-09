"""Unit tests for adaptive exposure normalization (pure numpy/cv2, no TF)."""
import numpy as np
import pytest

from firewatch.detection.exposure import EXPOSURE_METHODS, auto_exposure, jitter_exposure


def _img(fill: int) -> np.ndarray:
    return np.full((32, 48, 3), fill, dtype=np.uint8)


def test_methods_constant_includes_expected():
    assert EXPOSURE_METHODS == ("none", "clahe", "gamma")


def test_none_is_identity():
    img = _img(123)
    assert np.array_equal(auto_exposure(img, "none"), img)


@pytest.mark.parametrize("method", ["clahe", "gamma"])
def test_shape_and_dtype_preserved(method):
    img = _img(100)
    out = auto_exposure(img, method)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_gamma_brightens_dark_and_darkens_bright():
    dark, bright = _img(40), _img(220)
    assert auto_exposure(dark, "gamma").mean() > dark.mean()
    assert auto_exposure(bright, "gamma").mean() < bright.mean()


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        auto_exposure(_img(100), "bogus")


def test_reversed_view_is_handled():
    # detector feeds a BGR->RGB reversed (negative-stride) view; must not crash.
    bgr = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    out = auto_exposure(bgr[:, :, ::-1], "clahe")
    assert out.shape == bgr.shape and out.dtype == np.uint8


def test_jitter_preserves_shape_and_is_seed_deterministic():
    img = _img(128)
    a = jitter_exposure(img, rng=np.random.default_rng(7))
    b = jitter_exposure(img, rng=np.random.default_rng(7))
    assert a.shape == img.shape and a.dtype == np.uint8
    assert np.array_equal(a, b)
