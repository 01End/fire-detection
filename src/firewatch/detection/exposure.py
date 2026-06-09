"""Adaptive exposure normalization for detector input frames.

Pure numpy/cv2 (no TensorFlow import) so it's cheap to unit-test, like ``types.py``. The same
function is used at **inference** (``TFFireDetector._prepare``) and in the **training** pipeline,
so train/test input statistics match.

Idea: "bright frame -> lower, dark frame -> raise" — normalize each frame's exposure before the
model sees it, to help out-of-distribution indoor / over/under-exposed scenes. (Cannot recover
fully clipped pure-white highlights — that information is already lost.)

Methods:
  * ``"none"``  — identity (default; no behaviour change).
  * ``"clahe"`` — Contrast-Limited Adaptive Histogram Equalization on the L (luminance) channel;
                  lifts shadows and tames bright areas *locally* without over-amplifying noise.
  * ``"gamma"`` — global gamma correction pushing mean luminance toward ``target``.
"""
from __future__ import annotations

import numpy as np

EXPOSURE_METHODS = ("none", "clahe", "gamma")


def auto_exposure(rgb, method: str = "none", clip: float = 2.0, grid: int = 8,
                  target: float = 0.5):
    """Return an exposure-normalized copy of an HxWx3 uint8 RGB image.

    ``method`` is one of :data:`EXPOSURE_METHODS`. Shape and dtype (uint8) are preserved.
    """
    if method in (None, "none"):
        return rgb

    import cv2

    img = np.ascontiguousarray(rgb)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype("uint8")

    if method == "clahe":
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

    if method == "gamma":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mean = float(gray.mean()) / 255.0
        mean = min(max(mean, 1e-3), 1.0 - 1e-3)  # avoid log(0) / log(1)
        # output = input ** gamma  (input in [0,1]); gamma>1 darkens, gamma<1 brightens.
        # Solving mean**gamma = target gives gamma below; clamp so one odd frame can't explode.
        gamma = float(np.clip(np.log(target) / np.log(mean), 0.4, 2.5))
        lut = (np.power(np.arange(256) / 255.0, gamma) * 255.0).clip(0, 255).astype("uint8")
        return cv2.LUT(img, lut)

    raise ValueError(f"unknown exposure method {method!r}; use one of {EXPOSURE_METHODS}")


def jitter_exposure(rgb, rng=None):
    """Random brightness + gamma jitter for *training* augmentation.

    Distinct from :func:`auto_exposure` (which is deterministic normalization): this randomly
    varies exposure so the model learns to tolerate it. Returns an HxWx3 uint8 RGB image.
    """
    import cv2

    rng = rng if rng is not None else np.random
    img = np.ascontiguousarray(rgb)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype("uint8")

    gamma = float(rng.uniform(0.6, 1.6))           # random darken/brighten
    lut = (np.power(np.arange(256) / 255.0, gamma) * 255.0).clip(0, 255).astype("uint8")
    img = cv2.LUT(img, lut)

    gain = float(rng.uniform(0.8, 1.2))            # random brightness scale
    return np.clip(img.astype("float32") * gain, 0, 255).astype("uint8")
