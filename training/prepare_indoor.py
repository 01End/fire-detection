"""Merge the Roboflow 'Indoor Fire Smoke' dataset into a D-Fire tree.

The indoor set is YOLO format but its label files use **0 = FIRE, 1 = SMOKE** (verified by
inspecting the boxed crops), whereas **D-Fire label files use 0 = SMOKE, 1 = FIRE**. So we
remap indoor class ids to D-Fire's convention while copying, otherwise the two classes would
be swapped during training.

Images are copied with an ``indoor_`` prefix so they never collide with D-Fire filenames and
the added data stays identifiable/removable.

Appends, in place, into the D-Fire tree:
    <dfire>/train/{images,labels}        <- indoor train + valid
    <dfire>/test/{images,labels}         <- indoor test
    <dfire>/indoor_test/{images,labels}  <- indoor test only (for an indoor-only eval)

Usage (local or Colab — same script)::

    python -m training.prepare_indoor \
        --indoor "<...>/Indoor Fire Smoke" --dfire "<...>/D-Fire"
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _remap_label(src_txt: str, dst_txt: str, fire_index: int) -> None:
    """Copy a YOLO label file, remapping indoor classes to D-Fire (0=smoke, 1=fire)."""
    out = []
    if os.path.exists(src_txt):
        for line in open(src_txt, encoding="utf-8").read().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))
            dfire_cls = 1 if cls == fire_index else 0  # fire->1, everything else (smoke)->0
            out.append(" ".join([str(dfire_cls)] + parts[1:]))
    with open(dst_txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + ("\n" if out else ""))


def _copy_split(indoor_split: str, dst_img: str, dst_lbl: str, fire_index: int,
                prefix: str = "indoor_") -> int:
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)
    img_dir = os.path.join(indoor_split, "images")
    lbl_dir = os.path.join(indoor_split, "labels")
    n = 0
    for ip in glob.glob(os.path.join(img_dir, "*")):
        if os.path.splitext(ip)[1].lower() not in IMAGE_EXTS:
            continue
        stem = os.path.splitext(os.path.basename(ip))[0]
        shutil.copy(ip, os.path.join(dst_img, prefix + os.path.basename(ip)))
        _remap_label(os.path.join(lbl_dir, stem + ".txt"),
                     os.path.join(dst_lbl, prefix + stem + ".txt"), fire_index)
        n += 1
    return n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Merge indoor dataset into a D-Fire tree")
    p.add_argument("--indoor", required=True, help="path to the 'Indoor Fire Smoke' folder")
    p.add_argument("--dfire", required=True, help="D-Fire root (must contain train/ and test/)")
    p.add_argument("--indoor-fire-index", type=int, default=0,
                   help="class id meaning FIRE in the indoor labels (verified: 0)")
    a = p.parse_args(argv)
    fi = a.indoor_fire_index

    n_train = 0
    for split in ("train", "valid"):
        d = os.path.join(a.indoor, split)
        if os.path.isdir(d):
            n_train += _copy_split(
                d, os.path.join(a.dfire, "train", "images"),
                os.path.join(a.dfire, "train", "labels"), fi)

    n_test = 0
    dtest = os.path.join(a.indoor, "test")
    if os.path.isdir(dtest):
        n_test += _copy_split(
            dtest, os.path.join(a.dfire, "test", "images"),
            os.path.join(a.dfire, "test", "labels"), fi)
        _copy_split(  # standalone indoor-only eval set
            dtest, os.path.join(a.dfire, "indoor_test", "images"),
            os.path.join(a.dfire, "indoor_test", "labels"), fi)

    print(f"merged indoor -> D-Fire: +{n_train} train images, +{n_test} test images")
    print(f"indoor-only eval set: {os.path.join(a.dfire, 'indoor_test')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
