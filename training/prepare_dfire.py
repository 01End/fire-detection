"""Arrange a downloaded D-Fire dataset into the layout train.py expects.

Produces::

    <out>/train/images/*.jpg   <out>/train/labels/*.txt
    <out>/val/images/*.jpg     <out>/val/labels/*.txt
    <out>/class_map.json       # dataset-class-id -> our label index (fire=1, smoke=2)

It adapts to common D-Fire layouts (pre-split train/test, or a flat images+labels
folder it splits itself) and verifies the class order against a ``data.yaml`` if present,
so mislabeled redistributions don't silently flip fire/smoke.

Examples::

    # Quick demo subset (fast to train):
    python -m training.prepare_dfire --src "C:/Downloads/D-Fire" --train-subset 2000 --val-subset 400
    # Full dataset:
    python -m training.prepare_dfire --src "C:/Downloads/D-Fire"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
from typing import Dict, List, Optional, Tuple

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Our runtime label indices (0 = background).
OUR_FIRE, OUR_SMOKE = 1, 2


def _find_pair(folder: str) -> Optional[Tuple[str, str]]:
    """Return (images_dir, labels_dir) under ``folder`` if both exist."""
    for img_name in ("images", "JPEGImages", "img"):
        for lbl_name in ("labels", "annotations", "ann"):
            img = os.path.join(folder, img_name)
            lbl = os.path.join(folder, lbl_name)
            if os.path.isdir(img) and os.path.isdir(lbl):
                return img, lbl
    return None


def _discover_splits(src: str) -> Dict[str, Tuple[str, str]]:
    """Map our split names ('train'/'val') to (images_dir, labels_dir) in ``src``."""
    found: Dict[str, Tuple[str, str]] = {}
    aliases = {"train": "train", "val": "val", "valid": "val", "test": "val"}
    for name in os.listdir(src):
        path = os.path.join(src, name)
        if not os.path.isdir(path):
            continue
        ours = aliases.get(name.lower())
        if ours is None:
            continue
        pair = _find_pair(path)
        if pair:
            # Prefer a real 'val' over 'test' if both map to val.
            if ours == "val" and "val" in found and name.lower() == "test":
                continue
            found[ours] = pair
    # Fall back: a flat images/labels pair at the root -> split later.
    if not found:
        pair = _find_pair(src)
        if pair:
            found["__flat__"] = pair
    return found


def _list_images(images_dir: str) -> List[str]:
    return sorted(
        p for p in glob.glob(os.path.join(images_dir, "*"))
        if os.path.splitext(p)[1].lower() in IMAGE_EXTS
    )


def _label_for(image_path: str, labels_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(labels_dir, stem + ".txt")


def _copy_split(images: List[str], src_labels: str, out_dir: str) -> int:
    img_out = os.path.join(out_dir, "images")
    lbl_out = os.path.join(out_dir, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    n = 0
    for img in images:
        shutil.copy2(img, os.path.join(img_out, os.path.basename(img)))
        lbl = _label_for(img, src_labels)
        if os.path.exists(lbl):
            shutil.copy2(lbl, os.path.join(lbl_out, os.path.basename(lbl)))
        else:
            # No annotation file = background image; create an empty label.
            open(os.path.join(lbl_out, os.path.splitext(os.path.basename(img))[0] + ".txt"), "w").close()
        n += 1
    return n


def _detect_class_map(src: str) -> Tuple[Dict[int, int], str]:
    """Return (class_map dataset_id->our_index, explanation)."""
    for cand in ("data.yaml", "data.yml", "dataset.yaml"):
        path = os.path.join(src, cand)
        if os.path.exists(path):
            try:
                import yaml
                names = (yaml.safe_load(open(path, encoding="utf-8")) or {}).get("names")
            except Exception:
                names = None
            if names:
                cmap = {}
                for idx, nm in enumerate(names):
                    nm = str(nm).lower()
                    if "fire" in nm:
                        cmap[idx] = OUR_FIRE
                    elif "smoke" in nm:
                        cmap[idx] = OUR_SMOKE
                return cmap, f"from {cand}: names={names}"
    # D-Fire standard convention.
    return {0: OUR_SMOKE, 1: OUR_FIRE}, "default D-Fire convention (0=smoke, 1=fire) — VERIFY"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="extracted D-Fire root folder")
    p.add_argument("--out", default="data/dfire", help="output dataset root")
    p.add_argument("--train-subset", type=int, default=None, help="cap train images")
    p.add_argument("--val-subset", type=int, default=None, help="cap val images")
    p.add_argument("--val-ratio", type=float, default=0.1, help="val fraction if splitting a flat set")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)

    random.seed(a.seed)
    splits = _discover_splits(a.src)
    if not splits:
        print(f"ERROR: could not find images/labels under {a.src}", flush=True)
        return 2

    if "__flat__" in splits:
        img_dir, lbl_dir = splits["__flat__"]
        all_imgs = _list_images(img_dir)
        random.shuffle(all_imgs)
        k = int(len(all_imgs) * (1 - a.val_ratio))
        plan = {"train": (all_imgs[:k], lbl_dir), "val": (all_imgs[k:], lbl_dir)}
    else:
        if "val" not in splits and "train" in splits:
            # Split train into train/val if no separate val/test was found.
            img_dir, lbl_dir = splits["train"]
            all_imgs = _list_images(img_dir)
            random.shuffle(all_imgs)
            k = int(len(all_imgs) * (1 - a.val_ratio))
            plan = {"train": (all_imgs[:k], lbl_dir), "val": (all_imgs[k:], lbl_dir)}
        else:
            plan = {s: (_list_images(d[0]), d[1]) for s, d in splits.items()}

    # Apply subset caps.
    if a.train_subset and "train" in plan:
        imgs, lbl = plan["train"]
        random.shuffle(imgs)
        plan["train"] = (imgs[: a.train_subset], lbl)
    if a.val_subset and "val" in plan:
        imgs, lbl = plan["val"]
        random.shuffle(imgs)
        plan["val"] = (imgs[: a.val_subset], lbl)

    os.makedirs(a.out, exist_ok=True)
    for split, (imgs, lbl) in plan.items():
        n = _copy_split(imgs, lbl, os.path.join(a.out, split))
        print(f"{split}: {n} images")

    cmap, why = _detect_class_map(a.src)
    with open(os.path.join(a.out, "class_map.json"), "w", encoding="utf-8") as fh:
        json.dump({str(k): v for k, v in cmap.items()}, fh)
    print(f"class_map: {cmap}  ({why})")
    print(f"done -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
