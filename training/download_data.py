"""Fetch a public fire/smoke dataset — behind an explicit license-acknowledgement gate.

License posture (verify against the source before any commercial use — licenses change):

  D-Fire  (https://github.com/gaiasd/DFireDataset)
      ~21k images, YOLO labels (smoke/fire). Released by GAIA/UFMG for research; confirm
      commercial terms with the authors before deploying.

  FASDD   (Flame And Smoke Detection Dataset, hosted on Zenodo)
      Large multi-source set, typically CC-BY 4.0 — the commercial-clean fallback. Open
      the current Zenodo record and confirm the exact license/version you download.

This script will not download anything until you pass ``--accept-license``, affirming you
have read and accepted the chosen dataset's terms. Provide ``--url`` pointing at the
dataset archive (zip/tar) you are entitled to use; it is downloaded and extracted under
``data/<dataset>/``.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile

DATASETS = {
    "dfire": "D-Fire (GAIA/UFMG) — research use; verify commercial terms.",
    "fasdd": "FASDD (Zenodo) — typically CC-BY 4.0; verify the record's license.",
}


def _download(url: str, dest: str) -> str:
    print(f"downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)
    return dest


def _extract(archive: str, dest_dir: str) -> None:
    print(f"extracting {archive} -> {dest_dir}")
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)
    elif archive.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(archive) as tf:
            tf.extractall(dest_dir)
    else:
        raise ValueError(f"unsupported archive type: {archive}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--url", help="direct URL to the dataset archive you may use")
    p.add_argument("--accept-license", action="store_true",
                   help="affirm you have read and accept the dataset's license")
    p.add_argument("--out", default="data", help="root output directory")
    a = p.parse_args(argv)

    print(f"Dataset: {a.dataset}\nLicense note: {DATASETS[a.dataset]}\n")

    if not a.accept_license:
        print("Refusing to download: re-run with --accept-license once you have read and "
              "accepted the license above.", file=sys.stderr)
        return 2
    if not a.url:
        print("No --url given. Visit the dataset's official page, obtain the archive URL "
              "you are entitled to, and pass it via --url.", file=sys.stderr)
        return 2

    dest_dir = os.path.join(a.out, a.dataset)
    os.makedirs(dest_dir, exist_ok=True)
    archive = os.path.join(dest_dir, os.path.basename(a.url.split("?")[0]) or "archive.zip")
    _download(a.url, archive)
    _extract(archive, dest_dir)
    print(f"done. Arrange as {dest_dir}/train/{{images,labels}} and "
          f"{dest_dir}/val/{{images,labels}} for training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
