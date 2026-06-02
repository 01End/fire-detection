"""Standalone entry point for interactive floor-zone authoring.

Thin wrapper around ``firewatch setup-zones`` so the tool can be run directly:

    python tools/label_zones.py --config configs/cameras/cam1.yaml
    python tools/label_zones.py --config configs/cameras/cam1.yaml --auto 3
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from firewatch.cli import main  # noqa: E402

if __name__ == "__main__":
    argv = ["setup-zones"] + sys.argv[1:]
    raise SystemExit(main(argv))
