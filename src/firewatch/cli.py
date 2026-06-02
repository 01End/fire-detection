"""FireWatch command-line interface: ``run`` and ``setup-zones``."""
from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional, Tuple

from .config import build_pipeline, load_camera_config
from .detection.types import Detection


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load_camera_config(args.config)

    frame_hook = _make_display_hook() if args.display else None
    pipeline = build_pipeline(
        cfg, model_path=args.model, output_dir=args.output, frame_hook=frame_hook
    )
    logging.getLogger("firewatch").info(
        "monitoring camera '%s' (model=%s)", cfg.get("camera_id"), args.model
    )
    pipeline.run(max_frames=args.max_frames)
    return 0


def _make_display_hook():  # pragma: no cover - needs a display
    import cv2

    from .annotate import annotate

    def hook(frame, mapped: List[Tuple[Detection, Optional[int]]]):
        canvas = annotate(frame.image, mapped)
        cv2.imshow(f"FireWatch: {frame.camera_id}", canvas)
        cv2.waitKey(1)

    return hook


def _cmd_setup_zones(args: argparse.Namespace) -> int:
    from .zonetool import (
        auto_bands,
        collect_polygons_interactive,
        save_zones,
        snapshot_from_config,
    )

    cfg = load_camera_config(args.config)
    image = snapshot_from_config(cfg)
    h, w = image.shape[:2]

    if args.auto:
        zones = auto_bands(width=w, height=h, n_floors=args.auto)
    else:  # pragma: no cover - interactive
        zones = collect_polygons_interactive(image)
        if not zones:
            print("cancelled; no zones saved.")
            return 1

    save_zones(args.config, cfg, zones)
    print(f"saved {len(zones)} floor zone(s) to {args.config}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="firewatch", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run detection on a camera")
    p_run.add_argument("--config", required=True, help="camera config YAML")
    p_run.add_argument("--model", required=True, help="trained model checkpoint (.pt)")
    p_run.add_argument("--output", default=None, help="evidence/output dir")
    p_run.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    p_run.add_argument("--display", action="store_true", help="show annotated frames")
    p_run.set_defaults(func=_cmd_run)

    p_zones = sub.add_parser("setup-zones", help="define a camera's floor zones")
    p_zones.add_argument("--config", required=True, help="camera config YAML")
    p_zones.add_argument(
        "--auto", type=int, metavar="N",
        help="headless: split the frame into N equal horizontal floor bands",
    )
    p_zones.set_defaults(func=_cmd_setup_zones)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    _configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
