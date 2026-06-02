"""FireWatch command-line interface: ``run`` and ``setup-zones``."""
from __future__ import annotations

import argparse
import logging
import os
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


def _cmd_detect(args: argparse.Namespace) -> int:
    """Run the detector on an image or folder and write annotated copies."""
    import glob

    import cv2

    from .annotate import annotate

    if args.backend == "tf":
        from .detection.tf_detector import TFFireDetector as Detector
    else:
        from .detection.detector import FireDetector as Detector

    detector = Detector.from_checkpoint(
        args.model, arch=args.arch, score_threshold=args.score_thr
    )

    if os.path.isdir(args.input):
        paths = sorted(
            p for p in glob.glob(os.path.join(args.input, "*"))
            if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp")
        )
    else:
        paths = [args.input]

    os.makedirs(args.out, exist_ok=True)
    total = 0
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            continue
        dets = detector.detect(image)
        total += len(dets)
        canvas = annotate(image, [(d, None) for d in dets])
        out_path = os.path.join(args.out, os.path.basename(path))
        cv2.imwrite(out_path, canvas)
        summary = ", ".join(f"{d.label}:{d.confidence:.2f}" for d in dets) or "none"
        print(f"{os.path.basename(path)} -> {len(dets)} detection(s) [{summary}]")
    print(f"\nwrote {len(paths)} annotated image(s) to {args.out} ({total} detections)")
    return 0


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

    p_det = sub.add_parser("detect", help="run detection on an image/folder (no floors)")
    p_det.add_argument("--model", required=True, help="trained model checkpoint (.pt)")
    p_det.add_argument("--input", required=True, help="image file or folder of images")
    p_det.add_argument("--out", default="detect_out", help="dir for annotated images")
    p_det.add_argument("--backend", default="torch", choices=("torch", "tf"))
    p_det.add_argument("--arch", default="ssdlite", choices=("ssdlite", "retinanet"))
    p_det.add_argument("--score-thr", type=float, default=0.5)
    p_det.set_defaults(func=_cmd_detect)

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
