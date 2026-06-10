"""Run the TF fire/smoke detector on a video file or a live webcam (demo tool).

Two modes, chosen by ``--source``:
  * a path  -> reads the video, writes an annotated ``.mp4`` (``--out``).
  * an int  -> opens that webcam index and shows a live annotated window.

TF inference is CPU-only on native Windows (~3-5 s/frame at 512px), so detection runs on
every Nth frame (``--every``). Drop ``--image-size`` to 320/384 to speed it up.

Examples::

    # annotate a downloaded clip
    PYTHONPATH=src python tools/detect_video.py --model models/fire_retinanet.weights.h5 \
        --source data/demo_clip.mp4 --out demo_annotated.mp4 --every 15

    # live webcam (point it at a fire video on a screen, or a candle)
    PYTHONPATH=src python tools/detect_video.py --model models/fire_retinanet.weights.h5 \
        --source 0 --display --image-size 384
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="TF fire/smoke detection on video/webcam")
    p.add_argument("--model", required=True, help="TF .weights.h5 checkpoint")
    p.add_argument("--source", required=True,
                   help="video file path, OR a webcam index like 0")
    p.add_argument("--out", default=None, help="output annotated .mp4 (file mode)")
    p.add_argument("--image-size", type=int, default=512, help="match the training size")
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--every", type=int, default=15, help="run detection every Nth frame")
    p.add_argument("--display", action="store_true", help="show a live window")
    p.add_argument("--smooth", action="store_true",
                   help="write EVERY frame at full fps, keeping the last boxes between "
                        "detection frames -> smooth playback instead of a choppy slideshow")
    p.add_argument("--hold", type=float, default=0.0,
                   help="temporal persistence (seconds): when a detection frame comes back "
                        "EMPTY, keep showing the last boxes for up to this long instead of "
                        "dropping them instantly -> fills brief flicker gaps. This is also the "
                        "alert-smoothing idea: real fire/smoke persists, lone false positives "
                        "don't. 0 = off (clear as soon as a frame sees nothing).")
    p.add_argument("--exposure", default="none", choices=("none", "clahe", "gamma"),
                   help="adaptive exposure normalization applied before detection")
    a = p.parse_args(argv)

    import cv2

    from firewatch.annotate import annotate
    from firewatch.detection.tf_detector import TFFireDetector

    is_webcam = a.source.isdigit()
    detector = TFFireDetector.from_checkpoint(
        a.model, arch="retinanet", score_threshold=a.score_thr, image_size=a.image_size,
        exposure=a.exposure,
    )

    cap = cv2.VideoCapture(int(a.source) if is_webcam else a.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {a.source!r}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # how many frames a detection should persist when later frames come back empty
    hold_frames = int(a.hold * src_fps)

    writer = None
    if a.out and not is_webcam:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # smooth -> full source fps (every frame written); else one frame per detection.
        out_fps = src_fps if a.smooth else max(1.0, src_fps / max(1, a.every))
        writer = cv2.VideoWriter(
            a.out, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h)
        )

    show = a.display or is_webcam
    idx = processed = total_dets = 0
    last_dets: list = []
    frames_since_seen = hold_frames + 1   # start "expired" so nothing is held before the first hit
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            run_now = idx % max(1, a.every) == 0
            if run_now:
                dets = detector.detect(frame)
                processed += 1
                if dets:
                    last_dets = dets
                    total_dets += len(dets)
                    frames_since_seen = 0
                elif frames_since_seen > hold_frames:
                    # nothing seen and the hold window has expired -> drop the boxes
                    last_dets = []
                # else: empty frame but still within the hold window -> keep last_dets
                if processed % 10 == 0:
                    print(f"  processed {processed} frames, {total_dets} detections")
            frames_since_seen += 1
            if a.smooth:
                # draw the most recent detections on EVERY frame -> smooth full-fps video
                canvas = annotate(frame, [(d, None) for d in last_dets])
                if writer is not None:
                    writer.write(canvas)
                if show:
                    cv2.imshow("FireWatch demo (q to quit)", canvas)
            elif run_now:
                canvas = annotate(frame, [(d, None) for d in last_dets])
                if writer is not None:
                    writer.write(canvas)
                if show:
                    cv2.imshow("FireWatch demo (q to quit)", canvas)
            elif show and is_webcam:
                # keep the live feed moving between detections
                cv2.imshow("FireWatch demo (q to quit)", frame)
            if show and (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
            idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    print(f"\ndone: read {idx} frames, ran detection on {processed}, "
          f"{total_dets} total detections")
    if writer is not None:
        print(f"wrote annotated video -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
