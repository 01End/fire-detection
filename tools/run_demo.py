"""One-shot FireWatch demo pipeline:  download -> (trim if long) -> detect.

HOW TO USE:
    1. Set LINK and MODEL below (and tweak the other knobs if you want).
    2. Run with the venv Python:
         C:\\venvs\\firewatch\\Scripts\\python.exe tools/run_demo.py

What it does, automatically:
    * downloads LINK to  demos/demo.mp4
    * if that clip is longer than MAX_SECONDS -> trims START..START+DURATION first,
      otherwise uses the whole clip as-is (no trim)
    * runs the detector -> demos/demo_annotated.mp4

If the MODEL file isn't there yet (still training on Colab), it still does the
download/trim steps and then tells you what to do — just re-run once the model lands.
"""
from __future__ import annotations

import os
import subprocess
import sys

# =====================================================================
#  👇  THE ONLY THINGS YOU CHANGE
LINK = "https://youtu.be/whlymAuRtzU?si=w8HMTNPqL87pGOoN"   # video to download
MODEL = "models/fire_retinanet_indoor.weights.h5"      # best in+outdoor model (indoor-retrained)

MAX_SECONDS = 90      # clips longer than this get trimmed; shorter ones run whole
START = 60            # trim start in seconds  (1:00 — fire is clearly visible here)
DURATION = 60         # trim length in seconds (1:00 -> 2:00)

SCORE_THR = 0.6       # best operating point (peak F1 indoor 0.67 / overall 0.58 @ IoU0.5)
IMAGE_SIZE = 384      # the indoor-retrained model was fine-tuned at 384
EVERY = 15            # run detection on every Nth frame (keeps CPU runs tractable)
SMOOTH = True         # write every frame at full fps (smooth video) vs one frame per detection
EXPOSURE = "none"     # this model was trained with exposure none; keep it matched
HEIGHT = 720          # max download resolution (usually 360p without ffmpeg)
# =====================================================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "demos", "demo.mp4")
TRIMMED = os.path.join(ROOT, "demos", "demo_trimmed.mp4")
ANNOTATED = os.path.join(ROOT, "demos", "demo_annotated.mp4")


def download() -> None:
    from yt_dlp import YoutubeDL

    os.makedirs(os.path.dirname(DEMO), exist_ok=True)
    # Always fetch the CURRENT link fresh. Otherwise yt-dlp sees demos/demo.mp4 already
    # exists and SKIPS the download, so changing LINK silently has no effect (you'd keep
    # detecting on the previous video). Clear stale outputs first.
    for stale in (DEMO, TRIMMED):
        if os.path.exists(stale):
            os.remove(stale)
    fmt = f"b[ext=mp4][height<={HEIGHT}]/b[height<={HEIGHT}]/b[ext=mp4]/b"
    opts = {"format": fmt, "outtmpl": DEMO, "noplaylist": True, "quiet": False,
            "overwrites": True}
    print(f"[1/3] downloading {LINK}")
    with YoutubeDL(opts) as ydl:
        ydl.download([LINK])


def clip_seconds(path: str) -> float:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n / max(fps, 1.0)


def trim(src: str, dst: str) -> None:
    import cv2

    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_f, end_f = int(START * fps), int((START + DURATION) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    i = start_f
    while i < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        i += 1
    cap.release()
    writer.release()
    print(f"      trimmed -> {dst}  ({(i - start_f) / fps:.0f}s)")


def main() -> int:
    download()

    secs = clip_seconds(DEMO)
    print(f"[2/3] clip is {secs:.0f}s; threshold is {MAX_SECONDS}s")
    if secs > MAX_SECONDS:
        print(f"      long clip -> trimming {START}s..{START + DURATION}s")
        trim(DEMO, TRIMMED)
        source = TRIMMED
    else:
        print("      short clip -> using it whole (no trim)")
        source = DEMO

    model = os.path.join(ROOT, MODEL)
    if not os.path.exists(model):
        print(f"\n[3/3] SKIPPED — model not found at {model}")
        print("      Download fire_retinanet_full.weights.h5 from Drive into the")
        print("      models\\ folder, then run this script again to produce the demo.")
        return 0

    print("[3/3] running detection ...")
    cmd = [sys.executable, os.path.join(ROOT, "tools", "detect_video.py"),
           "--model", model, "--source", source, "--out", ANNOTATED,
           "--image-size", str(IMAGE_SIZE), "--score-thr", str(SCORE_THR),
           "--every", str(EVERY), "--exposure", EXPOSURE]
    if SMOOTH:
        cmd.append("--smooth")
    subprocess.run(cmd, check=True)
    print(f"\nDONE -> {ANNOTATED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
