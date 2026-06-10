"""One-shot FireWatch demo pipeline:  download -> (trim if long) -> detect.

Every run creates its own timestamped folder under  demos/runs/<timestamp>/  containing:
    original.mp4   - the downloaded video
    trimmed.mp4    - the trimmed clip (only if the video was long enough to trim)
    annotated.mp4  - the detector output
    notes.txt      - the settings used (link, model, threshold, window, ...)
So you can build up a library of demos without overwriting previous ones.

HOW TO USE:
    1. Set LINK and MODEL below (and tweak the other knobs if you want). SAVE the file.
    2. Run with the venv Python:
         C:\\venvs\\firewatch\\Scripts\\python.exe tools/run_demo.py
"""
from __future__ import annotations

import datetime
import os
import shutil
import subprocess
import sys

# =====================================================================
#  👇  THE ONLY THINGS YOU CHANGE  (save the file after editing!)
LINK = "https://www.youtube.com/watch?v=BJ9ng9L1CA0"   # video to download
MODEL = "models/fire_retinanet_indoor.weights.h5"      # best in+outdoor model (indoor-retrained)

MAX_SECONDS = 90      # clips longer than this get trimmed; shorter ones run whole
START = 60            # trim start in seconds
DURATION = 60         # trim length in seconds

SCORE_THR = 0.3       # alert-system threshold: catches early SMOKE (~0.3-0.42) = earliest warning
IMAGE_SIZE = 384      # the indoor-retrained model was fine-tuned at 384
EVERY = 15            # run detection on every Nth frame (keeps CPU runs tractable)
SMOOTH = True         # write every frame at full fps (smooth video) vs one frame per detection
HOLD = 1.0            # keep boxes on screen this many SECONDS after a frame stops detecting ->
                      # fills brief flicker gaps; 0 = clear instantly (this is the alert-smoothing idea)
EXPOSURE = "none"     # this model was trained with exposure none; keep it matched
HEIGHT = 720          # max download resolution (now true 720p — ffmpeg merges the DASH streams)
# =====================================================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(ROOT, "demos", "runs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
ORIGINAL = os.path.join(RUN_DIR, "original.mp4")
TRIMMED = os.path.join(RUN_DIR, "trimmed.mp4")
ANNOTATED = os.path.join(RUN_DIR, "annotated.mp4")

# A tiny download cache so re-running the SAME link (e.g. trying thresholds) is instant.
_CACHE = os.path.join(ROOT, "demos", "_cache")
_CACHE_VID = os.path.join(_CACHE, "video.mp4")
_CACHE_URL = os.path.join(_CACHE, "url.txt")


def find_ffmpeg() -> str | None:
    """Locate ffmpeg so yt-dlp can merge 720p DASH streams. Checks PATH first, then the
    WinGet install dir (winget's PATH entry only applies to new login sessions, so a freshly
    installed ffmpeg won't be on PATH yet)."""
    onpath = shutil.which("ffmpeg")
    if onpath:
        return onpath
    pkgs = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages")
    if os.path.isdir(pkgs):
        for root, _dirs, files in os.walk(pkgs):
            if "ffmpeg.exe" in files:
                return os.path.join(root, "ffmpeg.exe")
    return None


def fetch_original() -> None:
    """Get the video (reuse cache if the link is unchanged), then copy it into the run folder."""
    os.makedirs(_CACHE, exist_ok=True)
    cached = open(_CACHE_URL, encoding="utf-8").read().strip() if os.path.exists(_CACHE_URL) else None
    if cached == LINK and os.path.exists(_CACHE_VID):
        print(f"[1/3] reusing cached download of {LINK}")
    else:
        from yt_dlp import YoutubeDL
        if os.path.exists(_CACHE_VID):
            os.remove(_CACHE_VID)
        # Prefer a MERGED video+audio stream up to HEIGHT (needs ffmpeg, which yt-dlp uses to
        # mux the separate DASH video/audio tracks). On YouTube the best single-file mp4 is only
        # 360p, so without this selector you'd be stuck at 360p and the smoke is hard to see.
        fmt = (f"bv*[height<={HEIGHT}]+ba/b[height<={HEIGHT}]/"
               f"bv*+ba/b[ext=mp4]/b")
        opts = {"format": fmt, "outtmpl": _CACHE_VID, "noplaylist": True,
                "merge_output_format": "mp4", "quiet": False, "overwrites": True}
        ffmpeg = find_ffmpeg()
        if ffmpeg:
            opts["ffmpeg_location"] = os.path.dirname(ffmpeg)
            print(f"[1/3] downloading {LINK}  (up to {HEIGHT}p, ffmpeg merge enabled)")
        else:
            print(f"[1/3] downloading {LINK}  (ffmpeg not found -> capped at 360p single-stream)")
        with YoutubeDL(opts) as ydl:
            ydl.download([LINK])
        open(_CACHE_URL, "w", encoding="utf-8").write(LINK)
    shutil.copy(_CACHE_VID, ORIGINAL)


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


def write_notes(trimmed: bool, model: str) -> None:
    with open(os.path.join(RUN_DIR, "notes.txt"), "w", encoding="utf-8") as f:
        f.write(f"time:       {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"link:       {LINK}\n")
        f.write(f"model:      {MODEL}\n")
        f.write(f"score_thr:  {SCORE_THR}\n")
        f.write(f"image_size: {IMAGE_SIZE}\n")
        f.write(f"exposure:   {EXPOSURE}\n")
        f.write(f"every:      {EVERY}\n")
        f.write(f"hold:       {HOLD}s\n")
        f.write(f"max_res:    {HEIGHT}p\n")
        if trimmed:
            f.write(f"trim:       {START}s..{START + DURATION}s\n")
        else:
            f.write("trim:       none (whole clip)\n")


def main() -> int:
    os.makedirs(RUN_DIR, exist_ok=True)
    print(f"run folder: {RUN_DIR}")
    fetch_original()

    secs = clip_seconds(ORIGINAL)
    print(f"[2/3] clip is {secs:.0f}s; threshold is {MAX_SECONDS}s")
    trimmed = secs > MAX_SECONDS
    if trimmed:
        print(f"      long clip -> trimming {START}s..{START + DURATION}s")
        trim(ORIGINAL, TRIMMED)
        source = TRIMMED
    else:
        print("      short clip -> using it whole (no trim)")
        source = ORIGINAL

    model = os.path.join(ROOT, MODEL)
    write_notes(trimmed, model)

    if not os.path.exists(model):
        print(f"\n[3/3] SKIPPED — model not found at {model}")
        print(f"      Put the model at {MODEL} and re-run. (Folder kept: {RUN_DIR})")
        return 0

    print("[3/3] running detection ...")
    cmd = [sys.executable, os.path.join(ROOT, "tools", "detect_video.py"),
           "--model", model, "--source", source, "--out", ANNOTATED,
           "--image-size", str(IMAGE_SIZE), "--score-thr", str(SCORE_THR),
           "--every", str(EVERY), "--exposure", EXPOSURE, "--hold", str(HOLD)]
    if SMOOTH:
        cmd.append("--smooth")
    subprocess.run(cmd, check=True)
    print(f"\nDONE -> {RUN_DIR}")
    print("       (original.mp4 + trimmed.mp4 + annotated.mp4 + notes.txt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
