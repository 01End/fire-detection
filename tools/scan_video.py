"""Sample frames across a whole video and tile them into one montage image,
each labelled with its timestamp — so you can eyeball WHERE the fire actually is.

Run:  C:\\venvs\\firewatch\\Scripts\\python.exe tools/scan_video.py
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "demos", "demo.mp4")   # the full downloaded video
COLS, ROWS = 4, 4                                # 16 samples
TILE_W = 320


def main() -> int:
    import cv2

    cap = cv2.VideoCapture(SRC)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {SRC}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    count = COLS * ROWS

    tiles = []
    for k in range(count):
        idx = int(n * (k + 0.5) / count)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            frame = None
        secs = idx / fps
        ts = f"{int(secs // 60)}:{int(secs % 60):02d}"
        if frame is None:
            import numpy as np
            frame = (0 * cv2.cvtColor(cv2.UMat(1, 1, cv2.CV_8UC3), cv2.COLOR_BGR2BGR)).get()
        h, w = frame.shape[:2]
        tile = cv2.resize(frame, (TILE_W, int(TILE_W * h / w)))
        cv2.putText(tile, ts, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(tile, ts, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        tiles.append(tile)
    cap.release()

    th = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    rows = [cv2.hconcat(tiles[r * COLS:(r + 1) * COLS]) for r in range(ROWS)]
    montage = cv2.vconcat(rows)

    out = os.path.join(ROOT, "demos", "diag", "montage.jpg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, montage)
    print(f"saved montage ({COLS}x{ROWS}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
