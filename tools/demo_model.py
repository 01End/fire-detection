"""Teaching demo: push one image through the real detector and show the actual grids.

Builds the same SSDlite model the project uses, runs an image through it, and prints
the REAL feature-map grid sizes (the 'cells'), how many predictions the grid makes, and
the raw output shapes. Also saves an image with one grid drawn over it.

Run: python tools/demo_model.py
"""
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from firewatch.detection.model import build_model  # noqa: E402

IMG = 320  # SSDlite320 input size


def make_demo_image(path):
    """A simple synthetic scene so the grid overlay is visible."""
    img = np.full((IMG, IMG, 3), 30, dtype=np.uint8)
    cv2.rectangle(img, (120, 110), (200, 210), (40, 120, 220), -1)  # an 'object'
    cv2.circle(img, (160, 90), 25, (60, 200, 255), -1)
    cv2.imwrite(path, img)
    return img


def main():
    print("Building the real model (SSDlite + MobileNetV3)...\n")
    model = build_model(arch="ssdlite", pretrained_backbone=False).eval()

    img = make_demo_image("model_demo_input.jpg")
    rgb = img[:, :, ::-1]
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255

    print(f"INPUT image: {IMG}x{IMG}x3  (1 frame)\n")

    # The backbone returns several feature maps at different scales = the GRIDS.
    with torch.no_grad():
        feats = model.backbone(tensor.unsqueeze(0))

    print("The CNN turns that image into several GRIDS (feature maps):")
    print("-" * 64)
    total_cells = 0
    grids = []
    for name, fmap in feats.items():
        _, ch, h, w = fmap.shape
        cells = h * w
        total_cells += cells
        grids.append((h, w))
        role = "small objects" if cells >= 100 else "large objects"
        print(f"  grid {h:>2}x{w:<2} = {cells:>4} cells, {ch:>3} channels each  -> {role}")
    print("-" * 64)
    print(f"  TOTAL: {total_cells} cells across {len(grids)} grids")
    print("  Each cell asks: 'fire / smoke / nothing, and where exactly?'\n")

    # Run the full detector (untrained -> boxes are random, but shapes are real).
    with torch.no_grad():
        out = model([tensor])[0]
    print("RAW OUTPUT for this frame (one row per predicted box):")
    print(f"  boxes : {tuple(out['boxes'].shape)}   (x1,y1,x2,y2 each)")
    print(f"  labels: {tuple(out['labels'].shape)}")
    print(f"  scores: {tuple(out['scores'].shape)}")
    print("  (untrained model, so the boxes are meaningless here -- this shows the")
    print("   MECHANICS: the model really does emit boxes+labels+scores per frame.)\n")

    # Draw the coarsest informative grid over the image so 'cells' are visible.
    gh, gw = min(grids, key=lambda hw: abs(hw[0] - 10))  # pick a ~10x10 grid
    canvas = img.copy()
    for i in range(1, gw):
        x = round(i * IMG / gw)
        cv2.line(canvas, (x, 0), (x, IMG), (0, 255, 0), 1)
    for i in range(1, gh):
        y = round(i * IMG / gh)
        cv2.line(canvas, (0, y), (IMG, y), (0, 255, 0), 1)
    cv2.imwrite("model_demo_grid.jpg", canvas)
    print(f"Saved 'model_demo_grid.jpg' -> the {gh}x{gw} grid drawn over the image.")
    print("Saved 'model_demo_input.jpg' -> the original demo image.")


if __name__ == "__main__":
    main()
