"""Smoke tests for the training path on a tiny synthetic YOLO dataset (no downloads)."""
import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.dataset import YoloDetectionDataset, collate_fn
from training.train import run_training, eval_loss
from training.eval import box_iou
from firewatch.detection.detector import FireDetector


def _make_split(root, split, n=2):
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        cv2.imwrite(str(img_dir / f"img{i}.png"), img)
        # One centered box, class 1 (=fire) in YOLO normalized cx cy w h.
        (lbl_dir / f"img{i}.txt").write_text("1 0.5 0.5 0.4 0.4\n")


def test_dataset_parses_yolo_labels_to_xyxy(tmp_path):
    _make_split(tmp_path, "train", n=1)
    ds = YoloDetectionDataset(
        str(tmp_path / "train" / "images"), str(tmp_path / "train" / "labels")
    )
    image, target = ds[0]
    assert image.shape == (3, 64, 64)
    assert target["boxes"].shape == (1, 4)
    # cx,cy,w,h = .5,.5,.4,.4 on 64px -> x1=y1=64*0.3=19.2, x2=y2=64*0.7=44.8
    box = target["boxes"][0].tolist()
    assert box == pytest.approx([19.2, 19.2, 44.8, 44.8], abs=1e-3)
    assert target["labels"].tolist() == [1]  # fire


def test_box_iou_basic():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[0.0, 0.0, 10.0, 10.0], [10.0, 10.0, 20.0, 20.0]])
    iou = box_iou(a, b)
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 1] == pytest.approx(0.0)


def test_training_runs_one_epoch_and_saves_loadable_weights(tmp_path):
    _make_split(tmp_path, "train", n=2)
    _make_split(tmp_path, "val", n=2)
    out = tmp_path / "model.pt"

    saved = run_training(
        data_dir=str(tmp_path), arch="ssdlite", epochs=1, batch_size=2,
        lr=0.001, out=str(out), device="cpu", num_workers=0,
        pretrained_backbone=False,
    )

    assert saved == str(out) and out.exists()
    # The saved weights load straight into the runtime detector.
    det = FireDetector.from_checkpoint(str(out), arch="ssdlite", score_threshold=0.5)
    results = det.detect((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
    assert isinstance(results, list)
