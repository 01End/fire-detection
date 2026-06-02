"""Train the fire/smoke detector on a YOLO-format dataset.

Expected layout (D-Fire / FASDD style)::

    <data>/train/images/*.jpg   <data>/train/labels/*.txt
    <data>/val/images/*.jpg     <data>/val/labels/*.txt

Saves the best (lowest val loss) model weights to ``--out`` as a bare ``state_dict``
that ``FireDetector.from_checkpoint`` can load directly.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from firewatch.detection.model import build_model  # noqa: E402

from .dataset import YoloDetectionDataset, collate_fn  # noqa: E402


def _device(prefer: Optional[str]) -> torch.device:
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total = 0.0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return total / max(len(loader), 1)


@torch.no_grad()
def eval_loss(model, loader, device) -> float:
    # torchvision detection models only return losses in train() mode.
    model.train()
    total = 0.0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        total += float(sum(loss_dict.values()).item())
    return total / max(len(loader), 1)


def run_training(
    data_dir: str,
    arch: str = "ssdlite",
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 0.005,
    out: str = "models/fire_ssdlite.pt",
    device: Optional[str] = None,
    num_workers: int = 2,
    pretrained_backbone: bool = True,
) -> str:
    dev = _device(device)
    train_ds = YoloDetectionDataset(
        os.path.join(data_dir, "train", "images"),
        os.path.join(data_dir, "train", "labels"),
    )
    val_ds = YoloDetectionDataset(
        os.path.join(data_dir, "val", "images"),
        os.path.join(data_dir, "val", "labels"),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )

    model = build_model(arch=arch, pretrained_backbone=pretrained_backbone).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    best = float("inf")
    for epoch in range(epochs):
        tr = train_one_epoch(model, train_loader, optimizer, dev)
        vl = eval_loss(model, val_loader, dev)
        print(f"epoch {epoch + 1}/{epochs}  train_loss={tr:.4f}  val_loss={vl:.4f}")
        if vl < best:
            best = vl
            torch.save(model.state_dict(), out)
            print(f"  saved best -> {out} (val_loss={vl:.4f})")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train the fire/smoke detector")
    p.add_argument("--data", required=True, help="dataset root with train/ and val/")
    p.add_argument("--arch", default="ssdlite", choices=("ssdlite", "retinanet"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--out", default="models/fire_ssdlite.pt")
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=2)
    a = p.parse_args(argv)
    run_training(a.data, a.arch, a.epochs, a.batch_size, a.lr, a.out,
                 a.device, a.num_workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
