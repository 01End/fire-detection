"""Normalize a training checkpoint into a bare inference state_dict.

``train.py`` already saves a bare ``state_dict``; this helper exists for checkpoints that
wrap it (e.g. ``{"model_state_dict": ..., "optimizer": ...}``) so the runtime loader
always receives the clean weights it expects.
"""
from __future__ import annotations

import argparse

import torch


def export(checkpoint_path: str, out_path: str) -> str:
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    torch.save(state, out_path)
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Export a clean inference state_dict")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    print("exported ->", export(a.checkpoint, a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
