#!/usr/bin/env python3
"""Export epoch_45_model.pkl to TorchScript .pt using training repo architecture."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent
TRAIN_ROOT = Path("/mnt/data1/iorekhovsky/deepfake/detection/Unveiling_Deepfake")


def _pad_max_pool_traceable(self, x: torch.Tensor) -> torch.Tensor:
    h, w = x.shape[2], x.shape[3]
    padding = abs(h % 10 - 10) % 10
    if padding:
        x = F.pad(
            x,
            (padding // 2, (padding + 1) // 2, padding // 2, (padding + 1) // 2),
            mode="replicate",
        )
    h = x.shape[2]
    kernel = max(int(h) // 10, 1)
    return F.max_pool2d(x, kernel_size=kernel, stride=kernel, padding=0)


class InferWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export deepfake checkpoint to TorchScript")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "checkpoint" / "epoch_45_model.pkl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "checkpoint" / "epoch_45_model.pt",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def load_state_dict(path: Path, device: torch.device) -> dict:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    cleaned = {}
    for key, value in state.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def main() -> int:
    args = parse_args()
    if not TRAIN_ROOT.is_dir():
        raise FileNotFoundError(f"Training repo not found: {TRAIN_ROOT}")
    if not args.input.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.input}")

    sys.path.insert(0, str(TRAIN_ROOT))
    os.chdir(TRAIN_ROOT)

    from network.mymodel_bdct_dfcs_triplet_mi_loss import Xception_Net

    Xception_Net.pad_max_pool = _pad_max_pool_traceable

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Export requires CUDA because the model hardcodes .to('cuda') in features()")

    model = Xception_Net(pretrained=False)
    model.load_state_dict(load_state_dict(args.input, "cpu"))
    model.eval().to(device)

    wrapper = InferWrapper(model)
    dummy = torch.randn(1, 3, 299, 299, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy, strict=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(args.output))

    with torch.no_grad():
        ref = wrapper(dummy)
        out = traced(dummy)
        max_diff = (ref - out).abs().max().item()
    print(f"Saved: {args.output}")
    print(f"Max trace diff: {max_diff:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
