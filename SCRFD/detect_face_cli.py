#!/usr/bin/env python3
"""Detect largest face in an image. Run with SCRFD/.venv (onnxruntime-gpu)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from nets.nn import FaceDetector


def expand_square_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
    scale: float,
) -> tuple[int, int, int, int]:
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    side = max(x2 - x1, y2 - y1) * scale
    half = side * 0.5
    nx1 = int(round(cx - half))
    ny1 = int(round(cy - half))
    nx2 = int(round(cx + half))
    ny2 = int(round(cy + half))
    return max(0, nx1), max(0, ny1), min(img_w, nx2), min(img_h, ny2)


def pick_largest_face(det: np.ndarray | None) -> np.ndarray | None:
    if det is None or det.size == 0:
        return None
    areas = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
    return det[int(np.argmax(areas))]


def crop_face(
    frame: np.ndarray,
    det_row: np.ndarray,
    bbox_scale: float,
    min_size: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = expand_square_bbox(*det_row[:4], w, h, bbox_scale)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid face bbox")
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Empty face crop")
    side = max(crop.shape[0], crop.shape[1])
    if side < min_size:
        crop = cv2.resize(crop, (min_size, min_size), interpolation=cv2.INTER_LINEAR)
    return crop, (x1, y1, x2, y2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCRFD face detection CLI")
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument("--model", type=Path, default=Path("weights/model_1.onnx"))
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--input-size", type=int, nargs=2, default=[640, 640])
    parser.add_argument("--bbox-scale", type=float, default=1.5)
    parser.add_argument("--min-size", type=int, default=380)
    parser.add_argument("--crop-out", type=Path, default=None, help="Save face crop here")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scrfd_root = Path(__file__).resolve().parent
    model_path = args.model if args.model.is_absolute() else scrfd_root / args.model

    frame = cv2.imread(str(args.image))
    if frame is None:
        print(json.dumps({"status": "read_failed"}))
        return 1

    detector = FaceDetector(onnx_file=str(model_path))
    det, _ = detector.detect(
        frame,
        thresh=args.thresh,
        input_size=tuple(args.input_size),
        max_num=0,
    )
    face = pick_largest_face(det)
    if face is None:
        print(json.dumps({"status": "no_face"}))
        return 0

    crop, bbox = crop_face(frame, face, args.bbox_scale, args.min_size)
    payload = {
        "status": "ok",
        "bbox": list(bbox),
        "face_score": float(face[4]),
    }

    if args.crop_out is not None:
        args.crop_out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.crop_out), crop):
            print(json.dumps({"status": "write_failed"}))
            return 1
        payload["crop_path"] = str(args.crop_out.resolve())

    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
