#!/usr/bin/env python3
"""Deepfake detection inference on images and videos with SCRFD face preprocessing."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import albumentations as alb
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch.transforms import ToTensorV2

REPO_ROOT = Path(__file__).resolve().parent
SCRFD_ROOT = REPO_ROOT / "SCRFD"
TMP_ROOT = REPO_ROOT / "tmp"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

FACE_TRANSFORM = alb.Compose(
    [
        alb.Resize(299, 299, interpolation=1),
        alb.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ]
)

COLOR_REAL = (0, 255, 0)
COLOR_FAKE = (0, 0, 255)


@dataclass
class FramePrediction:
    frame_index: int
    bbox: tuple[int, int, int, int] | None
    fake_prob: float | None
    face_score: float | None
    status: str


@dataclass
class MediaResult:
    input_path: str
    media_type: str
    label: str
    fake_prob: float
    real_prob: float
    confidence: float
    frames: list[FramePrediction]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect real vs fake faces in images and videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Path to image or video")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoint" / "epoch_45_model.pt",
        help="TorchScript deepfake detector (.pt)",
    )
    parser.add_argument(
        "--scrfd-model",
        type=Path,
        default=SCRFD_ROOT / "weights" / "model_1.onnx",
        help="SCRFD ONNX weights",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for overlay image/video (used with --overlay)",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Draw face box and real/fake label on the output",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--bbox-scale",
        type=float,
        default=1.0,
        help="Square crop side = max(bbox_w, bbox_h) * scale (must match training)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=380,
        help="Upscale crop to this min side (px) if smaller; larger crops are kept as-is",
    )
    parser.add_argument("--scrfd-thresh", type=float, default=0.5)
    parser.add_argument("--scrfd-input-size", type=int, nargs=2, default=[640, 640])
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep extracted video frames in tmp/",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to save prediction JSON",
    )
    return parser.parse_args()


def resolve_input(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    raise ValueError(f"Unsupported input type: {path}")


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


def load_scrfd(model_path: Path):
    sys.path.insert(0, str(SCRFD_ROOT))
    from nets.nn import FaceDetector

    return FaceDetector(onnx_file=str(model_path))


def load_detector(checkpoint: Path, device: torch.device):
    if checkpoint.suffix.lower() != ".pt":
        raise ValueError(f"Expected TorchScript checkpoint .pt, got: {checkpoint}")
    model = torch.jit.load(str(checkpoint), map_location=device)
    model.eval()
    return model


def predict_face(
    model,
    face_bgr: np.ndarray,
    device: torch.device,
) -> float:
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    tensor = FACE_TRANSFORM(image=face_rgb)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
    return float(probs[1].item())


def detect_face_on_frame(
    detector,
    frame: np.ndarray,
    scrfd_thresh: float,
    scrfd_input_size: tuple[int, int],
    bbox_scale: float,
    min_size: int,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, float | None]:
    det, _ = detector.detect(
        frame,
        thresh=scrfd_thresh,
        input_size=scrfd_input_size,
        max_num=0,
    )
    face = pick_largest_face(det)
    if face is None:
        return None, None, None
    crop, bbox = crop_face(frame, face, bbox_scale, min_size)
    return crop, bbox, float(face[4])


def draw_overlay(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    label: str,
    confidence: float,
    color: tuple[int, int, int],
) -> np.ndarray:
    out = frame.copy()
    if bbox is None:
        text = f"{label} {confidence:.2%} (no face)"
        cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        return out

    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
    text = f"{label} {confidence:.2%}"
    text_y = max(28, y1 - 10)
    cv2.putText(out, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return out


def aggregate_label(frame_probs: list[float]) -> tuple[str, float, float]:
    if not frame_probs:
        return "unknown", 0.5, 0.5
    fake_prob = float(np.mean(frame_probs))
    real_prob = 1.0 - fake_prob
    label = "fake" if fake_prob >= 0.5 else "real"
    confidence = fake_prob if label == "fake" else real_prob
    return label, fake_prob, confidence


def extract_video_frames(video_path: Path, tmp_dir: Path) -> tuple[list[Path], float, tuple[int, int]]:
    frames_dir = tmp_dir / video_path.stem
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_paths: list[Path] = []
    index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_path = frames_dir / f"frame_{index:06d}.jpg"
        if not cv2.imwrite(str(frame_path), frame):
            cap.release()
            raise RuntimeError(f"Failed to write frame: {frame_path}")
        frame_paths.append(frame_path)
        index += 1

    cap.release()
    if not frame_paths:
        raise RuntimeError(f"No frames extracted from video: {video_path}")
    return frame_paths, fps, (width, height)


def process_image(
    input_path: Path,
    detector,
    model,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[MediaResult, np.ndarray | None]:
    frame = cv2.imread(str(input_path))
    if frame is None:
        raise RuntimeError(f"Failed to read image: {input_path}")

    crop, bbox, face_score = detect_face_on_frame(
        detector,
        frame,
        args.scrfd_thresh,
        tuple(args.scrfd_input_size),
        args.bbox_scale,
        args.min_size,
    )

    frame_pred = FramePrediction(
        frame_index=0,
        bbox=bbox,
        fake_prob=None,
        face_score=face_score,
        status="no_face",
    )
    fake_probs: list[float] = []

    if crop is not None:
        fake_prob = predict_face(model, crop, device)
        fake_probs.append(fake_prob)
        frame_pred.fake_prob = fake_prob
        frame_pred.status = "ok"

    label, fake_prob, confidence = aggregate_label(fake_probs)
    result = MediaResult(
        input_path=str(input_path),
        media_type="image",
        label=label,
        fake_prob=fake_prob,
        real_prob=1.0 - fake_prob,
        confidence=confidence,
        frames=[frame_pred],
    )

    overlay_frame = None
    if args.overlay:
        color = COLOR_FAKE if label == "fake" else COLOR_REAL
        overlay_frame = draw_overlay(frame, bbox, label, confidence, color)

    return result, overlay_frame


def process_video(
    input_path: Path,
    detector,
    model,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[MediaResult, list[np.ndarray] | None, float, tuple[int, int]]:
    tmp_dir = TMP_ROOT / input_path.stem
    frame_paths, fps, size = extract_video_frames(input_path, TMP_ROOT)

    frame_preds: list[FramePrediction] = []
    fake_probs: list[float] = []
    overlay_frames: list[np.ndarray] | None = [] if args.overlay else None

    for index, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            frame_preds.append(
                FramePrediction(index, None, None, None, "read_failed")
            )
            if overlay_frames is not None:
                overlay_frames.append(np.zeros((size[1], size[0], 3), dtype=np.uint8))
            continue

        crop, bbox, face_score = detect_face_on_frame(
            detector,
            frame,
            args.scrfd_thresh,
            tuple(args.scrfd_input_size),
            args.bbox_scale,
            args.min_size,
        )

        frame_pred = FramePrediction(
            frame_index=index,
            bbox=bbox,
            fake_prob=None,
            face_score=face_score,
            status="no_face",
        )

        if crop is not None:
            fake_prob = predict_face(model, crop, device)
            fake_probs.append(fake_prob)
            frame_pred.fake_prob = fake_prob
            frame_pred.status = "ok"

        frame_preds.append(frame_pred)

    label, fake_prob, confidence = aggregate_label(fake_probs)
    result = MediaResult(
        input_path=str(input_path),
        media_type="video",
        label=label,
        fake_prob=fake_prob,
        real_prob=1.0 - fake_prob,
        confidence=confidence,
        frames=frame_preds,
    )

    if overlay_frames is not None:
        color = COLOR_FAKE if label == "fake" else COLOR_REAL
        for frame_path, frame_pred in zip(frame_paths, frame_preds):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                overlay_frames.append(np.zeros((size[1], size[0], 3), dtype=np.uint8))
                continue
            overlay_frames.append(
                draw_overlay(frame, frame_pred.bbox, label, confidence, color)
            )

    if not args.keep_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return result, overlay_frames, fps, size


def default_output_path(input_path: Path, overlay: bool) -> Path:
    suffix = input_path.suffix or ".jpg"
    return input_path.with_name(f"{input_path.stem}_overlay{suffix}")


def save_overlay_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def save_overlay_video(path: Path, frames: list[np.ndarray], fps: float, size: tuple[int, int]) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found in PATH. Install it on the system, e.g.: "
            "sudo apt install ffmpeg"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        str(path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None

    try:
        for frame in frames:
            h, w = frame.shape[:2]
            if (w, h) != (width, height):
                frame = cv2.resize(frame, (width, height))
            proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    finally:
        proc.stdin.close()

    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed to write {path}:\n{stderr[-3000:]}")


def result_to_dict(result: MediaResult) -> dict:
    return {
        "input": result.input_path,
        "type": result.media_type,
        "label": result.label,
        "fake_prob": result.fake_prob,
        "real_prob": result.real_prob,
        "confidence": result.confidence,
        "frames": [
            {
                "index": frame.frame_index,
                "status": frame.status,
                "bbox": frame.bbox,
                "fake_prob": frame.fake_prob,
                "face_score": frame.face_score,
            }
            for frame in result.frames
        ],
    }


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    media_type = resolve_input(input_path)

    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    scrfd_model = args.scrfd_model if args.scrfd_model.is_absolute() else REPO_ROOT / args.scrfd_model

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not scrfd_model.is_file():
        raise FileNotFoundError(f"SCRFD model not found: {scrfd_model}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    TMP_ROOT.mkdir(parents=True, exist_ok=True)

    face_detector = load_scrfd(scrfd_model)
    model = load_detector(checkpoint, device)

    if media_type == "image":
        result, overlay = process_image(input_path, face_detector, model, device, args)
        if args.overlay and overlay is not None:
            out_path = args.output or default_output_path(input_path, True)
            save_overlay_image(out_path, overlay)
            print(f"Saved overlay: {out_path}", file=sys.stderr)
    else:
        result, overlay_frames, fps, size = process_video(
            input_path, face_detector, model, device, args
        )
        if args.overlay and overlay_frames is not None:
            out_path = args.output or default_output_path(input_path, True)
            save_overlay_video(out_path, overlay_frames, fps, size)
            print(f"Saved overlay: {out_path}", file=sys.stderr)

    payload = result_to_dict(result)
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
