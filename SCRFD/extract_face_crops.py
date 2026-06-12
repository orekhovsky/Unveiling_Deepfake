#!/usr/bin/env python3
"""
Extract per-frame face crops from videos using SCRFD.

BBox is expanded by --bbox-scale (default 1.5), square crop.
Output side is max(crop_side, --min-size): upscale to min-size if smaller, keep if larger.

Run with SCRFD venv (GPU required for onnxruntime):
  source .venv/bin/activate
  CUDA_VISIBLE_DEVICES=4 python extract_face_crops.py ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore


def progress_iter(iterable, total=None, desc="videos"):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="video", dynamic_ncols=True)

# FaceDetector imported in worker after CUDA_VISIBLE_DEVICES is set

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
FAKE_SKIP_DIRS = {"background_aug"}

_detector = None
_worker_args = None


def parse_gpus(gpu_str: str) -> list[str]:
    return [x.strip() for x in gpu_str.split(",") if x.strip()]


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
    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(img_w, nx2)
    ny2 = min(img_h, ny2)
    return nx1, ny1, nx2, ny2


def pick_largest_face(det: np.ndarray) -> np.ndarray | None:
    if det is None or det.size == 0:
        return None
    areas = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
    return det[int(np.argmax(areas))]


def crop_face(frame: np.ndarray, det_row: np.ndarray, bbox_scale: float, min_size: int):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = det_row[:4]
    x1, y1, x2, y2 = expand_square_bbox(x1, y1, x2, y2, w, h, bbox_scale)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    side = max(crop.shape[0], crop.shape[1])
    if side < min_size:
        crop = cv2.resize(crop, (min_size, min_size), interpolation=cv2.INTER_LINEAR)
    return crop


def output_dir_for_video(video: Path, fake_root: Path, real_root: Path, out_root: Path) -> tuple[Path, int]:
    video = video.resolve()
    fake_root = fake_root.resolve()
    real_root = real_root.resolve()

    if str(video).startswith(str(fake_root)):
        rel = video.relative_to(fake_root)
        parts = rel.parts
        if parts and parts[0] in FAKE_SKIP_DIRS:
            raise ValueError(f"skip {video}")
        if len(parts) > 1:
            prefix = parts[0]
            name = f"{prefix}_{video.stem}"
        else:
            name = video.stem
        return out_root / "fake" / name, 1

    if str(video).startswith(str(real_root)):
        return out_root / "real" / video.stem, 0

    raise ValueError(f"video outside roots: {video}")


def collect_videos(fake_root: Path, real_root: Path) -> list[Path]:
    videos: list[Path] = []
    for path in sorted(fake_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if FAKE_SKIP_DIRS.intersection(path.relative_to(fake_root).parts):
            continue
        videos.append(path)
    for path in sorted(real_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        videos.append(path)
    return videos


def process_video(task: dict) -> dict:
    global _detector, _worker_args
    video = Path(task["video"])
    out_dir = Path(task["out_dir"])
    label = task["label"]

    stats = {
        "video": str(video),
        "out_dir": str(out_dir),
        "label": label,
        "frames_read": 0,
        "frames_saved": 0,
        "frames_no_face": 0,
        "status": "ok",
    }

    existing = sorted(out_dir.glob("*.png"))
    if existing and not _worker_args.get("overwrite"):
        stats["frames_saved"] = len(existing)
        stats["status"] = "skipped"
        return stats

    out_dir.mkdir(parents=True, exist_ok=True)
    if _worker_args.get("overwrite"):
        for p in out_dir.glob("*.png"):
            p.unlink()

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        stats["status"] = "open_failed"
        return stats

    frame_stride = int(_worker_args["frame_stride"])
    max_frames = int(_worker_args["max_frames"])
    bbox_scale = float(_worker_args["bbox_scale"])
    min_size = int(_worker_args["min_size"])
    input_size = tuple(_worker_args["input_size"])
    thresh = float(_worker_args["thresh"])

    frame_idx = 0
    save_idx = 0

    while True:
        if save_idx >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        stats["frames_read"] += 1
        det, _ = _detector.detect(frame, thresh=thresh, input_size=input_size, max_num=0)
        face = pick_largest_face(det)
        if face is None:
            stats["frames_no_face"] += 1
            frame_idx += 1
            continue

        crop = crop_face(frame, face, bbox_scale, min_size)
        if crop is None:
            stats["frames_no_face"] += 1
            frame_idx += 1
            continue

        save_idx += 1
        out_path = out_dir / f"{save_idx:04d}.png"
        if not cv2.imwrite(str(out_path), crop):
            stats["status"] = "write_failed"
            break
        stats["frames_saved"] += 1
        frame_idx += 1

    cap.release()
    if stats["frames_saved"] == 0 and stats["status"] == "ok":
        stats["status"] = "no_faces"
    return stats


def init_worker(gpu_id: str, model_path: str, worker_args: dict):
    global _detector, _worker_args
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    _worker_args = worker_args
    # Import after CUDA_VISIBLE_DEVICES so ONNX binds to the right GPU
    from nets.nn import FaceDetector

    _detector = FaceDetector(onnx_file=model_path)


def build_tasks(
    videos: list[Path],
    fake_root: Path,
    real_root: Path,
    out_root: Path,
) -> list[dict]:
    tasks = []
    for video in videos:
        try:
            out_dir, label = output_dir_for_video(video, fake_root, real_root, out_root)
        except ValueError:
            continue
        tasks.append({"video": str(video), "out_dir": str(out_dir), "label": label})
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description="SCRFD face crop extraction for training")
    parser.add_argument(
        "--fake-root",
        type=Path,
        default=Path("/mnt/data1/iorekhovsky/deepfake/datasets/from_s3/final_dataset/videos/fake"),
    )
    parser.add_argument(
        "--real-root",
        type=Path,
        default=Path("/mnt/data1/iorekhovsky/deepfake/datasets/from_s3/final_dataset/videos/real"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/data1/iorekhovsky/deepfake/detection/Unveiling_Deepfake/my_data"),
    )
    parser.add_argument("--model", type=Path, default=Path("weights/model_1.onnx"))
    parser.add_argument("--gpus", type=str, default="4,5", help="Physical GPU ids, e.g. 4,5")
    parser.add_argument("--workers", type=int, default=0, help="0 = one worker per GPU")
    parser.add_argument("--bbox-scale", type=float, default=1.5)
    parser.add_argument("--min-size", type=int, default=380)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=24,
        help="Process every N-th frame (default: 24)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=40,
        help="Max saved crops per video (default: 40)",
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--input-size", type=int, nargs=2, default=[640, 640])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    scrfd_root = Path(__file__).resolve().parent
    model_path = args.model if args.model.is_absolute() else scrfd_root / args.model
    if not model_path.is_file():
        print(f"Model not found: {model_path}", file=sys.stderr)
        return 1

    gpus = parse_gpus(args.gpus)
    if not gpus:
        print("No GPUs specified", file=sys.stderr)
        return 1

    n_workers = args.workers if args.workers > 0 else len(gpus)
    n_workers = min(n_workers, len(gpus))

    videos = collect_videos(args.fake_root, args.real_root)
    if args.limit > 0:
        videos = videos[: args.limit]

    tasks = build_tasks(videos, args.fake_root, args.real_root, args.output)
    print(f"Videos: {len(videos)}, tasks: {len(tasks)}, GPUs: {gpus}, workers: {n_workers}")

    frame_stride = args.frame_stride
    if args.every_n is not None:
        frame_stride = args.every_n

    worker_args = {
        "frame_stride": frame_stride,
        "max_frames": args.max_frames,
        "bbox_scale": args.bbox_scale,
        "min_size": args.min_size,
        "input_size": list(args.input_size),
        "thresh": args.thresh,
        "overwrite": args.overwrite,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (args.output / "manifest.jsonl")

    if tqdm is None:
        print("Tip: pip install tqdm  (for progress bar)", file=sys.stderr)

    if n_workers == 1:
        init_worker(gpus[0], str(model_path), worker_args)
        results = []
        for task in progress_iter(tasks, total=len(tasks)):
            results.append(process_video(task))
    else:
        results = run_parallel(tasks, gpus, n_workers, str(model_path), worker_args)

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ok = sum(1 for r in results if r["status"] in ("ok", "skipped"))
    no_face = sum(1 for r in results if r["status"] == "no_faces")
    failed = len(results) - ok - no_face
    print(f"Done. ok/skipped={ok}, no_faces={no_face}, failed={failed}")
    print(f"Manifest: {manifest_path}")
    return 0


def run_parallel(tasks, gpus, n_workers, model_path, worker_args):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    gpu_assign = [gpus[i % len(gpus)] for i in range(n_workers)]

    in_queues = [ctx.Queue() for _ in range(n_workers)]
    out_queue = ctx.Queue()

    procs = []
    for wid in range(n_workers):
        p = ctx.Process(
            target=_worker_loop,
            args=(wid, gpu_assign[wid], model_path, worker_args, in_queues[wid], out_queue),
        )
        p.start()
        procs.append(p)

    for i, task in enumerate(tasks):
        in_queues[i % n_workers].put(task)
    for q in in_queues:
        q.put(None)

    results = []
    bar = progress_iter(range(len(tasks)), total=len(tasks))
    for _ in bar:
        row = out_queue.get()
        results.append(row)
        if tqdm is not None and hasattr(bar, "set_postfix_str"):
            bar.set_postfix_str(str(row.get("status", "?")))

    for p in procs:
        p.join()

    return results


def _worker_loop(wid, gpu_id, model_path, worker_args, in_queue, out_queue):
    init_worker(gpu_id, model_path, worker_args)
    while True:
        task = in_queue.get()
        if task is None:
            break
        try:
            out_queue.put(process_video(task))
        except Exception as exc:
            out_queue.put(
                {
                    "video": task.get("video"),
                    "status": "error",
                    "error": str(exc),
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
