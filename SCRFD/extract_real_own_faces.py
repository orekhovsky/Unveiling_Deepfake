#!/usr/bin/env python3
"""
Извлечение одного лица на исходное видео из папки real.

Сегменты одного YouTube-ролика объединяются в одну группу по ID после «____»
и полу (_f / _m). Разные таймкоды и partN одного ролика дают одно лицо.
Сохраняется лицо с именем того сегмента, из которого оно взято.
Если лицо не найдено в одном сегменте — пробуем следующий.

Запуск (GPU + venv SCRFD):
  source .venv/bin/activate
  CUDA_VISIBLE_DEVICES=4 python extract_real_own_faces.py
  CUDA_VISIBLE_DEVICES=4 python extract_real_own_faces.py --build-manifest-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

import extract_face_crops as efc
from extract_face_crops import (
    crop_face,
    init_worker,
    parse_gpus,
    pick_largest_face,
    progress_iter,
)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# dutkin_files____{video_id}_{HH_MM_SS-HH_MM_SS}[partN]_{f|m}
DUTKIN_STEM_RE = re.compile(
    r"^(?P<prefix>.+?)____"
    r"(?P<video_id>.+?)_"
    r"(?P<t0>\d{2})_(?P<t1>\d{2})_(?P<t2>\d{2})-"
    r"(?P<t3>\d{2})_(?P<t4>\d{2})_(?P<t5>\d{2})"
    r"(?:part(?P<part>\d+))?_(?P<gender>[mf])$",
    re.IGNORECASE,
)

PART_STEM_RE = re.compile(
    r"^(?P<base>.+?)part(?P<part>\d+)_(?P<gender>[mf])$",
    re.IGNORECASE,
)


def parse_dutkin_stem(stem: str) -> re.Match | None:
    return DUTKIN_STEM_RE.match(stem)


def video_group_key(stem: str) -> str:
    """Один ключ на YouTube-ролик: {video_id}_{gender}."""
    match = parse_dutkin_stem(stem)
    if match:
        return f"{match.group('video_id')}_{match.group('gender').lower()}"
    return stem


def parse_part_number(stem: str) -> int | None:
    match = parse_dutkin_stem(stem)
    if match and match.group("part"):
        return int(match.group("part"))
    match = PART_STEM_RE.match(stem)
    if match:
        return int(match.group("part"))
    return None


def segment_sort_key(path: Path) -> tuple:
    stem = path.stem
    match = parse_dutkin_stem(stem)
    if match:
        time_key = (
            match.group("t0"),
            match.group("t1"),
            match.group("t2"),
            match.group("t3"),
            match.group("t4"),
            match.group("t5"),
        )
        part = int(match.group("part") or 0)
        return (time_key, part, path.name)
    part = parse_part_number(stem)
    return ((part is None,), part or 0, path.name)


def collect_videos(video_root: Path) -> list[Path]:
    videos: list[Path] = []
    for path in sorted(video_root.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        videos.append(path)
    return videos


def build_groups(videos: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for video in videos:
        groups[video_group_key(video.stem)].append(video)

    for group_key in groups:
        groups[group_key].sort(key=segment_sort_key)
    return dict(groups)


def write_groups_manifest(groups: dict[str, list[Path]], manifest_path: Path) -> None:
    rows = []
    for group_key in sorted(groups):
        segments = []
        for video in groups[group_key]:
            part = parse_part_number(video.stem)
            segments.append(
                {
                    "filename": video.name,
                    "path": str(video.resolve()),
                    "part": part,
                }
            )
        rows.append({"group_key": group_key, "segments": segments})

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def try_extract_face_from_video(video: Path) -> np.ndarray | None:
    worker_args = efc._worker_args
    detector = efc._detector

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None

    frame_stride = int(worker_args["frame_stride"])
    max_frames = int(worker_args["max_frames_per_segment"])
    bbox_scale = float(worker_args["bbox_scale"])
    min_size = int(worker_args["min_size"])
    input_size = tuple(worker_args["input_size"])
    thresh = float(worker_args["thresh"])

    frame_idx = 0
    tried = 0

    while tried < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        tried += 1
        det, _ = detector.detect(frame, thresh=thresh, input_size=input_size, max_num=0)
        face = pick_largest_face(det)
        if face is None:
            frame_idx += 1
            continue

        crop = crop_face(frame, face, bbox_scale, min_size)
        if crop is not None:
            cap.release()
            return crop

        frame_idx += 1

    cap.release()
    return None


def process_group(task: dict) -> dict:
    group_key = task["group_key"]
    out_dir = Path(task["out_dir"])
    segments = task["segments"]

    result = {
        "group_key": group_key,
        "status": "no_faces",
        "source_video": None,
        "source_filename": None,
        "output_path": None,
        "segments_tried": [],
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    for segment in segments:
        video = Path(segment["path"])
        filename = segment["filename"]
        out_path = out_dir / f"{video.stem}.png"

        result["segments_tried"].append(filename)

        if out_path.is_file() and not efc._worker_args.get("overwrite"):
            result.update(
                {
                    "status": "skipped",
                    "source_video": str(video.resolve()),
                    "source_filename": filename,
                    "output_path": str(out_path.resolve()),
                }
            )
            return result

        crop = try_extract_face_from_video(video)
        if crop is None:
            continue

        if not cv2.imwrite(str(out_path), crop):
            result["status"] = "write_failed"
            result["source_video"] = str(video.resolve())
            result["source_filename"] = filename
            return result

        result.update(
            {
                "status": "ok",
                "source_video": str(video.resolve()),
                "source_filename": filename,
                "output_path": str(out_path.resolve()),
            }
        )
        return result

    return result


def _worker_loop_groups(gpu_id, model_path, worker_args, in_queue, out_queue):
    init_worker(gpu_id, model_path, worker_args)
    while True:
        task = in_queue.get()
        if task is None:
            break
        try:
            out_queue.put(process_group(task))
        except Exception as exc:
            out_queue.put(
                {
                    "group_key": task.get("group_key"),
                    "status": "error",
                    "error": str(exc),
                }
            )


def run_parallel_groups(tasks, gpus, n_workers, model_path, worker_args):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    gpu_assign = [gpus[i % len(gpus)] for i in range(n_workers)]
    in_queues = [ctx.Queue() for _ in range(n_workers)]
    out_queue = ctx.Queue()

    procs = []
    for wid in range(n_workers):
        p = ctx.Process(
            target=_worker_loop_groups,
            args=(gpu_assign[wid], model_path, worker_args, in_queues[wid], out_queue),
        )
        p.start()
        procs.append(p)

    for i, task in enumerate(tasks):
        in_queues[i % n_workers].put(task)
    for q in in_queues:
        q.put(None)

    results = []
    bar = progress_iter(range(len(tasks)), total=len(tasks), desc="groups")
    for _ in bar:
        row = out_queue.get()
        results.append(row)
        if tqdm is not None and hasattr(bar, "set_postfix_str"):
            bar.set_postfix_str(str(row.get("status", "?")))

    for p in procs:
        p.join()

    return results


def build_tasks(groups: dict[str, list[Path]], out_dir: Path) -> list[dict]:
    tasks = []
    for group_key in sorted(groups):
        segments = []
        for video in groups[group_key]:
            segments.append(
                {
                    "filename": video.name,
                    "path": str(video.resolve()),
                    "part": parse_part_number(video.stem),
                }
            )
        tasks.append(
            {
                "group_key": group_key,
                "out_dir": str(out_dir),
                "segments": segments,
            }
        )
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Одно лицо на исходное real-видео (SCRFD, группировка partN)"
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path(
            "/mnt/data1/iorekhovsky/deepfake/datasets/from_s3/final_dataset/videos/real"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/data1/iorekhovsky/deepfake/faces/real_own_faces"),
    )
    parser.add_argument(
        "--groups-manifest",
        type=Path,
        default=None,
        help="JSON со списком групп и сегментов (по умолчанию: <output>/groups_manifest.json)",
    )
    parser.add_argument("--model", type=Path, default=Path("weights/model_1.onnx"))
    parser.add_argument("--gpus", type=str, default="4,5")
    parser.add_argument("--workers", type=int, default=0, help="0 = по одному на GPU")
    parser.add_argument("--bbox-scale", type=float, default=1.4)
    parser.add_argument("--min-size", type=int, default=0, help="0 = без апскейла")
    parser.add_argument("--frame-stride", type=int, default=12)
    parser.add_argument(
        "--max-frames-per-segment",
        type=int,
        default=30,
        help="Сколько кадров пробовать в каждом сегменте",
    )
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--input-size", type=int, nargs=2, default=[640, 640])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--build-manifest-only",
        action="store_true",
        help="Только собрать groups_manifest.json, без детекции",
    )
    args = parser.parse_args()

    scrfd_root = Path(__file__).resolve().parent
    model_path = args.model if args.model.is_absolute() else scrfd_root / args.model

    if not args.video_root.is_dir():
        print(f"Video root not found: {args.video_root}", file=sys.stderr)
        return 1

    videos = collect_videos(args.video_root)
    groups = build_groups(videos)

    manifest_path = args.groups_manifest or (args.output / "groups_manifest.json")
    write_groups_manifest(groups, manifest_path)

    n_segments = sum(len(v) for v in groups.values())
    print(f"Videos: {len(videos)}, groups: {len(groups)}, segments total: {n_segments}")
    print(f"Groups manifest: {manifest_path.resolve()}")

    if args.build_manifest_only:
        return 0

    if not model_path.is_file():
        print(f"Model not found: {model_path}", file=sys.stderr)
        return 1

    gpus = parse_gpus(args.gpus)
    if not gpus:
        print("No GPUs specified", file=sys.stderr)
        return 1

    n_workers = args.workers if args.workers > 0 else len(gpus)
    n_workers = min(n_workers, len(gpus))

    tasks = build_tasks(groups, args.output)
    if args.limit > 0:
        tasks = tasks[: args.limit]

    worker_args = {
        "frame_stride": args.frame_stride,
        "max_frames_per_segment": args.max_frames_per_segment,
        "bbox_scale": args.bbox_scale,
        "min_size": args.min_size,
        "input_size": list(args.input_size),
        "thresh": args.thresh,
        "overwrite": args.overwrite,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "extraction_results.jsonl"

    if tqdm is None:
        print("Tip: pip install tqdm  (for progress bar)", file=sys.stderr)

    if n_workers == 1:
        init_worker(gpus[0], str(model_path), worker_args)
        results = []
        for task in progress_iter(tasks, total=len(tasks), desc="groups"):
            results.append(process_group(task))
    else:
        results = run_parallel_groups(tasks, gpus, n_workers, str(model_path), worker_args)

    with results_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ok = sum(1 for r in results if r["status"] in ("ok", "skipped"))
    no_face = sum(1 for r in results if r["status"] == "no_faces")
    failed = len(results) - ok - no_face
    print(f"Done. ok/skipped={ok}, no_faces={no_face}, failed={failed}")
    print(f"Faces: {args.output.resolve()}")
    print(f"Results: {results_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
