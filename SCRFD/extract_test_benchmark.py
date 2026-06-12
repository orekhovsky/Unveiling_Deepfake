#!/usr/bin/env python3
"""
Face-crop preprocessing for benchmark test sets (part1 videos + part2 images).

Labels from CSV (column ``label``). Output layout matches Unveiling_Deepfake MyDataset:
  test_data/part1/{obj_id}/0001.png ...
  test_data/part2/{obj_id}/0001.png
  test_data/test.txt

Run from SCRFD dir with venv + GPU:
  source .venv/bin/activate
  python extract_test_benchmark.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

from extract_face_crops import (
    crop_face,
    init_worker,
    parse_gpus,
    pick_largest_face,
    process_video,
    progress_iter,
    run_parallel,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def load_csv_labels(csv_path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj_id = row["obj_id"].strip()
            labels[obj_id] = int(row["label"])
    return labels


def find_media_path(media_dir: Path, obj_id: str) -> Path | None:
    for ext in list(VIDEO_EXTENSIONS) + list(IMAGE_EXTENSIONS):
        path = media_dir / f"{obj_id}{ext}"
        if path.is_file():
            return path
    return None


def process_image(task: dict) -> dict:
    from extract_face_crops import _detector, _worker_args

    image_path = Path(task["source"])
    out_dir = Path(task["out_dir"])
    label = task["label"]

    stats = {
        "obj_id": task["obj_id"],
        "kind": "image",
        "source": str(image_path),
        "out_dir": str(out_dir),
        "label": label,
        "frames_saved": 0,
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

    frame = cv2.imread(str(image_path))
    if frame is None:
        stats["status"] = "read_failed"
        return stats

    det, _ = _detector.detect(
        frame,
        thresh=float(_worker_args["thresh"]),
        input_size=tuple(_worker_args["input_size"]),
        max_num=0,
    )
    face = pick_largest_face(det)
    if face is None:
        stats["status"] = "no_faces"
        return stats

    crop = crop_face(
        frame,
        face,
        float(_worker_args["bbox_scale"]),
        int(_worker_args["min_size"]),
    )
    if crop is None:
        stats["status"] = "no_faces"
        return stats

    out_path = out_dir / "0001.png"
    if not cv2.imwrite(str(out_path), crop):
        stats["status"] = "write_failed"
        return stats

    stats["frames_saved"] = 1
    return stats


def process_task(task: dict) -> dict:
    if task["kind"] == "video":
        row = process_video(task)
        row["obj_id"] = task["obj_id"]
        row["kind"] = "video"
        return row
    return process_image(task)


def build_tasks(
    labels: dict[str, int],
    media_dir: Path,
    out_subdir: str,
    out_root: Path,
    kind: str,
) -> tuple[list[dict], list[dict]]:
    tasks: list[dict] = []
    missing: list[dict] = []
    for obj_id, label in sorted(labels.items()):
        src = find_media_path(media_dir, obj_id)
        out_dir = out_root / out_subdir / obj_id
        if src is None:
            missing.append({"obj_id": obj_id, "label": label, "status": "file_not_found"})
            continue
        if kind == "video":
            tasks.append(
                {
                    "kind": "video",
                    "obj_id": obj_id,
                    "video": str(src),
                    "out_dir": str(out_dir),
                    "label": label,
                }
            )
        else:
            tasks.append(
                {
                    "kind": "image",
                    "obj_id": obj_id,
                    "source": str(src),
                    "out_dir": str(out_dir),
                    "label": label,
                }
            )
    return tasks, missing


def write_test_lists(results: list[dict], out_root: Path) -> dict[str, int]:
    buckets: dict[str, list[tuple[int, str]]] = {
        "all": [],
        "part1": [],
        "part2": [],
    }
    for row in results:
        if row.get("frames_saved", 0) <= 0:
            continue
        if row.get("status") not in ("ok", "skipped"):
            continue
        out_dir = str(row["out_dir"])
        label = int(row["label"])
        entry = (label, out_dir)
        buckets["all"].append(entry)
        kind = row.get("kind")
        if kind == "video" or "/part1/" in out_dir:
            buckets["part1"].append(entry)
        elif kind == "image" or "/part2/" in out_dir:
            buckets["part2"].append(entry)

    counts: dict[str, int] = {}
    files = {
        "all": out_root / "test.txt",
        "part1": out_root / "test_part1.txt",
        "part2": out_root / "test_part2.txt",
    }
    for key, path in files.items():
        with path.open("w", encoding="utf-8") as f:
            for label, folder in sorted(buckets[key], key=lambda x: x[1]):
                f.write(f"{label},{Path(folder).resolve()}\n")
        counts[key] = len(buckets[key])
    return counts


def run_parallel_tasks(tasks, gpus, n_workers, model_path, worker_args):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    gpu_assign = [gpus[i % len(gpus)] for i in range(n_workers)]
    in_queues = [ctx.Queue() for _ in range(n_workers)]
    out_queue = ctx.Queue()

    procs = []
    for wid in range(n_workers):
        p = ctx.Process(
            target=_worker_loop_tasks,
            args=(gpu_assign[wid], model_path, worker_args, in_queues[wid], out_queue),
        )
        p.start()
        procs.append(p)

    for i, task in enumerate(tasks):
        in_queues[i % n_workers].put(task)
    for q in in_queues:
        q.put(None)

    results = []
    bar = progress_iter(range(len(tasks)), total=len(tasks), desc="items")
    for _ in bar:
        row = out_queue.get()
        results.append(row)
        try:
            from tqdm import tqdm

            if isinstance(bar, tqdm):
                bar.set_postfix_str(str(row.get("status", "?")))
        except ImportError:
            pass

    for p in procs:
        p.join()
    return results


def _worker_loop_tasks(gpu_id, model_path, worker_args, in_queue, out_queue):
    init_worker(gpu_id, model_path, worker_args)
    while True:
        task = in_queue.get()
        if task is None:
            break
        try:
            out_queue.put(process_task(task))
        except Exception as exc:
            out_queue.put(
                {
                    "obj_id": task.get("obj_id"),
                    "kind": task.get("kind"),
                    "status": "error",
                    "error": str(exc),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess benchmark test data (SCRFD crops)")
    parser.add_argument(
        "--test-root",
        type=Path,
        default=Path("/mnt/data1/iorekhovsky/deepfake/datasets/test"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/data1/iorekhovsky/deepfake/detection/Unveiling_Deepfake/test_data"),
    )
    parser.add_argument("--part1-csv", type=Path, default=None)
    parser.add_argument("--part2-csv", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=Path("weights/model_1.onnx"))
    parser.add_argument("--gpus", type=str, default="4,5")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bbox-scale", type=float, default=1.5)
    parser.add_argument("--min-size", type=int, default=380)
    parser.add_argument("--frame-stride", type=int, default=24)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--input-size", type=int, nargs=2, default=[640, 640])
    parser.add_argument("--limit-part1", type=int, default=0)
    parser.add_argument("--limit-part2", type=int, default=0)
    parser.add_argument("--skip-part2", action="store_true")
    parser.add_argument("--skip-part1", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    scrfd_root = Path(__file__).resolve().parent
    model_path = args.model if args.model.is_absolute() else scrfd_root / args.model
    if not model_path.is_file():
        print(f"Model not found: {model_path}", file=sys.stderr)
        return 1

    part1_csv = args.part1_csv or (args.test_root / "part1.csv")
    part2_csv = args.part2_csv or (args.test_root / "part2.csv")
    part1_dir = args.test_root / "part1"
    part2_dir = args.test_root / "part2"

    gpus = parse_gpus(args.gpus)
    n_workers = args.workers if args.workers > 0 else len(gpus)
    n_workers = min(n_workers, len(gpus))

    worker_args = {
        "frame_stride": args.frame_stride,
        "max_frames": args.max_frames,
        "bbox_scale": args.bbox_scale,
        "min_size": args.min_size,
        "input_size": list(args.input_size),
        "thresh": args.thresh,
        "overwrite": args.overwrite,
    }

    all_tasks: list[dict] = []
    missing: list[dict] = []

    if not args.skip_part1:
        p1_labels = load_csv_labels(part1_csv)
        p1_tasks, p1_missing = build_tasks(p1_labels, part1_dir, "part1", args.output, "video")
        if args.limit_part1 > 0:
            p1_tasks = p1_tasks[: args.limit_part1]
        all_tasks.extend(p1_tasks)
        missing.extend(p1_missing)
        print(f"part1: {len(p1_tasks)} videos ({len(p1_missing)} missing files)")

    if not args.skip_part2:
        p2_labels = load_csv_labels(part2_csv)
        p2_tasks, p2_missing = build_tasks(p2_labels, part2_dir, "part2", args.output, "image")
        if args.limit_part2 > 0:
            p2_tasks = p2_tasks[: args.limit_part2]
        all_tasks.extend(p2_tasks)
        missing.extend(p2_missing)
        print(f"part2: {len(p2_tasks)} images ({len(p2_missing)} missing files)")

    print(f"Total tasks: {len(all_tasks)}, GPUs: {gpus}, workers: {n_workers}")
    args.output.mkdir(parents=True, exist_ok=True)

    if not all_tasks:
        print("No tasks to process", file=sys.stderr)
        return 1

    if n_workers == 1:
        init_worker(gpus[0], str(model_path), worker_args)
        results = []
        for task in progress_iter(all_tasks, total=len(all_tasks), desc="items"):
            results.append(process_task(task))
    else:
        results = run_parallel_tasks(all_tasks, gpus, n_workers, str(model_path), worker_args)

    results.extend(missing)
    manifest_path = args.output / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    list_counts = write_test_lists(results, args.output)
    ok = sum(1 for r in results if r.get("status") in ("ok", "skipped"))
    no_face = sum(1 for r in results if r.get("status") == "no_faces")
    print(f"Done. ok/skipped={ok}, no_faces={no_face}")
    print(
        f"test.txt={list_counts.get('all', 0)}, "
        f"test_part1.txt={list_counts.get('part1', 0)}, "
        f"test_part2.txt={list_counts.get('part2', 0)}"
    )
    print(f"Output: {args.output.resolve()}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
