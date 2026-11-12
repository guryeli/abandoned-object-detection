"""
Benchmark runner.

Folder layout:
    data/
        videos/
            video1.mp4
            video2.mp4
        ground_truth.csv
            video,abandonment_start_sec,abandonment_end_sec
            video1.mp4,32.5,60.0
            video2.mp4,18.0,45.2
            ...
            (if the video has no abandoned object, do not add this row —
             it is counted automatically as a negative sample)

Metrics:
    - Detection: was it marked as abandoned?
    - Latency: seconds between the ground-truth start and the alarm time
    - False positive: did a video with no GT raise an alarm?
    - Precision / Recall

Usage:
    python -m eval.benchmark --videos data/videos --gt data/ground_truth.csv
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

from config import Config
from methods.method1_distance import Method1Pipeline
from core.ownership import BagState


def run_video(video_path: str, cfg: Config, headless: bool = True,
              frame_skip: int = 1, log_every_sec: float = 2.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    # Effective FPS for the abandonment timer: every Nth frame is processed
    # so timer's frame counter ticks at src_fps/frame_skip.
    cfg.target_fps = src_fps / max(1, frame_skip)

    pipeline = Method1Pipeline(cfg)
    first_alarm_frame = None
    alarm_count = 0
    processed_frames = 0
    read_idx = 0
    t_start = time.time()
    t_last_log = t_start

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        read_idx += 1
        if frame_skip > 1 and (read_idx - 1) % frame_skip != 0:
            continue
        processed_frames += 1
        annotated, bag_records, _, _ = pipeline.process_frame(frame)

        for rec in bag_records.values():
            if rec.state == BagState.ABANDONED:
                if first_alarm_frame is None:
                    first_alarm_frame = pipeline.frame_idx
                alarm_count += 1
                break

        now = time.time()
        if now - t_last_log >= log_every_sec:
            elapsed = now - t_start
            fps_proc = processed_frames / elapsed if elapsed > 0 else 0.0
            pct = (read_idx / total_video_frames * 100.0) if total_video_frames else 0.0
            alarm_str = "ALARM" if first_alarm_frame is not None else "----"
            print(f"      [{Path(video_path).name}] read={read_idx}/{total_video_frames} "
                  f"({pct:5.1f}%) proc={processed_frames} "
                  f"speed={fps_proc:5.1f}fps  {alarm_str}", flush=True)
            t_last_log = now

        if not headless:
            cv2.imshow("benchmark", annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    elapsed_total = time.time() - t_start
    # first_alarm_sec is in source-video time, which equals frame_idx * frame_skip / src_fps
    first_alarm_sec = None
    if first_alarm_frame is not None and src_fps:
        first_alarm_sec = first_alarm_frame * frame_skip / src_fps

    return {
        "video": Path(video_path).name,
        "fps": src_fps,
        "frame_skip": frame_skip,
        "total_frames": read_idx,
        "processed_frames": processed_frames,
        "duration_sec": read_idx / src_fps if src_fps else 0,
        "wall_time_sec": round(elapsed_total, 1),
        "speed_fps": round(processed_frames / elapsed_total, 1) if elapsed_total > 0 else 0,
        "first_alarm_frame": first_alarm_frame,
        "first_alarm_sec": first_alarm_sec,
        "any_alarm": first_alarm_frame is not None,
    }


def evaluate(results, gt_df):
    """Compute detection, latency, and false-positive metrics."""
    rows = []
    tp = fp = fn = tn = 0
    latencies = []

    for r in results:
        vid = r["video"]
        gt_row = gt_df[gt_df["video"] == vid]
        has_gt = len(gt_row) > 0
        gt_start = None
        if has_gt:
            raw = gt_row["abandonment_start_sec"].values[0]
            try:
                val = float(raw)
                # NaN -> latency is not computed, but has_gt stays True
                if val == val:  # NaN check
                    gt_start = val
            except (TypeError, ValueError):
                gt_start = None

        predicted = r["any_alarm"]
        if has_gt and predicted:
            tp += 1
            if r["first_alarm_sec"] is not None and gt_start is not None:
                latencies.append(r["first_alarm_sec"] - gt_start)
        elif has_gt and not predicted:
            fn += 1
        elif not has_gt and predicted:
            fp += 1
        else:
            tn += 1

        rows.append({
            **r,
            "gt_start_sec": gt_start,
            "label": "TP" if has_gt and predicted else
                     "FN" if has_gt and not predicted else
                     "FP" if not has_gt and predicted else "TN",
            "latency_sec": (r["first_alarm_sec"] - gt_start)
                           if (r["first_alarm_sec"] is not None and gt_start is not None) else None,
        })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else None

    summary = {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1": round(f1, 3),
        "AvgLatency_sec": round(avg_lat, 2) if avg_lat is not None else None,
    }
    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", required=True, help="video folder")
    parser.add_argument("--gt", required=True, help="ground truth csv")
    parser.add_argument("--out", default="benchmark_results.csv")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default=None, help="yolov8n.pt, yolov8s.pt, etc.")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO inference size (320/480/640)")
    parser.add_argument("--conf", type=float, default=None, help="YOLO confidence threshold")
    parser.add_argument("--frame-skip", type=int, default=1,
                        help="process every Nth frame (1 = all). 2-3 speeds it up and adds a little latency.")
    parser.add_argument("--abandon-seconds", type=float, default=None)
    parser.add_argument("--ownership-dist", type=float, default=None,
                        help="person-to-bag owner-assignment distance (pixels)")
    parser.add_argument("--detach-dist", type=float, default=None,
                        help="owner-to-bag detach distance (pixels)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N videos (debug)")
    parser.add_argument("--show", action="store_true", help="show a window for each video")
    args = parser.parse_args()

    cfg = Config()
    if args.device:
        cfg.device = args.device
    if args.model:
        cfg.yolo_model = args.model
    if args.imgsz:
        cfg.imgsz = args.imgsz
    if args.conf is not None:
        cfg.conf_threshold = args.conf
    if args.abandon_seconds is not None:
        cfg.abandonment_seconds = args.abandon_seconds
    if args.ownership_dist is not None:
        cfg.ownership_distance_thresh = args.ownership_dist
    if args.detach_dist is not None:
        cfg.detach_distance_thresh = args.detach_dist

    gt_df = pd.read_csv(args.gt) if os.path.exists(args.gt) else pd.DataFrame(
        columns=["video", "abandonment_start_sec", "abandonment_end_sec"])

    video_files = sorted([p for p in Path(args.videos).iterdir()
                          if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")])
    if args.limit:
        video_files = video_files[:args.limit]

    print(f"[CONFIG] device={cfg.device} model={cfg.yolo_model} imgsz={cfg.imgsz} "
          f"conf={cfg.conf_threshold} frame_skip={args.frame_skip} "
          f"abandon={cfg.abandonment_seconds}s "
          f"own_dist={cfg.ownership_distance_thresh} "
          f"detach_dist={cfg.detach_distance_thresh}", flush=True)
    print(f"[CONFIG] {len(video_files)} videos in the job queue", flush=True)

    results = []
    for i, vp in enumerate(video_files, 1):
        print(f"\n[RUN {i}/{len(video_files)}] {vp.name}", flush=True)
        r = run_video(str(vp), cfg, headless=not args.show, frame_skip=args.frame_skip)
        if r:
            results.append(r)
            print(f"      DONE alarm={r['any_alarm']} "
                  f"first_alarm_sec={r['first_alarm_sec']} "
                  f"wall={r['wall_time_sec']}s speed={r['speed_fps']}fps", flush=True)

    rows, summary = evaluate(results, gt_df)
    pd.DataFrame(rows).to_csv(args.out, index=False)

    print("\n=== SUMMARY (Method 1) ===", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v}", flush=True)
    print(f"\nDetails: {args.out}", flush=True)


if __name__ == "__main__":
    main()
