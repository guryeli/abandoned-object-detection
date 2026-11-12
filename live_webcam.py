"""
Runs Method 1 from a live webcam.

Usage:
    python live_webcam.py
    python live_webcam.py --source 0          # webcam
    python live_webcam.py --source video.mp4  # video file
    python live_webcam.py --device cpu        # CPU instead of MPS
"""
import argparse
import cv2
import sys

from config import Config
from methods.method1_distance import Method1Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0",
                        help="0 = webcam, or a path to a video file")
    parser.add_argument("--device", default=None, help="mps / cuda / cpu")
    parser.add_argument("--model", default=None, help="yolov8n.pt, yolov8s.pt, etc.")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO inference size")
    parser.add_argument("--conf", type=float, default=None, help="YOLO confidence threshold")
    parser.add_argument("--abandon-seconds", type=float, default=None)
    parser.add_argument("--ownership-dist", type=float, default=None,
                        help="person-to-bag owner-assignment distance (px). Try 250-350 on a webcam.")
    parser.add_argument("--detach-dist", type=float, default=None,
                        help="owner-to-bag detach distance (px). Try 350-500 on a webcam.")
    parser.add_argument("--save", action="store_true", help="save the output as output.mp4")
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

    src = int(args.source) if args.source.isdigit() else args.source
    if isinstance(src, int) and sys.platform == "darwin":
        cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {src}", file=sys.stderr)
        if isinstance(src, int):
            print("       On macOS, under System Settings → Privacy → Camera, "
                  "make sure Terminal/the IDE is allowed.", file=sys.stderr)
        sys.exit(1)

    # Take the real FPS from the source FPS
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if src_fps > 0:
        cfg.target_fps = src_fps

    print(f"[INFO] Source: {src}  FPS={cfg.target_fps:.1f}  Device={cfg.device}  Model={cfg.yolo_model}")
    print(f"[INFO] imgsz={cfg.imgsz}  conf={cfg.conf_threshold}  "
          f"own_dist={cfg.ownership_distance_thresh}px  detach_dist={cfg.detach_distance_thresh}px")
    print(f"[INFO] Abandonment threshold: {cfg.abandonment_seconds}s "
          f"({int(cfg.abandonment_seconds * cfg.target_fps)} frame)")
    print("[INFO] Exit: q or ESC")

    pipeline = Method1Pipeline(cfg)

    writer = None
    if args.save:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(cfg.output_path, fourcc, cfg.target_fps or 25.0, (w, h))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, bag_records, _, _ = pipeline.process_frame(frame)

            if writer is not None:
                writer.write(annotated)

            cv2.imshow("Abandoned Object Detection - Method 1", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
