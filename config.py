"""
Central configuration. All thresholds and parameters are set here.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ---- Detection ----
    yolo_model: str = "yolov8n.pt"          # 'n' or 's' recommended for Apple Silicon
    device: str = "mps"                      # Apple Silicon: 'mps', NVIDIA: 'cuda', CPU: 'cpu'
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    imgsz: int = 640                         # YOLO inference image size; 320/480 is faster

    # COCO class IDs
    person_class_id: int = 0
    bag_class_ids: List[int] = field(default_factory=lambda: [24, 26, 28])  # backpack, handbag, suitcase

    # ---- Tracking ----
    tracker_max_age: int = 30                # frames; how long to keep a track after it is lost
    tracker_n_init: int = 3                  # min frames before a track counts as confirmed

    # ---- Ownership (Method 1: distance-based) ----
    ownership_init_frames: int = 15          # wait before assigning an owner when a bag first appears
    ownership_distance_thresh: float = 150.0 # pixels; the nearest person within this range becomes the owner
    detach_distance_thresh: float = 250.0    # pixels; the owner is detached once they pass this distance
    abandonment_seconds: float = 8.0         # time spent detached → alarm
    # ABODA reference: the literature uses 5-30s. 8s is ideal for a demo.

    # ---- Visualization ----
    show_window: bool = True
    save_video: bool = False
    output_path: str = "output.mp4"

    # ---- FPS / source ----
    target_fps: float = 30.0                 # reference for alarm timing
