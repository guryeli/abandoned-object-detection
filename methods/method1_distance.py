"""
Method 1: YOLO + DeepSORT + Distance-Based Ownership

This class takes frames from a live source or a video file,
runs detection + tracking + the ownership state machine,
and returns (annotated_frame, bag_records) for each frame.
"""
import time
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

from detectors.yolo_detector import YoloDetector
from core.ownership import OwnershipManager
from core.visualizer import draw_overlay


class Method1Pipeline:
    def __init__(self, config):
        self.cfg = config
        self.detector = YoloDetector(
            model_path=config.yolo_model,
            device=config.device,
            conf=config.conf_threshold,
            iou=config.iou_threshold,
            person_id=config.person_class_id,
            bag_ids=config.bag_class_ids,
            imgsz=getattr(config, "imgsz", 640),
        )
        # Two separate trackers, one for people and one for bags. This prevents ID collisions.
        self.person_tracker = DeepSort(
            max_age=config.tracker_max_age,
            n_init=config.tracker_n_init,
            embedder="mobilenet",  # lightweight; runs on MPS/CPU
        )
        self.bag_tracker = DeepSort(
            max_age=config.tracker_max_age * 2,  # keep a bag longer once it is set down
            n_init=2,
            embedder="mobilenet",
        )
        self.ownership = OwnershipManager(
            init_frames=config.ownership_init_frames,
            own_dist=config.ownership_distance_thresh,
            detach_dist=config.detach_distance_thresh,
            abandon_seconds=config.abandonment_seconds,
            target_fps=config.target_fps,
        )
        self.frame_idx = 0

    def _to_deepsort_format(self, dets):
        """DeepSORT expects the [ [x, y, w, h], conf, class ] format."""
        out = []
        for d in dets:
            out.append((d.to_tlwh(), d.conf, str(d.cls_id)))
        return out

    @staticmethod
    def _track_to_dict(t):
        ltrb = t.to_ltrb()
        x1, y1, x2, y2 = ltrb
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return {"id": int(t.track_id), "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "center": (float(cx), float(cy))}

    def process_frame(self, frame):
        self.frame_idx += 1
        t0 = time.time()

        persons, bags = self.detector.detect(frame)

        person_dets = self._to_deepsort_format(persons)
        bag_dets = self._to_deepsort_format(bags)

        person_tracks_raw = self.person_tracker.update_tracks(person_dets, frame=frame)
        bag_tracks_raw = self.bag_tracker.update_tracks(bag_dets, frame=frame)

        person_tracks = [self._track_to_dict(t) for t in person_tracks_raw if t.is_confirmed()]
        bag_tracks = [self._track_to_dict(t) for t in bag_tracks_raw if t.is_confirmed()]

        bag_records = self.ownership.step(self.frame_idx, person_tracks, bag_tracks)

        dt = time.time() - t0
        fps = 1.0 / dt if dt > 0 else 0.0
        annotated = draw_overlay(
            frame.copy(), person_tracks, bag_tracks, bag_records,
            self.frame_idx, fps_text=f"FPS: {fps:.1f}"
        )

        return annotated, bag_records, person_tracks, bag_tracks
