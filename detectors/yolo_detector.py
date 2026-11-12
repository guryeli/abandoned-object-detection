"""
YOLO-based detector. Returns person and bag classes in separate lists.
"""
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple


class Detection:
    """A single detection. Bbox in xyxy format."""
    __slots__ = ("bbox", "conf", "cls_id", "label")

    def __init__(self, bbox, conf, cls_id, label):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.conf = float(conf)
        self.cls_id = int(cls_id)
        self.label = label

    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def to_tlwh(self):
        x1, y1, x2, y2 = self.bbox
        return [x1, y1, x2 - x1, y2 - y1]


class YoloDetector:
    """
    Detects both people and bags with a single YOLO model.
    verbose=False stops it from printing a log line on every frame.
    """

    COCO_NAMES = {0: "person", 24: "backpack", 26: "handbag", 28: "suitcase"}

    def __init__(self, model_path: str, device: str, conf: float, iou: float,
                 person_id: int, bag_ids: List[int], imgsz: int = 640):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.person_id = person_id
        self.bag_ids = set(bag_ids)
        self.target_classes = list({person_id, *bag_ids})
        self.imgsz = imgsz

    def detect(self, frame) -> Tuple[List[Detection], List[Detection]]:
        """frame: BGR numpy array. Returns (persons, bags)."""
        results = self.model.predict(
            source=frame,
            classes=self.target_classes,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )[0]

        persons, bags = [], []
        if results.boxes is None or len(results.boxes) == 0:
            return persons, bags

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clses = results.boxes.cls.cpu().numpy().astype(int)

        for box, c, cls in zip(boxes, confs, clses):
            label = self.COCO_NAMES.get(cls, str(cls))
            det = Detection(box.tolist(), c, cls, label)
            if cls == self.person_id:
                persons.append(det)
            elif cls in self.bag_ids:
                bags.append(det)
        return persons, bags
