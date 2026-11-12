"""
Ownership and abandonment state machine.

State machine for bag tracks:
  UNASSIGNED → OWNED → DETACHED → ABANDONED

- UNASSIGNED: the bag just appeared; no owner assigned yet
- OWNED: the bag is assigned to a person, and that person is nearby
- DETACHED: an owner was assigned but they moved away (no alarm yet)
- ABANDONED: the owner has stayed away longer than the set time → ALARM
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import math
import time


class BagState(Enum):
    UNASSIGNED = "unassigned"
    OWNED = "owned"
    DETACHED = "detached"
    ABANDONED = "abandoned"


@dataclass
class BagRecord:
    """Full state of one bag track."""
    bag_id: int
    first_seen_frame: int
    state: BagState = BagState.UNASSIGNED
    owner_id: Optional[int] = None           # person track id
    detached_since_frame: Optional[int] = None
    last_bbox: Optional[list] = None
    last_center: Optional[Tuple[float, float]] = None
    frames_seen: int = 0
    distance_history: list = field(default_factory=list)


def euclid(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class OwnershipManager:
    """
    Distance-based ownership assignment and detachment detection.
    Runs per frame and takes tracked bag and person lists from the caller.
    """

    def __init__(self, init_frames: int, own_dist: float, detach_dist: float,
                 abandon_seconds: float, target_fps: float):
        self.init_frames = init_frames
        self.own_dist = own_dist
        self.detach_dist = detach_dist
        self.abandon_frames = int(abandon_seconds * target_fps)
        self.bags: Dict[int, BagRecord] = {}
        self.frame_idx = 0

    def step(self, frame_idx: int, person_tracks, bag_tracks):
        """
        person_tracks / bag_tracks: list of dict
            {'id': int, 'bbox': [x1,y1,x2,y2], 'center': (cx,cy)}
        """
        self.frame_idx = frame_idx
        active_bag_ids = set()

        for bag in bag_tracks:
            bid = bag["id"]
            active_bag_ids.add(bid)

            if bid not in self.bags:
                self.bags[bid] = BagRecord(
                    bag_id=bid,
                    first_seen_frame=frame_idx,
                )

            rec = self.bags[bid]
            rec.frames_seen += 1
            rec.last_bbox = bag["bbox"]
            rec.last_center = bag["center"]

            # Find the nearest person
            nearest_pid, nearest_dist = self._nearest_person(bag["center"], person_tracks)

            # State machine
            if rec.state == BagState.UNASSIGNED:
                # Track the nearest person throughout the init period
                if rec.frames_seen >= self.init_frames:
                    if nearest_pid is not None and nearest_dist <= self.own_dist:
                        rec.owner_id = nearest_pid
                        rec.state = BagState.OWNED
                    # If nobody is present during init, treat the item as unowned (do not assign).
                    # If someone approaches later, it is still set to OWNED.
                else:
                    # Early owner assignment — we can wait so the decision is safer
                    pass

                # Even without that early assignment, assign if someone stays nearby
                if rec.state == BagState.UNASSIGNED and nearest_pid is not None and nearest_dist <= self.own_dist * 0.7:
                    rec.owner_id = nearest_pid
                    rec.state = BagState.OWNED

            elif rec.state == BagState.OWNED:
                owner_dist = self._dist_to_person(bag["center"], rec.owner_id, person_tracks)
                if owner_dist is None:
                    # Owner left the frame / the track was lost
                    rec.state = BagState.DETACHED
                    rec.detached_since_frame = frame_idx
                elif owner_dist > self.detach_dist:
                    rec.state = BagState.DETACHED
                    rec.detached_since_frame = frame_idx

            elif rec.state == BagState.DETACHED:
                owner_dist = self._dist_to_person(bag["center"], rec.owner_id, person_tracks)
                if owner_dist is not None and owner_dist <= self.own_dist:
                    # Owner came back
                    rec.state = BagState.OWNED
                    rec.detached_since_frame = None
                else:
                    elapsed = frame_idx - (rec.detached_since_frame or frame_idx)
                    if elapsed >= self.abandon_frames:
                        rec.state = BagState.ABANDONED

            elif rec.state == BagState.ABANDONED:
                # If the owner still comes back, the warning is cleared (optional)
                owner_dist = self._dist_to_person(bag["center"], rec.owner_id, person_tracks)
                if owner_dist is not None and owner_dist <= self.own_dist:
                    rec.state = BagState.OWNED
                    rec.detached_since_frame = None

        return self.bags

    def _nearest_person(self, point, person_tracks):
        best_id, best_d = None, float("inf")
        for p in person_tracks:
            d = euclid(point, p["center"])
            if d < best_d:
                best_d = d
                best_id = p["id"]
        if best_id is None:
            return None, None
        return best_id, best_d

    def _dist_to_person(self, point, pid, person_tracks):
        for p in person_tracks:
            if p["id"] == pid:
                return euclid(point, p["center"])
        return None
