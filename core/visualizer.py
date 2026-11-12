"""
Draws the bbox, track ID, and alarm overlay on the frame.
"""
import cv2
from core.ownership import BagState


COLORS = {
    BagState.UNASSIGNED: (160, 160, 160),    # gray
    BagState.OWNED:      (0, 200, 0),        # green
    BagState.DETACHED:   (0, 200, 255),      # orange
    BagState.ABANDONED:  (0, 0, 255),        # red (BGR)
}

PERSON_COLOR = (255, 200, 0)


def draw_overlay(frame, person_tracks, bag_tracks, bag_records, frame_idx, fps_text=""):
    h, w = frame.shape[:2]

    # Persons
    for p in person_tracks:
        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), PERSON_COLOR, 2)
        cv2.putText(frame, f"P#{p['id']}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, PERSON_COLOR, 2)

    # Bags
    alarm_count = 0
    for b in bag_tracks:
        bid = b["id"]
        rec = bag_records.get(bid)
        state = rec.state if rec else BagState.UNASSIGNED
        color = COLORS[state]
        x1, y1, x2, y2 = [int(v) for v in b["bbox"]]
        thickness = 3 if state == BagState.ABANDONED else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        owner_str = f" owner=P#{rec.owner_id}" if rec and rec.owner_id is not None else ""
        label = f"B#{bid} [{state.value}]{owner_str}"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if state == BagState.ABANDONED:
            alarm_count += 1

    # HUD
    bar = f"Frame {frame_idx}  {fps_text}"
    cv2.putText(frame, bar, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if alarm_count > 0:
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 200), -1)
        cv2.putText(frame, f"!! ABANDONED OBJECT DETECTED ({alarm_count}) !!",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return frame
