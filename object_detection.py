"""
CodeAlpha AI Internship — Task 4: Object Detection and Tracking
Author: CodeAlpha Intern
Description: Real-time object detection using YOLOv8 (ultralytics) + ByteTrack/SORT tracking.
             Supports webcam and video file input.

Usage:
    python object_detection.py                  # webcam
    python object_detection.py --source video.mp4
    python object_detection.py --source 0       # webcam device index
"""

import argparse
import sys
import time
import os

# ─────────────────────────── Import guards ───────────────────────────
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False
    print("❌ OpenCV not found. Install with:  pip install opencv-python")

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False
    print("❌ Ultralytics not found. Install with:  pip install ultralytics")

try:
    import numpy as np
    NP_OK = True
except ImportError:
    NP_OK = False


# ─────────────────────────── Simple SORT tracker ─────────────────────
class SimpleTracker:
    """
    Lightweight centroid-based tracker (no external deps).
    Assigns consistent IDs to objects across frames using IoU matching.
    """
    def __init__(self, max_lost: int = 30, iou_threshold: float = 0.3):
        self.next_id      = 1
        self.tracks: dict = {}   # {id: {"box": ..., "lost": int, "class": str}}
        self.max_lost     = max_lost
        self.iou_threshold = iou_threshold

    @staticmethod
    def _iou(b1, b2) -> float:
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    def update(self, detections: list[tuple]) -> list[tuple]:
        """
        Args:
            detections: [(x1,y1,x2,y2,conf,class_name), ...]
        Returns:
            tracked: [(x1,y1,x2,y2,track_id,class_name,conf), ...]
        """
        unmatched_dets = list(range(len(detections)))
        matched_ids    = set()

        for tid, track in list(self.tracks.items()):
            best_iou, best_det = 0, -1
            for di in unmatched_dets:
                iou = self._iou(track["box"], detections[di][:4])
                if iou > best_iou:
                    best_iou, best_det = iou, di
            if best_iou >= self.iou_threshold:
                self.tracks[tid]["box"]   = detections[best_det][:4]
                self.tracks[tid]["class"] = detections[best_det][5]
                self.tracks[tid]["conf"]  = detections[best_det][4]
                self.tracks[tid]["lost"]  = 0
                unmatched_dets.remove(best_det)
                matched_ids.add(tid)
            else:
                self.tracks[tid]["lost"] += 1

        # Remove old tracks
        self.tracks = {tid: t for tid, t in self.tracks.items()
                       if t["lost"] <= self.max_lost}

        # Create new tracks for unmatched detections
        for di in unmatched_dets:
            x1, y1, x2, y2, conf, cls = detections[di]
            self.tracks[self.next_id] = {
                "box": (x1, y1, x2, y2), "class": cls,
                "conf": conf, "lost": 0
            }
            self.next_id += 1

        return [
            (*t["box"], tid, t["class"], t["conf"])
            for tid, t in self.tracks.items() if t["lost"] == 0
        ]


# ─────────────────────────── Colour palette ──────────────────────────
PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10),  (146, 204, 23), (61, 219, 134),
    (26, 147, 52),  (0, 212, 187),  (44, 153, 168), (0, 194, 255),
    (52, 69, 147),  (100, 115, 255),(0, 24, 236),   (132, 56, 255),
    (82, 0, 133),   (203, 56, 255), (255, 149, 200),(255, 55, 199),
]

def get_color(track_id: int) -> tuple:
    return PALETTE[track_id % len(PALETTE)]


# ─────────────────────────── Drawing ─────────────────────────────────
def draw_track(frame, x1, y1, x2, y2, track_id, cls_name, conf, fps=None):
    color  = get_color(track_id)
    label  = f"#{track_id} {cls_name} {conf:.0%}"

    # Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(frame, n_tracks: int, fps: float):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 70), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, f"CodeAlpha — Object Tracker", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1)
    cv2.putText(frame, f"FPS: {fps:5.1f}   Objects: {n_tracks}", (8, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2)
    cv2.putText(frame, "Press Q to quit", (8, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)


# ─────────────────────────── Main loop ───────────────────────────────
def run(source, conf_threshold=0.4, model_name="yolov8n.pt"):
    if not CV2_OK or not YOLO_OK:
        print("❌ Required libraries not installed. See messages above.")
        sys.exit(1)

    print(f"📦 Loading YOLO model: {model_name} …")
    model   = YOLO(model_name)   # auto-downloads on first run
    tracker = SimpleTracker()

    # Open video source
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"❌ Cannot open source: {source}")
        sys.exit(1)

    print(f"✅ Streaming from: {source}")
    print("   Press Q in the window to quit.\n")

    prev_time = time.time()
    fps_smooth = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("📼 Stream ended.")
            break

        # ── YOLO inference ────────────────────────────────────────
        results = model(frame, verbose=False, conf=conf_threshold)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf  = float(box.conf[0])
            cls   = model.names[int(box.cls[0])]
            detections.append((x1, y1, x2, y2, conf, cls))

        # ── Tracking ──────────────────────────────────────────────
        tracked = tracker.update(detections)

        # ── Draw ──────────────────────────────────────────────────
        for item in tracked:
            x1, y1, x2, y2, tid, cls_name, conf = item
            draw_track(frame, x1, y1, x2, y2, tid, cls_name, conf)

        # FPS
        now = time.time()
        inst_fps   = 1.0 / (now - prev_time + 1e-9)
        fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps
        prev_time  = now

        draw_hud(frame, len(tracked), fps_smooth)
        cv2.imshow("CodeAlpha — Object Detection & Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Detection stopped.")


# ─────────────────────────── CLI ─────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CodeAlpha Task 4: Object Detection & Tracking")
    parser.add_argument("--source",    default="0",
                        help="Video source: 0/1 for webcam, path for video file")
    parser.add_argument("--conf",      type=float, default=0.4,
                        help="Detection confidence threshold (default 0.4)")
    parser.add_argument("--model",     default="yolov8n.pt",
                        choices=["yolov8n.pt","yolov8s.pt","yolov8m.pt","yolov8l.pt"],
                        help="YOLO model size (n=fastest, l=most accurate)")
    args = parser.parse_args()

    print("=" * 60)
    print("  CodeAlpha AI Internship — Task 4: Object Detection")
    print("=" * 60)
    print(f"  Source : {args.source}")
    print(f"  Model  : {args.model}")
    print(f"  Conf   : {args.conf}")
    print("=" * 60 + "\n")

    run(source=args.source, conf_threshold=args.conf, model_name=args.model)
