"""Batch ANPR scan over local video files (no Django/DB required).

Reuses the exact production pipeline from ``anpr_engine.vision`` and
``anpr_engine.tracking`` (platevision detector chain -> fast-plate-OCR with
EasyOCR rescue -> per-track temporal voter) so the readings match what the
live service would log, but runs straight on files in the ``Videos`` folder.

Usage:
    python scripts/read_plates.py [--sample 3] [--video <path>]

Output:
    - stdout table of confirmed plates per video
    - annotated frames + plate crops under runs/plate_reads/<video>/<plate>/
"""

import argparse
import glob
import os
import sys
from collections import defaultdict

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anpr_engine.tracking import IoUTracker, TemporalVoter
from anpr_engine.vision import (
    _clean_plate_text,
    _detect_plate_boxes_many,
    _encode_png,
    _ocr_plate_image_preferred,
    _preprocess_plate_crop,
    get_fast_alpr,
    get_plate_detector,
)

CONFIDENCE_MIN_THRESHOLD = 0.45
CONFIDENCE_VOTE_READ_FLOOR = 0.55
STABLE_CONFIRM_READS = 3
VOTER_WINDOW = 8
VOTER_MIN_DWELL = 5
VOTER_TOP_K = 3
TRACKER_IOU_THRESHOLD = 0.3
TRACKER_MAX_AGE = 30
TRACKER_MAX_CENTER_DISTANCE = 1.5
TRACKER_VELOCITY_SMOOTHING = 0.5
TRACKER_MAX_SPEED_SCALE = 2.0

VIDEOS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Videos',
)
OUT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'runs', 'plate_reads',
)


def _detect_boxes(frame):
    boxes = _detect_plate_boxes_many(get_plate_detector(), frame)
    if boxes:
        return boxes
    detector = get_fast_alpr()
    if detector is None:
        return []
    try:
        detections = detector.predict(frame)
    except Exception as exc:
        print(f'[WARN] fast-alpr prediction failed: {exc}')
        return []
    if detections and isinstance(detections[0], (list, tuple)):
        detections = detections[0]
    h, w = frame.shape[:2]
    out = []
    for det in detections:
        try:
            float(getattr(det, 'confidence', None))
        except (AttributeError, TypeError):
            continue
        bb = det.bounding_box
        x1, y1 = max(0, int(bb.x1)), max(0, int(bb.y1))
        x2, y2 = min(w, int(bb.x2)), min(h, int(bb.y2))
        if x2 > x1 and y2 > y1:
            out.append([x1, y1, x2, y2])
    return out


def _mean(values):
    if not values:
        return 0.0
    total = 0.0
    for value in values:
        total += float(value)
    return total / len(values)


def _safe_name(path):
    base = os.path.splitext(os.path.basename(path))[0]
    return ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in base)


def process_video(video_path, sample_rate):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f'Could not open video: {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = IoUTracker(
        iou_threshold=TRACKER_IOU_THRESHOLD,
        max_age=TRACKER_MAX_AGE,
        max_center_distance=TRACKER_MAX_CENTER_DISTANCE,
        velocity_smoothing=TRACKER_VELOCITY_SMOOTHING,
        max_speed_scale=TRACKER_MAX_SPEED_SCALE,
    )
    voter = TemporalVoter(
        window=VOTER_WINDOW, min_dwell=VOTER_MIN_DWELL, top_k=VOTER_TOP_K,
    )
    track_plate = {}
    stable_confirm = {}
    confirmed_total = defaultdict(list)  # plate -> [(frame_idx, timestamp, conf)]

    out_dir = os.path.join(OUT_ROOT, _safe_name(video_path))
    os.makedirs(out_dir, exist_ok=True)

    frame_idx = -1
    processed = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % sample_rate != 0:
            continue
        processed += 1
        timestamp = round(frame_idx / fps, 2)

        boxes = _detect_boxes(frame)
        tracked = tracker.update(boxes)
        for stale in tracker.expired_track_ids:
            voter.forget(stale)
            track_plate.pop(stale, None)
            stable_confirm.pop(stale, None)

        for tid, box, _predicted in tracked:
            x1, y1, x2, y2 = box
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            processed_crop = _preprocess_plate_crop(crop)
            raw_text, _, char_conf = _ocr_plate_image_preferred(crop)
            if not raw_text:
                continue
            plate_text = _clean_plate_text(raw_text)
            conf = float(0.0 if char_conf is None else _mean(char_conf))
            read_chars = None
            if char_conf and len(char_conf) == len(plate_text):
                read_chars = [float(c) for c in char_conf]
            if conf < CONFIDENCE_VOTE_READ_FLOOR:
                continue

            confirmed = voter.observe(tid, plate_text, conf, read_chars)
            confirmed_text = None
            if confirmed is not None:
                cleaned = _clean_plate_text(confirmed)
                if 6 <= len(cleaned) <= 12:
                    confirmed_text = cleaned
            if confirmed_text is None:
                continue

            prev, streak = stable_confirm.get(tid, (None, 0))
            streak = streak + 1 if prev == confirmed_text else 1
            stable_confirm[tid] = (confirmed_text, streak)
            if streak < STABLE_CONFIRM_READS:
                continue
            if conf < CONFIDENCE_MIN_THRESHOLD:
                continue

            plate = confirmed_text
            track_plate[tid] = plate
            confirmed_total[plate].append((frame_idx, timestamp, conf))

            plate_dir = os.path.join(out_dir, plate)
            os.makedirs(plate_dir, exist_ok=True)
            crop_path = os.path.join(
                plate_dir, f'f{frame_idx:06d}_t{timestamp:07.2f}_c{conf:.2f}.png'
            )
            with open(crop_path, 'wb') as fh:
                fh.write(_encode_png(processed_crop))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{plate} {conf:.2f}'
            cv2.putText(
                frame, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

        if frame_idx % int(sample_rate * 10) == 0 and processed % 10 == 0:
            pct = int((frame_idx / max(total_frames, 1)) * 100)
            print(f'  ... {_safe_name(video_path)} {pct}%  '
                  f'({len(confirmed_total)} plate(s) so far)', flush=True)

    cap.release()
    return total_frames, processed, confirmed_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=3)
    ap.add_argument('--video', default=None, help='Single file to scan')
    args = ap.parse_args()

    videos = [args.video] if args.video else sorted(glob.glob(
        os.path.join(VIDEOS_DIR, '*'),
    ))
    videos = [v for v in videos if v.lower().endswith(
        ('.mp4', '.mov', '.avi', '.mkv', '.webm'),
    )]
    if not videos:
        print(f'No videos found in {VIDEOS_DIR}')
        return 1

    for path in videos:
        print(f'\n=== {os.path.basename(path)} ===')
        total, processed, results = process_video(path, args.sample)
        if not results:
            print('  No plates confirmed.')
            continue
        print(f'  frames={total} scanned={processed}')
        for plate in sorted(results, key=lambda p: -max(c for _, _, c in results[p])):
            reads = results[plate]
            best = max(c for _, _, c in reads)
            span = f'{reads[0][1]:.1f}s-{reads[-1][1]:.1f}s'
            print(f'  {plate:<12} reads={len(reads):<3} best={best:.2f} '
                  f'seen {span}')
    return 0


if __name__ == '__main__':
    sys.exit(main())