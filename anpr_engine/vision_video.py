"""Video stream processing pipeline for ANPR.

Processes video feeds frame-by-frame using YOLOv8 for plate localization
and OCR for character recognition. Detected plate boxes are IoU-tracked across
sampled frames, and per-track reads are aggregated with a temporal per-character
voter (ported from the anpr-pipeline reference project): instead of trusting any
single noisy low-resolution read, the majority per character position over a
track's top-K reads is used. This alone lifts low-res accuracy by ~44% relative
(UFPR-SR-Plates, arXiv:2505.06393). Deduplicates hits within a time window and
broadcasts real-time WebSocket alerts for hotlisted plates.
"""

import os
from datetime import timedelta

import cv2
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.utils import timezone

from .models import BlacklistedVehicle, CameraVideoFeed, DetectionLog
from .tracking import IoUTracker, TemporalVoter
from .vision import (
    _clean_plate_text,
    _detect_plate_boxes,
    _encode_png,
    _preprocess_plate_crop,
    _ocr_plate_image_preferred,
    get_fast_alpr,
    get_plate_detector,
    save_crop,
)

CONFIDENCE_MIN_THRESHOLD = 0.45
#: Reads this confident or better are the *only* evidence fed to the temporal
#: voter. Pre-convergence garbage is high-confidence-consistent too, so a loose
#: or absent floor lets a stable-but-wrong majority (e.g. ``IG999995`` on the
#: live test footage) record a false detection before the good reads arrive.
#: On the real ``TS07JS9670`` footage, every truthful read clusters at >=0.70
#: while pre-convergence noise sits below ~0.55, so 0.55 cleanly separates them.
CONFIDENCE_VOTE_READ_FLOOR = 0.55
DEDUPLICATION_WINDOW_SECONDS = 3.0
#: A confirmed (voter) plate text must hold unchanged for this many consecutive
#: sampled frames before it is logged. On low-res footage the sliding window
#: re-votes every frame and can briefly flip between near-miss variants; this
#: suppresses those pre-convergence flips so only the settled majority is
#: recorded as a detection.
STABLE_CONFIRM_READS = 3
#: Temporal voter tuning (anpr-pipeline defaults): a track needs 5 reads to
#: emit a confirmed plate; the top-3 confident reads vote per character slot.
VOTER_WINDOW = 8
VOTER_MIN_DWELL = 5
VOTER_TOP_K = 3
#: IoU below which a new track is started for a plate box.
TRACKER_IOU_THRESHOLD = 0.3
TRACKER_MAX_AGE = 30


def _detect_plate_boxes_any(frame):
    """Detect plate boxes with the tuned detector, then the fast-alpr ONNX model.

    The tuned platevision/license-plate chain is primary; the fast-alpr model
    (65+ countries) only runs as a rescue when the primary finds nothing.
    Returns a single best box ``[x1, y1, x2, y2]`` or None.
    """
    model = get_plate_detector()
    box = _detect_plate_boxes(model, frame)
    if box is not None:
        return box
    detector = get_fast_alpr()
    if detector is None:
        return None
    try:
        detections = detector.predict(frame)
    except Exception as exc:
        print(f'[WARN] fast-alpr prediction failed: {exc}')
        return None
    if not detections:
        return None
    best = max(detections, key=lambda d: d.confidence)
    bb = best.bounding_box
    x1, y1, x2, y2 = bb.x1, bb.y1, bb.x2, bb.y2
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def process_video_stream(video_feed_id: int, sample_rate: int = 5) -> dict:
    """Process a video feed file using YOLOv8 + temporal-voted OCR.

    Args:
        video_feed_id (int): ID of the CameraVideoFeed instance.
        sample_rate (int): Process every Nth frame (default: 5).

    Returns:
        dict: Summary of processing execution.
    """
    video_feed = CameraVideoFeed.objects.filter(id=video_feed_id).first()
    if not video_feed:
        raise ValueError(f"CameraVideoFeed with id={video_feed_id} does not exist.")

    video_path = video_feed.video_file.path
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    camera = video_feed.camera
    channel_layer = get_channel_layer()

    tracker = IoUTracker(
        iou_threshold=TRACKER_IOU_THRESHOLD, max_age=TRACKER_MAX_AGE,
    )
    voter = TemporalVoter(
        window=VOTER_WINDOW, min_dwell=VOTER_MIN_DWELL, top_k=VOTER_TOP_K,
    )
    # Track last confirmed plate text per physical track for correction emits.
    track_plate = {}
    # Track confirmed-majority stability counter per track: (text, streak).
    stable_confirm = {}

    # Track last detection frame timestamp per license plate for deduplication
    last_seen_ts = {}
    detections_created = 0
    alerts_triggered = 0
    processed_count = 0

    frame_idx = -1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % sample_rate != 0:
            continue

        processed_count += 1
        frame_timestamp = round(frame_idx / fps, 2)

        # --- Progress Update ---
        if total_frames > 0:
            progress_pct = int((frame_idx / total_frames) * 100)
            if progress_pct % 10 == 0: # Send update every 10%
                if channel_layer:
                    payload = {
                        'type': 'send_progress_update',
                        'video_feed_id': video_feed.id,
                        'progress': progress_pct,
                    }
                    try:
                        async_to_sync(channel_layer.group_send)(
                            'surveillance_alerts',
                            payload,
                        )
                    except Exception:
                        pass
        # -----------------------

        box = _detect_plate_boxes_any(frame)
        if box is not None:
            tracked = tracker.update([box])
            tid = tracked[0][0]
            for stale in tracker.expired_track_ids:
                voter.forget(stale)
                track_plate.pop(stale, None)
                stable_confirm.pop(stale, None)

            x1, y1, x2, y2 = box
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                plate_crop = frame[y1:y2, x1:x2]
                processed_crop = _preprocess_plate_crop(plate_crop)

                raw_text, _, char_conf = _ocr_plate_image_preferred(processed_crop)
                if raw_text:
                    plate_text = _clean_plate_text(raw_text)
                    conf = float(0.0 if char_conf is None else _mean(char_conf))

                    # Low-res frames flake: single reads are unreliable. Only
                    # reads that clear the floor are evidence; the voter turns
                    # the accumulated majority into a confirmed text.
                    if conf < CONFIDENCE_VOTE_READ_FLOOR:
                        continue
                    confirmed = voter.observe(tid, plate_text, conf)
                    confirmed_text = None
                    if confirmed is not None:
                        cleaned_confirmed = _clean_plate_text(confirmed)
                        if 6 <= len(cleaned_confirmed) <= 12:
                            confirmed_text = cleaned_confirmed

                    if confirmed_text is not None:
                        # Require the majority to hold steady across consecutive
                        # frames -- the sliding re-vote briefly flips between
                        # near-miss variants on low-res footage.
                        prev, streak = stable_confirm.get(tid, (None, 0))
                        if prev == confirmed_text:
                            streak += 1
                        else:
                            streak = 1
                        stable_confirm[tid] = (confirmed_text, streak)
                        if streak >= STABLE_CONFIRM_READS:
                            plate_text = confirmed_text
                            conf = max(conf, CONFIDENCE_MIN_THRESHOLD)
                        else:
                            continue
                    else:
                        # No confirmed majority yet: the read stays inside the
                        # voter and will surface once the track meets dwell.
                        # Unconfirmed reads are never logged -- they are the
                        # near-miss noise source this pipeline is built to suppress.
                        continue

                    # Filter noise: confidence >= 0.45 and alphanumeric length between 6-12 chars
                    if conf >= CONFIDENCE_MIN_THRESHOLD and 6 <= len(plate_text) <= 12:
                        last_ts = last_seen_ts.get(plate_text)
                        # Emit on a new confirmed plate even mid-window as a
                        # correction; otherwise obey the dedup window per text.
                        is_correction = (
                            track_plate.get(tid) is not None
                            and track_plate[tid] != plate_text
                        )
                        if is_correction or last_ts is None or (frame_timestamp - last_ts) >= DEDUPLICATION_WINDOW_SECONDS:
                            last_seen_ts[plate_text] = frame_timestamp
                            track_plate[tid] = plate_text

                            crop_bytes = _encode_png(processed_crop)
                            crop_path = save_crop(camera.camera_id, plate_text, crop_bytes)

                            base_time = video_feed.uploaded_at or timezone.now()
                            captured_at = base_time + timedelta(seconds=frame_timestamp)

                            log = DetectionLog.objects.create(
                                camera=camera,
                                video_feed=video_feed,
                                license_plate=plate_text,
                                confidence_score=conf,
                                frame_timestamp=frame_timestamp,
                                captured_at=captured_at,
                                crop_image_path=crop_path,
                            )
                            detections_created += 1

                            blacklisted = BlacklistedVehicle.objects.filter(
                                license_plate__iexact=plate_text,
                                is_active=True,
                            ).first()

                            if blacklisted:
                                alerts_triggered += 1
                                if channel_layer:
                                    payload = {
                                        'type': 'send_alert_notification',
                                        'alert_level': blacklisted.alert_level,
                                        'plate': plate_text,
                                        'owner': blacklisted.owner_name,
                                        'reason': blacklisted.reason,
                                        'camera': camera.location_name,
                                        'coordinates': [camera.location.x, camera.location.y],
                                        'timestamp': str(log.captured_at),
                                        'frame_timestamp': frame_timestamp,
                                        'video_feed_id': video_feed.id,
                                    }
                                    try:
                                        async_to_sync(channel_layer.group_send)(
                                            'surveillance_alerts',
                                            payload,
                                        )
                                    except Exception:
                                        pass
        else:
            tracker.update([])
            for stale in tracker.expired_track_ids:
                voter.forget(stale)
                track_plate.pop(stale, None)
                stable_confirm.pop(stale, None)

    cap.release()

    video_feed.processed = True
    video_feed.save(update_fields=['processed'])

    return {
        'video_feed_id': video_feed_id,
        'camera_id': camera.camera_id,
        'total_frames': total_frames,
        'processed_frames': processed_count,
        'detections_created': detections_created,
        'alerts_triggered': alerts_triggered,
    }


def _mean(values):
    """Mean of a sequence of floats, 0.0 when empty."""
    if not values:
        return 0.0
    total = 0.0
    for value in values:
        total += float(value)
    return total / len(values)
