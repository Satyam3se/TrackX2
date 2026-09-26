import base64
import json
import logging

from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer

from .tracking import IoUTracker, TemporalVoter
from .vision import extract_license_plates
from .vision_video import (
    CONFIDENCE_VOTE_READ_FLOOR,
    TRACKER_IOU_THRESHOLD,
    TRACKER_MAX_AGE,
    TRACKER_MAX_CENTER_DISTANCE,
    TRACKER_MAX_SPEED_SCALE,
    TRACKER_VELOCITY_SMOOTHING,
    VOTER_MIN_DWELL,
    VOTER_TOP_K,
    VOTER_WINDOW,
)

logger = logging.getLogger(__name__)


class LiveVideoConsumer(AsyncWebsocketConsumer):
    channel_layer_alias = None

    async def connect(self):
        logger.info('LiveVideoConsumer.connect')
        await self.accept()
        self._session_id = None
        self._reset_tracking()

    def _reset_tracking(self):
        self._tracker = IoUTracker(
            iou_threshold=TRACKER_IOU_THRESHOLD,
            max_age=TRACKER_MAX_AGE,
            max_center_distance=TRACKER_MAX_CENTER_DISTANCE,
            velocity_smoothing=TRACKER_VELOCITY_SMOOTHING,
            max_speed_scale=TRACKER_MAX_SPEED_SCALE,
        )
        self._voter = TemporalVoter(
            window=VOTER_WINDOW,
            min_dwell=VOTER_MIN_DWELL,
            top_k=VOTER_TOP_K,
        )

    async def disconnect(self, close_code):
        logger.info('LiveVideoConsumer.disconnect code=%r', close_code)
        self._tracker = None
        self._voter = None
        self._session_id = None

    def _aggregate(self, detections, scale=1):
        if isinstance(detections, dict):
            detections = [detections]
        located = [detection for detection in (detections or []) if detection.get('bbox')]
        boxes = [detection['bbox'] for detection in located]
        tracked = self._tracker.update(boxes)
        for track_id in self._tracker.expired_track_ids:
            self._voter.forget(track_id)

        detections_by_box = {}
        for detection in located:
            box_key = tuple(detection['bbox'])
            detections_by_box.setdefault(box_key, []).append(detection)

        results = []
        for track_id, tracked_box, _predicted in tracked:
            box_key = tuple(tracked_box)
            candidates = detections_by_box.get(box_key)
            if not candidates:
                continue
            detection = candidates.pop(0)
            plate_text = detection.get('plate_text', '')
            confidence = float(detection.get('confidence') or 0.0)
            char_confidences = detection.get('char_confidences')
            confirmed = self._voter.confirmed_for(track_id)
            if plate_text and confidence >= CONFIDENCE_VOTE_READ_FLOOR:
                read_confidences = None
                if (
                    char_confidences
                    and len(char_confidences) == len(plate_text)
                ):
                    read_confidences = [float(value) for value in char_confidences]
                confirmed = self._voter.observe(
                    track_id,
                    plate_text,
                    confidence,
                    read_confidences,
                )
            result = {
                'track_id': track_id,
                'bbox': detection.get('bbox'),
                'vehicle_bbox': detection.get('vehicle_bbox'),
                'plate_text': plate_text,
                'confidence': confidence,
            }
            if confirmed:
                result['confirmed'] = confirmed
            results.append(result)

        primary = results[0] if results else {}
        return {
            'bbox': primary.get('bbox'),
            'vehicle_bbox': primary.get('vehicle_bbox'),
            'plate_text': primary.get('plate_text', ''),
            'confidence': primary.get('confidence', 0.0),
            'confirmed': primary.get('confirmed'),
            'scale': scale,
            'detections': results,
        }

    async def _process(
        self,
        image_bytes,
        scale,
        session_id=None,
        frame_id=None,
        media_time=None,
    ):
        detections = await sync_to_async(extract_license_plates)(image_bytes)
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            scale = 1.0
        if scale <= 0:
            scale = 1.0
        response = self._aggregate(detections, scale)
        if session_id is not None:
            response['session_id'] = session_id
        if frame_id is not None:
            response['frame_id'] = frame_id
        if media_time is not None:
            response['media_time'] = media_time
        await self.send(text_data=json.dumps(response))

    async def receive(self, text_data=None, bytes_data=None):
        response_context = {}
        try:
            if text_data:
                data = json.loads(text_data)
                session_id = data.get('session_id')
                frame_id = data.get('frame_id')
                media_time = data.get('media_time')
                response_context = {
                    key: value
                    for key, value in {
                        'session_id': session_id,
                        'frame_id': frame_id,
                        'media_time': media_time,
                    }.items()
                    if value is not None
                }
                if session_id and session_id != self._session_id:
                    self._session_id = session_id
                    self._reset_tracking()
                frame_data_url = data.get('frame')
                if frame_data_url and frame_data_url.startswith('data:image'):
                    _, base64_data = frame_data_url.split(',', 1)
                    image_bytes = base64.b64decode(base64_data)
                    await self._process(
                        image_bytes,
                        data.get('scale', 1),
                        session_id,
                        frame_id,
                        media_time,
                    )
            elif bytes_data:
                await self._process(bytes_data, 1)
        except Exception as exc:
            logger.exception('LiveVideoConsumer frame processing failed')
            try:
                await self.send(text_data=json.dumps({
                    'error': str(exc),
                    **response_context,
                }))
            except Exception:
                logger.exception('LiveVideoConsumer error response failed')
