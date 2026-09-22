import json

import base64

from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer

from .tracking import IoUTracker, TemporalVoter
from .vision import extract_license_plate
from .vision_video import (
    CONFIDENCE_VOTE_READ_FLOOR,
    TRACKER_IOU_THRESHOLD,
    TRACKER_MAX_AGE,
    VOTER_MIN_DWELL,
    VOTER_TOP_K,
    VOTER_WINDOW,
)


class LiveVideoConsumer(AsyncWebsocketConsumer):
    """
    Receives a frame from the frontend, runs ANPR on it, and immediately
    returns the results.

    Each connection keeps its own IoU tracker + temporal voter (ported from the
    anpr-pipeline reference project). The browser pushes every frame, and
    individual reads of a low-res plate flake frame-to-frame; grouping reads by
    their tracked physical plate and majority-voting them turns the noisy
    stream into a stable "confirmed" plate, sent alongside the raw read.
    """

    async def connect(self):
        await self.accept()
        self._tracker = IoUTracker(
            iou_threshold=TRACKER_IOU_THRESHOLD, max_age=TRACKER_MAX_AGE,
        )
        self._voter = TemporalVoter(
            window=VOTER_WINDOW, min_dwell=VOTER_MIN_DWELL, top_k=VOTER_TOP_K,
        )

    async def disconnect(self, close_code):
        self._tracker = None
        self._voter = None

    def _aggregate(self, detection, scale=1):
        """Feed one detection into the per-connection tracker/voter.

        Returns the JSON payload: the raw per-frame read plus a ``confirmed``
        plate once the temporal voter settles (dwell met on the tracked plate).
        """
        bbox = detection.get('bbox')
        plate_text = detection.get('plate_text', '')
        confidence = detection.get('confidence', 0.0)
        car_bbox = detection.get('car_bbox')
        all_cars = detection.get('all_cars', [])

        confirmed = None
        if bbox:
            tracked = self._tracker.update([bbox])
            for stale in self._tracker.expired_track_ids:
                self._voter.forget(stale)
            if tracked:
                track_id = tracked[0][0]
                if plate_text and confidence >= CONFIDENCE_VOTE_READ_FLOOR:
                    confirmed = self._voter.observe(
                        track_id, plate_text, float(confidence),
                    )
        else:
            self._tracker.update([])
            for stale in self._tracker.expired_track_ids:
                self._voter.forget(stale)

        response = {
            'bbox': bbox,
            'car_bbox': car_bbox,
            'all_cars': all_cars,
            'plate_text': plate_text,
            'confidence': confidence,
            'scale': scale,
        }
        if confirmed:
            response['confirmed'] = confirmed
        return response

    async def receive(self, text_data=None, bytes_data=None):
        if text_data:
            try:
                data = json.loads(text_data)
                frame_data_url = data.get('frame')
                scale = data.get('scale', 1)

                if frame_data_url and frame_data_url.startswith('data:image'):
                    # Extract the base64 part
                    _, base64_data = frame_data_url.split(',', 1)
                    image_bytes = base64.b64decode(base64_data)

                    # Run the heavy vision extraction synchronously in a thread pool
                    detection = await sync_to_async(extract_license_plate)(image_bytes)

                    # We don't need to send the cropped image back to save bandwidth
                    await self.send(
                        text_data=json.dumps(self._aggregate(detection, scale))
                    )
            except Exception as e:
                await self.send(text_data=json.dumps({'error': str(e)}))

        elif bytes_data:
            # Handle direct binary frame uploads if sent that way
            try:
                detection = await sync_to_async(extract_license_plate)(bytes_data)
                await self.send(
                    text_data=json.dumps(self._aggregate(detection))
                )
            except Exception as e:
                await self.send(text_data=json.dumps({'error': str(e)}))