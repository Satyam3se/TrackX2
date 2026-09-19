import json
from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
import base64

from .vision import extract_license_plate

class LiveVideoConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data=None, bytes_data=None):
        """
        Receives a frame from the frontend, runs ANPR on it,
        and immediately returns the results.
        """
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
                    response = {
                        'bbox': detection.get('bbox'),
                        'plate_text': detection.get('plate_text'),
                        'confidence': detection.get('confidence'),
                        'scale': scale,
                    }
                    
                    await self.send(text_data=json.dumps(response))
            except Exception as e:
                await self.send(text_data=json.dumps({'error': str(e)}))

        elif bytes_data:
            # Handle direct binary frame uploads if sent that way
            try:
                detection = await sync_to_async(extract_license_plate)(bytes_data)
                response = {
                    'bbox': detection.get('bbox'),
                    'plate_text': detection.get('plate_text'),
                    'confidence': detection.get('confidence'),
                }
                await self.send(text_data=json.dumps(response))
            except Exception as e:
                await self.send(text_data=json.dumps({'error': str(e)}))
