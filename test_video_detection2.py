import cv2
import sys
import os

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from anpr_engine.vision import extract_license_plate

video_path = r"c:/Users/acer/Documents/track-x/Videos/WhatsApp Video 2026-09-12 at 21.55.08.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print('Failed to open video')
    sys.exit(1)
ret, frame = cap.read()
if not ret:
    print('Failed to read frame')
    sys.exit(1)
# Encode frame to JPEG bytes
_, buffer = cv2.imencode('.jpg', frame)
image_bytes = buffer.tobytes()
result = extract_license_plate(image_bytes)
print('Detection result:', result)
cap.release()
