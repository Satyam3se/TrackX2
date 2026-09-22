import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trackx.settings')
django.setup()

from anpr_engine.vision_video import process_video_stream
from anpr_engine.models import CameraNode, CameraVideoFeed, DetectionLog
from django.core.files import File

def test_video():
    video_path = 'Videos/WhatsApp Video 2026-09-12 at 21.55.08.mp4'
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)

    print("="*60)
    print(f"Testing YOLO ANPR Pipeline on {video_path}")
    print("="*60)

    # 1. Create a mock CameraNode if none exists
    camera, _ = CameraNode.objects.get_or_create(
        camera_id='TEST_CAM_1',
        defaults={'location_name': 'Test Video Camera', 'latitude': 0, 'longitude': 0}
    )

    # 2. Upload the video to a CameraVideoFeed
    with open(video_path, 'rb') as f:
        feed = CameraVideoFeed.objects.create(
            camera=camera,
            title='WhatsApp Video Test',
            video_file=File(f, name=os.path.basename(video_path))
        )

    print(f"Created Feed ID: {feed.id}")
    print("Starting processing... (this may take a few minutes)")

    # 3. Process the feed
    summary = process_video_stream(feed.id, sample_rate=1)
    
    print("\n--- SUMMARY ---")
    print(f"Detections Created: {summary['detections_created']}")
    print(f"Alerts Triggered: {summary['alerts_triggered']}")
    
    print("\n--- DETECTED PLATES ---")
    logs = DetectionLog.objects.filter(video_feed_id=feed.id).order_by('frame_timestamp')
    for log in logs:
        print(f"[{log.frame_timestamp:.2f}s] {log.license_plate} (Conf: {log.confidence_score:.2f})")

if __name__ == '__main__':
    test_video()
