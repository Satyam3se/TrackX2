import os
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trackx.settings')
django.setup()

from anpr_engine.vision_video import process_video_stream
from anpr_engine.models import CameraNode, CameraVideoFeed, DetectionLog
from django.core.files import File

def process_folder(folder_path, sample_rate=1):
    print(f"\n{'='*60}\nProcessing folder: {folder_path}\n{'='*60}")
    # Ensure a test camera exists
    camera, _ = CameraNode.objects.get_or_create(
        camera_id='TEST_CAM_1',
        defaults={'location_name': 'Test Video Camera', 'latitude': 0, 'longitude': 0}
    )

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith('.mp4'):
            continue
        video_path = os.path.join(folder_path, fname)
        print(f"\n--- Processing video: {fname} ({os.path.getsize(video_path)//1024} KB) ---")
        with open(video_path, 'rb') as f:
            feed = CameraVideoFeed.objects.create(
                camera=camera,
                title=f"Test {fname}",
                video_file=File(f, name=fname)
            )
        print(f"Created Feed ID: {feed.id}")
        summary = process_video_stream(feed.id, sample_rate=sample_rate)
        print("Summary:", summary)
        logs = DetectionLog.objects.filter(video_feed_id=feed.id).order_by('frame_timestamp')
        if logs:
            print("Detected plates:")
            for log in logs:
                print(f"[{log.frame_timestamp:.2f}s] {log.license_plate} (Conf: {log.confidence_score:.2f})")
        else:
            print("No plates detected.")
        # Cleanup feed to avoid DB clutter
        feed.delete()

if __name__ == '__main__':
    testing_dataset_dir = r"C:\\Users\\acer\\Documents\\track-x\\testing_dataset"
    if os.path.isdir(testing_dataset_dir):
        process_folder(testing_dataset_dir, sample_rate=1)
    else:
        print('testing_dataset directory not found')
    testingmodelvideos_dir = r"C:\\Users\\acer\\Documents\\track-x\\testingmodelvideos"
    if os.path.isdir(testingmodelvideos_dir):
        process_folder(testingmodelvideos_dir, sample_rate=1)
    else:
        print('testingmodelvideos directory not found')
