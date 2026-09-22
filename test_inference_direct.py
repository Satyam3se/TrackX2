import os
import cv2
import sys

# Optional: Add project root to sys.path so we can import anpr_engine safely
sys.path.append(os.path.abspath('.'))

from anpr_engine.vision import extract_license_plate, get_plate_detector

def test_inference_direct():
    video_path = 'Videos/WhatsApp Video 2026-09-12 at 21.55.08.mp4'
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    print("="*60)
    print(f"Testing YOLO ANPR Inference DIRECTLY on {video_path}")
    print("="*60)

    # Force model load
    detector = get_plate_detector()
    if detector:
        print(f"Loaded YOLO weights: {getattr(detector.ckpt, 'filename', 'unknown')} or {detector.model.pt_path}")
    else:
        print("No detector loaded.")

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    detections = []

    print("Starting processing... (Scanning 1 frame per second to save time)")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 30
        
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process 1 frame per second approximately
        if frame_count % int(fps) == 0:
            sys.stdout.write(f"\rProcessing frame {frame_count}...")
            sys.stdout.flush()
            
            # Downscale frame for speed like the frontend does
            h, w = frame.shape[:2]
            MAX_WIDTH = 640
            scale = MAX_WIDTH / w if w > MAX_WIDTH else 1.0
            if scale != 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            result = extract_license_plate(frame)
            if result.get('plate_text'):
                timestamp = frame_count / fps
                print(f"\n[+{timestamp:.1f}s] DETECTED PLATE: {result['plate_text']} (Confidence: {result['confidence']:.2f})")
                detections.append(result)

        frame_count += 1

    cap.release()
    print(f"\nFinished processing. Found {len(detections)} plate reads in the video.")

if __name__ == '__main__':
    # No Django setup required!
    test_inference_direct()
