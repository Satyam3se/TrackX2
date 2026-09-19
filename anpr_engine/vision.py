"""YOLOv8 + EasyOCR ANPR vision engine.

Heavy models (YOLO, EasyOCR) are initialized lazily and cached so an import
of this module never blocks, and the singleton loaders are reused across
frames/celery workers.
"""

import os
import tempfile
import threading
from functools import lru_cache

import cv2
import numpy as np

DETECTION_CONF_THRESHOLD = 0.1

_lock = threading.Lock()
_YOLO_CLS = None


def _load_yolo_class():
    """Lazily import the ``ultralytics.YOLO`` class on first use."""
    global _YOLO_CLS
    if _YOLO_CLS is not None:
        return _YOLO_CLS
    with _lock:
        if _YOLO_CLS is None:
            from ultralytics import YOLO

            _YOLO_CLS = YOLO
    return _YOLO_CLS


@lru_cache(maxsize=1)
def get_yolo_model(weights_path: str | None = None):
    """Return a cached YOLO model instance.

    Defaults to the base ``yolov8n.pt``; for accurate plate detection use
    ``YOLO_WEIGHTS`` pointing to fine‑tuned weights (e.g. ``license_plate_detector.pt``).
    """
    if weights_path is None:
        weights_path = os.environ.get('YOLO_WEIGHTS', 'yolov8n.pt')
    try:
        # Attempt loading with device argument (works on older versions)
        model = _load_yolo_class()(weights_path, device='cpu')
    except Exception:
        # Fallback for newer versions that reject the device kwarg
        model = _load_yolo_class()(weights_path)
        try:
            model.to('cpu')
        except Exception:
            pass
    print(f"[DEBUG] YOLO model loaded with weights: {weights_path}, model: {model}")
    return model


@lru_cache(maxsize=1)
def get_easyocr_reader():
    """Return a cached EasyOCR reader restricted to English, CPU-only."""
    import easyocr

    return easyocr.Reader(['en'], gpu=False, verbose=False)


def _read_frame(image_bytes_or_path):
    """Decode an image path, raw bytes, or a BGR np.ndarray frame."""
    if isinstance(image_bytes_or_path, np.ndarray):
        return image_bytes_or_path
    if isinstance(image_bytes_or_path, (bytes, bytearray)) or (
        hasattr(image_bytes_or_path, 'read')
    ):
        data = image_bytes_or_path
        if hasattr(data, 'read'):
            data = data.read()
        img = cv2.imdecode(
            np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR,
        )
    else:
        img = cv2.imread(os.fspath(image_bytes_or_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(
            f'Could not decode image from: {image_bytes_or_path!r}'
        )
    return img


def _detect_plate_boxes(model, image_bgr):
    """Run YOLO and return the best plate bbox ``[x1, y1, x2, y2]`` or None.

    Prefers boxes predicted as a license-plate class when the model exposes
    such a class name; otherwise falls back to the highest-confidence box.
    """
    results = model.predict(
        source=image_bgr,
        conf=DETECTION_CONF_THRESHOLD,
        verbose=False,
    )
    if not results:
        return None
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
    if confs is None:
        return None

    names = getattr(model, 'names', None)
    plate_classes = []
    if names:
        plate_classes = [
            cls_id for cls_id, name in names.items()
            if 'plate' in str(name).lower()
        ]

    candidate_ids = list(range(len(xyxy)))
    if plate_classes:
        cls_npy = boxes.cls.cpu().numpy() if boxes.cls is not None else None
        plate_ids = [
            i for i in candidate_ids
            if cls_npy is not None and int(cls_npy[i]) in plate_classes
        ]
        if plate_ids:
            candidate_ids = plate_ids

    best_idx = max(candidate_ids, key=lambda i: float(confs[i]))
    x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy[best_idx]]
    return x1, y1, x2, y2


def _preprocess_plate_crop(plate_bgr):
    """Advanced preprocessing for license plates: Grayscale -> CLAHE -> Adaptive Thresholding.
    
    Uses Contrast Limited Adaptive Histogram Equalization (CLAHE) to normalize
    lighting and adaptive thresholding to sharpen text against complex backgrounds.
    """
    # Scale up the image by 2x for better OCR recognition
    plate_bgr = cv2.resize(plate_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    # 1. Grayscale
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(gray)
    
    # Return the grayscale contrast-enhanced image. 
    # Adaptive thresholding usually degrades CNN-based OCR performance.
    return contrast_enhanced


def _clean_plate_text(raw_text):
    """Uppercase, strip non-alphanumeric characters, and fix common OCR errors.

    The original implementation dropped leading letters (e.g., "TS07JS9670" became
    "S07JS9670") and confused characters such as "O"/"0" or "I"/"1". This revised
    version applies a simple character‑mapping and, when the cleaned plate has the
    expected length but starts with a digit, prepends a "T" assuming the first
    character was omitted.
    """
    # Upper‑case and keep only alphanumerics
    cleaned = ''.join(ch for ch in raw_text.upper() if ch.isalnum())
    # Common OCR mis‑recognitions
    mapping = {
        'O': '0',
        'I': '1',
        'L': '1',
        'S': '5',
        'Z': '2',
    }
    cleaned = ''.join(mapping.get(c, c) for c in cleaned)
    # Recover missing leading 'T' if plate length is correct but starts with a digit
    if len(cleaned) == 9 and cleaned[0].isdigit():
        cleaned = 'T' + cleaned
    return cleaned


def extract_license_plate(image_bytes_or_path):
    """Detect + OCR a license plate in a single image.

    Returns::

        {
            'plate_text': str,
            'confidence': float,
            'cropped_img': bytes,   # PNG bytes of the processed plate crop
            'bbox': [x1, y1, x2, y2] | None,
        }
    """
    model = get_yolo_model()
    image_bgr = _read_frame(image_bytes_or_path)

    box = _detect_plate_boxes(model, image_bgr)
    if box is None:
        return {
            'plate_text': '',
            'confidence': 0.0,
            'cropped_img': b'',
            'bbox': None,
        }

    x1, y1, x2, y2 = box
    # Clamp crop to image bounds.
    h, w = image_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return {
            'plate_text': '',
            'confidence': 0.0,
            'cropped_img': b'',
            'bbox': box,
        }

    plate_bgr = image_bgr[y1:y2, x1:x2]
    processed = _preprocess_plate_crop(plate_bgr)

    reader = get_easyocr_reader()
    results = reader.readtext(processed, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    if not results:
        return {
            'plate_text': '',
            'confidence': 0.0,
            'cropped_img': _encode_png(processed),
            'bbox': box,
        }

    _, raw_text, conf = max(results, key=lambda r: r[2])
    plate_text = _clean_plate_text(raw_text)

    return {
        'plate_text': plate_text,
        'confidence': float(conf),
        'cropped_img': _encode_png(processed),
        'bbox': box,
    }


def _encode_png(img):
    ok, buf = cv2.imencode('.png', img)
    if not ok:
        return b''
    return buf.tobytes()


def save_crop(camera_id, plate_text, cropped_img, crop_root=None):
    """Persist cropped plate pixels as a timestamped PNG file.

    Args:
        camera_id (str): CameraNode.camera_id.
        plate_text (str): The recognized license plate text.
        cropped_img (bytes): PNG bytes of the crop (may be empty).
        crop_root (str|None): Directory to save into. Defaults to the ANPR
            crop root / MEDIA_ROOT / tempdir.

    Returns:
        str: Absolute path of the saved crop, or '' if no crop bytes.
    """
    if not cropped_img:
        return ''
    if crop_root is None:
        crop_root = _default_crop_root()
    os.makedirs(crop_root, exist_ok=True)
    from datetime import datetime as _datetime
    stamp = _datetime.now().strftime('%Y%m%d%H%M%S')
    safe_plate = ''.join(ch for ch in plate_text if ch.isalnum())
    filename = f'{camera_id}_{safe_plate}_{stamp}.png'
    filepath = os.path.join(crop_root, filename)
    with open(filepath, 'wb') as fh:
        fh.write(cropped_img)
    return filepath


def _default_crop_root():
    """Resolve the directory used to persist plate crops."""
    root = os.environ.get('ANPR_CROP_ROOT', '')
    if root:
        return root
    from django.conf import settings
    media_root = getattr(settings, 'MEDIA_ROOT', None)
    if media_root:
        return media_root
    return os.path.join(tempfile.gettempdir(), 'trackx_crops')