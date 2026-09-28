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
#: Plate-geometry sanity bounds. Real (Indian) plates are wide rectangles;
#: full-frame squares & speck-sized boxes are detector false positives that
#: would otherwise make OCR run for seconds on useless pixels.
PLATE_MIN_ASPECT = 1.5
PLATE_MAX_ASPECT = 8.0
PLATE_MAX_AREA_FRACTION = 0.25
#: Cap on the largest side of the image handed to EasyOCR; keeps per-frame OCR
#: time bounded even for loose crops (full-frame OCR is what caused 15s frames).
OCR_MAX_SIDE = 640
#: Full-frame edge OCR costs ~3-5s per no-plate frame (PlateVision trick).
#: Off by default so video/stream pipelines stay fast; set ``1`` to enable it
#: for single-image calls where a missed plate matters more than latency.
FULL_IMAGE_FALLBACK = os.environ.get('ANPR_FULL_IMAGE_FALLBACK', '0') == '1'
#: Contour-heuristic plate locator (ported from the Kalsekar ANPR project).
#: Zero dependency: when the YOLO detector finds nothing on a single image, we
#: search lower-half edge blobs shaped like a plate and OCR the best candidate.
#: It returns a real bbox + heuristic confidence, so it is strictly better than
#: the whole-frame edge OCR above. On by default; the video/stream pipeline does
#: not go through it (it uses ``_detect_plate_boxes`` directly), so per-frame
#: latency is unaffected.
CONTOUR_FALLBACK = os.environ.get('ANPR_CONTOUR_FALLBACK', '1') == '1'
#: Use the fast-plate-ocr pretrained ONNX model (cct-s-v2-global, ~64MB of
#: cached weights) as the primary OCR pass. It reads a 128x64 plate crop in
#: ~milliseconds with per-character confidences -- ~100x faster than EasyOCR at
#: comparable (often better) accuracy -- and those per-char confidences feed
#: the temporal voter in the video pipeline. EasyOCR remains the fallback for
#: images fast-plate-ocr cannot read. Set ``0`` to force the legacy EasyOCR
#: path everywhere. The module import is lazy: nothing loads unless this is on.
FAST_OCR = os.environ.get('ANPR_FAST_OCR', '1') == '1'
#: Confidence at or above which a fast-plate-ocr read is trusted outright and
#: the EasyOCR second opinion is skipped. The engine is well calibrated on
#: low-res crops: measured over the 250-crop labelled set, reads at conf>=0.9
#: are 97.6% exact while reads below 0.7 are 0% exact. Raising the bar from
#: 0.5 to 0.8 sends the weak 0.5-0.8 band to EasyOCR (the stronger reader on
#: small/blurred crops): +4pp exact on the w96 stress bench. Costs ~340ms on
#: the extra escalated crops, so lower it to go faster.
FAST_OCR_TRUST_CONFIDENCE = float(os.environ.get('ANPR_FAST_OCR_TRUST', '0.8'))
#: Consult EasyOCR as a second opinion whenever the fast-plate-ocr read is not
#: trustworthy. fast-plate-ocr alone is weak on small/blurred crops (16.7%
#: exact, 34.2% char on ``syn_eval``) while EasyOCR alone reaches 31.7% / 62.9%.
#: The old path only reached for EasyOCR when fast-plate-ocr returned *nothing*,
#: which happens on under 1% of real crops -- so the far more accurate engine
#: almost never ran. Gating on trust instead lifts the ensemble to 35.0% exact
#: and 63.7% char. Costs ~340ms on the crops that are escalated, so set
#: ``ANPR_OCR_ENSEMBLE=0`` to restore fast-plate-ocr-only behaviour.
OCR_ENSEMBLE = os.environ.get('ANPR_OCR_ENSEMBLE', '1') == '1'
#: Use the fast-alpr pretrained ONNX detector (YOLOv9-t-384, detects plates of
#: 65+ countries) as an *additional* detector, tried only when the primary
#: platevision/license-plate chain locates nothing. On most footage the tuned
#: platevision model wins, so this rarely fires; it mainly rescues foreign
#: plates. Set ``0`` to disable (zero overhead beyond the lazy import).
FAST_DETECTOR = os.environ.get('ANPR_FAST_DETECTOR', '1') == '1'
#: Cap on decoded raster pixels for uploaded images (decompression-bomb guard,
#: ported from the anpr-pipeline reference project). A small compressed file
#: can *declare* huge dimensions, so the declared size is checked before pixel
#: data is materialized (Pillow HEADER-only probe) and the decoded shape is
#: re-checked afterwards.
MAX_IMAGE_PIXELS = 40_000_000
#: Fraction of the working-frame area a plate blob must occupy (Kalsekar:
#: min_plate_area=1200 on a ~vehicle crop). Real Indian plates are ~0.1-1.5%
#: of a full frame, so the fraction keeps that intent across resolutions.
CONTOUR_MIN_AREA_FRACTION = 0.0006
#: Aspect-ratio band for plate-shaped bounding boxes (Kalsekar: 2.0-7.5).
CONTOUR_MIN_ASPECT = 2.0
CONTOUR_MAX_ASPECT = 7.5
#: Heuristic confidence below which a contour candidate is not worth OCR-ing.
CONTOUR_CONF_THRESHOLD = 0.55

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


@lru_cache(maxsize=8)
def get_yolo_model(weights_path: str | None = None):
    """Return a cached YOLO model instance (per weights path).

    Defaults to the base COCO model at ``pretrained_weights/yolov8n.pt``; for
    accurate plate detection use ``YOLO_WEIGHTS`` pointing to fine‑tuned
    weights (e.g. ``pretrained_weights/license_plate_detector.pt``). The cache
    is keyed by path so multiple candidate detectors can be loaded without
    re-instantiating each frame.
    """
    if weights_path is None:
        weights_path = os.environ.get('YOLO_WEIGHTS', 'pretrained_weights/yolov8n.pt')
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
    print(f"[DEBUG] YOLO model loaded with weights: {weights_path}")
    return model


DETECTOR_CHAIN = (
    'pretrained_weights/platevision_plate_detector.pt',  # tuned plate detector (mAP50 .92)
    'pretrained_weights/license_plate_detector.pt',        # TrackX Indian-plate fine-tune
)


def get_plate_detector():
    """Return the best available plate detector along the fallback chain.

    An explicit ``YOLO_WEIGHTS`` env override (e.g. in ``.env``) always wins.
    Otherwise tries ``DETECTOR_CHAIN`` paths first (a tuned plate model wins),
    falling back to the base ``pretrained_weights/yolov8n.pt``. Missing weight
    files are skipped.
    """
    env_weights = os.environ.get('YOLO_WEIGHTS')
    if env_weights:
        return get_yolo_model(env_weights)
    for path in DETECTOR_CHAIN:
        if os.path.isfile(path):
            return get_yolo_model(path)
    return get_yolo_model(None)


@lru_cache(maxsize=1)
def get_easyocr_reader():
    """Return a cached EasyOCR reader restricted to English, CPU-only."""
    import easyocr

    return easyocr.Reader(['en'], gpu=False, verbose=False)


@lru_cache(maxsize=1)
def get_fast_plate_ocr():
    """Return a cached fast-plate-ocr recognizer (cct-s-v2-global ONNX model).

    Lazy + cached: the model weights (~64MB) are only loaded on first call and
    reused for every subsequent frame/crop. Returns ``None`` when the package
    is not installed, the model is missing, or ``ANPR_FAST_OCR=0`` -- callers
    must fall back to EasyOCR in that case.
    """
    if not FAST_OCR:
        return None
    try:
        from fast_plate_ocr import LicensePlateRecognizer

        return LicensePlateRecognizer(
            hub_ocr_model='cct-s-v2-global-model', device='auto',
        )
    except Exception as exc:
        print(f'[WARN] fast-plate-ocr unavailable, using EasyOCR: {exc}')
        return None


@lru_cache(maxsize=1)
def get_fast_alpr():
    """Return a cached fast-alpr plate detector (YOLOv9-t-384 ONNX model).

    Detects plates from 65+ countries (the model most likely to rescue a
    foreign plate the tuned platevision model never learned). Like
    ``get_fast_plate_ocr`` this is lazy + optional; ``None`` when unavailable.
    """
    if not FAST_DETECTOR:
        return None
    try:
        from fast_alpr.default_detector import DefaultDetector

        return DefaultDetector(conf_thresh=0.25)
    except Exception as exc:
        print(f'[WARN] fast-alpr unavailable, skipping extra detection: {exc}')
        return None


def _read_frame(image_bytes_or_path):
    """Decode an image path, raw bytes, or a BGR np.ndarray frame.

    Guarded against decompression bombs: untrusted upload bytes can declare a
    huge raster in a few KB (e.g. 30000x30000 px declares 2.7 GB decoded).
    Pillow's ``Image.open()`` is lazy (header-only), so the *declared* size is
    checked there before handing bytes to ``cv2.imdecode`` -- which materializes
    the whole raster before returning and has no bomb guard of its own.
    """
    if isinstance(image_bytes_or_path, np.ndarray):
        return image_bytes_or_path
    if isinstance(image_bytes_or_path, (bytes, bytearray)) or (
        hasattr(image_bytes_or_path, 'read')
    ):
        data = image_bytes_or_path
        if hasattr(data, 'read'):
            data = data.read()
        _check_declared_size(data)
        img = cv2.imdecode(
            np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR,
        )
    else:
        img = cv2.imread(os.fspath(image_bytes_or_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(
            f'Could not decode image from: {image_bytes_or_path!r}'
        )
    # Re-check the decoded raster even though the header was in budget -- a
    # lying header must not widen the bound after the fact.
    _check_decoded_size(img)
    return img


def _check_declared_size(data):
    """Reject image bytes whose *declared* pixel count exceeds the budget."""
    import io as _io
    import warnings as _warnings

    try:
        from PIL import Image

        with _warnings.catch_warnings():
            # Pillow emits a warning in the band (MAX, 2*MAX] and only an
            # error above 2*MAX; promote the warning so nothing slips through.
            _warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(_io.BytesIO(data)) as probe:
                width, height = probe.size
    except Image.DecompressionBombError as exc:
        raise ValueError(f'Image declares too many pixels: {exc}') from exc
    except Image.DecompressionBombWarning as exc:
        raise ValueError(f'Image declares too many pixels: {exc}') from exc
    except ValueError as exc:
        # Pillow's own guard (fine -- it tripped the budget itself).
        raise
    except Exception:
        # Not an image Pillow understands; let cv2.imdecode decide.
        return
    if width <= 0 or height <= 0:
        raise ValueError('Image declares invalid dimensions')
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(
            f'Image declares {width}x{height} = {width * height}px, '
            f'over the {MAX_IMAGE_PIXELS}px limit'
        )


def _check_decoded_size(img):
    """Reject a decoded raster wider/taller than the pixel budget."""
    height, width = img.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError('Image decoded with invalid dimensions')
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(
            f'Decoded image is {width}x{height} = {width * height}px, '
            f'over the {MAX_IMAGE_PIXELS}px limit'
        )


def _detect_plate_boxes(model, image_bgr):
    """Run YOLO and return the best plate bbox ``[x1, y1, x2, y2]`` or None.

    Prefers boxes predicted as a license-plate class when the model exposes
    such a class name; otherwise falls back to the highest-confidence box.

    A plate-geometry sanity filter rejects full-frame squares and tiny specks
    that some detectors emit when no plate is actually in frame -- feeding
    those to OCR wastes many seconds per frame and yields garbage text.
    """
    boxes = _detect_plate_boxes_many(model, image_bgr)
    return boxes[0] if boxes else None


def _detect_plate_boxes_many(model, image_bgr):
    """Return *all* geometry-valid plate bboxes ``[[x1, y1, x2, y2], ...]``.

    Same candidate selection and sanity filter as ``_detect_plate_boxes`` but
    keeps every passing box instead of collapsing to the best one, so a frame
    with several vehicles gets one track + OCR pass per plate (the
    anpr-pipeline reference detects and reads every plate per frame). Sorted
    by confidence, highest first.
    """
    results = model.predict(
        source=image_bgr,
        conf=DETECTION_CONF_THRESHOLD,
        verbose=False,
    )
    if not results:
        return []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
    if confs is None:
        return []

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

    h, w = image_bgr.shape[:2]
    frame_area = float(max(h * w, 1))
    valid = []
    for i in candidate_ids:
        x1, y1, x2, y2 = xyxy[i]
        bw = float(x2 - x1)
        bh = float(y2 - y1)
        if bw < 15 or bh < 8 or bh <= 0:
            continue  # too tiny to OCR meaningfully
        aspect = bw / bh
        area_frac = (bw * bh) / frame_area
        if aspect < PLATE_MIN_ASPECT or aspect > PLATE_MAX_ASPECT:
            continue
        if area_frac > PLATE_MAX_AREA_FRACTION:
            continue  # full-frame false positive
        valid.append(
            ([int(round(float(v))) for v in xyxy[i]], float(confs[i]))
        )

    valid.sort(key=lambda item: item[1], reverse=True)
    return [box for box, _ in valid]


def _detect_plate_with_contours(image_bgr):
    """Locate a plate with contour heuristics when YOLO plate detection fails.

    Ported from the Kalsekar ANPR project (``detector/plate_detector.py``).
    Faithful to their design: the contour search never runs on the raw full
    frame -- the plate detector runs inside a *vehicle* crop (found with a
    COCO object detector), which makes plate-area thresholds and the
    lower-half prior reliable, and keeps the OCR crop small so it stays fast.

    Returns ``[x1, y1, x2, y2]`` in original-frame coordinates, or None.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None
    h, w = image_bgr.shape[:2]
    if min(h, w) < 40:
        return None

    vehicle = _detect_vehicle_bbox(image_bgr)
    if vehicle is None:
        return None

    vx1, vy1, vx2, vy2 = vehicle
    # Crop tightly but keep a little slack so an edge-on plate stays inside.
    mx = max(8, int((vx2 - vx1) * 0.08))
    my = max(8, int((vy2 - vy1) * 0.08))
    cx1, cy1 = max(0, vx1 - mx), max(0, vy1 - my)
    cx2, cy2 = min(w, vx2 + mx), min(h, vy2 + my)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    vehicle_crop = image_bgr[cy1:cy2, cx1:cx2]

    candidates = _find_contour_candidates(vehicle_crop, lower_half_only=True)
    if not candidates:
        candidates = _find_contour_candidates(vehicle_crop, lower_half_only=False)
    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, bbox = candidates[0]
    rx1, ry1, rx2, ry2 = bbox

    confidence = _estimate_contour_confidence(bbox, vehicle_crop.shape[:2])
    if confidence < CONTOUR_CONF_THRESHOLD:
        return None

    x1, y1 = max(0, cx1 + rx1), max(0, cy1 + ry1)
    x2, y2 = min(w, cx1 + rx2), min(h, cy1 + ry2)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _detect_vehicle_bbox(image_bgr, conf_threshold=0.35):
    """Return a COCO vehicle bounding box ``[x1, y1, x2, y2]`` or None.

    Uses the base COCO model (``get_yolo_model(None)`` ->
    pretrained_weights/yolov8n.pt) and
    keeps car/bus/truck/auto/motorcycle classes. Vehicles sit lower in the
    frame, so among confident detections we prefer the one whose bottom is
    closest to the frame bottom (that is where a plate can be OCR'd).
    """
    model = _get_vehicle_model()
    if model is None:
        return None
    results = model.predict(
        source=image_bgr, conf=conf_threshold, verbose=False,
    )
    if not results:
        return None
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    names = getattr(model, 'names', None) or {}
    vehicle_classes = {
        cls_id for cls_id, name in names.items()
        if str(name).lower() in {'car', 'truck', 'bus', 'motorbike', 'auto'}
    }
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
    cls_npy = boxes.cls.cpu().numpy() if boxes.cls is not None else None
    h, w = image_bgr.shape[:2]
    best = None
    best_key = -1.0
    for i in range(len(xyxy)):
        if cls_npy is not None and vehicle_classes and int(cls_npy[i]) not in vehicle_classes:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy[i]]
        if x2 <= x1 or y2 <= y1:
            continue
        # Prefer large vehicles whose bottom is near the frame bottom.
        key = (float(y2 - y1) + float(y1)) * (confs[i] if confs is not None else 1.0)
        if key > best_key:
            best_key = key
            best = [x1, y1, x2, y2]
    return best


@lru_cache(maxsize=1)
def _get_vehicle_model():
    try:
        return get_yolo_model(os.environ.get('YOLO_VEHICLE_WEIGHTS') or 'pretrained_weights/yolov8n.pt')
    except Exception as exc:
        print(f'[WARN] vehicle model unavailable: {exc}')
        return None


def _detect_vehicle_boxes(image_bgr, conf_threshold=0.25):
    if image_bgr is None or image_bgr.size == 0:
        return []
    try:
        model = _get_vehicle_model()
        if model is None:
            return []
        results = model.predict(
            source=image_bgr, conf=conf_threshold, verbose=False,
        )
    except Exception as exc:
        print(f'[WARN] vehicle detection failed: {exc}')
        return []
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return []
    boxes = results[0].boxes
    names = getattr(model, 'names', None) or {}
    if isinstance(names, dict):
        vehicle_classes = {
            cls_id for cls_id, name in names.items()
            if str(name).lower() in {
                'car', 'truck', 'bus', 'motorcycle', 'motorbike', 'auto',
            }
        }
    elif isinstance(names, (list, tuple)):
        vehicle_classes = {
            index for index, name in enumerate(names)
            if str(name).lower() in {
                'car', 'truck', 'bus', 'motorcycle', 'motorbike', 'auto',
            }
        }
    else:
        vehicle_classes = set()
    if not vehicle_classes:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
    classes = boxes.cls.cpu().numpy() if boxes.cls is not None else None
    h, w = image_bgr.shape[:2]
    detected = []
    for index, coordinates in enumerate(xyxy):
        if vehicle_classes and (
            classes is None or int(classes[index]) not in vehicle_classes
        ):
            continue
        x1, y1, x2, y2 = [int(round(float(value))) for value in coordinates]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        confidence = float(confs[index]) if confs is not None else 0.0
        detected.append(([x1, y1, x2, y2], confidence))
    detected.sort(key=lambda item: item[1], reverse=True)
    return [box for box, _ in detected]


def _vehicle_for_plate(plate_bbox, vehicle_boxes):
    if not plate_bbox or not vehicle_boxes:
        return None
    x1, y1, x2, y2 = plate_bbox
    center_x = (float(x1) + float(x2)) / 2
    center_y = (float(y1) + float(y2)) / 2
    plate_area = max(1.0, (float(x2) - x1) * (float(y2) - y1))
    ranked = []
    for vehicle in vehicle_boxes:
        vx1, vy1, vx2, vy2 = vehicle
        intersection_width = max(0.0, min(x2, vx2) - max(x1, vx1))
        intersection_height = max(0.0, min(y2, vy2) - max(y1, vy1))
        intersection = intersection_width * intersection_height
        if intersection <= 0:
            continue
        vehicle_center_x = (float(vx1) + float(vx2)) / 2
        vehicle_center_y = (float(vy1) + float(vy2)) / 2
        distance = (
            (center_x - vehicle_center_x) ** 2
            + (center_y - vehicle_center_y) ** 2
        )
        contains_center = vx1 <= center_x <= vx2 and vy1 <= center_y <= vy2
        area = (float(vx2) - vx1) * (float(vy2) - vy1)
        ranked.append((
            contains_center,
            intersection / plate_area,
            -distance,
            -area,
            vehicle,
        ))
    return max(ranked)[-1] if ranked else None


def _find_contour_candidates(image_bgr, lower_half_only=True):
    """Return [(score, (x1, y1, x2, y2)), ...] for plate-shaped edge blobs.

    Scores are ``area * aspect`` so bigger, wider blobs win (Kalsekar's
    ranking). A full-frame ``min_area_fraction`` replaces their absolute
    ``1200`` px floor (which was tuned for a ~vehicle-sized crop) so the
    heuristic stays resolution-independent on entire frames.
    """
    h, w = image_bgr.shape[:2]
    offset_y = h // 3 if lower_half_only else 0
    region = image_bgr[offset_y:, :] if lower_half_only else image_bgr
    if region.size == 0:
        return []

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) \
        if len(region.shape) == 3 else region.copy()
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    frame_area = float(w * h)
    min_area = CONTOUR_MIN_AREA_FRACTION * frame_area

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw <= 0 or ch <= 0:
            continue
        aspect = cw / float(ch)
        if aspect < CONTOUR_MIN_ASPECT or aspect > CONTOUR_MAX_ASPECT:
            continue
        if cw < w * 0.08 or ch < h * 0.03:
            continue
        score = area * min(aspect, CONTOUR_MAX_ASPECT)
        candidates.append((score, (x, y + offset_y, x + cw, y + ch + offset_y)))
    return candidates


def _estimate_contour_confidence(bbox, image_shape):
    """Heuristic confidence for a contour plate candidate (Kalsekar port).

    Rewards aspect ratios near the 4.5 target and plates that occupy a
    "realistic" share of the frame; returns 0.45-0.95.
    """
    x1, y1, x2, y2 = bbox
    plate_w = max(1, x2 - x1)
    plate_h = max(1, y2 - y1)
    aspect = plate_w / float(plate_h)
    target_ratio = 4.5
    ratio_score = max(0.0, 1.0 - abs(aspect - target_ratio) / target_ratio)
    area_ratio = (plate_w * plate_h) / float(max(image_shape[0] * image_shape[1], 1))
    area_score = min(1.0, area_ratio * 10.0)
    confidence = 0.45 + 0.35 * ratio_score + 0.2 * area_score
    return round(min(float(confidence), 0.95), 3)


def _deskew_plate(plate_bgr):
    """Straighten a tilted plate crop via its dominant contour (PlateVision trick).

    Uses ``minAreaRect`` of the plate's largest edge contour to estimate the
    angle, then rotates so the plate baseline becomes horizontal.
    """
    edges = cv2.Canny(plate_bgr, 50, 150)
    contours = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]
    if not contours:
        return plate_bgr
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    angle = rect[2]
    if angle > 45:
        angle -= 90
    if angle < -45:
        angle += 90
    if abs(angle) < 6:
        return plate_bgr  # already horizontal enough
    h, w = plate_bgr.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        plate_bgr, matrix, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _sharpen(gray):
    """Unsharp-mask style sharpening kernel (PlateVision's step)."""
    kernel = np.array(
        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32,
    )
    return cv2.filter2D(gray, -1, kernel)


def _preprocess_plate_crop(plate_bgr, *, sharpen=True, upscale=3.0):
    """Advanced preprocessing for license plates (adapted from PlateVision-AI).

    Pipeline: deskew -> pad border -> Nx LANCZOS upscale -> grayscale ->
    bilateral filter -> CLAHE -> sharpen, tuned so EasyOCR reads the text
    region cleanly even on tilted or low-light crops.

    ``sharpen=False`` skips the final unsharp-mask step. An A/B against real
    low-res footage shows the unsharp step *degrades* fast-plate-ocr's reads
    (63% -> 70% correct on the TS07JS9670 window) while barely helping
    EasyOCR, so the fast-plate-ocr path feeds the unsharpened crop and only
    the EasyOCR fallback receives the fully-sharpened version.
    ``upscale`` overrides the 3x factor (use 4.5x for distant ~96px crops).
    """
    # 1. Bring tilted plates onto the horizontal baseline.
    plate_bgr = _deskew_plate(plate_bgr)

    # 2. Small uniform border so characters at plate edges aren't clipped.
    plate_bgr = cv2.copyMakeBorder(
        plate_bgr, 8, 8, 8, 8, cv2.BORDER_REPLICATE,
    )

    # 3. Big Nx LANCZOS upscale: sharper than the old 2x CUBIC. For plates
    #    already >900px wide (deep zoom / oversized boxes), upscaling further
    #    only slows OCR, so clamp the target area.
    target_max_area = 900 * 250
    area = plate_bgr.shape[0] * plate_bgr.shape[1]
    if area > target_max_area:
        scale = (target_max_area / area) ** 0.5
        plate_bgr = cv2.resize(
            plate_bgr, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    else:
        plate_bgr = cv2.resize(
            plate_bgr, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4,
        )

    # 4. Grayscale + bilateral noise reduction.
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 75, 75)

    # 5. CLAHE (Contrast Limited Adaptive Histogram Equalization).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    result = clahe.apply(gray)

    # Optional final unsharp mask. Off for the fast-plate-ocr path: real low-res
    # footage reads ~7% better without it and the synthetic benchmark agrees.
    if sharpen:
        result = _sharpen(result)

    # 6. Cap the largest side handed to EasyOCR. Without this a loose/attacker
    # box makes OCR scan huge images (a 360x360 crop at 3x = 1080x1080 just to
    # find nothing). Bounding it keeps per-frame cost predictable.
    hh, ww = result.shape[:2]
    longest = max(hh, ww)
    if longest > OCR_MAX_SIDE:
        scale = OCR_MAX_SIDE / longest
        result = cv2.resize(
            result, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    return result


def _clean_plate_text(raw_text):
    """Uppercase, strip non-alphanumeric characters, and fix common OCR errors.

    Uses the position-aware Indian plate corrector (``plate_text.clean_plate_text``),
    which resolves letter/digit OCR confusions by their slot in the plate layout
    (e.g. it will not convert an ``S`` that sits in a letter position into ``5``).
    """
    from .plate_text import clean_plate_text

    return clean_plate_text(raw_text)


def _ocr_plate_image(img, allowlist=True):
    """Run EasyOCR over an image (crop or full image) and return best line."""
    reader = get_easyocr_reader()
    kw = {}
    if allowlist:
        kw['allowlist'] = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    results = reader.readtext(img, **kw) if kw else reader.readtext(img)
    if not results:
        return None, 0.0
    return max(results, key=lambda r: r[2])[1], max(r[2] for r in results)


def _ocr_plate_image_fast(processed, allowlist=True):
    """OCR a (preprocessed) plate crop with fast-plate-ocr (cct-s-v2 ONNX).

    Returns ``(text, confidence, char_confidences)`` where ``char_confidences``
    is a per-plate-character confidence vector (one float per char of ``text``)
    used by the temporal voter. Returns ``(None, 0.0, None)`` when the recognizer
    is unavailable or produces nothing, so callers fall back to EasyOCR. The
    model wants RGB crop data, so the grayscale preprocessed pipeline output is
    broadcast to 3 channels (color is irrelevant after grayscale conversion).
    """
    reader = get_fast_plate_ocr()
    if reader is None:
        return None, 0.0, None
    if processed.ndim == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
    try:
        results = reader.run(processed, return_confidence=True)
    except Exception as exc:
        print(f'[WARN] fast-plate-ocr inference failed: {exc}')
        return None, 0.0, None
    if not results:
        return None, 0.0, None
    plate = str(results[0].plate)
    if not plate:
        return None, 0.0, None
    char_probs = np.atleast_1d(results[0].char_probs)
    if char_probs.size == 0:
        char_probs = np.array([0.0])
    n = min(len(plate), char_probs.size)
    conf = float(np.mean(char_probs[:n]))
    return plate, conf, [float(c) for c in char_probs[:n]]


def _read_is_structurally_valid(raw_text):
    """True when an OCR read resolves to a well-formed Indian plate layout.

    Cheap proxy for "the engine read a plate rather than some plate-shaped
    noise": it reuses the resolver that ``_clean_plate_text`` already runs, so
    no extra OCR work is needed.
    """
    if not raw_text:
        return False
    from .plate_text import is_valid_plate, resolve_plate_structure

    plate, _score = resolve_plate_structure(raw_text)
    return bool(plate) and is_valid_plate(plate)


def _ocr_plate_image_preferred(plate_crop, allowlist=True, *, trust_confidence=None):
    """Read a RAW plate crop: fast-plate-ocr first, EasyOCR as second opinion.

    ``plate_crop`` is the **raw BGR crop** pulled straight from the frame, and
    each engine gets the preprocessing it reads best on real low-res footage:

      * **fast-plate-ocr** (primary): the deskew -> 3x LANCZOS -> grayscale ->
        bilateral -> CLAHE pipeline *without* the final unsharp mask. A/B on
        real footage (TS07JS9670 window) showed the unsharp step degrading this
        model's reads (63% -> 70% correct without it); the synthetic benchmark
        agrees. Bilateral must stay (dropping it drops reads to ~54%).
      * **EasyOCR** (second opinion): the full legacy ``_preprocess_plate_crop``
        pipeline *including* the sharpen step, which is what it was tuned for
        (``_ocr_plate_image_rescue``).

    A read is only accepted from fast-plate-ocr when it is *trustworthy*:
    confidence >= ``trust_confidence`` **and** the text resolves to a valid
    plate layout. Otherwise EasyOCR is asked for its own read and the two are
    arbitrated by structural validity. EasyOCR is a much stronger reader on
    small, blurred, or rotated crops, but it costs ~340ms per crop versus
    ~35ms, so it is only paid for where it can change the answer. Rationale and
    measurements live in ``TRACKX_GUIDE.md``.

    Returns ``(text, confidence, char_confidences)``. ``char_confidences`` is
    ``None`` when the winning read came from EasyOCR (no per-char scores); the
    fast-plate-ocr path returns one float per character.
    """
    if trust_confidence is None:
        trust_confidence = FAST_OCR_TRUST_CONFIDENCE

    fast_text, fast_conf, fast_chars = _ocr_plate_image_fast(
        _preprocess_plate_crop(plate_crop, sharpen=False),
    )
    fast_conf = float(fast_conf or 0.0)

    # Trustworthy fast read: accept it and never touch the slow engine.
    if fast_text and not OCR_ENSEMBLE:
        return fast_text, fast_conf, fast_chars
    if (
        fast_text
        and fast_conf >= trust_confidence
        and _read_is_structurally_valid(fast_text)
    ):
        return fast_text, fast_conf, fast_chars

    processed = _preprocess_plate_crop(plate_crop, sharpen=True)
    eas_text, eas_conf = _ocr_plate_image_rescue(processed, allowlist)
    eas_conf = float(eas_conf or 0.0)

    # Second chance at a larger upscale for weak reads (tiny/distant plates).
    # At ~96px crop widths the standard 3x upscale leaves characters at
    # ~9px each and both engines return short garbage ('IO', 'CST', 'GE').
    # A 4.5x LANCZOS retry resolves several of those to full plates.
    # Only fires when the first EasyOCR read is weak, so clear plates
    # pay nothing and weak crops pay one extra ~300-500ms pass.
    OCR_RETRY_UPSCALE = 4.5
    OCR_RETRY_CONFIDENCE = 0.5
    if (
        eas_conf < OCR_RETRY_CONFIDENCE
        or not eas_text
        or not _read_is_structurally_valid(eas_text)
    ):
        retry_processed = _preprocess_plate_crop(
            plate_crop, sharpen=True, upscale=OCR_RETRY_UPSCALE,
        )
        retry_text, retry_conf = _ocr_plate_image_rescue(
            retry_processed, allowlist,
        )
        retry_conf = float(retry_conf or 0.0)
        # Adopt the retry only when it is strictly better: a valid read
        # beats an invalid one, otherwise clearly higher confidence wins.
        # (Never replace a valid read with an invalid one.)
        if retry_text and (
            not eas_text
            or (
                _read_is_structurally_valid(retry_text)
                and (
                    not _read_is_structurally_valid(eas_text)
                    or retry_conf > eas_conf + 0.05
                )
            )
        ):
            eas_text, eas_conf = retry_text, retry_conf

    if not fast_text:
        return eas_text, eas_conf, None
    if not eas_text:
        # EasyOCR is stricter and can come back empty on a crop the fast model
        # did read; never throw a usable read away.
        return fast_text, fast_conf, fast_chars

    # Both engines produced text. Prefer whichever resolves to a real plate
    # layout; when both are valid, prefer higher-confidence or longer read.
    # When neither is valid, keep the longer read as a best-effort guess.
    fast_valid = _read_is_structurally_valid(fast_text)
    eas_valid = _read_is_structurally_valid(eas_text)
    if fast_valid and eas_valid:
        if abs(fast_conf - eas_conf) > 0.05:
            return (fast_text, fast_conf, fast_chars) if fast_conf > eas_conf else (eas_text, eas_conf, None)
        return (fast_text, fast_conf, fast_chars) if len(fast_text) >= len(eas_text) else (eas_text, eas_conf, None)
    if eas_valid:
        return eas_text, eas_conf, None
    if fast_valid:
        return fast_text, fast_conf, fast_chars
    return (fast_text, fast_conf, fast_chars) if len(fast_text) >= len(eas_text) else (eas_text, eas_conf, None)


def _ocr_plate_image_rescue(processed, allowlist=True):
    """EasyOCR with a rescue variant, ported from the Kalsekar PlateReader.

    Kalsekar OCRs *all* preprocessing variants (gray/denoised/otsu_inv/
    adaptive/adaptive_inv) and keeps the best-confidence result. With EasyOCR
    that is ~6x the per-crop cost, so we run the primary pass and only when the
    read is empty or low-confidence try an inverted adaptive-threshold variant
    (rescues dark-on-light / light-on-dark plates cheaply). Returns
    ``(text, confidence)`` whichever variant scored higher.
    """
    raw_text, conf = _ocr_plate_image(processed, allowlist)
    if raw_text and conf >= 0.5:
        return raw_text, conf

    if len(processed.shape) == 3:
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        gray = processed
    adaptive_inv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 15,
    )
    resc_text, resc_conf = _ocr_plate_image(adaptive_inv, allowlist)
    if not resc_text:
        return raw_text, conf
    if raw_text and conf >= resc_conf:
        return raw_text, conf
    return resc_text, resc_conf


def _full_image_fallback(image_bgr):
    """PlateVision's trick: when the detector finds no box, OCR edge regions
    of the whole frame and look for a plate-shaped text candidate."""
    h, w = image_bgr.shape[:2]
    if min(h, w) < 40:
        return None
    # Downscale first so the whole-frame OCR stays bounded in time.
    longest = max(h, w)
    if longest > OCR_MAX_SIDE:
        scale = OCR_MAX_SIDE / longest
        image_bgr = cv2.resize(
            image_bgr, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        h, w = image_bgr.shape[:2]
    side = max(h, w) / 1200.0
    thresh_kernel = max(5, int(round(side * 12)) | 1)
    edges = cv2.Canny(image_bgr, 50, 150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(
        cv2.MORPH_RECT, (thresh_kernel, thresh_kernel)
    ))
    # EasyOCR reads the edge map as if it were a document.
    raw_text, _ = _ocr_plate_image(edges, allowlist=False)
    if not raw_text:
        return None
    plate_text = _clean_plate_text(raw_text)
    if len(plate_text) < 6:
        return None
    return plate_text


def extract_license_plates(image_bytes_or_path):
    model = get_plate_detector()
    image_bgr = _read_frame(image_bytes_or_path)
    boxes = _detect_plate_boxes_many(model, image_bgr)
    contour_confidences = {}
    if not boxes and CONTOUR_FALLBACK:
        candidates = sorted(
            _find_contour_candidates(image_bgr),
            key=lambda candidate: candidate[0],
            reverse=True,
        )
        boxes = [box for _, box in candidates]
        contour_confidences = {
            tuple(box): _estimate_contour_confidence(box, image_bgr.shape[:2])
            for box in boxes
        }

    vehicle_boxes = _detect_vehicle_boxes(image_bgr) if boxes else []
    h, w = image_bgr.shape[:2]
    detections = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        clipped_box = [x1, y1, x2, y2]
        pad = 12
        x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
        x2p, y2p = min(w, x2 + pad), min(h, y2 + pad)
        plate_bgr = image_bgr[y1p:y2p, x1p:x2p]
        raw_text, confidence, char_confidences = _ocr_plate_image_preferred(
            plate_bgr,
        )
        plate_text = _clean_plate_text(raw_text) if raw_text else ''
        contour_conf = contour_confidences.get(tuple(box))
        if contour_conf is not None:
            confidence = max(float(confidence), contour_conf * 0.9)
        detections.append({
            'plate_text': plate_text,
            'confidence': float(confidence),
            'bbox': clipped_box,
            'char_confidences': char_confidences,
            'vehicle_bbox': _vehicle_for_plate(clipped_box, vehicle_boxes),
        })
    return detections


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
    model = get_plate_detector()
    image_bgr = _read_frame(image_bytes_or_path)

    contour_conf = None
    box = _detect_plate_boxes(model, image_bgr)
    if box is None and CONTOUR_FALLBACK:
        # Ported from the Kalsekar ANPR project: when the model misses the
        # plate, search edge blobs shaped like a plate (lower-half prior,
        # aspect band 2.0-7.5, area * aspect scoring) and OCR the best one.
        # This is cheap (~10-30ms just to localise) and returns a real bbox
        # with a heuristic confidence, unlike the whole-frame edge OCR below.
        box = _detect_plate_with_contours(image_bgr)
        contour_conf = None
        if box is not None:
            contour_conf = _estimate_contour_confidence(box, image_bgr.shape[:2])

    if box is None:
        # No plate localised -- optional PlateVision fallback: OCR the whole
        # frame edges looking for a plate-shaped string (enable via
        # ANPR_FULL_IMAGE_FALLBACK=1; costs ~3-5s per no-plate frame).
        fallback_text = None
        if FULL_IMAGE_FALLBACK:
            fallback_text = _full_image_fallback(image_bgr)
        return {
            'plate_text': fallback_text or '',
            'confidence': 0.4 if fallback_text else 0.0,
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
    # Pad the crop so OCR characters at plate edges survive detection jitter
    # (PlateVision uses an 8-12px border; preprocess adds its own replicate pad).
    pad = 12
    x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
    x2p, y2p = min(w, x2 + pad), min(h, y2 + pad)
    plate_bgr = image_bgr[y1p:y2p, x1p:x2p]
    # Heavy EasyOCR-tuned crop (with the final sharpen step): used for the
    # fallback path inside _ocr_plate_image_preferred and for the persisted
    # crop image.
    processed = _preprocess_plate_crop(plate_bgr)

    # fast-plate-ocr reads its own unsharpened preprocessing of the RAW crop
    # (the final sharpen step degrades it); EasyOCR fallback lives inside.
    raw_text, conf, char_conf = _ocr_plate_image_preferred(plate_bgr)
    if not raw_text:
        return {
            'plate_text': '',
            'confidence': 0.0,
            'cropped_img': _encode_png(processed),
            'bbox': box,
        }

    plate_text = _clean_plate_text(raw_text)

    if contour_conf is not None:
        # Heuristic confidence from the contour locator can lift a weak read
        # that still produced plausible text; never above its 0.95 cap.
        conf = max(float(conf), contour_conf * 0.9)

    return {
        'plate_text': plate_text,
        'confidence': float(conf),
        'cropped_img': _encode_png(processed),
        'bbox': box,
        'char_confidences': char_conf,
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