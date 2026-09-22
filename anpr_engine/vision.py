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

from .tracking import iou

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

    Defaults to the base ``yolov8n.pt``; for accurate plate detection use
    ``YOLO_WEIGHTS`` pointing to fine‑tuned weights (e.g.
    ``license_plate_detector.pt``). The cache is keyed by path so multiple
    candidate detectors can be loaded without re-instantiating each frame.
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
    print(f"[DEBUG] YOLO model loaded with weights: {weights_path}")
    return model


DETECTOR_CHAIN = (
    'pretrained_weights/platevision_plate_detector.pt',
    'license_plate_detector.pt',
    'runs/detect/license_plate_model-3/weights/best.pt',
    'runs/detect/runs/detect/lp_quick/weights/best.pt',
)


def resolve_detector_path():
    """Return the detector weights path that get_plate_detector() will use."""
    env_weights = os.environ.get('YOLO_WEIGHTS')
    if env_weights:
        return env_weights
    for path in DETECTOR_CHAIN:
        if os.path.isfile(path):
            return path
    return 'yolov8n.pt'  # Default


def get_plate_detector():
    """Return the best available plate detector along the fallback chain.

    An explicit ``YOLO_WEIGHTS`` env override (e.g. in ``.env``) always wins.
    Otherwise tries ``DETECTOR_CHAIN`` paths first (a tuned plate model wins),
    falling back to the base ``yolov8n.pt``. Missing weight files are skipped.
    """
    env_weights = os.environ.get('YOLO_WEIGHTS')
    if env_weights:
        print(f"[TRACKX] Explicitly loading YOLO weights from ENV: {env_weights}")
        return get_yolo_model(env_weights)
    for path in DETECTOR_CHAIN:
        if os.path.isfile(path):
            print(f"[TRACKX] Auto-detected YOLO weights from DETECTOR_CHAIN: {path}")
            return get_yolo_model(path)
    print(f"[TRACKX] Falling back to default YOLO weights: yolov8n.pt")
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


def _detect_vehicles(image_bgr, conf_threshold=0.25):
    """Detect vehicles (cars, trucks, buses, motorcycles) in the frame.

    Returns a list of vehicle bounding boxes: ``[[x1, y1, x2, y2], ...]``.
    """
    if image_bgr is None or image_bgr.size == 0:
        return []
    model = get_yolo_model(None)
    results = model.predict(
        source=image_bgr, conf=conf_threshold, verbose=False,
    )
    if not results:
        return []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    names = getattr(model, 'names', None) or {}
    vehicle_classes = {
        cls_id for cls_id, name in names.items()
        if str(name).lower() in {'car', 'truck', 'bus', 'motorcycle', 'motorbike', 'auto'}
    }
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
    cls_npy = boxes.cls.cpu().numpy() if boxes.cls is not None else None
    h, w = image_bgr.shape[:2]
    vehicles = []
    for i in range(len(xyxy)):
        if cls_npy is not None and vehicle_classes and int(cls_npy[i]) not in vehicle_classes:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy[i]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            continue
        conf = float(confs[i]) if confs is not None else 1.0
        vehicles.append(([x1, y1, x2, y2], conf))
    # Sort largest/most confident first
    vehicles.sort(key=lambda v: (v[0][2] - v[0][0]) * (v[0][3] - v[0][1]) * v[1], reverse=True)
    return [v[0] for v in vehicles]


def _detect_plates_raw(model, crop_bgr):
    """Run plate YOLO detector directly on a crop (vehicle ROI or isolated plate)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return []
    results = model.predict(
        source=crop_bgr,
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

    h, w = crop_bgr.shape[:2]
    crop_area = float(max(h * w, 1))
    valid = []
    for i in candidate_ids:
        x1, y1, x2, y2 = xyxy[i]
        bw = float(x2 - x1)
        bh = float(y2 - y1)
        if bw < 15 or bh < 8 or bh <= 0:
            continue
        aspect = bw / bh
        area_frac = (bw * bh) / crop_area
        if aspect < PLATE_MIN_ASPECT or aspect > PLATE_MAX_ASPECT:
            continue
        if area_frac > 0.4:
            continue
        valid.append(
            ([int(round(float(v))) for v in xyxy[i]], float(confs[i]))
        )

    valid.sort(key=lambda item: item[1], reverse=True)
    return valid


def _find_contour_in_vehicle_crop(vehicle_crop):
    """Search for plate-shaped edge blobs in a vehicle crop."""
    if vehicle_crop is None or vehicle_crop.size == 0:
        return None
    candidates = _find_contour_candidates(vehicle_crop, lower_half_only=True)
    if not candidates:
        candidates = _find_contour_candidates(vehicle_crop, lower_half_only=False)
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _detect_plate_boxes(model, image_bgr, vehicles=None):
    """Run YOLO and return the best plate bbox ``[x1, y1, x2, y2]`` or None."""
    boxes = _detect_plate_boxes_many(model, image_bgr, vehicles=vehicles)
    return boxes[0] if boxes else None


def _detect_plate_boxes_many(model, image_bgr, vehicles=None):
    """Return *all* geometry-valid plate bboxes located on detected vehicles.

    First detects vehicles (car, truck, bus, motorcycle) in the frame, then
    detects license plates within each vehicle crop. This two-stage pipeline
    eliminates false-positive detections on background objects (trees, signs,
    buildings, pedestrians) and dramatically improves recognition on smaller
    or distant vehicles by preserving plate resolution in the crop.

    Returns ``[[x1, y1, x2, y2], ...]`` sorted by detection confidence.
    """
    if image_bgr is None or image_bgr.size == 0:
        return []
    h, w = image_bgr.shape[:2]

    # Stage 1: Vehicle detection
    if vehicles is None:
        vehicles = _detect_vehicles(image_bgr)

    valid_plates = []

    # Stage 2: Plate detection inside each detected vehicle
    if vehicles:
        for vx1, vy1, vx2, vy2 in vehicles:
            vw = vx2 - vx1
            vh = vy2 - vy1
            if vw < 20 or vh < 20:
                continue
            # Add small 4% padding around vehicle to prevent clipping bumper plates
            px = max(4, int(vw * 0.04))
            py = max(4, int(vh * 0.04))
            cx1 = max(0, vx1 - px)
            cy1 = max(0, vy1 - py)
            cx2 = min(w, vx2 + px)
            cy2 = min(h, vy2 + py)
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            vcrop = image_bgr[cy1:cy2, cx1:cx2]

            # Run plate detector on vehicle crop
            p_boxes = _detect_plates_raw(model, vcrop)
            for pbox, pconf in p_boxes:
                rx1, ry1, rx2, ry2 = pbox
                gx1 = cx1 + rx1
                gy1 = cy1 + ry1
                gx2 = cx1 + rx2
                gy2 = cy1 + ry2
                valid_plates.append(([gx1, gy1, gx2, gy2], pconf))

            # If model didn't detect plate in this vehicle and contour fallback is on:
            if not p_boxes and CONTOUR_FALLBACK:
                cbox = _find_contour_in_vehicle_crop(vcrop)
                if cbox is not None:
                    rx1, ry1, rx2, ry2 = cbox
                    c_conf = _estimate_contour_confidence(cbox, vcrop.shape[:2])
                    if c_conf >= CONTOUR_CONF_THRESHOLD:
                        valid_plates.append(([cx1 + rx1, cy1 + ry1, cx1 + rx2, cy1 + ry2], c_conf))

    else:
        # Fallback: if no vehicles were detected in the frame:
        # If the image itself is already a small/tight plate crop (e.g. uploaded plate photo),
        # allow direct plate detection on the image. Full scene frames with no vehicles
        # yield 0 plates, cleanly suppressing false positives on non-car objects.
        aspect = float(w) / max(float(h), 1.0)
        is_isolated_plate = (w < 800 and h < 500 and 1.5 <= aspect <= 7.0)
        if is_isolated_plate:
            p_boxes = _detect_plates_raw(model, image_bgr)
            for pbox, pconf in p_boxes:
                valid_plates.append((pbox, pconf))

    # Deduplicate overlapping plate boxes
    deduped = []
    valid_plates.sort(key=lambda item: item[1], reverse=True)
    for box, conf in valid_plates:
        overlap = False
        for abox, _ in deduped:
            if iou(box, abox) > 0.4:
                overlap = True
                break
        if not overlap:
            deduped.append((box, conf))

    return [item[0] for item in deduped]


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

    cbox = _find_contour_in_vehicle_crop(vehicle_crop)
    if cbox is None:
        return None
    rx1, ry1, rx2, ry2 = cbox

    confidence = _estimate_contour_confidence(cbox, vehicle_crop.shape[:2])
    if confidence < CONTOUR_CONF_THRESHOLD:
        return None

    x1, y1 = max(0, cx1 + rx1), max(0, cy1 + ry1)
    x2, y2 = min(w, cx1 + rx2), min(h, cy1 + ry2)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _detect_vehicle_bbox(image_bgr, conf_threshold=0.35):
    """Return a COCO vehicle bounding box ``[x1, y1, x2, y2]`` or None."""
    vehs = _detect_vehicles(image_bgr, conf_threshold=conf_threshold)
    return vehs[0] if vehs else None


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


def _preprocess_plate_crop(plate_bgr):
    """Advanced preprocessing for license plates (adapted from PlateVision-AI).

    Pipeline: deskew -> pad border -> 3x LANCZOS upscale -> grayscale ->
    bilateral filter -> CLAHE -> sharpen, tuned so EasyOCR reads the text
    region cleanly even on tilted or low-light crops.
    """
    # 1. Bring tilted plates onto the horizontal baseline.
    plate_bgr = _deskew_plate(plate_bgr)

    # 2. Small uniform border so characters at plate edges aren't clipped.
    plate_bgr = cv2.copyMakeBorder(
        plate_bgr, 8, 8, 8, 8, cv2.BORDER_REPLICATE,
    )

    # 3. Big 3x LANCZOS upscale: sharper than the old 2x CUBIC. For plates
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
            plate_bgr, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_LANCZOS4,
        )

    # 4. Grayscale + bilateral noise reduction.
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 75, 75)

    # 5. CLAHE (Contrast Limited Adaptive Histogram Equalization) + sharpen.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)

    result = _sharpen(contrast_enhanced)

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


def _ocr_plate_image_preferred(processed, allowlist=True):
    """Read a preprocessed plate crop, fast-plate-ocr first, EasyOCR fallback.

    fast-plate-ocr is the primary engine when available: ~100x faster than
    EasyOCR with per-character confidences. When it returns nothing (empty
    plate / low signal / model unavailable) we fall back to the EasyOCR rescue
    variant so every plate the old pipeline could read is still readable.
    ``_ocr_plate_image_rescue`` already includes the plain EasyOCR pass, so a
    single fallback call is enough.

    Returns ``(text, confidence, char_confidences)``. ``char_confidences`` is
    ``None`` on the EasyOCR path (it has no per-char scores); the FastOCR path
    returns one float per character.
    """
    fast_text, fast_conf, fast_chars = _ocr_plate_image_fast(processed, allowlist)
    if fast_text:
        return fast_text, fast_conf, fast_chars
    eas_text, eas_conf = _ocr_plate_image_rescue(processed, allowlist)
    return eas_text, eas_conf, None


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


def extract_license_plate(image_bytes_or_path):
    """Detect + OCR a license plate in a single image.

    Returns::

        {
            'plate_text': str,
            'confidence': float,
            'cropped_img': bytes,   # PNG bytes of the processed plate crop
            'bbox': [x1, y1, x2, y2] | None,
            'car_bbox': [x1, y1, x2, y2] | None,
            'all_cars': [[x1, y1, x2, y2], ...],
            'char_confidences': list | None,
        }
    """
    model = get_plate_detector()
    image_bgr = _read_frame(image_bytes_or_path)

    # 1. Detect cars / vehicles first
    vehicles = _detect_vehicles(image_bgr)

    # 2. Detect plates on cars
    contour_conf = None
    boxes = _detect_plate_boxes_many(model, image_bgr, vehicles=vehicles)
    box = boxes[0] if boxes else None

    car_bbox = None
    if box is not None and vehicles:
        px_mid = (box[0] + box[2]) / 2.0
        py_mid = (box[1] + box[3]) / 2.0
        for vx1, vy1, vx2, vy2 in vehicles:
            if vx1 <= px_mid <= vx2 and vy1 <= py_mid <= vy2:
                car_bbox = [vx1, vy1, vx2, vy2]
                break
        if car_bbox is None:
            car_bbox = vehicles[0]
    elif vehicles:
        car_bbox = vehicles[0]

    if box is None and CONTOUR_FALLBACK:
        box = _detect_plate_with_contours(image_bgr)
        if box is not None:
            contour_conf = _estimate_contour_confidence(box, image_bgr.shape[:2])

    if box is None:
        fallback_text = None
        if FULL_IMAGE_FALLBACK:
            fallback_text = _full_image_fallback(image_bgr)
        return {
            'plate_text': fallback_text or '',
            'confidence': 0.4 if fallback_text else 0.0,
            'cropped_img': b'',
            'bbox': None,
            'car_bbox': car_bbox,
            'all_cars': vehicles,
        }

    x1, y1, x2, y2 = box
    h, w = image_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return {
            'plate_text': '',
            'confidence': 0.0,
            'cropped_img': b'',
            'bbox': box,
            'car_bbox': car_bbox,
            'all_cars': vehicles,
        }

    pad = 12
    x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
    x2p, y2p = min(w, x2 + pad), min(h, y2 + pad)
    plate_bgr = image_bgr[y1p:y2p, x1p:x2p]
    processed = _preprocess_plate_crop(plate_bgr)

    raw_text, conf, char_conf = _ocr_plate_image_preferred(processed)
    if not raw_text:
        return {
            'plate_text': '',
            'confidence': 0.0,
            'cropped_img': _encode_png(processed),
            'bbox': box,
            'car_bbox': car_bbox,
            'all_cars': vehicles,
        }

    plate_text = _clean_plate_text(raw_text)

    if contour_conf is not None:
        conf = max(float(conf), contour_conf * 0.9)

    return {
        'plate_text': plate_text,
        'confidence': float(conf),
        'cropped_img': _encode_png(processed),
        'bbox': box,
        'car_bbox': car_bbox,
        'all_cars': vehicles,
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