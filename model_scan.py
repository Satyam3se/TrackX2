import logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)
import time, cv2

from fast_alpr.default_detector import DefaultDetector
from fast_plate_ocr import LicensePlateRecognizer

cap = cv2.VideoCapture('Videos/WhatsApp Video 2026-09-12 at 21.55.08.mp4')
def frame(i):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read(); return f

det = DefaultDetector(conf_thresh=0.25)
ocr = LicensePlateRecognizer(hub_ocr_model='cct-s-v2-global-model', device='auto')

for i in (192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640):
    f = frame(i)
    if f is None: continue
    t0 = time.time()
    raw = det.predict(f)
    t1 = time.time()
    roles = []
    for r in raw[:8]:
        bb = r.bounding_box
        x1, y1, x2, y2 = int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)
        crop = f[y1:y2, x1:x2]
        if crop.size == 0: continue
        try:
            res = ocr.run(crop, return_confidence=True)
        except Exception:
            continue
        if res:
            roles.append((str(res[0].plate), round(float(r.confidence), 2)))
    print('f%d det=%d t=%.3fs roles=%s' % (i, len(raw), t1-t0, roles))
cap.release()