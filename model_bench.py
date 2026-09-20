import logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)
import time, cv2, numpy as np

from fast_alpr.default_detector import DefaultDetector
from fast_plate_ocr import LicensePlateRecognizer

cap = cv2.VideoCapture('Videos/WhatsApp Video 2026-09-12 at 21.55.08.mp4')
def frame(i):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read(); return f

det = DefaultDetector(conf_thresh=0.25)
ocr = LicensePlateRecognizer(hub_ocr_model='cct-s-v2-global-model', device='auto')

for i in (256, 288, 320, 352, 384, 416, 480):
    f = frame(i)
    if f is None: continue
    raw = det.predict(f)
    print('f%d detections=%d' % (i, len(raw)))
    for r in raw[:6]:
        bb = r.bounding_box
        x1, y1, x2, y2 = int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)
        crop = f[y1:y2, x1:x2]
        try:
            res = ocr.run(crop, return_confidence=True)
        except Exception as e:
            print('  OCR ERR:', type(e).__name__, str(e)[:120]); continue
        if not res:
            continue
        p = res[0]
        try:
            cp = getattr(p, 'char_probs', None)
            cpl = [round(float(c), 2) for c in np.atleast_1d(cp)] if cp is not None and np.size(cp) else []
        except Exception:
            cpl = []
        print('  bbox', (x1, y1, x2, y2), 'det_conf=%.3f' % float(r.confidence),
              'ocr=', repr(str(p.plate)), 'char_conf=', cpl)
cap.release()