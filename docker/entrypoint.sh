#!/bin/sh
# ============================================================================
# TrackX — Web entrypoint
# Runs migrations + idempotent seed, then launches Daphne (ASGI).
# ============================================================================
set -e

echo "[entrypoint] Collecting static files..."
python manage.py collectstatic --noinput || echo "[entrypoint] collectstatic warning (non-fatal)"

echo "[entrypoint] Applying database migrations..."
python manage.py migrate --noinput

echo "[entrypoint] Seeding initial data (idempotent)..."
python manage.py seed_trackx

echo "[entrypoint] Warming up ANPR models (detector + OCR)..."
python -c "
import os, sys
sys.path.insert(0, '/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trackx.settings')
import django
django.setup()
import numpy as np
from anpr_engine.vision import extract_license_plate
extract_license_plate(np.zeros((400, 640, 3), dtype=np.uint8))
print('[warmup] ANPR detector + OCR models loaded')
"

echo "[entrypoint] Starting Daphne ASGI server on 0.0.0.0:8000..."
# Raise WebSocket frame/message limits above Daphne's 1 MiB default: the Live
# Video client posts JPEG frames as base64 data URLs (~1.2 MB after encoding),
# so the default aborts the socket as soon as a full-size frame arrives.
exec daphne -b 0.0.0.0 -p 8000 --ping-interval 45 --ping-timeout 120 \
    --websocket-max-message-size 16777216 \
    --websocket-max-frame-size 16777216 \
    trackx.asgi:application
