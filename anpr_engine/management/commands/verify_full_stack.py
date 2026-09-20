from django.core.management.base import BaseCommand
from django.db import connection
import os
import redis
import requests
from anpr_engine.models import CameraNode, DetectionLog, BlacklistedVehicle
from anpr_engine.vision import (
    get_plate_detector, 
    get_easyocr_reader, 
    resolve_detector_path,
    get_fast_plate_ocr,
    get_fast_alpr
)

class Command(BaseCommand):
    help = 'Full-stack diagnostic tool for TrackX system health'

    def handle(self, *args, **options):
        self.stdout.write(self.style.MIGRATE('\n' + '='*60))
        self.stdout.write(self.style.SUCCESS(' TRACKX SYSTEM INTEGRATION DIAGNOSTICS '))
        self.stdout.write('='*60 + '\n')

        checks = [
            ('Database (PostGIS)', self.check_db),
            ('AI Model Resolution', self.check_ai_models),
            ('Redis Broker', self.check_redis),
            ('API Gateway', self.check_api),
            ('Spatial Integrity', self.check_spatial_indices),
        ]

        for name, func in checks:
            try:
                status = func()
                if isinstance(status, tuple):
                    ok, msg = status
                    self.stdout.write(f'[{ "✓" if ok else "✗" }] {name}: {msg}')
                else:
                    self.stdout.write(f'[{ "✓" if status else "✗" }] {name}')
            except Exception as e:
                self.stdout.write(f'[{ "✗" }] {name} - ERROR: {str(e)}')

        self.stdout.write('='*60 + '\n')

    def check_db(self):
        with connection.cursor() as cursor:
            cursor.execute('SELECT version();')
        return True

    def check_ai_models(self):
        # Report the resolved detector path
        active_weights = resolve_detector_path()
        detector = get_plate_detector()
        ocr = get_easyocr_reader()
        
        # Check optional fast paths
        has_fast_ocr = get_fast_plate_ocr() is not None
        has_fast_alpr = get_fast_alpr() is not None
        
        info = f"Weights={active_weights} | FastOCR={has_fast_ocr} | FastALPR={has_fast_alpr}"
        ok = detector is not None and ocr is not None
        return ok, info

    def check_redis(self):
        url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
        try:
            r = redis.from_url(url, socket_timeout=2)
            return r.ping(), f"URL={url}"
        except:
            return False

    def check_api(self):
        for host in ['localhost', 'web', '127.0.0.1']:
            url = f'http://{host}:8000/api/v1/analytics/summary/'
            try:
                res = requests.get(url, timeout=1)
                if res.status_code == 200:
                    return True, f"OK via {host}"
            except:
                continue
        return False

    def check_spatial_indices(self):
        with connection.cursor() as cursor:
            # We look for GIST indices on any table
            cursor.execute("SELECT relname FROM pg_class WHERE relname LIKE '%_location_%' AND relname LIKE '%_gist%';")
            rows = cursor.fetchall()
            return len(rows) > 0, f"Found {len(rows)} indices"
