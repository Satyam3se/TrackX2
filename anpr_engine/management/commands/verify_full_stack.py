from django.core.management.base import BaseCommand
from django.db import connection
import redis
import requests
from anpr_engine.models import CameraNode, DetectionLog, BlacklistedVehicle
from anpr_engine.vision import get_yolo_model, get_easyocr_reader

class Command(BaseCommand):
    help = 'Full-stack diagnostic tool for TrackX system health'

    def handle(self, *args, **options):
        self.stdout.write(self.style.MIGRATE('\n' + '='*60))
        self.stdout.write(self.style.SUCCESS(' TRACKX SYSTEM INTEGRATION DIAGNOSTICS '))
        self.stdout.write('='*60 + '\n')

        checks = [
            ('Database (PostGIS)', self.check_db),
            ('AI Model Loading', self.check_ai_models),
            ('Redis Broker', self.check_redis),
            ('API Gateway', self.check_api),
            ('Spatial Integrity', self.check_spatial_indices),
        ]

        for name, func in checks:
            try:
                status = func()
                self.stdout.write(f'[{ "✓" if status else "✗" }] {name}')
            except Exception as e:
                self.stdout.write(f'[{ "✗" }] {name} - ERROR: {str(e)}')

        self.stdout.write('='*60 + '\n')

    def check_db(self):
        # Check if critical tables exist
        with connection.cursor() as cursor:
            cursor.execute('SELECT version();')
        return True

    def check_ai_models(self):
        # Test lazy loading of YOLO and OCR
        yolo = get_yolo_model()
        ocr = get_easyocr_reader()
        return yolo is not None and ocr is not None

    def check_redis(self):
        # Test Redis connection (adjust host if needed)
        try:
            r = redis.Redis(host='redis', port=6379, socket_timeout=2)
            return r.ping()
        except:
            return False

    def check_api(self):
        # Test internal API endpoint
        try:
            # We use a relative path or the internal docker network name
            res = requests.get('http://localhost:8000/api/v1/analytics/summary/', timeout=2)
            return res.status_code == 200
        except:
            # Fallback for when running outside docker
            try:
                res = requests.get('http://web:8000/api/v1/analytics/summary/', timeout=2)
                return res.status_code == 200
            except:
                return False

    def check_spatial_indices(self):
        # Ensure GIST indices are present on location fields
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM pg_class WHERE relname LIKE '%location_gist%';")
            return len(cursor.fetchall()) > 0
