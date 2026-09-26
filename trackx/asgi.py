import os

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.security.websocket import AllowedHostsOriginValidator
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trackx.settings')

# Initialize Django ASGI application early to ensure AppRegistry is populated
# before importing code that may import ORM models.
django_asgi_app = get_asgi_application()

import anpr_engine.routing

# Preload the ANPR models into THIS process before Daphne starts serving.
# Without this, the first live-video frame stalls the event loop for ~30-60s
# while the models load, ping timeouts fire, and the WebSocket is killed.
try:
    from anpr_engine.vision import get_fast_plate_ocr, get_plate_detector
    get_plate_detector()
    get_fast_plate_ocr()
    print('[warmup] ANPR models preloaded in-process')
except Exception as exc:  # pragma: no cover - never fatal
    print(f'[warmup] ANPR model preload skipped: {exc}')

application = ProtocolTypeRouter({
    'http': django_asgi_app,
    'websocket': AllowedHostsOriginValidator(
        AuthMiddlewareStack(
            URLRouter(anpr_engine.routing.websocket_urlpatterns)
        )
    ),
})