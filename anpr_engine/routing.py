from django.urls import path

from . import consumers
from . import consumers_live_video

websocket_urlpatterns = [
    path('ws/alerts/', consumers.AlertConsumer.as_asgi()),
    path('ws/live-video/', consumers_live_video.LiveVideoConsumer.as_asgi()),
]