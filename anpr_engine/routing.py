from django.urls import re_path

from . import consumers
from . import consumers_live_video

websocket_urlpatterns = [
    re_path(r'ws/alerts/$', consumers.AlertConsumer.as_asgi()),
    re_path(r'ws/live-video/$', consumers_live_video.LiveVideoConsumer.as_asgi()),
]