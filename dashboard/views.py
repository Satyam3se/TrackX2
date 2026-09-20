import json
from urllib.parse import quote

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.conf import settings
from django.http import HttpResponseRedirect, JsonResponse
from django.utils import timezone

from anpr_engine.models import BlacklistedVehicle, CameraNode


def index(request):
    """Single-UI handoff.

    The React SPA (``frontend/``) is the one and only TrackX command center.
    The Django-rendered prototype dashboards have been retired: visiting any
    of their routes now redirects to the SPA so there is exactly one UI and
    one origin, regardless of which port the app is opened on.
    """
    return HttpResponseRedirect(settings.REACT_APP_URL)


def map_dashboard(request):
    """Redirect the old template map to the SPA, preserving the plate query."""
    plate = (request.GET.get("plate") or "").strip()
    url = settings.REACT_APP_URL.rstrip("/") + "/"
    if plate:
        url = f"{url}?plate={quote(plate)}"
    return HttpResponseRedirect(url)


def fire_test_alert(request):
    """Broadcast a synthetic hotlist alert over the real WebSocket channel layer
    so the SPA's realtime alert feed can be exercised without a live ANPR
    camera. Trigger it from a browser or with curl, e.g.

        curl "http://localhost:8000/map/fire-alert/?plate=UP%2014%20AB%201234"
    """
    plate = (request.GET.get("plate") or "").strip() or "DL 3C AF 9021"
    camera_id = (request.GET.get("camera") or "").strip() or "CAM-042"

    camera = CameraNode.objects.filter(camera_id=camera_id).first()
    if not camera:
        return JsonResponse({"error": f"No camera {camera_id}"}, status=400)

    vehicle = BlacklistedVehicle.objects.filter(
        license_plate__iexact=plate
    ).first()

    if vehicle:
        payload = {
            "type": "send_alert_notification",
            "alert_level": vehicle.alert_level,
            "plate": vehicle.license_plate,
            "owner": vehicle.owner_name,
            "reason": vehicle.reason,
            "camera": camera.location_name,
            "coordinates": [camera.location.x, camera.location.y],
            "timestamp": str(timezone.now()),
        }
    else:
        payload = {
            "type": "send_alert_notification",
            "alert_level": "INFO",
            "plate": plate,
            "owner": "Unregistered plate",
            "reason": "Simulated realtime alert",
            "camera": camera.location_name,
            "coordinates": [camera.location.x, camera.location.y],
            "timestamp": str(timezone.now()),
        }

    channel_layer = get_channel_layer()
    if not channel_layer:
        return JsonResponse({"error": "Channel layer unavailable"}, status=500)

    async_to_sync(channel_layer.group_send)("surveillance_alerts", payload)
    return JsonResponse({"broadcast": True, "payload": payload})


def api_map_payload(request):
    """Backward-compatible GeoJSON payload for any legacy consumer that still
    calls the old template map's JSON endpoint directly."""
    features = []
    for cam in CameraNode.objects.all():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [cam.location.x, cam.location.y],
            },
            "properties": {
                "camera_id": cam.camera_id,
                "location_name": cam.location_name,
                "is_active": cam.is_active,
            },
        })
    return JsonResponse({"cameras": {"type": "FeatureCollection", "features": features}})