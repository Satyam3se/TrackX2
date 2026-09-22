import json
from datetime import timedelta

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.db.models import Avg, Count
from django.http import JsonResponse
from django.shortcuts import render
from django.utils import timezone

from anpr_engine.models import BlacklistedVehicle, CameraNode, DetectionLog


# ---------------------------------------------------------------------------
# Mock intelligence data for the TrackX command-center dashboard.
# In a production system this would be served from the ANPR ingest pipeline.
# ---------------------------------------------------------------------------

# Target vehicle chronological route across the city (New Delhi-ish grid).
TRAJECTORY_NODES = [
    {
        "id": 1, "name": "CAM-042 · Ring Rd / ITO Jn", "lat": 28.6289, "lng": 77.2410,
        "time": "14:02", "speed": 46, "heading": "NE", "status": "MATCH", "confidence": 98.4,
    },
    {
        "id": 2, "name": "CAM-118 · Rajghat Flyover", "lat": 28.6412, "lng": 77.2498,
        "time": "14:07", "speed": 61, "heading": "NE", "status": "MATCH", "confidence": 97.1,
    },
    {
        "id": 3, "name": "CAM-233 · Kashmere Gate ISBT", "lat": 28.6673, "lng": 77.2295,
        "time": "14:15", "speed": 33, "heading": "NW", "status": "MATCH", "confidence": 99.0,
    },
    {
        "id": 4, "name": "CAM-501 · GT Karnal Rd", "lat": 28.6905, "lng": 77.2011,
        "time": "14:24", "speed": 72, "heading": "NW", "status": "MATCH", "confidence": 95.8,
    },
    {
        "id": 5, "name": "CAM-612 · Azadpur Mandi", "lat": 28.7078, "lng": 77.1758,
        "time": "14:33", "speed": 28, "heading": "W", "status": "LAST SEEN", "confidence": 96.3,
    },
]

# Live traffic congestion heat points (weight 0..1, higher = heavier).
HEATMAP_POINTS = [
    [28.6289, 77.2410, 0.9], [28.6350, 77.2450, 0.7], [28.6412, 77.2498, 0.85],
    [28.6540, 77.2380, 0.5], [28.6673, 77.2295, 0.95], [28.6720, 77.2200, 0.6],
    [28.6905, 77.2011, 0.4], [28.7078, 77.1758, 0.3], [28.6100, 77.2600, 0.8],
    [28.6600, 77.2600, 0.65], [28.6800, 77.2400, 0.55], [28.6200, 77.2100, 0.75],
    [28.6450, 77.2000, 0.45], [28.6950, 77.2300, 0.35], [28.7000, 77.2050, 0.5],
]

# All active camera nodes shown as ambient green markers.
CAMERA_NODES = [
    {"id": "CAM-042", "lat": 28.6289, "lng": 77.2410},
    {"id": "CAM-118", "lat": 28.6412, "lng": 77.2498},
    {"id": "CAM-233", "lat": 28.6673, "lng": 77.2295},
    {"id": "CAM-501", "lat": 28.6905, "lng": 77.2011},
    {"id": "CAM-612", "lat": 28.7078, "lng": 77.1758},
    {"id": "CAM-077", "lat": 28.6540, "lng": 77.2650},
    {"id": "CAM-091", "lat": 28.6150, "lng": 77.2250},
    {"id": "CAM-144", "lat": 28.6820, "lng": 77.2500},
    {"id": "CAM-205", "lat": 28.6350, "lng": 77.1950},
    {"id": "CAM-318", "lat": 28.7010, "lng": 77.2200},
]

# Vertical timeline built from trajectory nodes (with computed segments).
TIMELINE = []
for i, n in enumerate(TRAJECTORY_NODES):
    seg = None
    if i > 0:
        prev = TRAJECTORY_NODES[i - 1]
        seg = {"avg_speed": round((prev["speed"] + n["speed"]) / 2)}
    TIMELINE.append({**n, "segment": seg, "is_last": i == len(TRAJECTORY_NODES) - 1})

HOTLIST = [
    {
        "level": "critical", "tag": "STOLEN VEHICLE",
        "plate": "DL 3C AF 9021", "desc": "Black Toyota Fortuner · Reported stolen 2d ago",
        "cam": "CAM-233 · Kashmere Gate", "time": "14:16",
    },
    {
        "level": "critical", "tag": "BLACKLIST HIT",
        "plate": "HR 26 DQ 4412", "desc": "Silver Honda City · Watchlist: NCB-DEL",
        "cam": "CAM-118 · Rajghat Flyover", "time": "14:09",
    },
    {
        "level": "warning", "tag": "ROUTE ANOMALY",
        "plate": "UP 14 AB 1234", "desc": "Suspicious looping detected · 3 passes / 20 min",
        "cam": "CAM-042 · ITO Junction", "time": "14:02",
    },
    {
        "level": "warning", "tag": "SPEED VIOLATION",
        "plate": "DL 8C X 7788", "desc": "112 km/h in 60 zone · auto-challan queued",
        "cam": "CAM-501 · GT Karnal Rd", "time": "13:58",
    },
    {
        "level": "warning", "tag": "UNREADABLE PLATE",
        "plate": "DL ?? ?? ????", "desc": "Obscured plate · manual review flagged",
        "cam": "CAM-612 · Azadpur", "time": "13:51",
    },
]

VELOCITY = {"avg_speed": 38, "active_vehicles": 42890, "congestion_index": 64}

OD_CORRIDORS = [
    {"name": "South Ext -> Connaught Place", "volume": 92, "vph": "8.4k"},
    {"name": "Dwarka -> Cyber Hub (GGN)", "volume": 78, "vph": "7.1k"},
    {"name": "Noida Sec-62 -> ITO", "volume": 71, "vph": "6.5k"},
    {"name": "Rohini -> Kashmere Gate", "volume": 63, "vph": "5.8k"},
    {"name": "Saket -> AIIMS Ring Rd", "volume": 54, "vph": "4.9k"},
    {"name": "Airport T3 -> Aerocity", "volume": 47, "vph": "4.2k"},
]

BOTTLENECKS = [
    {"name": "ITO Junction", "drop": 68, "speed": 11, "clear": "18 min"},
    {"name": "Rajghat Flyover", "drop": 55, "speed": 17, "clear": "12 min"},
    {"name": "Kashmere Gate ISBT", "drop": 74, "speed": 8, "clear": "24 min"},
    {"name": "AIIMS Ring Rd", "drop": 41, "speed": 22, "clear": "9 min"},
    {"name": "Azadpur Chowk", "drop": 62, "speed": 14, "clear": "15 min"},
]


def index(request):
    context = {
        "cameras_online": 1420,
        "cameras_total": 1450,
        "latency": 12,
        "alert_count": 3,
        "target": {
            "plate": "UP 14 AB 1234",
            "confidence": 98.4,
            "color": "White",
            "body": "SUV",
            "model": "Hyundai Creta",
            "owner_status": "Flagged",
        },
        "timeline": TIMELINE,
        "hotlist": HOTLIST,
        "velocity": VELOCITY,
        "od_corridors": OD_CORRIDORS,
        "bottlenecks": BOTTLENECKS,
        # JSON blobs consumed by the client-side map / charts.
        "trajectory_json": json.dumps(TRAJECTORY_NODES),
        "heatmap_json": json.dumps(HEATMAP_POINTS),
        "cameras_json": json.dumps(CAMERA_NODES),
        "velocity_json": json.dumps(VELOCITY),
    }
    return render(request, "dashboard/index.html", context)


def _camera_feature_collection():
    """All camera nodes as a GeoJSON FeatureCollection (green markers)."""
    features = []
    for cam in CameraNode.objects.all():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [cam.location.x, cam.location.y],
            },
            "properties": {
                "id": cam.camera_id,
                "camera_id": cam.camera_id,
                "location_name": cam.location_name,
                "is_active": cam.is_active,
            },
        })
    return {"type": "FeatureCollection", "features": features}


def _detection_feature_collection(hours=24):
    """Recent DetectionLogs as point features (vehicle markers + popups)."""
    since = timezone.now() - timedelta(hours=hours)
    logs = DetectionLog.objects.select_related("camera").filter(
        captured_at__gte=since,
    ).order_by("-captured_at")
    features = []
    for log in logs:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [log.camera.location.x, log.camera.location.y],
            },
            "properties": {
                "id": log.id,
                "license_plate": log.license_plate,
                "location_name": log.camera.location_name,
                "camera_id": log.camera.camera_id,
                "status": "Active",
                "timestamp": log.captured_at.strftime("%H:%M"),
                "confidence": round(log.confidence_score * 100, 1),
                "speed": log.speed_estimate,
            },
        })
    return {"type": "FeatureCollection", "features": features}


def _heatmap_feature_collection():
    """Detection density per camera as heatmap points (intensity 0..1)."""
    counts = (
        DetectionLog.objects
        .values("camera_id")
        .annotate(count=Count("id"))
    )
    cam_ids = [row["camera_id"] for row in counts]
    cams = {
        c.id: c
        for c in CameraNode.objects.filter(id__in=cam_ids)
    }
    max_count = max((row["count"] for row in counts), default=1) or 1
    features = []
    for row in counts:
        cam = cams.get(row["camera_id"])
        if not cam:
            continue
        intensity = row["count"] / max_count
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [cam.location.x, cam.location.y],
            },
            "properties": {"intensity": round(intensity, 3)},
        })
    return {"type": "FeatureCollection", "features": features}


def _trajectory_feature_collection(license_plate=None):
    """Trajectory (route + waypoints) for a license plate.

    If ``license_plate`` is provided it is used directly; otherwise the most
    recently flagged blacklisted plate that has detections is used.
    """
    hot = None
    if license_plate:
        hot = (
            BlacklistedVehicle.objects
            .filter(license_plate__iexact=license_plate, is_active=True)
            .first()
        )
        if not hot:
            # Still resolve the plate even if it isn't blacklisted.
            plate_has_logs = DetectionLog.objects.filter(
                license_plate__iexact=license_plate
            ).exists()
            if not plate_has_logs:
                return {
                    "type": "FeatureCollection",
                    "license_plate": license_plate,
                    "total_hits": 0,
                    "features": [],
                }
    if hot is None:
        plates_with_logs = (
            BlacklistedVehicle.objects
            .filter(is_active=True)
            .filter(license_plate__in=DetectionLog.objects.values_list("license_plate", flat=True))
        )
        hot = plates_with_logs.order_by("-flagged_at").first()
        if not hot:
            hot = (
                BlacklistedVehicle.objects
                .filter(is_active=True)
                .order_by("-flagged_at")
                .first()
            )
        if not hot:
            return {"type": "FeatureCollection", "license_plate": None, "total_hits": 0, "features": []}

    target_plate = license_plate or hot.license_plate
    logs = list(
        DetectionLog.objects.select_related("camera")
        .filter(license_plate__iexact=target_plate)
        .order_by("captured_at")
    )
    if not logs:
        return {
            "type": "FeatureCollection",
            "license_plate": target_plate,
            "total_hits": 0,
            "features": [],
        }

    coords = [[l.camera.location.x, l.camera.location.y] for l in logs]
    waypoints = []
    for l in logs:
        waypoints.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [l.camera.location.x, l.camera.location.y],
            },
            "properties": {
                "id": l.id,
                "camera_id": l.camera.camera_id,
                "location_name": l.camera.location_name,
                "timestamp": l.captured_at.strftime("%H:%M"),
                "confidence": round(l.confidence_score * 100, 1),
                "speed": l.speed_estimate,
            },
        })

    return {
        "type": "FeatureCollection",
        "license_plate": target_plate,
        "total_hits": len(logs),
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {"source": "trackx-anpr", "total_hits": len(logs)},
            },
            *waypoints,
        ],
    }


def map_dashboard(request):
    cameras = _camera_feature_collection()
    detections = _detection_feature_collection()
    heatmap = _heatmap_feature_collection()
    plate = (request.GET.get("plate") or "").strip() or None
    trajectory = _trajectory_feature_collection(plate)
    target_plate = trajectory.get("license_plate")

    # Center the map on the mean of the real camera nodes.
    try:
        mean = CameraNode.objects.aggregate(
            avg_lon=Avg("location__x"),
            avg_lat=Avg("location__y"),
        )
        center = [mean["avg_lon"] or 77.2090, mean["avg_lat"] or 28.6139]
    except Exception:
        center = [77.2090, 28.6139]

    context = {
        "map_center": center,
        "map_zoom": 11,
        "map_pitch": 45,
        "cameras_json": cameras,
        "detections_json": detections,
        "heatmap_json": heatmap,
        "trajectory_json": trajectory,
        "target_plate": target_plate,
    }
    return render(request, "dashboard/map.html", context)


def fire_test_alert(request):
    """Broadcast a synthetic hotlist alert over the real WebSocket channel layer
    so the /map/ realtime feed can be exercised without a live ANPR camera.
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
