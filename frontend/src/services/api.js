// Same-origin relative base is the default (works behind the Docker Nginx
// reverse proxy). Override for local dev via VITE_API_BASE_URL, e.g.
//   VITE_API_BASE_URL=http://localhost:8000/api/v1
const BASE =
  import.meta.env.VITE_API_BASE_URL ??
  (import.meta.env.PROD ? '/api/v1' : 'http://localhost:8000/api/v1');

async function request(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* non-JSON error body – use status text */
    }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

/**
 * Fetch all blacklisted vehicles.
 * @returns {Promise<Array>} List of BlacklistedVehicle records.
 */
export function getBlacklistedVehicles() {
  return request('/blacklisted-vehicles/');
}

/**
 * Add a vehicle to the blacklist.
 * @param {Object} data { license_plate, owner_name, reason, alert_level }
 * @returns {Promise<Object>} The created record.
 */
export async function addBlacklistedVehicle(data) {
  const res = await fetch(`${BASE}/blacklisted-vehicles/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error('Failed to add vehicle to hotlist');
  return res.json();
}

/**
 * Remove a vehicle from the blacklist.
 * @param {number} id Primary key.
 * @returns {Promise<void>}
 */
export async function deleteBlacklistedVehicle(id) {
  const res = await fetch(`${BASE}/blacklisted-vehicles/${id}/`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error('Failed to delete vehicle');
}

/**
 * Fetch all detection logs as a GeoJSON FeatureCollection for heatmap rendering.
 * @returns {Promise<Object>} GeoJSON FeatureCollection.
 */
export function getAllDetectionsGeoJSON() {
  return request('/analytics/detections-geojson/');
}

/**
 * Fetch GeoJSON trajectory path for a target license plate.
 * @param {string}  plateNumber
 * @param {string=} startDate  ISO-8601
 * @param {string=} endDate    ISO-8601
 * @returns {Promise<Object>}  GeoJSON FeatureCollection with LineString + waypoints
 */
export function getVehicleTrajectory(plateNumber, startDate, endDate) {
  const params = new URLSearchParams();
  if (startDate) params.set('start_date', startDate);
  if (endDate) params.set('end_date', endDate);
  const qs = params.toString();
  return request(
    `/trajectory/${encodeURIComponent(plateNumber)}${qs ? `?${qs}` : ''}`,
  );
}

/**
 * Fetch macro city traffic velocity + active camera counts.
 * @returns {Promise<Object>} { total_cameras, active_cameras, detections_today, active_blacklisted_count, avg_speed_kmh }
 */
export function getAnalyticsSummary() {
  return request('/analytics/summary/');
}

/**
 * Fetch all registered CCTV nodes for map pin rendering.
 * @returns {Promise<Object>} GeoJSON FeatureCollection with Point geometry per camera.
 */
export function getCameraNodes() {
  return request('/cameras/');
}

/**
 * Fetch all uploaded camera video feeds (newest first).
 * @returns {Promise<Array>} List of CameraVideoFeed records.
 */
export function getVideoFeeds() {
  return request('/video-feeds/');
}

/**
 * Upload a video feed and associate it with a camera.
 * @param {number} cameraId CameraNode primary key.
 * @param {string} title    Display title.
 * @param {File}   file     The video file.
 * @returns {Promise<Object>} The created CameraVideoFeed.
 */
export async function createVideoFeed(cameraId, title, file) {
  const formData = new FormData();
  formData.append('camera', String(cameraId));
  formData.append('title', title);
  formData.append('video_file', file);
  const res = await fetch(`${BASE}/video-feeds/`, { method: 'POST', body: formData });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = Object.values(body || {}).flat().join(', ') || detail;
    } catch {}
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

/**
 * Trigger async YOLOv8+EasyOCR processing for a video feed.
 * @param {number} feedId CameraVideoFeed id.
 * @param {number} sampleRate Process every Nth frame.
 * @returns {Promise<Object>} { video_feed_id, sample_rate, task_id, status }
 */
export async function processVideoFeed(feedId, sampleRate = 5) {
  const res = await fetch(`${BASE}/video-feeds/${feedId}/process_video/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sample_rate: sampleRate }),
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try { const b = await res.json(); detail = b?.detail ?? detail; } catch {}
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

/**
 * Fetch detection logs produced from a given video feed.
 * @param {number} feedId CameraVideoFeed id.
 * @returns {Promise<Array>} List of DetectionLogs for this feed.
 */
export function getVideoFeedDetections(feedId) {
  return request(`/video-feeds/${feedId}/detections/`);
}
