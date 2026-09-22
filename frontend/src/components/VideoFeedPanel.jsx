import { useCallback, useEffect, useRef, useState } from 'react';
import {
  createVideoFeed,
  getVideoFeeds,
  processVideoFeed,
  getVideoFeedDetections,
} from '../services/api';

export default function VideoFeedPanel({ cameras, onAlert }) {
  const [feeds, setFeeds] = useState([]);
  const [selectedCameraId, setSelectedCameraId] = useState('');
  const [title, setTitle] = useState('');
  const [file, setFile] = useState(null);
  const [processingId, setProcessingId] = useState(null);
  const [fetchedDetections, setFetchedDetections] = useState({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState({}); // Store progress per feed { feedId: percentage }
  const timerRef = useRef(null);

  /* camera list derived from the GeoJSON FeatureCollection */
  const cameraOptions =
    cameras?.features?.map((f) => ({
      id: f.id,
      camera_id: f.properties?.camera_id,
      location_name: f.properties?.location_name,
    })) ?? [];

  const refresh = useCallback(async () => {
    try {
      const data = await getVideoFeeds();
      setFeeds(data || []);
    } catch (err) {
      console.error('Failed to refresh feeds:', err);
    }
  }, []);

  useEffect(() => {
    refresh();
    return () => {
        if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [refresh]);

  /* poll feed list and progress updates */
  useEffect(() => {
    timerRef.current = setInterval(async () => {
      if (processingId) {
        await refresh();
      }
    }, 8000);
    return () => {
        if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [processingId, refresh]);

  const handleSubmit = useCallback(
    async (e) => {
      e.preventDefault();
      if (!file || !selectedCameraId || !title.trim()) return;
      setBusy(true);
      setError(null);
      try {
        await createVideoFeed(Number(selectedCameraId), title.trim(), file);
        setFile(null);
        setTitle('');
        setSelectedCameraId('');
        if (e.target.querySelector('input[type=file]')) {
            e.target.querySelector('input[type=file]').value = '';
        }
        await refresh();
      } catch (err) {
        setError(err.message ?? 'Upload failed.');
      } finally {
        setBusy(false);
      }
    },
    [file, selectedCameraId, title, refresh],
  );

  const handleProcess = useCallback(
    async (feed) => {
      setProcessingId(feed.id);
      setError(null);
      try {
        await processVideoFeed(feed.id, 5);
        onAlert?.(`Queued ${feed.title} for YOLOv8+EasyOCR processing`);
      } catch (err) {
        setError(err.message ?? 'Could not queue processing.');
      }
    },
    [onAlert],
  );

  const handleShowDetections = useCallback(
    async (feed) => {
      if (fetchedDetections[feed.id]) {
        const next = { ...fetchedDetections };
        delete next[feed.id];
        setFetchedDetections(next);
        return;
      }
      try {
        const dets = await getVideoFeedDetections(feed.id);
        setFetchedDetections((prev) => ({ ...prev, [feed.id]: dets }));
      } catch (err) {
        setError(err.message ?? 'Could not load detections.');
      }
    },
    [fetchedDetections],
  );

  return (
    <section className="card" style={{ border: '2px solid #00e676', marginBottom: '20px' }}>
      <div className="card-head">
        <div className="card-title">
          <span className="bar" style={{ background: '#00e676' }} />
          Video Feeds
        </div>
        <span className="card-sub">{feeds.length} uploaded</span>
      </div>

      <form className="vf-form" onSubmit={handleSubmit}>
        <div className="vf-fields">
          <select
            className="vf-select"
            value={selectedCameraId}
            onChange={(e) => setSelectedCameraId(e.target.value)}
            aria-label="Camera"
          >
            <option value="">Camera…</option>
            {cameraOptions.map((c) => (
              <option key={c.id} value={c.id}>
                {c.camera_id} — {c.location_name}
              </option>
            ))}
          </select>
          <input
            className="vf-input"
            placeholder="Title (e.g. North gate 14:00)"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            aria-label="Feed title"
          />
          <input
            className="vf-file"
            type="file"
            accept="video/*,.mp4,.avi,.mov,.mkv"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            aria-label="Video file"
          />
          <button
            type="submit"
            className="vf-upload"
            disabled={busy || !file || !selectedCameraId || !title.trim()}
          >
            {busy ? 'UPLOADING…' : 'UPLOAD'}
          </button>
        </div>
        {error && <div className="vf-error" style={{ color: 'red', fontSize: '12px', marginTop: '5px' }}>{error}</div>}
      </form>

      {feeds.length === 0 && !processingId && (
        <div className="empty">No video feeds yet. Upload one above.</div>
      )}

      {feeds.map((feed) => {
        const detections = fetchedDetections[feed.id];
        const processing = processingId === feed.id;
        const feedProgress = progress[feed.id] || 0;

        return (
          <div className="vf-item" key={feed.id}>
            <div className="vf-top">
              <span className="vf-title">{feed.title}</span>
              <span
                className={`vf-badge ${feed.processed ? 'ok' : 'pending'}`}
              >
                {processing ? `PROCESSING ${feedProgress}%` : feed.processed ? 'PROCESSED' : 'UNPROCESSED'}
              </span>
            </div>
            <div className="vf-meta">
              {feed.camera?.camera_id || 'Unknown'} &middot; {feed.camera?.location_name || 'Unknown'} &middot;{' '}
              {feed.uploaded_at ? new Date(feed.uploaded_at).toLocaleString() : 'N/A'}
            </div>
            <div className="vf-actions">
              {!feed.processed && (
                <button
                  className="mini-btn process"
                  onClick={() => handleProcess(feed)}
                  disabled={processing}
                >
                  {processing ? 'QUEUED' : 'RUN YOLO OCR'}
                </button>
              )}
              <button
                className="mini-btn"
                onClick={() => handleShowDetections(feed)}
              >
                {detections ? `HIDE ${detections.length}` : 'DETECTIONS'}
              </button>
            </div>

            {detections && (
              <div className="vf-dets">
                {detections.length === 0 && (
                  <div className="empty" style={{ padding: '8px' }}>
                    No detections in this feed.
                  </div>
                )}
                {detections.map((d) => (
                  <div className="vf-det" key={d.id}>
                    <span className="vf-plate">{d.license_plate}</span>
                    <span className="vf-conf">
                      OCR {Math.round((d.confidence_score || 0) * 100)}%
                    </span>
                    <span className="vf-ts">{d.location_name}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </section>
  );
}
