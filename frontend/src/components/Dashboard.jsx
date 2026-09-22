import { useCallback, useEffect, useRef, useState } from 'react';
import SurveillanceMap from './SurveillanceMap';
import VideoFeedPanel from './VideoFeedPanel';
import HotlistManager from './HotlistManager';
import {
  getVehicleTrajectory,
  getAnalyticsSummary,
  getCameraNodes,
} from '../services/api';
import useWebSocketAlerts from '../hooks/useWebSocketAlerts';

/* ------------------------------------------------------------------ */
/*  Dashboard                                                           */
/* ------------------------------------------------------------------ */

export default function Dashboard() {
  /* ---- search state ---- */
  const [plateInput, setPlateInput] = useState('');
  const [query, setQuery] = useState('');
  const [trajectory, setTrajectory] = useState(null);
  const [cameras, setCameras] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [pinnedAlert, setPinnedAlert] = useState(null);
  const [toast, setToast] = useState(null);
  const [isHotlistOpen, setIsHotlistOpen] = useState(false);
  const toastTimer = useRef(null);

  /* ---- audio toggle (persisted to localStorage) ---- */
  const [audioEnabled, setAudioEnabled] = useState(() => {
    try {
      return localStorage.getItem('trackx_audio') !== 'false';
    } catch {
      return true;
    }
  });

  const toggleAudio = useCallback(() => {
    setAudioEnabled((prev) => {
      const next = !prev;
      try {
        localStorage.setItem('trackx_audio', String(next));
      } catch {}
      return next;
    });
  }, []);

  /* ---- WebSocket alerts ---- */
  const { alerts, connected, clearAlerts, registerAlertHandler } =
    useWebSocketAlerts();
  const [videoProgress, setVideoProgress] = useState({}); // { feedId: percentage }

  const showToast = useCallback(
    (msg) => {
      setToast(msg);
      clearTimeout(toastTimer.current);
      toastTimer.current = setTimeout(() => setToast(null), 6000);
    },
    [],
  );

  useEffect(() => {
    registerAlertHandler((payload) => {
      // Handle Progress Updates
      if (payload.type === 'send_progress_update') {
        setVideoProgress((prev) => ({
          ...prev,
          [payload.video_feed_id]: payload.progress,
        }));
        return;
      }

      // Handle Regular Alerts
      showToast(
        `ALERT ${payload.alert_level}: ${payload.plate} at ${payload.camera}`,
      );
    });
  }, [registerAlertHandler, showToast]);

  /* ---- initial data fetches ---- */
  useEffect(() => {
    getCameraNodes().then(setCameras).catch(() => {});
    getAnalyticsSummary().then(setAnalytics).catch(() => {});
  }, []);

  /* ---- search handler ---- */
  const handleSearch = useCallback(
    async (e) => {
      e.preventDefault();
      const plate = plateInput.trim();
      if (!plate) return;

      setQuery(plate);
      setLoading(true);
      setError(null);

      try {
        const data = await getVehicleTrajectory(plate);
        if (data.total_hits === 0) {
          setTrajectory(null);
          setError(`No detections found for plate "${plate}".`);
        } else {
          setTrajectory(data);
          setError(null);
        }
      } catch (err) {
        setTrajectory(null);
        setError(err.message ?? 'Trajectory request failed.');
      } finally {
        setLoading(false);
      }
    },
    [plateInput],
  );

  /* ---- live clock ---- */
  const [clock, setClock] = useState(() => new Date().toLocaleTimeString());
  useEffect(() => {
    const id = setInterval(() => setClock(new Date().toLocaleTimeString()), 1000);
    return () => clearInterval(id);
  }, []);

  const waypoints =
    trajectory?.features?.filter((f) => f.geometry?.type === 'Point') ?? [];

  /* ================================================================ */
  /*  RENDER                                                            */
  /* ================================================================ */

  return (
    <div className="app">
      {/* ==================== HEADER ==================== */}
      <header className="header">
        <div className="brand">
          <div className="logo">
            <svg
              width="18" height="18" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="2.2"
              strokeLinecap="round" strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="3" />
              <path d="M12 2v3M12 19v3M2 12h3M19 12h3" />
              <path d="M4.9 4.9l2.1 2.1M17 17l2.1 2.1M19.1 4.9L17 7M7 17l-2.1 2.1" />
            </svg>
          </div>
          <div className="brand-text">
            <div className="brand-name">
              Track<b>X</b>
            </div>
            <div className="sys-status">
              <span className={`dot ${connected ? 'dot-green' : 'dot-red'}`} />
              WS {connected ? 'CONNECTED' : 'RECONNECTING'}&nbsp;&middot;&nbsp;ALERTS{' '}
              {alerts.length}
            </div>
          </div>
        </div>

        <form className="search-wrap" onSubmit={handleSearch} role="search">
          <div className="search">
            <svg
              width="16" height="16" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="2"
              strokeLinecap="round" strokeLinejoin="round"
            >
              <circle cx="11" cy="11" r="7" />
              <path d="m21 21-4.3-4.3" />
            </svg>
            <input
              type="text"
              placeholder="Search license plate (e.g. KDA 123A)…"
              value={plateInput}
              onChange={(e) => setPlateInput(e.target.value)}
              aria-label="License plate search"
            />
            <button type="submit" className="search-btn">
              SEARCH
            </button>
          </div>
        </form>

        <div className="hdr-right">
          <button 
            className="nav-link" 
            style={{ 
              background: 'none', border: 'none', cursor: 'pointer', 
              display: 'flex', alignItems: 'center', gap: '6px', 
              color: 'inherit', fontSize: 'inherit', fontFamily: 'inherit' 
            }}
            onClick={() => setIsHotlistOpen(true)}
            title="Manage Blacklist"
          >
            <svg
              width="16" height="16" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="2"
              strokeLinecap="round" strokeLinejoin="round"
            >
              <path d="M12 20h9" /><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L3 4a2.121 2.121 0 0 1 3-3z" />
            </svg>
            HOTLIST
          </button>
          <a
            className="nav-link"
            href={`/map/${(query || plateInput).trim() ? `?plate=${encodeURIComponent((query || plateInput).trim())}` : ''}`}
            title="Open live MapLibre map with the searched plate"
          >
            <svg
              width="16" height="16" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="2"
              strokeLinecap="round" strokeLinejoin="round"
            >
              <path d="M9 20l-5.5 2V5L9 3l6 2 5.5-2v17L15 21l-6-2z" />
              <path d="M9 3v17M15 5v17" />
            </svg>
            LIVE MAP
          </a>
          <div className="clock mono">{clock}</div>
          <button
            className={`icon-btn alarm ${audioEnabled ? 'active' : ''}`}
            onClick={toggleAudio}
            title="Toggle audio alarm"
            aria-label="Toggle audio alarm"
          >
            <svg
              width="17" height="17" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="2"
              strokeLinecap="round" strokeLinejoin="round"
            >
              <path d="M11 5 6 9H2v6h4l5 4V5z" />
              {audioEnabled && <path d="M15.5 8.5a5 5 0 0 1 0 7" />}
            </svg>
          </button>
          <div className="operator">
            <span className="avatar">RS</span>
            <div>
              <div className="op-name">Ops. R. Sharma</div>
              <div className="op-role">Command Center &middot; L3</div>
            </div>
          </div>
        </div>
      </header>

      {/* ==================== BODY ==================== */}
      <div className="body">
        {/* ---------- LEFT SIDEBAR ---------- */}
        <aside className="col left">
          {/* --- Video Feeds: upload + YOLOv8/EasyOCR processing --- */}
          <VideoFeedPanel 
            cameras={cameras} 
            onAlert={showToast} 
            progress={videoProgress} 
          />

          {/* --- Target Trajectory Breakdown --- */}
          <section className="card">
            <div className="card-head">
              <div className="card-title">
                <span className="bar" />
                Target Trajectory
              </div>
              {trajectory && (
                <span className="card-sub">
                  {trajectory.total_hits} HITS &middot;{' '}
                  {trajectory.summary?.distance_km} KM
                </span>
              )}
            </div>

            {query && (
              <div className="plate-crop">
                <div className="plate-big">{query}</div>
              </div>
            )}

            {trajectory?.time_span && (
              <div className="ocr-line">
                <span className="ocr-plate">{trajectory.license_plate}</span>
                <span className="conf">
                  {trajectory.summary?.avg_speed_kmh ?? '\u2014'} km/h avg
                </span>
              </div>
            )}

            {loading && <div className="empty">Loading trajectory…</div>}
            {error && <div className="empty error">{error}</div>}

            <div className="timeline">
              {waypoints.map((wp, i) => (
                <div
                  key={wp.properties.id ?? i}
                  className={`tl-item ${i === waypoints.length - 1 ? 'last' : ''}`}
                >
                  <span className="tl-node" />
                  <div className="tl-top">
                    <span className="tl-time">
                      {new Date(wp.properties.timestamp).toLocaleString()}
                    </span>
                    <span className="tl-conf">
                      OCR {Math.round(wp.properties.confidence_score ?? 0)}%
                    </span>
                  </div>
                  <div className="tl-name">{wp.properties.location_name}</div>
                  <div className="tl-seg">
                    {wp.properties.segment_speed_kmh != null ? (
                      <>
                        Inter-node avg speed{' '}
                        <b>{wp.properties.segment_speed_kmh} km/h</b>
                      </>
                    ) : (
                      'Origin detection'
                    )}
                  </div>
                </div>
              ))}
            </div>

            {!query && !loading && (
              <div className="empty">
                Search a plate above to view the trajectory timeline.
              </div>
            )}
          </section>

          {/* --- Real-Time Hotlist & Anomaly Feed --- */}
          <section className="card">
            <div className="card-head">
              <div className="card-title crimson">
                <span className="bar" />
                Hotlist &amp; Anomaly Feed
              </div>
              <span className="card-sub">LIVE</span>
            </div>

            {alerts.length === 0 && (
              <div className="empty">No active alerts.</div>
            )}

            {alerts.map((a) => (
              <div key={a.id} className={`alert ${a.alert_level}`}>
                <div className="alert-top">
                  <span className="alert-tag">{a.alert_level}</span>
                  <span className="alert-time">
                    {new Date(a.receivedAt).toLocaleTimeString()}
                  </span>
                </div>
                <div className="alert-plate">{a.plate}</div>
                <div className="alert-desc">{a.reason}</div>
                <div className="alert-cam">{a.camera}</div>
                <div className="alert-actions">
                  <button
                    className="mini-btn pin"
                    onClick={() =>
                      setPinnedAlert({
                        coordinates: a.coordinates,
                        plate: a.plate,
                      })
                    }
                  >
                    PIN TO MAP
                  </button>
                </div>
              </div>
            ))}
          </section>
        </aside>

        {/* ---------- CENTER MAP ---------- */}
        <main className="map-col">
          <div
            id="map"
            role="application"
            aria-label="GIS surveillance map"
            style={{ width: '100%', height: '100%' }}
          >
            <SurveillanceMap
              trajectory={trajectory}
              cameras={cameras}
              pinnedAlert={pinnedAlert}
            />
          </div>

          <div className="map-overlay-top">
            <div className="map-title-badge">
              <span className={`dot ${connected ? 'dot-green' : 'dot-red'}`} />
              <div>
                <div className="t">
                  GIS SURVEILLANCE GRID
                  {trajectory ? ` \u00B7 ${trajectory.license_plate}` : ''}
                </div>
                <div className="s">
                  {trajectory
                    ? `${trajectory.total_hits} node hits`
                    : 'Select a target'}
                </div>
              </div>
            </div>
          </div>

          <div className="map-legend">
            <div className="lh">Traffic Density</div>
            <div className="legend-row">
              <span className="swatch" style={{ background: '#00e676' }} />{' '}
              Smooth flow
            </div>
            <div className="legend-row">
              <span className="swatch" style={{ background: '#ffab00' }} />{' '}
              Moderate
            </div>
            <div className="legend-row">
              <span className="swatch" style={{ background: '#ff1744' }} />{' '}
              Heavy congestion
            </div>
            <div className="legend-row" style={{ marginTop: 7 }}>
              <span className="swatch" style={{ background: '#00e5ff' }} />{' '}
              Target trajectory
            </div>
          </div>
        </main>

        {/* ---------- RIGHT SIDEBAR ---------- */}
        <aside className="col right">
          <section className="card">
            <div className="card-head">
              <div className="card-title">
                <span className="bar" />
                City Traffic Analytics
              </div>
              <span className="card-sub">REALTIME</span>
            </div>

            <div className="gauges">
              <div className="gauge">
                <div className="gauge-ring" style={{ borderColor: '#00e5ff' }}>
                  <span className="g-val" style={{ color: '#00e5ff' }}>
                    {analytics?.avg_speed_kmh ?? '\u2014'}
                  </span>
                  <span className="g-unit">km/h</span>
                </div>
                <div className="g-label">Avg City Speed</div>
              </div>

              <div className="gauge">
                <div className="gauge-ring" style={{ borderColor: '#00e676' }}>
                  <span className="g-val" style={{ color: '#00e676' }}>
                    {analytics?.active_cameras ?? '\u2014'}/
                    {analytics?.total_cameras ?? '\u2014'}
                  </span>
                  <span className="g-unit">online</span>
                </div>
                <div className="g-label">Camera Nodes</div>
              </div>

              <div className="gauge">
                <div className="gauge-ring" style={{ borderColor: '#ffab00' }}>
                  <span className="g-val" style={{ color: '#ffab00' }}>
                    {analytics?.active_blacklisted_count ?? '\u2014'}
                  </span>
                  <span className="g-unit">active</span>
                </div>
                <div className="g-label">Blacklisted</div>
              </div>
            </div>

            <div className="stat-grid">
              <div className="stat">
                <span className="stat-val">
                  {analytics?.detections_today ?? '\u2014'}
                </span>
                <span className="stat-label">Detections Today</span>
              </div>
              <div className="stat">
                <span className="stat-val">
                  {analytics?.total_cameras ?? '\u2014'}
                </span>
                <span className="stat-label">Total Cameras</span>
              </div>
            </div>
          </section>

          {pinnedAlert && (
            <section className="card">
              <div className="card-head">
                <div className="card-title amber">
                  <span className="bar" />
                  Pinned Alert
                </div>
              </div>
              <div className="pinned-info">
                <div className="alert-plate">{pinnedAlert.plate}</div>
                <div className="pinned-coords">
                  [{pinnedAlert.coordinates?.join(', ')}]
                </div>
                <button
                  className="mini-btn"
                  onClick={() => setPinnedAlert(null)}
                >
                  CLEAR
                </button>
              </div>
            </section>
          )}
        </aside>
      </div>

      {/* ==================== TOAST ==================== */}
      {toast && (
        <div className="toast" role="alert">
          <div className="toast-inner">{toast}</div>
        </div>
      )}

      <HotlistManager 
        isOpen={isHotlistOpen} 
        onClose={() => setIsHotlistOpen(false)} 
        onUpdate={() => {
          getAnalyticsSummary().then(setAnalytics).catch(() => {});
        }}
      />
    </div>
  );
}
