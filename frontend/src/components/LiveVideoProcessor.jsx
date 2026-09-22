import React, { useState, useRef, useEffect, useCallback } from 'react';
// Same-origin relative path by default: proxied by Nginx (prod, :3000) and by
// Vite (dev, :5173 -> :8000), so no port/URL hardcoding is needed. Override
// only when the backend truly lives on another host.
const WS_LIVE_URL = import.meta.env.VITE_WS_LIVE_URL ?? '/ws/live-video/';
const MAX_RETRY_DELAY = 15_000;
export default function LiveVideoProcessor() {
  const [videoFile, setVideoFile] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [plateInfo, setPlateInfo] = useState(null);
  const [videoSize, setVideoSize] = useState(800);
  const isProcessingRef = useRef(false);
  const wsRef = useRef(null);
  const retryCountRef = useRef(0);
  const retryTimerRef = useRef(null);
  const disposedRef = useRef(false);
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const videoRef = useRef(null);

  // Connect once, with exponential-backoff auto-reconnect like the alerts hook.
  const connect = useCallback(() => {
    const ws = new WebSocket(WS_LIVE_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      retryCountRef.current = 0;
      setWsConnected(true);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.error) {
        console.error('WebSocket Error:', data.error);
        return;
      }

      // We got bounding box data or empty data, meaning the backend finished processing the frame
      isProcessingRef.current = false;

      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (!canvas || !video) {
        return;
      }

      const ctx = canvas.getContext('2d');
      // Set canvas size to match video dimensions
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Clear previous drawings
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const scale = data.scale || 1;

      // 1. Draw all detected CARS first (Cyberpunk cyan-blue dashed box)
      const cars = data.all_cars || (data.car_bbox ? [data.car_bbox] : []);
      if (cars.length > 0) {
        cars.forEach((cbox, idx) => {
          const cx1 = cbox[0] / scale;
          const cy1 = cbox[1] / scale;
          const cx2 = cbox[2] / scale;
          const cy2 = cbox[3] / scale;
          const cw = cx2 - cx1;
          const ch = cy2 - cy1;

          ctx.strokeStyle = 'rgba(0, 195, 255, 0.85)';
          ctx.lineWidth = 2.5;
          ctx.setLineDash([8, 4]);
          ctx.strokeRect(cx1, cy1, cw, ch);
          ctx.setLineDash([]);

          // Vehicle badge
          ctx.fillStyle = 'rgba(0, 195, 255, 0.9)';
          ctx.fillRect(cx1, Math.max(0, cy1 - 24), 70, 24);
          ctx.fillStyle = '#060d1d';
          ctx.font = 'bold 12px "JetBrains Mono", monospace';
          ctx.fillText(`CAR #${idx + 1}`, cx1 + 6, Math.max(16, cy1 - 7));
        });
      }

      // 2. Draw detected plates on cars (Glowing neon-green / cyan box)
      if (data.bbox) {
        const x1 = data.bbox[0] / scale;
        const y1 = data.bbox[1] / scale;
        const x2 = data.bbox[2] / scale;
        const y2 = data.bbox[3] / scale;
        const width = x2 - x1;
        const height = y2 - y1;

        ctx.strokeStyle = '#00ffcc';
        ctx.lineWidth = 4;
        ctx.strokeRect(x1, y1, width, height);

        const label = data.confirmed || data.plate_text;
        if (label) {
          const isConfirmed = Boolean(data.confirmed);
          const badgeWidth = Math.max(width, 180);
          ctx.fillStyle = isConfirmed ? 'rgba(0, 255, 204, 0.95)' : 'rgba(0, 0, 0, 0.85)';
          ctx.fillRect(x1, Math.max(0, y1 - 34), badgeWidth, 34);

          ctx.fillStyle = isConfirmed ? '#0b0f19' : '#00ffcc';
          ctx.font = 'bold 18px "JetBrains Mono", monospace';
          ctx.fillText(
            `${label}${isConfirmed ? ' ✓' : ''} (${(data.confidence * 100).toFixed(0)}%)`,
            x1 + 6,
            Math.max(22, y1 - 10)
          );

          setPlateInfo(`${label} (${(data.confidence * 100).toFixed(1)}%)`);
        }
      }

      // Frame progression is handled by requestAnimationFrame loop; no manual timeout needed
    };

    ws.onclose = () => {
      setWsConnected(false);
      if (disposedRef.current) return;
      const delay = Math.min(1000 * 2 ** retryCountRef.current, MAX_RETRY_DELAY);
      retryCountRef.current += 1;
      retryTimerRef.current = setTimeout(connect, delay);
    };

    ws.onerror = () => ws.close();
  }, []);

  useEffect(() => {
    disposedRef.current = false;
    connect();
    return () => {
      disposedRef.current = true;
      clearTimeout(retryTimerRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  // Use a fixed interval (e.g. 300ms = ~3.3 fps) to avoid overwhelming the backend
  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      processFrame();
    }, 300);
    return () => clearInterval(interval);
  }, [isPlaying]);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setVideoFile(url);
    }
  };

  // The magic loop that extracts frames and sends them to the backend
  const processFrame = () => {
    const video = videoRef.current;
    const ws = wsRef.current;

    if (isProcessingRef.current) {
      return; // Wait for the previous frame to finish!
    }

    if (video && !video.paused && !video.ended && ws && ws.readyState === WebSocket.OPEN) {
      isProcessingRef.current = true;
      // Create a temporary hidden canvas to extract the image data
      // Downscale to max 640 width to drastically speed up CPU YOLO inference
      const MAX_WIDTH = 640;
      const scale = video.videoWidth > MAX_WIDTH ? MAX_WIDTH / video.videoWidth : 1;
      const targetWidth = video.videoWidth * scale;
      const targetHeight = video.videoHeight * scale;

      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = targetWidth;
      tempCanvas.height = targetHeight;

      const ctx = tempCanvas.getContext('2d');
      ctx.drawImage(video, 0, 0, targetWidth, targetHeight);

      // Compress frame to save bandwidth (JPEG, quality 0.5)
      const frameDataUrl = tempCanvas.toDataURL('image/jpeg', 0.5);

      // Send to backend along with the scale factor so we can fix bounding boxes
      ws.send(JSON.stringify({ frame: frameDataUrl, scale: scale }));
    }
  };

  const togglePlay = () => {
    const video = videoRef.current;
    if (video.paused) {
      video.play();
      setIsPlaying(true);
      processFrame();
    } else {
      video.pause();
      setIsPlaying(false);
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    }
  };

  return (
    <div className="card" style={{ maxWidth: '1000px', margin: '0 auto', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '20px' }}>
      <div className="card-head" style={{ width: '100%' }}>
        <div className="card-title">
          <span className="bar"></span>
          Live Video ANPR Processing
        </div>
        {!wsConnected && (
          <span className="card-sub" style={{ background: 'rgba(255,23,68,0.1)', color: 'var(--neon-red)', borderColor: 'rgba(255,23,68,0.3)' }}>
            DISCONNECTED
          </span>
        )}
      </div>

      <div className="glass-panel" style={{
        width: '100%',
        display: 'flex',
        flexWrap: 'wrap',
        gap: '16px',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '16px 20px',
      }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', flex: 1, minWidth: '280px' }}>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <span style={{ fontSize: '11px', fontWeight: 700, color: 'var(--neon-cyan)', letterSpacing: '1px' }}>
              PIPELINE: 🚗 CARS FIRST → 🪪 PLATES
            </span>
          </div>
          <input
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            className="vf-file"
            style={{ width: '100%', background: 'rgba(0,0,0,0.3)', border: '1px dashed var(--glass-border)' }}
          />
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)', alignSelf: 'center' }}>Test Videos:</span>
            <button
              type="button"
              className="glass-btn"
              style={{ fontSize: '11px', padding: '4px 10px' }}
              onClick={() => {
                setVideoFile('/sample_videos/demo_recording.mp4');
                setIsPlaying(false);
              }}
            >
              📹 Recording 2026-09-10
            </button>
            <button
              type="button"
              className="glass-btn"
              style={{ fontSize: '11px', padding: '4px 10px' }}
              onClick={() => {
                setVideoFile('/sample_videos/demo_anpr.mp4');
                setIsPlaying(false);
              }}
            >
              📹 ANPR Video 1
            </button>
            <button
              type="button"
              className="glass-btn"
              style={{ fontSize: '11px', padding: '4px 10px' }}
              onClick={() => {
                setVideoFile('/sample_videos/demo_traffic.mp4');
                setIsPlaying(false);
              }}
            >
              📹 Traffic 1080p
            </button>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Screen Size</label>
          <button
            onClick={() => setVideoSize(Math.max(400, Number(videoSize) - 50))}
            className="icon-btn"
            style={{ width: '28px', height: '28px', borderRadius: '6px' }}
          >-</button>
          <input
            type="range"
            min="400"
            max="1200"
            step="50"
            value={videoSize}
            onChange={(e) => setVideoSize(Number(e.target.value))}
            style={{ cursor: 'pointer', accentColor: 'var(--neon-cyan)' }}
          />
          <button
            onClick={() => setVideoSize(Math.min(1200, Number(videoSize) + 50))}
            className="icon-btn"
            style={{ width: '28px', height: '28px', borderRadius: '6px' }}
          >+</button>
        </div>
        {videoFile && (
          <button
            onClick={togglePlay}
            className={`glass-btn ${isPlaying ? 'active' : ''}`}
            style={{ 
              background: isPlaying ? 'rgba(255,23,68,0.1)' : 'rgba(0,230,118,0.1)',
              color: isPlaying ? 'var(--neon-red)' : 'var(--neon-green)',
              borderColor: isPlaying ? 'var(--neon-red)' : 'var(--neon-green)'
            }}
          >
            {isPlaying ? 'PAUSE FEED' : 'START PROCESSING'}
          </button>
        )}
      </div>

      <div style={{ position: 'relative', width: '100%', maxWidth: `${videoSize}px`, borderRadius: '12px', overflow: 'hidden', boxShadow: '0 10px 40px rgba(0, 229, 255, 0.1)', border: '1px solid var(--glass-border)' }}>
        {videoFile ? (
          <>
            <video
              ref={videoRef}
              src={videoFile}
              style={{ width: '100%', display: 'block', background: '#000' }}
              controls={false}
              muted
              onEnded={() => setIsPlaying(false)}
            />
            {/* The transparent overlay canvas where we draw the bounding boxes */}
            <canvas
              ref={canvasRef}
              style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
            />
          </>
        ) : (
          <div style={{ width: '100%', aspectRatio: '16/9', background: 'rgba(0,0,0,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', border: '1px dashed var(--glass-border)', borderRadius: '12px' }}>
            <div style={{ textAlign: 'center' }}>
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" style={{ opacity: 0.5, marginBottom: '10px' }}>
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              <br/>
              Select a video file to engage ANPR
            </div>
          </div>
        )}
      </div>

      {/* Display latest detected plate in a dedicated box */}
      <div className="glass-panel" style={{
        marginTop: '10px',
        width: '100%',
        padding: '24px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        background: plateInfo ? 'rgba(0, 229, 255, 0.05)' : 'rgba(10, 17, 40, 0.5)',
        borderColor: plateInfo ? 'var(--neon-cyan)' : 'var(--glass-border)',
        boxShadow: plateInfo ? '0 0 30px rgba(0,229,255,0.15)' : 'none',
        transition: 'all 0.3s ease'
      }}>
        <h3 style={{ margin: '0 0 12px 0', color: 'var(--text-muted)', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '2px', fontWeight: 700 }}>
          Live Target Acquisition
        </h3>
        <div style={{
          padding: '16px 32px',
          minWidth: '350px',
          textAlign: 'center',
        }}>
          {plateInfo ? (
            <span style={{ 
              color: 'var(--neon-cyan)', 
              fontFamily: '"JetBrains Mono", monospace', 
              fontSize: '32px', 
              fontWeight: 800,
              textShadow: '0 0 20px var(--neon-cyan-glow)',
              letterSpacing: '3px'
            }}>
              {plateInfo}
            </span>
          ) : (
            <span style={{ color: 'var(--text-muted)', fontStyle: 'italic', fontSize: '14px', letterSpacing: '1px' }}>
              Awaiting plate signature...
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
