import React, { useState, useRef, useEffect } from 'react';
// Helper to compute correct WebSocket base URL
const getWsBase = () => {
  const host = window.location.hostname;
  // Django backend runs on port 9000 (8000 is held by Docker Desktop)
  return `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${host}:9000/ws/`;
};
export default function LiveVideoProcessor() {
  const [videoFile, setVideoFile] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [plateInfo, setPlateInfo] = useState(null);
  const [videoSize, setVideoSize] = useState(800);
  // Removed unused processing state
  const wsRef = useRef(null);
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const videoRef = useRef(null);

  // Connect to WebSocket on mount
  useEffect(() => {
    const ws = new WebSocket(`${getWsBase()}live-video/`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('Live Video WebSocket Connected');
      setWsConnected(true);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.error) {
        console.error('WebSocket Error:', data.error);
        return;
      }

      // We got bounding box data, let's draw it immediately on the canvas!
      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (!canvas || !video) {
        return;
      }
      if (!data.bbox) {
        // No detection in this frame; skip drawing.
        return;
      }
      // Continue to drawing below


      const ctx = canvas.getContext('2d');
      // Set canvas size to match video dimensions
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Clear previous drawings
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw Bounding Box
      // The backend returns coordinates relative to the downscaled frame.
      // We must scale them back up to the original video dimensions.
      const scale = data.scale || 1;
      const x1 = data.bbox[0] / scale;
      const y1 = data.bbox[1] / scale;
      const x2 = data.bbox[2] / scale;
      const y2 = data.bbox[3] / scale;
      const width = x2 - x1;
      const height = y2 - y1;

      ctx.strokeStyle = '#00ffcc'; // Cyberpunk cyan!
      ctx.lineWidth = 4;
      ctx.strokeRect(x1, y1, width, height);

      // Draw Plate Text if we found one
      if (data.plate_text) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(x1, y1 - 40, width, 40);

        ctx.fillStyle = '#00ffcc';
        ctx.font = '24px "JetBrains Mono", monospace';
        ctx.fillText(`${data.plate_text} (${(data.confidence * 100).toFixed(1)}%)`, x1 + 5, y1 - 10);

        // Update UI plate info state
        setPlateInfo(`${data.plate_text} (${(data.confidence * 100).toFixed(1)}%)`);
      }

      // Frame progression is handled by requestAnimationFrame loop; no manual timeout needed
    };

    ws.onclose = () => {
      console.log('Live Video WebSocket Disconnected');
      setWsConnected(false);
    };

    // Replace setTimeout loop with requestAnimationFrame for smoother processing
    let animationId = null;
    const loop = () => {
      if (isPlaying) {
        processFrame();
        animationRef.current = requestAnimationFrame(loop);
      }
    };
    if (isPlaying) {
      animationId = requestAnimationFrame(loop);
    }
    return () => {
      if (animationId) cancelAnimationFrame(animationId);
    };
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

    if (video && !video.paused && !video.ended && ws && ws.readyState === WebSocket.OPEN) {
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
    <div style={{ padding: '2rem', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem', background: '#0b0f19', color: '#e1e5ee', fontFamily: 'Inter, sans-serif' }}>
      <h1>Live Video ANPR Processing</h1>

      {!wsConnected && (
        <div style={{ color: '#ff4d4d', background: 'rgba(255, 77, 77, 0.1)', padding: '1rem', borderRadius: '8px' }}>
          Backend WebSocket Disconnected. Is the Django server running?
        </div>
      )}

      <div style={{
        width: '100%',
        maxWidth: '800px',
        display: 'flex',
        gap: '1rem',
        justifyContent: 'center',
        alignItems: 'center',
        position: 'sticky',
        top: '1rem',
        zIndex: 10,
        background: 'rgba(11, 15, 25, 0.9)',
        backdropFilter: 'blur(10px)',
        padding: '1rem',
        borderRadius: '8px',
        boxShadow: '0 4px 15px rgba(0,0,0,0.5)'
      }}>
        <input
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          style={{ padding: '0.5rem', background: '#1a2235', border: '1px solid #2a3553', borderRadius: '4px', color: 'white' }}
        />
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <label style={{ fontSize: '0.9rem', color: '#8a9ab8' }}>Screen Size:</label>
          <button
            onClick={() => setVideoSize(Math.max(400, Number(videoSize) - 50))}
            style={{ background: '#2a3553', color: 'white', border: 'none', borderRadius: '4px', width: '24px', height: '24px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          >-</button>
          <input
            type="range"
            min="400"
            max="1200"
            step="50"
            value={videoSize}
            onChange={(e) => setVideoSize(Number(e.target.value))}
            style={{ cursor: 'pointer' }}
          />
          <button
            onClick={() => setVideoSize(Math.min(1200, Number(videoSize) + 50))}
            style={{ background: '#2a3553', color: 'white', border: 'none', borderRadius: '4px', width: '24px', height: '24px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          >+</button>
          <button
            onClick={() => document.getElementById('live-video-scroller').scrollBy({ top: 300, behavior: 'smooth' })}
            style={{ background: '#2a3553', color: 'white', border: 'none', borderRadius: '4px', width: '24px', height: '24px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', marginLeft: '0.5rem' }}
            title="Scroll Down"
          >↓</button>
        </div>
        {videoFile && (
          <button
            onClick={togglePlay}
            style={{ padding: '0.5rem 1rem', background: isPlaying ? '#ff4d4d' : '#00ffcc', color: '#0b0f19', border: 'none', borderRadius: '4px', fontWeight: 'bold', cursor: 'pointer' }}
          >
            {isPlaying ? 'Pause' : 'Start Live Processing'}
          </button>
        )}
      </div>

      <div style={{ position: 'relative', width: '100%', maxWidth: `${videoSize}px`, borderRadius: '12px', overflow: 'hidden', boxShadow: '0 10px 30px rgba(0,255,204,0.1)', transition: 'max-width 0.3s ease' }}>
        {videoFile ? (
          <>
            <video
              ref={videoRef}
              src={videoFile}
              style={{ width: '100%', display: 'block' }}
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
          <div style={{ width: '100%', aspectRatio: '16/9', background: '#1a2235', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#6a7a9c' }}>
            Select a video file to begin
          </div>
        )}
        {/* Display latest detected plate in a dedicated box */}
        <div style={{
          marginTop: '1.5rem',
          width: '100%',
          padding: '1.5rem',
          background: 'rgba(11, 15, 25, 0.8)',
          border: '2px solid #2a3553',
          borderRadius: '8px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          boxShadow: '0 4px 15px rgba(0,0,0,0.5)'
        }}>
          <h3 style={{ margin: '0 0 1rem 0', color: '#8a9ab8', fontSize: '1rem', textTransform: 'uppercase', letterSpacing: '2px' }}>
            Detected License Plate
          </h3>
          <div style={{
            background: plateInfo ? 'rgba(0, 255, 204, 0.1)' : '#1a2235',
            border: plateInfo ? '2px solid #00ffcc' : '2px dashed #2a3553',
            borderRadius: '8px',
            padding: '1rem 2rem',
            minWidth: '300px',
            textAlign: 'center',
            transition: 'all 0.3s ease'
          }}>
            {plateInfo ? (
              <span style={{ color: '#00ffcc', fontFamily: '"JetBrains Mono", monospace', fontSize: '2rem', fontWeight: 'bold' }}>
                {plateInfo}
              </span>
            ) : (
              <span style={{ color: '#6a7a9c', fontStyle: 'italic', fontSize: '1.2rem' }}>
                Scanning for plates...
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
