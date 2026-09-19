import React, { useState, useRef, useEffect } from 'react';
// Helper to compute correct WebSocket base URL (Docker vs local dev)
const getWsBase = () => {
  const host = window.location.hostname;
  const isDocker = host === 'localhost' && window.location.port === '3000';
  return isDocker ? `ws://${host}:8000/ws/` : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${host}${window.location.port ? ':' + window.location.port : ''}/ws/`;
};
export default function LiveVideoProcessor() {
  const [videoFile, setVideoFile] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [plateInfo, setPlateInfo] = useState(null);
  const [videoSize, setVideoSize] = useState(800);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const animationRef = useRef(null);
  
  // Connect to WebSocket on mount
  useEffect(() => {
    const ws = new WebSocket(`${getWsBase()}live-video/`);
    
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
      if (!canvas || !video || !data.bbox) return;
      
      const ctx = canvas.getContext('2d');
      // Set canvas size to match video dimensions
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // Clear previous drawings
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw Bounding Box
      const [x1, y1, x2, y2] = data.bbox;
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
    };
    
    ws.onclose = () => {
      console.log('Live Video WebSocket Disconnected');
      setWsConnected(false);
    };
    
    wsRef.current = ws;
    
    return () => {
      if (ws.readyState === 1) ws.close();
    };
  }, []);
  
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
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = video.videoWidth;
      tempCanvas.height = video.videoHeight;
      const ctx = tempCanvas.getContext('2d');
      ctx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
      
      // Compress frame to save bandwidth (JPEG, quality 0.5)
      const frameDataUrl = tempCanvas.toDataURL('image/jpeg', 0.5);
      
      // Send to backend
      ws.send(JSON.stringify({ frame: frameDataUrl }));
    }
    
    // Schedule next frame extraction. 
    // We throttle this to roughly 10fps so we don't melt the backend.
    setTimeout(() => {
      if (isPlaying) {
         animationRef.current = requestAnimationFrame(processFrame);
      }
    }, 100); 
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
        {/* Display latest detected plate */}
        {plateInfo && (
          <div style={{ marginTop: '0.5rem', color: '#00ffcc', fontFamily: 'Inter, sans-serif', fontSize: '1.1rem' }}>
            Detected Plate: {plateInfo}
          </div>
        )}
      </div>
    </div>
  );
}
