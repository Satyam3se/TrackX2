import React, { useEffect, useRef, useState } from 'react';

const MAX_FRAME_WIDTH = 1280;
const FRAME_RESPONSE_TIMEOUT = 30000;
const PLATE_PATTERN = /[^A-Z0-9]/g;

function normalizePlate(value) {
  return String(value || '').toUpperCase().replace(PLATE_PATTERN, '');
}

function createSessionId() {
  return globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
}

function getLiveWsUrl() {
  const configured = import.meta.env.VITE_LIVE_WS_URL;
  if (configured) {
    if (configured.startsWith('ws')) return configured;
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${window.location.host}${configured.startsWith('/') ? configured : `/${configured}`}`;
  }
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}/ws/live-video/`;
}

function toFrameBox(box, scale, video) {
  if (!Array.isArray(box) || box.length !== 4) return null;
  const safeScale = Number(scale) > 0 ? Number(scale) : 1;
  const values = box.map((value) => Number(value) / safeScale);
  if (values.some((value) => !Number.isFinite(value))) return null;
  const x1 = Math.max(0, Math.min(video.videoWidth, values[0]));
  const y1 = Math.max(0, Math.min(video.videoHeight, values[1]));
  const x2 = Math.max(0, Math.min(video.videoWidth, values[2]));
  const y2 = Math.max(0, Math.min(video.videoHeight, values[3]));
  if (x2 <= x1 || y2 <= y1) return null;
  return [x1, y1, x2, y2];
}

function getDetectionLabel(detection) {
  return detection.confirmed || detection.plate_text || '';
}

function getPlateVariants(detection) {
  const values = detection?.variants?.length
    ? detection.variants
    : [detection?.confirmed, detection?.plate_text];
  return [...new Set(values.map(normalizePlate).filter(Boolean))];
}

function getPlateMatch(detection, query) {
  const normalizedQuery = normalizePlate(query);
  if (!normalizedQuery) return { matched: false, score: 0, type: '' };

  let best = { matched: false, score: 0, type: '' };
  for (const variant of getPlateVariants(detection)) {
    let score = 0;
    let type = '';
    if (variant === normalizedQuery) {
      score = 10000;
      type = 'exact';
    } else if (variant.startsWith(normalizedQuery)) {
      score = 8000 + normalizedQuery.length;
      type = 'prefix';
    } else if (variant.endsWith(normalizedQuery)) {
      score = 7000 + normalizedQuery.length;
      type = 'suffix';
    } else if (variant.includes(normalizedQuery)) {
      score = 6000 + normalizedQuery.length;
      type = 'contains';
    } else if (
      variant.length >= 4
      && normalizedQuery.includes(variant)
    ) {
      score = 5000 + variant.length;
      type = 'fragment';
    }
    if (score > best.score) best = { matched: true, score, type };
  }
  return best;
}

function getDetectionId(detection, index) {
  if (detection.track_id !== undefined && detection.track_id !== null) {
    return `track-${detection.track_id}`;
  }
  return `${normalizePlate(getDetectionLabel(detection))}-${index}`;
}

function rankPlateReads(plates, query) {
  return plates
    .map((plate) => ({ ...plate, match: getPlateMatch(plate, query) }))
    .sort((left, right) => {
      if (right.match.score !== left.match.score) {
        return right.match.score - left.match.score;
      }
      return right.confidence - left.confidence;
    });
}

function toPlateRead(plate) {
  return {
    id: plate.id,
    label: plate.label,
    confidence: plate.confidence * 100,
    confirmed: plate.confirmed,
    variants: plate.variants,
    matched: plate.match.matched,
    matchType: plate.match.type,
    matchScore: plate.match.score,
  };
}

function getMatchLabel(type) {
  if (type === 'exact') return 'EXACT MATCH';
  if (type === 'prefix') return 'PREFIX MATCH';
  if (type === 'suffix') return 'SUFFIX MATCH';
  if (type === 'fragment') return 'PARTIAL READ';
  return 'PARTIAL MATCH';
}

function drawLabel(ctx, text, x, y, width, height, matched, confirmed) {
  const fontSize = Math.max(16, Math.min(26, Math.round(ctx.canvas.width / 58)));
  const labelHeight = Math.max(30, fontSize + 12);
  const labelX = Math.max(4, Math.min(x, Math.max(4, ctx.canvas.width - 44)));
  const labelY = y >= labelHeight + 4 ? y - labelHeight : y + height + 4;
  const color = matched ? '#ff2d75' : confirmed ? '#00e676' : '#00e5ff';

  ctx.save();
  ctx.font = `600 ${fontSize}px "JetBrains Mono", monospace`;
  const labelWidth = Math.min(
    Math.max(40, ctx.canvas.width - labelX - 4),
    Math.max(width, ctx.measureText(text).width + 18),
  );
  ctx.fillStyle = matched ? 'rgba(255, 45, 117, 0.94)' : 'rgba(5, 12, 24, 0.88)';
  ctx.fillRect(labelX, labelY, labelWidth, labelHeight);
  ctx.fillStyle = matched ? '#ffffff' : color;
  ctx.textBaseline = 'middle';
  ctx.fillText(text, labelX + 9, labelY + labelHeight / 2, Math.max(20, labelWidth - 14));
  ctx.restore();
}

function drawBox(ctx, box, matched, dashed = false, focused = false) {
  const [x1, y1, x2, y2] = box;
  const width = x2 - x1;
  const height = y2 - y1;
  const color = matched ? '#ff2d75' : '#00e5ff';
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = focused ? 9 : matched ? 6 : 3;
  ctx.shadowColor = color;
  ctx.shadowBlur = focused ? 26 : matched ? 16 : 5;
  ctx.setLineDash(dashed ? [10, 7] : []);
  ctx.strokeRect(x1, y1, width, height);
  ctx.restore();
}

function drawDetections(video, canvas, payload, query, focusedId = '') {
  if (!video || !canvas || !video.videoWidth || !video.videoHeight) return;
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!payload) return;

  const detections = Array.isArray(payload.detections) && payload.detections.length
    ? payload.detections
    : payload.bbox
      ? [payload]
      : [];
  detections.forEach((detection, index) => {
    const plateBox = toFrameBox(detection.bbox, payload.scale, video);
    if (!plateBox) return;
    const label = getDetectionLabel(detection);
    const match = getPlateMatch(detection, query);
    const matched = match.matched;
    const detectionId = getDetectionId(detection, index);
    const focused = matched && detectionId === focusedId;
    const vehicleBox = toFrameBox(detection.vehicle_bbox, payload.scale, video);
    if (vehicleBox) {
      const sameBox = vehicleBox.every((value, index) => Math.abs(value - plateBox[index]) < 3);
      if (!sameBox) drawBox(ctx, vehicleBox, matched, !matched, focused);
    }
    drawBox(ctx, plateBox, matched, !matched, focused);
    if (label) {
      const confidence = Number.isFinite(Number(detection.confidence))
        ? `${(Number(detection.confidence) * 100).toFixed(1)}%`
        : '';
      const matchLabel = matched
        ? `${focused ? 'FOCUS · ' : ''}${match.type === 'exact' ? 'MATCH' : 'PARTIAL'} · `
        : '';
      const text = `${matchLabel}${label}${detection.confirmed ? ' ✓' : ''}${confidence ? ` ${confidence}` : ''}`;
      ctx.save();
      ctx.font = `600 ${Math.max(16, Math.min(26, Math.round(canvas.width / 58)))}px "JetBrains Mono", monospace`;
      drawLabel(ctx, text, plateBox[0], plateBox[1], plateBox[2] - plateBox[0], plateBox[3] - plateBox[1], matched, Boolean(detection.confirmed));
      ctx.restore();
    }
  });
}

export default function LiveVideoProcessor() {
  const [videoUrl, setVideoUrl] = useState(null);
  const [fileName, setFileName] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [processingError, setProcessingError] = useState('');
  const [plateInput, setPlateInput] = useState('');
  const [plateQuery, setPlateQuery] = useState('');
  const [bestPlate, setBestPlate] = useState(null);
  const [latestPlates, setLatestPlates] = useState([]);
  const [matchStatus, setMatchStatus] = useState(null);
  const [videoSize, setVideoSize] = useState(800);
  const wsRef = useRef(null);
  const videoUrlRef = useRef(null);
  const busyRef = useRef(false);
  const animationRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const bestPlateRef = useRef(null);
  const lastPayloadRef = useRef(null);
  const latestPlatesRef = useRef([]);
  const queryRef = useRef('');
  const matchStatusRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const reconnectAttemptRef = useRef(0);
  const processFrameRef = useRef(null);
  const sessionIdRef = useRef('');
  const frameIdRef = useRef(0);
  const requestTimerRef = useRef(null);

  const resetResults = () => {
    if (requestTimerRef.current) {
      window.clearTimeout(requestTimerRef.current);
      requestTimerRef.current = null;
    }
    busyRef.current = false;
    sessionIdRef.current = createSessionId();
    frameIdRef.current = 0;
    bestPlateRef.current = null;
    lastPayloadRef.current = null;
    latestPlatesRef.current = [];
    matchStatusRef.current = null;
    setBestPlate(null);
    setLatestPlates([]);
    setMatchStatus(null);
  };

  useEffect(() => {
    let disposed = false;
    let socket = null;

    const scheduleReconnect = () => {
      if (disposed || reconnectTimerRef.current) return;
      const delay = Math.min(1000 * (2 ** reconnectAttemptRef.current), 10000);
      reconnectAttemptRef.current += 1;
      reconnectTimerRef.current = window.setTimeout(() => {
        reconnectTimerRef.current = null;
        connect();
      }, delay);
    };

    const handleMessage = (event) => {
      busyRef.current = false;
      if (requestTimerRef.current) {
        window.clearTimeout(requestTimerRef.current);
        requestTimerRef.current = null;
      }
      let data;
      try {
        data = JSON.parse(event.data);
      } catch {
        setProcessingError('The processor returned an unreadable response.');
        return;
      }
      if (
        disposed
        || (data.session_id && data.session_id !== sessionIdRef.current)
        || (data.frame_id && data.frame_id !== frameIdRef.current)
      ) {
        return;
      }
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (data.error) {
        setProcessingError(data.error);
        if (video && canvas) drawDetections(video, canvas, null, queryRef.current);
        lastPayloadRef.current = null;
        latestPlatesRef.current = [];
        setLatestPlates([]);
        if (queryRef.current) {
          const nextStatus = { query: queryRef.current, count: 0, label: '' };
          matchStatusRef.current = nextStatus;
          setMatchStatus(nextStatus);
        }
        return;
      }
      reconnectAttemptRef.current = 0;
      setProcessingError('');
      lastPayloadRef.current = data;

      const query = queryRef.current;
      const rawDetections = Array.isArray(data.detections) && data.detections.length
        ? data.detections
        : data.bbox
          ? [data]
          : [];
      const currentPlates = rawDetections
        .map((detection, index) => {
          const label = getDetectionLabel(detection);
          return {
            id: getDetectionId(detection, index),
            label,
            variants: getPlateVariants(detection),
            confidence: Number(detection.confidence) || 0,
            confirmed: Boolean(detection.confirmed),
            trackId: detection.track_id,
          };
        })
        .filter((plate) => plate.label);
      const rankedPlates = rankPlateReads(currentPlates, query);
      latestPlatesRef.current = rankedPlates;
      setLatestPlates(rankedPlates);

      const focusedPlate = query
        ? rankedPlates.find((plate) => plate.match.matched)
        : rankedPlates[0];
      if (focusedPlate) {
        const read = toPlateRead(focusedPlate);
        const previous = bestPlateRef.current;
        if (query || !previous || read.confidence > previous.confidence) {
          bestPlateRef.current = read;
          setBestPlate(read);
        }
      }

      const matches = rankedPlates.filter((plate) => plate.match.matched);
      const nextStatus = query
        ? {
          query,
          count: matches.length,
          label: matches[0]?.label || '',
          matchType: matches[0]?.match.type || '',
        }
        : null;
      matchStatusRef.current = nextStatus;
      setMatchStatus(nextStatus);
      if (video && canvas) {
        drawDetections(video, canvas, data, query, focusedPlate?.id || '');
      }
    };

    const connect = () => {
      if (disposed) return;
      try {
        socket = new WebSocket(getLiveWsUrl());
        wsRef.current = socket;
        socket.onopen = () => {
          if (disposed) return;
          setWsConnected(true);
          setProcessingError('');
        };
        socket.onmessage = handleMessage;
        socket.onerror = () => {
          if (!disposed) {
            setProcessingError('Unable to reach the live ANPR processor.');
          }
        };
        socket.onclose = () => {
          if (disposed) return;
          busyRef.current = false;
          if (requestTimerRef.current) {
            window.clearTimeout(requestTimerRef.current);
            requestTimerRef.current = null;
          }
          videoRef.current?.pause();
          const video = videoRef.current;
          const canvas = canvasRef.current;
          if (video && canvas) drawDetections(video, canvas, null, queryRef.current);
          lastPayloadRef.current = null;
          setLatestPlates([]);
          if (queryRef.current) {
            const nextStatus = { query: queryRef.current, count: 0, label: '' };
            matchStatusRef.current = nextStatus;
            setMatchStatus(nextStatus);
          }
          setWsConnected(false);
          setProcessingError('Live processor disconnected. Reconnecting automatically…');
          if (wsRef.current === socket) wsRef.current = null;
          scheduleReconnect();
        };
      } catch {
        setWsConnected(false);
        scheduleReconnect();
      }
    };

    connect();
    return () => {
      disposed = true;
      if (reconnectTimerRef.current) {
        window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      if (requestTimerRef.current) {
        window.clearTimeout(requestTimerRef.current);
        requestTimerRef.current = null;
      }
      const activeSocket = wsRef.current;
      wsRef.current = null;
      if (activeSocket && activeSocket.readyState <= WebSocket.OPEN) {
        activeSocket.close();
      }
    };
  }, []);

  useEffect(() => {
    queryRef.current = plateQuery;
    const rankedPlates = rankPlateReads(latestPlatesRef.current, plateQuery);
    latestPlatesRef.current = rankedPlates;
    setLatestPlates(rankedPlates);

    const focusedPlate = plateQuery
      ? rankedPlates.find((plate) => plate.match.matched)
      : rankedPlates[0];
    if (focusedPlate) {
      const read = toPlateRead(focusedPlate);
      const previous = bestPlateRef.current;
      if (plateQuery || !previous || read.confidence > previous.confidence) {
        bestPlateRef.current = read;
        setBestPlate(read);
      }
    } else if (plateQuery) {
      bestPlateRef.current = null;
      setBestPlate(null);
    }

    const matches = rankedPlates.filter((plate) => plate.match.matched);
    const nextStatus = plateQuery
      ? {
        query: plateQuery,
        count: matches.length,
        label: matches[0]?.label || '',
        matchType: matches[0]?.match.type || '',
      }
      : null;
    matchStatusRef.current = nextStatus;
    setMatchStatus(nextStatus);
    if (videoRef.current && canvasRef.current) {
      drawDetections(
        videoRef.current,
        canvasRef.current,
        lastPayloadRef.current,
        plateQuery,
        focusedPlate?.id || '',
      );
    }
  }, [plateQuery]);

  useEffect(() => {
    if (!isPlaying) return undefined;
    let cancelled = false;
    const loop = () => {
      if (cancelled) return;
      processFrameRef.current?.();
      animationRef.current = window.requestAnimationFrame(loop);
    };
    animationRef.current = window.requestAnimationFrame(loop);
    return () => {
      cancelled = true;
      if (animationRef.current) window.cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    };
  }, [isPlaying]);

  useEffect(() => () => {
    if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current);
    if (animationRef.current) window.cancelAnimationFrame(animationRef.current);
    if (requestTimerRef.current) window.clearTimeout(requestTimerRef.current);
  }, []);

  const processFrame = () => {
    const video = videoRef.current;
    const socket = wsRef.current;
    if (
      !video
      || video.readyState < 2
      || video.paused
      || video.ended
      || !socket
      || socket.readyState !== WebSocket.OPEN
      || busyRef.current
    ) return;

    busyRef.current = true;
    const frameId = frameIdRef.current + 1;
    frameIdRef.current = frameId;
    try {
      const scale = video.videoWidth > MAX_FRAME_WIDTH
        ? MAX_FRAME_WIDTH / video.videoWidth
        : 1;
      const targetWidth = Math.max(1, Math.round(video.videoWidth * scale));
      const targetHeight = Math.max(1, Math.round(video.videoHeight * scale));
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = targetWidth;
      tempCanvas.height = targetHeight;
      const context = tempCanvas.getContext('2d');
      if (!context) throw new Error('Unable to create the frame processor.');
      context.drawImage(video, 0, 0, targetWidth, targetHeight);
      const frame = tempCanvas.toDataURL('image/jpeg', 0.9);
      socket.send(JSON.stringify({
        frame,
        scale,
        session_id: sessionIdRef.current,
        frame_id: frameId,
        media_time: video.currentTime,
      }));
      requestTimerRef.current = window.setTimeout(() => {
        if (!busyRef.current || frameIdRef.current !== frameId) return;
        busyRef.current = false;
        setProcessingError('The processor timed out. Reconnecting…');
        if (socket.readyState < WebSocket.CLOSING) socket.close();
      }, FRAME_RESPONSE_TIMEOUT);
    } catch {
      busyRef.current = false;
      if (requestTimerRef.current) {
        window.clearTimeout(requestTimerRef.current);
        requestTimerRef.current = null;
      }
      setProcessingError('The current video frame could not be processed.');
    }
  };
  processFrameRef.current = processFrame;

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    const video = videoRef.current;
    video?.pause();
    if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current);
    const url = URL.createObjectURL(file);
    videoUrlRef.current = url;
    setVideoUrl(url);
    setFileName(file.name);
    setIsPlaying(false);
    setProcessingError('');
    if (canvasRef.current) {
      canvasRef.current.getContext('2d')?.clearRect(
        0,
        0,
        canvasRef.current.width,
        canvasRef.current.height,
      );
    }
    resetResults();
  };

  const togglePlay = async () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused || video.ended) {
      if (video.ended) video.currentTime = 0;
      try {
        await video.play();
        setProcessingError('');
      } catch {
        setProcessingError('The browser could not start this video.');
      }
    } else {
      video.pause();
    }
  };

  const handleSearch = (event) => {
    event.preventDefault();
    setPlateQuery(normalizePlate(plateInput));
  };

  const clearSearch = () => {
    setPlateInput('');
    setPlateQuery('');
  };

  const resetBest = () => {
    bestPlateRef.current = null;
    setBestPlate(null);
  };

  const stageWidth = `${Math.min(videoSize, 1200)}px`;
  const detectedMatch = Boolean(
    bestPlate && plateQuery && getPlateMatch(bestPlate, plateQuery).matched,
  );

  return (
    <main className="live-page">
      <div className="live-shell">
        <header className="live-heading">
          <div>
            <p className="live-eyebrow">TRACKX / VIDEO INTELLIGENCE</p>
            <h1>Live Video ANPR Processing</h1>
            <p className="live-subtitle">
              Detect, search, and visually track a license plate as the vehicle moves through the frame.
            </p>
          </div>
          <div className={`live-connection ${wsConnected ? 'connected' : 'offline'}`}>
            <span className="live-connection-dot" />
            {wsConnected ? 'PROCESSOR ONLINE' : 'PROCESSOR OFFLINE'}
          </div>
        </header>

        {!wsConnected && (
          <div className="live-alert" role="status">
            <strong>Backend connection lost.</strong> Reconnecting automatically. Start processing when the indicator is online.
          </div>
        )}
        {processingError && (
          <div className="live-alert live-alert-error" role="alert">
            {processingError}
          </div>
        )}

        <section className="live-controls" aria-label="Live processing controls">
          <div className="live-file-control">
            <label htmlFor="live-video-file">VIDEO SOURCE</label>
            <input
              id="live-video-file"
              type="file"
              accept="video/*"
              onChange={handleFileChange}
            />
            {fileName && <span className="live-file-name">{fileName}</span>}
          </div>
          <div className="live-control-divider" />
          <div className="live-size-control">
            <div className="live-control-label">
              <span>Screen Size:</span>
              <output>{videoSize}px</output>
            </div>
            <div className="live-size-actions">
              <button
                type="button"
                className="live-icon-button"
                onClick={() => setVideoSize((size) => Math.max(400, size - 50))}
                aria-label="Decrease screen size"
              >
                −
              </button>
              <input
                type="range"
                min="400"
                max="1200"
                step="50"
                value={videoSize}
                onChange={(event) => setVideoSize(Number(event.target.value))}
                aria-label="Screen size"
              />
              <button
                type="button"
                className="live-icon-button"
                onClick={() => setVideoSize((size) => Math.min(1200, size + 50))}
                aria-label="Increase screen size"
              >
                +
              </button>
              <button
                type="button"
                className="live-icon-button live-scroll-button"
                onClick={() => document.getElementById('live-video-scroller')?.scrollBy({ top: 300, behavior: 'smooth' })}
                aria-label="Scroll down"
              >
                ↓
              </button>
            </div>
          </div>
          <button
            type="button"
            className={`live-start-button ${isPlaying ? 'playing' : ''}`}
            onClick={togglePlay}
            disabled={!videoUrl || !wsConnected}
          >
            <span className="live-start-icon">{isPlaying ? 'Ⅱ' : '▶'}</span>
            {isPlaying ? 'Pause Processing' : 'Start Live Processing'}
          </button>
        </section>

        <section className="live-search-row">
          <form className="live-plate-search" onSubmit={handleSearch} role="search">
            <div className="live-search-icon" aria-hidden="true">⌕</div>
            <input
              type="text"
              value={plateInput}
              onChange={(event) => setPlateInput(event.target.value)}
              placeholder="Search full or partial plate: UP10, 10U, 4714…"
              aria-label="Search a full or partial license plate"
            />
            <button type="submit">TRACK</button>
            {plateInput && (
              <button
                type="button"
                className="live-clear-search"
                onClick={clearSearch}
                aria-label="Clear plate search"
              >
                ×
              </button>
            )}
          </form>
          <div className={`live-match-status ${matchStatus?.count ? 'found' : plateQuery ? 'searching' : ''}`}>
            {plateQuery ? (
              matchStatus?.count ? (
                <>
                  <strong>{getMatchLabel(matchStatus.matchType)}</strong>
                  <span>Focused read: {matchStatus.label} · {matchStatus.count} vehicle{matchStatus.count === 1 ? '' : 's'}</span>
                </>
              ) : (
                <><strong>SEARCHING</strong><span>Looking for {plateQuery} in every detected plate</span></>
              )
            ) : (
              <><strong>PLATE SEARCH</strong><span>Use any first, middle, or last characters</span></>
            )}
          </div>
        </section>

        <section
          className={`live-video-stage ${videoUrl ? '' : 'empty'}`}
          style={{ '--live-stage-width': stageWidth }}
        >
          {videoUrl ? (
            <>
              <video
                ref={videoRef}
                src={videoUrl}
                muted
                playsInline
                controls={false}
                onLoadedMetadata={() => {
                  if (canvasRef.current && videoRef.current) {
                    canvasRef.current.width = videoRef.current.videoWidth;
                    canvasRef.current.height = videoRef.current.videoHeight;
                    drawDetections(
                      videoRef.current,
                      canvasRef.current,
                      lastPayloadRef.current,
                      queryRef.current,
                      bestPlateRef.current?.id || '',
                    );
                  }
                }}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
                onEnded={() => setIsPlaying(false)}
                onError={() => setProcessingError('This video format could not be loaded by the browser.')}
              />
              <canvas ref={canvasRef} aria-label="ANPR detection overlay" />
              <div className="live-stage-badge">
                <span className={isPlaying ? 'live-record-dot active' : 'live-record-dot'} />
                {isPlaying ? 'LIVE ANALYSIS' : 'ANALYSIS PAUSED'}
              </div>
              {plateQuery && (
                <div className="live-stage-query">
                  TARGET <strong>{plateQuery}</strong>
                </div>
              )}
            </>
          ) : (
            <div className="live-empty-state">
              <div className="live-empty-icon">▣</div>
              <strong>Select a video file to begin</strong>
              <span>MP4, WebM, and other browser-supported video formats</span>
            </div>
          )}
        </section>

        <section className="live-results" aria-label="Live recognition results">
          <div className="live-result-card live-result-primary">
            <div className="live-result-heading">
              <span>{plateQuery ? 'Focused Plate Read' : 'Detected License Plate'}</span>
              <span className="live-result-tag">{plateQuery ? 'SEARCH FOCUS' : 'BEST READ'}</span>
            </div>
            {bestPlate ? (
              <>
                <div className={`live-best-plate ${detectedMatch ? 'matched' : ''}`}>
                  {bestPlate.label} {bestPlate.confirmed ? '✓' : ''}
                </div>
                <div className="live-result-meta">
                  {detectedMatch
                    ? `${getMatchLabel(bestPlate.matchType)} · query ${plateQuery}`
                    : `Highest accuracy: ${bestPlate.confidence.toFixed(1)}%`}
                </div>
              </>
            ) : (
              <div className="live-scanning">Scanning for plates…</div>
            )}
            {bestPlate && <button type="button" className="live-reset-button" onClick={resetBest}>Reset best</button>}
          </div>
          <div className="live-result-card">
            <div className="live-result-heading">
              <span>Plates in current frame</span>
              <span className="live-count-badge">{latestPlates.length}</span>
            </div>
            {latestPlates.length ? (
              <div className="live-plate-list">
                {latestPlates.map((plate) => (
                  <div
                    className={`live-plate-row ${plateQuery && plate.match.matched ? 'matched' : ''}`}
                    key={plate.id}
                  >
                    <span className="live-plate-name">{plate.label}</span>
                    {plateQuery && plate.match.matched && (
                      <span className="live-match-kind">{getMatchLabel(plate.match.type)}</span>
                    )}
                    <span className="live-plate-confidence">{(plate.confidence * 100).toFixed(1)}%</span>
                    {plate.confirmed && <span className="live-confirmed">✓</span>}
                  </div>
                ))}
              </div>
            ) : (
              <div className="live-scanning">No readable plate in this frame</div>
            )}
          </div>
          <div className="live-result-card live-instructions">
            <div className="live-result-heading"><span>How to use</span></div>
            <ol>
              <li>Choose a local video source.</li>
              <li>Press <strong>Start Live Processing</strong>.</li>
              <li>Enter a full plate or any first, middle, or last characters and press <strong>TRACK</strong>.</li>
            </ol>
            <p>Every detected plate is searched; the strongest fragment match is focused and read automatically.</p>
          </div>
        </section>
      </div>
    </main>
  );
}
