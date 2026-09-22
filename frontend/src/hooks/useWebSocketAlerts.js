import { useCallback, useEffect, useRef, useState } from 'react';

// Same-origin relative path is the default (works behind the Docker Nginx
// WebSocket proxy). Override for local dev via VITE_WS_URL, e.g.
//   VITE_WS_URL=ws://localhost:8000/ws/alerts/
const WS_URL =
  import.meta.env.VITE_WS_URL ??
  (import.meta.env.PROD
    ? '/ws/alerts/'
    : 'ws://localhost:8000/ws/alerts/');
const MAX_RETRY_DELAY = 30_000;
const MAX_ALERTS = 100;

/* ------------------------------------------------------------------ */
/*  Tiny Web Audio siren – no external assets needed                    */
/* ------------------------------------------------------------------ */
let audioCtx = null;

function getAudioContext() {
  if (!audioCtx && typeof AudioContext !== 'undefined') {
    audioCtx = new AudioContext();
  }
  return audioCtx;
}

function playAlertBeep() {
  const ctx = getAudioContext();
  if (!ctx) return;
  if (ctx.state === 'suspended') ctx.resume();

  [0, 0.15, 0.3].forEach((offset) => {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'square';
    osc.frequency.value = 880;
    gain.gain.setValueAtTime(0.25, ctx.currentTime + offset);
    gain.gain.exponentialRampToValueAtTime(
      0.001,
      ctx.currentTime + offset + 0.12,
    );
    osc.connect(gain).connect(ctx.destination);
    osc.start(ctx.currentTime + offset);
    osc.stop(ctx.currentTime + offset + 0.12);
  });
}

/* ------------------------------------------------------------------ */
/*  Hook                                                                */
/* ------------------------------------------------------------------ */

export default function useWebSocketAlerts() {
  const [alerts, setAlerts] = useState([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);
  const retryCountRef = useRef(0);
  const onAlertRef = useRef(null);
  const disposedRef = useRef(false);

  /** Register an external callback invoked when a fresh alert arrives. */
  const registerAlertHandler = useCallback((cb) => {
    onAlertRef.current = cb;
  }, []);

  /* -------- WebSocket lifecycle with exponential-backoff retry -------- */
  useEffect(() => {
    disposedRef.current = false;
    let retryTimer = null;

    const connect = () => {
      if (disposedRef.current) return;

      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        retryCountRef.current = 0;
        setConnected(true);
      };

      ws.onmessage = (event) => {
        let payload;
        try {
          payload = JSON.parse(event.data);
        } catch {
          return;
        }

        // Handle Progress Updates (No alert beep)
        if (payload.type === 'send_progress_update') {
          onAlertRef.current?.(payload);
          return;
        }

        const alert = {
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          receivedAt: new Date().toISOString(),
          ...payload,
        };

        setAlerts((prev) => [alert, ...prev].slice(0, MAX_ALERTS));
        playAlertBeep();
        onAlertRef.current?.(alert);
      };

      ws.onclose = () => {
        setConnected(false);
        if (disposedRef.current) return;
        const delay = Math.min(
          1_000 * 2 ** retryCountRef.current,
          MAX_RETRY_DELAY,
        );
        retryCountRef.current += 1;
        retryTimer = setTimeout(connect, delay);
      };

      ws.onerror = () => ws.close();
    };

    connect();

    return () => {
      disposedRef.current = true;
      clearTimeout(retryTimer);
      wsRef.current?.close();
    };
  }, []);

  const removeAlert = useCallback(
    (id) => setAlerts((prev) => prev.filter((a) => a.id !== id)),
    [],
  );

  const clearAlerts = useCallback(() => setAlerts([]), []);

  return { alerts, connected, removeAlert, clearAlerts, registerAlertHandler };
}
