# 🛰️ TrackX — Advanced Urban Vehicle Surveillance System

TrackX is a professional-grade, AI-powered vehicle tracking and surveillance ecosystem. It integrates real-time Computer Vision (CV), spatial databases, and asynchronous task processing to monitor vehicle movements across a city-wide network of cameras, providing instant alerts for blacklisted vehicles and deep trajectory analytics.

---

## 🚀 1. Core Technology Stack

TrackX is built on a modern, distributed architecture designed for scalability and low-latency response.

### 🧠 Artificial Intelligence (The ANPR Engine)
- **YOLOv8 (Ultralytics):** Used for **Object Detection**. It scans every camera frame to locate the precise bounding box of a vehicle's license plate.
- **fast-plate-ocr (ONNX) + EasyOCR:** Used for **Optical Character Recognition (OCR)**. Once YOLOv8 isolates the plate, the fast ONNX recognizer transcribes the crop (EasyOCR is the automatic fallback if the ONNX model is unavailable).
- **Temporal voting + IoU tracking:** A vehicle's plate is only "confirmed" after it is read the same way across multiple frames, filtering single-frame OCR noise.
- **Asynchronous Processing (Celery + Redis):** Video processing is computationally expensive. TrackX offloads the AI pipeline to background workers, ensuring the main application remains responsive.

### 🌐 Backend (The Orchestrator)
- **Django & Django REST Framework (DRF):** Provides a robust API for managing cameras, blacklisted vehicles, and detection logs. API base: `/api/v1/`.
- **Django Channels (ASGI) + Daphne:** Enables **WebSockets**, allowing the server to "push" critical alerts to the dashboard instantly and to process live-video frames in real time.
- **PostgreSQL + PostGIS:** A spatial database that stores not just data, but **geography**. It allows TrackX to perform complex spatial queries (e.g., "find all cameras within 1km of this detection").

### 🎨 Frontend (The Command Center)
- **React.js:** A high-performance UI framework for the real-time dashboard.
- **MapLibre GL / OpenStreetMap:** Provides an interactive map for visualizing camera nodes, vehicle trajectories, and traffic heatmaps.
- **Vite:** A lightning-fast build tool and development server.

### 🐳 Infrastructure
- **Docker & Docker Compose:** Entire system is containerized for "one-click" deployment, ensuring it runs the same on every teammate's machine.
- **Nginx:** Serves the built React frontend and reverse-proxies REST `/api/` and WebSocket `/ws/` traffic to the Django backend, resolving the backend via Docker's embedded DNS (no stale-IP 502s after restarts).

---

## 🛠️ 2. Installation & Setup Guide

### Prerequisites
- **Docker Desktop** (Must be installed and running)
- **Git**

### Step-by-Step Deployment
1. **Clone the Repository**
   ```bash
   git clone https://github.com/Satyam3se/TrackX2
   cd track-x3
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   ```
   *Open `.env` and set a strong `DJANGO_SECRET_KEY`.*

3. **Launch the System** *(migrations, seed data, and ANPR model warm-up all run automatically on the first boot — no manual `migrate` needed)*
   ```bash
   docker compose up --build -d
   ```

4. **Create an Admin Account** *(only if one does not already exist)*
   ```bash
   docker compose exec web python manage.py createsuperuser
   ```

   > **Demo note:** on this machine an account already exists — username `admin`, password `admin123` (change it before any public demo).

### ⏳ First-Boot Behaviour (read this)
- The first time the `web` container starts with an empty cache, it **downloads the ANPR model files** (~5 MB) and **preloads them into memory**. Until that finishes, the backend is briefly unavailable — give it **1–3 minutes** after `docker compose up` before testing.
- The model cache lives in Docker volume `trackx_web_cache`, so later restarts are fast and **never re-download**.
- Monitor startup with: `docker compose logs -f web`

### Accessing the System
| What | URL | Notes |
|---|---|---|
| **Command Center (UI)** | `http://localhost:3000` | The full React dashboard + Live Video page |
| **Plate/Trajectory search** | Type a plate in the dashboard search box | e.g. `DL 3C AF 9021` |
| **Admin Panel** | `http://localhost:8000/admin/` | Django admin: cameras, hotlist, logs |
| **API** | `http://localhost:8000/api/v1/` | Same API is served on `:3000/api/v1/...` too |

> **Ports explained:** `:3000` is the Nginx frontend (what your browser uses). `:8000` talks straight to the Django backend — useful for admin/API. Requests to `:3000` are internally proxied to the backend, so both are **one app**.

---

## 🕹️ 3. How to Use TrackX

### A. Managing the Infrastructure (Admin Panel)
Log into `http://localhost:8000/admin/` to:
- **Register Cameras:** Add `CameraNode` entries with their GPS coordinates.
- **Set the Hotlist:** Add vehicles to `BlacklistedVehicle`. Assign an **Alert Level**:
    - `CRITICAL`: Immediate dispatch required.
    - `WARNING`: Review detection.
    - `INFO`: General analytics.
- **Review Logs:** Browse every single detection captured by the AI (`DetectionLog`).

### B. Operating the Dashboard
1. **Real-time Monitoring:** Watch the **Anomaly Feed**. When a blacklisted car is spotted, a toast notification appears. Click **PIN TO MAP** to instantly zoom into that camera's location.
2. **Vehicle Tracking:** Enter a license plate in the search bar (e.g. `DL 3C AF 9021`). The map will draw a **Cyan Trajectory Line** showing everywhere that vehicle has been seen, along with a timeline of detections.
3. **City Analytics:** Check the right sidebar for average city speed and total detection counts to monitor urban traffic flow.

### C. Live Video ANPR Processing (new)
The Live Video page (also at `http://localhost:3000`) reads plates **directly from a video you upload** — this is the in-browser ANPR demo, no camera hardware needed:

1. On the **Live Video ANPR Processing** panel, click **Choose File** and pick any `.mp4` (e.g. `pexels-...mp4`).
2. Click **Start Live Processing**.
3. Within ~2 seconds the **first plate appears** in the *Detected License Plate* box, and a cyan bounding box (with ✓ when the plate is confirmed by temporal voting) is drawn over the video.
4. Adjust the screen size with **+ / −**; **Pause** stops processing, the **↓** button scrolls to the detection panel.

How it works under the hood:
- Each video frame is sent over a WebSocket to the backend, which runs YOLO + OCR and returns the plate + bounding box.
- **Flow control is built in:** only one frame is in flight at a time, so the connection stays healthy and plates stream at ~1–5 per second. There is no "Start" explosion — it self-throttles.
- If you see **"Backend WebSocket Disconnected"**, wait 1–3 seconds — or check the backend is up (`docker compose ps`) and reload the page (Ctrl+Shift+R).

### D. Testing the System (Developer Tools)
Use the built-in management commands to simulate a live environment:
```bash
# Run a full end-to-end demo (seeds data & triggers alerts)
docker compose exec web python manage.py run_teacher_demo

# Trigger a hotlist alert manually on an arbitrary plate
docker compose exec web python manage.py trigger_test_alert --plate "KDA 123A"

# Re-seed cameras, detection logs and the hotlist (idempotent)
docker compose exec web python manage.py seed_trackx
```

---

## ⚙️ 4. The ANPR Pipeline (Technical Flow)

1. **Ingestion:** A camera frame (or a live-video WebSocket frame) is handed to the detection pipeline.
2. **Localization:** **YOLOv8** identifies the license plate → crops the image. If it misses, a contour-based plate locator is tried.
3. **Transcription:** **fast-plate-ocr (ONNX)** transcribes the crop into a string (e.g., `DL3CAF9021`); EasyOCR is the fallback. The crop gets the deskew → 3x LANCZOS → grayscale → bilateral → CLAHE pipeline *without* the final unsharp-mask step (an A/B on real low-res footage showed the unsharp step degrading reads ~63% → ~70% correct without it). Per-character confidences from the ONNX model feed the temporal voter.
4. **Stability check:** **IoU tracking + temporal voting** accept the plate only after consistent multi-frame reads ("confirmed" plate). The voter weights each character by its per-character confidence when available.
5. **Validation:** The reading is checked against the `BlacklistedVehicle` table in **PostGIS**; if matched, an alert is broadcast over **WebSockets** → React Dashboard.
6. **Persistence:** Every confirmed/flagged detection is stored as a `DetectionLog` (camera, plate, timestamp, confidence), which feeds trajectory reconstruction and analytics.

### Why `v1.1` reads low-res footage better
- The fast-OCR input is the RAW crop passed through the heavy pipeline *minus* the unsharp mask. The unsharp mask helps synthetic clean renders but **hurts real blurry low-res plates** (measured: 63% → 70% correct reads on the `TS07JS9670` test window) because it amplifies sensor/compression noise.
- EasyOCR fallback still uses the fully-sharpened pipeline it was tuned for.
- Labeled synthetic-benchmark tooling lives in `scripts/eval_ocr.py` (exact + per-character accuracy over `data/syn_eval/`, with `--width` resolution-scaling); it tracked no regression for this change (w96 exact 18.3% → 19.2%, char 45.1% → 47.9%).

---

## 🛠️ 5. Operational Notes & Troubleshooting

### Container map
| Container | Role |
|---|---|
| `trackx-web` | Django ASGI (Daphne) — REST + WebSocket + **models preloaded in-process** |
| `trackx-frontend` | Nginx — SPA + proxy (`/api/`, `/ws/`, `/map/`) with dynamic DNS |
| `trackx-celery` | Celery worker — video/video-feed ANPR pipeline |
| `trackx-redis` | Broker for Celery + Channels |
| `trackx-db` | PostGIS 15 — `trackx_db` (user `postgres`, password `trackx_pass`) |

### Common commands
```bash
docker compose ps                 # health of all services
docker compose logs -f web        # follow backend logs (ANPR/Channels messages now logged)
docker compose logs -f frontend   # nginx access/errors
docker compose exec db psql -U postgres -d trackx_db   # inspect the database directly
docker restart trackx-frontend    # ONLY needed if you edited nginx.conf by hand
```

### Known gotchas & how they are handled
- **No more 502 after backend restarts:** Nginx resolves the `web` hostname via Docker's embedded DNS (`resolver 127.0.0.11`) per request, so recreated containers are picked up automatically — no manual frontend restart.
- **No more "WebSocket Disconnected" from model loading:** the ANPR models are preloaded into the Daphne process at startup (`trackx/asgi.py`), and Daphne's ping interval/timeout were raised (`docker/entrypoint.sh`) so slow frames never kill sockets.
- **Live-video sockets stay connected:** the live-video consumer disables the per-connection Redis channel-layer loop (`channel_layer_alias = None`) and the Redis channel layer uses `socket_timeout=None` — this removes the ~5s "Timeout reading from redis" socket drops on `trackx/settings.py`.
- **Large video frames are accepted:** Daphne's WebSocket max message/frame size was raised to 16 MB (`docker/entrypoint.sh`) so full-resolution video frames aren't rejected at the ~1 MB default.
- **Windows + WSL gotcha:** a WSL `wslrelay` process can hijack IPv6 `::1` port(s) on the host. The app is unaffected (it uses port `3000`), but scripts probing `localhost:8000` should use **`127.0.0.1`** instead.
- **Model cache volume:** ANPR weights are stored in `trackx_web_cache`; delete it only if you intentionally want re-downloads:
  ```bash
  docker compose down
  docker volume rm track-x3_trackx_web_cache
  ```

---

## 📈 6. Project Roadmap & Future Scope
- [ ] **Multi-Feed Support:** Processing multiple RTSP streams concurrently.
- [ ] **Predictive Analytics:** Predicting the next likely camera a vehicle will hit based on trajectory.
- [ ] **Advanced Filtering:** Filtering detections by vehicle color or type.
- [ ] **Mobile Alerts:** Integration with Push Notifications/SMS for critical alerts.