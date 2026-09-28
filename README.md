# 🛰️ TrackX — Advanced Urban Vehicle Surveillance System

[![Docker](https://img.shields.io/badge/Docker-Docker%20Compose-blue?logo=docker)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-Rest%20Framework-green?logo=django)](https://www.djangoproject.com/)
[![React](https://img.shields.io/badge/React-18%2B-black?logo=react)](https://reactjs.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red?logo=ultralytics)](https://github.com/ultralytics/ultralytics)
[![PostGIS](https://img.shields.io/badge/PostGIS-Spatial%20DB-orange?logo=postgresql)](https://postgis.net/)

TrackX is a professional-grade, AI-powered vehicle tracking and surveillance ecosystem. It integrates real-time Computer Vision (CV), spatial databases, and asynchronous task processing to monitor vehicle movements across a city-wide network of cameras, providing instant alerts for blacklisted vehicles and deep trajectory analytics.

## ✨ Features

- **Real-time ANPR** — Automatic Number Plate Recognition using YOLOv8 detection + fast-plate-ocr (ONNX) transcription with temporal voting for noise filtering
- **Live Video Processing** — Upload any `.mp4` and watch plates get transcribed in real-time via WebSockets
- **Blacklisted Vehicle Hotlist** — Instant WebSocket alerts with severity levels (CRITICAL, WARNING, INFO)
- **Vehicle Trajectory Mapping** — Visualize a vehicle's entire movement history across city camera nodes on an interactive map
- **City Analytics Dashboard** — Real-time traffic flow stats, average speeds, and detection heatmaps
- **Spatial Queries** — PostGIS-powered geographic queries (e.g., "find all cameras within 1km of this detection")
- **Async Pipeline** — Celery + Redis offloads computationally heavy AI processing from the main application
- **Containerized Deployment** — Fully Dockerized for one-click, reproducible deployment

## 🚀 Technology Stack

| Layer | Technology |
|---|---|
| **AI/ML** | YOLOv8 (object detection), fast-plate-ocr ONNX + EasyOCR (OCR), temporal voting + IoU tracking |
| **Backend** | Django & Django REST Framework, Django Channels (WebSockets), ASGI (Daphne) |
| **Database** | PostgreSQL + PostGIS (spatial database) |
| **Task Queue** | Celery + Redis |
| **Frontend** | React.js, MapLibre GL / OpenStreetMap, Vite |
| **Infrastructure** | Docker & Docker Compose, Nginx (reverse proxy + static serving) |

## 🛠️ Getting Started

### Prerequisites
- **Docker Desktop** (required)
- **Git**

### Quick Start
```bash
git clone https://github.com/Satyam3se/TrackX2
cd track-x3
cp .env.example .env
docker compose up --build -d
```

### Access the System
| What | URL |
|---|---|
| **Dashboard (UI)** | `http://localhost:3000` |
| **Admin Panel** | `http://localhost:8000/admin/` |
| **API** | `http://localhost:8000/api/v1/` |

> **First boot:** The ANPR model files (~5 MB) download automatically. Wait **1–3 minutes** after `docker compose up` before testing.
> **Demo credentials:** `admin` / `admin123` (change before any public demo).

## 🕹️ Usage

### Real-time Monitoring
Watch the **Anomaly Feed** on the dashboard. When a blacklisted car is spotted, a toast notification appears instantly with a **PIN TO MAP** button to zoom into the camera location.

### Vehicle Tracking
Enter a license plate (e.g., `DL 3C AF 9021`) in the search bar. The map draws a **Cyan Trajectory Line** showing every location the vehicle has been seen.

### Live Video ANPR
1. Go to the **Live Video** page
2. Upload any `.mp4` file
3. Click **Start Live Processing**
4. Watch plates appear in real-time with bounding boxes (✓ = confirmed by temporal voting)

### Admin Panel
- Register cameras with GPS coordinates
- Manage the blacklisted vehicle hotlist with alert levels
- Browse all detection logs

## 📁 Project Structure

```
track-x3/
├── anpr_engine/          # Core ANPR app (detection, OCR, tracking, WebSocket consumers)
├── trackx/               # Django project settings, URLs, ASGI config
├── dashboard/            # Dashboard Django app (admin, views, templates)
├── frontend/             # React.js dashboard (Vite)
├── docker/               # Docker configs & entrypoints
├── scripts/              # Training, evaluation & utility scripts
├── notebooks/            # Exploratory Jupyter notebooks
├── static/               # Django static assets
├── pretrained_weights/   # Model weights (YOLOv8 base, plate detectors)
├── data/                 # Datasets, training runs, test videos (gitignored)
│   └── syn_eval/         #   labelled OCR benchmark (small, so tracked)
├── manage.py             # Django management script
├── requirements.txt      # Python dependencies
├── docker-compose.yml    # Multi-container orchestration
└── Dockerfile            # Application container image
```

> **Note:** everything under `data/` is gitignored **except** `data/syn_eval/`
> (small + labelled, so it stays in version control), as is
> `pretrained_weights/` (large binaries). Clone fresh and populate those before
> training — see [TRACKX_GUIDE.md](./TRACKX_GUIDE.md). Scripts under `scripts/`
> use paths relative to the repo root, so run them as
> `python scripts/<name>.py` from there.

## 🔧 Management Commands
```bash
# Run a full end-to-end demo
docker compose exec web python manage.py run_teacher_demo

# Trigger a hotlist alert manually
docker compose exec web python manage.py trigger_test_alert --plate "KDA 123A"

# Re-seed cameras, detection logs, and hotlist
docker compose exec web python manage.py seed_trackx
```

## 🐳 Common Commands
```bash
docker compose ps              # Check service health
docker compose logs -f web     # Follow backend logs
docker compose exec db psql -U postgres -d trackx_db  # Direct database access
```

## 📈 Roadmap
- [ ] Multi-Feed Support (concurrent RTSP streams)
- [ ] Predictive Analytics (next camera prediction)
- [ ] Advanced Filtering (vehicle color/type)
- [ ] Mobile Alerts (Push/SMS notifications)

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🔗 References
- [TRACKX_GUIDE.md](./TRACKX_GUIDE.md) — Detailed setup guide and troubleshooting
- [Issue Tracker](https://github.com/Satyam3se/TrackX2/issues)
- [Contributing](https://github.com/Satyam3se/TrackX2/blob/main/CONTRIBUTING.md)
