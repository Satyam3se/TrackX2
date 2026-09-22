# 🛰️ TrackX — Advanced Urban Vehicle Surveillance System

TrackX is a professional-grade, AI-powered vehicle tracking and surveillance ecosystem. It integrates real-time Computer Vision (CV), spatial databases, and asynchronous task processing to monitor vehicle movements across a city-wide network of cameras, providing instant alerts for blacklisted vehicles and deep trajectory analytics.

---

## 🚀 1. Core Technology Stack

TrackX is built on a modern, distributed architecture designed for scalability and low-latency response.

### 🧠 Artificial Intelligence (The ANPR Engine)
- **YOLOv8 (Ultralytics):** Used for **Object Detection**. It scans every camera frame to locate the precise bounding box of a vehicle's license plate.
- **EasyOCR:** Used for **Optical Character Recognition (OCR)**. Once YOLOv8 isolates the plate, EasyOCR transcribes the image into digital text (the license plate number).
- **Asynchronous Processing (Celery + Redis):** Video processing is computationally expensive. TrackX offloads the AI pipeline to background workers, ensuring the main application remains responsive.

### 🌐 Backend (The Orchestrator)
- **Django & Django REST Framework (DRF):** Provides a robust API for managing cameras, blacklisted vehicles, and detection logs.
- **Django Channels (ASGI):** Enables **WebSockets**, allowing the server to "push" critical alerts to the dashboard instantly without the user needing to refresh the page.
- **PostgreSQL + PostGIS:** A spatial database that stores not just data, but **geography**. It allows TrackX to perform complex spatial queries (e.g., "find all cameras within 1km of this detection").

### 🎨 Frontend (The Command Center)
- **React.js:** A high-performance UI framework for the real-time dashboard.
- **MapLibre GL / OpenStreetMap:** Provides an interactive map for visualizing camera nodes, vehicle trajectories, and traffic heatmaps.
- **Vite:** A lightning-fast build tool and development server.

### 🐳 Infrastructure
- **Docker & Docker Compose:** Entire system is containerized for "one-click" deployment, ensuring it runs the same on every teammate's machine.
- **Nginx:** Acts as a reverse proxy to serve the React frontend and route API requests to the Django backend.

---

## 🛠️ 2. Installation & Setup Guide

### Prerequisites
- **Docker Desktop** (Must be installed and running)
- **Git**

### Step-by-Step Deployment
1. **Clone the Repository**
   ```bash
   git clone https://github.com/Satyam3se/TrackX2
   cd track-x
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   ```
   *Open `.env` and set a strong `DJANGO_SECRET_KEY`.*

3. **Launch the System**
   ```bash
   docker compose up --build -d
   ```

4. **Initialize the Database**
   ```bash
   docker compose exec web python manage.py migrate
   docker compose exec web python manage.py createsuperuser
   ```

### Accessing the System
- **Command Center (UI):** `http://localhost:3000`
- **Admin Panel (Management):** `http://localhost:3000/admin/`
- **API Documentation:** `http://localhost:3000/api/v1/`

---

## 🕹️ 3. How to Use TrackX

### A. Managing the Infrastructure (Admin Panel)
Log into `http://localhost:3000/admin/` to:
- **Register Cameras:** Add `CameraNode` entries with their GPS coordinates.
- **Set the Hotlist:** Add vehicles to `BlacklistedVehicle`. Assign an **Alert Level**:
    - `CRITICAL`: Immediate dispatch required.
    - `WARNING`: Review detection.
    - `INFO`: General analytics.
- **Review Logs:** Browse every single detection captured by the AI.

### B. Operating the Dashboard
1. **Real-time Monitoring:** Watch the **Anomaly Feed**. When a blacklisted car is spotted, a toast notification appears. Click **PIN TO MAP** to instantly zoom into that camera's location.
2. **Vehicle Tracking:** Enter a license plate in the search bar. The map will draw a **Cyan Trajectory Line** showing everywhere that vehicle has been seen, along with a timeline of detections.
3. **City Analytics:** Check the right sidebar for average city speed and total detection counts to monitor urban traffic flow.

### C. Testing the System (Developer Tools)
Use the built-in management commands to simulate a live environment:
```bash
# Run a full end-to-end demo (Seeds data & triggers alerts)
docker compose exec web python manage.py run_teacher_demo
```

---

## ⚙️ 4. The ANPR Pipeline (Technical Flow)

1. **Ingestion:** A camera frame is sent to the `process_camera_frame` Celery task.
2. **Localization:** **YOLOv8** identifies the license plate $\rightarrow$ crops the image.
3. **Transcription:** **EasyOCR** converts the crop into a string (e.g., "ABC-1234").
4. **Validation:** The system checks the string against the `BlacklistedVehicle` table in **PostGIS**.
5. **Notification:** If a match is found, a message is sent via **WebSockets** $\rightarrow$ React Dashboard $\rightarrow$ User Alert.

---

## 📈 5. Project Roadmap & Future Scope
- [ ] **Multi-Feed Support:** Processing multiple RTSP streams concurrently.
- [ ] **Predictive Analytics:** Predicting the next likely camera a vehicle will hit based on trajectory.
- [ ] **Advanced Filtering:** Filtering detections by vehicle color or type.
- [ ] **Mobile Alerts:** Integration with Push Notifications/SMS for critical alerts.
