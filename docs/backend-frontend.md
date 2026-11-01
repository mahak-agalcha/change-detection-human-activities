# Part 4: Backend, Frontend and Database

Web application for the "Change Detection Due to Human Activities Using
Satellite Imagery" project.

## Stack
- Backend: FastAPI (Python)
- Database: SQLite via SQLAlchemy (PostgreSQL planned for deployment)
- Frontend: React (Vite) + Leaflet map
- Reports: ReportLab (PDF)

## Features
- Upload old and new satellite images (validated: type, size, real image)
- Run analysis and store detected change regions
- View both images, changes table, map and statistics
- History of past analyses
- Download PDF report

## Run locally (Windows)
Backend:
```
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```
Frontend (new terminal):
```
cd frontend
npm install
npm run dev
```
Open http://localhost:5173 (API docs: http://127.0.0.1:8000/docs).

## API
| Method | Endpoint | Purpose |
|---|---|---|
| POST | /analyses | Upload old_image and new_image |
| POST | /analyses/{id}/run | Run inference, save regions |
| GET | /analyses | List past analyses |
| GET | /analyses/{id} | Result with regions (GeoJSON) |
| GET | /analyses/{id}/report | Download PDF report |
| GET | /analyses/stats/summary | Summary statistics |

## Model integration contract
`backend/app/services/inference.py` -> `run_inference(old_path, new_path)`
must return a list of dicts:
- `object_type`: "building" | "road" | "vehicle"
- `cause_label`: "human" | "natural" | "unknown"
- `confidence`: float between 0 and 1
- `geometry`: GeoJSON geometry in (lon, lat)

## Current limitations
- Inference is a **stub** returning placeholder data until the ML model
  is integrated. Placeholder results are not real detections.
- Map coordinates are placeholders. Real coordinates need georeferenced
  images (e.g. GeoTIFF).
- Images show what changed, not why. "Unknown" is a valid cause result.

## Screenshots
See `docs/screenshots/`.