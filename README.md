# AIManthanBackend

FastAPI backend for 3D lattice inverse design, surrogate property prediction, and physics verification.

## Architecture

- **FastAPI Core & Routing (`server/app.py`)**: Asynchronous REST endpoints and SSE streaming.
- **Data Contracts (`server/schemas.py`)**: Strict Pydantic models for design targets, candidate lattices, physical evaluations, and job tracking.
- **Physical & Normalized Units (`server/units.py`)**: Conversion between physical dimensional targets (GPa, MPa, g, mm) and dimensionless surrogate features.
- **Job Store & Persistence (`server/store.py`)**: SQLite job status tracking and filesystem storage for candidate results.
- **Adaptive Escalation Pipeline (`server/jobs.py` & `core.py`)**:
  - GPU search offloaded to a thread pool executor.
  - Multi-state candidate physics verification (unsteered, 1/3, final) batched across process pool workers.
  - Undirected edge hashing and geometric deduplication.
  - Adaptive $K$-escalation loop ($K = 100, 300, 1000$) with dynamic SSE progress streaming.

## API Endpoints

- `GET /api/health`: Health status.
- `GET /api/materials`: Available materials and mechanical properties.
- `POST /api/designs`: Submit design requirements and initiate background job (HTTP 202).
- `GET /api/jobs/{job_id}`: Query job progress and generated candidates.
- `GET /api/jobs/{job_id}/events`: Server-Sent Events (SSE) live progress stream.
- `GET /api/designs/{design_id}`: Retrieve detailed candidate lattice geometry and verified properties.

## Setup & Running

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Running Tests

```bash
pytest
```
