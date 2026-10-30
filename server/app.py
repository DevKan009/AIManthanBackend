from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import AsyncGenerator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

try:
    from server.jobs import active_sse_queues, run_pipeline
    from server.schemas import (
        AxisValues,
        Candidate,
        DesignRequest,
        FailureMode,
        FailureModeAxis,
        HealthResponse,
        JobCreateResponse,
        JobResponse,
        LatticeSpec,
        Material,
        PropertyEvaluation,
    )
    from server.store import create_job, get_job, init_db
except ImportError:
    from jobs import active_sse_queues, run_pipeline
    from schemas import (
        AxisValues,
        Candidate,
        DesignRequest,
        FailureMode,
        FailureModeAxis,
        HealthResponse,
        JobCreateResponse,
        JobResponse,
        LatticeSpec,
        Material,
        PropertyEvaluation,
    )
    from store import create_job, get_job, init_db

# Initialize executor pools for physics solver and GPU search
WORKERS = int(os.environ.get("WORKERS", max(1, (os.cpu_count() or 2) - 1)))
process_pool = ProcessPoolExecutor(max_workers=WORKERS)
thread_pool = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context initializing SQLite on startup and cleaning up pools."""
    init_db()
    yield
    process_pool.shutdown(wait=False)
    thread_pool.shutdown(wait=False)


app = FastAPI(
    title="Lattice Design ML/Physics Engine Backend",
    description="Backend API for 3D lattice inverse design, surrogate property prediction, and physics verification.",
    version="0.1.0",
    lifespan=lifespan,
)

# Initialize database at module load time as well
init_db()

# CORS middleware allowing all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
async def get_health() -> dict:
    """Backend service health check."""
    return {"status": "ok", "model_loaded": False, "gpu": False}


@app.get("/api/materials", response_model=list[Material])
async def get_materials() -> list:
    """List supported materials with mechanical properties."""
    return [
        {
            "id": "ti64",
            "name": "Ti-6Al-4V",
            "E_gpa": 114,
            "yield_mpa": 880,
            "density_g_cm3": 4.43,
        }
    ]


@app.post(
    "/api/designs",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobCreateResponse,
)
async def create_design(
    payload: DesignRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """Generate a UUID, insert job into SQLite, start background simulation, and return 202."""
    job_id = f"j_{uuid.uuid4().hex[:8]}"

    create_job(
        job_id=job_id,
        status="queued",
        stage=None,
        progress=0.0,
        message="Job queued for processing",
    )

    if job_id not in active_sse_queues:
        active_sse_queues[job_id] = asyncio.Queue()

    background_tasks.add_task(run_pipeline, job_id, payload)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
async def get_job_endpoint(job_id: str) -> JobResponse:
    """Query current status, stage, progress, and results of a design job from SQLite."""
    job_data = get_job(job_id)
    if not job_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )
    return JobResponse(**job_data)


@app.get("/api/jobs/{job_id}/events")
async def get_job_events(job_id: str) -> EventSourceResponse:
    """Server-Sent Events (SSE) stream reporting real-time job stage and progress updates."""
    job = get_job(job_id)

    # If the job is already done or failed in SQLite, yield a final close message and exit
    if job and job.get("status") in ("done", "failed"):
        async def closed_generator():
            yield {
                "event": "close",
                "data": json.dumps({
                    "job_id": job_id,
                    "status": job["status"],
                    "stage": job.get("stage"),
                    "progress": job.get("progress", 1.0),
                    "message": f"Job is already {job['status']}.",
                }),
            }

        return EventSourceResponse(closed_generator())

    if job_id not in active_sse_queues:
        active_sse_queues[job_id] = asyncio.Queue()

    async def event_generator():
        queue = active_sse_queues[job_id]
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
                yield {"data": json.dumps(event)}

                if event.get("status") in ("done", "failed"):
                    yield {
                        "event": "close",
                        "data": json.dumps({
                            "job_id": job_id,
                            "status": event["status"],
                            "message": "Job finished",
                        }),
                    }
                    break
            except TimeoutError:
                # Check database status in case of queue timeouts
                current_job = get_job(job_id)
                if current_job and current_job.get("status") in ("done", "failed"):
                    yield {
                        "event": "close",
                        "data": json.dumps({
                            "job_id": job_id,
                            "status": current_job["status"],
                            "message": f"Job finished ({current_job['status']}).",
                        }),
                    }
                    break

    return EventSourceResponse(event_generator())


def _find_candidate_in_runs(design_id: str) -> Candidate | None:
    """Helper to locate a saved candidate from disk runs."""
    for results_file in Path("runs").glob("*/results.json"):
        try:
            with open(results_file, encoding="utf-8") as f:
                cands = json.load(f)
                for c in cands:
                    if isinstance(c, dict) and c.get("id") == design_id:
                        return Candidate(**c)
        except (OSError, json.JSONDecodeError):
            pass
    return None


@app.get("/api/designs/{design_id}", response_model=Candidate)
async def get_design(design_id: str) -> Candidate:
    """Retrieve details, lattice geometry, predicted, and verified properties of a candidate design."""
    saved_cand = _find_candidate_in_runs(design_id)
    if saved_cand is not None:
        return saved_cand

    # Fallback to default dummy candidate structure
    return Candidate(
        id=design_id,
        rank=1,
        match_error_pct=4.2,
        rel_density=0.12,
        mass_g=54.1,
        lattice=LatticeSpec(
            format="octant27",
            nodes=[[0.0, 0.0, 0.0]],
            edges=[[0, 0]],
            cell_size_mm=10.0,
            strut_radius_mm=0.41,
        ),
        predicted=PropertyEvaluation(
            stiffness_gpa=AxisValues(x=8.0, y=8.0, z=12.0),
            yield_strength_mpa=AxisValues(x=90.0, y=90.0, z=None),
            buckling_strength_mpa=AxisValues(x=100.0, y=100.0, z=120.0),
            strength_mpa=AxisValues(x=90.0, y=90.0, z=None),
            governing_failure=FailureModeAxis(
                x=FailureMode.YIELD, y=FailureMode.YIELD, z=FailureMode.YIELD
            ),
        ),
        verified=PropertyEvaluation(
            stiffness_gpa=AxisValues(x=8.0, y=8.0, z=12.0),
            yield_strength_mpa=AxisValues(x=90.0, y=90.0, z=None),
            buckling_strength_mpa=AxisValues(x=100.0, y=100.0, z=120.0),
            strength_mpa=AxisValues(x=90.0, y=90.0, z=None),
            governing_failure=FailureModeAxis(
                x=FailureMode.YIELD, y=FailureMode.YIELD, z=FailureMode.YIELD
            ),
        ),
    )
