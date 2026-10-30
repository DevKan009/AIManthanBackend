import asyncio
import json
from pathlib import Path

import pytest

from server.jobs import active_sse_queues, run_mock_pipeline
from server.store import (
    create_job,
    get_db_connection,
    get_job,
    get_job_results,
    init_db,
    save_results,
    update_job_status,
)


@pytest.fixture
def temp_store(tmp_path):
    """Fixture providing a temporary SQLite db and runs directory."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    db_path = runs_dir / "jobs.db"
    init_db(db_path=db_path)
    return {"runs_dir": runs_dir, "db_path": db_path}


def test_init_db_creates_table(temp_store):
    """Verify init_db creates jobs table with all expected columns."""
    db_path = temp_store["db_path"]
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute("PRAGMA table_info(jobs)")
        columns = {row["name"]: row["type"] for row in cursor.fetchall()}
    finally:
        conn.close()

    expected_columns = {
        "job_id": "TEXT",
        "status": "TEXT",
        "stage": "TEXT",
        "progress": "REAL",
        "message": "TEXT",
        "updated_at": "TIMESTAMP",
    }
    for col, col_type in expected_columns.items():
        assert col in columns
        assert columns[col] == col_type


def test_create_and_get_job(temp_store):
    """Verify create_job inserts record and get_job retrieves it."""
    db_path = temp_store["db_path"]
    job_id = "j_test_create"

    created = create_job(
        job_id=job_id,
        status="queued",
        stage=None,
        progress=0.0,
        message="Queued test job",
        db_path=db_path,
    )
    assert created["job_id"] == job_id
    assert created["status"] == "queued"
    assert created["stage"] is None
    assert created["progress"] == 0.0

    retrieved = get_job(job_id, db_path=db_path)
    assert retrieved is not None
    assert retrieved["job_id"] == job_id
    assert retrieved["status"] == "queued"
    assert retrieved["message"] == "Queued test job"
    assert "updated_at" in retrieved
    assert retrieved["candidates"] == []


def test_update_job_status(temp_store):
    """Verify update_job_status updates fields and timestamp."""
    db_path = temp_store["db_path"]
    job_id = "j_test_update"
    create_job(job_id=job_id, status="queued", db_path=db_path)

    updated = update_job_status(
        job_id=job_id,
        status="running",
        stage="verify",
        progress=0.75,
        message="Running FEA verification",
        db_path=db_path,
    )
    assert updated is not None
    assert updated["status"] == "running"
    assert updated["stage"] == "verify"
    assert pytest.approx(updated["progress"], rel=1e-5) == 0.75
    assert updated["message"] == "Running FEA verification"


def test_save_and_get_results(temp_store):
    """Verify save_results writes JSON list to runs/{job_id}/results.json."""
    runs_dir = temp_store["runs_dir"]
    job_id = "j_test_results"

    dummy_candidates = [
        {"id": "cand_1", "rank": 1, "mass_g": 50.0},
        {"id": "cand_2", "rank": 2, "mass_g": 52.5},
    ]

    saved_path = save_results(job_id, dummy_candidates, base_dir=runs_dir)
    assert saved_path.exists()
    assert saved_path == runs_dir / job_id / "results.json"

    with open(saved_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert len(loaded) == 2
    assert loaded[0]["id"] == "cand_1"

    results = get_job_results(job_id, base_dir=runs_dir)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_run_mock_pipeline_execution():
    """Verify run_mock_pipeline executes 4 stages, updates SQLite, pushes to SSE queue, and writes results."""
    init_db()
    job_id = "j_mock_pipeline_test"
    create_job(job_id=job_id, status="queued")

    active_sse_queues[job_id] = asyncio.Queue()

    # Run mock pipeline with very fast delay
    await run_mock_pipeline(
        job_id=job_id,
        request_data={"material": "ti64"},
        stage_delay=0.02,
    )

    # Verify SQLite status marked as done
    job = get_job(job_id)
    assert job is not None
    assert job["status"] == "done"
    assert job["stage"] == "rank"
    assert pytest.approx(job["progress"], rel=1e-5) == 1.0

    # Verify 5 candidates saved in results.json
    results_file = Path("runs") / job_id / "results.json"
    assert results_file.exists()
    candidates = get_job_results(job_id)
    assert len(candidates) == 5

    # Verify SSE queue received events for the 4 stages plus completion
    received_events = []
    while not active_sse_queues[job_id].empty():
        received_events.append(await active_sse_queues[job_id].get())

    stages_seen = [e.get("stage") for e in received_events if "stage" in e]
    assert "search" in stages_seen
    assert "repair" in stages_seen
    assert "verify" in stages_seen
    assert "rank" in stages_seen
