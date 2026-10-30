"""Persistence layer using SQLite and local filesystem storage for design jobs and candidate results."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_RUNS_DIR = Path("runs")
DEFAULT_DB_PATH = DEFAULT_RUNS_DIR / "jobs.db"


def get_db_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create a configured SQLite connection with row access."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Set up the SQLite database file (runs/jobs.db) with the jobs table.

    Columns: job_id (TEXT PK), status (TEXT), stage (TEXT), progress (REAL), message (TEXT), updated_at (TIMESTAMP).
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_db_connection(path)
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    stage TEXT,
                    progress REAL DEFAULT 0.0,
                    message TEXT,
                    updated_at TIMESTAMP NOT NULL
                )
                """
            )
    finally:
        conn.close()


def create_job(
    job_id: str,
    status: str = "queued",
    stage: str | None = None,
    progress: float = 0.0,
    message: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Insert or replace a job record into SQLite."""
    now_iso = datetime.now(UTC).isoformat()
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, status, stage, progress, message, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status = excluded.status,
                    stage = excluded.stage,
                    progress = excluded.progress,
                    message = excluded.message,
                    updated_at = excluded.updated_at
                """,
                (job_id, status, stage, float(progress), message, now_iso),
            )
    finally:
        conn.close()

    return {
        "job_id": job_id,
        "status": status,
        "stage": stage,
        "progress": float(progress),
        "message": message,
        "updated_at": now_iso,
    }


def update_job_status(
    job_id: str,
    status: str | None = None,
    stage: str | None = None,
    progress: float | None = None,
    message: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any] | None:
    """Update job fields and updated_at timestamp in SQLite, inserting if missing."""
    now_iso = datetime.now(UTC).isoformat()
    conn = get_db_connection(db_path)
    try:
        # Check if record exists
        row = conn.execute("SELECT job_id FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if not row:
            conn.close()
            return create_job(
                job_id=job_id,
                status=status or "running",
                stage=stage,
                progress=progress or 0.0,
                message=message,
                db_path=db_path,
            )

        fields: list[str] = []
        params: list[Any] = []

        if status is not None:
            fields.append("status = ?")
            params.append(status)
        if stage is not None:
            fields.append("stage = ?")
            params.append(stage)
        if progress is not None:
            fields.append("progress = ?")
            params.append(float(progress))
        if message is not None:
            fields.append("message = ?")
            params.append(message)

        fields.append("updated_at = ?")
        params.append(now_iso)
        params.append(job_id)

        with conn:
            conn.execute(
                f"UPDATE jobs SET {', '.join(fields)} WHERE job_id = ?",
                params,
            )
    finally:
        conn.close()

    return get_job(job_id, db_path=db_path)


def save_results(
    job_id: str,
    candidates: list[dict[str, Any]],
    base_dir: str | Path = DEFAULT_RUNS_DIR,
) -> Path:
    """Save the final candidate JSON list to runs/{job_id}/results.json."""
    results_dir = Path(base_dir) / job_id
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)
    return results_file


def get_job_results(
    job_id: str,
    base_dir: str | Path = DEFAULT_RUNS_DIR,
) -> list[dict[str, Any]]:
    """Load candidates from runs/{job_id}/results.json if it exists."""
    results_file = Path(base_dir) / job_id / "results.json"
    if not results_file.exists():
        return []
    try:
        with open(results_file, encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def get_job(
    job_id: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    include_candidates: bool = True,
) -> dict[str, Any] | None:
    """Retrieve job details by job_id from SQLite, along with any saved candidates."""
    conn = get_db_connection(db_path)
    try:
        row = conn.execute(
            "SELECT job_id, status, stage, progress, message, updated_at FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if not row:
            return None
        job_data = {
            "job_id": row["job_id"],
            "status": row["status"],
            "stage": row["stage"],
            "progress": float(row["progress"]) if row["progress"] is not None else 0.0,
            "message": row["message"],
            "updated_at": row["updated_at"],
        }
    finally:
        conn.close()

    if include_candidates:
        runs_dir = Path(db_path).parent
        job_data["candidates"] = get_job_results(job_id, base_dir=runs_dir)

    return job_data
