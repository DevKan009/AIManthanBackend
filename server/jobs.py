"""Background pipeline execution with ML surrogate search, physics verification, and adaptive escalation."""

from __future__ import annotations

import asyncio
import hashlib
from concurrent.futures import Executor
from typing import Any

try:
    import core
except ImportError:
    from server import core

try:
    from server.schemas import DesignRequest
    from server.store import save_results, update_job_status
except ImportError:
    from schemas import DesignRequest
    from store import save_results, update_job_status

# Global dictionary mapping job_id to event queue for SSE streaming
active_sse_queues: dict[str, asyncio.Queue] = {}

# Material library fallback
MATERIALS_DB: dict[str, dict[str, Any]] = {
    "ti64": {
        "id": "ti64",
        "name": "Ti-6Al-4V",
        "E_gpa": 114.0,
        "yield_mpa": 880.0,
        "density_g_cm3": 4.43,
    }
}


def hash_lattice_edges(edges: list[list[int]] | list[tuple[int, int]]) -> str:
    """Compute canonical hash of undirected lattice edges to detect geometric duplicates."""
    canonical_edges = tuple(sorted({(min(e[0], e[1]), max(e[0], e[1])) for e in edges if len(e) >= 2}))
    return hashlib.sha256(str(canonical_edges).encode("utf-8")).hexdigest()


def drop_geometric_duplicates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hash the lattice edges and drop geometric duplicates."""
    seen_hashes: set[str] = set()
    unique: list[dict[str, Any]] = []

    for cand in candidates:
        lattice = cand.get("lattice")
        if not lattice and "lattice_states" in cand:
            states = cand["lattice_states"]
            first_state = next(iter(states.values())) if isinstance(states, dict) else states[0]
            lattice = first_state.get("lattice")

        edges = lattice.get("edges", []) if isinstance(lattice, dict) else []
        edge_hash = hash_lattice_edges(edges)
        if edge_hash not in seen_hashes:
            seen_hashes.add(edge_hash)
            unique.append(cand)

    return unique


def chunk_candidates(candidates: list[Any], batch_size: int = 50) -> list[list[Any]]:
    """Chunk the candidates into batches of 50 to avoid overloading ProcessPool with individual tasks."""
    return [candidates[i : i + batch_size] for i in range(0, len(candidates), batch_size)]


def verify_batch_worker(
    batch: list[dict[str, Any]],
    material_dict: dict[str, Any],
    target_dict: dict[str, Any],
) -> list[dict[str, Any]]:
    """Worker function executed inside ProcessPoolExecutor.

    Sequentially evaluates core.verify on each candidate across its 3 states
    (unsteered, 1/3, final), selects the state with the lowest match error,
    and returns resolved single-candidate models.
    """
    verified_results = []
    for cand in batch:
        lattice_states = cand.get("lattice_states") or [cand.get("lattice")]
        rel_density = cand.get("rel_density", 0.15)

        evaluated_states = core.verify(
            lattice_states=lattice_states,
            rel_density=rel_density,
            material=material_dict,
            target=target_dict,
        )

        # For each candidate's 3 states, keep only the state with the lowest match error
        best_state = min(evaluated_states, key=lambda s: s.get("match_error_pct", float("inf")))

        # Fulfill single-candidate API contract
        resolved_cand = {
            "id": cand["id"],
            "rank": 0,
            "match_error_pct": best_state["match_error_pct"],
            "rel_density": rel_density,
            "mass_g": cand.get("mass_g", 54.1),
            "lattice": best_state["lattice"],
            "predicted": best_state.get("predicted") or {},
            "verified": best_state["verified"],
        }
        verified_results.append(resolved_cand)

    return verified_results


async def run_pipeline(
    job_id: str,
    request: DesignRequest | dict[str, Any] | None = None,
    process_pool: Executor | None = None,
    thread_pool: Executor | None = None,
    k_values: list[int] | None = None,
    request_data: dict[str, Any] | None = None,
    stage_delay: float | None = None,
) -> None:
    """Execute the adaptive escalation search and physics verification pipeline.

    Loops over k_values = [100, 300, 1000].
    Offloads core.search to ThreadPoolExecutor and batched core.verify to ProcessPoolExecutor.
    Breaks loop early if tolerance is met, dropping geometric duplicates and ranking final candidates.
    """
    # Ensure active queue exists for SSE events
    if job_id not in active_sse_queues:
        active_sse_queues[job_id] = asyncio.Queue()

    req_payload = request if request is not None else (request_data or {})
    if isinstance(req_payload, dict):
        if "part" not in req_payload:
            req_payload = {
                "material": req_payload.get("material", "ti64"),
                "part": {"size_mm": [40.0, 40.0, 40.0], "cell_mm": 10.0},
                "target": req_payload.get("target", {}),
                "modes": req_payload.get("modes", {}),
                "tolerance_pct": req_payload.get("tolerance_pct", 10.0),
                "n_results": req_payload.get("n_results", 5),
            }
        request = DesignRequest(**req_payload)

    if k_values is None:
        k_values = [100, 300, 1000]

    # Resolve executors
    if process_pool is None or thread_pool is None:
        try:
            from server.app import process_pool as app_p_pool
            from server.app import thread_pool as app_t_pool
            process_pool = process_pool or app_p_pool
            thread_pool = thread_pool or app_t_pool
        except (ImportError, AttributeError):
            from concurrent.futures import ThreadPoolExecutor
            process_pool = process_pool or ThreadPoolExecutor(max_workers=4)
            thread_pool = thread_pool or ThreadPoolExecutor(max_workers=1)

    loop = asyncio.get_running_loop()

    # Resolve material properties
    material_key = request.material.lower()
    material_dict = MATERIALS_DB.get(material_key, {
        "id": request.material,
        "name": request.material,
        "E_gpa": 114.0,
        "yield_mpa": 880.0,
        "density_g_cm3": 4.43,
    })

    target_dict = request.target.model_dump() if hasattr(request.target, "model_dump") else request.target

    all_verified_candidates: list[dict[str, Any]] = []
    tolerance_met = False

    try:
        for k_idx, k in enumerate(k_values):
            # Dynamic stage progress calculation
            # k=100 -> search: 0.1, repair: 0.15, verify: 0.25, escalation: 0.3
            # k=300 -> search: 0.4, repair: 0.45, verify: 0.55, escalation: 0.6
            # k=1000 -> search: 0.7, repair: 0.75, verify: 0.85
            base_p = 0.3 * k_idx
            p_search = round(base_p + 0.10, 2)
            p_repair = round(base_p + 0.15, 2)
            p_verify = round(base_p + 0.25, 2)
            p_escalate = round(base_p + 0.30, 2)

            # ---------------------------------------------------------------
            # 1. Search stage (offloaded to ThreadPool for GPU PyTorch)
            # ---------------------------------------------------------------
            search_msg = f"GPU surrogate searching {k} candidate lattice topologies..."
            update_job_status(job_id=job_id, status="running", stage="search", progress=p_search, message=search_msg)
            await active_sse_queues[job_id].put({"stage": "search", "progress": p_search, "message": search_msg})

            raw_candidates = await loop.run_in_executor(
                thread_pool,
                core.search,
                target_dict,
                k,
            )

            # ---------------------------------------------------------------
            # 2. Repair stage (deduplicate and filter topology)
            # ---------------------------------------------------------------
            repair_msg = f"Deduplicating geometric topologies among {len(raw_candidates)} candidates..."
            update_job_status(job_id=job_id, status="running", stage="repair", progress=p_repair, message=repair_msg)
            await active_sse_queues[job_id].put({"stage": "repair", "progress": p_repair, "message": repair_msg})

            unique_candidates = drop_geometric_duplicates(raw_candidates)

            # ---------------------------------------------------------------
            # 3. Verification stage (batched to ProcessPoolExecutor)
            # ---------------------------------------------------------------
            verify_msg = f"Physics solver verifying {len(unique_candidates)} candidates in batches of 50..."
            update_job_status(job_id=job_id, status="running", stage="verify", progress=p_verify, message=verify_msg)
            await active_sse_queues[job_id].put({"stage": "verify", "progress": p_verify, "message": verify_msg})

            batches = chunk_candidates(unique_candidates, batch_size=50)

            # Offload each batch to ProcessPool
            batch_tasks = [
                loop.run_in_executor(
                    process_pool,
                    verify_batch_worker,
                    batch,
                    material_dict,
                    target_dict,
                )
                for batch in batches
            ]

            batch_results = await asyncio.gather(*batch_tasks)
            verified_batch = [cand for sublist in batch_results for cand in sublist]

            # Filter duplicates again after single-candidate state resolution
            verified_unique = drop_geometric_duplicates(verified_batch)
            all_verified_candidates.extend(verified_unique)

            # ---------------------------------------------------------------
            # 4. Tolerance check & adaptive escalation
            # ---------------------------------------------------------------
            matching_in_this_k = [
                cand for cand in verified_unique
                if cand.get("match_error_pct", float("inf")) <= request.tolerance_pct
            ]

            if matching_in_this_k:
                tolerance_met = True
                break

            # If tolerance not met and there are further k levels, escalate
            if k_idx < len(k_values) - 1:
                next_k = k_values[k_idx + 1]
                esc_msg = f"Tolerance not met. Escalating to {next_k} candidates..."
                update_job_status(job_id=job_id, status="running", stage="search", progress=p_escalate, message=esc_msg)
                await active_sse_queues[job_id].put({
                    "stage": "search",
                    "progress": p_escalate,
                    "message": esc_msg,
                })

        # -------------------------------------------------------------------
        # 5. Rank & Save stage
        # -------------------------------------------------------------------
        update_job_status(job_id=job_id, status="running", stage="rank", progress=0.95, message="Ranking top candidates...")
        await active_sse_queues[job_id].put({"stage": "rank", "progress": 0.95, "message": "Ranking top candidates..."})

        # Deduplicate all accumulated candidates
        unique_final = drop_geometric_duplicates(all_verified_candidates)
        # Sort by match error percentage ascending
        unique_final.sort(key=lambda c: c.get("match_error_pct", float("inf")))

        n_results = request.n_results or 5
        top_candidates = unique_final[:n_results]

        # Assign ranks
        for rank_idx, cand in enumerate(top_candidates, start=1):
            cand["rank"] = rank_idx

        # Persist results to runs/{job_id}/results.json
        save_results(job_id, top_candidates)

        final_msg = (
            f"Pipeline complete. Met tolerance ({request.tolerance_pct}%) with top candidate error "
            f"{top_candidates[0]['match_error_pct']}%."
            if top_candidates and tolerance_met
            else f"Pipeline complete. Top {len(top_candidates)} candidates ranked and saved."
        )

        update_job_status(
            job_id=job_id,
            status="done",
            stage="rank",
            progress=1.0,
            message=final_msg,
        )
        await active_sse_queues[job_id].put({
            "status": "done",
            "stage": "rank",
            "progress": 1.0,
            "message": final_msg,
        })

    except Exception as e:  # noqa: BLE001 - catch-all for background task
        err_msg = f"Pipeline execution failed: {e!s}"
        update_job_status(job_id=job_id, status="failed", message=err_msg)
        if job_id in active_sse_queues:
            await active_sse_queues[job_id].put({"status": "failed", "message": err_msg})


# Backward compatibility alias
run_mock_pipeline = run_pipeline
