from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TargetMode(str, Enum):
    """Optimization target comparison mode."""
    ABOUT = "about"
    AT_LEAST = "at_least"
    AT_MOST = "at_most"
    EXACT = "exact"
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class JobStatus(str, Enum):
    """Lifecycle status of a design search and verification job."""
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobStage(str, Enum):
    """Processing stage of a running design job."""
    SEARCH = "search"
    REPAIR = "repair"
    VERIFY = "verify"
    RANK = "rank"


class FailureMode(str, Enum):
    """Governing structural failure mode."""
    YIELD = "yield"
    BUCKLING = "buckling"


# ---------------------------------------------------------------------------
# Common / Directional Models
# ---------------------------------------------------------------------------

class AxisValues(BaseModel):
    """Directional 3D physical property values (axes can be None)."""
    x: float | None = None
    y: float | None = None
    z: float | None = None


class FailureModeAxis(BaseModel):
    """Governing failure mode per spatial axis ("yield", "buckling", or None)."""
    x: FailureMode | None = None
    y: FailureMode | None = None
    z: FailureMode | None = None


# ---------------------------------------------------------------------------
# Health & Material Models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Backend service health and readiness status."""
    status: str = "ok"
    model_loaded: bool = False
    gpu: bool = False


class Material(BaseModel):
    """Physical and mechanical material definition."""
    id: str = Field(..., description="Unique material identifier, e.g. 'ti64'")
    name: str = Field(..., description="Human-readable material name, e.g. 'Ti-6Al-4V'")
    E_gpa: float = Field(..., description="Young's modulus in GPa")
    yield_mpa: float = Field(..., description="Yield strength in MPa")
    density_g_cm3: float = Field(..., description="Material density in g/cm³")


MaterialResponse = Material


# ---------------------------------------------------------------------------
# Design Request Models (POST /api/designs)
# ---------------------------------------------------------------------------

class PartSpec(BaseModel):
    """Geometry specification for the target lattice part."""
    size_mm: list[float] = Field(..., min_length=3, max_length=3, description="Bounding dimensions [x, y, z] in mm")
    cell_mm: float = Field(..., gt=0, description="Unit cell dimension in mm")


class DesignTarget(BaseModel):
    """Target performance criteria for the lattice optimization."""
    stiffness_gpa: AxisValues | None = None
    strength_mpa: AxisValues | None = None
    max_mass_g: float | None = None


class DesignModes(BaseModel):
    """Mode configurations for target criteria."""
    stiffness: TargetMode = TargetMode.ABOUT
    strength: TargetMode = TargetMode.AT_LEAST
    weight: TargetMode = TargetMode.AT_MOST


class DesignRequest(BaseModel):
    """Request payload for initiating a lattice design search."""
    material: str = Field(..., description="Material ID, e.g. 'ti64'")
    part: PartSpec = Field(..., description="Target part dimensions and cell sizing")
    target: DesignTarget = Field(..., description="Target physical properties")
    modes: DesignModes = Field(default_factory=DesignModes, description="Optimization modes")
    tolerance_pct: float = Field(default=10.0, description="Allowed deviation percentage")
    n_results: int = Field(default=5, ge=1, description="Number of candidate lattice designs to return")


CreateDesignRequest = DesignRequest
DesignCreate = DesignRequest


# ---------------------------------------------------------------------------
# Lattice & Property Evaluation Models
# ---------------------------------------------------------------------------

class LatticeSpec(BaseModel):
    """Lattice graph topology and structural specifications."""
    format: str = Field(default="octant27", description="Lattice representation format")
    nodes: list[list[float]] = Field(..., description="Lattice node 3D coordinates [[x, y, z], ...]")
    edges: list[list[int]] = Field(..., description="Node connectivity graph edge pairs [[node_a, node_b], ...]")
    cell_size_mm: float = Field(..., description="Unit cell size in mm")
    strut_radius_mm: float = Field(..., description="Strut radius in mm")


class PropertyEvaluation(BaseModel):
    """Directional mechanical and failure evaluations."""
    stiffness_gpa: AxisValues = Field(..., description="Stiffness along x, y, z in GPa")
    yield_strength_mpa: AxisValues = Field(..., description="Yield strength along x, y, z in MPa")
    buckling_strength_mpa: AxisValues = Field(..., description="Buckling strength along x, y, z in MPa")
    strength_mpa: AxisValues = Field(..., description="Effective strength along x, y, z in MPa")
    governing_failure: FailureModeAxis = Field(..., description="Governing failure mode along x, y, z ('yield' or 'buckling')")


PropertyPrediction = PropertyEvaluation
CandidateProperties = PropertyEvaluation


class Candidate(BaseModel):
    """Candidate lattice design with topology, predicted, and verified properties."""
    id: str = Field(..., description="Unique candidate identifier")
    rank: int = Field(..., ge=1, description="Ranking among candidate solutions")
    match_error_pct: float = Field(..., description="Percentage error relative to target requirements")
    rel_density: float = Field(..., ge=0.0, le=1.0, description="Relative volume density fraction")
    mass_g: float = Field(..., ge=0.0, description="Total mass in grams")
    lattice: LatticeSpec = Field(..., description="Lattice structural geometry")
    predicted: PropertyEvaluation = Field(..., description="Properties predicted by the ML surrogate model")
    verified: PropertyEvaluation | None = Field(default=None, description="Properties verified by the physics solver")


DesignCandidate = Candidate
DesignResponse = Candidate


# ---------------------------------------------------------------------------
# Job Response & Event Models (GET /api/jobs/{job_id}, etc.)
# ---------------------------------------------------------------------------

class JobCreateResponse(BaseModel):
    """Response returned upon successfully queueing a design job."""
    job_id: str = Field(..., description="Unique generated job identifier")
    status: JobStatus = Field(default=JobStatus.QUEUED, description="Initial status of the queued job")


class JobResponse(BaseModel):
    """Status and result response for a design job."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status (queued, running, done, failed)")
    stage: JobStage | None = Field(default=None, description="Current processing stage (search, repair, verify, rank)")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Completion progress from 0.0 to 1.0")
    message: str | None = Field(default=None, description="Informational or diagnostic status message")
    candidates: list[Candidate] = Field(default_factory=list, description="List of generated lattice design candidates")


DesignJobResponse = JobResponse


class JobEvent(BaseModel):
    """Streaming or polled progress event for a design job."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    stage: JobStage | None = Field(default=None, description="Current processing stage")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Completion progress from 0.0 to 1.0")
    message: str | None = Field(default=None, description="Status update message")
    candidates: list[Candidate] = Field(default_factory=list, description="Latest available candidates")


JobProgressEvent = JobEvent
