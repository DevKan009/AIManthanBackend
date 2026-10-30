"""Units conversion layer for physical design targets and normalized ML features.

Converts between dimensional physical properties (GPa, MPa, g, mm) and
dimensionless normalized quantities used by machine learning surrogate models
and optimization algorithms.
"""

from __future__ import annotations

from typing import Any

try:
    from server.schemas import AxisValues, DesignRequest
except ImportError:
    from schemas import AxisValues, DesignRequest


def calculate_volume_cm3(size_mm: list[float] | tuple[float, float, float]) -> float:
    """Calculate volume in cm³ from dimensions in mm [x, y, z].

    Each dimension is converted from mm to cm by dividing by 10 before multiplying.
    """
    if len(size_mm) != 3:
        raise ValueError(f"size_mm must contain exactly 3 dimensions [x, y, z], got {size_mm}")
    return (size_mm[0] / 10.0) * (size_mm[1] / 10.0) * (size_mm[2] / 10.0)


def mass_to_relative_density(
    mass_g: float | None,
    volume_cm3: float,
    material_density_g_cm3: float,
) -> float | None:
    """Convert mass in grams to dimensionless relative density (rho*).

    relative_density = mass_g / (volume_cm3 * material_density_g_cm3)
    Returns None if mass_g is None.
    """
    if mass_g is None:
        return None
    solid_mass_g = volume_cm3 * material_density_g_cm3
    if solid_mass_g <= 0.0:
        raise ValueError("Solid mass must be positive to calculate relative density.")
    return mass_g / solid_mass_g


def relative_density_to_mass(
    relative_density: float | None,
    volume_cm3: float,
    material_density_g_cm3: float,
) -> float | None:
    """Convert dimensionless relative density (rho*) back to mass in grams.

    mass_g = relative_density * volume_cm3 * material_density_g_cm3
    Returns None if relative_density is None.
    """
    if relative_density is None:
        return None
    return relative_density * volume_cm3 * material_density_g_cm3


def normalize_stiffness(stiffness_gpa: float | None, material_E_gpa: float) -> float | None:
    """Normalize stiffness by the base material's Young's modulus (E* = E / E_s).

    Returns None if stiffness_gpa is None.
    """
    if stiffness_gpa is None:
        return None
    if material_E_gpa <= 0.0:
        raise ValueError("Material Young's modulus E_gpa must be positive.")
    return stiffness_gpa / material_E_gpa


def denormalize_stiffness(norm_stiffness: float | None, material_E_gpa: float) -> float | None:
    """Denormalize dimensionless stiffness back to GPa (E = E* * E_s).

    Returns None if norm_stiffness is None.
    """
    if norm_stiffness is None:
        return None
    return norm_stiffness * material_E_gpa


def normalize_strength(strength_mpa: float | None, material_yield_mpa: float) -> float | None:
    """Normalize strength by the base material's yield strength (sigma* = sigma / sigma_ys).

    Returns None if strength_mpa is None.
    """
    if strength_mpa is None:
        return None
    if material_yield_mpa <= 0.0:
        raise ValueError("Material yield strength yield_mpa must be positive.")
    return strength_mpa / material_yield_mpa


def denormalize_strength(norm_strength: float | None, material_yield_mpa: float) -> float | None:
    """Denormalize dimensionless strength back to MPa (sigma = sigma* * sigma_ys).

    Returns None if norm_strength is None.
    """
    if norm_strength is None:
        return None
    return norm_strength * material_yield_mpa


def _extract_material_properties(material_dict: dict[str, Any]) -> tuple[float, float, float]:
    """Helper to extract (E_gpa, yield_mpa, density_g_cm3) flexibly from a material dict."""
    E_gpa = material_dict.get("E_gpa")
    if E_gpa is None:
        E_gpa = material_dict.get("material_E_gpa")
    if E_gpa is None:
        raise KeyError("Material dictionary must contain 'E_gpa' (or 'material_E_gpa').")

    yield_mpa = material_dict.get("yield_mpa")
    if yield_mpa is None:
        yield_mpa = material_dict.get("material_yield_mpa")
    if yield_mpa is None:
        raise KeyError("Material dictionary must contain 'yield_mpa' (or 'material_yield_mpa').")

    density_g_cm3 = material_dict.get("density_g_cm3")
    if density_g_cm3 is None:
        density_g_cm3 = material_dict.get("material_density_g_cm3")
    if density_g_cm3 is None:
        raise KeyError("Material dictionary must contain 'density_g_cm3' (or 'material_density_g_cm3').")

    return float(E_gpa), float(yield_mpa), float(density_g_cm3)


def normalize_request(request: DesignRequest | dict[str, Any], material_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert physical design targets into normalized ML targets.

    Calculates:
    - relative_density = mass_g / (volume_cm3 * material_density_g_cm3)
    - normalized_stiffness = stiffness_gpa / material_E_gpa (per axis)
    - normalized_strength = strength_mpa / material_yield_mpa (per axis)

    Handles None values gracefully by passing them through as None.
    """
    E_gpa, yield_mpa, density_g_cm3 = _extract_material_properties(material_dict)

    # Extract part size
    if isinstance(request, dict):
        part_data = request.get("part", {})
        size_mm = part_data.get("size_mm", [10.0, 10.0, 10.0])
        target_data = request.get("target", {})
        stiffness_val = target_data.get("stiffness_gpa")
        strength_val = target_data.get("strength_mpa")
        max_mass_g = target_data.get("max_mass_g")
    else:
        size_mm = request.part.size_mm
        stiffness_val = request.target.stiffness_gpa
        strength_val = request.target.strength_mpa
        max_mass_g = request.target.max_mass_g

    # Calculate volume
    volume_cm3 = calculate_volume_cm3(size_mm)

    # Relative density
    rel_density = mass_to_relative_density(max_mass_g, volume_cm3, density_g_cm3)

    # Normalized stiffness per axis
    if stiffness_val is None:
        norm_stiffness = {"x": None, "y": None, "z": None}
    elif isinstance(stiffness_val, dict):
        norm_stiffness = {
            "x": normalize_stiffness(stiffness_val.get("x"), E_gpa),
            "y": normalize_stiffness(stiffness_val.get("y"), E_gpa),
            "z": normalize_stiffness(stiffness_val.get("z"), E_gpa),
        }
    elif isinstance(stiffness_val, AxisValues):
        norm_stiffness = {
            "x": normalize_stiffness(stiffness_val.x, E_gpa),
            "y": normalize_stiffness(stiffness_val.y, E_gpa),
            "z": normalize_stiffness(stiffness_val.z, E_gpa),
        }
    else:
        norm_stiffness = {"x": None, "y": None, "z": None}

    # Normalized strength per axis
    if strength_val is None:
        norm_strength = {"x": None, "y": None, "z": None}
    elif isinstance(strength_val, dict):
        norm_strength = {
            "x": normalize_strength(strength_val.get("x"), yield_mpa),
            "y": normalize_strength(strength_val.get("y"), yield_mpa),
            "z": normalize_strength(strength_val.get("z"), yield_mpa),
        }
    elif isinstance(strength_val, AxisValues):
        norm_strength = {
            "x": normalize_strength(strength_val.x, yield_mpa),
            "y": normalize_strength(strength_val.y, yield_mpa),
            "z": normalize_strength(strength_val.z, yield_mpa),
        }
    else:
        norm_strength = {"x": None, "y": None, "z": None}

    return {
        "relative_density": rel_density,
        "normalized_stiffness": norm_stiffness,
        "normalized_strength": norm_strength,
    }


def denormalize_targets(
    normalized_targets: dict[str, Any],
    material_dict: dict[str, Any],
    volume_cm3: float | None = None,
    size_mm: list[float] | tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """Convert normalized ML targets/properties back to physical units (vice-versa).

    Converts:
    - relative_density -> mass_g (if volume_cm3 or size_mm is provided)
    - normalized_stiffness -> stiffness_gpa (per axis)
    - normalized_strength -> strength_mpa (per axis)
    """
    E_gpa, yield_mpa, density_g_cm3 = _extract_material_properties(material_dict)

    if volume_cm3 is None and size_mm is not None:
        volume_cm3 = calculate_volume_cm3(size_mm)

    rel_density = normalized_targets.get("relative_density")
    if rel_density is None:
        rel_density = normalized_targets.get("rel_density")

    mass_g = None
    if rel_density is not None and volume_cm3 is not None:
        mass_g = relative_density_to_mass(rel_density, volume_cm3, density_g_cm3)

    norm_stiffness = normalized_targets.get("normalized_stiffness") or normalized_targets.get("stiffness") or {}
    stiffness_gpa = {
        "x": denormalize_stiffness(norm_stiffness.get("x"), E_gpa),
        "y": denormalize_stiffness(norm_stiffness.get("y"), E_gpa),
        "z": denormalize_stiffness(norm_stiffness.get("z"), E_gpa),
    }

    norm_strength = normalized_targets.get("normalized_strength") or normalized_targets.get("strength") or {}
    strength_mpa = {
        "x": denormalize_strength(norm_strength.get("x"), yield_mpa),
        "y": denormalize_strength(norm_strength.get("y"), yield_mpa),
        "z": denormalize_strength(norm_strength.get("z"), yield_mpa),
    }

    return {
        "mass_g": mass_g,
        "stiffness_gpa": stiffness_gpa,
        "strength_mpa": strength_mpa,
    }
