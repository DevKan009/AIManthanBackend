import pytest

from server.schemas import (
    AxisValues,
    DesignModes,
    DesignRequest,
    DesignTarget,
    PartSpec,
)
from server.units import (
    calculate_volume_cm3,
    denormalize_stiffness,
    denormalize_strength,
    denormalize_targets,
    mass_to_relative_density,
    normalize_request,
    normalize_stiffness,
    normalize_strength,
    relative_density_to_mass,
)


@pytest.fixture
def dummy_material():
    """Dummy test material as specified in requirements:

    E=100 GPa, Yield=1000 MPa, Density=5.0 g/cm3.
    """
    return {
        "id": "dummy_mat",
        "name": "Dummy Test Material",
        "E_gpa": 100.0,
        "yield_mpa": 1000.0,
        "density_g_cm3": 5.0,
    }


def test_normalize_request_full_values(dummy_material):
    """Test 1: Verify full mathematical conversion of request targets to normalized ML values."""
    # Part: [40, 40, 40] mm -> [4, 4, 4] cm -> volume = 64.0 cm³
    # Solid mass = 64.0 * 5.0 = 320.0 g
    # Target mass = 64.0 g -> relative_density = 64.0 / 320.0 = 0.20
    # Stiffness: 8.0 / 100 = 0.08, 12.0 / 100 = 0.12
    # Strength: 100.0 / 1000 = 0.10, 250.0 / 1000 = 0.25, 500.0 / 1000 = 0.50
    request = DesignRequest(
        material="dummy_mat",
        part=PartSpec(size_mm=[40.0, 40.0, 40.0], cell_mm=10.0),
        target=DesignTarget(
            stiffness_gpa=AxisValues(x=8.0, y=8.0, z=12.0),
            strength_mpa=AxisValues(x=100.0, y=250.0, z=500.0),
            max_mass_g=64.0,
        ),
        modes=DesignModes(),
        tolerance_pct=10.0,
        n_results=5,
    )

    norm = normalize_request(request, dummy_material)

    assert pytest.approx(norm["relative_density"], rel=1e-5) == 0.20
    assert pytest.approx(norm["normalized_stiffness"]["x"], rel=1e-5) == 0.08
    assert pytest.approx(norm["normalized_stiffness"]["y"], rel=1e-5) == 0.08
    assert pytest.approx(norm["normalized_stiffness"]["z"], rel=1e-5) == 0.12
    assert pytest.approx(norm["normalized_strength"]["x"], rel=1e-5) == 0.10
    assert pytest.approx(norm["normalized_strength"]["y"], rel=1e-5) == 0.25
    assert pytest.approx(norm["normalized_strength"]["z"], rel=1e-5) == 0.50


def test_normalize_request_with_none_values(dummy_material):
    """Test 2: Verify that None values on any axis or mass pass through gracefully as None."""
    request = DesignRequest(
        material="dummy_mat",
        part=PartSpec(size_mm=[20.0, 30.0, 40.0], cell_mm=10.0),
        target=DesignTarget(
            stiffness_gpa=AxisValues(x=15.0, y=None, z=20.0),
            strength_mpa=AxisValues(x=90.0, y=90.0, z=None),
            max_mass_g=None,
        ),
        modes=DesignModes(),
    )

    norm = normalize_request(request, dummy_material)

    # Relative density should be None when max_mass_g is None
    assert norm["relative_density"] is None

    # Stiffness y should pass through as None
    assert pytest.approx(norm["normalized_stiffness"]["x"], rel=1e-5) == 0.15
    assert norm["normalized_stiffness"]["y"] is None
    assert pytest.approx(norm["normalized_stiffness"]["z"], rel=1e-5) == 0.20

    # Strength z should pass through as None
    assert pytest.approx(norm["normalized_strength"]["x"], rel=1e-5) == 0.09
    assert pytest.approx(norm["normalized_strength"]["y"], rel=1e-5) == 0.09
    assert norm["normalized_strength"]["z"] is None


def test_denormalize_targets_vice_versa(dummy_material):
    """Test 3: Verify vice-versa denormalization and roundtrip conversion back to physical units."""
    size_mm = [40.0, 40.0, 40.0]
    volume_cm3 = calculate_volume_cm3(size_mm)  # 64.0 cm³

    normalized_targets = {
        "relative_density": 0.25,
        "normalized_stiffness": {"x": 0.08, "y": 0.08, "z": 0.12},
        "normalized_strength": {"x": 0.15, "y": 0.20, "z": None},
    }

    denorm = denormalize_targets(
        normalized_targets,
        dummy_material,
        volume_cm3=volume_cm3,
    )

    # mass_g = 0.25 * 64.0 * 5.0 = 80.0 g
    assert pytest.approx(denorm["mass_g"], rel=1e-5) == 80.0

    # stiffness_gpa = normalized * 100.0
    assert pytest.approx(denorm["stiffness_gpa"]["x"], rel=1e-5) == 8.0
    assert pytest.approx(denorm["stiffness_gpa"]["y"], rel=1e-5) == 8.0
    assert pytest.approx(denorm["stiffness_gpa"]["z"], rel=1e-5) == 12.0

    # strength_mpa = normalized * 1000.0
    assert pytest.approx(denorm["strength_mpa"]["x"], rel=1e-5) == 150.0
    assert pytest.approx(denorm["strength_mpa"]["y"], rel=1e-5) == 200.0
    assert denorm["strength_mpa"]["z"] is None


def test_individual_conversion_functions(dummy_material):
    """Test 4: Verify discrete unit math and edge functions."""
    # Volume calculation mm -> cm³: 100mm x 50mm x 20mm = 10cm x 5cm x 2cm = 100 cm³
    vol = calculate_volume_cm3([100.0, 50.0, 20.0])
    assert pytest.approx(vol, rel=1e-5) == 100.0

    # Mass to relative density: 50g in 100 cm³ with density 5.0 g/cm³ -> 50 / (100 * 5) = 0.10
    rd = mass_to_relative_density(50.0, vol, dummy_material["density_g_cm3"])
    assert pytest.approx(rd, rel=1e-5) == 0.10

    # Relative density back to mass: 0.10 * 100 * 5.0 = 50.0g
    mass = relative_density_to_mass(rd, vol, dummy_material["density_g_cm3"])
    assert pytest.approx(mass, rel=1e-5) == 50.0

    # Stiffness conversions
    norm_e = normalize_stiffness(15.0, dummy_material["E_gpa"])
    assert pytest.approx(norm_e, rel=1e-5) == 0.15
    denorm_e = denormalize_stiffness(norm_e, dummy_material["E_gpa"])
    assert pytest.approx(denorm_e, rel=1e-5) == 15.0

    # Strength conversions
    norm_s = normalize_strength(250.0, dummy_material["yield_mpa"])
    assert pytest.approx(norm_s, rel=1e-5) == 0.25
    denorm_s = denormalize_strength(norm_s, dummy_material["yield_mpa"])
    assert pytest.approx(denorm_s, rel=1e-5) == 250.0
