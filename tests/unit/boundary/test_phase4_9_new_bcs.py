"""Tests for Phase 4-9 new boundary conditions.

Tests for:
- FixedFluxPressure2BC: enhanced fixed flux pressure with buoyancy correction
- InletOutlet3BC: enhanced inlet/outlet with turbulence-aware treatment
"""

import math

import pytest
import torch

from pyfoam.boundary import (
    BoundaryCondition,
    FixedFluxPressure2BC,
    InletOutlet3BC,
)
from pyfoam.boundary.boundary_condition import Patch


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def pressure_patch() -> Patch:
    """A patch for pressure BC testing."""
    return Patch(
        name="outlet",
        face_indices=torch.tensor([3, 4, 5]),
        face_normals=torch.tensor([
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ], dtype=torch.float64),
        face_areas=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
        delta_coeffs=torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64),
        owner_cells=torch.tensor([3, 4, 5]),
    )


@pytest.fixture
def inlet_patch() -> Patch:
    """A patch for inlet BC testing."""
    return Patch(
        name="inlet",
        face_indices=torch.tensor([0, 1, 2]),
        face_normals=torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=torch.float64),
        face_areas=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
        delta_coeffs=torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64),
        owner_cells=torch.tensor([0, 1, 2]),
    )


# ============================================================================
# FixedFluxPressure2BC tests
# ============================================================================


class TestFixedFluxPressure2BC:
    """Tests for fixedFluxPressure2 boundary condition."""

    def test_registration(self):
        """fixedFluxPressure2 is registered in RTS."""
        assert "fixedFluxPressure2" in BoundaryCondition.available_types()

    def test_factory_creation(self, pressure_patch):
        """BC can be created via factory method."""
        bc = BoundaryCondition.create(
            "fixedFluxPressure2", pressure_patch,
        )
        assert isinstance(bc, FixedFluxPressure2BC)

    def test_default_gravity(self, pressure_patch):
        """Default gravity is (0, -9.81, 0)."""
        bc = FixedFluxPressure2BC(pressure_patch)
        expected = torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64)
        assert torch.allclose(bc.g, expected, atol=1e-10)

    def test_custom_gravity(self, pressure_patch):
        """Custom gravity vector is stored correctly."""
        bc = FixedFluxPressure2BC(
            pressure_patch,
            {"g": (0.0, 0.0, -1.0)},
        )
        expected = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64)
        assert torch.allclose(bc.g, expected)

    def test_rho_ref_property(self, pressure_patch):
        """Reference density is stored correctly."""
        bc = FixedFluxPressure2BC(pressure_patch, {"rho": 1000.0})
        assert bc.rho_ref == 1000.0

    def test_default_rho_ref(self, pressure_patch):
        """Default reference density is 1.0."""
        bc = FixedFluxPressure2BC(pressure_patch)
        assert bc.rho_ref == 1.0

    def test_apply_zero_gradient_no_buoyancy(self, pressure_patch):
        """With zero gravity, behaves like zeroGradient."""
        bc = FixedFluxPressure2BC(
            pressure_patch,
            {"g": (0.0, 0.0, 0.0)},
        )
        field = torch.zeros(10, dtype=torch.float64)
        field[3:6] = torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(
            field[3:6],
            torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64),
        )

    def test_apply_with_y_normal_and_gravity(self):
        """Buoyancy correction is non-zero for y-normal face with y-gravity."""
        y_patch = Patch(
            name="top",
            face_indices=torch.tensor([0, 1, 2]),
            face_normals=torch.tensor([
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ], dtype=torch.float64),
            face_areas=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
            delta_coeffs=torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64),
            owner_cells=torch.tensor([0, 1, 2]),
        )

        bc = FixedFluxPressure2BC(
            y_patch,
            {"g": (0.0, -9.81, 0.0), "rho": 1000.0},
        )
        field = torch.zeros(10, dtype=torch.float64)
        bc.apply(field)

        # g . n = (0, -9.81, 0) . (0, 1, 0) = -9.81
        # correction = rho * g . n / delta = 1000 * (-9.81) / 2.0 = -4905.0
        expected = -4905.0
        assert torch.allclose(
            field[0:3],
            torch.full((3,), expected, dtype=torch.float64),
            atol=1e-6,
        )

    def test_apply_with_tensor_density(self):
        """Apply with per-face density tensor works correctly."""
        y_patch = Patch(
            name="top",
            face_indices=torch.tensor([0, 1, 2]),
            face_normals=torch.tensor([
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ], dtype=torch.float64),
            face_areas=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
            delta_coeffs=torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64),
            owner_cells=torch.tensor([0, 1, 2]),
        )

        bc = FixedFluxPressure2BC(
            y_patch,
            {"g": (0.0, -9.81, 0.0)},
        )
        field = torch.zeros(10, dtype=torch.float64)
        rho_tensor = torch.tensor(
            [500.0, 1000.0, 1500.0],
            dtype=torch.float64,
        )
        bc.apply(field, rho=rho_tensor)

        # For face 0: rho=500, g.n=-9.81, delta=2.0
        # correction = 500 * (-9.81) / 2.0 = -2452.5
        assert abs(field[0].item() - (-2452.5)) < 1e-4
        # For face 1: rho=1000, correction = 1000 * (-9.81) / 2.0 = -4905.0
        assert abs(field[1].item() - (-4905.0)) < 1e-4

    def test_matrix_contributions_zero(self, pressure_patch):
        """Matrix contributions are zero (like zeroGradient)."""
        bc = FixedFluxPressure2BC(pressure_patch)
        field = torch.zeros(10, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 10)
        assert torch.allclose(diag, torch.zeros(10, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(10, dtype=torch.float64))


# ============================================================================
# InletOutlet3BC tests
# ============================================================================


class TestInletOutlet3BC:
    """Tests for inletOutlet3 boundary condition."""

    def test_registration(self):
        """inletOutlet3 is registered in RTS."""
        assert "inletOutlet3" in BoundaryCondition.available_types()

    def test_factory_creation(self, inlet_patch):
        """BC can be created via factory method."""
        bc = BoundaryCondition.create(
            "inletOutlet3", inlet_patch,
        )
        assert isinstance(bc, InletOutlet3BC)

    def test_default_properties(self, inlet_patch):
        """Default properties are zero."""
        bc = InletOutlet3BC(inlet_patch)
        assert bc.turbulence_intensity == 0.0
        assert bc.mixing_length == 0.0

    def test_custom_properties(self, inlet_patch):
        """Custom turbulence properties are stored."""
        bc = InletOutlet3BC(
            inlet_patch,
            {"turbulenceIntensity": 0.05, "mixingLength": 0.1},
        )
        assert bc.turbulence_intensity == 0.05
        assert bc.mixing_length == 0.1

    def test_apply_outflow_zero_gradient(self, inlet_patch):
        """Outflow (positive flux) applies zero gradient."""
        bc = InletOutlet3BC(inlet_patch, {"value": 42.0})
        field = torch.zeros(10, dtype=torch.float64)
        field[0:3] = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        flux = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        bc.apply(field, flux=flux)
        assert torch.allclose(
            field[0:3],
            torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64),
        )

    def test_apply_inflow_fixed_value(self, inlet_patch):
        """Inflow (negative flux) applies prescribed value."""
        bc = InletOutlet3BC(inlet_patch, {"value": 42.0})
        field = torch.zeros(10, dtype=torch.float64)
        flux = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64)
        bc.apply(field, flux=flux)
        assert torch.allclose(
            field[0:3],
            torch.full((3,), 42.0, dtype=torch.float64),
        )

    def test_apply_mixed_flow(self, inlet_patch):
        """Mixed inflow/outflow applies correct values per face."""
        bc = InletOutlet3BC(inlet_patch, {"value": 100.0})
        field = torch.zeros(10, dtype=torch.float64)
        field[0:3] = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        flux = torch.tensor([-1.0, 1.0, -1.0], dtype=torch.float64)
        bc.apply(field, flux=flux)
        assert abs(field[0].item() - 100.0) < 1e-10
        assert abs(field[1].item() - 20.0) < 1e-10
        assert abs(field[2].item() - 100.0) < 1e-10

    def test_turbulent_k_computation(self, inlet_patch):
        """Turbulent kinetic energy is computed correctly."""
        bc = InletOutlet3BC(
            inlet_patch,
            {"turbulenceIntensity": 0.1, "value": 0.0},
        )
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        k = bc._compute_turbulent_k(velocity)
        expected = 1.5 * (0.1 * 10.0) ** 2
        assert torch.allclose(k, torch.full((3,), expected, dtype=torch.float64))

    def test_apply_inflow_with_turbulence(self, inlet_patch):
        """Inflow with turbulence intensity adds k to value."""
        bc = InletOutlet3BC(
            inlet_patch,
            {"turbulenceIntensity": 0.1, "value": 0.0},
        )
        field = torch.zeros(10, dtype=torch.float64)
        flux = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64)
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.apply(field, flux=flux, velocity=velocity)
        expected = 1.5
        assert torch.allclose(
            field[0:3],
            torch.full((3,), expected, dtype=torch.float64),
        )

    def test_turbulent_epsilon_computation(self, inlet_patch):
        """Turbulent dissipation rate is computed correctly."""
        bc = InletOutlet3BC(
            inlet_patch,
            {"turbulenceIntensity": 0.1, "mixingLength": 0.1},
        )
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        eps = bc._compute_turbulent_epsilon(velocity)
        k = 1.5
        expected = 0.09 ** 0.75 * k ** 1.5 / 0.1
        assert torch.allclose(eps, torch.full((3,), expected, dtype=torch.float64))

    def test_turbulent_omega_computation(self, inlet_patch):
        """Specific dissipation rate is computed correctly."""
        bc = InletOutlet3BC(
            inlet_patch,
            {"turbulenceIntensity": 0.1, "mixingLength": 0.1},
        )
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        omega = bc._compute_turbulent_omega(velocity)
        k = 1.5
        expected = k ** 0.5 / (0.09 ** 0.25 * 0.1)
        assert torch.allclose(omega, torch.full((3,), expected, dtype=torch.float64))

    def test_matrix_contributions_inflow(self, inlet_patch):
        """Inflow matrix contributions use penalty method."""
        bc = InletOutlet3BC(inlet_patch, {"value": 42.0})
        field = torch.zeros(10, dtype=torch.float64)
        flux = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 10, flux=flux)
        assert diag.shape == (10,)
        assert source.shape == (10,)
        assert diag[0] > 0
        assert diag[1] > 0
        assert diag[2] > 0

    def test_matrix_contributions_outflow(self, inlet_patch):
        """Outflow matrix contributions are zero."""
        bc = InletOutlet3BC(inlet_patch, {"value": 42.0})
        field = torch.zeros(10, dtype=torch.float64)
        flux = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 10, flux=flux)
        assert torch.allclose(diag, torch.zeros(10, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(10, dtype=torch.float64))

    def test_velocity_based_flow_direction(self, inlet_patch):
        """Flow direction can be determined from velocity."""
        bc = InletOutlet3BC(inlet_patch, {"value": 50.0})
        field = torch.zeros(10, dtype=torch.float64)
        velocity = torch.tensor([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert abs(field[0].item() - 50.0) < 1e-10
        assert abs(field[1].item() - 0.0) < 1e-10
        assert abs(field[2].item() - 50.0) < 1e-10
