"""Tests for Phase 9 boundary conditions.

Tests for:
- Velocity BCs: flowRateInletVelocity, pressureInletOutletVelocity, rotatingWallVelocity
- Pressure BCs: totalPressure, fixedFluxPressure, prghPressure, waveTransmissive
- Turbulence BCs: turbulentIntensityKineticEnergyInlet, turbulentMixingLengthDissipationRateInlet,
  turbulentMixingLengthFrequencyInlet
- VOF BCs: constantAlphaContactAngle
"""

import math

import pytest
import torch

from pyfoam.boundary import (
    BoundaryCondition,
    FlowRateInletVelocityBC,
    PressureInletOutletVelocityBC,
    RotatingWallVelocityBC,
    TotalPressureBC,
    FixedFluxPressureBC,
    PrghPressureBC,
    WaveTransmissiveBC,
    TurbulentIntensityKineticEnergyInletBC,
    TurbulentMixingLengthDissipationRateInletBC,
    TurbulentMixingLengthFrequencyInletBC,
    ConstantAlphaContactAngleBC,
)
from pyfoam.boundary.boundary_condition import Patch


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def velocity_patch() -> Patch:
    """A patch for velocity BC testing."""
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
def wall_patch() -> Patch:
    """A patch for wall BC testing."""
    return Patch(
        name="wall",
        face_indices=torch.tensor([6, 7, 8]),
        face_normals=torch.tensor([
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=torch.float64),
        face_areas=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
        delta_coeffs=torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64),
        owner_cells=torch.tensor([6, 7, 8]),
    )


# ============================================================================
# T9.1: Velocity BCs
# ============================================================================


class TestFlowRateInletVelocityBC:
    """Tests for flowRateInletVelocity BC."""

    def test_registration(self):
        """flowRateInletVelocity is registered in RTS."""
        assert "flowRateInletVelocity" in BoundaryCondition.available_types()

    def test_factory_creation(self, velocity_patch):
        """BC can be created via factory method."""
        bc = BoundaryCondition.create(
            "flowRateInletVelocity",
            velocity_patch,
            {"volumetricFlowRate": 1.0},
        )
        assert isinstance(bc, FlowRateInletVelocityBC)

    def test_volumetric_flow_rate(self, velocity_patch):
        """Volumetric flow rate is parsed correctly."""
        bc = FlowRateInletVelocityBC(velocity_patch, {"volumetricFlowRate": 0.5})
        assert bc.volumetric_flow_rate == 0.5

    def test_mass_flow_rate(self, velocity_patch):
        """Mass flow rate is converted to volumetric."""
        bc = FlowRateInletVelocityBC(
            velocity_patch,
            {"massFlowRate": 1.0, "rho": 2.0},
        )
        assert bc.volumetric_flow_rate == 0.5

    def test_default_flow_rate(self, velocity_patch):
        """Default flow rate is zero."""
        bc = FlowRateInletVelocityBC(velocity_patch)
        assert bc.volumetric_flow_rate == 0.0

    def test_apply_sets_velocity(self, velocity_patch):
        """apply() sets velocity based on flow rate."""
        bc = FlowRateInletVelocityBC(velocity_patch, {"volumetricFlowRate": 3.0})
        field = torch.zeros(10, 3, dtype=torch.float64)
        bc.apply(field)
        # Total area = 3.0, velocity magnitude = 3.0/3.0 = 1.0
        # Velocity = 1.0 * face_normal = (1, 0, 0) per face
        assert torch.allclose(field[0:3, 0], torch.ones(3, dtype=torch.float64))
        assert torch.allclose(field[0:3, 1], torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(field[0:3, 2], torch.zeros(3, dtype=torch.float64))

    def test_matrix_contributions(self, velocity_patch):
        """Matrix contributions use penalty method."""
        bc = FlowRateInletVelocityBC(velocity_patch, {"volumetricFlowRate": 3.0})
        field = torch.zeros(10, 3, dtype=torch.float64)
        n_cells = 10
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # Owner cells 0, 1, 2 should have contributions
        assert diag[0] > 0
        assert diag[1] > 0
        assert diag[2] > 0


class TestPressureInletOutletVelocityBC:
    """Tests for pressureInletOutletVelocity BC."""

    def test_registration(self):
        """pressureInletOutletVelocity is registered in RTS."""
        assert "pressureInletOutletVelocity" in BoundaryCondition.available_types()

    def test_factory_creation(self, velocity_patch):
        """BC can be created via factory method."""
        bc = BoundaryCondition.create(
            "pressureInletOutletVelocity",
            velocity_patch,
        )
        assert isinstance(bc, PressureInletOutletVelocityBC)

    def test_apply_zero_gradient(self, velocity_patch):
        """apply() copies owner cell values (zero gradient)."""
        bc = PressureInletOutletVelocityBC(velocity_patch)
        field = torch.zeros(10, 3, dtype=torch.float64)
        field[0:3] = torch.tensor([[1.0, 2.0, 3.0]] * 3, dtype=torch.float64)
        bc.apply(field)
        # Boundary faces should have owner cell values
        assert torch.allclose(field[0:3], field[0:3])

    def test_matrix_contributions_zero(self, velocity_patch):
        """Matrix contributions are zero (like zeroGradient)."""
        bc = PressureInletOutletVelocityBC(velocity_patch)
        field = torch.zeros(10, 3, dtype=torch.float64)
        n_cells = 10
        diag, source = bc.matrix_contributions(field, n_cells)
        assert torch.allclose(diag, torch.zeros(n_cells, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(n_cells, dtype=torch.float64))


class TestRotatingWallVelocityBC:
    """Tests for rotatingWallVelocity BC."""

    def test_registration(self):
        """rotatingWallVelocity is registered in RTS."""
        assert "rotatingWallVelocity" in BoundaryCondition.available_types()

    def test_factory_creation(self, wall_patch):
        """BC can be created via factory method."""
        bc = BoundaryCondition.create(
            "rotatingWallVelocity",
            wall_patch,
            {"origin": [0, 0, 0], "axis": [0, 0, 1], "omega": 10.0},
        )
        assert isinstance(bc, RotatingWallVelocityBC)

    def test_properties(self, wall_patch):
        """Properties are parsed correctly."""
        bc = RotatingWallVelocityBC(
            wall_patch,
            {"origin": [1, 2, 3], "axis": [0, 0, 1], "omega": 5.0},
        )
        assert bc.omega == 5.0
        assert torch.allclose(bc.origin, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        # Axis should be normalized
        assert torch.allclose(bc.axis, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))

    def test_default_values(self, wall_patch):
        """Default values are correct."""
        bc = RotatingWallVelocityBC(wall_patch)
        assert bc.omega == 0.0
        assert torch.allclose(bc.origin, torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(bc.axis, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))


# ============================================================================
# T9.2: Pressure BCs
# ============================================================================


class TestTotalPressureBC:
    """Tests for totalPressure BC."""

    def test_registration(self):
        """totalPressure is registered in RTS."""
        assert "totalPressure" in BoundaryCondition.available_types()

    def test_factory_creation(self, pressure_patch):
        """BC can be created via factory method."""
        bc = BoundaryCondition.create(
            "totalPressure",
            pressure_patch,
            {"p0": 101325.0},
        )
        assert isinstance(bc, TotalPressureBC)

    def test_properties(self, pressure_patch):
        """Properties are parsed correctly."""
        bc = TotalPressureBC(pressure_patch, {"p0": 100000.0, "gamma": 1.4})
        assert bc.p0 == 100000.0
        assert bc.gamma == 1.4

    def test_default_values(self, pressure_patch):
        """Default values are correct."""
        bc = TotalPressureBC(pressure_patch)
        assert bc.p0 == 101325.0
        assert bc.gamma == 1.4

    def test_apply_without_velocity(self, pressure_patch):
        """apply() without velocity sets total pressure directly."""
        bc = TotalPressureBC(pressure_patch, {"p0": 100000.0})
        field = torch.zeros(10, dtype=torch.float64)
        bc.apply(field)
        # Should set p0 at boundary faces
        assert torch.allclose(field[3:6], torch.full((3,), 100000.0, dtype=torch.float64))

    def test_apply_with_velocity(self, pressure_patch):
        """apply() with velocity computes static pressure."""
        bc = TotalPressureBC(pressure_patch, {"p0": 100000.0})
        field = torch.zeros(10, dtype=torch.float64)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        rho = 1.0
        bc.apply(field, velocity=velocity, rho=rho)
        # p = p0 - 0.5 * rho * |U|² = 100000 - 0.5*1*100 = 99950
        expected = 100000.0 - 0.5 * 1.0 * 100.0
        assert torch.allclose(field[3:6], torch.full((3,), expected, dtype=torch.float64))

    def test_matrix_contributions(self, pressure_patch):
        """Matrix contributions use penalty method."""
        bc = TotalPressureBC(pressure_patch, {"p0": 100000.0})
        field = torch.zeros(10, dtype=torch.float64)
        n_cells = 10
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # Owner cells 3, 4, 5 should have contributions
        assert diag[3] > 0
        assert diag[4] > 0
        assert diag[5] > 0


class TestFixedFluxPressureBC:
    """Tests for fixedFluxPressure BC."""

    def test_registration(self):
        """fixedFluxPressure is registered in RTS."""
        assert "fixedFluxPressure" in BoundaryCondition.available_types()

    def test_factory_creation(self, pressure_patch):
        """BC can be created via factory method."""
        bc = BoundaryCondition.create(
            "fixedFluxPressure",
            pressure_patch,
        )
        assert isinstance(bc, FixedFluxPressureBC)

    def test_apply_zero_gradient(self, pressure_patch):
        """apply() copies owner cell values (zero gradient)."""
        bc = FixedFluxPressureBC(pressure_patch)
        field = torch.zeros(10, dtype=torch.float64)
        field[3:6] = torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64)
        bc.apply(field)
        # Boundary faces should have owner cell values
        assert torch.allclose(field[3:6], torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64))

    def test_matrix_contributions_zero(self, pressure_patch):
        """Matrix contributions are zero (like zeroGradient)."""
        bc = FixedFluxPressureBC(pressure_patch)
        field = torch.zeros(10, dtype=torch.float64)
        n_cells = 10
        diag, source = bc.matrix_contributions(field, n_cells)
        assert torch.allclose(diag, torch.zeros(n_cells, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(n_cells, dtype=torch.float64))


class TestPrghPressureBC:
    """Tests for prghPressure BC."""

    def test_registration(self):
        """prghPressure is registered in RTS."""
        assert "prghPressure" in BoundaryCondition.available_types()

    def test_factory_creation(self, pressure_patch):
        """BC can be created via factory method."""
        bc = BoundaryCondition.create(
            "prghPressure",
            pressure_patch,
            {"p0": 101325.0},
        )
        assert isinstance(bc, PrghPressureBC)

    def test_properties(self, pressure_patch):
        """Properties are parsed correctly."""
        bc = PrghPressureBC(pressure_patch, {"p0": 100000.0})
        assert bc.p0 == 100000.0

    def test_default_values(self, pressure_patch):
        """Default values are correct."""
        bc = PrghPressureBC(pressure_patch)
        assert bc.p0 == 101325.0

    def test_apply_sets_pressure(self, pressure_patch):
        """apply() sets p_rgh to reference pressure."""
        bc = PrghPressureBC(pressure_patch, {"p0": 100000.0})
        field = torch.zeros(10, dtype=torch.float64)
        bc.apply(field)
        # Should set p0 at boundary faces
        assert torch.allclose(field[3:6], torch.full((3,), 100000.0, dtype=torch.float64))

    def test_matrix_contributions(self, pressure_patch):
        """Matrix contributions use penalty method."""
        bc = PrghPressureBC(pressure_patch, {"p0": 100000.0})
        field = torch.zeros(10, dtype=torch.float64)
        n_cells = 10
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # Owner cells 3, 4, 5 should have contributions
        assert diag[3] > 0
        assert diag[4] > 0
        assert diag[5] > 0


class TestWaveTransmissiveBC:
    """Tests for waveTransmissive BC."""

    def test_registration(self):
        """waveTransmissive is registered in RTS."""
        assert "waveTransmissive" in BoundaryCondition.available_types()

    def test_factory_creation(self, pressure_patch):
        """BC can be created via factory method."""
        bc = BoundaryCondition.create(
            "waveTransmissive",
            pressure_patch,
            {"fieldInf": 101325.0, "lInf": 1.0},
        )
        assert isinstance(bc, WaveTransmissiveBC)

    def test_properties(self, pressure_patch):
        """Properties are parsed correctly."""
        bc = WaveTransmissiveBC(
            pressure_patch,
            {"fieldInf": 100000.0, "lInf": 0.5, "gamma": 1.4},
        )
        assert bc.field_inf == 100000.0
        assert bc.l_inf == 0.5
        assert bc.gamma == 1.4

    def test_default_values(self, pressure_patch):
        """Default values are correct."""
        bc = WaveTransmissiveBC(pressure_patch)
        assert bc.field_inf == 101325.0
        assert bc.l_inf == 1.0
        assert bc.gamma == 1.4

    def test_apply_zero_gradient(self, pressure_patch):
        """apply() without velocity uses zero gradient."""
        bc = WaveTransmissiveBC(pressure_patch, {"fieldInf": 101325.0})
        field = torch.zeros(10, dtype=torch.float64)
        field[3:6] = torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64)
        bc.apply(field)
        # Should copy owner values
        assert torch.allclose(field[3:6], torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64))

    def test_matrix_contributions_zero(self, pressure_patch):
        """Matrix contributions are zero (like zeroGradient)."""
        bc = WaveTransmissiveBC(pressure_patch)
        field = torch.zeros(10, dtype=torch.float64)
        n_cells = 10
        diag, source = bc.matrix_contributions(field, n_cells)
        assert torch.allclose(diag, torch.zeros(n_cells, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(n_cells, dtype=torch.float64))


# ============================================================================
# T9.3: Turbulence BCs
# ============================================================================


class TestTurbulentIntensityKineticEnergyInletBC:
    """Tests for turbulentIntensityKineticEnergyInlet BC."""

    def test_registration(self):
        """turbulentIntensityKineticEnergyInlet is registered in RTS."""
        assert "turbulentIntensityKineticEnergyInlet" in BoundaryCondition.available_types()

    def test_factory_creation(self, velocity_patch):
        """BC can be created via factory method."""
        bc = BoundaryCondition.create(
            "turbulentIntensityKineticEnergyInlet",
            velocity_patch,
            {"intensity": 0.05},
        )
        assert isinstance(bc, TurbulentIntensityKineticEnergyInletBC)

    def test_properties(self, velocity_patch):
        """Properties are parsed correctly."""
        bc = TurbulentIntensityKineticEnergyInletBC(
            velocity_patch,
            {"intensity": 0.1},
        )
        assert bc.intensity == 0.1

    def test_default_intensity(self, velocity_patch):
        """Default intensity is 0.05."""
        bc = TurbulentIntensityKineticEnergyInletBC(velocity_patch)
        assert bc.intensity == 0.05

    def test_apply_with_velocity(self, velocity_patch):
        """apply() computes k from velocity and intensity."""
        bc = TurbulentIntensityKineticEnergyInletBC(
            velocity_patch,
            {"intensity": 0.1},
        )
        field = torch.zeros(10, dtype=torch.float64)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        # k = 1.5 * (I * |U|)² = 1.5 * (0.1 * 10)² = 1.5 * 1 = 1.5
        expected = 1.5 * (0.1 * 10.0) ** 2
        assert torch.allclose(field[0:3], torch.full((3,), expected, dtype=torch.float64))

    def test_apply_without_velocity(self, velocity_patch):
        """apply() without velocity uses default value."""
        bc = TurbulentIntensityKineticEnergyInletBC(velocity_patch)
        field = torch.zeros(10, dtype=torch.float64)
        bc.apply(field)
        # Should use default k = 0.01
        assert torch.allclose(field[0:3], torch.full((3,), 0.01, dtype=torch.float64))

    def test_matrix_contributions(self, velocity_patch):
        """Matrix contributions use penalty method."""
        bc = TurbulentIntensityKineticEnergyInletBC(velocity_patch)
        field = torch.zeros(10, dtype=torch.float64)
        n_cells = 10
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # Owner cells 0, 1, 2 should have contributions
        assert diag[0] > 0
        assert diag[1] > 0
        assert diag[2] > 0


class TestTurbulentMixingLengthDissipationRateInletBC:
    """Tests for turbulentMixingLengthDissipationRateInlet BC."""

    def test_registration(self):
        """turbulentMixingLengthDissipationRateInlet is registered in RTS."""
        assert "turbulentMixingLengthDissipationRateInlet" in BoundaryCondition.available_types()

    def test_factory_creation(self, velocity_patch):
        """BC can be created via factory method."""
        bc = BoundaryCondition.create(
            "turbulentMixingLengthDissipationRateInlet",
            velocity_patch,
            {"mixingLength": 0.01, "Cmu": 0.09},
        )
        assert isinstance(bc, TurbulentMixingLengthDissipationRateInletBC)

    def test_properties(self, velocity_patch):
        """Properties are parsed correctly."""
        bc = TurbulentMixingLengthDissipationRateInletBC(
            velocity_patch,
            {"mixingLength": 0.05, "Cmu": 0.09},
        )
        assert bc.mixing_length == 0.05
        assert bc.C_mu == 0.09

    def test_default_values(self, velocity_patch):
        """Default values are correct."""
        bc = TurbulentMixingLengthDissipationRateInletBC(velocity_patch)
        assert bc.mixing_length == 0.01
        assert bc.C_mu == 0.09

    def test_apply_with_k(self, velocity_patch):
        """apply() computes epsilon from k and mixing length."""
        bc = TurbulentMixingLengthDissipationRateInletBC(
            velocity_patch,
            {"mixingLength": 0.01, "Cmu": 0.09},
        )
        field = torch.zeros(10, dtype=torch.float64)
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        bc.apply(field, k=k)
        # epsilon = C_mu^0.75 * k^1.5 / l = 0.09^0.75 * 1.0 / 0.01
        expected = (0.09 ** 0.75) * (1.0 ** 1.5) / 0.01
        assert torch.allclose(field[0:3], torch.full((3,), expected, dtype=torch.float64))

    def test_apply_without_k(self, velocity_patch):
        """apply() without k uses default value."""
        bc = TurbulentMixingLengthDissipationRateInletBC(velocity_patch)
        field = torch.zeros(10, dtype=torch.float64)
        bc.apply(field)
        # Should use default epsilon = 0.01
        assert torch.allclose(field[0:3], torch.full((3,), 0.01, dtype=torch.float64))

    def test_matrix_contributions(self, velocity_patch):
        """Matrix contributions use penalty method."""
        bc = TurbulentMixingLengthDissipationRateInletBC(velocity_patch)
        field = torch.zeros(10, dtype=torch.float64)
        n_cells = 10
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # Owner cells 0, 1, 2 should have contributions
        assert diag[0] > 0
        assert diag[1] > 0
        assert diag[2] > 0


class TestTurbulentMixingLengthFrequencyInletBC:
    """Tests for turbulentMixingLengthFrequencyInlet BC."""

    def test_registration(self):
        """turbulentMixingLengthFrequencyInlet is registered in RTS."""
        assert "turbulentMixingLengthFrequencyInlet" in BoundaryCondition.available_types()

    def test_factory_creation(self, velocity_patch):
        """BC can be created via factory method."""
        bc = BoundaryCondition.create(
            "turbulentMixingLengthFrequencyInlet",
            velocity_patch,
            {"mixingLength": 0.01, "Cmu": 0.09},
        )
        assert isinstance(bc, TurbulentMixingLengthFrequencyInletBC)

    def test_properties(self, velocity_patch):
        """Properties are parsed correctly."""
        bc = TurbulentMixingLengthFrequencyInletBC(
            velocity_patch,
            {"mixingLength": 0.05, "Cmu": 0.09, "beta": 0.075},
        )
        assert bc.mixing_length == 0.05
        assert bc.C_mu == 0.09
        assert bc.beta == 0.075

    def test_default_values(self, velocity_patch):
        """Default values are correct."""
        bc = TurbulentMixingLengthFrequencyInletBC(velocity_patch)
        assert bc.mixing_length == 0.01
        assert bc.C_mu == 0.09
        assert bc.beta == 0.075

    def test_apply_with_k(self, velocity_patch):
        """apply() computes omega from k and mixing length."""
        bc = TurbulentMixingLengthFrequencyInletBC(
            velocity_patch,
            {"mixingLength": 0.01, "Cmu": 0.09},
        )
        field = torch.zeros(10, dtype=torch.float64)
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        bc.apply(field, k=k)
        # omega = k^0.5 / (C_mu^0.25 * l) = 1.0 / (0.09^0.25 * 0.01)
        expected = 1.0 / (0.09 ** 0.25 * 0.01)
        assert torch.allclose(field[0:3], torch.full((3,), expected, dtype=torch.float64))

    def test_apply_without_k(self, velocity_patch):
        """apply() without k uses default value."""
        bc = TurbulentMixingLengthFrequencyInletBC(velocity_patch)
        field = torch.zeros(10, dtype=torch.float64)
        bc.apply(field)
        # Should use default omega = 0.01
        assert torch.allclose(field[0:3], torch.full((3,), 0.01, dtype=torch.float64))

    def test_matrix_contributions(self, velocity_patch):
        """Matrix contributions use penalty method."""
        bc = TurbulentMixingLengthFrequencyInletBC(velocity_patch)
        field = torch.zeros(10, dtype=torch.float64)
        n_cells = 10
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # Owner cells 0, 1, 2 should have contributions
        assert diag[0] > 0
        assert diag[1] > 0
        assert diag[2] > 0


# ============================================================================
# T9.4: VOF BCs
# ============================================================================


class TestConstantAlphaContactAngleBC:
    """Tests for constantAlphaContactAngle BC."""

    def test_registration(self):
        """constantAlphaContactAngle is registered in RTS."""
        assert "constantAlphaContactAngle" in BoundaryCondition.available_types()

    def test_factory_creation(self, wall_patch):
        """BC can be created via factory method."""
        bc = BoundaryCondition.create(
            "constantAlphaContactAngle",
            wall_patch,
            {"theta0": 90.0},
        )
        assert isinstance(bc, ConstantAlphaContactAngleBC)

    def test_properties(self, wall_patch):
        """Properties are parsed correctly."""
        bc = ConstantAlphaContactAngleBC(wall_patch, {"theta0": 45.0})
        assert bc.theta0 == 45.0
        assert abs(bc.theta0_rad - math.radians(45.0)) < 1e-10

    def test_default_values(self, wall_patch):
        """Default values are correct."""
        bc = ConstantAlphaContactAngleBC(wall_patch)
        assert bc.theta0 == 90.0

    def test_apply_neutral_wetting(self, wall_patch):
        """apply() with theta=90 sets alpha=0.5."""
        bc = ConstantAlphaContactAngleBC(wall_patch, {"theta0": 90.0})
        field = torch.zeros(10, dtype=torch.float64)
        bc.apply(field)
        # alpha = 0.5 * (1 + cos(90°)) = 0.5 * (1 + 0) = 0.5
        assert torch.allclose(field[6:9], torch.full((3,), 0.5, dtype=torch.float64))

    def test_apply_fully_wetting(self, wall_patch):
        """apply() with theta=0 sets alpha=1."""
        bc = ConstantAlphaContactAngleBC(wall_patch, {"theta0": 0.0})
        field = torch.zeros(10, dtype=torch.float64)
        bc.apply(field)
        # alpha = 0.5 * (1 + cos(0°)) = 0.5 * (1 + 1) = 1.0
        assert torch.allclose(field[6:9], torch.ones(3, dtype=torch.float64))

    def test_apply_fully_non_wetting(self, wall_patch):
        """apply() with theta=180 sets alpha=0."""
        bc = ConstantAlphaContactAngleBC(wall_patch, {"theta0": 180.0})
        field = torch.zeros(10, dtype=torch.float64)
        bc.apply(field)
        # alpha = 0.5 * (1 + cos(180°)) = 0.5 * (1 - 1) = 0.0
        assert torch.allclose(field[6:9], torch.zeros(3, dtype=torch.float64))

    def test_apply_partial_wetting(self, wall_patch):
        """apply() with theta=45 sets alpha correctly."""
        bc = ConstantAlphaContactAngleBC(wall_patch, {"theta0": 45.0})
        field = torch.zeros(10, dtype=torch.float64)
        bc.apply(field)
        # alpha = 0.5 * (1 + cos(45°)) = 0.5 * (1 + 0.7071) ≈ 0.8536
        expected = 0.5 * (1.0 + math.cos(math.radians(45.0)))
        assert torch.allclose(field[6:9], torch.full((3,), expected, dtype=torch.float64))

    def test_matrix_contributions(self, wall_patch):
        """Matrix contributions use penalty method."""
        bc = ConstantAlphaContactAngleBC(wall_patch, {"theta0": 90.0})
        field = torch.zeros(10, dtype=torch.float64)
        n_cells = 10
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # Owner cells 6, 7, 8 should have contributions
        assert diag[6] > 0
        assert diag[7] > 0
        assert diag[8] > 0
