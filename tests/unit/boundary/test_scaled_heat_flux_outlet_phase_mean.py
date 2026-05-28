"""Tests for scaledHeatFlux and outletPhaseMeanVelocity boundary conditions.

Tests cover:
- RTS registration
- Factory creation
- Property access
- apply() behaviour
- matrix_contributions
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.boundary_condition import Patch
from pyfoam.core.dtype import CFD_DTYPE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wall_patch() -> Patch:
    """A wall patch with 3 faces."""
    return Patch(
        name="wall",
        face_indices=torch.tensor([30, 31, 32]),
        face_normals=torch.tensor([
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
        ], dtype=torch.float64),
        face_areas=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
        delta_coeffs=torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64),
        owner_cells=torch.tensor([0, 1, 2]),
    )


@pytest.fixture
def inlet_patch() -> Patch:
    """An inlet patch with 3 faces."""
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


# ===========================================================================
# ScaledHeatFluxBC tests
# ===========================================================================


class TestScaledHeatFluxBC:
    """Tests for scaledHeatFlux boundary condition."""

    def test_registration(self):
        """scaledHeatFlux is registered in the RTS registry."""
        assert "scaledHeatFlux" in BoundaryCondition.available_types()

    def test_factory_creation(self, wall_patch):
        """BC can be created via the factory method."""
        from pyfoam.boundary.scaled_heat_flux import ScaledHeatFluxBC

        bc = BoundaryCondition.create(
            "scaledHeatFlux", wall_patch,
            coeffs={"scale": 2.0, "q_ref": 500.0, "k": 0.025},
        )
        assert isinstance(bc, ScaledHeatFluxBC)

    def test_default_properties(self, wall_patch):
        """Default coefficients are set correctly."""
        from pyfoam.boundary.scaled_heat_flux import ScaledHeatFluxBC

        bc = ScaledHeatFluxBC(wall_patch)
        assert bc.scale == pytest.approx(1.0)
        assert bc.q_ref == pytest.approx(0.0)
        assert bc.k == pytest.approx(0.025)
        assert bc.T_ref == pytest.approx(300.0)

    def test_custom_properties(self, wall_patch):
        """Custom coefficients are stored correctly."""
        from pyfoam.boundary.scaled_heat_flux import ScaledHeatFluxBC

        bc = ScaledHeatFluxBC(
            wall_patch,
            coeffs={"scale": 3.0, "q_ref": 1000.0, "k": 0.05, "value": 350.0},
        )
        assert bc.scale == pytest.approx(3.0)
        assert bc.q_ref == pytest.approx(1000.0)
        assert bc.k == pytest.approx(0.05)
        assert bc.T_ref == pytest.approx(350.0)

    def test_effective_heat_flux(self, wall_patch):
        """q property returns scale * q_ref."""
        from pyfoam.boundary.scaled_heat_flux import ScaledHeatFluxBC

        bc = ScaledHeatFluxBC(
            wall_patch,
            coeffs={"scale": 2.0, "q_ref": 500.0},
        )
        assert bc.q == pytest.approx(1000.0)

    def test_gradient(self, wall_patch):
        """gradient = -(scale * q_ref) / k."""
        from pyfoam.boundary.scaled_heat_flux import ScaledHeatFluxBC

        bc = ScaledHeatFluxBC(
            wall_patch,
            coeffs={"scale": 2.0, "q_ref": 500.0, "k": 0.025},
        )
        # gradient = -1000 / 0.025 = -40000
        assert bc.gradient == pytest.approx(-40000.0)

    def test_scale_setter(self, wall_patch):
        """scale can be updated."""
        from pyfoam.boundary.scaled_heat_flux import ScaledHeatFluxBC

        bc = ScaledHeatFluxBC(wall_patch)
        bc.scale = 5.0
        assert bc.scale == pytest.approx(5.0)

    def test_apply_sets_temperature(self, wall_patch):
        """apply() sets face temperature based on gradient."""
        from pyfoam.boundary.scaled_heat_flux import ScaledHeatFluxBC

        bc = ScaledHeatFluxBC(
            wall_patch,
            coeffs={"scale": 1.0, "q_ref": 0.0, "value": 300.0},
        )
        field = torch.zeros(35, dtype=torch.float64)
        bc.apply(field)
        # q=0 → gradient=0 → T_face = T_ref = 300
        assert torch.allclose(field[30:33], torch.full((3,), 300.0, dtype=torch.float64))

    def test_apply_with_patch_idx(self, wall_patch):
        """apply() with patch_idx sets correct location."""
        from pyfoam.boundary.scaled_heat_flux import ScaledHeatFluxBC

        bc = ScaledHeatFluxBC(wall_patch, coeffs={"value": 350.0})
        field = torch.zeros(40, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert field[5] == pytest.approx(350.0)
        assert field[6] == pytest.approx(350.0)
        assert field[7] == pytest.approx(350.0)

    def test_matrix_contributions(self, wall_patch):
        """Matrix contributions add heat flux to source."""
        from pyfoam.boundary.scaled_heat_flux import ScaledHeatFluxBC

        bc = ScaledHeatFluxBC(
            wall_patch,
            coeffs={"scale": 2.0, "q_ref": 500.0},
        )
        field = torch.zeros(3, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (3,)
        assert source.shape == (3,)
        # q = 1000, area = 1.0 → source = 1000 per face
        expected_source = torch.full((3,), 1000.0, dtype=torch.float64)
        assert torch.allclose(source, expected_source)


# ===========================================================================
# OutletPhaseMeanVelocityBC tests
# ===========================================================================


class TestOutletPhaseMeanVelocityBC:
    """Tests for outletPhaseMeanVelocity boundary condition."""

    def test_registration(self):
        """outletPhaseMeanVelocity is registered in the RTS registry."""
        assert "outletPhaseMeanVelocity" in BoundaryCondition.available_types()

    def test_factory_creation(self, inlet_patch):
        """BC can be created via the factory method."""
        from pyfoam.boundary.outlet_phase_mean_velocity import OutletPhaseMeanVelocityBC

        bc = BoundaryCondition.create(
            "outletPhaseMeanVelocity", inlet_patch,
            coeffs={"Umean": [1.0, 0.0, 0.0], "phaseName": "gas"},
        )
        assert isinstance(bc, OutletPhaseMeanVelocityBC)

    def test_Umean_property(self, inlet_patch):
        """Umean returns the prescribed velocity."""
        from pyfoam.boundary.outlet_phase_mean_velocity import OutletPhaseMeanVelocityBC

        bc = OutletPhaseMeanVelocityBC(
            inlet_patch,
            coeffs={"Umean": [2.0, 0.5, 0.0]},
        )
        expected = torch.tensor([2.0, 0.5, 0.0], dtype=torch.float64)
        assert torch.allclose(bc.Umean, expected)

    def test_phase_name_property(self, inlet_patch):
        """Phase name is stored correctly."""
        from pyfoam.boundary.outlet_phase_mean_velocity import OutletPhaseMeanVelocityBC

        bc = OutletPhaseMeanVelocityBC(
            inlet_patch,
            coeffs={"phaseName": "gas"},
        )
        assert bc.phase_name == "gas"

    def test_alpha_field_property(self, inlet_patch):
        """Alpha field name is stored correctly."""
        from pyfoam.boundary.outlet_phase_mean_velocity import OutletPhaseMeanVelocityBC

        bc = OutletPhaseMeanVelocityBC(
            inlet_patch,
            coeffs={"alphaField": "alpha.air"},
        )
        assert bc.alpha_field == "alpha.air"

    def test_alpha_min_property(self, inlet_patch):
        """Alpha min threshold is stored correctly."""
        from pyfoam.boundary.outlet_phase_mean_velocity import OutletPhaseMeanVelocityBC

        bc = OutletPhaseMeanVelocityBC(
            inlet_patch,
            coeffs={"alphaMin": 1e-3},
        )
        assert bc.alpha_min == pytest.approx(1e-3)

    def test_apply_sets_velocity(self, inlet_patch):
        """apply() sets face velocities to Umean (no alpha → single-phase)."""
        from pyfoam.boundary.outlet_phase_mean_velocity import OutletPhaseMeanVelocityBC

        bc = OutletPhaseMeanVelocityBC(
            inlet_patch,
            coeffs={"Umean": [1.0, 0.0, 0.0]},
        )
        field = torch.zeros(35, 3, dtype=torch.float64)
        bc.apply(field)
        for i in range(3):
            assert torch.allclose(field[i], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))

    def test_apply_with_alpha(self, inlet_patch):
        """apply() with alpha divides velocity by alpha."""
        from pyfoam.boundary.outlet_phase_mean_velocity import OutletPhaseMeanVelocityBC

        bc = OutletPhaseMeanVelocityBC(
            inlet_patch,
            coeffs={"Umean": [1.0, 0.0, 0.0]},
        )
        field = torch.zeros(35, 3, dtype=torch.float64)
        alpha = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
        bc.apply(field, alpha=alpha)
        # U_phase = Umean / alpha = [2, 0, 0] for alpha=0.5
        for i in range(3):
            assert torch.allclose(field[i], torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64))

    def test_apply_low_alpha_clamped(self, inlet_patch):
        """apply() clamps alpha at alphaMin for outlet robustness."""
        from pyfoam.boundary.outlet_phase_mean_velocity import OutletPhaseMeanVelocityBC

        bc = OutletPhaseMeanVelocityBC(
            inlet_patch,
            coeffs={"Umean": [1.0, 0.0, 0.0], "alphaMin": 1e-4},
        )
        field = torch.zeros(35, 3, dtype=torch.float64)
        alpha = torch.tensor([1e-10, 1e-10, 1e-10], dtype=torch.float64)
        bc.apply(field, alpha=alpha)
        # alpha clamped to 1e-4 → U = 1/1e-4 = 10000
        expected_mag = 1.0 / 1e-4
        for i in range(3):
            assert field[i, 0] == pytest.approx(expected_mag)

    def test_matrix_contributions(self, inlet_patch):
        """Matrix contributions use reduced penalty factor (0.5x)."""
        from pyfoam.boundary.outlet_phase_mean_velocity import OutletPhaseMeanVelocityBC

        bc = OutletPhaseMeanVelocityBC(
            inlet_patch,
            coeffs={"Umean": [1.0, 0.0, 0.0]},
        )
        field = torch.zeros(3, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (3,)
        assert source.shape == (3,)
        assert (diag > 0).any()

    def test_matrix_contributions_half_penalty(self, inlet_patch):
        """Diagonal is half of the full penalty (outlet treatment)."""
        from pyfoam.boundary.outlet_phase_mean_velocity import OutletPhaseMeanVelocityBC
        from pyfoam.boundary.phase_mean_velocity import PhaseMeanVelocityBC

        # Compare with inlet variant
        bc_outlet = OutletPhaseMeanVelocityBC(
            inlet_patch,
            coeffs={"Umean": [1.0, 0.0, 0.0]},
        )
        bc_inlet = PhaseMeanVelocityBC(
            inlet_patch,
            coeffs={"Umean": [1.0, 0.0, 0.0]},
        )

        field = torch.zeros(3, dtype=torch.float64)
        n_cells = 3

        diag_outlet, _ = bc_outlet.matrix_contributions(field.clone(), n_cells)
        diag_inlet, _ = bc_inlet.matrix_contributions(field.clone(), n_cells)

        # Outlet should have 0.5x the diagonal of inlet
        nonzero = diag_inlet > 0
        if nonzero.any():
            ratio = diag_outlet[nonzero] / diag_inlet[nonzero]
            assert torch.allclose(ratio, torch.full_like(ratio, 0.5), rtol=1e-10)
