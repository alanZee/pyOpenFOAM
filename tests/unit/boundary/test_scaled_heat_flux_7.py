"""Tests for v7 enhanced scaled heat flux boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.scaled_heat_flux_7 import ScaledHeatFlux7BC


class TestScaledHeatFlux7BC:
    """Test the scaledHeatFlux7 boundary condition."""

    def test_registration(self):
        assert "scaledHeatFlux7" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "scaledHeatFlux7", simple_patch,
            {"scale": 2.0, "q_ref": 500.0},
        )
        assert isinstance(bc, ScaledHeatFlux7BC)

    def test_type_name(self, simple_patch):
        bc = ScaledHeatFlux7BC(simple_patch)
        assert bc.type_name == "scaledHeatFlux7"

    def test_default_properties(self, simple_patch):
        bc = ScaledHeatFlux7BC(simple_patch)
        assert bc.scale == pytest.approx(1.0)
        assert bc.q_ref == pytest.approx(0.0)
        assert bc.k == pytest.approx(0.025)
        assert bc.history_coeff == pytest.approx(0.0)
        assert bc.inertia_damp == pytest.approx(0.5)
        assert bc.spatial_corr == pytest.approx(0.0)

    def test_custom_properties(self, simple_patch):
        bc = ScaledHeatFlux7BC(simple_patch, {
            "scale": 3.0, "q_ref": 1000.0, "historyCoeff": 0.1,
            "inertiaDamp": 0.8, "spatialCorr": 0.2, "spatialPeriod": 0.5,
        })
        assert bc.scale == pytest.approx(3.0)
        assert bc.q_ref == pytest.approx(1000.0)
        assert bc.history_coeff == pytest.approx(0.1)
        assert bc.inertia_damp == pytest.approx(0.8)
        assert bc.spatial_corr == pytest.approx(0.2)
        assert bc.spatial_period == pytest.approx(0.5)

    def test_q_property(self, simple_patch):
        bc = ScaledHeatFlux7BC(simple_patch, {
            "scale": 2.0, "q_ref": 500.0, "timeModulation": 0.8, "spatialWeight": 0.9,
        })
        assert bc.q == pytest.approx(2.0 * 500.0 * 0.8 * 0.9)

    def test_gradient_property(self, simple_patch):
        bc = ScaledHeatFlux7BC(simple_patch, {"scale": 1.0, "q_ref": 100.0, "k": 0.5})
        grad = bc.gradient
        assert grad == pytest.approx(-100.0 / 0.5)

    def test_apply_basic(self, simple_patch):
        bc = ScaledHeatFlux7BC(simple_patch, {"scale": 1.0, "q_ref": 500.0})
        field = torch.full((15,), 300.0, dtype=torch.float64)
        bc.apply(field)
        assert torch.all(torch.isfinite(field))

    def test_apply_with_T_field(self, simple_patch):
        bc = ScaledHeatFlux7BC(simple_patch, {"scale": 2.0, "q_ref": 500.0, "k": 0.025})
        field = torch.full((15,), 300.0, dtype=torch.float64)
        T = torch.tensor([310.0, 320.0, 330.0], dtype=torch.float64)
        bc.apply(field, T_field=T)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_spatial_periodicity(self, simple_patch):
        bc = ScaledHeatFlux7BC(simple_patch, {
            "scale": 1.0, "q_ref": 500.0, "spatialDecay": 1.0,
            "spatialCorr": 0.3, "spatialPeriod": 0.5,
        })
        field = torch.full((15,), 300.0, dtype=torch.float64)
        T = torch.tensor([310.0, 320.0, 330.0], dtype=torch.float64)
        bc.apply(field, T_field=T)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_history(self, simple_patch):
        bc = ScaledHeatFlux7BC(simple_patch, {
            "scale": 1.0, "q_ref": 500.0, "historyCoeff": 0.1, "k": 0.025,
        })
        field = torch.full((15,), 300.0, dtype=torch.float64)
        T = torch.tensor([310.0, 320.0, 330.0], dtype=torch.float64)
        # First step: sets T_prev but no history yet
        bc.apply(field, T_field=T, dt=1e-3)
        assert torch.all(torch.isfinite(field[10:13]))
        # Second step: history correction should activate
        T2 = torch.tensor([312.0, 322.0, 332.0], dtype=torch.float64)
        bc.apply(field, T_field=T2, dt=1e-3)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = ScaledHeatFlux7BC(simple_patch)
        field = torch.full((20,), 300.0, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.all(torch.isfinite(field[5:8]))

    def test_matrix_contributions(self, simple_patch):
        bc = ScaledHeatFlux7BC(simple_patch, {"scale": 1.0, "q_ref": 500.0})
        field = torch.full((15,), 300.0, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)

    def test_scale_setter(self, simple_patch):
        bc = ScaledHeatFlux7BC(simple_patch, {"scale": 1.0, "q_ref": 500.0})
        assert bc.q == pytest.approx(500.0)
        bc.scale = 3.0
        assert bc.q == pytest.approx(1500.0)
