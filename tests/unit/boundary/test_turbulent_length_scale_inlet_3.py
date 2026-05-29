"""Tests for v3 enhanced turbulent length scale inlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_length_scale_inlet_3 import TurbulentLengthScaleInlet3BC


class TestTurbulentLengthScaleInlet3BC:
    """Test the turbulentLengthScaleInlet3 boundary condition."""

    def test_registration(self):
        assert "turbulentLengthScaleInlet3" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentLengthScaleInlet3", simple_patch,
            {"intensity": 0.05, "lengthScale": 0.01},
        )
        assert isinstance(bc, TurbulentLengthScaleInlet3BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentLengthScaleInlet3BC(simple_patch)
        assert bc.type_name == "turbulentLengthScaleInlet3"

    def test_default_properties(self, simple_patch):
        bc = TurbulentLengthScaleInlet3BC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.length_scale == pytest.approx(0.01)
        assert bc.length_scale_min == pytest.approx(1e-6)
        assert bc.length_scale_fraction == pytest.approx(0.07)
        assert bc.hydraulic_diameter == pytest.approx(0.1)
        assert bc.alpha == pytest.approx(0.8)

    def test_custom_properties(self, simple_patch):
        bc = TurbulentLengthScaleInlet3BC(simple_patch, {
            "Cmu": 0.1, "intensity": 0.10, "lengthScale": 0.05,
            "lengthScaleMin": 1e-5, "lengthScaleFraction": 0.1,
            "hydraulicDiameter": 0.2, "alpha": 0.6,
        })
        assert bc.C_mu == pytest.approx(0.1)
        assert bc.intensity == pytest.approx(0.10)
        assert bc.length_scale == pytest.approx(0.05)
        assert bc.alpha == pytest.approx(0.6)

    def test_apply_with_k_epsilon(self, simple_patch):
        bc = TurbulentLengthScaleInlet3BC(simple_patch, {"alpha": 0.8})
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon)

        l_computed = (0.09 ** 0.75) * (1.0 ** 1.5) / 0.1
        l_ref = 0.07 * 0.1  # fraction * hydraulicDiameter
        expected = 0.8 * l_computed + 0.2 * l_ref
        # Also clamped
        l_max = 0.07 * 0.1
        expected = max(1e-6, min(expected, l_max))

        assert field[10] == pytest.approx(expected, rel=1e-6)

    def test_apply_with_velocity_only(self, simple_patch):
        bc = TurbulentLengthScaleInlet3BC(simple_patch)
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        assert torch.all(field[10:13] > 0)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_without_args(self, simple_patch):
        bc = TurbulentLengthScaleInlet3BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)

        l_max = 0.07 * 0.1
        expected = max(1e-6, min(0.01, l_max))
        assert field[10] == pytest.approx(expected, rel=1e-6)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentLengthScaleInlet3BC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert field[5] > 0

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentLengthScaleInlet3BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([0.02, 0.02, 0.02], dtype=torch.float64))
