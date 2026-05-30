"""Tests for v9 enhanced turbulent length scale inlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_length_scale_inlet_9 import TurbulentLengthScaleInlet9BC


class TestTurbulentLengthScaleInlet9BC:
    """Test the turbulentLengthScaleInlet9 boundary condition."""

    def test_registration(self):
        assert "turbulentLengthScaleInlet9" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentLengthScaleInlet9", simple_patch,
            {"lengthScale": 0.01},
        )
        assert isinstance(bc, TurbulentLengthScaleInlet9BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentLengthScaleInlet9BC(simple_patch)
        assert bc.type_name == "turbulentLengthScaleInlet9"

    def test_default_properties(self, simple_patch):
        bc = TurbulentLengthScaleInlet9BC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.length_scale == pytest.approx(0.01)
        assert bc.pg_coeff == pytest.approx(0.05)
        assert bc.pg_norm_ref == pytest.approx(1.0)
        assert bc.rho == pytest.approx(1.225)

    def test_custom_properties(self, simple_patch):
        bc = TurbulentLengthScaleInlet9BC(simple_patch, {
            "pgCoeff": 0.1, "pgNormRef": 2.0, "rho": 1.0,
        })
        assert bc.pg_coeff == pytest.approx(0.1)
        assert bc.pg_norm_ref == pytest.approx(2.0)
        assert bc.rho == pytest.approx(1.0)

    def test_apply_with_k_and_epsilon(self, simple_patch):
        bc = TurbulentLengthScaleInlet9BC(simple_patch)
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon)
        assert torch.all(field[10:13] > 0)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_pressure_gradient(self, simple_patch):
        bc = TurbulentLengthScaleInlet9BC(simple_patch, {"pgCoeff": 0.1})
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        dp_dx = torch.tensor([100.0, -50.0, 0.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon, pressure_gradient=dp_dx)
        assert torch.all(field[10:13] > 0)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_wall_model(self, simple_patch):
        bc = TurbulentLengthScaleInlet9BC(simple_patch, {"wallDist": 0.01})
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon, nu=1e-5)
        assert torch.all(field[10:13] > 0)

    def test_apply_fallback_without_nu(self, simple_patch):
        bc = TurbulentLengthScaleInlet9BC(simple_patch, {"lengthScale": 0.02})
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon)
        # v9 clamps to l_max = lengthScaleFraction * hydraulicDiameter = 0.007
        expected = min((0.09 ** 0.75) * (1.0 ** 1.5) / 0.1, 0.07 * 0.1)
        assert field[10] == pytest.approx(expected, rel=1e-3)

    def test_apply_with_velocity_only(self, simple_patch):
        bc = TurbulentLengthScaleInlet9BC(simple_patch)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert torch.all(field[10:13] > 0)

    def test_apply_without_args(self, simple_patch):
        bc = TurbulentLengthScaleInlet9BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        # v9 clamps to l_max = lengthScaleFraction * hydraulicDiameter = 0.007
        assert field[10] == pytest.approx(0.07 * 0.1)

    def test_apply_clamping(self, simple_patch):
        bc = TurbulentLengthScaleInlet9BC(simple_patch, {
            "lengthScaleMin": 1e-4, "lengthScaleFraction": 0.07, "hydraulicDiameter": 0.1,
        })
        k = torch.tensor([1e-10, 1e6, 1.0], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon)
        assert torch.all(field[10:13] >= 1e-4 - 1e-10)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentLengthScaleInlet9BC(simple_patch)
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5, k=k)
        assert field[5] > 0

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentLengthScaleInlet9BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([0.02, 0.02, 0.02], dtype=torch.float64))
