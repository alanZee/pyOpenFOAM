"""Tests for enhanced turbulent viscosity inlet boundary condition (v2)."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_viscosity_inlet_2 import TurbulentViscosityInlet2BC


class TestTurbulentViscosityInlet2BC:
    """Test the turbulentViscosityInlet2 boundary condition."""

    def test_registration(self):
        assert "turbulentViscosityInlet2" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentViscosityInlet2", simple_patch,
            {"intensity": 0.05, "lengthScale": 0.01},
        )
        assert isinstance(bc, TurbulentViscosityInlet2BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentViscosityInlet2BC(simple_patch)
        assert bc.type_name == "turbulentViscosityInlet2"

    def test_default_properties(self, simple_patch):
        bc = TurbulentViscosityInlet2BC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.length_scale == pytest.approx(0.01)
        assert bc.nut_min == pytest.approx(1e-10)
        assert bc.nut_max == pytest.approx(1e4)

    def test_custom_properties(self, simple_patch):
        bc = TurbulentViscosityInlet2BC(simple_patch, {
            "Cmu": 0.1, "intensity": 0.10, "lengthScale": 0.05,
            "nutMin": 1e-8, "nutMax": 1e3,
        })
        assert bc.C_mu == pytest.approx(0.1)
        assert bc.intensity == pytest.approx(0.10)
        assert bc.nut_min == pytest.approx(1e-8)
        assert bc.nut_max == pytest.approx(1e3)

    def test_apply_with_velocity(self, simple_patch):
        """nut = C_mu * k^2 / epsilon, derived from intensity and length scale."""
        bc = TurbulentViscosityInlet2BC(simple_patch, {
            "intensity": 0.05, "lengthScale": 0.01, "Cmu": 0.09,
        })
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        # All values should be positive
        assert field[10] > 0
        assert field[11] > 0
        assert field[12] > 0

    def test_apply_clamping(self, simple_patch):
        """Values should be clamped to [nutMin, nutMax]."""
        bc = TurbulentViscosityInlet2BC(simple_patch, {
            "nutMin": 1e-6, "nutMax": 0.5,
        })

        # With very high velocity, nut should be clamped to nutMax
        velocity = torch.tensor([
            [1e6, 0.0, 0.0],
            [1e6, 0.0, 0.0],
            [1e6, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        assert field[10] <= 0.5
        assert field[10] >= 1e-6

    def test_apply_without_velocity(self, simple_patch):
        bc = TurbulentViscosityInlet2BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)

        assert field[10] == pytest.approx(0.001)

    def test_apply_with_k_epsilon(self, simple_patch):
        bc = TurbulentViscosityInlet2BC(simple_patch)
        k = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon)

        # nut = C_mu * k^2 / epsilon = 0.09 * 1.0 / 0.1 = 0.9
        assert field[10] == pytest.approx(0.09 * 1.0 / 0.1, rel=1e-6)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentViscosityInlet2BC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert field[5] == pytest.approx(0.001)

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentViscosityInlet2BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([0.002, 0.002, 0.002], dtype=torch.float64))
