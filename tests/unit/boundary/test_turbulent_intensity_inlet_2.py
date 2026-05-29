"""Tests for enhanced turbulent intensity inlet boundary condition (v2)."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_intensity_inlet_2 import TurbulentIntensityInlet2BC


class TestTurbulentIntensityInlet2BC:
    """Test the turbulentIntensityInlet2 boundary condition."""

    def test_registration(self):
        assert "turbulentIntensityInlet2" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentIntensityInlet2", simple_patch,
            {"intensity": 0.05},
        )
        assert isinstance(bc, TurbulentIntensityInlet2BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentIntensityInlet2BC(simple_patch)
        assert bc.type_name == "turbulentIntensityInlet2"

    def test_default_properties(self, simple_patch):
        bc = TurbulentIntensityInlet2BC(simple_patch)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.k_min == pytest.approx(1e-10)
        assert bc.k_max == pytest.approx(100.0)

    def test_custom_properties(self, simple_patch):
        bc = TurbulentIntensityInlet2BC(simple_patch, {
            "intensity": 0.10, "kMin": 1e-8, "kMax": 50.0,
        })
        assert bc.intensity == pytest.approx(0.10)
        assert bc.k_min == pytest.approx(1e-8)
        assert bc.k_max == pytest.approx(50.0)

    def test_apply_with_velocity(self, simple_patch):
        """k = 1.5 * (I * |U|)^2, clamped."""
        bc = TurbulentIntensityInlet2BC(simple_patch, {"intensity": 0.05})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        expected_k0 = 1.5 * (0.05 * 10.0) ** 2
        assert field[10] == pytest.approx(expected_k0, rel=1e-10)

    def test_apply_clamping_max(self, simple_patch):
        """k should be clamped to kMax."""
        bc = TurbulentIntensityInlet2BC(simple_patch, {
            "intensity": 0.5, "kMax": 0.1,
        })
        velocity = torch.tensor([
            [100.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        assert field[10] <= 0.1 + 1e-12

    def test_apply_without_velocity(self, simple_patch):
        bc = TurbulentIntensityInlet2BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)

        assert field[10] == pytest.approx(0.01)

    def test_compute_kinetic_energy(self, simple_patch):
        bc = TurbulentIntensityInlet2BC(simple_patch, {"intensity": 0.05})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)

        k = bc.compute_kinetic_energy(velocity)
        expected_k0 = 1.5 * (0.05 * 10.0) ** 2
        assert k[0] == pytest.approx(expected_k0, rel=1e-10)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentIntensityInlet2BC(simple_patch, {"intensity": 0.05})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5, velocity=velocity)

        expected = 1.5 * (0.05 * 10.0) ** 2
        assert field[5] == pytest.approx(expected, rel=1e-10)

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentIntensityInlet2BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # k_default = max(kMin, 0.01) = 0.01
        assert torch.allclose(source, torch.tensor([0.02, 0.02, 0.02], dtype=torch.float64))
