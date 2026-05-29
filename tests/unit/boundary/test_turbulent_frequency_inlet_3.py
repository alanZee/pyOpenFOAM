"""Tests for v3 enhanced turbulent frequency inlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_frequency_inlet_3 import TurbulentFrequencyInlet3BC


class TestTurbulentFrequencyInlet3BC:
    """Test the turbulentFrequencyInlet3 boundary condition."""

    def test_registration(self):
        assert "turbulentFrequencyInlet3" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentFrequencyInlet3", simple_patch,
            {"intensity": 0.05, "mixingLength": 0.01},
        )
        assert isinstance(bc, TurbulentFrequencyInlet3BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentFrequencyInlet3BC(simple_patch)
        assert bc.type_name == "turbulentFrequencyInlet3"

    def test_default_properties(self, simple_patch):
        bc = TurbulentFrequencyInlet3BC(simple_patch)
        assert bc.mixing_length == pytest.approx(0.01)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.alpha == pytest.approx(1.0)

    def test_custom_properties(self, simple_patch):
        bc = TurbulentFrequencyInlet3BC(simple_patch, {
            "mixingLength": 0.05, "Cmu": 0.1, "intensity": 0.10, "alpha": 0.5,
        })
        assert bc.mixing_length == pytest.approx(0.05)
        assert bc.C_mu == pytest.approx(0.1)
        assert bc.intensity == pytest.approx(0.10)
        assert bc.alpha == pytest.approx(0.5)

    def test_apply_with_velocity_only(self, simple_patch):
        """Without k, uses intensity-based omega."""
        bc = TurbulentFrequencyInlet3BC(simple_patch, {
            "intensity": 0.05, "mixingLength": 0.01, "Cmu": 0.09, "alpha": 1.0,
        })
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        k = 1.5 * (0.05 * 10.0) ** 2
        expected = (k ** 0.5) / (0.09 ** 0.25 * 0.01)
        assert field[10] == pytest.approx(expected, rel=1e-6)

    def test_apply_blended(self, simple_patch):
        """With k, velocity, and alpha < 1, omega is blended."""
        bc = TurbulentFrequencyInlet3BC(simple_patch, {
            "intensity": 0.05, "mixingLength": 0.01, "Cmu": 0.09, "alpha": 0.6,
        })
        k = torch.tensor([4.0, 4.0, 4.0], dtype=torch.float64)
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, velocity=velocity)

        omega_k = (4.0 ** 0.5) / (0.09 ** 0.25 * 0.01)
        k_est = 1.5 * (0.05 * 10.0) ** 2
        omega_intensity = (k_est ** 0.5) / (0.09 ** 0.25 * 0.01)
        expected = 0.6 * omega_intensity + 0.4 * omega_k

        assert field[10] == pytest.approx(expected, rel=1e-6)

    def test_apply_with_k_only(self, simple_patch):
        bc = TurbulentFrequencyInlet3BC(simple_patch, {"mixingLength": 0.02})
        k = torch.tensor([4.0, 9.0, 16.0], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k)

        expected = (4.0 ** 0.5) / (0.09 ** 0.25 * 0.02)
        assert field[10] == pytest.approx(expected, rel=1e-6)

    def test_apply_without_args(self, simple_patch):
        bc = TurbulentFrequencyInlet3BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)

        assert field[10] == pytest.approx(0.01)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentFrequencyInlet3BC(simple_patch)
        k = torch.tensor([4.0, 4.0, 4.0], dtype=torch.float64)

        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5, k=k)

        assert field[5] > 0

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentFrequencyInlet3BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([0.02, 0.02, 0.02], dtype=torch.float64))
