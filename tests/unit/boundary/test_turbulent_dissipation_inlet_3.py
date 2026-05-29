"""Tests for v3 enhanced turbulent dissipation inlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_dissipation_inlet_3 import TurbulentDissipationInlet3BC


class TestTurbulentDissipationInlet3BC:
    """Test the turbulentDissipationInlet3 boundary condition."""

    def test_registration(self):
        assert "turbulentDissipationInlet3" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentDissipationInlet3", simple_patch,
            {"intensity": 0.05, "mixingLength": 0.01},
        )
        assert isinstance(bc, TurbulentDissipationInlet3BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentDissipationInlet3BC(simple_patch)
        assert bc.type_name == "turbulentDissipationInlet3"

    def test_default_properties(self, simple_patch):
        bc = TurbulentDissipationInlet3BC(simple_patch)
        assert bc.mixing_length == pytest.approx(0.01)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.alpha == pytest.approx(1.0)

    def test_custom_properties(self, simple_patch):
        bc = TurbulentDissipationInlet3BC(simple_patch, {
            "mixingLength": 0.05, "Cmu": 0.1, "intensity": 0.10, "alpha": 0.5,
        })
        assert bc.mixing_length == pytest.approx(0.05)
        assert bc.C_mu == pytest.approx(0.1)
        assert bc.intensity == pytest.approx(0.10)
        assert bc.alpha == pytest.approx(0.5)

    def test_apply_with_velocity_only(self, simple_patch):
        """Without k, uses intensity-based epsilon."""
        bc = TurbulentDissipationInlet3BC(simple_patch, {
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
        expected = (0.09 ** 0.75) * (k ** 1.5) / 0.01
        assert field[10] == pytest.approx(expected, rel=1e-6)

    def test_apply_blended(self, simple_patch):
        """With k, velocity, and alpha < 1, epsilon is blended."""
        bc = TurbulentDissipationInlet3BC(simple_patch, {
            "intensity": 0.05, "mixingLength": 0.01, "Cmu": 0.09, "alpha": 0.7,
        })
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, velocity=velocity)

        eps_k = (0.09 ** 0.75) * (1.0 ** 1.5) / 0.01
        k_est = 1.5 * (0.05 * 10.0) ** 2
        eps_intensity = (0.09 ** 0.75) * (k_est ** 1.5) / 0.01
        expected = 0.7 * eps_intensity + 0.3 * eps_k

        assert field[10] == pytest.approx(expected, rel=1e-6)

    def test_apply_with_k_only(self, simple_patch):
        """With k and alpha=1.0 (default), only k-based epsilon is used."""
        bc = TurbulentDissipationInlet3BC(simple_patch, {"mixingLength": 0.02})
        k = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k)

        expected = (0.09 ** 0.75) * (1.0 ** 1.5) / 0.02
        assert field[10] == pytest.approx(expected, rel=1e-6)

    def test_apply_without_args(self, simple_patch):
        bc = TurbulentDissipationInlet3BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)

        assert field[10] == pytest.approx(0.01)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentDissipationInlet3BC(simple_patch)
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5, k=k)

        assert field[5] > 0

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentDissipationInlet3BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([0.02, 0.02, 0.02], dtype=torch.float64))
