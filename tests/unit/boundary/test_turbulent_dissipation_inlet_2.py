"""Tests for enhanced turbulent dissipation inlet boundary condition (v2)."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_dissipation_inlet_2 import TurbulentDissipationInlet2BC


class TestTurbulentDissipationInlet2BC:
    """Test the turbulentDissipationInlet2 boundary condition."""

    def test_registration(self):
        assert "turbulentDissipationInlet2" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentDissipationInlet2", simple_patch,
            {"intensity": 0.05, "lengthScale": 0.01},
        )
        assert isinstance(bc, TurbulentDissipationInlet2BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentDissipationInlet2BC(simple_patch)
        assert bc.type_name == "turbulentDissipationInlet2"

    def test_default_properties(self, simple_patch):
        bc = TurbulentDissipationInlet2BC(simple_patch)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.length_scale == pytest.approx(0.01)
        assert bc.C_mu == pytest.approx(0.09)

    def test_custom_properties(self, simple_patch):
        bc = TurbulentDissipationInlet2BC(simple_patch, {
            "intensity": 0.10, "lengthScale": 0.05, "Cmu": 0.1,
        })
        assert bc.intensity == pytest.approx(0.10)
        assert bc.length_scale == pytest.approx(0.05)
        assert bc.C_mu == pytest.approx(0.1)

    def test_apply_with_velocity(self, simple_patch):
        """epsilon = C_mu^0.75 * k^1.5 / l_mix from intensity."""
        bc = TurbulentDissipationInlet2BC(simple_patch, {
            "intensity": 0.05, "lengthScale": 0.01, "Cmu": 0.09,
        })
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        # Compute expected for first face
        k = 1.5 * (0.05 * 10.0) ** 2
        expected = (0.09 ** 0.75) * (k ** 1.5) / 0.01
        assert field[10] == pytest.approx(expected, rel=1e-6)

    def test_apply_with_k(self, simple_patch):
        bc = TurbulentDissipationInlet2BC(simple_patch, {"lengthScale": 0.02})
        k = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k)

        expected = (0.09 ** 0.75) * (1.0 ** 1.5) / 0.02
        assert field[10] == pytest.approx(expected, rel=1e-6)

    def test_apply_without_args(self, simple_patch):
        bc = TurbulentDissipationInlet2BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)

        assert field[10] == pytest.approx(0.01)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentDissipationInlet2BC(simple_patch)
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5, k=k)

        assert field[5] > 0

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentDissipationInlet2BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([0.02, 0.02, 0.02], dtype=torch.float64))
