"""Tests for turbulent specific dissipation rate inlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_specific_dissipation_rate_inlet import (
    TurbulentSpecificDissipationRateInletBC,
)


class TestTurbulentSpecificDissipationRateInletBC:
    """Test the turbulentSpecificDissipationRateInlet boundary condition."""

    def test_registration(self):
        assert "turbulentSpecificDissipationRateInlet" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentSpecificDissipationRateInlet", simple_patch,
            {"mixingLength": 0.01, "Cmu": 0.09},
        )
        assert isinstance(bc, TurbulentSpecificDissipationRateInletBC)

    def test_type_name(self, simple_patch):
        bc = TurbulentSpecificDissipationRateInletBC(simple_patch)
        assert bc.type_name == "turbulentSpecificDissipationRateInlet"

    def test_default_properties(self, simple_patch):
        bc = TurbulentSpecificDissipationRateInletBC(simple_patch)
        assert bc.mixing_length == pytest.approx(0.01)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.intensity == pytest.approx(0.1)

    def test_custom_properties(self, simple_patch):
        bc = TurbulentSpecificDissipationRateInletBC(simple_patch, {
            "mixingLength": 0.05, "Cmu": 0.09, "intensity": 0.05,
        })
        assert bc.mixing_length == pytest.approx(0.05)
        assert bc.intensity == pytest.approx(0.05)

    def test_apply_with_k(self, simple_patch):
        """omega = k^0.5 / (C_mu^0.25 * l_mix)"""
        bc = TurbulentSpecificDissipationRateInletBC(simple_patch, {
            "mixingLength": 0.01, "Cmu": 0.09,
        })
        k = torch.tensor([0.01, 0.04, 0.09], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k)

        C_mu = 0.09
        l_mix = 0.01
        expected_o0 = (0.01 ** 0.5) / (C_mu ** 0.25 * l_mix)
        expected_o1 = (0.04 ** 0.5) / (C_mu ** 0.25 * l_mix)
        expected_o2 = (0.09 ** 0.5) / (C_mu ** 0.25 * l_mix)

        assert field[10] == pytest.approx(expected_o0, rel=1e-10)
        assert field[11] == pytest.approx(expected_o1, rel=1e-10)
        assert field[12] == pytest.approx(expected_o2, rel=1e-10)

    def test_apply_with_velocity(self, simple_patch):
        """omega estimated from velocity with fallback intensity."""
        bc = TurbulentSpecificDissipationRateInletBC(simple_patch, {
            "mixingLength": 0.01, "Cmu": 0.09, "intensity": 0.10,
        })
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        C_mu = 0.09
        l_mix = 0.01
        I = 0.10
        u0 = 10.0
        k0 = 1.5 * (I * u0) ** 2
        expected = (k0 ** 0.5) / (C_mu ** 0.25 * l_mix)
        assert field[10] == pytest.approx(expected, rel=1e-10)

    def test_apply_without_k_or_velocity(self, simple_patch):
        """Default omega = 0.01 when no info provided."""
        bc = TurbulentSpecificDissipationRateInletBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)

        assert field[10] == pytest.approx(0.01)
        assert field[11] == pytest.approx(0.01)
        assert field[12] == pytest.approx(0.01)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentSpecificDissipationRateInletBC(simple_patch, {
            "mixingLength": 0.01, "Cmu": 0.09,
        })
        k = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)

        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5, k=k)

        expected = (0.01 ** 0.5) / (0.09 ** 0.25 * 0.01)
        assert field[5] == pytest.approx(expected, rel=1e-10)

    def test_apply_k_priority_over_velocity(self, simple_patch):
        """When both k and velocity are provided, k takes priority."""
        bc = TurbulentSpecificDissipationRateInletBC(simple_patch, {
            "mixingLength": 0.01, "Cmu": 0.09, "intensity": 0.10,
        })
        k = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)
        velocity = torch.tensor([
            [100.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, velocity=velocity)

        expected = (0.01 ** 0.5) / (0.09 ** 0.25 * 0.01)
        assert field[10] == pytest.approx(expected, rel=1e-10)

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentSpecificDissipationRateInletBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([0.02, 0.02, 0.02], dtype=torch.float64))
