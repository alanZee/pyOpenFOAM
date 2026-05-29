"""Tests for turbulentMixingLengthDissipationRateInlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulence_bcs import TurbulentMixingLengthDissipationRateInletBC


class TestTurbulentMixingLengthDissipationRateInletBC:
    """Test the turbulentMixingLengthDissipationRateInlet boundary condition."""

    def test_registration(self):
        assert "turbulentMixingLengthDissipationRateInlet" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentMixingLengthDissipationRateInlet", simple_patch,
            {"mixingLength": 0.01, "Cmu": 0.09},
        )
        assert isinstance(bc, TurbulentMixingLengthDissipationRateInletBC)

    def test_type_name(self, simple_patch):
        bc = TurbulentMixingLengthDissipationRateInletBC(simple_patch)
        assert bc.type_name == "turbulentMixingLengthDissipationRateInlet"

    def test_default_coefficients(self, simple_patch):
        bc = TurbulentMixingLengthDissipationRateInletBC(simple_patch)
        assert bc.mixing_length == pytest.approx(0.01)
        assert bc.C_mu == pytest.approx(0.09)

    def test_custom_coefficients(self, simple_patch):
        bc = TurbulentMixingLengthDissipationRateInletBC(simple_patch, {
            "mixingLength": 0.05, "Cmu": 0.09,
        })
        assert bc.mixing_length == pytest.approx(0.05)

    def test_apply_with_k(self, simple_patch):
        """epsilon = C_mu^0.75 * k^1.5 / l."""
        C_mu = 0.09
        l_mix = 0.01
        bc = TurbulentMixingLengthDissipationRateInletBC(simple_patch, {
            "mixingLength": l_mix, "Cmu": C_mu,
        })
        k = torch.tensor([0.01, 0.04, 0.09], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k)

        for i in range(3):
            expected = (C_mu ** 0.75) * (k[i].item() ** 1.5) / l_mix
            assert field[10 + i].item() == pytest.approx(expected, rel=1e-10)

    def test_apply_with_velocity(self, simple_patch):
        """With velocity (no k), estimates k from velocity."""
        C_mu = 0.09
        l_mix = 0.01
        bc = TurbulentMixingLengthDissipationRateInletBC(simple_patch, {
            "mixingLength": l_mix, "Cmu": C_mu,
        })
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        for i, u in enumerate([10.0, 20.0, 30.0]):
            k_est = 1.5 * (0.1 * u) ** 2
            expected = (C_mu ** 0.75) * (k_est ** 1.5) / l_mix
            assert field[10 + i].item() == pytest.approx(expected, rel=1e-10)

    def test_apply_no_info(self, simple_patch):
        """Without k or velocity, uses default epsilon."""
        bc = TurbulentMixingLengthDissipationRateInletBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)

        for i in range(3):
            assert field[10 + i].item() == pytest.approx(0.01)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentMixingLengthDissipationRateInletBC(simple_patch, {
            "mixingLength": 0.01, "Cmu": 0.09,
        })
        k = torch.tensor([0.01, 0.04, 0.09], dtype=torch.float64)
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5, k=k)

        expected = (0.09 ** 0.75) * (0.01 ** 1.5) / 0.01
        assert field[5].item() == pytest.approx(expected, rel=1e-10)

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentMixingLengthDissipationRateInletBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)

        assert diag.shape == (3,)
        assert source.shape == (3,)
        # coeff = delta * area = 2.0 * 1.0 = 2.0
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # source = coeff * epsilon_default = 2.0 * 0.01 = 0.02
        assert torch.allclose(source, torch.tensor([0.02, 0.02, 0.02], dtype=torch.float64))
