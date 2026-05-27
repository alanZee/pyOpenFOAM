"""Tests for turbulence inlet boundary conditions."""

import pytest
import torch

from pyfoam.boundary import (
    BoundaryCondition,
    TurbulentIntensityKineticEnergyInletBC,
    TurbulentMixingLengthDissipationRateInletBC,
    TurbulentMixingLengthFrequencyInletBC,
)


class TestTurbulentIntensityKineticEnergyInletBC:
    """Test the turbulentIntensityKineticEnergyInlet boundary condition."""

    def test_registration(self):
        """turbulentIntensityKineticEnergyInlet is registered in the RTS registry."""
        assert "turbulentIntensityKineticEnergyInlet" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create(
            "turbulentIntensityKineticEnergyInlet",
            simple_patch,
            {"intensity": 0.05},
        )
        assert isinstance(bc, TurbulentIntensityKineticEnergyInletBC)

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = TurbulentIntensityKineticEnergyInletBC(simple_patch, {"intensity": 0.05})
        assert bc.type_name == "turbulentIntensityKineticEnergyInlet"

    def test_default_intensity(self, simple_patch):
        """Default turbulence intensity is 0.05."""
        bc = TurbulentIntensityKineticEnergyInletBC(simple_patch)
        assert bc.intensity == pytest.approx(0.05)

    def test_custom_intensity(self, simple_patch):
        """Custom turbulence intensity is stored correctly."""
        bc = TurbulentIntensityKineticEnergyInletBC(simple_patch, {"intensity": 0.10})
        assert bc.intensity == pytest.approx(0.10)

    def test_apply_with_velocity(self, simple_patch):
        """k = 1.5 * (I * |U|)^2 when velocity is provided."""
        bc = TurbulentIntensityKineticEnergyInletBC(simple_patch, {"intensity": 0.05})
        # Uniform velocity of 10 m/s in x-direction
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        field = bc.apply(field, velocity=velocity)

        # k = 1.5 * (0.05 * 10)^2 = 1.5 * 0.25 = 0.375
        expected = torch.full((3,), 0.375, dtype=torch.float64)
        assert torch.allclose(field[10:13], expected)

    def test_apply_without_velocity(self, simple_patch):
        """Default k = 0.01 when no velocity is provided."""
        bc = TurbulentIntensityKineticEnergyInletBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field = bc.apply(field)

        expected = torch.full((3,), 0.01, dtype=torch.float64)
        assert torch.allclose(field[10:13], expected)

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() with explicit patch_idx sets values at offset."""
        bc = TurbulentIntensityKineticEnergyInletBC(simple_patch, {"intensity": 0.05})
        velocity = torch.tensor([
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(20, dtype=torch.float64)
        field = bc.apply(field, patch_idx=5, velocity=velocity)

        # k = 1.5 * (0.05 * 4)^2 = 1.5 * 0.04 = 0.06
        expected = torch.full((3,), 0.06, dtype=torch.float64)
        assert torch.allclose(field[5:8], expected)

    def test_matrix_contributions(self, simple_patch):
        """Matrix contributions use penalty method with default k = 0.01."""
        bc = TurbulentIntensityKineticEnergyInletBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0 per face
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # source = coeff * k_default = 2.0 * 0.01 = 0.02
        assert torch.allclose(source, torch.tensor([0.02, 0.02, 0.02], dtype=torch.float64))


class TestTurbulentMixingLengthDissipationRateInletBC:
    """Test the turbulentMixingLengthDissipationRateInlet boundary condition."""

    def test_registration(self):
        """turbulentMixingLengthDissipationRateInlet is registered in the RTS registry."""
        assert "turbulentMixingLengthDissipationRateInlet" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create(
            "turbulentMixingLengthDissipationRateInlet",
            simple_patch,
            {"mixingLength": 0.01},
        )
        assert isinstance(bc, TurbulentMixingLengthDissipationRateInletBC)

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = TurbulentMixingLengthDissipationRateInletBC(simple_patch)
        assert bc.type_name == "turbulentMixingLengthDissipationRateInlet"

    def test_default_coefficients(self, simple_patch):
        """Default mixing_length=0.01, C_mu=0.09."""
        bc = TurbulentMixingLengthDissipationRateInletBC(simple_patch)
        assert bc.mixing_length == pytest.approx(0.01)
        assert bc.C_mu == pytest.approx(0.09)

    def test_custom_coefficients(self, simple_patch):
        """Custom coefficients are stored correctly."""
        bc = TurbulentMixingLengthDissipationRateInletBC(
            simple_patch, {"mixingLength": 0.05, "Cmu": 0.08}
        )
        assert bc.mixing_length == pytest.approx(0.05)
        assert bc.C_mu == pytest.approx(0.08)

    def test_apply_with_k(self, simple_patch):
        """epsilon = C_mu^0.75 * k^1.5 / l when k is provided."""
        bc = TurbulentMixingLengthDissipationRateInletBC(
            simple_patch, {"mixingLength": 0.01, "Cmu": 0.09}
        )
        k = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        field = bc.apply(field, k=k)

        expected_val = 0.09**0.75 * 0.1**1.5 / 0.01
        expected = torch.full((3,), expected_val, dtype=torch.float64)
        assert torch.allclose(field[10:13], expected)

    def test_apply_with_velocity(self, simple_patch):
        """epsilon from estimated k when velocity is provided."""
        bc = TurbulentMixingLengthDissipationRateInletBC(
            simple_patch, {"mixingLength": 0.01, "Cmu": 0.09}
        )
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        field = bc.apply(field, velocity=velocity)

        # k_est = 1.5 * (0.1 * 10)^2 = 1.5 * 1.0 = 1.5
        k_est = 1.5
        expected_val = 0.09**0.75 * k_est**1.5 / 0.01
        expected = torch.full((3,), expected_val, dtype=torch.float64)
        assert torch.allclose(field[10:13], expected)

    def test_apply_without_info(self, simple_patch):
        """Default epsilon = 0.01 when no k or velocity is provided."""
        bc = TurbulentMixingLengthDissipationRateInletBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field = bc.apply(field)

        expected = torch.full((3,), 0.01, dtype=torch.float64)
        assert torch.allclose(field[10:13], expected)

    def test_matrix_contributions(self, simple_patch):
        """Matrix contributions use penalty method with default epsilon = 0.01."""
        bc = TurbulentMixingLengthDissipationRateInletBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # source = coeff * epsilon_default = 2.0 * 0.01 = 0.02
        assert torch.allclose(source, torch.tensor([0.02, 0.02, 0.02], dtype=torch.float64))


class TestTurbulentMixingLengthFrequencyInletBC:
    """Test the turbulentMixingLengthFrequencyInlet boundary condition."""

    def test_registration(self):
        """turbulentMixingLengthFrequencyInlet is registered in the RTS registry."""
        assert "turbulentMixingLengthFrequencyInlet" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create(
            "turbulentMixingLengthFrequencyInlet",
            simple_patch,
            {"mixingLength": 0.01},
        )
        assert isinstance(bc, TurbulentMixingLengthFrequencyInletBC)

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = TurbulentMixingLengthFrequencyInletBC(simple_patch)
        assert bc.type_name == "turbulentMixingLengthFrequencyInlet"

    def test_default_coefficients(self, simple_patch):
        """Default mixing_length=0.01, C_mu=0.09, beta=0.075."""
        bc = TurbulentMixingLengthFrequencyInletBC(simple_patch)
        assert bc.mixing_length == pytest.approx(0.01)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.beta == pytest.approx(0.075)

    def test_custom_coefficients(self, simple_patch):
        """Custom coefficients are stored correctly."""
        bc = TurbulentMixingLengthFrequencyInletBC(
            simple_patch, {"mixingLength": 0.05, "Cmu": 0.08, "beta": 0.1}
        )
        assert bc.mixing_length == pytest.approx(0.05)
        assert bc.C_mu == pytest.approx(0.08)
        assert bc.beta == pytest.approx(0.1)

    def test_apply_with_k(self, simple_patch):
        """omega = k^0.5 / (C_mu^0.25 * l) when k is provided."""
        bc = TurbulentMixingLengthFrequencyInletBC(
            simple_patch, {"mixingLength": 0.01, "Cmu": 0.09}
        )
        k = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        field = bc.apply(field, k=k)

        expected_val = 0.1**0.5 / (0.09**0.25 * 0.01)
        expected = torch.full((3,), expected_val, dtype=torch.float64)
        assert torch.allclose(field[10:13], expected)

    def test_apply_with_velocity(self, simple_patch):
        """omega from estimated k when velocity is provided."""
        bc = TurbulentMixingLengthFrequencyInletBC(
            simple_patch, {"mixingLength": 0.01, "Cmu": 0.09}
        )
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        field = bc.apply(field, velocity=velocity)

        # k_est = 1.5 * (0.1 * 10)^2 = 1.5
        k_est = 1.5
        expected_val = k_est**0.5 / (0.09**0.25 * 0.01)
        expected = torch.full((3,), expected_val, dtype=torch.float64)
        assert torch.allclose(field[10:13], expected)

    def test_apply_without_info(self, simple_patch):
        """Default omega = 0.01 when no k or velocity is provided."""
        bc = TurbulentMixingLengthFrequencyInletBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field = bc.apply(field)

        expected = torch.full((3,), 0.01, dtype=torch.float64)
        assert torch.allclose(field[10:13], expected)

    def test_matrix_contributions(self, simple_patch):
        """Matrix contributions use penalty method with default omega = 0.01."""
        bc = TurbulentMixingLengthFrequencyInletBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # source = coeff * omega_default = 2.0 * 0.01 = 0.02
        assert torch.allclose(source, torch.tensor([0.02, 0.02, 0.02], dtype=torch.float64))
