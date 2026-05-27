"""Tests for oscillatingVelocity and turbulentMixingLength boundary conditions."""

import math

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.oscillating_velocity import OscillatingVelocityBC
from pyfoam.boundary.turbulent_mixing_length import TurbulentMixingLengthBC


# ======================================================================
# OscillatingVelocityBC
# ======================================================================


class TestOscillatingVelocityBC:
    """Test the oscillatingVelocity boundary condition."""

    def test_registration(self):
        assert "oscillatingVelocity" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "oscillatingVelocity", simple_patch,
            {"meanVelocity": [1, 0, 0], "amplitude": [0.5, 0, 0], "omega": 1.0},
        )
        assert isinstance(bc, OscillatingVelocityBC)

    def test_type_name(self, simple_patch):
        bc = OscillatingVelocityBC(simple_patch)
        assert bc.type_name == "oscillatingVelocity"

    def test_default_properties(self, simple_patch):
        bc = OscillatingVelocityBC(simple_patch)
        assert torch.allclose(bc.mean_velocity, torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(bc.amplitude, torch.zeros(3, dtype=torch.float64))
        assert bc.omega == pytest.approx(0.0)
        assert bc.phi == pytest.approx(0.0)

    def test_custom_properties(self, simple_patch):
        bc = OscillatingVelocityBC(
            simple_patch,
            {"meanVelocity": [1, 2, 3], "amplitude": [0.5, 0.5, 0.5], "omega": 6.28, "phi": 1.57},
        )
        assert torch.allclose(
            bc.mean_velocity, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        )
        assert torch.allclose(
            bc.amplitude, torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
        )
        assert bc.omega == pytest.approx(6.28)
        assert bc.phi == pytest.approx(1.57)

    def test_apply_at_t_zero(self, simple_patch):
        """At t=0 with phi=0, sin(0)=0 => U = U_mean."""
        bc = OscillatingVelocityBC(
            simple_patch,
            {"meanVelocity": [2, 0, 0], "amplitude": [1, 0, 0], "omega": 1.0, "phi": 0.0},
        )
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field, time=0.0)

        expected = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[10], expected)
        assert torch.allclose(field[11], expected)
        assert torch.allclose(field[12], expected)

    def test_apply_at_peak(self, simple_patch):
        """At omega*t + phi = pi/2, sin = 1 => U = U_mean + U_amp."""
        omega = 1.0
        t = math.pi / (2 * omega)  # omega * t = pi/2
        bc = OscillatingVelocityBC(
            simple_patch,
            {"meanVelocity": [1, 0, 0], "amplitude": [0.5, 0, 0], "omega": omega, "phi": 0.0},
        )
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field, time=t)

        expected = torch.tensor([1.5, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[10], expected, atol=1e-10)

    def test_apply_at_trough(self, simple_patch):
        """At omega*t + phi = 3*pi/2, sin = -1 => U = U_mean - U_amp."""
        omega = 1.0
        t = 3 * math.pi / (2 * omega)
        bc = OscillatingVelocityBC(
            simple_patch,
            {"meanVelocity": [1, 0, 0], "amplitude": [0.5, 0, 0], "omega": omega, "phi": 0.0},
        )
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field, time=t)

        expected = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[10], expected, atol=1e-10)

    def test_apply_with_phi_offset(self, simple_patch):
        """Phase offset shifts the oscillation."""
        omega = 2.0
        phi = math.pi / 2
        t = 0.0
        # omega*t + phi = pi/2 => sin = 1
        bc = OscillatingVelocityBC(
            simple_patch,
            {"meanVelocity": [0, 0, 0], "amplitude": [3, 0, 0], "omega": omega, "phi": phi},
        )
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field, time=t)

        expected = torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[10], expected, atol=1e-10)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = OscillatingVelocityBC(
            simple_patch,
            {"meanVelocity": [5, 0, 0], "amplitude": [1, 0, 0], "omega": 0.0, "phi": 0.0},
        )
        field = torch.zeros((20, 3), dtype=torch.float64)
        bc.apply(field, patch_idx=5, time=0.0)

        expected = torch.tensor([5.0, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[5], expected)

    def test_preserves_internal_field(self, simple_patch):
        bc = OscillatingVelocityBC(
            simple_patch,
            {"meanVelocity": [1, 0, 0], "amplitude": [0, 0, 0], "omega": 1.0},
        )
        field = torch.ones((15, 3), dtype=torch.float64) * 99.0
        original = field.clone()
        bc.apply(field, time=0.0)
        assert torch.allclose(field[:10], original[:10])
        assert torch.allclose(field[13:], original[13:])

    def test_matrix_contributions(self, simple_patch):
        bc = OscillatingVelocityBC(
            simple_patch,
            {"meanVelocity": [1, 0, 0], "amplitude": [0.5, 0, 0], "omega": 0.0},
        )
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells, time=0.0)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0 per face
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # U_mean[0] + U_amp[0]*sin(0) = 1.0 + 0.5*0 = 1.0
        assert torch.allclose(source, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))

    def test_matrix_contributions_at_peak(self, simple_patch):
        """At omega*t = pi/2, source should reflect peak velocity."""
        omega = 1.0
        t = math.pi / (2 * omega)
        bc = OscillatingVelocityBC(
            simple_patch,
            {"meanVelocity": [1, 0, 0], "amplitude": [2, 0, 0], "omega": omega},
        )
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells, time=t)

        # velocity_x = 1 + 2*sin(pi/2) = 3
        # source = coeff * 3.0 = 2.0 * 3.0 = 6.0
        assert torch.allclose(source, torch.tensor([6.0, 6.0, 6.0], dtype=torch.float64), atol=1e-10)


# ======================================================================
# TurbulentMixingLengthBC
# ======================================================================


class TestTurbulentMixingLengthBC:
    """Test the turbulentMixingLength boundary condition."""

    def test_registration(self):
        assert "turbulentMixingLength" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentMixingLength", simple_patch,
            {"mixingLength": 0.01, "mode": "epsilon"},
        )
        assert isinstance(bc, TurbulentMixingLengthBC)

    def test_type_name(self, simple_patch):
        bc = TurbulentMixingLengthBC(simple_patch)
        assert bc.type_name == "turbulentMixingLength"

    def test_default_properties(self, simple_patch):
        bc = TurbulentMixingLengthBC(simple_patch)
        assert bc.mixing_length == pytest.approx(0.01)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.mode == "epsilon"

    def test_custom_properties(self, simple_patch):
        bc = TurbulentMixingLengthBC(
            simple_patch,
            {"mixingLength": 0.05, "Cmu": 0.08, "intensity": 0.1, "mode": "omega"},
        )
        assert bc.mixing_length == pytest.approx(0.05)
        assert bc.C_mu == pytest.approx(0.08)
        assert bc.intensity == pytest.approx(0.1)
        assert bc.mode == "omega"

    def test_apply_epsilon_mode_default(self, simple_patch):
        """Without k or velocity, uses default k=0.01."""
        bc = TurbulentMixingLengthBC(
            simple_patch,
            {"mixingLength": 0.01, "mode": "epsilon"},
        )
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)

        k_default = 0.01
        C_mu = 0.09
        l = 0.01
        expected = (C_mu ** 0.75) * (k_default ** 1.5) / l
        assert torch.allclose(field[10], torch.tensor(expected, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(expected, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(expected, dtype=torch.float64))

    def test_apply_omega_mode_default(self, simple_patch):
        """Omega mode with default k."""
        bc = TurbulentMixingLengthBC(
            simple_patch,
            {"mixingLength": 0.01, "mode": "omega"},
        )
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)

        k_default = 0.01
        C_mu = 0.09
        l = 0.01
        expected = math.sqrt(k_default) / (C_mu ** 0.25 * l)
        assert torch.allclose(field[10], torch.tensor(expected, dtype=torch.float64))

    def test_apply_with_explicit_k(self, simple_patch):
        bc = TurbulentMixingLengthBC(
            simple_patch,
            {"mixingLength": 0.02, "mode": "epsilon"},
        )
        k = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k)

        C_mu = 0.09
        l = 0.02
        expected_0 = (C_mu ** 0.75) * (0.1 ** 1.5) / l
        expected_1 = (C_mu ** 0.75) * (0.2 ** 1.5) / l
        expected_2 = (C_mu ** 0.75) * (0.3 ** 1.5) / l
        assert torch.allclose(field[10], torch.tensor(expected_0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(expected_1, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(expected_2, dtype=torch.float64))

    def test_apply_with_velocity(self, simple_patch):
        """k estimated from velocity: k = 1.5 * (I * |U|)^2."""
        bc = TurbulentMixingLengthBC(
            simple_patch,
            {"mixingLength": 0.01, "intensity": 0.05, "mode": "epsilon"},
        )
        velocity = torch.tensor(
            [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        C_mu = 0.09
        l = 0.01
        I = 0.05
        for i, u_mag in enumerate([10.0, 20.0, 30.0]):
            k_est = 1.5 * (I * u_mag) ** 2
            expected = (C_mu ** 0.75) * (k_est ** 1.5) / l
            assert torch.allclose(
                field[10 + i], torch.tensor(expected, dtype=torch.float64)
            )

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentMixingLengthBC(
            simple_patch,
            {"mixingLength": 0.01, "mode": "epsilon"},
        )
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert field[5] != 0.0

    def test_preserves_internal_field(self, simple_patch):
        bc = TurbulentMixingLengthBC(simple_patch)
        field = torch.ones(15, dtype=torch.float64) * 42.0
        original = field.clone()
        bc.apply(field)
        assert torch.allclose(field[:10], original[:10])
        assert torch.allclose(field[13:], original[13:])

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentMixingLengthBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0 per face
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # default_val = 0.01 => source = 2.0 * 0.01 = 0.02
        assert torch.allclose(source, torch.tensor([0.02, 0.02, 0.02], dtype=torch.float64))
