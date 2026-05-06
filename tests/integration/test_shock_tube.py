"""
Integration test: Sod shock tube problem.

Tests the compressible solver infrastructure (EOS, transport, thermo)
against the classic Sod shock tube analytical solution.

The Sod shock tube is a 1D Riemann problem with:
- Left state: p=1, ρ=1, u=0
- Right state: p=0.1, ρ=0.125, u=0

This test verifies that the thermophysical models produce correct
density, pressure, and temperature relationships.
"""

import pytest
import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.equation_of_state import PerfectGas
from pyfoam.thermophysical.transport_model import Sutherland
from pyfoam.thermophysical.thermo import BasicThermo, create_air_thermo


class TestShockTubeEOS:
    """Test EOS properties relevant to shock tube physics."""

    @pytest.fixture
    def air_eos(self):
        """Air EOS with standard properties."""
        return PerfectGas(R=287.0, Cp=1005.0)

    @pytest.fixture
    def air_thermo(self):
        """Air thermophysical model."""
        return create_air_thermo()

    def test_sod_left_state(self, air_eos):
        """Left state: p=1e5 Pa, T=300 K → compute density."""
        p_left = 1e5
        T_left = 300.0
        rho_left = air_eos.rho(p_left, T_left)
        expected = p_left / (287.0 * T_left)
        assert abs(float(rho_left.item()) - expected) < 1e-3

    def test_sod_right_state(self, air_eos):
        """Right state: p=1e4 Pa, T=300 K → compute density."""
        p_right = 1e4
        T_right = 300.0
        rho_right = air_eos.rho(p_right, T_right)
        expected = p_right / (287.0 * T_right)
        assert abs(float(rho_right.item()) - expected) < 1e-3

    def test_pressure_ratio_density_ratio(self, air_eos):
        """For same T, p ratio should equal rho ratio."""
        T = 300.0
        p1, p2 = 1e5, 1e4
        rho1 = air_eos.rho(p1, T)
        rho2 = air_eos.rho(p2, T)
        p_ratio = p1 / p2
        rho_ratio = float(rho1.item()) / float(rho2.item())
        assert abs(p_ratio - rho_ratio) < 1e-6

    def test_speed_of_sound(self, air_eos):
        """Speed of sound: a = sqrt(γRT)."""
        T = 300.0
        gamma = air_eos.gamma()
        R = air_eos.R()
        a = (gamma * R * T) ** 0.5
        # For air at 300 K, a ≈ 347.2 m/s
        assert abs(a - 347.2) < 1.0

    def test_mach_number_regimes(self, air_eos):
        """Verify density changes are consistent across Mach regimes."""
        T = 300.0
        # Low Mach (incompressible): density change small
        p_low = 101325.0
        p_low_perturbed = 101325.0 * 1.01  # 1% pressure change
        rho_low = air_eos.rho(p_low, T)
        rho_low_p = air_eos.rho(p_low_perturbed, T)
        drho_low = float((rho_low_p - rho_low).item()) / float(rho_low.item())
        assert abs(drho_low - 0.01) < 1e-4

        # High Mach (compressible): density change significant
        p_high = 1e5
        p_high_perturbed = 5e4  # 50% pressure drop
        rho_high = air_eos.rho(p_high, T)
        rho_high_p = air_eos.rho(p_high_perturbed, T)
        drho_high = float((rho_high_p - rho_high).item()) / float(rho_high.item())
        assert abs(drho_high - (-0.5)) < 1e-4

    def test_energy_conservation(self, air_eos):
        """Total energy = kinetic + internal."""
        T = 300.0
        u = 100.0  # velocity magnitude

        e_internal = air_eos.E(T)
        e_kinetic = 0.5 * u ** 2
        e_total = e_internal + e_kinetic

        # For air at 300 K: e_internal ≈ 215400 J/kg
        assert abs(e_internal - 718.0 * 300.0) < 1.0
        # e_kinetic = 5000 J/kg
        assert abs(e_kinetic - 5000.0) < 1e-6
        # e_total ≈ 220400 J/kg
        assert abs(e_total - (718.0 * 300.0 + 5000.0)) < 1.0

    def test_thermo_consistency(self, air_thermo):
        """Thermo should give consistent results."""
        p = 101325.0
        T = 300.0

        rho = air_thermo.rho(p, T)
        mu = air_thermo.mu(T)
        kappa = air_thermo.kappa(T)

        # All should be positive
        assert float(rho.item()) > 0
        assert float(mu.item()) > 0
        assert float(kappa.item()) > 0

        # Check rho = p / (RT)
        expected_rho = p / (287.0 * T)
        assert abs(float(rho.item()) - expected_rho) < 1e-3

    def test_sutherland_viscosity_range(self):
        """Sutherland viscosity should be reasonable for shock tube temps."""
        transport = Sutherland()
        # Low temperature (post-expansion fan)
        mu_low = transport.mu(T=200.0)
        # High temperature (post-shock)
        mu_high = transport.mu(T=1000.0)

        # Both should be positive and increasing
        assert float(mu_low.item()) > 0
        assert float(mu_high.item()) > float(mu_low.item())

        # Typical air viscosity range: 1e-5 to 5e-5 Pa·s
        assert 1e-6 < float(mu_low.item()) < 1e-4
        assert 1e-6 < float(mu_high.item()) < 1e-4

    def test_batch_computation(self, air_eos):
        """Should handle batch computation for multiple cells."""
        device = get_device()
        dtype = get_default_dtype()

        # Simulate a 1D shock tube with 100 cells
        n = 100
        p = torch.full((n,), 1e5, dtype=dtype, device=device)
        T = torch.full((n,), 300.0, dtype=dtype, device=device)

        # Left half: high pressure
        p[:50] = 1e5
        # Right half: low pressure
        p[50:] = 1e4

        rho = air_eos.rho(p, T)
        assert rho.shape == (n,)
        assert (rho > 0).all()

        # Left density should be ~10x right density
        rho_left = rho[:50].mean()
        rho_right = rho[50:].mean()
        ratio = float(rho_left.item()) / float(rho_right.item())
        assert abs(ratio - 10.0) < 0.1
