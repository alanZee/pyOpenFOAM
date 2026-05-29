"""Tests for compressible N-phase VOF model.

Tests cover:
- CompressibleMultiphaseVoF initialisation and EOS validation
- Phase density from different EOS types
- Mixture density, viscosity, pressure, sound speed
- Volume fraction constraint
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE


class TestCompressibleMultiphaseVoF:
    """Tests for CompressibleMultiphaseVoF."""

    def test_init_two_phases(self):
        from pyfoam.multiphase.compressible_multiphase_vof import CompressibleMultiphaseVoF

        model = CompressibleMultiphaseVoF(
            ["gas", "liquid"],
            ["perfectGas", "incompressible"],
            [1.225, 998.0],
            [1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
        )
        assert model.n_phases == 2
        assert model.phase_names == ["gas", "liquid"]
        assert model.eos_type == ["perfectGas", "incompressible"]

    def test_init_invalid_eos_raises(self):
        from pyfoam.multiphase.compressible_multiphase_vof import CompressibleMultiphaseVoF

        with pytest.raises(ValueError, match="Unknown EOS"):
            CompressibleMultiphaseVoF(
                ["a", "b"],
                ["perfectGas", "invalid"],
                [1.0, 1.0],
                [1e-5, 1e-5],
                R=[287.0, None],
                gamma=[1.4, None],
            )

    def test_init_perfect_gas_without_R_raises(self):
        from pyfoam.multiphase.compressible_multiphase_vof import CompressibleMultiphaseVoF

        with pytest.raises(ValueError, match="R > 0"):
            CompressibleMultiphaseVoF(
                ["gas", "liquid"],
                ["perfectGas", "incompressible"],
                [1.225, 998.0],
                [1.8e-5, 1.002e-3],
                R=[None, None],
                gamma=[1.4, None],
            )

    def test_phase_density_perfect_gas(self):
        from pyfoam.multiphase.compressible_multiphase_vof import CompressibleMultiphaseVoF

        model = CompressibleMultiphaseVoF(
            ["gas", "liquid"],
            ["perfectGas", "incompressible"],
            [1.225, 998.0],
            [1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
        )
        p = torch.full((5,), 101325.0, dtype=CFD_DTYPE)
        T = torch.full((5,), 300.0, dtype=CFD_DTYPE)
        rho = model.phase_density(0, p, T)
        expected = 101325.0 / (287.0 * 300.0)
        assert torch.allclose(rho, torch.full((5,), expected, dtype=CFD_DTYPE), rtol=1e-5)

    def test_phase_density_incompressible(self):
        from pyfoam.multiphase.compressible_multiphase_vof import CompressibleMultiphaseVoF

        model = CompressibleMultiphaseVoF(
            ["gas", "liquid"],
            ["perfectGas", "incompressible"],
            [1.225, 998.0],
            [1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
        )
        p = torch.full((5,), 101325.0, dtype=CFD_DTYPE)
        T = torch.full((5,), 300.0, dtype=CFD_DTYPE)
        rho = model.phase_density(1, p, T)
        assert torch.allclose(rho, torch.full((5,), 998.0, dtype=CFD_DTYPE))

    def test_mixture_density(self):
        from pyfoam.multiphase.compressible_multiphase_vof import CompressibleMultiphaseVoF

        model = CompressibleMultiphaseVoF(
            ["gas", "liquid"],
            ["perfectGas", "incompressible"],
            [1.225, 1000.0],
            [1.8e-5, 1e-3],
            R=[287.0, None],
            gamma=[1.4, None],
        )
        alphas = torch.full((5, 1), 0.3, dtype=CFD_DTYPE)
        p = torch.full((5,), 101325.0, dtype=CFD_DTYPE)
        T = torch.full((5,), 300.0, dtype=CFD_DTYPE)
        rho_m = model.mixture_density(alphas, p, T)
        assert rho_m.shape == (5,)
        assert (rho_m > 0).all()

    def test_mixture_viscosity(self):
        from pyfoam.multiphase.compressible_multiphase_vof import CompressibleMultiphaseVoF

        model = CompressibleMultiphaseVoF(
            ["gas", "liquid"],
            ["perfectGas", "incompressible"],
            [1.225, 998.0],
            [1.8e-5, 1e-3],
            R=[287.0, None],
            gamma=[1.4, None],
        )
        alphas = torch.full((5, 1), 0.5, dtype=CFD_DTYPE)
        mu_m = model.mixture_viscosity(alphas)
        expected = 0.5 * 1.8e-5 + 0.5 * 1e-3
        assert torch.allclose(mu_m, torch.full((5,), expected, dtype=CFD_DTYPE), atol=1e-8)

    def test_mixture_pressure(self):
        from pyfoam.multiphase.compressible_multiphase_vof import CompressibleMultiphaseVoF

        model = CompressibleMultiphaseVoF(
            ["gas", "liquid"],
            ["perfectGas", "incompressible"],
            [1.225, 998.0],
            [1.8e-5, 1e-3],
            R=[287.0, None],
            gamma=[1.4, None],
        )
        alphas = torch.full((3, 1), 0.3, dtype=CFD_DTYPE)
        T = torch.full((3,), 300.0, dtype=CFD_DTYPE)
        p_m = model.mixture_pressure(alphas, T)
        assert p_m.shape == (3,)
        assert (p_m > 0).all()

    def test_sound_speed(self):
        from pyfoam.multiphase.compressible_multiphase_vof import CompressibleMultiphaseVoF

        model = CompressibleMultiphaseVoF(
            ["gas", "liquid"],
            ["perfectGas", "incompressible"],
            [1.225, 998.0],
            [1.8e-5, 1e-3],
            R=[287.0, None],
            gamma=[1.4, None],
        )
        alphas = torch.full((3, 1), 0.3, dtype=CFD_DTYPE)
        p = torch.full((3,), 101325.0, dtype=CFD_DTYPE)
        T = torch.full((3,), 300.0, dtype=CFD_DTYPE)
        a = model.sound_speed(alphas, p, T)
        assert a.shape == (3,)
        assert (a > 0).all()

    def test_validate_alphas(self):
        from pyfoam.multiphase.compressible_multiphase_vof import CompressibleMultiphaseVoF

        model = CompressibleMultiphaseVoF(
            ["gas", "liquid"],
            ["perfectGas", "incompressible"],
            [1.225, 998.0],
            [1.8e-5, 1e-3],
            R=[287.0, None],
            gamma=[1.4, None],
        )
        alphas = torch.tensor([[0.6], [-0.1]], dtype=CFD_DTYPE)
        fixed = model.validate_alphas(alphas)
        assert (fixed >= 0).all()
        assert (fixed <= 1).all()

    def test_repr(self):
        from pyfoam.multiphase.compressible_multiphase_vof import CompressibleMultiphaseVoF

        model = CompressibleMultiphaseVoF(
            ["gas", "liquid"],
            ["perfectGas", "incompressible"],
            [1.225, 998.0],
            [1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
        )
        r = repr(model)
        assert "CompressibleMultiphaseVoF" in r
        assert "gas" in r
