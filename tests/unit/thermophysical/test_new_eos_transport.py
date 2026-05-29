"""
Tests for new EOS (CubicEOS, PengRobinson, RedlichKwong, VanDerWaals, IcoTabulated)
and transport models (TabulatedTransport, WilkeTransport).
"""

import pytest
import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.equation_of_state import (
    EquationOfState,
    CubicEOS,
    PengRobinsonEOS,
    RedlichKwongEOS,
    VanDerWaalsEOS,
    IcoTabulatedEOS,
)
from pyfoam.thermophysical.tabulated_transport import TabulatedTransport
from pyfoam.thermophysical.wilke_transport import WilkeTransport
from pyfoam.thermophysical.transport_model import ConstantViscosity, Sutherland


# ======================================================================
# PengRobinsonEOS tests
# ======================================================================


class TestPengRobinsonEOS:
    """Tests for Peng-Robinson cubic EOS."""

    def _make_co2(self):
        """Create a CO2 Peng-Robinson EOS."""
        return PengRobinsonEOS(
            Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, accentric=0.228
        )

    def test_constructor_valid(self):
        """Valid parameters should not raise."""
        eos = self._make_co2()
        assert eos._Mw == pytest.approx(44.0)
        assert eos._Tc == pytest.approx(304.13)
        assert eos._Pc == pytest.approx(7.377e6)

    def test_R_value(self):
        """Specific gas constant R = R_universal / Mw."""
        eos = self._make_co2()
        expected_R = 8.314462618 / (44.0 * 1e-3)
        assert eos.R() == pytest.approx(expected_R, rel=1e-6)

    def test_Cp_Cv_gamma(self):
        """Cp, Cv, gamma should be consistent."""
        eos = self._make_co2()
        assert eos.Cp() == pytest.approx(846.0)
        assert eos.Cv() == pytest.approx(846.0 - eos.R())
        assert eos.gamma() == pytest.approx(eos.Cp() / eos.Cv())

    def test_rho_positive(self):
        """Density should be positive at any valid p, T."""
        eos = self._make_co2()
        rho = eos.rho(p=1e6, T=300.0)
        assert float(rho.item()) > 0

    def test_rho_increases_with_pressure(self):
        """Density should increase with pressure at constant T."""
        eos = self._make_co2()
        rho_low = eos.rho(p=1e6, T=350.0)
        rho_high = eos.rho(p=5e6, T=350.0)
        assert float(rho_high.item()) > float(rho_low.item())

    def test_rho_decreases_with_temperature(self):
        """Density should decrease with temperature at constant p."""
        eos = self._make_co2()
        rho_cold = eos.rho(p=2e6, T=300.0)
        rho_hot = eos.rho(p=2e6, T=500.0)
        assert float(rho_hot.item()) < float(rho_cold.item())

    def test_p_roundtrip(self):
        """rho(p(rho, T), T) should approximately return original rho."""
        eos = self._make_co2()
        rho_orig = torch.tensor([10.0, 20.0, 30.0])
        T = torch.tensor([350.0, 400.0, 450.0])
        p = eos.p(rho_orig, T)
        rho_back = eos.rho(p, T)
        # cubic EOS 可能有较大误差，检查相对误差
        assert torch.allclose(rho_orig, rho_back, rtol=0.15)

    def test_tensor_input(self):
        """Should handle tensor inputs."""
        eos = self._make_co2()
        p = torch.tensor([1e6, 2e6, 3e6])
        T = torch.tensor([300.0, 350.0, 400.0])
        rho = eos.rho(p, T)
        assert rho.shape == (3,)
        assert (rho > 0).all()

    def test_H_E(self):
        """Enthalpy and internal energy."""
        eos = self._make_co2()
        T = 300.0
        assert eos.H(T) == pytest.approx(846.0 * 300.0)
        assert eos.E(T) == pytest.approx(eos.Cv() * 300.0)

    def test_invalid_Mw(self):
        with pytest.raises(ValueError, match="Mw must be positive"):
            PengRobinsonEOS(Mw=0, Tc=300, Pc=1e6, Cp=1000)

    def test_invalid_Tc(self):
        with pytest.raises(ValueError, match="Tc must be positive"):
            PengRobinsonEOS(Mw=44, Tc=0, Pc=1e6, Cp=1000)

    def test_invalid_Pc(self):
        with pytest.raises(ValueError, match="Pc must be positive"):
            PengRobinsonEOS(Mw=44, Tc=300, Pc=0, Cp=1000)

    def test_invalid_Cp(self):
        with pytest.raises(ValueError, match="Cp must be positive"):
            PengRobinsonEOS(Mw=44, Tc=300, Pc=1e6, Cp=0)

    def test_repr(self):
        eos = self._make_co2()
        r = repr(eos)
        assert "PengRobinsonEOS" in r
        assert "44" in r


# ======================================================================
# RedlichKwongEOS tests
# ======================================================================


class TestRedlichKwongEOS:
    """Tests for Redlich-Kwong cubic EOS."""

    def _make_methane(self):
        """Create a methane Redlich-Kwong EOS."""
        return RedlichKwongEOS(
            Mw=16.0, Tc=190.56, Pc=4.599e6, Cp=2220.0
        )

    def test_constructor_valid(self):
        eos = self._make_methane()
        assert eos._Mw == pytest.approx(16.0)
        assert eos._Tc == pytest.approx(190.56)

    def test_R_value(self):
        eos = self._make_methane()
        expected_R = 8.314462618 / (16.0 * 1e-3)
        assert eos.R() == pytest.approx(expected_R, rel=1e-6)

    def test_rho_positive(self):
        eos = self._make_methane()
        rho = eos.rho(p=2e6, T=200.0)
        assert float(rho.item()) > 0

    def test_rho_increases_with_pressure(self):
        eos = self._make_methane()
        rho_low = eos.rho(p=1e6, T=250.0)
        rho_high = eos.rho(p=5e6, T=250.0)
        assert float(rho_high.item()) > float(rho_low.item())

    def test_tensor_input(self):
        eos = self._make_methane()
        p = torch.tensor([1e6, 3e6, 5e6])
        T = torch.tensor([200.0, 300.0, 400.0])
        rho = eos.rho(p, T)
        assert rho.shape == (3,)
        assert (rho > 0).all()

    def test_H_E(self):
        eos = self._make_methane()
        T = 300.0
        assert eos.H(T) == pytest.approx(2220.0 * 300.0)
        assert eos.E(T) == pytest.approx(eos.Cv() * 300.0)

    def test_repr(self):
        eos = self._make_methane()
        r = repr(eos)
        assert "RedlichKwongEOS" in r


# ======================================================================
# VanDerWaalsEOS tests
# ======================================================================


class TestVanDerWaalsEOS:
    """Tests for Van der Waals cubic EOS."""

    def _make_n2(self):
        """Create a nitrogen Van der Waals EOS."""
        return VanDerWaalsEOS(
            Mw=28.0, Tc=126.2, Pc=3.39e6, Cp=1040.0
        )

    def test_constructor_valid(self):
        eos = self._make_n2()
        assert eos._Mw == pytest.approx(28.0)
        assert eos._Tc == pytest.approx(126.2)

    def test_R_value(self):
        eos = self._make_n2()
        expected_R = 8.314462618 / (28.0 * 1e-3)
        assert eos.R() == pytest.approx(expected_R, rel=1e-6)

    def test_rho_positive(self):
        eos = self._make_n2()
        rho = eos.rho(p=2e6, T=200.0)
        assert float(rho.item()) > 0

    def test_rho_increases_with_pressure(self):
        eos = self._make_n2()
        rho_low = eos.rho(p=1e6, T=200.0)
        rho_high = eos.rho(p=5e6, T=200.0)
        assert float(rho_high.item()) > float(rho_low.item())

    def test_tensor_input(self):
        eos = self._make_n2()
        p = torch.tensor([1e6, 3e6, 5e6])
        T = torch.tensor([200.0, 300.0, 400.0])
        rho = eos.rho(p, T)
        assert rho.shape == (3,)
        assert (rho > 0).all()

    def test_H_E(self):
        eos = self._make_n2()
        T = 300.0
        assert eos.H(T) == pytest.approx(1040.0 * 300.0)
        assert eos.E(T) == pytest.approx(eos.Cv() * 300.0)

    def test_repr(self):
        eos = self._make_n2()
        r = repr(eos)
        assert "VanDerWaalsEOS" in r

    def test_alpha_constant(self):
        """VDW alpha should be 1 for all temperatures."""
        eos = self._make_n2()
        T = torch.tensor([200.0, 300.0, 500.0])
        alpha = eos._alpha(T)
        assert torch.allclose(alpha, torch.ones_like(T))


# ======================================================================
# IcoTabulatedEOS tests
# ======================================================================


class TestIcoTabulatedEOS:
    """Tests for IcoTabulatedEOS."""

    def _make_eos(self):
        """Create tabulated EOS with simple data."""
        p_data = [1e5, 2e5, 3e5, 4e5, 5e5]
        T_data = [300, 350, 400, 450, 500]
        # 密度随 p 递增、随 T 递减
        rho_data = [
            [1.16, 1.00, 0.87, 0.77, 0.70],
            [2.32, 2.00, 1.74, 1.55, 1.39],
            [3.48, 3.00, 2.61, 2.32, 2.09],
            [4.64, 4.00, 3.48, 3.09, 2.79],
            [5.80, 5.00, 4.35, 3.87, 3.49],
        ]
        return IcoTabulatedEOS(
            p_data=p_data, T_data=T_data, rho_data=rho_data
        )

    def test_constructor_valid(self):
        eos = self._make_eos()
        assert len(eos._p_data) == 5
        assert len(eos._T_data) == 5

    def test_exact_table_points(self):
        """At table points, should return exact values."""
        eos = self._make_eos()
        rho = eos.rho(p=1e5, T=300.0)
        assert float(rho.item()) == pytest.approx(1.16, rel=1e-3)

    def test_interpolated_value(self):
        """Between table points, should interpolate."""
        eos = self._make_eos()
        rho = eos.rho(p=1.5e5, T=325.0)
        # 插值在 [1.16, 2.32] 和 [1.00, 2.00] 之间
        assert 1.0 < float(rho.item()) < 2.5

    def test_rho_increases_with_p(self):
        """Density should increase with pressure."""
        eos = self._make_eos()
        rho1 = eos.rho(p=1e5, T=400.0)
        rho2 = eos.rho(p=3e5, T=400.0)
        assert float(rho2.item()) > float(rho1.item())

    def test_rho_decreases_with_T(self):
        """Density should decrease with temperature at fixed p."""
        eos = self._make_eos()
        rho1 = eos.rho(p=2e5, T=300.0)
        rho2 = eos.rho(p=2e5, T=500.0)
        assert float(rho2.item()) < float(rho1.item())

    def test_clamp_at_boundaries(self):
        """Outside data range, should clamp to boundary values."""
        eos = self._make_eos()
        # Below p range
        rho = eos.rho(p=0.5e5, T=400.0)
        rho_min = eos.rho(p=1e5, T=400.0)
        assert float(rho.item()) == pytest.approx(float(rho_min.item()), rel=1e-3)

    def test_p_inverse(self):
        """p(rho(p, T), T) should approximately recover original p."""
        eos = self._make_eos()
        p_target = 2.5e5
        T = 400.0
        rho = eos.rho(p=p_target, T=T)
        p_back = eos.p(rho=rho, T=T)
        assert float(p_back.item()) == pytest.approx(p_target, rel=0.05)

    def test_tensor_input(self):
        """Should handle tensor inputs."""
        eos = self._make_eos()
        p = torch.tensor([1.5e5, 2.5e5, 3.5e5])
        T = torch.tensor([325.0, 375.0, 425.0])
        rho = eos.rho(p, T)
        assert rho.shape == (3,)
        assert (rho > 0).all()

    def test_H_E(self):
        eos = self._make_eos()
        T = 300.0
        assert eos.H(T) == pytest.approx(1005.0 * 300.0)
        assert eos.E(T) == pytest.approx(eos.Cv() * 300.0)

    def test_invalid_p_data(self):
        with pytest.raises(ValueError, match="at least 2 points"):
            IcoTabulatedEOS(
                p_data=[1e5], T_data=[300, 400], rho_data=[[1, 2]]
            )

    def test_invalid_T_data(self):
        with pytest.raises(ValueError, match="at least 2 points"):
            IcoTabulatedEOS(
                p_data=[1e5, 2e5], T_data=[300], rho_data=[[1], [2]]
            )

    def test_mismatched_rho_data_rows(self):
        with pytest.raises(ValueError, match="rho_data rows"):
            IcoTabulatedEOS(
                p_data=[1e5, 2e5, 3e5],
                T_data=[300, 400],
                rho_data=[[1, 2], [3, 4]],  # only 2 rows for 3 p points
            )

    def test_mismatched_rho_data_cols(self):
        with pytest.raises(ValueError, match="rho_data row"):
            IcoTabulatedEOS(
                p_data=[1e5, 2e5],
                T_data=[300, 400, 500],
                rho_data=[[1, 2], [3, 4, 5]],  # row 0 has 2 cols but T has 3
            )

    def test_repr(self):
        eos = self._make_eos()
        r = repr(eos)
        assert "IcoTabulatedEOS" in r
        assert "100000" in r or "1e5" in r or "grid=5x5" in r


# ======================================================================
# TabulatedTransport tests
# ======================================================================


class TestTabulatedTransport:
    """Tests for TabulatedTransport model."""

    def _make_transport(self, with_kappa=False):
        T_data = [200, 300, 400, 500, 600]
        mu_data = [1.0e-5, 1.8e-5, 2.5e-5, 3.2e-5, 3.8e-5]
        kappa_data = [0.018, 0.026, 0.033, 0.040, 0.046] if with_kappa else None
        return TabulatedTransport(
            T_data=T_data, mu_data=mu_data, kappa_data=kappa_data
        )

    def test_exact_table_point(self):
        """At a table point, should return exact value."""
        transport = self._make_transport()
        mu = transport.mu(T=300.0)
        assert float(mu.item()) == pytest.approx(1.8e-5, rel=1e-4)

    def test_interpolated_value(self):
        """Between table points, should linearly interpolate."""
        transport = self._make_transport()
        mu = transport.mu(T=350.0)
        expected = 0.5 * (1.8e-5 + 2.5e-5)
        assert float(mu.item()) == pytest.approx(expected, rel=1e-4)

    def test_clamp_at_boundaries(self):
        """Outside range, should clamp to boundary values."""
        transport = self._make_transport()
        # Below range
        mu_below = transport.mu(T=100.0)
        mu_min = transport.mu(T=200.0)
        assert float(mu_below.item()) == pytest.approx(float(mu_min.item()))
        # Above range
        mu_above = transport.mu(T=800.0)
        mu_max = transport.mu(T=600.0)
        assert float(mu_above.item()) == pytest.approx(float(mu_max.item()))

    def test_tensor_input(self):
        """Should handle tensor inputs."""
        transport = self._make_transport()
        T = torch.tensor([250.0, 350.0, 450.0])
        mu = transport.mu(T)
        assert mu.shape == (3,)
        assert (mu > 0).all()

    def test_monotonically_increasing(self):
        """Viscosity should increase with temperature."""
        transport = self._make_transport()
        T = torch.tensor([200.0, 300.0, 400.0, 500.0, 600.0])
        mu = transport.mu(T)
        assert torch.all(mu[1:] > mu[:-1])

    def test_kappa_without_data(self):
        """Without kappa_data, kappa = mu * Cp / Pr."""
        transport = self._make_transport(with_kappa=False)
        Cp, Pr = 1005.0, 0.7
        kappa = transport.kappa(T=300.0, Cp=Cp, Pr=Pr)
        mu = transport.mu(T=300.0)
        expected = float(mu.item()) * Cp / Pr
        assert float(kappa.item()) == pytest.approx(expected, rel=1e-6)

    def test_kappa_with_data(self):
        """With kappa_data, should interpolate from table."""
        transport = self._make_transport(with_kappa=True)
        kappa = transport.kappa(T=400.0)
        assert float(kappa.item()) == pytest.approx(0.033, rel=1e-3)

    def test_kappa_interpolated(self):
        """Kappa between table points should interpolate."""
        transport = self._make_transport(with_kappa=True)
        kappa = transport.kappa(T=350.0)
        expected = 0.5 * (0.026 + 0.033)
        assert float(kappa.item()) == pytest.approx(expected, rel=1e-3)

    def test_T_data_property(self):
        transport = self._make_transport()
        assert transport.T_data == [200, 300, 400, 500, 600]

    def test_mu_data_property(self):
        transport = self._make_transport()
        assert len(transport.mu_data) == 5

    def test_kappa_data_property_none(self):
        transport = self._make_transport(with_kappa=False)
        assert transport.kappa_data is None

    def test_kappa_data_property_copy(self):
        transport = self._make_transport(with_kappa=True)
        data = transport.kappa_data
        data.append(999)
        assert len(transport.kappa_data) == 5  # unchanged

    def test_nu(self):
        """Kinematic viscosity = mu / rho."""
        transport = self._make_transport()
        mu = float(transport.mu(T=300.0).item())
        nu = transport.nu(T=300.0, rho=1.2)
        assert float(nu.item()) == pytest.approx(mu / 1.2)

    def test_insufficient_T_data(self):
        with pytest.raises(ValueError, match="at least 2 points"):
            TabulatedTransport(T_data=[300], mu_data=[1.8e-5])

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="mu_data length"):
            TabulatedTransport(T_data=[300, 400], mu_data=[1.8e-5])

    def test_non_increasing_T_data(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            TabulatedTransport(T_data=[300, 300, 400], mu_data=[1e-5, 2e-5, 3e-5])

    def test_repr(self):
        transport = self._make_transport()
        r = repr(transport)
        assert "TabulatedTransport" in r
        assert "Pr-based" in r

    def test_repr_with_kappa(self):
        transport = self._make_transport(with_kappa=True)
        r = repr(transport)
        assert "tabulated" in r


# ======================================================================
# WilkeTransport tests
# ======================================================================


class TestWilkeTransport:
    """Tests for WilkeTransport mixing rule."""

    def _make_air(self):
        """Binary mixture: N2 (79%) + O2 (21%)."""
        return WilkeTransport(
            transport_models=[
                ConstantViscosity(mu=1.76e-5),
                ConstantViscosity(mu=2.05e-5),
            ],
            Mw=[28.014, 31.998],
        )

    def test_constructor_valid(self):
        wilke = self._make_air()
        assert wilke.n_species == 2
        assert wilke.Mw == [28.014, 31.998]

    def test_pure_component_recovery(self):
        """Pure component (x=[1,0]) should recover that component's viscosity."""
        model = ConstantViscosity(mu=1.76e-5)
        wilke = WilkeTransport(
            transport_models=[model, ConstantViscosity(mu=2.05e-5)],
            Mw=[28.014, 31.998],
        )
        mu = wilke.mu(T=300.0, x=[1.0, 0.0])
        assert float(mu.item()) == pytest.approx(1.76e-5, rel=1e-6)

    def test_pure_component_2(self):
        """Pure component (x=[0,1]) should recover that component's viscosity."""
        model = ConstantViscosity(mu=2.05e-5)
        wilke = WilkeTransport(
            transport_models=[ConstantViscosity(mu=1.76e-5), model],
            Mw=[28.014, 31.998],
        )
        mu = wilke.mu(T=300.0, x=[0.0, 1.0])
        assert float(mu.item()) == pytest.approx(2.05e-5, rel=1e-6)

    def test_mixture_between_pure(self):
        """Mixture viscosity should lie between pure component viscosities."""
        wilke = self._make_air()
        mu = wilke.mu(T=300.0, x=[0.79, 0.21])
        mu_val = float(mu.item())
        assert 1.76e-5 <= mu_val <= 2.05e-5

    def test_tensor_T_input(self):
        """Should handle tensor T input."""
        wilke = self._make_air()
        T = torch.tensor([300.0, 400.0, 500.0])
        mu = wilke.mu(T=T, x=[0.79, 0.21])
        assert mu.shape == (3,)
        assert (mu > 0).all()

    def test_kappa_default(self):
        """Kappa should use mu * Cp / Pr."""
        wilke = self._make_air()
        Cp, Pr = 1005.0, 0.7
        kappa = wilke.kappa(T=300.0, x=[0.79, 0.21], Cp=Cp, Pr=Pr)
        mu = wilke.mu(T=300.0, x=[0.79, 0.21])
        expected = float(mu.item()) * Cp / Pr
        assert float(kappa.item()) == pytest.approx(expected, rel=1e-6)

    def test_kappa_per_species(self):
        """Kappa with per-species Cp and Pr."""
        wilke = self._make_air()
        kappa = wilke.kappa(
            T=300.0, x=[0.79, 0.21],
            Cp=[1040.0, 919.0], Pr=[0.71, 0.72],
        )
        assert float(kappa.item()) > 0

    def test_sutherland_models(self):
        """Should work with Sutherland transport models."""
        wilke = WilkeTransport(
            transport_models=[
                Sutherland(mu_ref=1.76e-5, T_ref=273.15, S=111.0),
                Sutherland(mu_ref=2.05e-5, T_ref=273.15, S=139.0),
            ],
            Mw=[28.014, 31.998],
        )
        mu = wilke.mu(T=300.0, x=[0.79, 0.21])
        assert float(mu.item()) > 0

    def test_invalid_empty_models(self):
        with pytest.raises(ValueError, match="must not be empty"):
            WilkeTransport(transport_models=[], Mw=[])

    def test_invalid_mismatched_Mw(self):
        with pytest.raises(ValueError, match="Mw length"):
            WilkeTransport(
                transport_models=[ConstantViscosity(mu=1.8e-5)],
                Mw=[28.0, 32.0],
            )

    def test_invalid_Mw_negative(self):
        with pytest.raises(ValueError, match="Mw.*must be positive"):
            WilkeTransport(
                transport_models=[ConstantViscosity(mu=1.8e-5)],
                Mw=[0.0],
            )

    def test_invalid_x_length(self):
        wilke = self._make_air()
        with pytest.raises(ValueError, match="x length"):
            wilke.mu(T=300.0, x=[0.79])

    def test_repr(self):
        wilke = self._make_air()
        r = repr(wilke)
        assert "WilkeTransport" in r
        assert "2" in r

    def test_T_data_property(self):
        """Mw property should return a copy."""
        wilke = self._make_air()
        mw = wilke.Mw
        mw.append(999)
        assert wilke.Mw == [28.014, 31.998]
