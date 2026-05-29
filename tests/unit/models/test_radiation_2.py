"""
Unit tests for additional radiation models (radiation_3).

Tests FvDOM, ViewFactor, OpaqueSolid, ConstantAbsorption, and WSGGM models.
"""

from __future__ import annotations

import math

import torch
import pytest

from pyfoam.models.radiation_3 import (
    ConstantAbsorption,
    WSGGM,
    FvDOMModel,
    ViewFactorModel,
    OpaqueSolidModel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def T_field():
    """Standard 8-cell temperature field (K)."""
    return torch.tensor(
        [300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0],
        dtype=torch.float64,
    )


@pytest.fixture
def T_uniform():
    """Uniform temperature field (K)."""
    return torch.full((6,), 500.0, dtype=torch.float64)


@pytest.fixture
def cell_centres_3d():
    """8 cell centres arranged in a cube."""
    return torch.tensor(
        [
            [0.25, 0.25, 0.25],
            [0.75, 0.25, 0.25],
            [0.25, 0.75, 0.25],
            [0.75, 0.75, 0.25],
            [0.25, 0.25, 0.75],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
            [0.75, 0.75, 0.75],
        ],
        dtype=torch.float64,
    )


# ---------------------------------------------------------------------------
# ConstantAbsorption
# ---------------------------------------------------------------------------


class TestConstantAbsorption:

    def test_import(self):
        assert ConstantAbsorption is not None

    def test_default_kappa(self):
        model = ConstantAbsorption()
        assert model.kappa == pytest.approx(0.1)

    def test_custom_kappa(self):
        model = ConstantAbsorption(kappa=0.75)
        assert model.kappa == pytest.approx(0.75)

    def test_negative_kappa_raises(self):
        with pytest.raises(ValueError, match=">= 0"):
            ConstantAbsorption(kappa=-0.1)

    def test_zero_kappa_allowed(self):
        model = ConstantAbsorption(kappa=0.0)
        assert model.kappa == 0.0

    def test_absorption_coeff_shape(self, T_field):
        model = ConstantAbsorption(kappa=0.3)
        kappa = model.absorption_coeff(T_field)
        assert kappa.shape == T_field.shape
        assert torch.allclose(kappa, torch.full_like(T_field, 0.3))

    def test_emission_coeff_equals_absorption(self, T_field):
        model = ConstantAbsorption(kappa=0.5)
        assert torch.allclose(
            model.emission_coeff(T_field),
            model.absorption_coeff(T_field),
        )

    def test_repr(self):
        model = ConstantAbsorption(kappa=0.42)
        assert "ConstantAbsorption" in repr(model)
        assert "0.42" in repr(model)


# ---------------------------------------------------------------------------
# WSGGM
# ---------------------------------------------------------------------------


class TestWSGGM:

    def test_import(self):
        assert WSGGM is not None

    def test_single_gray_gas(self):
        """Single gas: kappa = a * p * exp(-b * T)."""
        model = WSGGM(a_coeffs=[0.5], b_coeffs=[0.001], pressure=0.2)
        T = torch.tensor([300.0, 600.0, 1200.0], dtype=torch.float64)
        kappa = model.absorption_coeff(T)
        expected = 0.5 * 0.2 * torch.exp(-0.001 * T)
        assert torch.allclose(kappa, expected, atol=1e-10)

    def test_multi_gas_sum(self):
        """Multiple gases: kappa = sum(a_i * p * exp(-b_i * T))."""
        a = [0.0, 0.446, 0.161]
        b = [0.0, 12.1, 2.7]
        model = WSGGM(a_coeffs=a, b_coeffs=b, pressure=0.1)
        T = torch.tensor([800.0, 1000.0], dtype=torch.float64)
        kappa = model.absorption_coeff(T)
        # Manual computation
        expected = torch.zeros(2, dtype=torch.float64)
        for a_i, b_i in zip(a, b):
            expected = expected + a_i * 0.1 * torch.exp(-b_i * T)
        assert torch.allclose(kappa, expected, atol=1e-10)

    def test_emission_equals_absorption(self, T_field):
        """Kirchhoff's law: emission == absorption for gray gas."""
        model = WSGGM(a_coeffs=[0.1, 0.3], b_coeffs=[0.0, 0.005], pressure=0.2)
        assert torch.allclose(
            model.emission_coeff(T_field),
            model.absorption_coeff(T_field),
        )

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            WSGGM(a_coeffs=[0.1, 0.2], b_coeffs=[0.001])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            WSGGM(a_coeffs=[], b_coeffs=[])

    def test_high_temperature_increases_absorption(self):
        """With positive b, higher T decreases exp(-bT) so total kappa
        decreases for non-zero b; for b=0 (clear gas) it's constant."""
        model = WSGGM(a_coeffs=[1.0], b_coeffs=[0.0], pressure=0.2)
        T_low = torch.tensor([300.0], dtype=torch.float64)
        T_high = torch.tensor([3000.0], dtype=torch.float64)
        # b=0 -> exp(0)=1 always, so kappa is constant
        assert torch.allclose(
            model.absorption_coeff(T_low),
            model.absorption_coeff(T_high),
        )

    def test_properties(self):
        model = WSGGM(
            a_coeffs=[0.0, 0.5], b_coeffs=[0.0, 0.01], pressure=0.3
        )
        assert model.a_coeffs == [0.0, 0.5]
        assert model.b_coeffs == [0.0, 0.01]
        assert model.pressure == pytest.approx(0.3)

    def test_repr(self):
        model = WSGGM(a_coeffs=[0.0, 0.5], b_coeffs=[0.0, 0.01])
        assert "WSGGM" in repr(model)
        assert "n_gases=2" in repr(model)


# ---------------------------------------------------------------------------
# FvDOMModel
# ---------------------------------------------------------------------------


class TestFvDOMModel:

    def test_import(self):
        assert FvDOMModel is not None

    def test_construction(self):
        dom = FvDOMModel(n_theta=3, n_phi=6, absorption_coeff=0.2)
        assert dom.n_directions == 18
        assert dom.directions.shape == (18, 3)
        assert dom.weights.shape == (18,)

    def test_invalid_n_theta(self):
        with pytest.raises(ValueError, match="n_theta"):
            FvDOMModel(n_theta=1, n_phi=4)

    def test_invalid_n_phi(self):
        with pytest.raises(ValueError, match="n_phi"):
            FvDOMModel(n_theta=4, n_phi=1)

    def test_directions_are_unit_vectors(self):
        dom = FvDOMModel(n_theta=4, n_phi=8)
        norms = dom.directions.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-10)

    def test_weights_positive(self):
        dom = FvDOMModel(n_theta=4, n_phi=8)
        assert (dom.weights > 0).all()

    def test_weights_sum_to_4pi(self):
        """Total solid angle over all directions should equal 4*pi sr."""
        dom = FvDOMModel(n_theta=8, n_phi=16)
        total = dom.weights.sum().item()
        assert total == pytest.approx(4.0 * math.pi, rel=0.05)

    def test_sh_returns_correct_shape(self, T_field):
        dom = FvDOMModel(n_theta=4, n_phi=8)
        S = dom.Sh(T_field)
        assert S.shape == T_field.shape
        assert torch.isfinite(S).all()

    def test_high_T_emits_radiation(self):
        """Non-uniform T should produce non-zero source."""
        dom = FvDOMModel(n_theta=4, n_phi=8, absorption_coeff=0.5)
        T = torch.tensor([300.0, 1200.0], dtype=torch.float64)
        S = dom.Sh(T)
        assert S.abs().sum() > 0

    def test_correct_is_noop(self):
        dom = FvDOMModel(n_theta=2, n_phi=4)
        T = torch.tensor([500.0], dtype=torch.float64)
        S1 = dom.Sh(T)
        dom.correct()
        S2 = dom.Sh(T)
        assert torch.allclose(S1, S2)

    def test_repr(self):
        dom = FvDOMModel(n_theta=3, n_phi=6, absorption_coeff=0.25)
        r = repr(dom)
        assert "FvDOMModel" in r
        assert "18" in r  # n_dirs = 3*6


# ---------------------------------------------------------------------------
# ViewFactorModel
# ---------------------------------------------------------------------------


class TestViewFactorModel:

    def test_import(self):
        assert ViewFactorModel is not None

    def test_construction(self, cell_centres_3d):
        vf = ViewFactorModel(cell_centres_3d, emissivity=0.85)
        assert vf.emissivity == pytest.approx(0.85)
        assert vf.view_factors.shape == (8, 8)

    def test_view_factors_row_sum_to_one(self, cell_centres_3d):
        vf = ViewFactorModel(cell_centres_3d)
        row_sum = vf.view_factors.sum(dim=1)
        assert torch.allclose(
            row_sum, torch.ones(8, dtype=torch.float64), atol=1e-10
        )

    def test_view_factors_diagonal_zero(self, cell_centres_3d):
        vf = ViewFactorModel(cell_centres_3d)
        diag = vf.view_factors.diag()
        assert torch.allclose(diag, torch.zeros(8, dtype=torch.float64))

    def test_view_factors_symmetric_by_construction(self, cell_centres_3d):
        """Distance-based F is symmetric because dist(i,j) == dist(j,i)."""
        vf = ViewFactorModel(cell_centres_3d)
        assert torch.allclose(vf.view_factors, vf.view_factors.T, atol=1e-10)

    def test_invalid_emissivity(self, cell_centres_3d):
        with pytest.raises(ValueError, match="emissivity"):
            ViewFactorModel(cell_centres_3d, emissivity=0.0)
        with pytest.raises(ValueError, match="emissivity"):
            ViewFactorModel(cell_centres_3d, emissivity=1.5)

    def test_sh_returns_correct_shape(self, cell_centres_3d, T_field_8=None):
        T = torch.tensor([300.0, 400.0, 500.0, 600.0,
                          700.0, 800.0, 900.0, 1000.0], dtype=torch.float64)
        vf = ViewFactorModel(cell_centres_3d)
        S = vf.Sh(T)
        assert S.shape == (8,)
        assert torch.isfinite(S).all()

    def test_uniform_T_gives_zero_source(self, cell_centres_3d):
        """Uniform temperature -> no net exchange."""
        T = torch.full((8,), 500.0, dtype=torch.float64)
        vf = ViewFactorModel(cell_centres_3d)
        S = vf.Sh(T)
        assert torch.allclose(S, torch.zeros(8, dtype=torch.float64), atol=1e-10)

    def test_nonuniform_T_gives_nonzero_source(self, cell_centres_3d):
        T = torch.tensor([300.0, 300.0, 300.0, 300.0,
                          1000.0, 1000.0, 1000.0, 1000.0], dtype=torch.float64)
        vf = ViewFactorModel(cell_centres_3d, emissivity=0.9)
        S = vf.Sh(T)
        assert S.abs().sum() > 0

    def test_repr(self, cell_centres_3d):
        vf = ViewFactorModel(cell_centres_3d, emissivity=0.8)
        r = repr(vf)
        assert "ViewFactorModel" in r
        assert "8" in r


# ---------------------------------------------------------------------------
# OpaqueSolidModel
# ---------------------------------------------------------------------------


class TestOpaqueSolidModel:

    def test_import(self):
        assert OpaqueSolidModel is not None

    def test_construction(self):
        solid = OpaqueSolidModel(n_cells=50)
        assert solid.n_cells == 50

    def test_sh_returns_zeros(self, T_field):
        solid = OpaqueSolidModel(n_cells=len(T_field))
        S = solid.Sh(T_field)
        assert torch.allclose(S, torch.zeros_like(T_field))

    def test_sh_shape_matches_input(self):
        solid = OpaqueSolidModel(n_cells=3)
        T = torch.tensor([400.0, 500.0, 600.0], dtype=torch.float64)
        S = solid.Sh(T)
        assert S.shape == (3,)

    def test_correct_is_noop(self):
        solid = OpaqueSolidModel(n_cells=5)
        solid.correct()  # should not raise

    def test_repr(self):
        solid = OpaqueSolidModel(n_cells=128)
        r = repr(solid)
        assert "OpaqueSolidModel" in r
        assert "128" in r
