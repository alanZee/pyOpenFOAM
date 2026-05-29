"""
Tests for fvSources: SolidificationMeltingModel, RASourceModel, GravitationalBodyForce.

Test cases:
1. RTS registry
2. SolidificationMeltingModel source terms
3. RASourceModel source terms
4. GravitationalBodyForce source terms
5. Edge cases and validation
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.fv_matrix import FvMatrix
from pyfoam.fv.fv_models import FvModel
from pyfoam.fv.fv_sources import (
    SolidificationMeltingModel,
    RASourceModel,
    GravitationalBodyForce,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_matrix(n_cells: int = 5) -> FvMatrix:
    """Create a minimal FvMatrix for testing (no internal faces)."""
    owner = torch.tensor([], dtype=torch.long)
    neighbour = torch.tensor([], dtype=torch.long)
    return FvMatrix(n_cells, owner, neighbour)


# ---------------------------------------------------------------------------
# RTS Registry
# ---------------------------------------------------------------------------


class TestFvSourcesRegistry:
    """Test that the new fvModels are registered in the RTS registry."""

    def test_solidification_melting_registered(self):
        assert "solidificationMelting" in FvModel.available_types()

    def test_radiation_absorption_registered(self):
        assert "radiationAbsorption" in FvModel.available_types()

    def test_gravitational_body_force_registered(self):
        assert "gravitationalBodyForce" in FvModel.available_types()

    def test_create_solidification_via_factory(self):
        m = FvModel.create("solidificationMelting", T_solidus=273.15)
        assert isinstance(m, SolidificationMeltingModel)

    def test_create_radiation_via_factory(self):
        m = FvModel.create("radiationAbsorption", a=0.05)
        assert isinstance(m, RASourceModel)

    def test_create_gravity_via_factory(self):
        m = FvModel.create("gravitationalBodyForce")
        assert isinstance(m, GravitationalBodyForce)


# ---------------------------------------------------------------------------
# SolidificationMeltingModel
# ---------------------------------------------------------------------------


class TestSolidificationMeltingModel:
    """Test SolidificationMeltingModel (phase change source)."""

    def test_liquid_fraction_below_solidus(self):
        """f_l = 0 when T < T_solidus."""
        f_l = SolidificationMeltingModel._liquid_fraction(
            torch.tensor([250.0]), 273.15, 373.15,
        )
        assert float(f_l[0].item()) == 0.0

    def test_liquid_fraction_above_liquidus(self):
        """f_l = 1 when T > T_liquidus."""
        f_l = SolidificationMeltingModel._liquid_fraction(
            torch.tensor([400.0]), 273.15, 373.15,
        )
        assert float(f_l[0].item()) == 1.0

    def test_liquid_fraction_mid_range(self):
        """f_l linearly interpolates between solidus and liquidus."""
        T_mid = 323.15  # midpoint
        f_l = SolidificationMeltingModel._liquid_fraction(
            torch.tensor([T_mid]), 273.15, 373.15,
        )
        expected = (323.15 - 273.15) / (373.15 - 273.15)
        assert abs(float(f_l[0].item()) - expected) < 1e-6

    def test_pure_substance_case(self):
        """When T_solidus == T_liquidus, f_l is step function."""
        f_l = SolidificationMeltingModel._liquid_fraction(
            torch.tensor([270.0, 273.15, 280.0]), 273.15, 273.15,
        )
        assert float(f_l[0].item()) == 0.0
        assert float(f_l[1].item()) == 1.0
        assert float(f_l[2].item()) == 1.0

    def test_apply_in_mushy_zone(self):
        """In mushy zone (T_solidus < T < T_liquidus), Sp should be non-zero."""
        model = SolidificationMeltingModel(
            T_solidus=273.15, T_liquidus=373.15, L=3.34e5, rho=1000.0,
        )
        matrix = _make_matrix(3)
        T = torch.tensor([300.0, 320.0, 350.0], dtype=torch.float64)
        model.apply(matrix, T)

        # 在糊状区内，Sp 应为负值
        assert float(matrix.diag[0].item()) < 0.0
        assert float(matrix.diag[1].item()) < 0.0
        assert float(matrix.diag[2].item()) < 0.0

    def test_apply_outside_mushy_zone(self):
        """Outside mushy zone, Sp should be zero."""
        model = SolidificationMeltingModel(
            T_solidus=273.15, T_liquidus=373.15, L=3.34e5, rho=1000.0,
        )
        matrix = _make_matrix(3)
        T = torch.tensor([200.0, 500.0, 1000.0], dtype=torch.float64)
        model.apply(matrix, T)

        # 固相和液相区 df_l/dT = 0 → Sp = 0
        assert abs(float(matrix.diag[0].item())) < 1e-10
        assert abs(float(matrix.diag[1].item())) < 1e-10
        assert abs(float(matrix.diag[2].item())) < 1e-10

    def test_cell_restriction(self):
        """Source restricted to specified cells."""
        model = SolidificationMeltingModel(
            T_solidus=273.15, T_liquidus=373.15, L=3.34e5, rho=1000.0,
            cells=[0, 2],
        )
        matrix = _make_matrix(4)
        T = torch.tensor([300.0, 300.0, 300.0, 300.0], dtype=torch.float64)
        model.apply(matrix, T)

        # 只有 cells 0 和 2 有贡献
        assert float(matrix.diag[0].item()) != 0.0
        assert abs(float(matrix.diag[1].item())) < 1e-10
        assert float(matrix.diag[2].item()) != 0.0
        assert abs(float(matrix.diag[3].item())) < 1e-10

    def test_inactive_does_nothing(self):
        """Inactive model should not modify matrix."""
        model = SolidificationMeltingModel(T_solidus=273.15, T_liquidus=373.15)
        model.active = False
        matrix = _make_matrix(3)
        orig_diag = matrix.diag.clone()
        model.apply(matrix, torch.tensor([300.0, 300.0, 300.0], dtype=torch.float64))
        assert torch.allclose(matrix.diag, orig_diag)

    def test_invalid_T_liquidus_raises(self):
        """T_liquidus < T_solidus should raise ValueError."""
        with pytest.raises(ValueError, match="T_liquidus"):
            SolidificationMeltingModel(T_solidus=400.0, T_liquidus=300.0)

    def test_negative_L_raises(self):
        """Negative latent heat should raise ValueError."""
        with pytest.raises(ValueError, match="L"):
            SolidificationMeltingModel(L=-1.0)

    def test_zero_rho_raises(self):
        """Zero or negative rho should raise ValueError."""
        with pytest.raises(ValueError, match="rho"):
            SolidificationMeltingModel(rho=0.0)

    def test_properties(self):
        """Properties return correct values."""
        model = SolidificationMeltingModel(
            T_solidus=273.15, T_liquidus=373.15, L=3.34e5, rho=1000.0,
        )
        assert model.T_solidus == 273.15
        assert model.T_liquidus == 373.15
        assert model.L == 3.34e5
        assert model.rho == 1000.0

    def test_repr(self):
        model = SolidificationMeltingModel(T_solidus=273.15, T_liquidus=373.15)
        r = repr(model)
        assert "SolidificationMeltingModel" in r
        assert "273.15" in r

    def test_type_name(self):
        model = SolidificationMeltingModel()
        assert model.type_name == "solidificationMelting"


# ---------------------------------------------------------------------------
# RASourceModel
# ---------------------------------------------------------------------------


class TestRASourceModel:
    """Test RASourceModel (radiation absorption heat source)."""

    def test_basic_source(self):
        """Q_absorbed = a * I_rad."""
        model = RASourceModel(a=0.1, I_rad=1000.0)
        assert abs(model.Q_absorbed - 100.0) < 1e-6

    def test_apply_all_cells(self):
        """Source applied to all cells when cells=None."""
        model = RASourceModel(a=0.05, I_rad=1000.0)
        matrix = _make_matrix(4)
        T = torch.ones(4, dtype=torch.float64) * 300.0
        model.apply(matrix, T)

        expected_su = torch.full((4,), 50.0, dtype=torch.float64)
        assert torch.allclose(matrix.source, expected_su, atol=1e-10)

    def test_apply_partial_cells(self):
        """Source applied only to specified cells."""
        model = RASourceModel(a=0.1, I_rad=500.0, cells=[1, 3])
        matrix = _make_matrix(5)
        T = torch.ones(5, dtype=torch.float64) * 300.0
        model.apply(matrix, T)

        expected = torch.tensor([0.0, 50.0, 0.0, 50.0, 0.0], dtype=torch.float64)
        assert torch.allclose(matrix.source, expected, atol=1e-10)

    def test_no_implicit_contribution(self):
        """RASourceModel should have no implicit (diagonal) contribution."""
        model = RASourceModel(a=0.1, I_rad=1000.0)
        matrix = _make_matrix(3)
        orig_diag = matrix.diag.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64) * 300.0)
        assert torch.allclose(matrix.diag, orig_diag)

    def test_zero_absorption(self):
        """a=0 produces no source."""
        model = RASourceModel(a=0.0, I_rad=1000.0)
        matrix = _make_matrix(3)
        orig_src = matrix.source.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64) * 300.0)
        assert torch.allclose(matrix.source, orig_src)

    def test_inactive_does_nothing(self):
        """Inactive model should not modify matrix."""
        model = RASourceModel(a=0.1, I_rad=1000.0)
        model.active = False
        matrix = _make_matrix(3)
        orig_src = matrix.source.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64) * 300.0)
        assert torch.allclose(matrix.source, orig_src)

    def test_negative_a_raises(self):
        """Negative absorption coefficient should raise ValueError."""
        with pytest.raises(ValueError, match="a"):
            RASourceModel(a=-0.1)

    def test_negative_I_rad_raises(self):
        """Negative radiation intensity should raise ValueError."""
        with pytest.raises(ValueError, match="I_rad"):
            RASourceModel(I_rad=-100.0)

    def test_properties(self):
        """Properties return correct values."""
        model = RASourceModel(a=0.05, I_rad=800.0)
        assert model.a == 0.05
        assert model.I_rad == 800.0
        assert abs(model.Q_absorbed - 40.0) < 1e-6

    def test_repr(self):
        model = RASourceModel(a=0.1, I_rad=1000.0)
        r = repr(model)
        assert "RASourceModel" in r
        assert "0.1" in r

    def test_type_name(self):
        model = RASourceModel()
        assert model.type_name == "radiationAbsorption"


# ---------------------------------------------------------------------------
# GravitationalBodyForce
# ---------------------------------------------------------------------------


class TestGravitationalBodyForce:
    """Test GravitationalBodyForce (gravity source)."""

    def test_default_gravity(self):
        """Default gravity should be [0, 0, -9.81]."""
        model = GravitationalBodyForce()
        g = model.g
        assert abs(float(g[0].item())) < 1e-10
        assert abs(float(g[1].item())) < 1e-10
        assert abs(float(g[2].item()) - (-9.81)) < 1e-10

    def test_g_mag(self):
        """g_mag should return magnitude."""
        model = GravitationalBodyForce(g=[0, 0, -9.81])
        assert abs(model.g_mag - 9.81) < 1e-6

    def test_apply_full_gravity(self):
        """Full gravity: F = rho * g_z."""
        model = GravitationalBodyForce(g=[0, 0, -9.81])
        matrix = _make_matrix(3)
        rho = torch.tensor([1.0, 1.225, 2.0], dtype=torch.float64)
        model.apply(matrix, rho)

        # g_z = -9.81, F = rho * (-9.81)
        expected = torch.tensor([-9.81, -1.225 * 9.81, -19.62], dtype=torch.float64)
        assert torch.allclose(matrix.source, expected, atol=1e-10)

    def test_apply_boussinesq(self):
        """Boussinesq: F = (rho - rho_ref) * g_z."""
        model = GravitationalBodyForce(g=[0, 0, -9.81], rho_ref=1.0)
        matrix = _make_matrix(3)
        rho = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
        model.apply(matrix, rho)

        # F = (rho - 1.0) * (-9.81)
        expected = torch.tensor(
            [(0.5 - 1.0) * (-9.81), 0.0, (1.5 - 1.0) * (-9.81)],
            dtype=torch.float64,
        )
        assert torch.allclose(matrix.source, expected, atol=1e-10)

    def test_cell_restriction(self):
        """Source restricted to specified cells."""
        model = GravitationalBodyForce(g=[0, 0, -9.81], cells=[0, 2])
        matrix = _make_matrix(4)
        rho = torch.ones(4, dtype=torch.float64)
        model.apply(matrix, rho)

        assert abs(float(matrix.source[0].item()) - (-9.81)) < 1e-10
        assert abs(float(matrix.source[1].item())) < 1e-10
        assert abs(float(matrix.source[2].item()) - (-9.81)) < 1e-10
        assert abs(float(matrix.source[3].item())) < 1e-10

    def test_inactive_does_nothing(self):
        """Inactive model should not modify matrix."""
        model = GravitationalBodyForce()
        model.active = False
        matrix = _make_matrix(3)
        orig_src = matrix.source.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64))
        assert torch.allclose(matrix.source, orig_src)

    def test_custom_gravity_vector(self):
        """Custom gravity vector should work."""
        model = GravitationalBodyForce(g=[1.0, 2.0, -3.0])
        assert abs(float(model.g[0].item()) - 1.0) < 1e-10
        assert abs(float(model.g[1].item()) - 2.0) < 1e-10
        assert abs(float(model.g[2].item()) - (-3.0)) < 1e-10

    def test_invalid_alpha_raises(self):
        """alpha outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            GravitationalBodyForce(alpha=1.5)
        with pytest.raises(ValueError, match="alpha"):
            GravitationalBodyForce(alpha=-0.1)

    def test_properties(self):
        """Properties return correct values."""
        model = GravitationalBodyForce(rho_ref=1.225, alpha=0.5)
        assert model.rho_ref == 1.225
        assert model.alpha == 0.5

    def test_repr(self):
        model = GravitationalBodyForce()
        r = repr(model)
        assert "GravitationalBodyForce" in r
        assert "-9.81" in r

    def test_type_name(self):
        model = GravitationalBodyForce()
        assert model.type_name == "gravitationalBodyForce"

    def test_zero_density_zero_force(self):
        """Zero density should produce zero force."""
        model = GravitationalBodyForce(g=[0, 0, -9.81])
        matrix = _make_matrix(3)
        rho = torch.zeros(3, dtype=torch.float64)
        model.apply(matrix, rho)

        assert torch.allclose(matrix.source, torch.zeros(3, dtype=torch.float64))
