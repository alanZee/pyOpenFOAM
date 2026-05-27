"""Tests for fvModels source term framework."""

import pytest
import torch

from pyfoam.core.fv_matrix import FvMatrix
from pyfoam.fv.fv_models import (
    FvModel,
    SemiImplicitSource,
    MassSource,
    HeatSource,
    PorosityForce,
    CodedFvModel,
    create_fv_model,
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


class TestFvModelRegistry:
    """Test the RTS (Run-Time Selection) registry."""

    def test_available_types(self):
        """All five built-in models are registered."""
        types = FvModel.available_types()
        assert "semiImplicitSource" in types
        assert "massSource" in types
        assert "heatSource" in types
        assert "porosityForce" in types
        assert "codedFvModel" in types

    def test_create_semi_implicit(self):
        """SemiImplicitSource created via factory."""
        m = FvModel.create("semiImplicitSource", Su=10.0, Sp=-1.0)
        assert isinstance(m, SemiImplicitSource)

    def test_create_mass_source(self):
        """MassSource created via factory."""
        m = FvModel.create("massSource", mass_source=0.01)
        assert isinstance(m, MassSource)

    def test_create_heat_source(self):
        """HeatSource created via factory."""
        m = FvModel.create("heatSource", Q=1e6)
        assert isinstance(m, HeatSource)

    def test_create_porosity_force(self):
        """PorosityForce created via factory."""
        m = FvModel.create("porosityForce", D=1e8)
        assert isinstance(m, PorosityForce)

    def test_create_coded(self):
        """CodedFvModel created via factory."""
        m = FvModel.create("codedFvModel", code=lambda f: (0.0, 0.0))
        assert isinstance(m, CodedFvModel)

    def test_create_unknown_raises(self):
        """Unknown model name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown fvModel"):
            FvModel.create("nonExistent")

    def test_create_fv_model_function(self):
        """create_fv_model convenience function works."""
        m = create_fv_model("semiImplicitSource", Su=5.0)
        assert isinstance(m, SemiImplicitSource)

    def test_duplicate_registration_raises(self):
        """Registering the same name twice raises ValueError."""
        with pytest.raises(ValueError, match="already registered"):
            @FvModel.register("semiImplicitSource")
            class Duplicate(FvModel):
                def apply(self, matrix, field):
                    pass

    def test_type_name_property(self):
        """type_name returns the registered name."""
        m = SemiImplicitSource(Su=1.0)
        assert m.type_name == "semiImplicitSource"

    def test_coeffs_property(self):
        """coeffs stores the constructor kwargs."""
        m = SemiImplicitSource(Su=1.0, Sp=-0.5)
        assert m.coeffs["Su"] == 1.0
        assert m.coeffs["Sp"] == -0.5

    def test_active_toggle(self):
        """Models can be toggled active/inactive."""
        m = SemiImplicitSource(Su=10.0)
        assert m.active is True
        m.active = False
        assert m.active is False


# ---------------------------------------------------------------------------
# SemiImplicitSource
# ---------------------------------------------------------------------------


class TestSemiImplicitSource:
    """Test SemiImplicitSource (Su + Sp * field)."""

    def test_explicit_only(self):
        """Purely explicit source adds Su to RHS."""
        matrix = _make_matrix(3)
        field = torch.ones(3, dtype=torch.float64)
        model = SemiImplicitSource(Su=10.0, Sp=0.0)
        model.apply(matrix, field)

        expected_source = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
        assert torch.allclose(matrix.source, expected_source)
        # Diagonal unchanged
        assert torch.allclose(matrix.diag, torch.zeros(3, dtype=torch.float64))

    def test_implicit_only(self):
        """Purely implicit source adds Sp to diagonal."""
        matrix = _make_matrix(3)
        field = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        model = SemiImplicitSource(Su=0.0, Sp=-2.0)
        model.apply(matrix, field)

        expected_diag = torch.tensor([-2.0, -2.0, -2.0], dtype=torch.float64)
        assert torch.allclose(matrix.diag, expected_diag)
        assert torch.allclose(matrix.source, torch.zeros(3, dtype=torch.float64))

    def test_semi_implicit(self):
        """Both Su and Sp are applied correctly."""
        matrix = _make_matrix(3)
        field = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        model = SemiImplicitSource(Su=100.0, Sp=-0.5)
        model.apply(matrix, field)

        expected_source = torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64)
        expected_diag = torch.tensor([-0.5, -0.5, -0.5], dtype=torch.float64)
        assert torch.allclose(matrix.source, expected_source)
        assert torch.allclose(matrix.diag, expected_diag)

    def test_tensor_su_sp(self):
        """Per-cell Su and Sp tensors work."""
        matrix = _make_matrix(3)
        field = torch.ones(3, dtype=torch.float64)
        Su = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        Sp = torch.tensor([-0.1, -0.2, -0.3], dtype=torch.float64)
        model = SemiImplicitSource(Su=Su, Sp=Sp)
        model.apply(matrix, field)

        assert torch.allclose(matrix.source, Su)
        assert torch.allclose(matrix.diag, Sp)

    def test_cell_mask(self):
        """Source is restricted to specified cells."""
        matrix = _make_matrix(5)
        field = torch.ones(5, dtype=torch.float64)
        model = SemiImplicitSource(Su=10.0, Sp=-1.0, cells=[1, 3])
        model.apply(matrix, field)

        expected_source = torch.tensor([0.0, 10.0, 0.0, 10.0, 0.0], dtype=torch.float64)
        expected_diag = torch.tensor([0.0, -1.0, 0.0, -1.0, 0.0], dtype=torch.float64)
        assert torch.allclose(matrix.source, expected_source)
        assert torch.allclose(matrix.diag, expected_diag)

    def test_inactive_does_nothing(self):
        """Inactive model does not modify the matrix."""
        matrix = _make_matrix(3)
        original_source = matrix.source.clone()
        original_diag = matrix.diag.clone()
        model = SemiImplicitSource(Su=999.0)
        model.active = False
        model.apply(matrix, torch.ones(3, dtype=torch.float64))
        assert torch.allclose(matrix.source, original_source)
        assert torch.allclose(matrix.diag, original_diag)

    def test_repr(self):
        """repr includes class name and coeffs."""
        m = SemiImplicitSource(Su=1.0, Sp=-0.5)
        r = repr(m)
        assert "SemiImplicitSource" in r

    def test_returns_none(self):
        """apply() returns None (modifies matrix in-place)."""
        m = SemiImplicitSource(Su=1.0)
        result = m.apply(_make_matrix(2), torch.zeros(2, dtype=torch.float64))
        assert result is None


# ---------------------------------------------------------------------------
# MassSource
# ---------------------------------------------------------------------------


class TestMassSource:
    """Test MassSource (continuity equation)."""

    def test_explicit_mass_source(self):
        """Fully explicit mass source (alpha=0)."""
        matrix = _make_matrix(3)
        field = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        model = MassSource(mass_source=0.01, alpha=0.0)
        model.apply(matrix, field)

        expected = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)
        assert torch.allclose(matrix.source, expected)
        assert torch.allclose(matrix.diag, torch.zeros(3, dtype=torch.float64))

    def test_semi_implicit_mass_source(self):
        """Semi-implicit mass source (alpha=0.5)."""
        matrix = _make_matrix(3)
        field = torch.tensor([2.0, 4.0, 1.0], dtype=torch.float64)
        model = MassSource(mass_source=2.0, alpha=0.5)
        model.apply(matrix, field)

        # Su = (1 - 0.5) * 2.0 = 1.0
        expected_su = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        # Sp = 0.5 * 2.0 / field
        expected_sp = torch.tensor([0.5, 0.25, 1.0], dtype=torch.float64)
        assert torch.allclose(matrix.source, expected_su, atol=1e-12)
        assert torch.allclose(matrix.diag, expected_sp, atol=1e-12)

    def test_negative_mass_source(self):
        """Negative mass source (sink) works."""
        matrix = _make_matrix(2)
        field = torch.ones(2, dtype=torch.float64)
        model = MassSource(mass_source=-0.5)
        model.apply(matrix, field)

        expected = torch.tensor([-0.5, -0.5], dtype=torch.float64)
        assert torch.allclose(matrix.source, expected)

    def test_invalid_alpha_raises(self):
        """Alpha outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            MassSource(mass_source=1.0, alpha=-0.1)
        with pytest.raises(ValueError, match="alpha"):
            MassSource(mass_source=1.0, alpha=1.5)

    def test_cell_restriction(self):
        """Source applied only to specified cells."""
        matrix = _make_matrix(4)
        field = torch.ones(4, dtype=torch.float64)
        model = MassSource(mass_source=5.0, alpha=0.0, cells=[0, 2])
        model.apply(matrix, field)

        expected = torch.tensor([5.0, 0.0, 5.0, 0.0], dtype=torch.float64)
        assert torch.allclose(matrix.source, expected)


# ---------------------------------------------------------------------------
# HeatSource
# ---------------------------------------------------------------------------


class TestHeatSource:
    """Test HeatSource (energy equation)."""

    def test_explicit_heat_source(self):
        """Fully explicit heat source (alpha=0)."""
        matrix = _make_matrix(3)
        T = torch.tensor([300.0, 500.0, 400.0], dtype=torch.float64)
        model = HeatSource(Q=1e6, Cp=1005.0, alpha=0.0)
        model.apply(matrix, T)

        expected_su = torch.full((3,), 1e6, dtype=torch.float64)
        assert torch.allclose(matrix.source, expected_su)
        assert torch.allclose(matrix.diag, torch.zeros(3, dtype=torch.float64))

    def test_semi_implicit_heat_source(self):
        """Semi-implicit heat source with alpha > 0."""
        matrix = _make_matrix(3)
        T = torch.tensor([300.0, 500.0, 400.0], dtype=torch.float64)
        alpha = 0.3
        Q = 1e6
        Cp = 1005.0
        model = HeatSource(Q=Q, Cp=Cp, alpha=alpha)
        model.apply(matrix, T)

        expected_su = (1.0 - alpha) * Q * torch.ones(3, dtype=torch.float64)
        # Sp = -alpha * Q / (Cp * T)
        expected_sp = -alpha * Q / (Cp * T)
        assert torch.allclose(matrix.source, expected_su, atol=1e-6)
        assert torch.allclose(matrix.diag, expected_sp, atol=1e-6)

    def test_negative_heat_cooling(self):
        """Negative Q represents cooling."""
        matrix = _make_matrix(2)
        T = torch.tensor([300.0, 300.0], dtype=torch.float64)
        model = HeatSource(Q=-5e5)
        model.apply(matrix, T)

        expected = torch.tensor([-5e5, -5e5], dtype=torch.float64)
        assert torch.allclose(matrix.source, expected)

    def test_invalid_cp_raises(self):
        """Non-positive Cp raises ValueError."""
        with pytest.raises(ValueError, match="Cp"):
            HeatSource(Q=1.0, Cp=0.0)
        with pytest.raises(ValueError, match="Cp"):
            HeatSource(Q=1.0, Cp=-100.0)

    def test_invalid_alpha_raises(self):
        """Alpha outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            HeatSource(Q=1.0, alpha=-0.1)

    def test_cell_restriction(self):
        """Heat source restricted to specified cells."""
        matrix = _make_matrix(4)
        T = torch.ones(4, dtype=torch.float64) * 300.0
        model = HeatSource(Q=1e6, cells=[1])
        model.apply(matrix, T)

        expected = torch.tensor([0.0, 1e6, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(matrix.source, expected)


# ---------------------------------------------------------------------------
# PorosityForce
# ---------------------------------------------------------------------------


class TestPorosityForce:
    """Test PorosityForce (Darcy-Forchheimer porosity resistance)."""

    def test_darcy_only(self):
        """Darcy resistance (purely implicit, Sp contribution)."""
        matrix = _make_matrix(3)
        U = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        D = 1e6
        mu = 1e-3
        model = PorosityForce(D=D, F_coeff=0.0, mu=mu)
        model.apply(matrix, U)

        # Sp_darcy = -mu * D = -1e-3 * 1e6 = -1000
        expected_sp = torch.full((3,), -mu * D, dtype=torch.float64)
        assert torch.allclose(matrix.diag, expected_sp)
        # No explicit source
        assert torch.allclose(matrix.source, torch.zeros(3, dtype=torch.float64))

    def test_forchheimer_only(self):
        """Forchheimer resistance (semi-implicit)."""
        matrix = _make_matrix(3)
        U = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        F_coeff = 10.0
        rho = 1.225
        model = PorosityForce(D=0.0, F_coeff=F_coeff, rho=rho)
        model.apply(matrix, U)

        # Sp_forch = -0.5 * rho * F_coeff * |U|
        expected_sp = -0.5 * rho * F_coeff * U.abs()
        assert torch.allclose(matrix.diag, expected_sp, atol=1e-12)

    def test_combined_darcy_forchheimer(self):
        """Combined Darcy + Forchheimer."""
        matrix = _make_matrix(3)
        U = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
        D = 1e4
        F_coeff = 5.0
        mu = 1e-3
        rho = 1.225
        model = PorosityForce(D=D, F_coeff=F_coeff, mu=mu, rho=rho)
        model.apply(matrix, U)

        sp_darcy = -mu * D
        sp_forch = -0.5 * rho * F_coeff * U.abs()
        expected_sp = sp_darcy + sp_forch
        assert torch.allclose(matrix.diag, expected_sp, atol=1e-12)

    def test_zero_velocity_no_forchheimer(self):
        """Zero velocity => Forchheimer contribution is zero."""
        matrix = _make_matrix(3)
        U = torch.zeros(3, dtype=torch.float64)
        model = PorosityForce(D=1e6, F_coeff=10.0, mu=1e-3, rho=1.225)
        model.apply(matrix, U)

        # Only Darcy: -mu * D
        expected = torch.full((3,), -1e-3 * 1e6, dtype=torch.float64)
        assert torch.allclose(matrix.diag, expected, atol=1e-12)

    def test_invalid_D_raises(self):
        """Negative D raises ValueError."""
        with pytest.raises(ValueError, match="D"):
            PorosityForce(D=-1.0)

    def test_invalid_F_coeff_raises(self):
        """Negative F_coeff raises ValueError."""
        with pytest.raises(ValueError, match="F_coeff"):
            PorosityForce(F_coeff=-1.0)

    def test_cell_restriction(self):
        """Porosity restricted to specified cells."""
        matrix = _make_matrix(4)
        U = torch.ones(4, dtype=torch.float64)
        model = PorosityForce(D=1e6, mu=1e-3, cells=[0, 3])
        model.apply(matrix, U)

        expected_sp = torch.tensor(
            [-1e3, 0.0, 0.0, -1e3], dtype=torch.float64
        )
        assert torch.allclose(matrix.diag, expected_sp, atol=1e-10)


# ---------------------------------------------------------------------------
# CodedFvModel
# ---------------------------------------------------------------------------


class TestCodedFvModel:
    """Test CodedFvModel (user-defined Python function)."""

    def test_constant_source(self):
        """User function returning constant Su, Sp."""
        def const_source(field):
            Su = torch.full_like(field, 42.0)
            Sp = torch.full_like(field, -1.0)
            return Su, Sp

        matrix = _make_matrix(3)
        field = torch.ones(3, dtype=torch.float64)
        model = CodedFvModel(code=const_source, name="const42")
        model.apply(matrix, field)

        assert torch.allclose(
            matrix.source, torch.full((3,), 42.0, dtype=torch.float64)
        )
        assert torch.allclose(
            matrix.diag, torch.full((3,), -1.0, dtype=torch.float64)
        )

    def test_field_dependent_source(self):
        """User function returning field-dependent values."""
        def quadratic_sink(field):
            Su = torch.zeros_like(field)
            Sp = -0.1 * field  # nonlinear sink linearised
            return Su, Sp

        matrix = _make_matrix(3)
        field = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        model = CodedFvModel(code=quadratic_sink, name="quad")
        model.apply(matrix, field)

        expected_sp = -0.1 * field
        assert torch.allclose(matrix.diag, expected_sp, atol=1e-12)

    def test_scalar_return(self):
        """User function returning scalars broadcasts correctly."""
        def scalar_source(field):
            return 5.0, -0.5

        matrix = _make_matrix(3)
        model = CodedFvModel(code=scalar_source)
        model.apply(matrix, torch.ones(3, dtype=torch.float64))

        assert torch.allclose(
            matrix.source, torch.full((3,), 5.0, dtype=torch.float64)
        )
        assert torch.allclose(
            matrix.diag, torch.full((3,), -0.5, dtype=torch.float64)
        )

    def test_non_callable_raises(self):
        """Non-callable code raises TypeError."""
        with pytest.raises(TypeError, match="callable"):
            CodedFvModel(code="not a function")

    def test_name_property(self):
        """name property returns descriptive name."""
        m = CodedFvModel(code=lambda f: (0, 0), name="myModel")
        assert m.name == "myModel"

    def test_repr_shows_name(self):
        """repr includes the model name."""
        m = CodedFvModel(code=lambda f: (0, 0), name="gravity")
        assert "gravity" in repr(m)

    def test_inactive_does_nothing(self):
        """Inactive coded model does not modify matrix."""
        matrix = _make_matrix(3)
        orig_src = matrix.source.clone()
        orig_diag = matrix.diag.clone()
        model = CodedFvModel(code=lambda f: (999.0, 999.0))
        model.active = False
        model.apply(matrix, torch.ones(3, dtype=torch.float64))
        assert torch.allclose(matrix.source, orig_src)
        assert torch.allclose(matrix.diag, orig_diag)


# ---------------------------------------------------------------------------
# Custom registration
# ---------------------------------------------------------------------------


class TestCustomRegistration:
    """Test that users can register custom fvModels."""

    def test_custom_model(self):
        """Register and use a custom model via decorator."""

        @FvModel.register("testCustomSource")
        class CustomSource(FvModel):
            def __init__(self, value: float = 1.0, **kwargs):
                super().__init__(value=value, **kwargs)
                self._value = value

            def apply(self, matrix, field):
                matrix._source += self._value

        m = FvModel.create("testCustomSource", value=7.0)
        assert isinstance(m, CustomSource)

        matrix = _make_matrix(3)
        m.apply(matrix, torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(
            matrix.source, torch.full((3,), 7.0, dtype=torch.float64)
        )

        # Clean up
        del FvModel._registry["testCustomSource"]


# ---------------------------------------------------------------------------
# Integration: multiple models on one matrix
# ---------------------------------------------------------------------------


class TestMultipleModels:
    """Test applying multiple models sequentially to one matrix."""

    def test_combined_sources(self):
        """Heat source + porosity on the same matrix."""
        n = 3
        matrix = _make_matrix(n)
        T = torch.tensor([300.0, 400.0, 500.0], dtype=torch.float64)

        heat = HeatSource(Q=1e6, Cp=1005.0)
        heat.apply(matrix, T)

        # Verify heat added source
        assert torch.allclose(
            matrix.source,
            torch.full((n,), 1e6, dtype=torch.float64),
        )

        # Now add porosity (on a different matrix for a momentum equation)
        mom_matrix = _make_matrix(n)
        U = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        poro = PorosityForce(D=1e6, mu=1e-3)
        poro.apply(mom_matrix, U)

        expected_diag = torch.full((n,), -1e3, dtype=torch.float64)
        assert torch.allclose(mom_matrix.diag, expected_diag, atol=1e-10)

    def test_additive_sources(self):
        """Multiple sources accumulate on the same matrix."""
        matrix = _make_matrix(3)
        field = torch.ones(3, dtype=torch.float64)

        src1 = SemiImplicitSource(Su=10.0, Sp=-1.0)
        src2 = SemiImplicitSource(Su=20.0, Sp=-2.0)
        src1.apply(matrix, field)
        src2.apply(matrix, field)

        assert torch.allclose(
            matrix.source, torch.full((3,), 30.0, dtype=torch.float64)
        )
        assert torch.allclose(
            matrix.diag, torch.full((3,), -3.0, dtype=torch.float64)
        )
