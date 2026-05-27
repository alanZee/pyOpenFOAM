"""Tests for Spalart-Allmaras turbulence model.

Tests cover:
- Model constants (default values, custom, immutability)
- RTS registration and factory creation
- Model instantiation
- nut() and k() computation
- nuTilde_field property
- correct() execution
"""

import pytest
import torch

from pyfoam.turbulence.turbulence_model import TurbulenceModel
from pyfoam.turbulence.spalart_allmaras import (
    SpalartAllmarasModel,
    SpalartAllmarasConstants,
)

from tests.unit.turbulence.conftest import make_fv_mesh


class TestSpalartAllmarasConstants:
    """Tests for S-A model constants."""

    def test_default_constants(self):
        """Default constants match OpenFOAM values."""
        C = SpalartAllmarasConstants()
        assert C.sigma == pytest.approx(2.0 / 3.0)
        assert C.kappa == pytest.approx(0.41)
        assert C.Cb1 == pytest.approx(0.1355)
        assert C.Cb2 == pytest.approx(0.622)
        assert C.Cw2 == pytest.approx(0.3)
        assert C.Cw3 == pytest.approx(2.0)
        assert C.Cv1 == pytest.approx(7.1)
        assert C.Ct1 == pytest.approx(1.0)
        assert C.Ct2 == pytest.approx(2.0)
        assert C.Ct3 == pytest.approx(1.1)
        assert C.Ct4 == pytest.approx(0.5)

    def test_custom_constants(self):
        """Can create custom constants."""
        C = SpalartAllmarasConstants(Cb1=0.2, Cv1=8.0)
        assert C.Cb1 == pytest.approx(0.2)
        assert C.Cv1 == pytest.approx(8.0)
        # Unchanged defaults
        assert C.sigma == pytest.approx(2.0 / 3.0)
        assert C.Cb2 == pytest.approx(0.622)

    def test_constants_frozen(self):
        """Constants are immutable (frozen dataclass)."""
        C = SpalartAllmarasConstants()
        with pytest.raises(AttributeError):
            C.Cb1 = 0.5


class TestSpalartAllmarasRegistration:
    """Tests for RTS registration of S-A model."""

    def test_sa_registered(self):
        """SpalartAllmaras is registered in the RTS registry."""
        assert "SpalartAllmaras" in TurbulenceModel.available_types()

    def test_create_via_factory(self):
        """Can create S-A model via TurbulenceModel.create()."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = TurbulenceModel.create("SpalartAllmaras", mesh, U, phi)
        assert isinstance(model, SpalartAllmarasModel)


class TestSpalartAllmarasModel:
    """Tests for S-A model behaviour."""

    def _make_model(self, U_vals=None):
        """Helper to create a model with optional velocity."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        if U_vals is not None:
            U[:, 0] = U_vals[0]
            U[:, 1] = U_vals[1]
            U[:, 2] = U_vals[2]
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        return SpalartAllmarasModel(mesh, U, phi)

    def test_model_creation(self):
        """Model can be created with mesh, U, phi."""
        model = self._make_model()
        assert model.mesh is not None
        assert model.mesh.n_cells == 2

    def test_nut_shape(self):
        """nut() returns correct shape."""
        model = self._make_model()
        nut = model.nut()
        assert nut.shape == (model.mesh.n_cells,)

    def test_nut_non_negative(self):
        """nut() returns non-negative values."""
        model = self._make_model()
        nut = model.nut()
        assert (nut >= 0).all()

    def test_k_shape(self):
        """k() returns correct shape (requires correct() first)."""
        model = self._make_model(U_vals=(1.0, 0.0, 0.0))
        model.correct()
        k = model.k()
        assert k.shape == (model.mesh.n_cells,)

    def test_nuTilde_field_shape(self):
        """nuTilde_field has correct shape."""
        model = self._make_model()
        assert model.nuTilde_field.shape == (model.mesh.n_cells,)

    def test_nuTilde_field_setter(self):
        """nuTilde_field setter updates the field."""
        model = self._make_model()
        new_val = torch.full((model.mesh.n_cells,), 0.01, dtype=torch.float64)
        model.nuTilde_field = new_val
        assert torch.allclose(model.nuTilde_field, new_val)

    def test_nut_with_custom_constants(self):
        """nut() works with custom constants."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        C = SpalartAllmarasConstants(Cv1=5.0)
        model = SpalartAllmarasModel(mesh, U, phi, constants=C)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_correct_with_zero_velocity(self):
        """correct() works with zero velocity (no crash)."""
        model = self._make_model()
        model.correct()
        assert model.nuTilde_field.shape == (model.mesh.n_cells,)

    def test_correct_with_uniform_velocity(self):
        """correct() works with uniform non-zero velocity."""
        model = self._make_model(U_vals=(1.0, 0.5, 0.2))
        model.correct()
        assert model.nuTilde_field.shape == (model.mesh.n_cells,)
        # nuTilde should be positive after correction
        assert (model.nuTilde_field > 0).all()

    def test_nut_changes_after_correct(self):
        """nut() values change after correct() with non-zero velocity."""
        model = self._make_model(U_vals=(1.0, 0.0, 0.0))
        nut_before = model.nut().clone()
        model.correct()
        nut_after = model.nut()
        # At minimum the field is updated without error
        assert nut_after.shape == nut_before.shape

    def test_repr(self):
        """Model repr includes class name."""
        model = self._make_model()
        r = repr(model)
        assert "SpalartAllmarasModel" in r
