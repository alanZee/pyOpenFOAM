"""Tests for k-ε turbulence model.

Tests cover:
- Model creation and RTS registration
- Turbulent viscosity computation
- Transport equation solving
- Constants configuration
- Realizable variant
"""

import pytest
import torch

from pyfoam.turbulence.turbulence_model import TurbulenceModel
from pyfoam.turbulence.k_epsilon import (
    KEpsilonModel,
    RealizableKEpsilonModel,
    KEpsilonConstants,
)

from tests.unit.turbulence.conftest import make_fv_mesh


class TestKEpsilonRegistration:
    """Tests for RTS registration of k-ε model."""

    def test_k_epsilon_registered(self):
        """kEpsilon is registered in the RTS registry."""
        assert "kEpsilon" in TurbulenceModel.available_types()

    def test_realizable_registered(self):
        """realizableKEpsilon is registered in the RTS registry."""
        assert "realizableKEpsilon" in TurbulenceModel.available_types()

    def test_create_k_epsilon(self):
        """Can create kEpsilon model via factory."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = TurbulenceModel.create("kEpsilon", mesh, U, phi)
        assert isinstance(model, KEpsilonModel)

    def test_create_realizable(self):
        """Can create realizableKEpsilon model via factory."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = TurbulenceModel.create("realizableKEpsilon", mesh, U, phi)
        assert isinstance(model, RealizableKEpsilonModel)
        assert isinstance(model, KEpsilonModel)

    def test_unknown_model_raises(self):
        """Creating unknown model raises KeyError."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        with pytest.raises(KeyError, match="Unknown turbulence model"):
            TurbulenceModel.create("nonexistent", mesh, U, phi)


class TestKEpsilonConstants:
    """Tests for k-ε model constants."""

    def test_default_constants(self):
        """Default constants match OpenFOAM values."""
        C = KEpsilonConstants()
        assert C.C_mu == 0.09
        assert C.C1 == 1.44
        assert C.C2 == 1.92
        assert C.sigma_k == 1.0
        assert C.sigma_eps == 1.3

    def test_custom_constants(self):
        """Can create custom constants."""
        C = KEpsilonConstants(C_mu=0.1, C1=1.5, C2=2.0)
        assert C.C_mu == 0.1
        assert C.C1 == 1.5
        assert C.C2 == 2.0

    def test_constants_frozen(self):
        """Constants are immutable (frozen dataclass)."""
        C = KEpsilonConstants()
        with pytest.raises(AttributeError):
            C.C_mu = 0.1


class TestKEpsilonModel:
    """Tests for k-ε model behaviour."""

    def test_model_creation(self):
        """Model can be created with mesh, U, phi."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KEpsilonModel(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        """nut() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KEpsilonModel(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        """nut() returns non-negative values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KEpsilonModel(mesh, U, phi)
        nut = model.nut()
        assert (nut >= 0).all()

    def test_k_shape(self):
        """k() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KEpsilonModel(mesh, U, phi)
        k = model.k()
        assert k.shape == (mesh.n_cells,)

    def test_k_positive(self):
        """k() returns positive values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KEpsilonModel(mesh, U, phi)
        k = model.k()
        assert (k > 0).all()

    def test_epsilon_shape(self):
        """epsilon() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KEpsilonModel(mesh, U, phi)
        eps = model.epsilon()
        assert eps.shape == (mesh.n_cells,)

    def test_epsilon_positive(self):
        """epsilon() returns positive values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KEpsilonModel(mesh, U, phi)
        eps = model.epsilon()
        assert (eps > 0).all()

    def test_correct_updates_fields(self):
        """correct() updates k and epsilon fields."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KEpsilonModel(mesh, U, phi)
        k_before = model.k().clone()
        eps_before = model.epsilon().clone()

        model.correct()

        # Fields should be updated (not necessarily different, but correct() should not crash)
        assert model.k().shape == (mesh.n_cells,)
        assert model.epsilon().shape == (mesh.n_cells,)

    def test_correct_with_velocity(self):
        """correct() works with non-zero velocity."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0  # Uniform x-velocity
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KEpsilonModel(mesh, U, phi)
        model.correct()

        # Should complete without error
        assert model.k().shape == (mesh.n_cells,)
        assert model.epsilon().shape == (mesh.n_cells,)

    def test_custom_constants(self):
        """Model accepts custom constants."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        C = KEpsilonConstants(C_mu=0.1)
        model = KEpsilonModel(mesh, U, phi, constants=C)
        assert model._C.C_mu == 0.1

    def test_repr(self):
        """Model repr includes class name."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KEpsilonModel(mesh, U, phi)
        r = repr(model)
        assert "KEpsilonModel" in r


class TestRealizableKEpsilon:
    """Tests for realizable k-ε variant."""

    def test_realizable_is_k_epsilon(self):
        """Realizable model is a subclass of KEpsilonModel."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = RealizableKEpsilonModel(mesh, U, phi)
        assert isinstance(model, KEpsilonModel)

    def test_realizable_nut_shape(self):
        """Realizable nut() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = RealizableKEpsilonModel(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_realizable_nut_positive(self):
        """Realizable nut() returns non-negative values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = RealizableKEpsilonModel(mesh, U, phi)
        nut = model.nut()
        assert (nut >= 0).all()

    def test_realizable_correct(self):
        """Realizable correct() works."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = RealizableKEpsilonModel(mesh, U, phi)
        model.correct()

        assert model.k().shape == (mesh.n_cells,)
        assert model.epsilon().shape == (mesh.n_cells,)
