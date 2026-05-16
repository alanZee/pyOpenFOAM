"""Tests for RNG k-ε turbulence model.

Tests cover:
- Model creation and RTS registration
- R-term computation
- Transport equation solving
"""

import pytest
import torch

from pyfoam.turbulence.turbulence_model import TurbulenceModel
from pyfoam.turbulence.rng_k_epsilon import RNGkEpsilonModel, RNGkEpsilonConstants

from tests.unit.turbulence.conftest import make_fv_mesh


class TestRNGkEpsilonRegistration:
    """Tests for RTS registration."""

    def test_registered(self):
        """RNGkEpsilon is registered."""
        assert "RNGkEpsilon" in TurbulenceModel.available_types()

    def test_create_model(self):
        """Can create model via factory."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = TurbulenceModel.create("RNGkEpsilon", mesh, U, phi)
        assert isinstance(model, RNGkEpsilonModel)


class TestRNGkEpsilonConstants:
    """Tests for model constants."""

    def test_default_constants(self):
        """Default constants match RNG values."""
        C = RNGkEpsilonConstants()
        assert C.C_mu == pytest.approx(0.0845)
        assert C.C1 == pytest.approx(1.42)
        assert C.C2 == pytest.approx(1.68)


class TestRNGkEpsilonModel:
    """Tests for model behaviour."""

    def test_model_creation(self):
        """Model can be created."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = RNGkEpsilonModel(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        """nut() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = RNGkEpsilonModel(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        """nut() returns non-negative values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = RNGkEpsilonModel(mesh, U, phi)
        nut = model.nut()
        assert (nut >= 0).all()

    def test_correct_updates_fields(self):
        """correct() updates fields."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = RNGkEpsilonModel(mesh, U, phi)
        model.correct()

        assert model.k().shape == (mesh.n_cells,)
        assert model.epsilon().shape == (mesh.n_cells,)

    def test_r_term_nonnegative(self):
        """R term is non-negative (clamped)."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0  # Need non-zero velocity for gradient computation
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = RNGkEpsilonModel(mesh, U, phi)
        # Must call correct() first to compute gradients
        model.correct()
        P_k = torch.ones(mesh.n_cells, dtype=torch.float64)
        R = model._R_term(P_k)
        assert (R >= 0).all()
