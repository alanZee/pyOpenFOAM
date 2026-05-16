"""Tests for Launder-Sharma low-Re k-ε turbulence model.

Tests cover:
- Model creation and RTS registration
- Wall-damping functions
- Transport equation solving
"""

import pytest
import torch

from pyfoam.turbulence.turbulence_model import TurbulenceModel
from pyfoam.turbulence.launder_sharma_ke import (
    LaunderSharmaKEModel,
    LaunderSharmaKEConstants,
)

from tests.unit.turbulence.conftest import make_fv_mesh


class TestLaunderSharmaKERegistration:
    """Tests for RTS registration."""

    def test_registered(self):
        """LaunderSharmaKE is registered."""
        assert "LaunderSharmaKE" in TurbulenceModel.available_types()

    def test_create_model(self):
        """Can create model via factory."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = TurbulenceModel.create("LaunderSharmaKE", mesh, U, phi)
        assert isinstance(model, LaunderSharmaKEModel)


class TestLaunderSharmaKEConstants:
    """Tests for model constants."""

    def test_default_constants(self):
        """Default constants match standard values."""
        C = LaunderSharmaKEConstants()
        assert C.C_mu == 0.09
        assert C.C1 == 1.44
        assert C.C2 == 1.92
        assert C.sigma_k == 1.0
        assert C.sigma_eps == 1.3


class TestLaunderSharmaKEModel:
    """Tests for model behaviour."""

    def test_model_creation(self):
        """Model can be created."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LaunderSharmaKEModel(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        """nut() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LaunderSharmaKEModel(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        """nut() returns non-negative values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LaunderSharmaKEModel(mesh, U, phi)
        nut = model.nut()
        assert (nut >= 0).all()

    def test_k_positive(self):
        """k() returns positive values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LaunderSharmaKEModel(mesh, U, phi)
        assert (model.k() > 0).all()

    def test_epsilon_positive(self):
        """epsilon() returns positive values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LaunderSharmaKEModel(mesh, U, phi)
        assert (model.epsilon() > 0).all()

    def test_correct_updates_fields(self):
        """correct() updates fields."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LaunderSharmaKEModel(mesh, U, phi)
        model.correct()

        assert model.k().shape == (mesh.n_cells,)
        assert model.epsilon().shape == (mesh.n_cells,)

    def test_damping_functions(self):
        """Wall-damping functions return correct shapes."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LaunderSharmaKEModel(mesh, U, phi)
        assert model._f_mu().shape == (mesh.n_cells,)
        assert model._f_1().shape == (mesh.n_cells,)
        assert model._f_2().shape == (mesh.n_cells,)
