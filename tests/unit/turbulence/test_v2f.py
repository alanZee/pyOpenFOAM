"""Tests for v²-f turbulence model.

Tests cover:
- Model creation and RTS registration
- Turbulent time and length scales
- Transport equation solving
"""

import pytest
import torch

from pyfoam.turbulence.turbulence_model import TurbulenceModel
from pyfoam.turbulence.v2f import V2FModel, V2FConstants

from tests.unit.turbulence.conftest import make_fv_mesh


class TestV2FRegistration:
    """Tests for RTS registration."""

    def test_registered(self):
        """v2f is registered."""
        assert "v2f" in TurbulenceModel.available_types()

    def test_create_model(self):
        """Can create model via factory."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = TurbulenceModel.create("v2f", mesh, U, phi)
        assert isinstance(model, V2FModel)


class TestV2FConstants:
    """Tests for model constants."""

    def test_default_constants(self):
        """Default constants match Durbin (1995) values."""
        C = V2FConstants()
        assert C.C_mu == 0.22
        assert C.C1 == 1.4
        assert C.C2 == 0.3


class TestV2FModel:
    """Tests for model behaviour."""

    def test_model_creation(self):
        """Model can be created."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = V2FModel(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        """nut() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = V2FModel(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        """nut() returns non-negative values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = V2FModel(mesh, U, phi)
        nut = model.nut()
        assert (nut >= 0).all()

    def test_v2_positive(self):
        """v2_field returns positive values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = V2FModel(mesh, U, phi)
        assert (model.v2_field > 0).all()

    def test_correct_updates_fields(self):
        """correct() updates fields."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = V2FModel(mesh, U, phi)
        model.correct()

        assert model.k().shape == (mesh.n_cells,)
        assert model.epsilon().shape == (mesh.n_cells,)
        assert model.v2_field.shape == (mesh.n_cells,)

    def test_time_scale(self):
        """Time scale returns positive values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = V2FModel(mesh, U, phi)
        T = model._time_scale()
        assert T.shape == (mesh.n_cells,)
        assert (T > 0).all()

    def test_length_scale(self):
        """Length scale returns positive values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = V2FModel(mesh, U, phi)
        L = model._length_scale()
        assert L.shape == (mesh.n_cells,)
        assert (L > 0).all()
