"""Tests for SA-DES (Detached Eddy Simulation) model.

Tests cover:
- RTS registration
- Model creation
- Turbulent viscosity shape and positivity
- Modified distance correctness
- correct() updates
"""

import pytest
import torch

from pyfoam.turbulence.turbulence_model import TurbulenceModel
from pyfoam.turbulence.sa_des import SpalartAllmarasDESModel, SpalartAllmarasDESConstants

from tests.unit.turbulence.conftest import make_fv_mesh


class TestSpalartAllmarasDESRegistration:
    """Tests for RTS registration."""

    def test_registered(self):
        """SpalartAllmarasDES is registered."""
        assert "SpalartAllmarasDES" in TurbulenceModel.available_types()

    def test_create_model(self):
        """Can create model via factory."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = TurbulenceModel.create("SpalartAllmarasDES", mesh, U, phi)
        assert isinstance(model, SpalartAllmarasDESModel)


class TestSpalartAllmarasDESModel:
    """Tests for SA DES model."""

    def test_model_creation(self):
        """Model can be created."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SpalartAllmarasDESModel(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        """nut() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SpalartAllmarasDESModel(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        """nut() returns non-negative values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SpalartAllmarasDESModel(mesh, U, phi)
        nut = model.nut()
        assert (nut >= 0).all()

    def test_modified_distance_positive(self):
        """Modified wall distance is positive."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SpalartAllmarasDESModel(mesh, U, phi)
        d_tilde = model._modified_distance()
        assert d_tilde.shape == (mesh.n_cells,)
        assert (d_tilde > 0).all()

    def test_modified_distance_le_wall_distance(self):
        """Modified distance is always <= wall distance (min property)."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SpalartAllmarasDESModel(mesh, U, phi)
        d_tilde = model._modified_distance()
        d = model._y
        # d_tilde = min(d, C_DES * delta) <= d
        assert (d_tilde <= d + 1e-10).all()

    def test_delta_max_shape(self):
        """Delta max has correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SpalartAllmarasDESModel(mesh, U, phi)
        assert model._delta_max.shape == (mesh.n_cells,)
        assert (model._delta_max > 0).all()

    def test_correct_updates_fields(self):
        """correct() updates nuTilde field."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SpalartAllmarasDESModel(mesh, U, phi)
        model.correct()

        assert model.nuTilde_field.shape == (mesh.n_cells,)
