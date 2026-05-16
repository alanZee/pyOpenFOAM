"""Tests for DES (Detached Eddy Simulation) models.

Tests cover:
- k-ω SST DES model
- Spalart-Allmaras DDES model
"""

import pytest
import torch

from pyfoam.turbulence.turbulence_model import TurbulenceModel
from pyfoam.turbulence.k_omega_sst_des import KOmegaSSTDESModel, KOmegaSSTDESConstants
from pyfoam.turbulence.sa_ddes import SpalartAllmarasDDESModel, SpalartAllmarasDDESConstants

from tests.unit.turbulence.conftest import make_fv_mesh


class TestKOmegaSSTDESRegistration:
    """Tests for RTS registration."""

    def test_registered(self):
        """kOmegaSSTDES is registered."""
        assert "kOmegaSSTDES" in TurbulenceModel.available_types()

    def test_create_model(self):
        """Can create model via factory."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = TurbulenceModel.create("kOmegaSSTDES", mesh, U, phi)
        assert isinstance(model, KOmegaSSTDESModel)


class TestKOmegaSSTDESModel:
    """Tests for k-ω SST DES model."""

    def test_model_creation(self):
        """Model can be created."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTDESModel(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        """nut() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTDESModel(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        """nut() returns non-negative values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTDESModel(mesh, U, phi)
        nut = model.nut()
        assert (nut >= 0).all()

    def test_correct_updates_fields(self):
        """correct() updates fields."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTDESModel(mesh, U, phi)
        model.correct()

        assert model.k().shape == (mesh.n_cells,)
        assert model.omega().shape == (mesh.n_cells,)


class TestSpalartAllmarasDDESRegistration:
    """Tests for RTS registration."""

    def test_registered(self):
        """SpalartAllmarasDDES is registered."""
        assert "SpalartAllmarasDDES" in TurbulenceModel.available_types()

    def test_create_model(self):
        """Can create model via factory."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = TurbulenceModel.create("SpalartAllmarasDDES", mesh, U, phi)
        assert isinstance(model, SpalartAllmarasDDESModel)


class TestSpalartAllmarasDDESModel:
    """Tests for SA DDES model."""

    def test_model_creation(self):
        """Model can be created."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SpalartAllmarasDDESModel(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        """nut() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SpalartAllmarasDDESModel(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        """nut() returns non-negative values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SpalartAllmarasDDESModel(mesh, U, phi)
        nut = model.nut()
        assert (nut >= 0).all()

    def test_delay_function_range(self):
        """Delay function f_d is in [0, 1]."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SpalartAllmarasDDESModel(mesh, U, phi)
        f_d = model._f_d()
        assert f_d.shape == (mesh.n_cells,)
        assert (f_d >= 0).all()
        assert (f_d <= 1).all()

    def test_modified_distance_positive(self):
        """Modified wall distance is positive."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SpalartAllmarasDDESModel(mesh, U, phi)
        d_tilde = model._modified_distance()
        assert d_tilde.shape == (mesh.n_cells,)
        assert (d_tilde > 0).all()

    def test_correct_updates_fields(self):
        """correct() updates fields."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SpalartAllmarasDDESModel(mesh, U, phi)
        model.correct()

        assert model.nuTilde_field.shape == (mesh.n_cells,)
