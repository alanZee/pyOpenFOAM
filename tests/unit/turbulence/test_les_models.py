"""Tests for LES subgrid-scale models.

Tests cover:
- Dynamic Smagorinsky model
- Lagrangian dynamic model
- One-equation k model
"""

import pytest
import torch

from pyfoam.turbulence.dynamic_smagorinsky import DynamicSmagorinskyModel
from pyfoam.turbulence.dynamic_lagrangian import DynamicLagrangianModel
from pyfoam.turbulence.k_eqn import KEqnModel, KEqnConstants

from tests.unit.turbulence.conftest import make_fv_mesh


class TestDynamicSmagorinsky:
    """Tests for dynamic Smagorinsky model."""

    def test_model_creation(self):
        """Model can be created."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = DynamicSmagorinskyModel(mesh, U, phi)
        assert model._mesh is mesh

    def test_nut_before_correct_raises(self):
        """nut() before correct() raises RuntimeError."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = DynamicSmagorinskyModel(mesh, U, phi)
        with pytest.raises(RuntimeError, match="correct\\(\\) must be called"):
            model.nut()

    def test_correct_and_nut(self):
        """correct() and nut() work together."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = DynamicSmagorinskyModel(mesh, U, phi)
        model.correct()

        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_cs2_clipping(self):
        """Cs2 is clipped to physical range."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = DynamicSmagorinskyModel(mesh, U, phi, Cs_min=0.0, Cs_max=0.5)
        model.correct()

        Cs2 = model.Cs2
        assert Cs2 is not None
        assert (Cs2 >= 0.0).all()
        assert (Cs2 <= 0.5).all()


class TestDynamicLagrangian:
    """Tests for Lagrangian dynamic model."""

    def test_model_creation(self):
        """Model can be created."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = DynamicLagrangianModel(mesh, U, phi)
        assert model._mesh is mesh

    def test_nut_before_correct_raises(self):
        """nut() before correct() raises RuntimeError."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = DynamicLagrangianModel(mesh, U, phi)
        with pytest.raises(RuntimeError, match="correct\\(\\) must be called"):
            model.nut()

    def test_correct_and_nut(self):
        """correct() and nut() work together."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = DynamicLagrangianModel(mesh, U, phi)
        model.correct()

        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()


class TestKEqnModel:
    """Tests for one-equation k SGS model."""

    def test_model_creation(self):
        """Model can be created."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KEqnModel(mesh, U, phi)
        assert model._mesh is mesh

    def test_nut_shape(self):
        """nut() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KEqnModel(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        """nut() returns non-negative values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KEqnModel(mesh, U, phi)
        nut = model.nut()
        assert (nut >= 0).all()

    def test_k_sgs_positive(self):
        """k_sgs returns positive values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KEqnModel(mesh, U, phi)
        assert (model.k_sgs_field > 0).all()

    def test_correct_updates_fields(self):
        """correct() updates fields."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KEqnModel(mesh, U, phi)
        model.correct()

        assert model.k_sgs_field.shape == (mesh.n_cells,)
        assert (model.k_sgs_field > 0).all()

    def test_custom_constants(self):
        """Model accepts custom constants."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        C = KEqnConstants(C_k=0.1)
        model = KEqnModel(mesh, U, phi, constants=C)
        assert model._C.C_k == 0.1
