"""Tests for k-ω turbulence model.

Tests cover:
- Model creation and RTS registration
- Turbulent viscosity computation
- Transport equation solving
- Constants configuration
"""

import pytest
import torch

from pyfoam.turbulence.turbulence_model import TurbulenceModel
from pyfoam.turbulence.k_omega import KOmegaModel, KOmegaConstants

from tests.unit.turbulence.conftest import make_fv_mesh


class TestKOmegaRegistration:
    """Tests for RTS registration of k-ω model."""

    def test_k_omega_registered(self):
        """kOmega is registered in the RTS registry."""
        assert "kOmega" in TurbulenceModel.available_types()

    def test_create_k_omega(self):
        """Can create kOmega model via factory."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = TurbulenceModel.create("kOmega", mesh, U, phi)
        assert isinstance(model, KOmegaModel)


class TestKOmegaConstants:
    """Tests for k-ω model constants."""

    def test_default_constants(self):
        """Default constants match Wilcox (2006) values."""
        C = KOmegaConstants()
        assert C.alpha == pytest.approx(5.0 / 9.0)
        assert C.beta == pytest.approx(3.0 / 40.0)
        assert C.beta_star == pytest.approx(9.0 / 100.0)
        assert C.sigma == 0.5
        assert C.sigma_star == 0.5

    def test_custom_constants(self):
        """Can create custom constants."""
        C = KOmegaConstants(alpha=0.5, beta=0.075)
        assert C.alpha == 0.5
        assert C.beta == 0.075

    def test_constants_frozen(self):
        """Constants are immutable (frozen dataclass)."""
        C = KOmegaConstants()
        with pytest.raises(AttributeError):
            C.alpha = 0.5


class TestKOmegaModel:
    """Tests for k-ω model behaviour."""

    def test_model_creation(self):
        """Model can be created with mesh, U, phi."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaModel(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        """nut() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaModel(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        """nut() returns non-negative values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaModel(mesh, U, phi)
        nut = model.nut()
        assert (nut >= 0).all()

    def test_k_shape(self):
        """k() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaModel(mesh, U, phi)
        k = model.k()
        assert k.shape == (mesh.n_cells,)

    def test_k_positive(self):
        """k() returns positive values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaModel(mesh, U, phi)
        k = model.k()
        assert (k > 0).all()

    def test_omega_shape(self):
        """omega() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaModel(mesh, U, phi)
        omega = model.omega()
        assert omega.shape == (mesh.n_cells,)

    def test_omega_positive(self):
        """omega() returns positive values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaModel(mesh, U, phi)
        omega = model.omega()
        assert (omega > 0).all()

    def test_epsilon_shape(self):
        """epsilon() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaModel(mesh, U, phi)
        eps = model.epsilon()
        assert eps.shape == (mesh.n_cells,)

    def test_correct_updates_fields(self):
        """correct() updates k and omega fields."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaModel(mesh, U, phi)
        model.correct()

        assert model.k().shape == (mesh.n_cells,)
        assert model.omega().shape == (mesh.n_cells,)

    def test_correct_with_velocity(self):
        """correct() works with non-zero velocity."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaModel(mesh, U, phi)
        model.correct()

        assert model.k().shape == (mesh.n_cells,)
        assert model.omega().shape == (mesh.n_cells,)

    def test_custom_constants(self):
        """Model accepts custom constants."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        C = KOmegaConstants(alpha=0.5)
        model = KOmegaModel(mesh, U, phi, constants=C)
        assert model._C.alpha == 0.5

    def test_repr(self):
        """Model repr includes class name."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaModel(mesh, U, phi)
        r = repr(model)
        assert "KOmegaModel" in r
