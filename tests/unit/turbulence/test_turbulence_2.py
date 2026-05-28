"""Tests for enhanced turbulence models (kOmegaSST2003, realizableKE2).

Tests cover:
- RTS registration and factory creation
- Constants configuration
- Model creation and basic interface
- nut / k / epsilon shapes and positivity
- correct() with and without velocity
"""

import pytest
import torch

from pyfoam.turbulence.turbulence_model import TurbulenceModel
from tests.unit.turbulence.conftest import make_fv_mesh


class TestKOmegaSST2003Registration:
    """Tests for RTS registration of kOmegaSST2003."""

    def test_registered(self):
        assert "kOmegaSST2003" in TurbulenceModel.available_types()

    def test_factory_create(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = TurbulenceModel.create("kOmegaSST2003", mesh, U, phi)
        from pyfoam.turbulence.turbulence_2 import KOmegaSST2003Model
        assert isinstance(model, KOmegaSST2003Model)


class TestKOmegaSST2003Constants:
    """Tests for kOmegaSST2003 constants."""

    def test_default_constants(self):
        from pyfoam.turbulence.turbulence_2 import KOmegaSST2003Constants

        C = KOmegaSST2003Constants()
        assert C.sigma_k1 == 0.85
        assert C.sigma_k2 == 1.0
        assert C.sigma_omega1 == 0.5
        assert C.sigma_omega2 == 0.856
        assert C.beta1 == 0.075
        assert C.beta2 == 0.0828
        assert C.a1 == 0.31
        assert C.beta_star == 0.09
        assert C.c1 == 10.0

    def test_custom_constants(self):
        from pyfoam.turbulence.turbulence_2 import KOmegaSST2003Constants

        C = KOmegaSST2003Constants(a1=0.5, c1=15.0)
        assert C.a1 == 0.5
        assert C.c1 == 15.0

    def test_constants_frozen(self):
        from pyfoam.turbulence.turbulence_2 import KOmegaSST2003Constants

        C = KOmegaSST2003Constants()
        with pytest.raises(AttributeError):
            C.a1 = 0.5


class TestKOmegaSST2003Model:
    """Tests for kOmegaSST2003 model behaviour."""

    def test_model_creation(self):
        from pyfoam.turbulence.turbulence_2 import KOmegaSST2003Model

        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSST2003Model(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        from pyfoam.turbulence.turbulence_2 import KOmegaSST2003Model

        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSST2003Model(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        from pyfoam.turbulence.turbulence_2 import KOmegaSST2003Model

        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSST2003Model(mesh, U, phi)
        assert (model.nut() >= 0).all()

    def test_k_shape(self):
        from pyfoam.turbulence.turbulence_2 import KOmegaSST2003Model

        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSST2003Model(mesh, U, phi)
        assert model.k().shape == (mesh.n_cells,)
        assert (model.k() > 0).all()

    def test_omega_shape(self):
        from pyfoam.turbulence.turbulence_2 import KOmegaSST2003Model

        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSST2003Model(mesh, U, phi)
        assert model.omega().shape == (mesh.n_cells,)
        assert (model.omega() > 0).all()

    def test_epsilon(self):
        from pyfoam.turbulence.turbulence_2 import KOmegaSST2003Model

        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSST2003Model(mesh, U, phi)
        eps = model.epsilon()
        assert eps.shape == (mesh.n_cells,)

        # Verify ε = β* ω k
        expected = model._C.beta_star * model.omega_field * model.k_field
        assert torch.allclose(eps, expected, atol=1e-10)

    def test_correct_updates_fields(self):
        from pyfoam.turbulence.turbulence_2 import KOmegaSST2003Model

        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSST2003Model(mesh, U, phi)
        model.correct()
        assert model.k().shape == (mesh.n_cells,)
        assert model.omega().shape == (mesh.n_cells,)

    def test_correct_with_velocity(self):
        from pyfoam.turbulence.turbulence_2 import KOmegaSST2003Model

        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSST2003Model(mesh, U, phi)
        model.correct()
        assert model.k().shape == (mesh.n_cells,)

    def test_blending_functions_shape(self):
        from pyfoam.turbulence.turbulence_2 import KOmegaSST2003Model

        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSST2003Model(mesh, U, phi)
        assert model._F1().shape == (mesh.n_cells,)
        assert model._F2().shape == (mesh.n_cells,)

    def test_blending_functions_range(self):
        from pyfoam.turbulence.turbulence_2 import KOmegaSST2003Model

        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSST2003Model(mesh, U, phi)
        F1 = model._F1()
        F2 = model._F2()
        assert (F1 >= 0).all() and (F1 <= 1).all()
        assert (F2 >= 0).all() and (F2 <= 1).all()

    def test_repr(self):
        from pyfoam.turbulence.turbulence_2 import KOmegaSST2003Model

        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSST2003Model(mesh, U, phi)
        assert "KOmegaSST2003Model" in repr(model)


class TestRealizableKE2Registration:
    """Tests for RTS registration of realizableKE2."""

    def test_registered(self):
        assert "realizableKE2" in TurbulenceModel.available_types()

    def test_factory_create(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = TurbulenceModel.create("realizableKE2", mesh, U, phi)
        from pyfoam.turbulence.turbulence_2 import RealizableKE2Model
        assert isinstance(model, RealizableKE2Model)


class TestRealizableKE2Constants:
    """Tests for realizableKE2 constants."""

    def test_default_constants(self):
        from pyfoam.turbulence.turbulence_2 import RealizableKE2Constants

        C = RealizableKE2Constants()
        assert C.C_mu_base == 0.09
        assert C.C1 == 1.44
        assert C.C2 == 1.9
        assert C.sigma_k == 1.0
        assert C.sigma_eps == 1.2
        assert C.A0 == 4.0
        assert C.C_lim == 0.43

    def test_custom_constants(self):
        from pyfoam.turbulence.turbulence_2 import RealizableKE2Constants

        C = RealizableKE2Constants(C2=1.8, C_lim=0.5)
        assert C.C2 == 1.8
        assert C.C_lim == 0.5

    def test_constants_frozen(self):
        from pyfoam.turbulence.turbulence_2 import RealizableKE2Constants

        C = RealizableKE2Constants()
        with pytest.raises(AttributeError):
            C.C2 = 2.0


class TestRealizableKE2Model:
    """Tests for realizableKE2 model behaviour."""

    def test_model_creation(self):
        from pyfoam.turbulence.turbulence_2 import RealizableKE2Model

        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = RealizableKE2Model(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        from pyfoam.turbulence.turbulence_2 import RealizableKE2Model

        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = RealizableKE2Model(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        from pyfoam.turbulence.turbulence_2 import RealizableKE2Model

        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = RealizableKE2Model(mesh, U, phi)
        assert (model.nut() >= 0).all()

    def test_k_shape(self):
        from pyfoam.turbulence.turbulence_2 import RealizableKE2Model

        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = RealizableKE2Model(mesh, U, phi)
        assert model.k().shape == (mesh.n_cells,)
        assert (model.k() > 0).all()

    def test_epsilon_shape(self):
        from pyfoam.turbulence.turbulence_2 import RealizableKE2Model

        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = RealizableKE2Model(mesh, U, phi)
        assert model.epsilon().shape == (mesh.n_cells,)
        assert (model.epsilon() > 0).all()

    def test_correct_updates_fields(self):
        from pyfoam.turbulence.turbulence_2 import RealizableKE2Model

        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = RealizableKE2Model(mesh, U, phi)
        model.correct()
        assert model.k().shape == (mesh.n_cells,)
        assert model.epsilon().shape == (mesh.n_cells,)

    def test_correct_with_velocity(self):
        from pyfoam.turbulence.turbulence_2 import RealizableKE2Model

        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = RealizableKE2Model(mesh, U, phi)
        model.correct()
        assert model.k().shape == (mesh.n_cells,)

    def test_dynamic_C_mu_range(self):
        """Dynamic C_mu should be in a reasonable range."""
        from pyfoam.turbulence.turbulence_2 import RealizableKE2Model

        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = RealizableKE2Model(mesh, U, phi)
        model.correct()  # compute gradients

        C_mu = model._compute_C_mu()
        assert C_mu.shape == (mesh.n_cells,)
        assert (C_mu > 0).all()
        assert (C_mu <= 0.5).all()

    def test_repr(self):
        from pyfoam.turbulence.turbulence_2 import RealizableKE2Model

        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = RealizableKE2Model(mesh, U, phi)
        assert "RealizableKE2Model" in repr(model)
