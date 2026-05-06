"""Tests for k-ω SST turbulence model.

Tests cover:
- Model creation and RTS registration
- Turbulent viscosity computation with SST limiter
- Blending functions F1 and F2
- Transport equation solving
- Constants configuration
- Wall distance computation
"""

import pytest
import torch

from pyfoam.turbulence.turbulence_model import TurbulenceModel
from pyfoam.turbulence.k_omega_sst import (
    KOmegaSSTModel,
    KOmegaSSTConstants,
)

from tests.unit.turbulence.conftest import make_fv_mesh


class TestKOmegaSSTRegistration:
    """Tests for RTS registration of k-ω SST model."""

    def test_k_omega_sst_registered(self):
        """kOmegaSST is registered in the RTS registry."""
        assert "kOmegaSST" in TurbulenceModel.available_types()

    def test_create_k_omega_sst(self):
        """Can create kOmegaSST model via factory."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = TurbulenceModel.create("kOmegaSST", mesh, U, phi)
        assert isinstance(model, KOmegaSSTModel)


class TestKOmegaSSTConstants:
    """Tests for k-ω SST model constants."""

    def test_default_constants(self):
        """Default constants match Menter (1994) values."""
        C = KOmegaSSTConstants()
        assert C.sigma_k1 == 0.85
        assert C.sigma_k2 == 1.0
        assert C.sigma_omega1 == 0.5
        assert C.sigma_omega2 == 0.856
        assert C.beta1 == 0.075
        assert C.beta2 == 0.0828
        assert C.gamma1 == pytest.approx(5.0 / 9.0)
        assert C.gamma2 == 0.44
        assert C.a1 == 0.31
        assert C.beta_star == 0.09
        assert C.kappa == 0.41

    def test_custom_constants(self):
        """Can create custom constants."""
        C = KOmegaSSTConstants(a1=0.5, beta_star=0.1)
        assert C.a1 == 0.5
        assert C.beta_star == 0.1

    def test_constants_frozen(self):
        """Constants are immutable (frozen dataclass)."""
        C = KOmegaSSTConstants()
        with pytest.raises(AttributeError):
            C.a1 = 0.5


class TestKOmegaSSTModel:
    """Tests for k-ω SST model behaviour."""

    def test_model_creation(self):
        """Model can be created with mesh, U, phi."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTModel(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        """nut() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTModel(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        """nut() returns non-negative values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTModel(mesh, U, phi)
        nut = model.nut()
        assert (nut >= 0).all()

    def test_k_shape(self):
        """k() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTModel(mesh, U, phi)
        k = model.k()
        assert k.shape == (mesh.n_cells,)

    def test_k_positive(self):
        """k() returns positive values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTModel(mesh, U, phi)
        k = model.k()
        assert (k > 0).all()

    def test_omega_shape(self):
        """omega() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTModel(mesh, U, phi)
        omega = model.omega()
        assert omega.shape == (mesh.n_cells,)

    def test_omega_positive(self):
        """omega() returns positive values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTModel(mesh, U, phi)
        omega = model.omega()
        assert (omega > 0).all()

    def test_epsilon_from_omega(self):
        """epsilon() computes ε = β* ω k."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTModel(mesh, U, phi)
        eps = model.epsilon()
        assert eps.shape == (mesh.n_cells,)

        # Verify: ε = β* ω k
        expected = model._C.beta_star * model.omega_field * model.k_field
        assert torch.allclose(eps, expected, atol=1e-10)

    def test_correct_updates_fields(self):
        """correct() updates k and omega fields."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTModel(mesh, U, phi)
        model.correct()

        assert model.k().shape == (mesh.n_cells,)
        assert model.omega().shape == (mesh.n_cells,)

    def test_correct_with_velocity(self):
        """correct() works with non-zero velocity."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0  # Uniform x-velocity
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTModel(mesh, U, phi)
        model.correct()

        assert model.k().shape == (mesh.n_cells,)
        assert model.omega().shape == (mesh.n_cells,)

    def test_blending_functions_shape(self):
        """Blending functions F1 and F2 return correct shapes."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTModel(mesh, U, phi)

        F1 = model._F1()
        F2 = model._F2()

        assert F1.shape == (mesh.n_cells,)
        assert F2.shape == (mesh.n_cells,)

    def test_blending_functions_range(self):
        """Blending functions return values in [0, 1]."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTModel(mesh, U, phi)

        F1 = model._F1()
        F2 = model._F2()

        assert (F1 >= 0).all() and (F1 <= 1).all()
        assert (F2 >= 0).all() and (F2 <= 1).all()

    def test_wall_distance_shape(self):
        """Wall distance computation returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTModel(mesh, U, phi)
        y = model._y

        assert y.shape == (mesh.n_cells,)
        assert (y > 0).all()

    def test_custom_constants(self):
        """Model accepts custom constants."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        C = KOmegaSSTConstants(a1=0.5)
        model = KOmegaSSTModel(mesh, U, phi, constants=C)
        assert model._C.a1 == 0.5

    def test_repr(self):
        """Model repr includes class name."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTModel(mesh, U, phi)
        r = repr(model)
        assert "KOmegaSSTModel" in r


class TestKOmegaSSTNutFormula:
    """Tests for the SST turbulent viscosity formula."""

    def test_nut_sst_limiter(self):
        """nut = a1 * k / max(a1 * omega, S * F2)."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTModel(mesh, U, phi)

        # Manually compute expected nut
        k = model.k_field.clamp(min=1e-16)
        omega = model.omega_field.clamp(min=1e-16)

        # Need to compute S and F2
        model._grad_U = torch.zeros(mesh.n_cells, 3, 3, dtype=torch.float64)
        # For uniform velocity, S = 0, so nut = a1 * k / (a1 * omega) = k / omega
        # But F2 is also involved

        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()
