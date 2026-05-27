"""Tests for Wilcox 2006 k-ω turbulence model (kOmega2006).

Tests cover:
- Model creation and RTS registration
- Turbulent viscosity computation
- Cross-diffusion term
- Low-Re β* correction
- Transport equation solving
- Constants configuration
"""

import pytest
import torch

from pyfoam.turbulence.turbulence_model import TurbulenceModel
from pyfoam.turbulence.k_omega_2006 import KOmega2006Model, KOmega2006Constants

from tests.unit.turbulence.conftest import make_fv_mesh


class TestKOmega2006Registration:
    """Tests for RTS registration of kOmega2006 model."""

    def test_komega2006_registered(self):
        """kOmega2006 is registered in the RTS registry."""
        assert "kOmega2006" in TurbulenceModel.available_types()

    def test_create_komega2006(self):
        """Can create kOmega2006 model via factory."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = TurbulenceModel.create("kOmega2006", mesh, U, phi)
        assert isinstance(model, KOmega2006Model)


class TestKOmega2006Constants:
    """Tests for kOmega2006 model constants."""

    def test_default_constants(self):
        """Default constants match Wilcox (2006) values."""
        C = KOmega2006Constants()
        assert C.alpha == pytest.approx(5.0 / 9.0)
        assert C.beta == pytest.approx(3.0 / 40.0)
        assert C.beta_star_0 == pytest.approx(9.0 / 100.0)
        assert C.sigma == 0.5
        assert C.sigma_star == 0.5
        assert C.sigma_d == pytest.approx(1.0 / 8.0)

    def test_custom_constants(self):
        """Can create custom constants."""
        C = KOmega2006Constants(alpha=0.5, beta=0.075, sigma_d=0.2)
        assert C.alpha == 0.5
        assert C.beta == 0.075
        assert C.sigma_d == 0.2

    def test_constants_frozen(self):
        """Constants are immutable (frozen dataclass)."""
        C = KOmega2006Constants()
        with pytest.raises(AttributeError):
            C.alpha = 0.5


class TestKOmega2006Model:
    """Tests for kOmega2006 model behaviour."""

    def test_model_creation(self):
        """Model can be created with mesh, U, phi."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmega2006Model(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        """nut() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmega2006Model(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        """nut() returns non-negative values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmega2006Model(mesh, U, phi)
        nut = model.nut()
        assert (nut >= 0).all()

    def test_k_shape(self):
        """k() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmega2006Model(mesh, U, phi)
        k = model.k()
        assert k.shape == (mesh.n_cells,)

    def test_k_positive(self):
        """k() returns positive values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmega2006Model(mesh, U, phi)
        k = model.k()
        assert (k > 0).all()

    def test_omega_shape(self):
        """omega() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmega2006Model(mesh, U, phi)
        omega = model.omega()
        assert omega.shape == (mesh.n_cells,)

    def test_omega_positive(self):
        """omega() returns positive values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmega2006Model(mesh, U, phi)
        omega = model.omega()
        assert (omega > 0).all()

    def test_epsilon_shape(self):
        """epsilon() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmega2006Model(mesh, U, phi)
        eps = model.epsilon()
        assert eps.shape == (mesh.n_cells,)

    def test_correct_updates_fields(self):
        """correct() updates k and omega fields."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmega2006Model(mesh, U, phi)
        model.correct()

        assert model.k().shape == (mesh.n_cells,)
        assert model.omega().shape == (mesh.n_cells,)

    def test_correct_with_velocity(self):
        """correct() works with non-zero velocity."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmega2006Model(mesh, U, phi)
        model.correct()

        assert model.k().shape == (mesh.n_cells,)
        assert model.omega().shape == (mesh.n_cells,)

    def test_custom_constants(self):
        """Model accepts custom constants."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        C = KOmega2006Constants(alpha=0.5, sigma_d=0.2)
        model = KOmega2006Model(mesh, U, phi, constants=C)
        assert model._C.alpha == 0.5
        assert model._C.sigma_d == 0.2

    def test_repr(self):
        """Model repr includes class name."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmega2006Model(mesh, U, phi)
        r = repr(model)
        assert "KOmega2006Model" in r


class TestKOmega2006BetaStarCorrection:
    """Tests for the low-Re β* correction."""

    def test_beta_star_shape(self):
        """_beta_star_correction returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmega2006Model(mesh, U, phi)
        beta_star = model._beta_star_correction()
        assert beta_star.shape == (mesh.n_cells,)

    def test_beta_star_positive(self):
        """β* correction is always positive."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmega2006Model(mesh, U, phi)
        beta_star = model._beta_star_correction()
        assert (beta_star > 0).all()

    def test_beta_star_at_high_re(self):
        """At high Re_t, β* should approach β*₀."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmega2006Model(mesh, U, phi)

        # Set high k and low omega to get high Re_t
        model._k = torch.full((mesh.n_cells,), 10.0, dtype=torch.float64)
        model._omega = torch.full((mesh.n_cells,), 0.01, dtype=torch.float64)

        # Need grad_U for vorticity computation in beta_star correction
        # Compute a simple grad_U
        grad_U = torch.zeros(mesh.n_cells, 3, 3, dtype=torch.float64)
        grad_U[:, 0, 1] = 1.0  # some shear
        model._grad_U = grad_U

        beta_star = model._beta_star_correction()
        # At high Re_t with non-trivial vorticity, f_beta_star -> 1
        # so beta_star -> beta_star_0
        C = KOmega2006Constants()
        # Values should be close to beta_star_0
        assert torch.allclose(
            beta_star, torch.full_like(beta_star, C.beta_star_0), atol=1e-6
        )


class TestKOmega2006CrossDiffusion:
    """Tests for the cross-diffusion term."""

    def test_cross_diffusion_shape(self):
        """Cross-diffusion returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmega2006Model(mesh, U, phi)
        k_safe = model._k.clamp(min=1e-16)
        omega_safe = model._omega.clamp(min=1e-16)
        cd = model._cross_diffusion(k_safe, omega_safe)
        assert cd.shape == (mesh.n_cells,)

    def test_cross_diffusion_non_negative(self):
        """Cross-diffusion is non-negative (uses max(..., 0))."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmega2006Model(mesh, U, phi)
        k_safe = model._k.clamp(min=1e-16)
        omega_safe = model._omega.clamp(min=1e-16)
        cd = model._cross_diffusion(k_safe, omega_safe)
        assert (cd >= 0).all()


class TestKOmega2006VsKOmega:
    """Tests that kOmega2006 differs from standard kOmega."""

    def test_different_model_class(self):
        """kOmega2006 is a distinct model from kOmega."""
        from pyfoam.turbulence.k_omega import KOmegaModel

        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model_2006 = TurbulenceModel.create("kOmega2006", mesh, U, phi)
        model_std = TurbulenceModel.create("kOmega", mesh, U, phi)

        assert type(model_2006) is not type(model_std)
        assert isinstance(model_2006, KOmega2006Model)
        assert isinstance(model_std, KOmegaModel)

    def test_different_default_constants(self):
        """kOmega2006 has sigma_d constant not present in standard kOmega."""
        C_2006 = KOmega2006Constants()
        from pyfoam.turbulence.k_omega import KOmegaConstants

        C_std = KOmegaConstants()

        assert C_2006.alpha == pytest.approx(C_std.alpha)
        assert C_2006.beta == pytest.approx(C_std.beta)
        assert hasattr(C_2006, "sigma_d")
        assert not hasattr(C_std, "sigma_d")
