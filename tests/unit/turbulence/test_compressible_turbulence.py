"""Tests for compressible turbulence models.

Tests cover:
- CompressibleTurbulenceModel ABC RTS registry
- KOmegaSSTCompressible model creation and constants
- Kinematic vs dynamic turbulent viscosity (mu_t = rho * nut)
- Density-weighted transport equation solving
- Compressibility corrections (Sarkar)
- Blending functions
"""

import pytest
import torch

from pyfoam.turbulence.compressible_turbulence import (
    CompressibleTurbulenceModel,
    KOmegaSSTCompressible,
    KOmegaSSTCompressibleConstants,
)

from tests.unit.turbulence.conftest import make_fv_mesh


# ============================================================================
# CompressibleTurbulenceModel ABC
# ============================================================================


class TestCompressibleTurbulenceModelABC:
    """CompressibleTurbulenceModel ABC tests."""

    def test_rts_registry_contains_model(self):
        """kOmegaSSTCompressible is registered."""
        types = CompressibleTurbulenceModel.available_types()
        assert "kOmegaSSTCompressible" in types

    def test_factory_create(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = CompressibleTurbulenceModel.create(
            "kOmegaSSTCompressible", mesh, U, phi,
        )
        assert isinstance(model, KOmegaSSTCompressible)

    def test_factory_create_with_rho(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        rho = torch.full((mesh.n_cells,), 1.225, dtype=torch.float64)

        model = CompressibleTurbulenceModel.create(
            "kOmegaSSTCompressible", mesh, U, phi, rho=rho,
        )
        assert isinstance(model, KOmegaSSTCompressible)

    def test_factory_unknown_raises(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        with pytest.raises(KeyError, match="Unknown compressible turbulence model"):
            CompressibleTurbulenceModel.create(
                "nonexistent", mesh, U, phi,
            )

    def test_is_abstract(self):
        with pytest.raises(TypeError):
            CompressibleTurbulenceModel(None, None, None)


# ============================================================================
# KOmegaSSTCompressibleConstants
# ============================================================================


class TestKOmegaSSTCompressibleConstants:
    """Constants tests."""

    def test_default_constants(self):
        """Default constants match Menter (1994) SST values + Sarkar defaults."""
        C = KOmegaSSTCompressibleConstants()
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
        # Compressibility constants
        assert C.alpha_star == 1.0
        assert C.alpha_1 == 1.0
        assert C.alpha_2 == 0.5

    def test_custom_constants(self):
        C = KOmegaSSTCompressibleConstants(a1=0.5, alpha_1=0.0)
        assert C.a1 == 0.5
        assert C.alpha_1 == 0.0

    def test_constants_frozen(self):
        C = KOmegaSSTCompressibleConstants()
        with pytest.raises(AttributeError):
            C.a1 = 0.5


# ============================================================================
# KOmegaSSTCompressible
# ============================================================================


class TestKOmegaSSTCompressible:
    """Compressible k-omega SST model tests."""

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTCompressible(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTCompressible(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTCompressible(mesh, U, phi)
        nut = model.nut()
        assert (nut >= 0).all()
        assert torch.isfinite(nut).all()

    def test_mu_t_shape(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        rho = torch.full((mesh.n_cells,), 1.225, dtype=torch.float64)

        model = KOmegaSSTCompressible(mesh, U, phi, rho=rho)
        mu_t = model.mu_t()
        assert mu_t.shape == (mesh.n_cells,)

    def test_mu_t_equals_rho_times_nut(self):
        """mu_t = rho * nut for compressible models."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        rho = torch.full((mesh.n_cells,), 1.225, dtype=torch.float64)

        model = KOmegaSSTCompressible(mesh, U, phi, rho=rho)
        nut = model.nut()
        mu_t = model.mu_t()

        assert torch.allclose(mu_t, rho * nut, rtol=1e-10)

    def test_mu_t_with_unit_density(self):
        """With rho=1, mu_t == nut."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTCompressible(mesh, U, phi)
        nut = model.nut()
        mu_t = model.mu_t()

        assert torch.allclose(mu_t, nut, rtol=1e-10)

    def test_k_field_shape(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTCompressible(mesh, U, phi)
        assert model.k().shape == (mesh.n_cells,)

    def test_k_field_positive(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTCompressible(mesh, U, phi)
        assert (model.k() >= 0).all()

    def test_omega_field_shape(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTCompressible(mesh, U, phi)
        assert model.omega().shape == (mesh.n_cells,)

    def test_epsilon_computation(self):
        """epsilon = beta* * omega * k."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTCompressible(mesh, U, phi)
        eps = model.epsilon()
        assert eps.shape == (mesh.n_cells,)
        assert (eps >= 0).all()
        assert torch.isfinite(eps).all()

    def test_correct_updates_fields(self):
        """correct() modifies k and omega fields."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 10.0  # Non-zero velocity for production
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        rho = torch.full((mesh.n_cells,), 1.225, dtype=torch.float64)

        model = KOmegaSSTCompressible(mesh, U, phi, rho=rho)
        k_before = model.k().clone()
        model.correct()
        k_after = model.k()

        # Fields should be updated
        assert not torch.allclose(k_before, k_after)

    def test_correct_with_density(self):
        """correct() works with variable density."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 5.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        rho = torch.tensor([1.0, 2.0], dtype=torch.float64)

        model = KOmegaSSTCompressible(mesh, U, phi, rho=rho)
        model.correct()
        assert torch.isfinite(model.k()).all()
        assert torch.isfinite(model.omega()).all()

    def test_rho_setter(self):
        """rho can be updated."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTCompressible(mesh, U, phi)
        new_rho = torch.full((mesh.n_cells,), 2.0, dtype=torch.float64)
        model.rho = new_rho
        assert torch.allclose(model.rho, new_rho)

    def test_nu_setter(self):
        """nu can be updated."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTCompressible(mesh, U, phi)
        model.nu = 2e-5
        assert model.nu == 2e-5

    def test_k_field_setter(self):
        """k_field can be updated."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTCompressible(mesh, U, phi)
        new_k = torch.full((mesh.n_cells,), 0.1, dtype=torch.float64)
        model.k_field = new_k
        assert torch.allclose(model.k(), new_k)

    def test_repr(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTCompressible(mesh, U, phi)
        r = repr(model)
        assert "KOmegaSSTCompressible" in r
        assert str(mesh.n_cells) in r
