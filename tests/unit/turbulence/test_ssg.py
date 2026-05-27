"""Tests for SSG Reynolds stress model.

Tests cover:
- Model creation and RTS registration
- Reynolds stress tensor initialization and shape
- Turbulent viscosity computation
- k computation from trace of R
- Transport equation solving
- Anisotropy tensor computation
- Pressure-strain correlation
- Constants configuration
"""

import pytest
import torch

from pyfoam.turbulence.turbulence_model import TurbulenceModel
from pyfoam.turbulence.ssg import SSGModel, SSGConstants

from tests.unit.turbulence.conftest import make_fv_mesh


class TestSSGRegistration:
    """Tests for RTS registration of SSG model."""

    def test_ssg_registered(self):
        """SSG is registered in the RTS registry."""
        assert "SSG" in TurbulenceModel.available_types()

    def test_create_ssg(self):
        """Can create SSG model via factory."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = TurbulenceModel.create("SSG", mesh, U, phi)
        assert isinstance(model, SSGModel)


class TestSSGConstants:
    """Tests for SSG model constants."""

    def test_default_constants(self):
        """Default constants match Speziale-Sarkar-Gatski 1991 values."""
        C = SSGConstants()
        assert C.C1 == 3.4
        assert C.C1star == 1.8
        assert C.C2 == 4.2
        assert C.C3 == 0.8
        assert C.C4 == 1.2
        assert C.C5 == 0.4
        assert C.Ceps1 == 1.44
        assert C.Ceps2 == 1.06
        assert C.Cmu == 0.09
        assert C.Cs == 0.22

    def test_custom_constants(self):
        """Can create custom constants."""
        C = SSGConstants(C1=3.0, C1star=1.5, C2=4.0)
        assert C.C1 == 3.0
        assert C.C1star == 1.5
        assert C.C2 == 4.0

    def test_constants_frozen(self):
        """Constants are immutable (frozen dataclass)."""
        C = SSGConstants()
        with pytest.raises(AttributeError):
            C.C1 = 5.0


class TestSSGModel:
    """Tests for SSG model behaviour."""

    def test_model_creation(self):
        """Model can be created with mesh, U, phi."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SSGModel(mesh, U, phi)
        assert model.mesh is mesh

    def test_R_field_shape(self):
        """R_field has correct shape (n_cells, 3, 3)."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SSGModel(mesh, U, phi)
        R = model.R_field
        assert R.shape == (mesh.n_cells, 3, 3)

    def test_R_field_initially_isotropic(self):
        """R_field is initially isotropic: R_ii = 2/3 k_0."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SSGModel(mesh, U, phi)
        R = model.R_field
        k0 = 1e-4
        expected_diag = 2.0 / 3.0 * k0

        # Diagonal components equal
        assert torch.allclose(R[:, 0, 0], R[:, 1, 1], atol=1e-12)
        assert torch.allclose(R[:, 1, 1], R[:, 2, 2], atol=1e-12)

        # Value check
        assert torch.allclose(
            R[:, 0, 0],
            torch.full((mesh.n_cells,), expected_diag, dtype=torch.float64),
            atol=1e-12,
        )

        # Off-diagonal zero
        assert torch.allclose(R[:, 0, 1], torch.zeros(mesh.n_cells, dtype=torch.float64), atol=1e-12)
        assert torch.allclose(R[:, 0, 2], torch.zeros(mesh.n_cells, dtype=torch.float64), atol=1e-12)
        assert torch.allclose(R[:, 1, 2], torch.zeros(mesh.n_cells, dtype=torch.float64), atol=1e-12)

    def test_R_field_symmetric(self):
        """R_field is symmetric: R_ij = R_ji."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SSGModel(mesh, U, phi)
        R = model.R_field
        assert torch.allclose(R, R.transpose(-1, -2), atol=1e-12)

    def test_nut_shape(self):
        """nut() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SSGModel(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        """nut() returns non-negative values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SSGModel(mesh, U, phi)
        nut = model.nut()
        assert (nut >= 0).all()

    def test_k_shape(self):
        """k() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SSGModel(mesh, U, phi)
        k = model.k()
        assert k.shape == (mesh.n_cells,)

    def test_k_is_half_trace(self):
        """k = 0.5 * trace(R)."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SSGModel(mesh, U, phi)
        R = model.R_field
        k = model.k()
        k_expected = 0.5 * (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2])
        assert torch.allclose(k, k_expected.clamp(min=1e-16), atol=1e-12)

    def test_k_positive(self):
        """k() returns positive values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SSGModel(mesh, U, phi)
        k = model.k()
        assert (k > 0).all()

    def test_epsilon_shape(self):
        """epsilon() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SSGModel(mesh, U, phi)
        eps = model.epsilon()
        assert eps.shape == (mesh.n_cells,)

    def test_epsilon_positive(self):
        """epsilon() returns positive values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SSGModel(mesh, U, phi)
        eps = model.epsilon()
        assert (eps > 0).all()

    def test_devReff_shape(self):
        """devReff() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SSGModel(mesh, U, phi)
        tau = model.devReff()
        assert tau.shape == (mesh.n_cells, 3, 3)

    def test_correct_updates_fields(self):
        """correct() updates R and epsilon fields without error."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SSGModel(mesh, U, phi)
        model.correct()

        assert model.R_field.shape == (mesh.n_cells, 3, 3)
        assert model.epsilon().shape == (mesh.n_cells,)

    def test_correct_with_velocity(self):
        """correct() works with non-zero velocity."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0  # Uniform x-velocity
        U[:, 2] = 0.5  # Some z-velocity
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SSGModel(mesh, U, phi)
        model.correct()

        # Should complete without error
        assert model.R_field.shape == (mesh.n_cells, 3, 3)
        assert model.epsilon().shape == (mesh.n_cells,)

    def test_correct_preserves_symmetry(self):
        """correct() preserves R symmetry."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SSGModel(mesh, U, phi)
        model.correct()

        R = model.R_field
        assert torch.allclose(R, R.transpose(-1, -2), atol=1e-10)

    def test_correct_diagonal_nonnegative(self):
        """correct() ensures diagonal components remain non-negative."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        U[:, 2] = 0.5
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SSGModel(mesh, U, phi)
        model.correct()

        R = model.R_field
        assert (R[:, 0, 0] >= 0).all()
        assert (R[:, 1, 1] >= 0).all()
        assert (R[:, 2, 2] >= 0).all()

    def test_multiple_correct_calls(self):
        """Multiple correct() calls work without error."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SSGModel(mesh, U, phi)
        for _ in range(3):
            model.correct()

        assert model.R_field.shape == (mesh.n_cells, 3, 3)
        assert model.k().shape == (mesh.n_cells,)
        assert model.epsilon().shape == (mesh.n_cells,)

    def test_custom_constants(self):
        """Model accepts custom constants."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        C = SSGConstants(C1=3.0, Cmu=0.1)
        model = SSGModel(mesh, U, phi, constants=C)
        assert model._C.C1 == 3.0
        assert model._C.Cmu == 0.1

    def test_repr(self):
        """Model repr includes class name."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SSGModel(mesh, U, phi)
        r = repr(model)
        assert "SSGModel" in r


class TestSSGAnisotropy:
    """Tests for anisotropy tensor computation."""

    def test_anisotropy_initially_zero(self):
        """Anisotropy b_ij is zero for isotropic initial conditions."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SSGModel(mesh, U, phi)
        # Trigger gradient computation
        from pyfoam.discretisation.operators import fvc

        grad_U = torch.zeros(mesh.n_cells, 3, 3, dtype=torch.float64)
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(U[:, i], "Gauss linear", mesh=mesh)
        model._grad_U = grad_U

        b = model._compute_anisotropy()
        # For isotropic R: b_ij = R_ij/(2k) - delta_ij/3
        # R_ii = 2k/3 => R_ii/(2k) = 1/3, so b_ii = 1/3 - 1/3 = 0
        # Off-diag: 0/(2k) - 0 = 0
        assert torch.allclose(b, torch.zeros_like(b), atol=1e-10)

    def test_anisotropy_trace_zero(self):
        """Anisotropy tensor is traceless: b_ii = 0."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SSGModel(mesh, U, phi)
        model.correct()

        b = model._compute_anisotropy()
        trace = b[:, 0, 0] + b[:, 1, 1] + b[:, 2, 2]
        assert torch.allclose(trace, torch.zeros_like(trace), atol=1e-10)


class TestSSGPressureStrain:
    """Tests for pressure-strain correlation."""

    def test_pressure_strain_shape(self):
        """Pressure-strain tensor has correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SSGModel(mesh, U, phi)
        n_cells = mesh.n_cells

        b = torch.zeros(n_cells, 3, 3, dtype=torch.float64)
        S = torch.zeros(n_cells, 3, 3, dtype=torch.float64)
        W = torch.zeros(n_cells, 3, 3, dtype=torch.float64)
        k = torch.full((n_cells,), 1e-4, dtype=torch.float64)
        eps = torch.full((n_cells,), 1e-4, dtype=torch.float64)

        Phi = model._pressure_strain(b, S, W, k, eps)
        assert Phi.shape == (n_cells, 3, 3)

    def test_pressure_strain_symmetric(self):
        """Pressure-strain tensor is symmetric."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SSGModel(mesh, U, phi)
        n_cells = mesh.n_cells

        b = torch.zeros(n_cells, 3, 3, dtype=torch.float64)
        S = torch.zeros(n_cells, 3, 3, dtype=torch.float64)
        W = torch.zeros(n_cells, 3, 3, dtype=torch.float64)
        k = torch.full((n_cells,), 1e-4, dtype=torch.float64)
        eps = torch.full((n_cells,), 1e-4, dtype=torch.float64)

        Phi = model._pressure_strain(b, S, W, k, eps)
        assert torch.allclose(Phi, Phi.transpose(-1, -2), atol=1e-12)

    def test_pressure_strain_zero_for_zero_anisotropy_and_strain(self):
        """Phi_ij = 0 when b=0 and S=0."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = SSGModel(mesh, U, phi)
        n_cells = mesh.n_cells

        b = torch.zeros(n_cells, 3, 3, dtype=torch.float64)
        S = torch.zeros(n_cells, 3, 3, dtype=torch.float64)
        W = torch.zeros(n_cells, 3, 3, dtype=torch.float64)
        k = torch.full((n_cells,), 1e-4, dtype=torch.float64)
        eps = torch.full((n_cells,), 1e-4, dtype=torch.float64)

        Phi = model._pressure_strain(b, S, W, k, eps)
        assert torch.allclose(Phi, torch.zeros_like(Phi), atol=1e-15)
