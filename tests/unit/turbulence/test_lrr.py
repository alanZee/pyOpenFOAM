"""Tests for LRR Reynolds stress transport model.

Tests cover:
- Model creation and RTS registration
- Reynolds stress tensor properties
- Turbulent viscosity computation (nut = Cmu * k^2 / eps)
- Transport equation solving (correct())
- Pressure-strain correlation (linear return-to-isotropy)
- Constants configuration
- Turbulent kinetic energy from trace(R_ij)
"""

import pytest
import torch

from pyfoam.turbulence.turbulence_model import TurbulenceModel
from pyfoam.turbulence.lrr import LRRModel, LRRConstants

from tests.unit.turbulence.conftest import make_fv_mesh


class TestLRRRegistration:
    """Tests for RTS registration of LRR model."""

    def test_lrr_registered(self):
        """LRR is registered in the RTS registry."""
        assert "LRR" in TurbulenceModel.available_types()

    def test_create_lrr(self):
        """Can create LRR model via factory."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = TurbulenceModel.create("LRR", mesh, U, phi)
        assert isinstance(model, LRRModel)


class TestLRRConstants:
    """Tests for LRR model constants."""

    def test_default_constants(self):
        """Default constants match Launder-Reece-Rodi 1975 values."""
        C = LRRConstants()
        assert C.C1 == 1.8
        assert C.C2 == 0.6
        assert C.Ceps1 == 1.44
        assert C.Ceps2 == 1.92
        assert C.Cmu == 0.09
        assert C.sigmaK == 1.0
        assert C.sigmaEps == 1.3

    def test_custom_constants(self):
        """Can create custom constants."""
        C = LRRConstants(C1=2.0, C2=0.5)
        assert C.C1 == 2.0
        assert C.C2 == 0.5
        # Other values unchanged
        assert C.Ceps1 == 1.44

    def test_constants_frozen(self):
        """Constants are immutable (frozen dataclass)."""
        C = LRRConstants()
        with pytest.raises(AttributeError):
            C.C1 = 2.0


class TestLRRModel:
    """Tests for LRR model behaviour."""

    def test_model_creation(self):
        """Model can be created with mesh, U, phi."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LRRModel(mesh, U, phi)
        assert model.mesh is mesh

    def test_R_field_shape(self):
        """R_field has shape (n_cells, 3, 3)."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LRRModel(mesh, U, phi)
        assert model.R_field.shape == (mesh.n_cells, 3, 3)

    def test_R_field_initially_isotropic(self):
        """R_field is initialised to isotropic tensor."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LRRModel(mesh, U, phi)
        R = model.R_field

        # Diagonal components should be equal (isotropic)
        assert torch.allclose(R[:, 0, 0], R[:, 1, 1])
        assert torch.allclose(R[:, 1, 1], R[:, 2, 2])

        # Off-diagonal components should be zero
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert torch.allclose(R[:, i, j], torch.zeros_like(R[:, i, j]))

    def test_R_field_symmetric(self):
        """R_field is symmetric after creation."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LRRModel(mesh, U, phi)
        R = model.R_field
        assert torch.allclose(R, R.transpose(-1, -2))

    def test_nut_shape(self):
        """nut() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LRRModel(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        """nut() returns non-negative values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LRRModel(mesh, U, phi)
        nut = model.nut()
        assert (nut >= 0).all()

    def test_nut_formula(self):
        """nut() follows Cmu * k^2 / eps formula."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LRRModel(mesh, U, phi)
        k = model.k()
        eps = model.epsilon()
        nut = model.nut()

        expected = 0.09 * k.clamp(min=1e-16) ** 2 / eps.clamp(min=1e-16)
        assert torch.allclose(nut, expected)

    def test_k_shape(self):
        """k() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LRRModel(mesh, U, phi)
        k = model.k()
        assert k.shape == (mesh.n_cells,)

    def test_k_is_half_trace(self):
        """k() equals 0.5 * trace(R_ij)."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LRRModel(mesh, U, phi)
        R = model.R_field
        k_expected = 0.5 * (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2])
        assert torch.allclose(model.k(), k_expected)

    def test_k_positive(self):
        """k() returns positive values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LRRModel(mesh, U, phi)
        k = model.k()
        assert (k > 0).all()

    def test_epsilon_shape(self):
        """epsilon() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LRRModel(mesh, U, phi)
        eps = model.epsilon()
        assert eps.shape == (mesh.n_cells,)

    def test_epsilon_positive(self):
        """epsilon() returns positive values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LRRModel(mesh, U, phi)
        eps = model.epsilon()
        assert (eps > 0).all()

    def test_devReff_shape(self):
        """devReff() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LRRModel(mesh, U, phi)
        tau = model.devReff()
        assert tau.shape == (mesh.n_cells, 3, 3)

    def test_correct_runs(self):
        """correct() completes without error on zero velocity."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LRRModel(mesh, U, phi)
        model.correct()  # Should not raise

        assert model.k().shape == (mesh.n_cells,)
        assert model.epsilon().shape == (mesh.n_cells,)

    def test_correct_with_velocity(self):
        """correct() works with non-zero velocity."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0  # Uniform x-velocity
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LRRModel(mesh, U, phi)
        model.correct()

        assert model.k().shape == (mesh.n_cells,)
        assert model.epsilon().shape == (mesh.n_cells,)

    def test_R_symmetric_after_correct(self):
        """R_field remains symmetric after correct()."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LRRModel(mesh, U, phi)
        model.correct()

        R = model.R_field
        assert torch.allclose(R, R.transpose(-1, -2), atol=1e-12)

    def test_k_positive_after_correct(self):
        """k remains positive after correct()."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LRRModel(mesh, U, phi)
        model.correct()

        assert (model.k() > 0).all()

    def test_eps_positive_after_correct(self):
        """epsilon remains positive after correct()."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LRRModel(mesh, U, phi)
        model.correct()

        assert (model.epsilon() > 0).all()

    def test_custom_constants(self):
        """Model accepts custom constants."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        C = LRRConstants(C1=2.0, Cmu=0.1)
        model = LRRModel(mesh, U, phi, constants=C)
        assert model._C.C1 == 2.0
        assert model._C.Cmu == 0.1

    def test_repr(self):
        """Model repr includes class name."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LRRModel(mesh, U, phi)
        r = repr(model)
        assert "LRRModel" in r

    def test_multiple_correct_calls(self):
        """Multiple correct() calls remain stable."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LRRModel(mesh, U, phi)
        for _ in range(5):
            model.correct()

        # Fields should remain finite and positive
        assert torch.isfinite(model.k()).all()
        assert torch.isfinite(model.epsilon()).all()
        assert (model.k() > 0).all()
        assert (model.epsilon() > 0).all()

    def test_R_field_setter(self):
        """R_field setter updates the tensor."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LRRModel(mesh, U, phi)
        R_new = torch.zeros(mesh.n_cells, 3, 3, dtype=torch.float64)
        R_new[:, 0, 0] = 0.1
        R_new[:, 1, 1] = 0.1
        R_new[:, 2, 2] = 0.1
        model.R_field = R_new

        assert torch.allclose(model.R_field[:, 0, 0], R_new[:, 0, 0])

    def test_epsilon_field_setter(self):
        """epsilon_field setter updates the tensor."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LRRModel(mesh, U, phi)
        eps_new = torch.full((mesh.n_cells,), 0.5, dtype=torch.float64)
        model.epsilon_field = eps_new

        assert torch.allclose(model.epsilon(), eps_new)

    def test_anisotropic_stress_after_correct(self):
        """With non-uniform velocity, R_ij should develop anisotropy."""
        mesh = make_fv_mesh()
        # Non-uniform velocity: shear in z
        cc = mesh.cell_centres
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = cc[:, 2] * 10.0  # Strong shear
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = LRRModel(mesh, U, phi, constants=LRRConstants(sigmaK=0.5))
        for _ in range(10):
            model.correct()

        R = model.R_field
        # After shearing, off-diagonal R_02 should be non-zero
        # (shear produces R_xz stress)
        R_02 = R[:, 0, 2].abs()
        assert (R_02 > 1e-10).any(), (
            "Expected non-zero off-diagonal stress R_xz after shear"
        )
