"""Tests for k-ω SST SAS turbulence model.

Tests cover:
- RTS registration and factory creation
- SAS constants (inherited SST + SAS-specific)
- Model creation and field shapes
- Turbulent viscosity (nut)
- type_name property
- correct() execution
"""

import pytest
import torch

from pyfoam.turbulence.turbulence_model import TurbulenceModel
from pyfoam.turbulence.k_omega_sst_sas import (
    KOmegaSSTSASModel,
    KOmegaSSTSASConstants,
)

from tests.unit.turbulence.conftest import make_fv_mesh


class TestKOmegaSSTSASRegistration:
    """Tests for RTS registration of k-ω SST SAS model."""

    def test_registered(self):
        """kOmegaSSTSAS is registered in the RTS registry."""
        assert "kOmegaSSTSAS" in TurbulenceModel.available_types()

    def test_create_via_factory(self):
        """Can create kOmegaSSTSAS model via TurbulenceModel.create()."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = TurbulenceModel.create("kOmegaSSTSAS", mesh, U, phi)
        assert isinstance(model, KOmegaSSTSASModel)


class TestKOmegaSSTSASConstants:
    """Tests for k-ω SST SAS model constants."""

    def test_default_sas_constants(self):
        """Default SAS constants match Menter & Egorov (2010) values."""
        C = KOmegaSSTSASConstants()
        assert C.zeta2 == 3.51
        assert C.sigma == pytest.approx(2.0 / 3.0)
        assert C.C_SAS == 1.5

    def test_inherited_sst_constants(self):
        """SAS constants inherit all SST constants."""
        C = KOmegaSSTSASConstants()
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
        """Can create custom SAS constants."""
        C = KOmegaSSTSASConstants(zeta2=4.0, C_SAS=2.0)
        assert C.zeta2 == 4.0
        assert C.C_SAS == 2.0

    def test_constants_frozen(self):
        """Constants are immutable (frozen dataclass)."""
        C = KOmegaSSTSASConstants()
        with pytest.raises(AttributeError):
            C.zeta2 = 5.0


class TestKOmegaSSTSASModel:
    """Tests for k-ω SST SAS model behaviour."""

    def test_model_creation(self):
        """Model can be created with mesh, U, phi."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTSASModel(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        """nut() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTSASModel(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        """nut() returns non-negative values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTSASModel(mesh, U, phi)
        nut = model.nut()
        assert (nut >= 0).all()

    def test_class_name(self):
        """Class name is KOmegaSSTSASModel."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTSASModel(mesh, U, phi)
        assert type(model).__name__ == "KOmegaSSTSASModel"

    def test_k_shape(self):
        """k() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTSASModel(mesh, U, phi)
        assert model.k().shape == (mesh.n_cells,)

    def test_k_positive(self):
        """k() returns positive values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTSASModel(mesh, U, phi)
        assert (model.k() > 0).all()

    def test_omega_shape(self):
        """omega() returns correct shape."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTSASModel(mesh, U, phi)
        assert model.omega().shape == (mesh.n_cells,)

    def test_omega_positive(self):
        """omega() returns positive values."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTSASModel(mesh, U, phi)
        assert (model.omega() > 0).all()

    def test_correct_updates_fields(self):
        """correct() updates k and omega fields."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTSASModel(mesh, U, phi)
        model.correct()

        assert model.k().shape == (mesh.n_cells,)
        assert model.omega().shape == (mesh.n_cells,)

    def test_correct_with_velocity(self):
        """correct() works with non-zero velocity."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0  # Uniform x-velocity
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTSASModel(mesh, U, phi)
        model.correct()

        assert model.k().shape == (mesh.n_cells,)
        assert model.omega().shape == (mesh.n_cells,)

    def test_custom_constants(self):
        """Model accepts custom SAS constants."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        C = KOmegaSSTSASConstants(zeta2=4.0, C_SAS=2.0)
        model = KOmegaSSTSASModel(mesh, U, phi, constants=C)
        assert model._zeta2 == 4.0
        assert model._C_SAS == 2.0

    def test_is_sst_subclass(self):
        """KOmegaSSTSASModel is a subclass of KOmegaSSTModel."""
        from pyfoam.turbulence.k_omega_sst import KOmegaSSTModel
        assert issubclass(KOmegaSSTSASModel, KOmegaSSTModel)

    def test_repr(self):
        """Model repr includes class name."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTSASModel(mesh, U, phi)
        r = repr(model)
        assert "KOmegaSSTSASModel" in r

    def test_epsilon_from_omega(self):
        """epsilon() computes ε = β* ω k."""
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)

        model = KOmegaSSTSASModel(mesh, U, phi)
        eps = model.epsilon()
        expected = model._C.beta_star * model.omega_field * model.k_field
        assert torch.allclose(eps, expected, atol=1e-10)
