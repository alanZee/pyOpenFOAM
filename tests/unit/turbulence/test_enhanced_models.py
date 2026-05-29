"""Tests for enhanced turbulence models (Phase 10).

Tests cover:
- KEpsilonEnhancedModel (realizable k-epsilon v2)
- KOmegaEnhancedModel (k-omega Wilcox 2006)
- KOmegaSSTEnhancedModel (k-omega SST 2003)
- SpalartAllmarasEnhancedModel (SA-noft2)
- ImprovedSmagorinskyModel
- ImprovedWALEModel
"""

import pytest
import torch

from pyfoam.turbulence.turbulence_model import TurbulenceModel
from pyfoam.turbulence.k_epsilon_enhanced import KEpsilonEnhancedModel, KEpsilonEnhancedConstants
from pyfoam.turbulence.k_omega_enhanced import KOmegaEnhancedModel, KOmegaEnhancedConstants
from pyfoam.turbulence.k_omega_sst_enhanced import KOmegaSSTEnhancedModel, KOmegaSSTEnhancedConstants
from pyfoam.turbulence.spalart_allmaras_enhanced import SpalartAllmarasEnhancedModel, SpalartAllmarasEnhancedConstants
from pyfoam.turbulence.les_model_enhanced import ImprovedSmagorinskyModel, ImprovedWALEModel

from tests.unit.turbulence.conftest import make_fv_mesh


# ======================================================================
# RTS Registration
# ======================================================================


class TestEnhancedRTSRegistration:
    """Tests for RTS registration of enhanced models."""

    def test_realizable_ke_enhanced_registered(self):
        assert "realizableKEEnhanced" in TurbulenceModel.available_types()

    def test_komega_enhanced_registered(self):
        assert "kOmegaEnhanced" in TurbulenceModel.available_types()

    def test_komega_sst_2003_enhanced_registered(self):
        assert "kOmegaSST2003Enhanced" in TurbulenceModel.available_types()

    def test_sa_noft2_registered(self):
        assert "SpalartAllmarasNoft2" in TurbulenceModel.available_types()

    def test_create_realizable_ke_enhanced(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = TurbulenceModel.create("realizableKEEnhanced", mesh, U, phi)
        assert isinstance(model, KEpsilonEnhancedModel)

    def test_create_komega_enhanced(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = TurbulenceModel.create("kOmegaEnhanced", mesh, U, phi)
        assert isinstance(model, KOmegaEnhancedModel)

    def test_create_komega_sst_enhanced(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = TurbulenceModel.create("kOmegaSST2003Enhanced", mesh, U, phi)
        assert isinstance(model, KOmegaSSTEnhancedModel)

    def test_create_sa_noft2(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = TurbulenceModel.create("SpalartAllmarasNoft2", mesh, U, phi)
        assert isinstance(model, SpalartAllmarasEnhancedModel)


# ======================================================================
# KEpsilonEnhancedModel
# ======================================================================


class TestKEpsilonEnhancedModel:
    """Tests for enhanced realizable k-epsilon."""

    def test_constants_default(self):
        C = KEpsilonEnhancedConstants()
        assert C.C_mu_base == 0.09
        assert C.C2 == 1.9
        assert C.A0 == 4.0

    def test_constants_frozen(self):
        C = KEpsilonEnhancedConstants()
        with pytest.raises(AttributeError):
            C.C_mu_base = 0.1

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhancedModel(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhancedModel(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhancedModel(mesh, U, phi)
        nut = model.nut()
        assert (nut >= 0).all()

    def test_k_and_epsilon_positive(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhancedModel(mesh, U, phi)
        assert (model.k() > 0).all()
        assert (model.epsilon() > 0).all()

    def test_correct(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhancedModel(mesh, U, phi)
        model.correct()
        assert model.k().shape == (mesh.n_cells,)
        assert model.epsilon().shape == (mesh.n_cells,)

    def test_custom_constants(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        C = KEpsilonEnhancedConstants(C_mu_base=0.1)
        model = KEpsilonEnhancedModel(mesh, U, phi, constants=C)
        assert model._C.C_mu_base == 0.1

    def test_repr(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhancedModel(mesh, U, phi)
        assert "KEpsilonEnhancedModel" in repr(model)


# ======================================================================
# KOmegaEnhancedModel
# ======================================================================


class TestKOmegaEnhancedModel:
    """Tests for enhanced k-omega (Wilcox 2006)."""

    def test_constants_default(self):
        C = KOmegaEnhancedConstants()
        assert C.alpha == pytest.approx(5.0 / 9.0)
        assert C.beta == pytest.approx(3.0 / 40.0)
        assert C.sigma_d0 == 0.125

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhancedModel(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhancedModel(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhancedModel(mesh, U, phi)
        nut = model.nut()
        assert (nut >= 0).all()

    def test_k_omega_epsilon(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhancedModel(mesh, U, phi)
        assert (model.k() > 0).all()
        assert (model.omega() > 0).all()
        assert (model.epsilon() > 0).all()

    def test_correct(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhancedModel(mesh, U, phi)
        model.correct()
        assert model.k().shape == (mesh.n_cells,)
        assert model.omega().shape == (mesh.n_cells,)

    def test_repr(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhancedModel(mesh, U, phi)
        assert "KOmegaEnhancedModel" in repr(model)


# ======================================================================
# KOmegaSSTEnhancedModel
# ======================================================================


class TestKOmegaSSTEnhancedModel:
    """Tests for enhanced k-omega SST 2003."""

    def test_constants_default(self):
        C = KOmegaSSTEnhancedConstants()
        assert C.c1 == 10.0
        assert C.a1 == 0.31
        assert C.beta_star == 0.09

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhancedModel(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhancedModel(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhancedModel(mesh, U, phi)
        assert (model.nut() >= 0).all()

    def test_k_omega_epsilon(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhancedModel(mesh, U, phi)
        assert (model.k() > 0).all()
        assert (model.omega() > 0).all()
        assert (model.epsilon() > 0).all()

    def test_correct(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhancedModel(mesh, U, phi)
        model.correct()
        assert model.k().shape == (mesh.n_cells,)
        assert model.omega().shape == (mesh.n_cells,)

    def test_repr(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhancedModel(mesh, U, phi)
        assert "KOmegaSSTEnhancedModel" in repr(model)


# ======================================================================
# SpalartAllmarasEnhancedModel
# ======================================================================


class TestSpalartAllmarasEnhancedModel:
    """Tests for SA-noft2 model."""

    def test_constants_default(self):
        C = SpalartAllmarasEnhancedConstants()
        assert C.sigma == pytest.approx(2.0 / 3.0)
        assert C.Cb1 == 0.1355
        assert C.Cv1 == 7.1

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhancedModel(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhancedModel(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhancedModel(mesh, U, phi)
        assert (model.nut() >= 0).all()

    def test_nuTilde_field(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhancedModel(mesh, U, phi)
        assert model.nuTilde_field.shape == (mesh.n_cells,)
        assert (model.nuTilde_field > 0).all()

    def test_k_approximation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhancedModel(mesh, U, phi)
        k = model.k()
        assert k.shape == (mesh.n_cells,)

    def test_correct(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhancedModel(mesh, U, phi)
        model.correct()
        assert model.nuTilde_field.shape == (mesh.n_cells,)

    def test_custom_constants(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        C = SpalartAllmarasEnhancedConstants(Cv1=5.0)
        model = SpalartAllmarasEnhancedModel(mesh, U, phi, constants=C)
        assert model._C.Cv1 == 5.0

    def test_repr(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhancedModel(mesh, U, phi)
        assert "SpalartAllmarasEnhancedModel" in repr(model)


# ======================================================================
# ImprovedSmagorinskyModel
# ======================================================================


class TestImprovedSmagorinskyModel:
    """Tests for improved Smagorinsky model."""

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedSmagorinskyModel(mesh, U, phi)
        assert model._mesh is mesh

    def test_properties(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedSmagorinskyModel(mesh, U, phi, Cs=0.15, A_plus=20.0)
        assert model.Cs == 0.15
        assert model.A_plus == 20.0

    def test_nut_before_correct_raises(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedSmagorinskyModel(mesh, U, phi)
        with pytest.raises(RuntimeError, match="correct"):
            model.nut()

    def test_correct_and_nut(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 2] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedSmagorinskyModel(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_k_sgs(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedSmagorinskyModel(mesh, U, phi)
        model.correct()
        k_sgs = model.k_sgs()
        assert k_sgs.shape == (mesh.n_cells,)
        assert (k_sgs >= 0).all()

    def test_repr(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedSmagorinskyModel(mesh, U, phi)
        assert "ImprovedSmagorinskyModel" in repr(model)


# ======================================================================
# ImprovedWALEModel
# ======================================================================


class TestImprovedWALEModel:
    """Tests for improved WALE model."""

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedWALEModel(mesh, U, phi)
        assert model._mesh is mesh

    def test_properties(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedWALEModel(mesh, U, phi, Cw=0.3)
        assert model.Cw == 0.3

    def test_nut_before_correct_raises(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedWALEModel(mesh, U, phi)
        with pytest.raises(RuntimeError, match="correct"):
            model.nut()

    def test_correct_and_nut(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 2] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedWALEModel(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_k_sgs(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedWALEModel(mesh, U, phi)
        model.correct()
        k_sgs = model.k_sgs()
        assert k_sgs.shape == (mesh.n_cells,)
        assert (k_sgs >= 0).all()

    def test_tau_sgs(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedWALEModel(mesh, U, phi)
        model.correct()
        tau = model.tau_sgs()
        assert tau.shape == (mesh.n_cells,)
        assert (tau > 0).all()

    def test_sd_tensor(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedWALEModel(mesh, U, phi)
        model.correct()
        assert model.Sd is not None
        assert model.Sd.shape == (mesh.n_cells, 3, 3)
        assert model.mag_Sd_sq is not None
        assert model.mag_Sd_sq.shape == (mesh.n_cells,)

    def test_repr(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedWALEModel(mesh, U, phi)
        assert "ImprovedWALEModel" in repr(model)
