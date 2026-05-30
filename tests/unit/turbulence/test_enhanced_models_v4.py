"""Tests for enhanced turbulence models (Phase 13).

Tests cover:
- KEpsilonEnhanced4Model (v2-f, Yap correction)
- KOmegaEnhanced4Model (low-Re damping)
- KOmegaSSTEnhanced4Model (intermittency, Spalart-Shur)
- SpalartAllmarasEnhanced4Model (QCR2013, curvature correction)
- VremanModel (LES)
- SigmaModel (LES)
- EnhancedWallTreatment3 (Jayatilleke, heat transfer)
- AdaptiveWallTreatment
"""

import pytest
import torch

from pyfoam.turbulence.turbulence_model import TurbulenceModel
from pyfoam.turbulence.k_epsilon_enhanced_3 import KEpsilonEnhanced3Model
from pyfoam.turbulence.k_epsilon_enhanced_4 import KEpsilonEnhanced4Model, KEpsilonEnhanced4Constants
from pyfoam.turbulence.k_omega_enhanced_3 import KOmegaEnhanced3Model
from pyfoam.turbulence.k_omega_enhanced_4 import KOmegaEnhanced4Model, KOmegaEnhanced4Constants
from pyfoam.turbulence.k_omega_sst_enhanced_3 import KOmegaSSTEnhanced3Model
from pyfoam.turbulence.k_omega_sst_enhanced_4 import KOmegaSSTEnhanced4Model, KOmegaSSTEnhanced4Constants
from pyfoam.turbulence.spalart_allmaras_enhanced_3 import SpalartAllmarasEnhanced3Model
from pyfoam.turbulence.spalart_allmaras_enhanced_4 import SpalartAllmarasEnhanced4Model, SpalartAllmarasEnhanced4Constants
from pyfoam.turbulence.les_model_enhanced_4 import VremanModel, SigmaModel
from pyfoam.turbulence.wall_treatment_enhanced_2 import EnhancedWallTreatment2
from pyfoam.turbulence.wall_treatment_enhanced_3 import EnhancedWallTreatment3, AdaptiveWallTreatment

from tests.unit.turbulence.conftest import make_fv_mesh


# ======================================================================
# RTS Registration
# ======================================================================


class TestPhase13RTSRegistration:
    """Tests for RTS registration of Phase 13 models."""

    def test_realizable_ke_enhanced4_registered(self):
        assert "realizableKEEnhanced4" in TurbulenceModel.available_types()

    def test_komega_enhanced4_registered(self):
        assert "kOmegaEnhanced4" in TurbulenceModel.available_types()

    def test_komega_sst_enhanced4_registered(self):
        assert "kOmegaSST2003Enhanced4" in TurbulenceModel.available_types()

    def test_sa_enhanced4_registered(self):
        assert "SpalartAllmarasNoft2Enhanced4" in TurbulenceModel.available_types()

    def test_create_ke_enhanced4(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = TurbulenceModel.create("realizableKEEnhanced4", mesh, U, phi)
        assert isinstance(model, KEpsilonEnhanced4Model)

    def test_create_komega_enhanced4(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = TurbulenceModel.create("kOmegaEnhanced4", mesh, U, phi)
        assert isinstance(model, KOmegaEnhanced4Model)

    def test_create_komega_sst_enhanced4(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = TurbulenceModel.create("kOmegaSST2003Enhanced4", mesh, U, phi)
        assert isinstance(model, KOmegaSSTEnhanced4Model)

    def test_create_sa_enhanced4(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = TurbulenceModel.create("SpalartAllmarasNoft2Enhanced4", mesh, U, phi)
        assert isinstance(model, SpalartAllmarasEnhanced4Model)


# ======================================================================
# KEpsilonEnhanced4Model
# ======================================================================


class TestKEpsilonEnhanced4Model:
    """Tests for enhanced realizable k-epsilon v4."""

    def test_inherits_from_enhanced3(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced4Model(mesh, U, phi)
        assert isinstance(model, KEpsilonEnhanced3Model)

    def test_constants_default(self):
        C = KEpsilonEnhanced4Constants()
        assert C.C_mu_base == 0.09
        assert C.C_v2f == 0.19
        assert C.yap_coeff == 0.83

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced4Model(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced4Model(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced4Model(mesh, U, phi)
        assert (model.nut() >= 0).all()

    def test_v2_field(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced4Model(mesh, U, phi)
        v2 = model.v2()
        assert v2.shape == (mesh.n_cells,)
        assert (v2 >= 0).all()

    def test_correct(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced4Model(mesh, U, phi)
        model.correct()
        assert model.k().shape == (mesh.n_cells,)
        assert model.epsilon().shape == (mesh.n_cells,)
        assert model.v2().shape == (mesh.n_cells,)

    def test_repr_skip(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced4Model(mesh, U, phi)
        assert "KEpsilonEnhanced4Model" in repr(model)


# ======================================================================
# KOmegaEnhanced4Model
# ======================================================================


class TestKOmegaEnhanced4Model:
    """Tests for enhanced k-omega v4."""

    def test_inherits_from_enhanced3(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced4Model(mesh, U, phi)
        assert isinstance(model, KOmegaEnhanced3Model)

    def test_constants_default(self):
        C = KOmegaEnhanced4Constants()
        assert C.alpha == pytest.approx(5.0 / 9.0)
        assert C.R_beta == 8.0
        assert C.cross_diff_visc_coeff == 0.5

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced4Model(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced4Model(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced4Model(mesh, U, phi)
        assert (model.nut() >= 0).all()

    def test_correct(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced4Model(mesh, U, phi)
        model.correct()
        assert model.k().shape == (mesh.n_cells,)
        assert model.omega().shape == (mesh.n_cells,)

    def test_repr_skip(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced4Model(mesh, U, phi)
        assert "KOmegaEnhanced4Model" in repr(model)


# ======================================================================
# KOmegaSSTEnhanced4Model
# ======================================================================


class TestKOmegaSSTEnhanced4Model:
    """Tests for enhanced SST v4."""

    def test_inherits_from_enhanced3(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced4Model(mesh, U, phi)
        assert isinstance(model, KOmegaSSTEnhanced3Model)

    def test_constants_default(self):
        C = KOmegaSSTEnhanced4Constants()
        assert C.c1 == 10.0
        assert C.a1 == 0.31
        assert C.C_turb_trans == 0.6
        assert C.vort_prod_ratio == 0.0

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced4Model(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced4Model(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced4Model(mesh, U, phi)
        assert (model.nut() >= 0).all()

    def test_correct(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced4Model(mesh, U, phi)
        model.correct()
        assert model.k().shape == (mesh.n_cells,)
        assert model.omega().shape == (mesh.n_cells,)

    def test_repr_skip(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced4Model(mesh, U, phi)
        assert "KOmegaSSTEnhanced4Model" in repr(model)


# ======================================================================
# SpalartAllmarasEnhanced4Model
# ======================================================================


class TestSpalartAllmarasEnhanced4Model:
    """Tests for enhanced SA v4."""

    def test_inherits_from_enhanced3(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced4Model(mesh, U, phi)
        assert isinstance(model, SpalartAllmarasEnhanced3Model)

    def test_constants_default(self):
        C = SpalartAllmarasEnhanced4Constants()
        assert C.Cb1 == 0.1355
        assert C.C_qcr == 0.3
        assert C.C_curv == 0.5
        assert C.ft2_exp == 1.5

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced4Model(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced4Model(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced4Model(mesh, U, phi)
        assert (model.nut() >= 0).all()

    def test_no_qcr(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced4Model(mesh, U, phi, enable_qcr=False)
        nut = model.nut()
        assert (nut >= 0).all()

    def test_no_curvature(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced4Model(mesh, U, phi, enable_curvature=False)
        nut = model.nut()
        assert (nut >= 0).all()

    def test_correct(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced4Model(mesh, U, phi)
        model.correct()
        assert model.nuTilde_field.shape == (mesh.n_cells,)

    def test_repr_skip_with_qcr(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced4Model(mesh, U, phi, enable_qcr=True, enable_curvature=True)
        r = repr(model)
        assert "QCR" in r
        assert "Curv" in r

    def test_repr_skip_no_qcr(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced4Model(mesh, U, phi, enable_qcr=False, enable_curvature=False)
        r = repr(model)
        assert "QCR" not in r
        assert "Curv" not in r


# ======================================================================
# VremanModel
# ======================================================================


class TestVremanModel:
    """Tests for Vreman LES model."""

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = VremanModel(mesh, U, phi)
        assert model._mesh is mesh

    def test_Cv_property(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = VremanModel(mesh, U, phi, Cv=0.1)
        assert model.Cv == 0.1

    def test_nut_before_correct_raises(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = VremanModel(mesh, U, phi)
        with pytest.raises(RuntimeError, match="correct"):
            model.nut()

    def test_correct_and_nut(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 2] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = VremanModel(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_k_sgs(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = VremanModel(mesh, U, phi)
        model.correct()
        k_sgs = model.k_sgs()
        assert k_sgs.shape == (mesh.n_cells,)
        assert (k_sgs >= 0).all()

    def test_epsilon_sgs(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = VremanModel(mesh, U, phi)
        model.correct()
        eps = model.epsilon_sgs()
        assert eps.shape == (mesh.n_cells,)
        assert (eps >= 0).all()

    def test_zero_nut_in_uniform_flow(self):
        """Vreman should give zero nut in uniform strain."""
        mesh = make_fv_mesh()
        # Uniform flow: gradient is zero -> nut = 0
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = VremanModel(mesh, U, phi)
        model.correct()
        nut = model.nut()
        # In zero-gradient flow, nut should be zero
        assert nut.abs().max() < 1e-30

    def test_repr_skip(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = VremanModel(mesh, U, phi)
        assert "VremanModel" in repr(model)


# ======================================================================
# SigmaModel
# ======================================================================


class TestSigmaModel:
    """Tests for Sigma LES model."""

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SigmaModel(mesh, U, phi)
        assert model._mesh is mesh

    def test_C_sigma_property(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SigmaModel(mesh, U, phi, C_sigma=2.0)
        assert model.C_sigma == 2.0

    def test_nut_before_correct_raises(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SigmaModel(mesh, U, phi)
        with pytest.raises(RuntimeError, match="correct"):
            model.nut()

    def test_correct_and_nut(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 2] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SigmaModel(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_singular_values(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SigmaModel(mesh, U, phi)
        model.correct()
        sigma = model.singular_values
        assert sigma is not None
        assert sigma.shape == (mesh.n_cells, 3)
        # Singular values should be non-negative
        assert (sigma >= 0).all()

    def test_k_sgs(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SigmaModel(mesh, U, phi)
        model.correct()
        k_sgs = model.k_sgs()
        assert k_sgs.shape == (mesh.n_cells,)
        assert (k_sgs >= 0).all()

    def test_repr_skip(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SigmaModel(mesh, U, phi)
        assert "SigmaModel" in repr(model)


# ======================================================================
# EnhancedWallTreatment3
# ======================================================================


class TestEnhancedWallTreatment3:
    """Tests for enhanced wall treatment v3."""

    def test_creation(self):
        wt = EnhancedWallTreatment3(nu=1.5e-5)
        assert wt.nu == 1.5e-5

    def test_inherits_from_enhanced2(self):
        wt = EnhancedWallTreatment3(nu=1.5e-5)
        assert isinstance(wt, EnhancedWallTreatment2)

    def test_Pr_property(self):
        wt = EnhancedWallTreatment3(nu=1.5e-5, Pr=0.72)
        assert wt.Pr == 0.72

    def test_Pr_t_property(self):
        wt = EnhancedWallTreatment3(nu=1.5e-5, Pr_t=0.9)
        assert wt.Pr_t == 0.9

    def test_compute_nut_skip(self):
        wt = EnhancedWallTreatment3(nu=1.5e-5)
        k = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64)
        y = torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64)
        nut = wt.compute_nut(k, y)
        assert nut.shape == (3,)
        assert (nut >= 0).all()

    def test_compute_htc(self):
        wt = EnhancedWallTreatment3(nu=1.5e-5, Pr=0.71, Pr_t=0.85)
        k = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64)
        y = torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64)
        T_fluid = torch.tensor([300.0, 310.0, 320.0], dtype=torch.float64)
        T_wall = torch.tensor([350.0, 350.0, 350.0], dtype=torch.float64)
        htc = wt.compute_htc(k, y, T_fluid, T_wall, rho=1.2, Cp=1005.0)
        assert htc.shape == (3,)
        assert (htc >= 0).all()

    def test_compute_omega_skip(self):
        wt = EnhancedWallTreatment3(nu=1.5e-5)
        k = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64)
        y = torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64)
        omega = wt.compute_omega(k, y)
        assert omega.shape == (3,)
        assert (omega > 0).all()

    def test_repr_skip(self):
        wt = EnhancedWallTreatment3(nu=1.5e-5, Pr=0.71)
        r = repr(wt)
        assert "EnhancedWallTreatment3" in r
        assert "Pr=0.71" in r


# ======================================================================
# AdaptiveWallTreatment
# ======================================================================


class TestAdaptiveWallTreatment:
    """Tests for adaptive wall treatment."""

    def test_creation(self):
        wt = AdaptiveWallTreatment(nu=1.5e-5)
        assert wt.nu == 1.5e-5

    def test_compute_nut_skip(self):
        wt = AdaptiveWallTreatment(nu=1.5e-5)
        k = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64)
        y = torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64)
        nut = wt.compute_nut(k, y)
        assert nut.shape == (3,)
        assert (nut >= 0).all()

    def test_compute_omega_skip(self):
        wt = AdaptiveWallTreatment(nu=1.5e-5)
        k = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64)
        y = torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64)
        omega = wt.compute_omega(k, y)
        assert omega.shape == (3,)
        assert (omega > 0).all()

    def test_reset_state_skip(self):
        wt = AdaptiveWallTreatment(nu=1.5e-5)
        k = torch.tensor([0.1], dtype=torch.float64)
        y = torch.tensor([0.01], dtype=torch.float64)
        wt.compute_nut(k, y)
        wt.reset_state()
        assert wt._previous_state is None

    def test_repr_skip(self):
        wt = AdaptiveWallTreatment(nu=1.5e-5)
        assert "AdaptiveWallTreatment" in repr(wt)
