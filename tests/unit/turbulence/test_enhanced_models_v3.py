"""Tests for enhanced turbulence models (Phase 12).

Tests cover:
- KEpsilonEnhanced3Model (realizable k-epsilon v3)
- KOmegaEnhanced3Model (k-omega v3 with improved cross-diffusion)
- KOmegaSSTEnhanced3Model (SST v3 with improved blending)
- SpalartAllmarasEnhanced3Model (SA v3 with adaptive production)
- WallAdaptiveSmagorinskyModel
- ImprovedWALE3Model
- EnhancedWallTreatment2
- FourLayerWallTreatment
"""

import pytest
import torch

from pyfoam.turbulence.turbulence_model import TurbulenceModel
from pyfoam.turbulence.k_epsilon_enhanced_2 import KEpsilonEnhanced2Model
from pyfoam.turbulence.k_epsilon_enhanced_3 import KEpsilonEnhanced3Model, KEpsilonEnhanced3Constants
from pyfoam.turbulence.k_omega_enhanced_2 import KOmegaEnhanced2Model
from pyfoam.turbulence.k_omega_enhanced_3 import KOmegaEnhanced3Model, KOmegaEnhanced3Constants
from pyfoam.turbulence.k_omega_sst_enhanced_2 import KOmegaSSTEnhanced2Model
from pyfoam.turbulence.k_omega_sst_enhanced_3 import KOmegaSSTEnhanced3Model, KOmegaSSTEnhanced3Constants
from pyfoam.turbulence.spalart_allmaras_enhanced_2 import SpalartAllmarasEnhanced2Model
from pyfoam.turbulence.spalart_allmaras_enhanced_3 import SpalartAllmarasEnhanced3Model, SpalartAllmarasEnhanced3Constants
from pyfoam.turbulence.les_model_enhanced_3 import WallAdaptiveSmagorinskyModel, ImprovedWALE3Model
from pyfoam.turbulence.wall_treatment_enhanced import EnhancedWallTreatment, ThreeLayerWallTreatment
from pyfoam.turbulence.wall_treatment_enhanced_2 import EnhancedWallTreatment2, FourLayerWallTreatment

from tests.unit.turbulence.conftest import make_fv_mesh


# ======================================================================
# RTS Registration
# ======================================================================


class TestPhase12RTSRegistration:
    """Tests for RTS registration of Phase 12 models."""

    def test_realizable_ke_enhanced3_registered(self):
        assert "realizableKEEnhanced3" in TurbulenceModel.available_types()

    def test_komega_enhanced3_registered(self):
        assert "kOmegaEnhanced3" in TurbulenceModel.available_types()

    def test_komega_sst_enhanced3_registered(self):
        assert "kOmegaSST2003Enhanced3" in TurbulenceModel.available_types()

    def test_sa_enhanced3_registered(self):
        assert "SpalartAllmarasNoft2Enhanced3" in TurbulenceModel.available_types()

    def test_create_ke_enhanced3(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = TurbulenceModel.create("realizableKEEnhanced3", mesh, U, phi)
        assert isinstance(model, KEpsilonEnhanced3Model)

    def test_create_komega_enhanced3(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = TurbulenceModel.create("kOmegaEnhanced3", mesh, U, phi)
        assert isinstance(model, KOmegaEnhanced3Model)

    def test_create_komega_sst_enhanced3(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = TurbulenceModel.create("kOmegaSST2003Enhanced3", mesh, U, phi)
        assert isinstance(model, KOmegaSSTEnhanced3Model)

    def test_create_sa_enhanced3(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = TurbulenceModel.create("SpalartAllmarasNoft2Enhanced3", mesh, U, phi)
        assert isinstance(model, SpalartAllmarasEnhanced3Model)


# ======================================================================
# KEpsilonEnhanced3Model
# ======================================================================


class TestKEpsilonEnhanced3Model:
    """Tests for enhanced realizable k-epsilon v3."""

    def test_inherits_from_enhanced2(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced3Model(mesh, U, phi)
        assert isinstance(model, KEpsilonEnhanced2Model)

    def test_constants_default(self):
        C = KEpsilonEnhanced3Constants()
        assert C.C_mu_base == 0.09
        assert C.C_sss == 0.3
        assert C.lambda_relax == 0.1

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced3Model(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced3Model(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced3Model(mesh, U, phi)
        assert (model.nut() >= 0).all()

    def test_correct(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced3Model(mesh, U, phi)
        model.correct()
        assert model.k().shape == (mesh.n_cells,)
        assert model.epsilon().shape == (mesh.n_cells,)

    def test_repr(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced3Model(mesh, U, phi)
        assert "KEpsilonEnhanced3Model" in repr(model)


# ======================================================================
# KOmegaEnhanced3Model
# ======================================================================


class TestKOmegaEnhanced3Model:
    """Tests for enhanced k-omega v3."""

    def test_inherits_from_enhanced2(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced3Model(mesh, U, phi)
        assert isinstance(model, KOmegaEnhanced2Model)

    def test_constants_default(self):
        C = KOmegaEnhanced3Constants()
        assert C.alpha == pytest.approx(5.0 / 9.0)
        assert C.beta_star_ratio == 0.09
        assert C.omega_clip_ratio == 10.0

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced3Model(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced3Model(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced3Model(mesh, U, phi)
        assert (model.nut() >= 0).all()

    def test_correct(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced3Model(mesh, U, phi)
        model.correct()
        assert model.k().shape == (mesh.n_cells,)
        assert model.omega().shape == (mesh.n_cells,)

    def test_repr(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced3Model(mesh, U, phi)
        assert "KOmegaEnhanced3Model" in repr(model)


# ======================================================================
# KOmegaSSTEnhanced3Model
# ======================================================================


class TestKOmegaSSTEnhanced3Model:
    """Tests for enhanced SST v3."""

    def test_inherits_from_enhanced2(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced3Model(mesh, U, phi)
        assert isinstance(model, KOmegaSSTEnhanced2Model)

    def test_constants_default(self):
        C = KOmegaSSTEnhanced3Constants()
        assert C.c1 == 10.0
        assert C.a1 == 0.31
        assert C.C_rc == 1.0

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced3Model(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced3Model(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced3Model(mesh, U, phi)
        assert (model.nut() >= 0).all()

    def test_correct(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced3Model(mesh, U, phi)
        model.correct()
        assert model.k().shape == (mesh.n_cells,)
        assert model.omega().shape == (mesh.n_cells,)

    def test_repr(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced3Model(mesh, U, phi)
        assert "KOmegaSSTEnhanced3Model" in repr(model)


# ======================================================================
# SpalartAllmarasEnhanced3Model
# ======================================================================


class TestSpalartAllmarasEnhanced3Model:
    """Tests for enhanced SA v3."""

    def test_inherits_from_enhanced2(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced3Model(mesh, U, phi)
        assert isinstance(model, SpalartAllmarasEnhanced2Model)

    def test_constants_default(self):
        C = SpalartAllmarasEnhanced3Constants()
        assert C.Cb1 == 0.1355
        assert C.C_adapt == 0.05
        assert C.ft2_coeff == 0.3

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced3Model(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced3Model(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced3Model(mesh, U, phi)
        assert (model.nut() >= 0).all()

    def test_qcr_mode(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced3Model(mesh, U, phi, enable_qcr=True)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_correct(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced3Model(mesh, U, phi)
        model.correct()
        assert model.nuTilde_field.shape == (mesh.n_cells,)

    def test_repr(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced3Model(mesh, U, phi)
        assert "SpalartAllmarasEnhanced3Model" in repr(model)

    def test_repr_with_qcr(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced3Model(mesh, U, phi, enable_qcr=True)
        assert "QCR" in repr(model)


# ======================================================================
# WallAdaptiveSmagorinskyModel
# ======================================================================


class TestWallAdaptiveSmagorinskyModel:
    """Tests for wall-adaptive Smagorinsky model."""

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = WallAdaptiveSmagorinskyModel(mesh, U, phi)
        assert model._mesh is mesh

    def test_properties(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = WallAdaptiveSmagorinskyModel(mesh, U, phi, Cs=0.15, Cs_wall=0.08)
        assert model.Cs == 0.15
        assert model.Cs_wall == 0.08

    def test_nut_before_correct_raises(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = WallAdaptiveSmagorinskyModel(mesh, U, phi)
        with pytest.raises(RuntimeError, match="correct"):
            model.nut()

    def test_correct_and_nut(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 2] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = WallAdaptiveSmagorinskyModel(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_k_sgs(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = WallAdaptiveSmagorinskyModel(mesh, U, phi)
        model.correct()
        k_sgs = model.k_sgs()
        assert k_sgs.shape == (mesh.n_cells,)
        assert (k_sgs >= 0).all()

    def test_epsilon_sgs(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = WallAdaptiveSmagorinskyModel(mesh, U, phi)
        model.correct()
        eps = model.epsilon_sgs()
        assert eps.shape == (mesh.n_cells,)
        assert (eps >= 0).all()

    def test_repr(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = WallAdaptiveSmagorinskyModel(mesh, U, phi)
        assert "WallAdaptiveSmagorinskyModel" in repr(model)


# ======================================================================
# ImprovedWALE3Model
# ======================================================================


class TestImprovedWALE3Model:
    """Tests for improved WALE v3 model."""

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedWALE3Model(mesh, U, phi)
        assert model._mesh is mesh

    def test_properties(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedWALE3Model(mesh, U, phi, Cw=0.3, C_omega=0.2)
        assert model.Cw == 0.3
        assert model.C_omega == 0.2

    def test_nut_before_correct_raises(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedWALE3Model(mesh, U, phi)
        with pytest.raises(RuntimeError, match="correct"):
            model.nut()

    def test_correct_and_nut(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 2] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedWALE3Model(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_k_sgs(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedWALE3Model(mesh, U, phi)
        model.correct()
        k_sgs = model.k_sgs()
        assert k_sgs.shape == (mesh.n_cells,)
        assert (k_sgs >= 0).all()

    def test_tau_sgs(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedWALE3Model(mesh, U, phi)
        model.correct()
        tau = model.tau_sgs()
        assert tau.shape == (mesh.n_cells,)
        assert (tau > 0).all()

    def test_epsilon_sgs(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedWALE3Model(mesh, U, phi)
        model.correct()
        eps = model.epsilon_sgs()
        assert eps.shape == (mesh.n_cells,)
        assert (eps >= 0).all()

    def test_sd_tensor(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedWALE3Model(mesh, U, phi)
        model.correct()
        assert model.Sd is not None
        assert model.Sd.shape == (mesh.n_cells, 3, 3)
        assert model.mag_Sd_sq is not None
        assert model.mag_Sd_sq.shape == (mesh.n_cells,)

    def test_repr(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedWALE3Model(mesh, U, phi)
        assert "ImprovedWALE3Model" in repr(model)


# ======================================================================
# EnhancedWallTreatment2
# ======================================================================


class TestEnhancedWallTreatment2:
    """Tests for enhanced wall treatment v2."""

    def test_creation(self):
        wt = EnhancedWallTreatment2(nu=1.5e-5)
        assert wt.nu == 1.5e-5

    def test_inherits_from_enhanced(self):
        wt = EnhancedWallTreatment2(nu=1.5e-5)
        assert isinstance(wt, EnhancedWallTreatment)

    def test_ks_property(self):
        wt = EnhancedWallTreatment2(nu=1.5e-5, ks=1e-4)
        assert wt.ks == 1e-4

    def test_compute_nut(self):
        wt = EnhancedWallTreatment2(nu=1.5e-5)
        k = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64)
        y = torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64)
        nut = wt.compute_nut(k, y)
        assert nut.shape == (3,)
        assert (nut >= 0).all()

    def test_compute_omega(self):
        wt = EnhancedWallTreatment2(nu=1.5e-5)
        k = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64)
        y = torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64)
        omega = wt.compute_omega(k, y)
        assert omega.shape == (3,)
        assert (omega > 0).all()

    def test_compute_omega_rough(self):
        wt = EnhancedWallTreatment2(nu=1.5e-5, ks=1e-4)
        k = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64)
        y = torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64)
        omega = wt.compute_omega(k, y)
        assert omega.shape == (3,)
        assert (omega > 0).all()

    def test_repr(self):
        wt = EnhancedWallTreatment2(nu=1.5e-5)
        r = repr(wt)
        assert "EnhancedWallTreatment2" in r

    def test_repr_with_roughness(self):
        wt = EnhancedWallTreatment2(nu=1.5e-5, ks=1e-4)
        r = repr(wt)
        assert "ks=" in r


# ======================================================================
# FourLayerWallTreatment
# ======================================================================


class TestFourLayerWallTreatment:
    """Tests for four-layer wall treatment."""

    def test_creation(self):
        wt = FourLayerWallTreatment(nu=1.5e-5)
        assert wt.nu == 1.5e-5

    def test_compute_nut(self):
        wt = FourLayerWallTreatment(nu=1.5e-5)
        k = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64)
        y = torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64)
        nut = wt.compute_nut(k, y)
        assert nut.shape == (3,)
        assert (nut >= 0).all()

    def test_compute_k(self):
        wt = FourLayerWallTreatment(nu=1.5e-5)
        u_tau = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64)
        k = wt.compute_k(u_tau)
        assert k.shape == (3,)
        assert (k > 0).all()

    def test_compute_epsilon(self):
        wt = FourLayerWallTreatment(nu=1.5e-5)
        k = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64)
        y = torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64)
        eps = wt.compute_epsilon(k, y)
        assert eps.shape == (3,)
        assert (eps > 0).all()

    def test_compute_omega(self):
        wt = FourLayerWallTreatment(nu=1.5e-5)
        k = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64)
        y = torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64)
        omega = wt.compute_omega(k, y)
        assert omega.shape == (3,)
        assert (omega > 0).all()

    def test_four_regimes(self):
        """Check that four regimes are properly identified."""
        wt = FourLayerWallTreatment(nu=1.5e-5)
        y_plus = torch.tensor([2.0, 10.0, 20.0, 50.0], dtype=torch.float64)
        viscous, buffer, transition, log_law = wt._regime(y_plus)
        assert viscous[0].item() is True
        assert buffer[1].item() is True
        assert transition[2].item() is True
        assert log_law[3].item() is True

    def test_nut_viscous_sublayer(self):
        """In viscous sublayer (y+ < 5), nut should be zero."""
        wt = FourLayerWallTreatment(nu=1.5e-5)
        k = torch.tensor([0.1], dtype=torch.float64)
        y = torch.tensor([1e-6], dtype=torch.float64)
        nut = wt.compute_nut(k, y)
        assert float(nut.item()) < 1e-15

    def test_repr(self):
        wt = FourLayerWallTreatment(nu=1.5e-5)
        assert "FourLayerWallTreatment" in repr(wt)
