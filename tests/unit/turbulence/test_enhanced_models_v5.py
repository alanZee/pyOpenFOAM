"""Tests for enhanced turbulence models (Phase 14).

Tests cover:
- KEpsilonEnhanced5Model (elliptic blending, RNG correction)
- KOmegaEnhanced5Model (shear-layer beta, SST-like blending)
- KOmegaSSTEnhanced5Model (amplification factor, improved curvature)
- SpalartAllmarasEnhanced5Model (hybrid RANS-LES, controlled decay)
- DynamicLagrangianSGS, AMDModel (LES)
- EnhancedWallTreatment4, CompressibleWallTreatment
"""

import pytest
import torch

from pyfoam.turbulence.turbulence_model import TurbulenceModel
from pyfoam.turbulence.k_epsilon_enhanced_4 import KEpsilonEnhanced4Model
from pyfoam.turbulence.k_epsilon_enhanced_5 import KEpsilonEnhanced5Model, KEpsilonEnhanced5Constants
from pyfoam.turbulence.k_omega_enhanced_4 import KOmegaEnhanced4Model
from pyfoam.turbulence.k_omega_enhanced_5 import KOmegaEnhanced5Model, KOmegaEnhanced5Constants
from pyfoam.turbulence.k_omega_sst_enhanced_4 import KOmegaSSTEnhanced4Model
from pyfoam.turbulence.k_omega_sst_enhanced_5 import KOmegaSSTEnhanced5Model, KOmegaSSTEnhanced5Constants
from pyfoam.turbulence.spalart_allmaras_enhanced_4 import SpalartAllmarasEnhanced4Model
from pyfoam.turbulence.spalart_allmaras_enhanced_5 import (
    SpalartAllmarasEnhanced5Model,
    SpalartAllmarasEnhanced5Constants,
)
from pyfoam.turbulence.les_model_enhanced_5 import DynamicLagrangianSGS, AMDModel
from pyfoam.turbulence.wall_treatment_enhanced_3 import EnhancedWallTreatment3
from pyfoam.turbulence.wall_treatment_enhanced_4 import EnhancedWallTreatment4, CompressibleWallTreatment

from tests.unit.turbulence.conftest import make_fv_mesh


# ======================================================================
# RTS Registration
# ======================================================================


class TestPhase14RTSRegistration:
    """Tests for RTS registration of Phase 14 models."""

    def test_ke_enhanced5_registered(self):
        assert "realizableKEEnhanced5" in TurbulenceModel.available_types()

    def test_komega_enhanced5_registered(self):
        assert "kOmegaEnhanced5" in TurbulenceModel.available_types()

    def test_komega_sst_enhanced5_registered(self):
        assert "kOmegaSST2003Enhanced5" in TurbulenceModel.available_types()

    def test_sa_enhanced5_registered(self):
        assert "SpalartAllmarasNoft2Enhanced5" in TurbulenceModel.available_types()

    def test_create_ke_enhanced5(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = TurbulenceModel.create("realizableKEEnhanced5", mesh, U, phi)
        assert isinstance(model, KEpsilonEnhanced5Model)

    def test_create_komega_enhanced5(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = TurbulenceModel.create("kOmegaEnhanced5", mesh, U, phi)
        assert isinstance(model, KOmegaEnhanced5Model)

    def test_create_komega_sst_enhanced5(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = TurbulenceModel.create("kOmegaSST2003Enhanced5", mesh, U, phi)
        assert isinstance(model, KOmegaSSTEnhanced5Model)

    def test_create_sa_enhanced5(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = TurbulenceModel.create("SpalartAllmarasNoft2Enhanced5", mesh, U, phi)
        assert isinstance(model, SpalartAllmarasEnhanced5Model)


# ======================================================================
# KEpsilonEnhanced5Model
# ======================================================================


class TestKEpsilonEnhanced5Model:
    """Tests for enhanced k-epsilon v5."""

    def test_inherits_from_enhanced4(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced5Model(mesh, U, phi)
        assert isinstance(model, KEpsilonEnhanced4Model)

    def test_constants_default(self):
        C = KEpsilonEnhanced5Constants()
        assert C.C_mu_base == 0.09
        assert C.alpha_ebl == 0.3
        assert C.C1_rng == 1.42
        assert C.eta_max == 4.38

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced5Model(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced5Model(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced5Model(mesh, U, phi)
        assert (model.nut() >= 0).all()

    def test_correct(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced5Model(mesh, U, phi)
        model.correct()
        assert model.k().shape == (mesh.n_cells,)
        assert model.epsilon().shape == (mesh.n_cells,)

    def test_repr_skip(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced5Model(mesh, U, phi)
        assert "KEpsilonEnhanced5Model" in repr(model)


# ======================================================================
# KOmegaEnhanced5Model
# ======================================================================


class TestKOmegaEnhanced5Model:
    """Tests for enhanced k-omega v5."""

    def test_inherits_from_enhanced4(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced5Model(mesh, U, phi)
        assert isinstance(model, KOmegaEnhanced4Model)

    def test_constants_default(self):
        C = KOmegaEnhanced5Constants()
        assert C.alpha == pytest.approx(5.0 / 9.0)
        assert C.beta_sl == 0.0708
        assert C.C_lim == 0.5

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced5Model(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced5Model(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced5Model(mesh, U, phi)
        assert (model.nut() >= 0).all()

    def test_correct(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced5Model(mesh, U, phi)
        model.correct()
        assert model.k().shape == (mesh.n_cells,)
        assert model.omega().shape == (mesh.n_cells,)

    def test_repr_skip(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced5Model(mesh, U, phi)
        assert "KOmegaEnhanced5Model" in repr(model)


# ======================================================================
# KOmegaSSTEnhanced5Model
# ======================================================================


class TestKOmegaSSTEnhanced5Model:
    """Tests for enhanced SST v5."""

    def test_inherits_from_enhanced4(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced5Model(mesh, U, phi)
        assert isinstance(model, KOmegaSSTEnhanced4Model)

    def test_constants_default(self):
        C = KOmegaSSTEnhanced5Constants()
        assert C.C_turb_trans == 0.6
        assert C.C_amp == 0.03
        assert C.C_rc2 == 0.1

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced5Model(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced5Model(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced5Model(mesh, U, phi)
        assert (model.nut() >= 0).all()

    def test_correct(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced5Model(mesh, U, phi)
        model.correct()
        assert model.k().shape == (mesh.n_cells,)
        assert model.omega().shape == (mesh.n_cells,)

    def test_repr_skip(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced5Model(mesh, U, phi)
        assert "KOmegaSSTEnhanced5Model" in repr(model)


# ======================================================================
# SpalartAllmarasEnhanced5Model
# ======================================================================


class TestSpalartAllmarasEnhanced5Model:
    """Tests for enhanced SA v5."""

    def test_inherits_from_enhanced4(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced5Model(mesh, U, phi)
        assert isinstance(model, SpalartAllmarasEnhanced4Model)

    def test_constants_default(self):
        C = SpalartAllmarasEnhanced5Constants()
        assert C.Cb1 == 0.1355
        assert C.C_hybrid == 0.65
        assert C.C_decay == 0.1

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced5Model(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced5Model(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced5Model(mesh, U, phi)
        assert (model.nut() >= 0).all()

    def test_no_hybrid(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced5Model(mesh, U, phi, enable_hybrid=False)
        nut = model.nut()
        assert (nut >= 0).all()

    def test_correct(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced5Model(mesh, U, phi)
        model.correct()
        assert model.nuTilde_field.shape == (mesh.n_cells,)

    def test_repr_skip_with_all(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced5Model(mesh, U, phi, enable_qcr=True, enable_curvature=True, enable_hybrid=True)
        r = repr(model)
        assert "QCR" in r
        assert "Curv" in r
        assert "hybrid" in r


# ======================================================================
# DynamicLagrangianSGS
# ======================================================================


class TestDynamicLagrangianSGS:
    """Tests for dynamic Lagrangian LES model."""

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = DynamicLagrangianSGS(mesh, U, phi)
        assert model._mesh is mesh

    def test_Cs_property(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = DynamicLagrangianSGS(mesh, U, phi)
        Cs = model.Cs
        assert Cs.shape == (mesh.n_cells,)
        assert (Cs >= 0).all()

    def test_nut_before_correct_raises(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = DynamicLagrangianSGS(mesh, U, phi)
        with pytest.raises(RuntimeError, match="correct"):
            model.nut()

    def test_correct_and_nut(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 2] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = DynamicLagrangianSGS(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_k_sgs(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = DynamicLagrangianSGS(mesh, U, phi)
        model.correct()
        k_sgs = model.k_sgs()
        assert k_sgs.shape == (mesh.n_cells,)
        assert (k_sgs >= 0).all()

    def test_repr_skip(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = DynamicLagrangianSGS(mesh, U, phi)
        assert "DynamicLagrangianSGS" in repr(model)


# ======================================================================
# AMDModel
# ======================================================================


class TestAMDModel:
    """Tests for AMD LES model."""

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = AMDModel(mesh, U, phi)
        assert model._mesh is mesh

    def test_C_AMD_property(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = AMDModel(mesh, U, phi, C_AMD=0.5)
        assert model.C_AMD == 0.5

    def test_nut_before_correct_raises(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = AMDModel(mesh, U, phi)
        with pytest.raises(RuntimeError, match="correct"):
            model.nut()

    def test_correct_and_nut(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 2] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = AMDModel(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_k_sgs(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = AMDModel(mesh, U, phi)
        model.correct()
        k_sgs = model.k_sgs()
        assert k_sgs.shape == (mesh.n_cells,)
        assert (k_sgs >= 0).all()

    def test_repr_skip(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = AMDModel(mesh, U, phi)
        assert "AMDModel" in repr(model)


# ======================================================================
# EnhancedWallTreatment4
# ======================================================================


class TestEnhancedWallTreatment4:
    """Tests for enhanced wall treatment v4."""

    def test_creation(self):
        wt = EnhancedWallTreatment4(nu=1.5e-5)
        assert wt.nu == 1.5e-5

    def test_inherits_from_enhanced3(self):
        wt = EnhancedWallTreatment4(nu=1.5e-5)
        assert isinstance(wt, EnhancedWallTreatment3)

    def test_Le_property(self):
        wt = EnhancedWallTreatment4(nu=1.5e-5, Le=1.2)
        assert wt.Le == 1.2

    def test_van_driest_A_property(self):
        wt = EnhancedWallTreatment4(nu=1.5e-5, van_driest_A=25.0)
        assert wt.van_driest_A == 25.0

    def test_van_driest_length(self):
        wt = EnhancedWallTreatment4(nu=1.5e-5)
        y_plus = torch.tensor([1.0, 10.0, 100.0], dtype=torch.float64)
        l_m = wt.van_driest_length(y_plus)
        assert l_m.shape == (3,)
        assert (l_m >= 0).all()
        # At y+ = 0, l_m = 0
        assert l_m[0] > 0  # y+ = 1 should be non-zero

    def test_compute_nut_skip(self):
        wt = EnhancedWallTreatment4(nu=1.5e-5)
        k = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64)
        y = torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64)
        nut = wt.compute_nut(k, y)
        assert nut.shape == (3,)
        assert (nut >= 0).all()

    def test_compute_htc_with_lewis(self):
        wt = EnhancedWallTreatment4(nu=1.5e-5, Pr=0.71, Pr_t=0.85, Le=1.0)
        k = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64)
        y = torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64)
        T_fluid = torch.tensor([300.0, 310.0, 320.0], dtype=torch.float64)
        T_wall = torch.tensor([350.0, 350.0, 350.0], dtype=torch.float64)
        htc = wt.compute_htc(k, y, T_fluid, T_wall, rho=1.2, Cp=1005.0)
        assert htc.shape == (3,)
        assert (htc >= 0).all()

    def test_repr_skip(self):
        wt = EnhancedWallTreatment4(nu=1.5e-5, Pr=0.71, Le=1.2)
        r = repr(wt)
        assert "EnhancedWallTreatment4" in r
        assert "Le=1.2" in r


# ======================================================================
# CompressibleWallTreatment
# ======================================================================


class TestCompressibleWallTreatment:
    """Tests for compressible wall treatment."""

    def test_creation(self):
        wt = CompressibleWallTreatment(nu=1.5e-5)
        assert wt.nu == 1.5e-5

    def test_van_driest_damping_skip(self):
        wt = CompressibleWallTreatment(nu=1.5e-5)
        y_plus = torch.tensor([0.1, 1.0, 10.0, 100.0], dtype=torch.float64)
        f_VD = wt.van_driest_damping(y_plus)
        assert f_VD.shape == (4,)
        assert (f_VD >= 0).all()
        assert (f_VD <= 1.0).all()
        # At y+ = 0, damping is 0
        assert f_VD[0] < 0.1

    def test_compute_nut_skip(self):
        wt = CompressibleWallTreatment(nu=1.5e-5)
        k = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64)
        y = torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64)
        nut = wt.compute_nut(k, y)
        assert nut.shape == (3,)
        assert (nut >= 0).all()

    def test_compute_omega_skip(self):
        wt = CompressibleWallTreatment(nu=1.5e-5)
        k = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64)
        y = torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64)
        omega = wt.compute_omega(k, y)
        assert omega.shape == (3,)
        assert (omega > 0).all()

    def test_repr_skip(self):
        wt = CompressibleWallTreatment(nu=1.5e-5, van_driest_A=25.0)
        assert "CompressibleWallTreatment" in repr(wt)
