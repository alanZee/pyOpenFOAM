"""Tests for enhanced turbulence models (Phase 11).

Tests cover:
- KEpsilonEnhanced2Model (realizable k-epsilon v2)
- KOmegaEnhanced2Model (k-omega v2 with improved cross-diffusion)
- KOmegaSSTEnhanced2Model (SST v2 with F4 blending)
- SpalartAllmarasEnhanced2Model (SA v2 with QCR/SARC)
- DynamicLikeSmagorinskyModel
- ImprovedWALE2Model
- EnhancedWallTreatment
- ThreeLayerWallTreatment
"""

import pytest
import torch

from pyfoam.turbulence.turbulence_model import TurbulenceModel
from pyfoam.turbulence.k_epsilon_enhanced import KEpsilonEnhancedModel
from pyfoam.turbulence.k_epsilon_enhanced_2 import KEpsilonEnhanced2Model, KEpsilonEnhanced2Constants
from pyfoam.turbulence.k_omega_enhanced import KOmegaEnhancedModel
from pyfoam.turbulence.k_omega_enhanced_2 import KOmegaEnhanced2Model, KOmegaEnhanced2Constants
from pyfoam.turbulence.k_omega_sst_enhanced import KOmegaSSTEnhancedModel
from pyfoam.turbulence.k_omega_sst_enhanced_2 import KOmegaSSTEnhanced2Model, KOmegaSSTEnhanced2Constants
from pyfoam.turbulence.spalart_allmaras_enhanced import SpalartAllmarasEnhancedModel
from pyfoam.turbulence.spalart_allmaras_enhanced_2 import SpalartAllmarasEnhanced2Model, SpalartAllmarasEnhanced2Constants
from pyfoam.turbulence.les_model_enhanced_2 import DynamicLikeSmagorinskyModel, ImprovedWALE2Model
from pyfoam.turbulence.wall_treatment_enhanced import EnhancedWallTreatment, ThreeLayerWallTreatment

from tests.unit.turbulence.conftest import make_fv_mesh


# ======================================================================
# RTS Registration
# ======================================================================


class TestPhase11RTSRegistration:
    """Tests for RTS registration of Phase 11 models."""

    def test_realizable_ke_enhanced2_registered(self):
        assert "realizableKEEnhanced2" in TurbulenceModel.available_types()

    def test_komega_enhanced2_registered(self):
        assert "kOmegaEnhanced2" in TurbulenceModel.available_types()

    def test_komega_sst_enhanced2_registered(self):
        assert "kOmegaSST2003Enhanced2" in TurbulenceModel.available_types()

    def test_sa_enhanced2_registered(self):
        assert "SpalartAllmarasNoft2Enhanced" in TurbulenceModel.available_types()

    def test_create_ke_enhanced2(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = TurbulenceModel.create("realizableKEEnhanced2", mesh, U, phi)
        assert isinstance(model, KEpsilonEnhanced2Model)

    def test_create_komega_enhanced2(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = TurbulenceModel.create("kOmegaEnhanced2", mesh, U, phi)
        assert isinstance(model, KOmegaEnhanced2Model)

    def test_create_komega_sst_enhanced2(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = TurbulenceModel.create("kOmegaSST2003Enhanced2", mesh, U, phi)
        assert isinstance(model, KOmegaSSTEnhanced2Model)

    def test_create_sa_enhanced2(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = TurbulenceModel.create("SpalartAllmarasNoft2Enhanced", mesh, U, phi)
        assert isinstance(model, SpalartAllmarasEnhanced2Model)


# ======================================================================
# KEpsilonEnhanced2Model
# ======================================================================


class TestKEpsilonEnhanced2Model:
    """Tests for enhanced realizable k-epsilon v2."""

    def test_inherits_from_enhanced(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced2Model(mesh, U, phi)
        assert isinstance(model, KEpsilonEnhancedModel)

    def test_constants_default(self):
        C = KEpsilonEnhanced2Constants()
        assert C.C_mu_base == 0.09
        assert C.C_w == 0.3
        assert C.C_eta_2 == 5.0

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced2Model(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced2Model(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced2Model(mesh, U, phi)
        assert (model.nut() >= 0).all()

    def test_correct(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced2Model(mesh, U, phi)
        model.correct()
        assert model.k().shape == (mesh.n_cells,)
        assert model.epsilon().shape == (mesh.n_cells,)

    def test_repr(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced2Model(mesh, U, phi)
        assert "KEpsilonEnhanced2Model" in repr(model)


# ======================================================================
# KOmegaEnhanced2Model
# ======================================================================


class TestKOmegaEnhanced2Model:
    """Tests for enhanced k-omega v2."""

    def test_inherits_from_enhanced(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced2Model(mesh, U, phi)
        assert isinstance(model, KOmegaEnhancedModel)

    def test_constants_default(self):
        C = KOmegaEnhanced2Constants()
        assert C.alpha == pytest.approx(5.0 / 9.0)
        assert C.a1 == 0.31

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced2Model(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced2Model(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced2Model(mesh, U, phi)
        assert (model.nut() >= 0).all()

    def test_correct(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced2Model(mesh, U, phi)
        model.correct()
        assert model.k().shape == (mesh.n_cells,)
        assert model.omega().shape == (mesh.n_cells,)

    def test_repr(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced2Model(mesh, U, phi)
        assert "KOmegaEnhanced2Model" in repr(model)


# ======================================================================
# KOmegaSSTEnhanced2Model
# ======================================================================


class TestKOmegaSSTEnhanced2Model:
    """Tests for enhanced SST v2."""

    def test_inherits_from_enhanced(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced2Model(mesh, U, phi)
        assert isinstance(model, KOmegaSSTEnhancedModel)

    def test_constants_default(self):
        C = KOmegaSSTEnhanced2Constants()
        assert C.c1 == 10.0
        assert C.a1 == 0.31
        assert C.C_prod_lim == 10.0

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced2Model(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced2Model(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced2Model(mesh, U, phi)
        assert (model.nut() >= 0).all()

    def test_kato_launder_mode(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        C = KOmegaSSTEnhanced2Constants(kato_launder=True)
        model = KOmegaSSTEnhanced2Model(mesh, U, phi, constants=C)
        model.correct()
        assert model.k().shape == (mesh.n_cells,)

    def test_correct(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced2Model(mesh, U, phi)
        model.correct()
        assert model.k().shape == (mesh.n_cells,)
        assert model.omega().shape == (mesh.n_cells,)

    def test_repr(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced2Model(mesh, U, phi)
        assert "KOmegaSSTEnhanced2Model" in repr(model)


# ======================================================================
# SpalartAllmarasEnhanced2Model
# ======================================================================


class TestSpalartAllmarasEnhanced2Model:
    """Tests for enhanced SA v2."""

    def test_inherits_from_enhanced(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced2Model(mesh, U, phi)
        assert isinstance(model, SpalartAllmarasEnhancedModel)

    def test_constants_default(self):
        C = SpalartAllmarasEnhanced2Constants()
        assert C.sigma == pytest.approx(2.0 / 3.0)
        assert C.Cb1 == 0.1355
        assert C.C_rot1 == 1.0

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced2Model(mesh, U, phi)
        assert model.mesh is mesh

    def test_nut_shape(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced2Model(mesh, U, phi)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)

    def test_nut_positive(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced2Model(mesh, U, phi)
        assert (model.nut() >= 0).all()

    def test_qcr_mode(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced2Model(mesh, U, phi, enable_qcr=True)
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_correct(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced2Model(mesh, U, phi)
        model.correct()
        assert model.nuTilde_field.shape == (mesh.n_cells,)

    def test_repr(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced2Model(mesh, U, phi)
        assert "SpalartAllmarasEnhanced2Model" in repr(model)

    def test_repr_with_qcr(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced2Model(mesh, U, phi, enable_qcr=True)
        assert "QCR" in repr(model)


# ======================================================================
# DynamicLikeSmagorinskyModel
# ======================================================================


class TestDynamicLikeSmagorinskyModel:
    """Tests for dynamic-like Smagorinsky model."""

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = DynamicLikeSmagorinskyModel(mesh, U, phi)
        assert model._mesh is mesh

    def test_properties(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = DynamicLikeSmagorinskyModel(mesh, U, phi, Cs=0.15, dynamic_factor=0.3)
        assert model.Cs == 0.15
        assert model.dynamic_factor == 0.3

    def test_nut_before_correct_raises(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = DynamicLikeSmagorinskyModel(mesh, U, phi)
        with pytest.raises(RuntimeError, match="correct"):
            model.nut()

    def test_correct_and_nut(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 2] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = DynamicLikeSmagorinskyModel(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_k_sgs(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = DynamicLikeSmagorinskyModel(mesh, U, phi)
        model.correct()
        k_sgs = model.k_sgs()
        assert k_sgs.shape == (mesh.n_cells,)
        assert (k_sgs >= 0).all()

    def test_repr(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = DynamicLikeSmagorinskyModel(mesh, U, phi)
        assert "DynamicLikeSmagorinskyModel" in repr(model)


# ======================================================================
# ImprovedWALE2Model
# ======================================================================


class TestImprovedWALE2Model:
    """Tests for improved WALE v2 model."""

    def test_model_creation(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedWALE2Model(mesh, U, phi)
        assert model._mesh is mesh

    def test_properties(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedWALE2Model(mesh, U, phi, Cw=0.3)
        assert model.Cw == 0.3

    def test_nut_before_correct_raises(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedWALE2Model(mesh, U, phi)
        with pytest.raises(RuntimeError, match="correct"):
            model.nut()

    def test_correct_and_nut(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 2] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedWALE2Model(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_k_sgs(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedWALE2Model(mesh, U, phi)
        model.correct()
        k_sgs = model.k_sgs()
        assert k_sgs.shape == (mesh.n_cells,)
        assert (k_sgs >= 0).all()

    def test_tau_sgs(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedWALE2Model(mesh, U, phi)
        model.correct()
        tau = model.tau_sgs()
        assert tau.shape == (mesh.n_cells,)
        assert (tau > 0).all()

    def test_sd_tensor(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        U[:, 0] = 1.0
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedWALE2Model(mesh, U, phi)
        model.correct()
        assert model.Sd is not None
        assert model.Sd.shape == (mesh.n_cells, 3, 3)
        assert model.mag_Sd_sq is not None
        assert model.mag_Sd_sq.shape == (mesh.n_cells,)

    def test_repr(self):
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = ImprovedWALE2Model(mesh, U, phi)
        assert "ImprovedWALE2Model" in repr(model)


# ======================================================================
# EnhancedWallTreatment
# ======================================================================


class TestEnhancedWallTreatment:
    """Tests for enhanced wall treatment."""

    def test_creation(self):
        wt = EnhancedWallTreatment(nu=1.5e-5)
        assert wt.nu == 1.5e-5

    def test_compute_nut(self):
        wt = EnhancedWallTreatment(nu=1.5e-5)
        k = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64)
        y = torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64)
        nut = wt.compute_nut(k, y)
        assert nut.shape == (3,)
        assert (nut >= 0).all()

    def test_compute_k(self):
        wt = EnhancedWallTreatment(nu=1.5e-5)
        u_tau = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64)
        k = wt.compute_k(u_tau)
        assert k.shape == (3,)
        assert (k > 0).all()

    def test_compute_epsilon(self):
        wt = EnhancedWallTreatment(nu=1.5e-5)
        k = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64)
        y = torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64)
        eps = wt.compute_epsilon(k, y)
        assert eps.shape == (3,)
        assert (eps > 0).all()

    def test_compute_omega(self):
        wt = EnhancedWallTreatment(nu=1.5e-5)
        k = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64)
        y = torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64)
        omega = wt.compute_omega(k, y)
        assert omega.shape == (3,)
        assert (omega > 0).all()

    def test_blending_smoothness(self):
        """Blending factor should vary smoothly from 0 to 1."""
        wt = EnhancedWallTreatment(nu=1.5e-5, y_plus_low=5.0, y_plus_high=30.0)
        y_plus = torch.linspace(0, 100, 200, dtype=torch.float64)
        blend = wt._blending_factor(y_plus)
        # Should start near 0 and end near 1
        assert float(blend[0].item()) < 0.1
        assert float(blend[-1].item()) > 0.9

    def test_repr(self):
        wt = EnhancedWallTreatment(nu=1.5e-5)
        assert "EnhancedWallTreatment" in repr(wt)


# ======================================================================
# ThreeLayerWallTreatment
# ======================================================================


class TestThreeLayerWallTreatment:
    """Tests for three-layer wall treatment."""

    def test_creation(self):
        wt = ThreeLayerWallTreatment(nu=1.5e-5)
        assert wt.nu == 1.5e-5

    def test_compute_nut(self):
        wt = ThreeLayerWallTreatment(nu=1.5e-5)
        k = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64)
        y = torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64)
        nut = wt.compute_nut(k, y)
        assert nut.shape == (3,)
        assert (nut >= 0).all()

    def test_compute_k(self):
        wt = ThreeLayerWallTreatment(nu=1.5e-5)
        u_tau = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64)
        k = wt.compute_k(u_tau)
        assert k.shape == (3,)
        assert (k > 0).all()

    def test_compute_epsilon(self):
        wt = ThreeLayerWallTreatment(nu=1.5e-5)
        k = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64)
        y = torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64)
        eps = wt.compute_epsilon(k, y)
        assert eps.shape == (3,)
        assert (eps > 0).all()

    def test_compute_omega(self):
        wt = ThreeLayerWallTreatment(nu=1.5e-5)
        k = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float64)
        y = torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64)
        omega = wt.compute_omega(k, y)
        assert omega.shape == (3,)
        assert (omega > 0).all()

    def test_three_regimes(self):
        """Check that three regimes are properly identified."""
        wt = ThreeLayerWallTreatment(nu=1.5e-5)
        y_plus = torch.tensor([2.0, 15.0, 50.0], dtype=torch.float64)
        viscous, buffer, log_law = wt._regime(y_plus)
        assert viscous[0].item() is True
        assert buffer[1].item() is True
        assert log_law[2].item() is True

    def test_nut_viscous_sublayer(self):
        """In viscous sublayer (y+ < 5), nut should be zero."""
        wt = ThreeLayerWallTreatment(nu=1.5e-5)
        k = torch.tensor([0.1], dtype=torch.float64)
        y = torch.tensor([1e-6], dtype=torch.float64)  # Very small y -> small y+
        nut = wt.compute_nut(k, y)
        assert float(nut.item()) < 1e-15

    def test_repr(self):
        wt = ThreeLayerWallTreatment(nu=1.5e-5)
        assert "ThreeLayerWallTreatment" in repr(wt)
