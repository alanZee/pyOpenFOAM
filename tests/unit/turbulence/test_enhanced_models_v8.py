"""Tests for enhanced turbulence models (Phase 18)."""

import pytest
import torch

from tests.unit.turbulence.conftest import make_fv_mesh


# ---- k-epsilon v8 ----

class TestKEpsilonEnhanced8:
    def test_import(self):
        from pyfoam.turbulence.k_epsilon_enhanced_8 import KEpsilonEnhanced8Model, KEpsilonEnhanced8Constants
        assert KEpsilonEnhanced8Model is not None

    def test_inherits(self):
        from pyfoam.turbulence.k_epsilon_enhanced_7 import KEpsilonEnhanced7Model
        from pyfoam.turbulence.k_epsilon_enhanced_8 import KEpsilonEnhanced8Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced8Model(mesh, U, phi)
        assert isinstance(model, KEpsilonEnhanced7Model)

    def test_correct(self):
        from pyfoam.turbulence.k_epsilon_enhanced_8 import KEpsilonEnhanced8Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced8Model(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_turbulence_spectrum(self):
        from pyfoam.turbulence.k_epsilon_enhanced_8 import KEpsilonEnhanced8Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced8Model(mesh, U, phi)
        model.correct()
        spec = model.turbulence_spectrum(n_modes=5)
        assert "frequencies" in spec
        assert "energy" in spec
        assert spec["frequencies"].numel() == 5


# ---- k-omega v8 ----

class TestKOmegaEnhanced8:
    def test_import(self):
        from pyfoam.turbulence.k_omega_enhanced_8 import KOmegaEnhanced8Model
        assert KOmegaEnhanced8Model is not None

    def test_correct(self):
        from pyfoam.turbulence.k_omega_enhanced_8 import KOmegaEnhanced8Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced8Model(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()


# ---- SST v8 ----

class TestKOmegaSSTEnhanced8:
    def test_import(self):
        from pyfoam.turbulence.k_omega_sst_enhanced_8 import KOmegaSSTEnhanced8Model
        assert KOmegaSSTEnhanced8Model is not None

    def test_correct(self):
        from pyfoam.turbulence.k_omega_sst_enhanced_8 import KOmegaSSTEnhanced8Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced8Model(mesh, U, phi, enable_sas=False)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()


# ---- SA v8 ----

class TestSpalartAllmarasEnhanced8:
    def test_import(self):
        from pyfoam.turbulence.spalart_allmaras_enhanced_8 import SpalartAllmarasEnhanced8Model
        assert SpalartAllmarasEnhanced8Model is not None

    def test_correct(self):
        from pyfoam.turbulence.spalart_allmaras_enhanced_8 import SpalartAllmarasEnhanced8Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced8Model(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()


# ---- LES v8 ----

class TestLESModelEnhanced8:
    def test_mts_import(self):
        from pyfoam.turbulence.les_model_enhanced_8 import MixedTimeScaleSGS
        assert MixedTimeScaleSGS is not None

    def test_wawale_import(self):
        from pyfoam.turbulence.les_model_enhanced_8 import WallAdaptiveWALE
        assert WallAdaptiveWALE is not None

    def test_dynamic_tensor_import(self):
        from pyfoam.turbulence.les_model_enhanced_8 import DynamicTensorSGS
        assert DynamicTensorSGS is not None

    def test_mts_correct(self):
        from pyfoam.turbulence.les_model_enhanced_8 import MixedTimeScaleSGS
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = MixedTimeScaleSGS(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_wawale_correct(self):
        from pyfoam.turbulence.les_model_enhanced_8 import WallAdaptiveWALE
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = WallAdaptiveWALE(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()


# ---- Wall treatment v7 ----

class TestWallTreatmentEnhanced7:
    def test_import(self):
        from pyfoam.turbulence.wall_treatment_enhanced_7 import EnhancedWallTreatment7, HeatFluxDecomposition
        assert EnhancedWallTreatment7 is not None

    def test_blended_u_plus(self):
        from pyfoam.turbulence.wall_treatment_enhanced_7 import EnhancedWallTreatment7
        wt = EnhancedWallTreatment7(nu=1.5e-5)
        u_plus = wt.blended_u_plus(15.0)
        assert u_plus > 0

    def test_blended_nut(self):
        from pyfoam.turbulence.wall_treatment_enhanced_7 import EnhancedWallTreatment7
        wt = EnhancedWallTreatment7(nu=1.5e-5)
        nu_t = wt.blended_nut(15.0, 0.05)
        assert nu_t >= 0

    def test_heat_flux_decomposition(self):
        from pyfoam.turbulence.wall_treatment_enhanced_7 import EnhancedWallTreatment7
        wt = EnhancedWallTreatment7(nu=1.5e-5, heat_flux_decomposition=True)
        hf = wt.wall_heat_flux_decomposition(350.0, 300.0, 0.001)
        assert "q_convective" in hf
        assert "q_conductive" in hf
        assert hf["q_total"] > 0

    def test_adaptive_regime(self):
        from pyfoam.turbulence.wall_treatment_enhanced_7 import EnhancedWallTreatment7
        wt = EnhancedWallTreatment7(nu=1.5e-5, adaptive_switching=True)
        r1 = wt.adaptive_regime(2.0)
        assert r1 == "viscous"
        r2 = wt.adaptive_regime(15.0)
        assert r2 in ("buffer", "log_law")
        r3 = wt.adaptive_regime(100.0)
        assert r3 == "log_law"

    def test_heat_flux_standalone(self):
        from pyfoam.turbulence.wall_treatment_enhanced_7 import HeatFluxDecomposition
        hfd = HeatFluxDecomposition(Pr=0.71, Pr_t=0.85)
        k_t = hfd.turbulent_conductivity(1e-4, rho=1.0, Cp=1005.0)
        assert k_t > 0

    def test_repr(self):
        from pyfoam.turbulence.wall_treatment_enhanced_7 import EnhancedWallTreatment7
        wt = EnhancedWallTreatment7(nu=1.5e-5, n_species=2)
        assert "EnhancedWallTreatment7" in repr(wt)
