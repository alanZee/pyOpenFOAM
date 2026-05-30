"""Tests for enhanced turbulence models (Phase 19)."""

import pytest
import torch

from tests.unit.turbulence.conftest import make_fv_mesh


# ---- k-epsilon v9 ----

class TestKEpsilonEnhanced9:
    def test_import(self):
        from pyfoam.turbulence.k_epsilon_enhanced_9 import KEpsilonEnhanced9Model, KEpsilonEnhanced9Constants
        assert KEpsilonEnhanced9Model is not None

    def test_inherits(self):
        from pyfoam.turbulence.k_epsilon_enhanced_8 import KEpsilonEnhanced8Model
        from pyfoam.turbulence.k_epsilon_enhanced_9 import KEpsilonEnhanced9Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced9Model(mesh, U, phi)
        assert isinstance(model, KEpsilonEnhanced8Model)

    def test_correct(self):
        from pyfoam.turbulence.k_epsilon_enhanced_9 import KEpsilonEnhanced9Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced9Model(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_asm_anisotropy(self):
        from pyfoam.turbulence.k_epsilon_enhanced_9 import KEpsilonEnhanced9Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced9Model(mesh, U, phi, enable_asm=True)
        model.correct()
        b = model.asm_anisotropy()
        assert b.shape == (mesh.n_cells, 3, 3)

    def test_transport_budget(self):
        from pyfoam.turbulence.k_epsilon_enhanced_9 import KEpsilonEnhanced9Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced9Model(mesh, U, phi, enable_transport_diag=True)
        model.correct()
        budget = model.transport_budget()
        assert "production" in budget
        assert "dissipation" in budget


# ---- k-omega v9 ----

class TestKOmegaEnhanced9:
    def test_import(self):
        from pyfoam.turbulence.k_omega_enhanced_9 import KOmegaEnhanced9Model
        assert KOmegaEnhanced9Model is not None

    def test_correct(self):
        from pyfoam.turbulence.k_omega_enhanced_9 import KOmegaEnhanced9Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced9Model(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_variable_sigma(self):
        from pyfoam.turbulence.k_omega_enhanced_9 import KOmegaEnhanced9Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced9Model(mesh, U, phi, enable_variable_sigma=True)
        sigma = model.variable_sigma_k()
        assert sigma.shape == (mesh.n_cells,)


# ---- SST v9 ----

class TestKOmegaSSTEnhanced9:
    def test_import(self):
        from pyfoam.turbulence.k_omega_sst_enhanced_9 import KOmegaSSTEnhanced9Model
        assert KOmegaSSTEnhanced9Model is not None

    def test_correct(self):
        from pyfoam.turbulence.k_omega_sst_enhanced_9 import KOmegaSSTEnhanced9Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced9Model(mesh, U, phi, enable_sas=False)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_blending_diagnostics(self):
        from pyfoam.turbulence.k_omega_sst_enhanced_9 import KOmegaSSTEnhanced9Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced9Model(mesh, U, phi, enable_sas=False)
        model.correct()
        diag = model.blending_diagnostics()
        assert "F1_mean" in diag


# ---- SA v9 ----

class TestSpalartAllmarasEnhanced9:
    def test_import(self):
        from pyfoam.turbulence.spalart_allmaras_enhanced_9 import SpalartAllmarasEnhanced9Model
        assert SpalartAllmarasEnhanced9Model is not None

    def test_correct(self):
        from pyfoam.turbulence.spalart_allmaras_enhanced_9 import SpalartAllmarasEnhanced9Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced9Model(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_stress_limiter(self):
        from pyfoam.turbulence.spalart_allmaras_enhanced_9 import SpalartAllmarasEnhanced9Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced9Model(mesh, U, phi)
        sl = model.stress_limiter()
        assert sl.shape == (mesh.n_cells,)
        assert (sl >= 0).all()


# ---- LES v9 ----

class TestLESModelEnhanced9:
    def test_amd_v2_import(self):
        from pyfoam.turbulence.les_model_enhanced_9 import AnisotropicMDv2
        assert AnisotropicMDv2 is not None

    def test_svv_import(self):
        from pyfoam.turbulence.les_model_enhanced_9 import SpectralVanishingViscosity
        assert SpectralVanishingViscosity is not None

    def test_hybrid_import(self):
        from pyfoam.turbulence.les_model_enhanced_9 import HybridSGS
        assert HybridSGS is not None

    def test_amd_v2_correct(self):
        from pyfoam.turbulence.les_model_enhanced_9 import AnisotropicMDv2
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = AnisotropicMDv2(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_hybrid_correct(self):
        from pyfoam.turbulence.les_model_enhanced_9 import HybridSGS
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = HybridSGS(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()


# ---- Wall treatment v8 ----

class TestWallTreatmentEnhanced8:
    def test_import(self):
        from pyfoam.turbulence.wall_treatment_enhanced_8 import EnhancedWallTreatment8, ConjugateWallFunction
        assert EnhancedWallTreatment8 is not None

    def test_conjugate_heat_flux(self):
        from pyfoam.turbulence.wall_treatment_enhanced_8 import EnhancedWallTreatment8
        wt = EnhancedWallTreatment8(nu=1.5e-5, k_solid=50.0)
        hf = wt.conjugate_heat_flux(300.0, 350.0, 0.001)
        assert "q" in hf
        assert hf["q"] > 0

    def test_multi_layer_u_plus(self):
        from pyfoam.turbulence.wall_treatment_enhanced_8 import EnhancedWallTreatment8
        wt = EnhancedWallTreatment8(nu=1.5e-5, n_layers=4)
        u = wt.multi_layer_u_plus(15.0)
        assert u > 0

    def test_y_plus_uncertain(self):
        from pyfoam.turbulence.wall_treatment_enhanced_8 import EnhancedWallTreatment8
        wt = EnhancedWallTreatment8(nu=1.5e-5, y_plus_uncertainty=True)
        result = wt.y_plus_uncertain(0.001, 0.05, mesh_quality=0.9)
        assert "y_plus" in result
        assert "y_plus_low" in result
        assert "y_plus_high" in result

    def test_conjugate_standalone(self):
        from pyfoam.turbulence.wall_treatment_enhanced_8 import ConjugateWallFunction
        cw = ConjugateWallFunction(k_fluid=0.026, k_solid=50.0)
        T_interface = cw.interface_temperature(300.0, 350.0, 0.001, 0.01)
        assert 300.0 < T_interface < 350.0

    def test_repr(self):
        from pyfoam.turbulence.wall_treatment_enhanced_8 import EnhancedWallTreatment8
        wt = EnhancedWallTreatment8(nu=1.5e-5, n_layers=4)
        assert "EnhancedWallTreatment8" in repr(wt)
