"""Tests for enhanced turbulence models (Phase 20)."""

import pytest
import torch

from tests.unit.turbulence.conftest import make_fv_mesh


# ---- k-epsilon v10 ----

class TestKEpsilonEnhanced10:
    def test_import(self):
        from pyfoam.turbulence.k_epsilon_enhanced_10 import KEpsilonEnhanced10Model, KEpsilonEnhanced10Constants
        assert KEpsilonEnhanced10Model is not None

    def test_inherits(self):
        from pyfoam.turbulence.k_epsilon_enhanced_9 import KEpsilonEnhanced9Model
        from pyfoam.turbulence.k_epsilon_enhanced_10 import KEpsilonEnhanced10Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced10Model(mesh, U, phi)
        assert isinstance(model, KEpsilonEnhanced9Model)

    def test_correct(self):
        from pyfoam.turbulence.k_epsilon_enhanced_10 import KEpsilonEnhanced10Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced10Model(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_length_scale_limit(self):
        from pyfoam.turbulence.k_epsilon_enhanced_10 import KEpsilonEnhanced10Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced10Model(mesh, U, phi, enable_length_limiter=True)
        P_k = torch.ones(mesh.n_cells, dtype=torch.float64) * 1e-3
        P_lim = model.length_scale_limit(P_k)
        assert P_lim.shape == P_k.shape

    def test_pressure_strain(self):
        from pyfoam.turbulence.k_epsilon_enhanced_10 import KEpsilonEnhanced10Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced10Model(mesh, U, phi, enable_pressure_strain=True)
        P_k = torch.ones(mesh.n_cells, dtype=torch.float64) * 1e-3
        P_corr = model.pressure_strain_correction(P_k)
        assert P_corr.shape == P_k.shape


# ---- k-omega v10 ----

class TestKOmegaEnhanced10:
    def test_import(self):
        from pyfoam.turbulence.k_omega_enhanced_10 import KOmegaEnhanced10Model
        assert KOmegaEnhanced10Model is not None

    def test_correct(self):
        from pyfoam.turbulence.k_omega_enhanced_10 import KOmegaEnhanced10Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced10Model(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_low_re_damping(self):
        from pyfoam.turbulence.k_omega_enhanced_10 import KOmegaEnhanced10Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced10Model(mesh, U, phi, enable_low_re_damping=True)
        f = model.low_re_damping()
        assert f.shape == (mesh.n_cells,)
        assert (f >= 0).all()


# ---- SST v10 ----

class TestKOmegaSSTEnhanced10:
    def test_import(self):
        from pyfoam.turbulence.k_omega_sst_enhanced_10 import KOmegaSSTEnhanced10Model
        assert KOmegaSSTEnhanced10Model is not None

    def test_correct(self):
        from pyfoam.turbulence.k_omega_sst_enhanced_10 import KOmegaSSTEnhanced10Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced10Model(mesh, U, phi, enable_sas=False)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_production_limiter(self):
        from pyfoam.turbulence.k_omega_sst_enhanced_10 import KOmegaSSTEnhanced10Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced10Model(mesh, U, phi, enable_production_limit=True)
        P_k = torch.ones(mesh.n_cells, dtype=torch.float64) * 100.0
        P_lim = model.production_limiter(P_k)
        assert P_lim.shape == P_k.shape
        assert (P_lim <= P_k).all()

    def test_tke_budget(self):
        from pyfoam.turbulence.k_omega_sst_enhanced_10 import KOmegaSSTEnhanced10Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced10Model(mesh, U, phi, enable_tke_budget=True)
        model.correct()
        budget = model.tke_budget()
        assert "production" in budget
        assert "dissipation" in budget


# ---- SA v10 ----

class TestSpalartAllmarasEnhanced10:
    def test_import(self):
        from pyfoam.turbulence.spalart_allmaras_enhanced_10 import SpalartAllmarasEnhanced10Model
        assert SpalartAllmarasEnhanced10Model is not None

    def test_correct(self):
        from pyfoam.turbulence.spalart_allmaras_enhanced_10 import SpalartAllmarasEnhanced10Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced10Model(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_rotation_correction(self):
        from pyfoam.turbulence.spalart_allmaras_enhanced_10 import SpalartAllmarasEnhanced10Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced10Model(mesh, U, phi, enable_rotation_correction=True)
        f_rot = model.rotation_correction()
        assert f_rot.shape == (mesh.n_cells,)
        assert (f_rot >= 1.0).all()

    def test_adaptive_ft2(self):
        from pyfoam.turbulence.spalart_allmaras_enhanced_10 import SpalartAllmarasEnhanced10Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced10Model(mesh, U, phi)
        ft2 = model.adaptive_ft2()
        assert ft2.shape == (mesh.n_cells,)


# ---- LES v10 ----

class TestLESModelEnhanced10:
    def test_localized_dynamic_import(self):
        from pyfoam.turbulence.les_model_enhanced_10 import LocalizedDynamicSGS
        assert LocalizedDynamicSGS is not None

    def test_tensor_diffusivity_import(self):
        from pyfoam.turbulence.les_model_enhanced_10 import TensorDiffusivitySGS
        assert TensorDiffusivitySGS is not None

    def test_wall_adaptive_import(self):
        from pyfoam.turbulence.les_model_enhanced_10 import WallAdaptiveBlendedSGS
        assert WallAdaptiveBlendedSGS is not None

    def test_localized_dynamic_correct(self):
        from pyfoam.turbulence.les_model_enhanced_10 import LocalizedDynamicSGS
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = LocalizedDynamicSGS(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()

    def test_wall_adaptive_correct(self):
        from pyfoam.turbulence.les_model_enhanced_10 import WallAdaptiveBlendedSGS
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = WallAdaptiveBlendedSGS(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)


# ---- Wall treatment v9 ----

class TestWallTreatmentEnhanced9:
    def test_import(self):
        from pyfoam.turbulence.wall_treatment_enhanced_9 import EnhancedWallTreatment9, ThermalWallFunction
        assert EnhancedWallTreatment9 is not None
        assert ThermalWallFunction is not None

    def test_thermal_wall_temperature(self):
        from pyfoam.turbulence.wall_treatment_enhanced_9 import EnhancedWallTreatment9
        wt = EnhancedWallTreatment9(nu=1.5e-5, Pr=0.71)
        result = wt.thermal_wall_temperature(300.0, 310.0, 15.0, 0.05)
        assert "heat_flux" in result
        assert "Nusselt" in result

    def test_species_wall_flux(self):
        from pyfoam.turbulence.wall_treatment_enhanced_9 import EnhancedWallTreatment9
        wt = EnhancedWallTreatment9(nu=1.5e-5, n_species=2, Sc_w=0.7)
        result = wt.species_wall_flux(0.21, 0.23, 15.0, 0.05)
        assert "flux" in result
        assert "Sherwood" in result

    def test_adaptive_y_plus_target(self):
        from pyfoam.turbulence.wall_treatment_enhanced_9 import EnhancedWallTreatment9
        wt = EnhancedWallTreatment9(nu=1.5e-5, adaptive_y_plus=True)
        assert wt.adaptive_y_plus_target(50.0) == 1.0
        assert wt.adaptive_y_plus_target(200.0) == 5.0
        assert wt.adaptive_y_plus_target(1000.0) == 30.0

    def test_thermal_wall_function_standalone(self):
        from pyfoam.turbulence.wall_treatment_enhanced_9 import ThermalWallFunction
        twf = ThermalWallFunction(Pr=0.71)
        T_plus = twf.T_plus(15.0)
        assert T_plus > 0
        T_plus_vis = twf.T_plus(5.0)
        assert T_plus_vis > 0

    def test_repr(self):
        from pyfoam.turbulence.wall_treatment_enhanced_9 import EnhancedWallTreatment9
        wt = EnhancedWallTreatment9(nu=1.5e-5, n_species=3)
        assert "EnhancedWallTreatment9" in repr(wt)
