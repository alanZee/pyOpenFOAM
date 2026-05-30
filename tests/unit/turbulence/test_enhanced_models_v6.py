"""Tests for enhanced turbulence models (Phase 16)."""

import pytest
import torch

from tests.unit.turbulence.conftest import make_fv_mesh


# ---- k-epsilon v6 ----

class TestKEpsilonEnhanced6:
    def test_import(self):
        from pyfoam.turbulence.k_epsilon_enhanced_6 import KEpsilonEnhanced6Model, KEpsilonEnhanced6Constants
        assert KEpsilonEnhanced6Model is not None

    def test_inherits(self):
        from pyfoam.turbulence.k_epsilon_enhanced_5 import KEpsilonEnhanced5Model
        from pyfoam.turbulence.k_epsilon_enhanced_6 import KEpsilonEnhanced6Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced6Model(mesh, U, phi)
        assert isinstance(model, KEpsilonEnhanced5Model)

    def test_correct(self):
        from pyfoam.turbulence.k_epsilon_enhanced_6 import KEpsilonEnhanced6Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced6Model(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()


# ---- k-omega v6 ----

class TestKOmegaEnhanced6:
    def test_import(self):
        from pyfoam.turbulence.k_omega_enhanced_6 import KOmegaEnhanced6Model
        assert KOmegaEnhanced6Model is not None

    def test_correct(self):
        from pyfoam.turbulence.k_omega_enhanced_6 import KOmegaEnhanced6Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced6Model(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()


# ---- SST v6 ----

class TestKOmegaSSTEnhanced6:
    def test_import(self):
        from pyfoam.turbulence.k_omega_sst_enhanced_6 import KOmegaSSTEnhanced6Model
        assert KOmegaSSTEnhanced6Model is not None

    def test_correct(self):
        from pyfoam.turbulence.k_omega_sst_enhanced_6 import KOmegaSSTEnhanced6Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced6Model(mesh, U, phi, enable_sas=False)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()


# ---- SA v6 ----

class TestSpalartAllmarasEnhanced6:
    def test_import(self):
        from pyfoam.turbulence.spalart_allmaras_enhanced_6 import SpalartAllmarasEnhanced6Model
        assert SpalartAllmarasEnhanced6Model is not None

    def test_correct(self):
        from pyfoam.turbulence.spalart_allmaras_enhanced_6 import SpalartAllmarasEnhanced6Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced6Model(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()


# ---- LES v6 ----

class TestLESModelEnhanced6:
    def test_wmle_import(self):
        from pyfoam.turbulence.les_model_enhanced_6 import WMLEModel
        assert WMLEModel is not None

    def test_tensor_visc_import(self):
        from pyfoam.turbulence.les_model_enhanced_6 import TensorViscositySGS
        assert TensorViscositySGS is not None

    def test_wmle_correct(self):
        from pyfoam.turbulence.les_model_enhanced_6 import WMLEModel
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = WMLEModel(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()


# ---- Wall treatment v5 ----

class TestWallTreatmentEnhanced5:
    def test_import(self):
        from pyfoam.turbulence.wall_treatment_enhanced_5 import EnhancedWallTreatment5, ConjugateHeatTransfer
        assert EnhancedWallTreatment5 is not None

    def test_conjugate_ht(self):
        from pyfoam.turbulence.wall_treatment_enhanced_5 import EnhancedWallTreatment5
        wt = EnhancedWallTreatment5(nu=1.5e-5, k_solid=50.0, solid_thickness=0.001)
        T_wall = wt.conjugate_wall_temperature(300.0, 100.0, 500.0)
        assert 300.0 < T_wall < 500.0

    def test_y_plus_tracking(self):
        from pyfoam.turbulence.wall_treatment_enhanced_5 import EnhancedWallTreatment5
        wt = EnhancedWallTreatment5()
        wt.update_y_plus_tracking(15.0)
        wt.update_y_plus_tracking(20.0)
        assert wt.y_plus_ema > 0

    def test_predict(self):
        from pyfoam.turbulence.wall_treatment_enhanced_5 import EnhancedWallTreatment5
        wt = EnhancedWallTreatment5()
        wt.update_y_plus_tracking(15.0)
        yp = wt.predict_y_plus()
        assert yp > 0

    def test_conjugate_htc(self):
        from pyfoam.turbulence.wall_treatment_enhanced_5 import ConjugateHeatTransfer
        cht = ConjugateHeatTransfer(k_fluid=0.026, k_solid=50.0, thickness=0.001)
        u = cht.overall_htc(100.0)
        assert u > 0

    def test_repr(self):
        from pyfoam.turbulence.wall_treatment_enhanced_5 import EnhancedWallTreatment5
        wt = EnhancedWallTreatment5()
        r = repr(wt)
        assert "EnhancedWallTreatment5" in r
