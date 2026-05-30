"""Tests for enhanced turbulence models (Phase 17)."""

import pytest
import torch

from tests.unit.turbulence.conftest import make_fv_mesh


# ---- k-epsilon v7 ----

class TestKEpsilonEnhanced7:
    def test_import(self):
        from pyfoam.turbulence.k_epsilon_enhanced_7 import KEpsilonEnhanced7Model, KEpsilonEnhanced7Constants
        assert KEpsilonEnhanced7Model is not None

    def test_inherits(self):
        from pyfoam.turbulence.k_epsilon_enhanced_6 import KEpsilonEnhanced6Model
        from pyfoam.turbulence.k_epsilon_enhanced_7 import KEpsilonEnhanced7Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced7Model(mesh, U, phi)
        assert isinstance(model, KEpsilonEnhanced6Model)

    def test_correct(self):
        from pyfoam.turbulence.k_epsilon_enhanced_7 import KEpsilonEnhanced7Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KEpsilonEnhanced7Model(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()


# ---- k-omega v7 ----

class TestKOmegaEnhanced7:
    def test_import(self):
        from pyfoam.turbulence.k_omega_enhanced_7 import KOmegaEnhanced7Model
        assert KOmegaEnhanced7Model is not None

    def test_correct(self):
        from pyfoam.turbulence.k_omega_enhanced_7 import KOmegaEnhanced7Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaEnhanced7Model(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()


# ---- SST v7 ----

class TestKOmegaSSTEnhanced7:
    def test_import(self):
        from pyfoam.turbulence.k_omega_sst_enhanced_7 import KOmegaSSTEnhanced7Model
        assert KOmegaSSTEnhanced7Model is not None

    def test_correct(self):
        from pyfoam.turbulence.k_omega_sst_enhanced_7 import KOmegaSSTEnhanced7Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = KOmegaSSTEnhanced7Model(mesh, U, phi, enable_sas=False)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()


# ---- SA v7 ----

class TestSpalartAllmarasEnhanced7:
    def test_import(self):
        from pyfoam.turbulence.spalart_allmaras_enhanced_7 import SpalartAllmarasEnhanced7Model
        assert SpalartAllmarasEnhanced7Model is not None

    def test_correct(self):
        from pyfoam.turbulence.spalart_allmaras_enhanced_7 import SpalartAllmarasEnhanced7Model
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = SpalartAllmarasEnhanced7Model(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()


# ---- LES v7 ----

class TestLESModelEnhanced7:
    def test_amd_import(self):
        from pyfoam.turbulence.les_model_enhanced_7 import AnisotropicMDModel
        assert AnisotropicMDModel is not None

    def test_sf_import(self):
        from pyfoam.turbulence.les_model_enhanced_7 import StructureFunctionSGS
        assert StructureFunctionSGS is not None

    def test_amd_correct(self):
        from pyfoam.turbulence.les_model_enhanced_7 import AnisotropicMDModel
        mesh = make_fv_mesh()
        U = torch.zeros(mesh.n_cells, 3, dtype=torch.float64)
        phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
        model = AnisotropicMDModel(mesh, U, phi)
        model.correct()
        nut = model.nut()
        assert nut.shape == (mesh.n_cells,)
        assert (nut >= 0).all()


# ---- Wall treatment v6 ----

class TestWallTreatmentEnhanced6:
    def test_import(self):
        from pyfoam.turbulence.wall_treatment_enhanced_6 import EnhancedWallTreatment6, RoughnessCorrelation
        assert EnhancedWallTreatment6 is not None

    def test_species_wall_transfer(self):
        from pyfoam.turbulence.wall_treatment_enhanced_6 import EnhancedWallTreatment6
        wt = EnhancedWallTreatment6(nu=1.5e-5, n_species=3, Sc=0.7)
        k = wt.species_wall_transfer_coefficient(15.0, 0.05)
        assert k >= 0

    def test_colebrook_white(self):
        from pyfoam.turbulence.wall_treatment_enhanced_6 import EnhancedWallTreatment6
        wt = EnhancedWallTreatment6(nu=1.5e-5, ks=1e-4)
        f = wt.colebrook_white_friction(1e5)
        assert f > 0

    def test_roughness_correlation(self):
        from pyfoam.turbulence.wall_treatment_enhanced_6 import RoughnessCorrelation
        rc = RoughnessCorrelation(ks=1e-4)
        du = rc.delta_u_rough(50.0)
        assert du > 0

    def test_repr(self):
        from pyfoam.turbulence.wall_treatment_enhanced_6 import EnhancedWallTreatment6
        wt = EnhancedWallTreatment6(nu=1.5e-5, n_species=2)
        r = repr(wt)
        assert "EnhancedWallTreatment6" in r
