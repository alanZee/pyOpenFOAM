"""Tests for enhanced postprocessing models (Phase 20)."""

import pytest
import torch


# ---- FieldMinMaxEnhanced11 ----

class TestFieldMinMaxEnhanced11:
    def test_import(self):
        from pyfoam.postprocessing.field_min_max_enhanced_11 import FieldMinMaxEnhanced11
        assert FieldMinMaxEnhanced11 is not None

    def test_inherits(self):
        from pyfoam.postprocessing.field_min_max_enhanced_10 import FieldMinMaxEnhanced10
        from pyfoam.postprocessing.field_min_max_enhanced_11 import FieldMinMaxEnhanced11
        fo = FieldMinMaxEnhanced11("test", {"field": "p"})
        assert isinstance(fo, FieldMinMaxEnhanced10)

    def test_default_params(self):
        from pyfoam.postprocessing.field_min_max_enhanced_11 import FieldMinMaxEnhanced11
        fo = FieldMinMaxEnhanced11("test", {"field": "p"})
        assert fo._persistence is False
        assert fo._spatial_grad is False

    def test_custom_params(self):
        from pyfoam.postprocessing.field_min_max_enhanced_11 import FieldMinMaxEnhanced11
        fo = FieldMinMaxEnhanced11("test", {
            "field": "p",
            "persistenceTracking": True,
            "spatialGradient": True,
        })
        assert fo._persistence is True
        assert fo._spatial_grad is True

    def test_dataclass(self):
        from pyfoam.postprocessing.field_min_max_enhanced_11 import PersistenceExtreme, GradientAtExtreme
        pe = PersistenceExtreme(field_name="p", value=100.0, persistence_count=5)
        assert pe.persistence_count == 5
        ge = GradientAtExtreme(field_name="p", gradient_magnitude=50.0, is_steep=True)
        assert ge.is_steep is True


# ---- ProbesEnhanced11 ----

class TestProbesEnhanced11:
    def test_import(self):
        from pyfoam.postprocessing.probes_enhanced_11 import ProbesEnhanced11
        assert ProbesEnhanced11 is not None

    def test_inherits(self):
        from pyfoam.postprocessing.probes_enhanced_10 import ProbesEnhanced10
        from pyfoam.postprocessing.probes_enhanced_11 import ProbesEnhanced11
        p = ProbesEnhanced11("test", {})
        assert isinstance(p, ProbesEnhanced10)

    def test_default_params(self):
        from pyfoam.postprocessing.probes_enhanced_11 import ProbesEnhanced11
        p = ProbesEnhanced11("test", {})
        assert p._temporal_coherence is False
        assert p._probe_clustering is False

    def test_custom_params(self):
        from pyfoam.postprocessing.probes_enhanced_11 import ProbesEnhanced11
        p = ProbesEnhanced11("test", {
            "temporalCoherence": True,
            "probeClustering": True,
            "nClusters": 5,
        })
        assert p._temporal_coherence is True
        assert p._n_clusters == 5

    def test_dataclass(self):
        from pyfoam.postprocessing.probes_enhanced_11 import TemporalCoherence, ProbeCluster
        tc = TemporalCoherence(field_name="p", coherence=0.95, lag=0.01)
        assert tc.coherence == 0.95
        pc = ProbeCluster(cluster_id=1, probe_indices=[0, 2, 4], similarity=0.8)
        assert pc.cluster_id == 1


# ---- ForcesEnhanced10 ----

class TestForcesEnhanced10:
    def test_import(self):
        from pyfoam.postprocessing.forces_enhanced_10 import ForcesEnhanced10
        assert ForcesEnhanced10 is not None

    def test_inherits(self):
        from pyfoam.postprocessing.forces_enhanced_9 import ForcesEnhanced9
        from pyfoam.postprocessing.forces_enhanced_10 import ForcesEnhanced10
        f = ForcesEnhanced10("test", {"patches": ["wall"]})
        assert isinstance(f, ForcesEnhanced9)

    def test_default_params(self):
        from pyfoam.postprocessing.forces_enhanced_10 import ForcesEnhanced10
        f = ForcesEnhanced10("test", {"patches": ["wall"]})
        assert f._spectral_moments is False
        assert f._steady_detect is False

    def test_custom_params(self):
        from pyfoam.postprocessing.forces_enhanced_10 import ForcesEnhanced10
        f = ForcesEnhanced10("test", {
            "patches": ["wall"],
            "spectralMoments": True,
            "steadyStateDetection": True,
            "steadyTolerance": 0.005,
        })
        assert f._spectral_moments is True
        assert f._steady_tol == 0.005

    def test_dataclass(self):
        from pyfoam.postprocessing.forces_enhanced_10 import SpectralMoment, SteadyStateIndicator
        sm = SpectralMoment(m0=1.0, m2=0.5, m4=0.3, zero_crossing_freq=10.0)
        assert sm.zero_crossing_freq == 10.0
        ss = SteadyStateIndicator(is_steady=True, convergence_rate=0.001)
        assert ss.is_steady is True


# ---- WallShearStressEnhanced10 ----

class TestWallShearStressEnhanced10:
    def test_import(self):
        from pyfoam.postprocessing.wall_shear_stress_enhanced_10 import WallShearStressEnhanced10
        assert WallShearStressEnhanced10 is not None

    def test_inherits(self):
        from pyfoam.postprocessing.wall_shear_stress_enhanced_9 import WallShearStressEnhanced9
        from pyfoam.postprocessing.wall_shear_stress_enhanced_10 import WallShearStressEnhanced10
        w = WallShearStressEnhanced10("test", {"patches": ["wall"]})
        assert isinstance(w, WallShearStressEnhanced9)

    def test_default_params(self):
        from pyfoam.postprocessing.wall_shear_stress_enhanced_10 import WallShearStressEnhanced10
        w = WallShearStressEnhanced10("test", {"patches": ["wall"]})
        assert w._pressure_coupling is False
        assert w._reynolds_analogy is False

    def test_dataclass(self):
        from pyfoam.postprocessing.wall_shear_stress_enhanced_10 import ReynoldsAnalogyResult, FrictionDecomposition
        ra = ReynoldsAnalogyResult(patch_name="wall", Stanton_number=0.002, heat_flux=500.0)
        assert ra.Stanton_number == 0.002
        fd = FrictionDecomposition(patch_name="wall", tau_viscous=0.1, tau_turbulent=0.5, ratio=5.0)
        assert fd.ratio == 5.0


# ---- YPlusEnhanced11 ----

class TestYPlusEnhanced11:
    def test_import(self):
        from pyfoam.postprocessing.y_plus_enhanced_11 import YPlusEnhanced11
        assert YPlusEnhanced11 is not None

    def test_inherits(self):
        from pyfoam.postprocessing.y_plus_enhanced_10 import YPlusEnhanced10
        from pyfoam.postprocessing.y_plus_enhanced_11 import YPlusEnhanced11
        y = YPlusEnhanced11("test", {"rho": 1.0, "mu": 1e-5})
        assert isinstance(y, YPlusEnhanced10)

    def test_default_params(self):
        from pyfoam.postprocessing.y_plus_enhanced_11 import YPlusEnhanced11
        y = YPlusEnhanced11("test", {"rho": 1.0, "mu": 1e-5})
        assert y._wall_htc is False
        assert y._patch_compare is False

    def test_custom_params(self):
        from pyfoam.postprocessing.y_plus_enhanced_11 import YPlusEnhanced11
        y = YPlusEnhanced11("test", {
            "rho": 1.0, "mu": 1e-5,
            "wallHeatTransfer": True,
            "multiPatchComparison": True,
            "Pr": 0.71,
        })
        assert y._wall_htc is True
        assert y._Pr == 0.71

    def test_dataclass(self):
        from pyfoam.postprocessing.y_plus_enhanced_11 import WallHeatTransferCoeff, PatchRanking
        wh = WallHeatTransferCoeff(patch_name="wall", htc=100.0, Nusselt=50.0)
        assert wh.htc == 100.0
        pr = PatchRanking(best_patch="wall1", worst_patch="wall3")
        assert pr.best_patch == "wall1"
