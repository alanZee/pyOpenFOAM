"""Tests for enhanced postprocessing models (Phase 18)."""

import pytest
import torch


# ---- FieldMinMaxEnhanced9 ----

class TestFieldMinMaxEnhanced9:
    def test_import(self):
        from pyfoam.postprocessing.field_min_max_enhanced_9 import FieldMinMaxEnhanced9
        assert FieldMinMaxEnhanced9 is not None

    def test_inherits(self):
        from pyfoam.postprocessing.field_min_max_enhanced_8 import FieldMinMaxEnhanced8
        from pyfoam.postprocessing.field_min_max_enhanced_9 import FieldMinMaxEnhanced9
        fo = FieldMinMaxEnhanced9("test", {"field": "p"})
        assert isinstance(fo, FieldMinMaxEnhanced8)

    def test_default_params(self):
        from pyfoam.postprocessing.field_min_max_enhanced_9 import FieldMinMaxEnhanced9
        fo = FieldMinMaxEnhanced9("test", {"field": "p"})
        assert fo._spatial_clustering is False
        assert fo._predictive_thresholds is False
        assert fo._spc_limits is False

    def test_custom_params(self):
        from pyfoam.postprocessing.field_min_max_enhanced_9 import FieldMinMaxEnhanced9
        fo = FieldMinMaxEnhanced9("test", {
            "field": "p",
            "spatialClustering": True,
            "spcLimits": True,
            "spcWindowSize": 30,
        })
        assert fo._spatial_clustering is True
        assert fo._spc_limits is True
        assert fo._spc_window == 30

    def test_dataclass(self):
        from pyfoam.postprocessing.field_min_max_enhanced_9 import SpatialCluster, SPCLimit
        sc = SpatialCluster(center_idx=5, n_members=10, mean_value=100.0, spatial_extent=0.1)
        assert sc.n_members == 10
        spc = SPCLimit(field_name="p", ucl=110.0, lcl=90.0, center_line=100.0, is_violated=False)
        assert spc.center_line == 100.0


# ---- ProbesEnhanced9 ----

class TestProbesEnhanced9:
    def test_import(self):
        from pyfoam.postprocessing.probes_enhanced_9 import ProbesEnhanced9
        assert ProbesEnhanced9 is not None

    def test_inherits(self):
        from pyfoam.postprocessing.probes_enhanced_8 import ProbesEnhanced8
        from pyfoam.postprocessing.probes_enhanced_9 import ProbesEnhanced9
        p = ProbesEnhanced9("test", {})
        assert isinstance(p, ProbesEnhanced8)

    def test_default_params(self):
        from pyfoam.postprocessing.probes_enhanced_9 import ProbesEnhanced9
        p = ProbesEnhanced9("test", {})
        assert p._signal_recon is False
        assert p._network_topology is False
        assert p._adaptive_sampling is False

    def test_custom_params(self):
        from pyfoam.postprocessing.probes_enhanced_9 import ProbesEnhanced9
        p = ProbesEnhanced9("test", {
            "signalReconstruction": True,
            "adaptiveSampling": True,
            "samplingRateMin": 2.0,
            "samplingRateMax": 50.0,
        })
        assert p._signal_recon is True
        assert p._rate_min == 2.0
        assert p._rate_max == 50.0

    def test_dataclass(self):
        from pyfoam.postprocessing.probes_enhanced_9 import ReconstructedSignal, NetworkTopology
        rs = ReconstructedSignal(field_name="p", n_original=100, n_reconstructed=5, reconstruction_error=0.05)
        assert rs.n_reconstructed == 5
        nt = NetworkTopology(n_probes=10, mean_spacing=0.05, max_spacing=0.15, coverage_ratio=0.8)
        assert nt.coverage_ratio == 0.8


# ---- ForcesEnhanced8 ----

class TestForcesEnhanced8:
    def test_import(self):
        from pyfoam.postprocessing.forces_enhanced_8 import ForcesEnhanced8
        assert ForcesEnhanced8 is not None

    def test_inherits(self):
        from pyfoam.postprocessing.forces_enhanced_7 import ForcesEnhanced7
        from pyfoam.postprocessing.forces_enhanced_8 import ForcesEnhanced8
        f = ForcesEnhanced8("test", {"patches": ["wall"]})
        assert isinstance(f, ForcesEnhanced7)

    def test_default_params(self):
        from pyfoam.postprocessing.forces_enhanced_8 import ForcesEnhanced8
        f = ForcesEnhanced8("test", {"patches": ["wall"]})
        assert f._pod_enabled is False
        assert f._freq_domain is False

    def test_custom_params(self):
        from pyfoam.postprocessing.forces_enhanced_8 import ForcesEnhanced8
        f = ForcesEnhanced8("test", {
            "patches": ["wall"],
            "podAnalysis": True,
            "frequencyDomain": True,
            "nPodModes": 8,
        })
        assert f._pod_enabled is True
        assert f._freq_domain is True
        assert f._n_pod_modes == 8

    def test_dataclass(self):
        from pyfoam.postprocessing.forces_enhanced_8 import PODMode, FrequencyDomainResult
        pm = PODMode(mode_idx=0, energy_fraction=0.8, modal_coefficient=1.5)
        assert pm.energy_fraction == 0.8
        fr = FrequencyDomainResult(n_frequencies=128, dominant_frequency=2.5, dominant_amplitude=0.1)
        assert fr.dominant_frequency == 2.5


# ---- WallShearStressEnhanced8 ----

class TestWallShearStressEnhanced8:
    def test_import(self):
        from pyfoam.postprocessing.wall_shear_stress_enhanced_8 import WallShearStressEnhanced8
        assert WallShearStressEnhanced8 is not None

    def test_inherits(self):
        from pyfoam.postprocessing.wall_shear_stress_enhanced_7 import WallShearStressEnhanced7
        from pyfoam.postprocessing.wall_shear_stress_enhanced_8 import WallShearStressEnhanced8
        w = WallShearStressEnhanced8("test", {"patches": ["wall"]})
        assert isinstance(w, WallShearStressEnhanced7)

    def test_default_params(self):
        from pyfoam.postprocessing.wall_shear_stress_enhanced_8 import WallShearStressEnhanced8
        w = WallShearStressEnhanced8("test", {"patches": ["wall"]})
        assert w._streak_spacing is False
        assert w._skin_friction_topo is False

    def test_dataclass(self):
        from pyfoam.postprocessing.wall_shear_stress_enhanced_8 import StreakSpacing, SkinFrictionTopology
        ss = StreakSpacing(patch_name="wall", mean_spacing=100.0, spacing_std=20.0)
        assert ss.mean_spacing == 100.0
        topo = SkinFrictionTopology(patch_name="wall", n_separation=5, n_attachment=3, separation_fraction=0.1)
        assert topo.n_separation == 5


# ---- YPlusEnhanced9 ----

class TestYPlusEnhanced9:
    def test_import(self):
        from pyfoam.postprocessing.y_plus_enhanced_9 import YPlusEnhanced9
        assert YPlusEnhanced9 is not None

    def test_inherits(self):
        from pyfoam.postprocessing.y_plus_enhanced_8 import YPlusEnhanced8
        from pyfoam.postprocessing.y_plus_enhanced_9 import YPlusEnhanced9
        y = YPlusEnhanced9("test", {"rho": 1.0, "mu": 1e-5})
        assert isinstance(y, YPlusEnhanced8)

    def test_default_params(self):
        from pyfoam.postprocessing.y_plus_enhanced_9 import YPlusEnhanced9
        y = YPlusEnhanced9("test", {"rho": 1.0, "mu": 1e-5})
        assert y._multi_patch is False
        assert y._tbl_classify is False
        assert y._cell_height is False

    def test_custom_params(self):
        from pyfoam.postprocessing.y_plus_enhanced_9 import YPlusEnhanced9
        y = YPlusEnhanced9("test", {
            "rho": 1.0, "mu": 1e-5,
            "multiPatchComparison": True,
            "tblClassification": True,
            "targetYPlus": 0.5,
        })
        assert y._multi_patch is True
        assert y._tbl_classify is True
        assert y._target_y_plus == 0.5

    def test_dataclass(self):
        from pyfoam.postprocessing.y_plus_enhanced_9 import PatchComparison, TBLClassification, CellHeightSuggestion
        pc = PatchComparison(time=1.0, best_patch="wall", worst_patch="inlet")
        assert pc.best_patch == "wall"
        tbl = TBLClassification(patch_name="wall", regime="turbulent", Re_theta=500.0)
        assert tbl.regime == "turbulent"
        ch = CellHeightSuggestion(target_y_plus=1.0, suggested_height=1e-5, height_ratio=0.5)
        assert ch.height_ratio == 0.5
