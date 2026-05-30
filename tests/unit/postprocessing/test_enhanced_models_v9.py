"""Tests for enhanced postprocessing models (Phase 19)."""

import pytest
import torch


# ---- FieldMinMaxEnhanced10 ----

class TestFieldMinMaxEnhanced10:
    def test_import(self):
        from pyfoam.postprocessing.field_min_max_enhanced_10 import FieldMinMaxEnhanced10
        assert FieldMinMaxEnhanced10 is not None

    def test_inherits(self):
        from pyfoam.postprocessing.field_min_max_enhanced_9 import FieldMinMaxEnhanced9
        from pyfoam.postprocessing.field_min_max_enhanced_10 import FieldMinMaxEnhanced10
        fo = FieldMinMaxEnhanced10("test", {"field": "p"})
        assert isinstance(fo, FieldMinMaxEnhanced9)

    def test_default_params(self):
        from pyfoam.postprocessing.field_min_max_enhanced_10 import FieldMinMaxEnhanced10
        fo = FieldMinMaxEnhanced10("test", {"field": "p"})
        assert fo._topo_extremes is False
        assert fo._multi_field is False

    def test_custom_params(self):
        from pyfoam.postprocessing.field_min_max_enhanced_10 import FieldMinMaxEnhanced10
        fo = FieldMinMaxEnhanced10("test", {
            "field": "p",
            "topologicalExtremes": True,
            "multiFieldCoupling": True,
        })
        assert fo._topo_extremes is True
        assert fo._multi_field is True

    def test_dataclass(self):
        from pyfoam.postprocessing.field_min_max_enhanced_10 import TopologicalExtreme, MultiFieldCorrelation
        te = TopologicalExtreme(field_name="p", is_local_max=True, local_max_count=3, global_max_idx=10)
        assert te.is_local_max is True
        assert te.local_max_count == 3
        mc = MultiFieldCorrelation(fields=["p", "T"], max_correlation=0.8)
        assert mc.max_correlation == 0.8


# ---- ProbesEnhanced10 ----

class TestProbesEnhanced10:
    def test_import(self):
        from pyfoam.postprocessing.probes_enhanced_10 import ProbesEnhanced10
        assert ProbesEnhanced10 is not None

    def test_inherits(self):
        from pyfoam.postprocessing.probes_enhanced_9 import ProbesEnhanced9
        from pyfoam.postprocessing.probes_enhanced_10 import ProbesEnhanced10
        p = ProbesEnhanced10("test", {})
        assert isinstance(p, ProbesEnhanced9)

    def test_default_params(self):
        from pyfoam.postprocessing.probes_enhanced_10 import ProbesEnhanced10
        p = ProbesEnhanced10("test", {})
        assert p._probe_corr is False
        assert p._spectral_entropy is False

    def test_custom_params(self):
        from pyfoam.postprocessing.probes_enhanced_10 import ProbesEnhanced10
        p = ProbesEnhanced10("test", {
            "probeCorrelation": True,
            "spectralEntropy": True,
        })
        assert p._probe_corr is True
        assert p._spectral_entropy is True

    def test_dataclass(self):
        from pyfoam.postprocessing.probes_enhanced_10 import ProbeCorrelation, SpectralEntropy
        pc = ProbeCorrelation(field_name="p", max_correlation=0.9, mean_correlation=0.5, n_pairs=10)
        assert pc.n_pairs == 10
        se = SpectralEntropy(field_name="p", entropy=1.5, peak_frequency=0.1)
        assert se.entropy == 1.5


# ---- ForcesEnhanced9 ----

class TestForcesEnhanced9:
    def test_import(self):
        from pyfoam.postprocessing.forces_enhanced_9 import ForcesEnhanced9
        assert ForcesEnhanced9 is not None

    def test_inherits(self):
        from pyfoam.postprocessing.forces_enhanced_8 import ForcesEnhanced8
        from pyfoam.postprocessing.forces_enhanced_9 import ForcesEnhanced9
        f = ForcesEnhanced9("test", {"patches": ["wall"]})
        assert isinstance(f, ForcesEnhanced8)

    def test_default_params(self):
        from pyfoam.postprocessing.forces_enhanced_9 import ForcesEnhanced9
        f = ForcesEnhanced9("test", {"patches": ["wall"]})
        assert f._load_history is False
        assert f._fatigue_damage is False

    def test_custom_params(self):
        from pyfoam.postprocessing.forces_enhanced_9 import ForcesEnhanced9
        f = ForcesEnhanced9("test", {
            "patches": ["wall"],
            "loadHistory": True,
            "fatigueDamage": True,
            "SN_exponent": 5.0,
        })
        assert f._load_history is True
        assert f._sn_exp == 5.0

    def test_dataclass(self):
        from pyfoam.postprocessing.forces_enhanced_9 import LoadCycle, FatigueDamage
        lc = LoadCycle(amplitude=10.0, mean_value=5.0, is_full=True)
        assert lc.amplitude == 10.0
        fd = FatigueDamage(damage_sum=0.01, n_cycles=100, damage_rate=0.001)
        assert fd.n_cycles == 100


# ---- WallShearStressEnhanced9 ----

class TestWallShearStressEnhanced9:
    def test_import(self):
        from pyfoam.postprocessing.wall_shear_stress_enhanced_9 import WallShearStressEnhanced9
        assert WallShearStressEnhanced9 is not None

    def test_inherits(self):
        from pyfoam.postprocessing.wall_shear_stress_enhanced_8 import WallShearStressEnhanced8
        from pyfoam.postprocessing.wall_shear_stress_enhanced_9 import WallShearStressEnhanced9
        w = WallShearStressEnhanced9("test", {"patches": ["wall"]})
        assert isinstance(w, WallShearStressEnhanced8)

    def test_default_params(self):
        from pyfoam.postprocessing.wall_shear_stress_enhanced_9 import WallShearStressEnhanced9
        w = WallShearStressEnhanced9("test", {"patches": ["wall"]})
        assert w._time_avg_topo is False
        assert w._streak_dynamics is False

    def test_dataclass(self):
        from pyfoam.postprocessing.wall_shear_stress_enhanced_9 import AveragedTopology, StreakDynamics
        at = AveragedTopology(patch_name="wall", mean_separation_fraction=0.05, topology_stability=0.8)
        assert at.topology_stability == 0.8
        sd = StreakDynamics(patch_name="wall", current_spacing=100.0, spacing_trend=-0.5)
        assert sd.current_spacing == 100.0


# ---- YPlusEnhanced10 ----

class TestYPlusEnhanced10:
    def test_import(self):
        from pyfoam.postprocessing.y_plus_enhanced_10 import YPlusEnhanced10
        assert YPlusEnhanced10 is not None

    def test_inherits(self):
        from pyfoam.postprocessing.y_plus_enhanced_9 import YPlusEnhanced9
        from pyfoam.postprocessing.y_plus_enhanced_10 import YPlusEnhanced10
        y = YPlusEnhanced10("test", {"rho": 1.0, "mu": 1e-5})
        assert isinstance(y, YPlusEnhanced9)

    def test_default_params(self):
        from pyfoam.postprocessing.y_plus_enhanced_10 import YPlusEnhanced10
        y = YPlusEnhanced10("test", {"rho": 1.0, "mu": 1e-5})
        assert y._wf_consistency is False
        assert y._mesh_convergence is False

    def test_custom_params(self):
        from pyfoam.postprocessing.y_plus_enhanced_10 import YPlusEnhanced10
        y = YPlusEnhanced10("test", {
            "rho": 1.0, "mu": 1e-5,
            "wallFunctionConsistency": True,
            "meshConvergence": True,
            "wallFunctionType": "lowRe",
        })
        assert y._wf_consistency is True
        assert y._wf_type == "lowRe"

    def test_dataclass(self):
        from pyfoam.postprocessing.y_plus_enhanced_10 import WallFunctionConsistency, MeshConvergenceIndicator
        wfc = WallFunctionConsistency(patch_name="wall", y_plus_mean=15.0, is_consistent=True, consistency_score=0.9)
        assert wfc.is_consistent is True
        mci = MeshConvergenceIndicator(patch_name="wall", y_plus_uniformity=0.85, convergence_level="good")
        assert mci.convergence_level == "good"
