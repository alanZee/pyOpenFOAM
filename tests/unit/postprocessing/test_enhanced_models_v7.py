"""Tests for enhanced postprocessing models (Phase 17)."""

import pytest
import torch

from tests.unit.postprocessing.conftest import fv_mesh, sample_fields


# ---- FieldMinMaxEnhanced8 ----

class TestFieldMinMaxEnhanced8:
    def test_import(self):
        from pyfoam.postprocessing.field_min_max_enhanced_8 import FieldMinMaxEnhanced8
        assert FieldMinMaxEnhanced8 is not None

    def test_inherits(self):
        from pyfoam.postprocessing.field_min_max_enhanced_7 import FieldMinMaxEnhanced7
        from pyfoam.postprocessing.field_min_max_enhanced_8 import FieldMinMaxEnhanced8
        fo = FieldMinMaxEnhanced8("test", {"field": "p"})
        assert isinstance(fo, FieldMinMaxEnhanced7)

    def test_default_params(self):
        from pyfoam.postprocessing.field_min_max_enhanced_8 import FieldMinMaxEnhanced8
        fo = FieldMinMaxEnhanced8("test", {"field": "p"})
        assert fo._temporal_clustering is False
        assert fo._cross_field_corr is False

    def test_custom_params(self):
        from pyfoam.postprocessing.field_min_max_enhanced_8 import FieldMinMaxEnhanced8
        fo = FieldMinMaxEnhanced8("test", {
            "field": "p",
            "temporalClustering": True,
            "nClusters": 3,
        })
        assert fo._temporal_clustering is True
        assert fo._n_clusters == 3

    def test_dataclass(self):
        from pyfoam.postprocessing.field_min_max_enhanced_8 import (
            TemporalCluster, CrossFieldCorrelation, AlertRule,
        )
        c = TemporalCluster(time=1.0, cluster_id=0, n_members=5, mean_intensity=0.5)
        assert c.n_members == 5
        cr = CrossFieldCorrelation(field_a="p", field_b="T", correlation=0.8)
        assert cr.correlation == 0.8
        a = AlertRule(severity="critical", value=100.0, threshold=50.0)
        assert a.severity == "critical"


# ---- ProbesEnhanced8 ----

class TestProbesEnhanced8:
    def test_import(self):
        from pyfoam.postprocessing.probes_enhanced_8 import ProbesEnhanced8
        assert ProbesEnhanced8 is not None

    def test_inherits(self):
        from pyfoam.postprocessing.probes_enhanced_7 import ProbesEnhanced7
        from pyfoam.postprocessing.probes_enhanced_8 import ProbesEnhanced8
        p = ProbesEnhanced8("test", {})
        assert isinstance(p, ProbesEnhanced7)

    def test_default_params(self):
        from pyfoam.postprocessing.probes_enhanced_8 import ProbesEnhanced8
        p = ProbesEnhanced8("test", {})
        assert p._streaming_enabled is False
        assert p._multi_fidelity is False

    def test_custom_params(self):
        from pyfoam.postprocessing.probes_enhanced_8 import ProbesEnhanced8
        p = ProbesEnhanced8("test", {
            "streamingMode": True,
            "streamingBufferSize": 500,
        })
        assert p._streaming_enabled is True
        assert p._buffer_size == 500

    def test_dataclass(self):
        from pyfoam.postprocessing.probes_enhanced_8 import StreamingStats, ProbeHealth
        s = StreamingStats(field_name="p", n_samples=100, running_mean=101325.0)
        assert s.n_samples == 100
        h = ProbeHealth(probe_idx=0, is_active=True, signal_quality=0.95)
        assert h.signal_quality == 0.95


# ---- ForcesEnhanced7 ----

class TestForcesEnhanced7:
    def test_import(self):
        from pyfoam.postprocessing.forces_enhanced_7 import ForcesEnhanced7
        assert ForcesEnhanced7 is not None

    def test_inherits(self):
        from pyfoam.postprocessing.forces_enhanced_6 import ForcesEnhanced6
        from pyfoam.postprocessing.forces_enhanced_7 import ForcesEnhanced7
        f = ForcesEnhanced7("test", {"patches": ["wall"]})
        assert isinstance(f, ForcesEnhanced6)

    def test_default_params(self):
        from pyfoam.postprocessing.forces_enhanced_7 import ForcesEnhanced7
        f = ForcesEnhanced7("test", {"patches": ["wall"]})
        assert f._wavelet_enabled is False
        assert f._multi_body is False

    def test_custom_params(self):
        from pyfoam.postprocessing.forces_enhanced_7 import ForcesEnhanced7
        f = ForcesEnhanced7("test", {
            "patches": ["wall"],
            "waveletAnalysis": True,
            "waveletScales": 10,
        })
        assert f._wavelet_enabled is True
        assert f._wavelet_scales == 10

    def test_dataclass(self):
        from pyfoam.postprocessing.forces_enhanced_7 import (
            WaveletDecomposition, MultiBodyForce, CoefficientStats,
        )
        w = WaveletDecomposition(n_scales=8, dominant_scale=3)
        assert w.dominant_scale == 3
        mb = MultiBodyForce(body_a="cylinder", body_b="sphere", force_interaction=1.5)
        assert mb.force_interaction == 1.5
        cs = CoefficientStats(Cd_mean=1.2, Cl_mean=0.5, confidence_95=(0.4, 0.6))
        assert cs.Cd_mean == 1.2


# ---- WallShearStressEnhanced7 ----

class TestWallShearStressEnhanced7:
    def test_import(self):
        from pyfoam.postprocessing.wall_shear_stress_enhanced_7 import WallShearStressEnhanced7
        assert WallShearStressEnhanced7 is not None

    def test_inherits(self):
        from pyfoam.postprocessing.wall_shear_stress_enhanced_6 import WallShearStressEnhanced6
        from pyfoam.postprocessing.wall_shear_stress_enhanced_7 import WallShearStressEnhanced7
        w = WallShearStressEnhanced7("test", {"patches": ["wall"]})
        assert isinstance(w, WallShearStressEnhanced6)

    def test_default_params(self):
        from pyfoam.postprocessing.wall_shear_stress_enhanced_7 import WallShearStressEnhanced7
        w = WallShearStressEnhanced7("test", {"patches": ["wall"]})
        assert w._drag_decomp is False
        assert w._wall_turb_stats is False

    def test_dataclass(self):
        from pyfoam.postprocessing.wall_shear_stress_enhanced_7 import (
            DragDecomposition, WallTurbulenceStats,
        )
        dd = DragDecomposition(patch_name="wall", pressure_drag=0.5, viscous_drag=0.3, total_drag=0.8)
        assert dd.total_drag == 0.8
        wt = WallTurbulenceStats(tau_rms=0.01, tau_skewness=0.5, tau_flatness=3.0)
        assert wt.tau_rms == 0.01


# ---- YPlusEnhanced8 ----

class TestYPlusEnhanced8:
    def test_import(self):
        from pyfoam.postprocessing.y_plus_enhanced_8 import YPlusEnhanced8
        assert YPlusEnhanced8 is not None

    def test_inherits(self):
        from pyfoam.postprocessing.y_plus_enhanced_7 import YPlusEnhanced7
        from pyfoam.postprocessing.y_plus_enhanced_8 import YPlusEnhanced8
        y = YPlusEnhanced8("test", {"rho": 1.0, "mu": 1e-5})
        assert isinstance(y, YPlusEnhanced7)

    def test_default_params(self):
        from pyfoam.postprocessing.y_plus_enhanced_8 import YPlusEnhanced8
        y = YPlusEnhanced8("test", {"rho": 1.0, "mu": 1e-5})
        assert y._spectral_enabled is False
        assert y._adaptation_enabled is False

    def test_custom_params(self):
        from pyfoam.postprocessing.y_plus_enhanced_8 import YPlusEnhanced8
        y = YPlusEnhanced8("test", {
            "rho": 1.0, "mu": 1e-5,
            "spectralAnalysis": True,
            "meshAdaptation": True,
            "nSpectralBins": 30,
        })
        assert y._spectral_enabled is True
        assert y._n_bins == 30

    def test_dataclass(self):
        from pyfoam.postprocessing.y_plus_enhanced_8 import YPlusSpectrum, MeshAdaptationCriterion
        s = YPlusSpectrum(patch_name="wall", n_bins=20, uniformity_metric=0.3)
        assert s.n_bins == 20
        m = MeshAdaptationCriterion(refine_fraction=0.2, coarsen_fraction=0.1, quality_score=0.8)
        assert m.quality_score == 0.8
