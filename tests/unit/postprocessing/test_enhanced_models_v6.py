"""Tests for enhanced postprocessing models (Phase 16)."""

import pytest
import torch

from tests.unit.postprocessing.conftest import fv_mesh, sample_fields


# ---- FieldMinMaxEnhanced7 ----

class TestFieldMinMaxEnhanced7:
    def test_import(self):
        from pyfoam.postprocessing.field_min_max_enhanced_7 import FieldMinMaxEnhanced7
        assert FieldMinMaxEnhanced7 is not None

    def test_inherits(self):
        from pyfoam.postprocessing.field_min_max_enhanced_6 import FieldMinMaxEnhanced6
        from pyfoam.postprocessing.field_min_max_enhanced_7 import FieldMinMaxEnhanced7
        fo = FieldMinMaxEnhanced7("test", {"field": "p"})
        assert isinstance(fo, FieldMinMaxEnhanced6)

    def test_default_params(self):
        from pyfoam.postprocessing.field_min_max_enhanced_7 import FieldMinMaxEnhanced7
        fo = FieldMinMaxEnhanced7("test", {"field": "p"})
        assert fo._multivariate_anomaly is False
        assert fo._adaptive_thresholds is False

    def test_custom_params(self):
        from pyfoam.postprocessing.field_min_max_enhanced_7 import FieldMinMaxEnhanced7
        fo = FieldMinMaxEnhanced7("test", {
            "field": "p",
            "multivariateAnomaly": True,
            "adaptiveThresholds": True,
            "anomalySigma": 2.5,
        })
        assert fo._multivariate_anomaly is True
        assert fo._anomaly_sigma == 2.5

    def test_dataclass(self):
        from pyfoam.postprocessing.field_min_max_enhanced_7 import MultivariateAnomaly, AdaptiveThreshold
        a = MultivariateAnomaly(time=1.0, mahalanobis_distance=5.0, is_anomaly=True)
        assert a.is_anomaly
        t = AdaptiveThreshold(time=1.0, field_name="p", upper_threshold=1.0, lower_threshold=-1.0)
        assert t.field_name == "p"


# ---- ProbesEnhanced7 ----

class TestProbesEnhanced7:
    def test_import(self):
        from pyfoam.postprocessing.probes_enhanced_7 import ProbesEnhanced7
        assert ProbesEnhanced7 is not None

    def test_inherits(self):
        from pyfoam.postprocessing.probes_enhanced_6 import ProbesEnhanced6
        from pyfoam.postprocessing.probes_enhanced_7 import ProbesEnhanced7
        p = ProbesEnhanced7("test", {})
        assert isinstance(p, ProbesEnhanced6)

    def test_default_params(self):
        from pyfoam.postprocessing.probes_enhanced_7 import ProbesEnhanced7
        p = ProbesEnhanced7("test", {})
        assert p._cs_enabled is False
        assert p._rom_enabled is False

    def test_dataclass(self):
        from pyfoam.postprocessing.probes_enhanced_7 import CompressedSensingResult, SensorPlacementResult
        r = CompressedSensingResult(field_name="p", residual=1e-7, n_iterations=10)
        assert r.n_iterations == 10
        s = SensorPlacementResult(n_sensors=5, coverage_metric=0.8)
        assert s.coverage_metric == 0.8


# ---- ForcesEnhanced6 ----

class TestForcesEnhanced6:
    def test_import(self):
        from pyfoam.postprocessing.forces_enhanced_6 import ForcesEnhanced6
        assert ForcesEnhanced6 is not None

    def test_inherits(self):
        from pyfoam.postprocessing.forces_enhanced_5 import ForcesEnhanced5
        from pyfoam.postprocessing.forces_enhanced_6 import ForcesEnhanced6
        f = ForcesEnhanced6("test", {"patches": ["wall"]})
        assert isinstance(f, ForcesEnhanced5)

    def test_default_params(self):
        from pyfoam.postprocessing.forces_enhanced_6 import ForcesEnhanced6
        f = ForcesEnhanced6("test", {"patches": ["wall"]})
        assert f._dmd_enabled is False
        assert f._freq_tracking is False

    def test_dataclass(self):
        from pyfoam.postprocessing.forces_enhanced_6 import DMDMode, FrequencyTracker
        d = DMDMode(n_modes=5)
        assert d.n_modes == 5
        ft = FrequencyTracker(dominant_freq_drag=10.0, dominant_freq_lift=20.0)
        assert ft.dominant_freq_drag == 10.0


# ---- WallShearStressEnhanced6 ----

class TestWallShearStressEnhanced6:
    def test_import(self):
        from pyfoam.postprocessing.wall_shear_stress_enhanced_6 import WallShearStressEnhanced6
        assert WallShearStressEnhanced6 is not None

    def test_inherits(self):
        from pyfoam.postprocessing.wall_shear_stress_enhanced_5 import WallShearStressEnhanced5
        from pyfoam.postprocessing.wall_shear_stress_enhanced_6 import WallShearStressEnhanced6
        w = WallShearStressEnhanced6("test", {"patches": ["wall"]})
        assert isinstance(w, WallShearStressEnhanced5)

    def test_dataclass(self):
        from pyfoam.postprocessing.wall_shear_stress_enhanced_6 import WMLESInterface, PressureStrainCorrelation
        w = WMLESInterface(patch_name="wall", u_tau_model=0.05)
        assert w.u_tau_model == 0.05
        ps = PressureStrainCorrelation(phi_iw1=0.3, phi_iw2=-0.1)
        assert ps.phi_iw1 == 0.3


# ---- YPlusEnhanced7 ----

class TestYPlusEnhanced7:
    def test_import(self):
        from pyfoam.postprocessing.y_plus_enhanced_7 import YPlusEnhanced7
        assert YPlusEnhanced7 is not None

    def test_inherits(self):
        from pyfoam.postprocessing.y_plus_enhanced_6 import YPlusEnhanced6
        from pyfoam.postprocessing.y_plus_enhanced_7 import YPlusEnhanced7
        y = YPlusEnhanced7("test", {"rho": 1.0, "mu": 1e-5})
        assert isinstance(y, YPlusEnhanced6)

    def test_default_params(self):
        from pyfoam.postprocessing.y_plus_enhanced_7 import YPlusEnhanced7
        y = YPlusEnhanced7("test", {"rho": 1.0, "mu": 1e-5})
        assert y._uq_enabled is False
        assert y._ensemble_enabled is False

    def test_custom_params(self):
        from pyfoam.postprocessing.y_plus_enhanced_7 import YPlusEnhanced7
        y = YPlusEnhanced7("test", {
            "rho": 1.0, "mu": 1e-5,
            "uncertaintyQuantification": True,
            "ensembleAnalysis": True,
        })
        assert y._uq_enabled is True
        assert y._ensemble_enabled is True

    def test_dataclass(self):
        from pyfoam.postprocessing.y_plus_enhanced_7 import YPlusUncertainty, WallFunctionEnsemble
        u = YPlusUncertainty(y_plus_mean=15.0, total_uncertainty=2.5)
        assert u.y_plus_mean == 15.0
        e = WallFunctionEnsemble(best_estimate=0.05, confidence_interval=(0.04, 0.06))
        assert e.best_estimate == 0.05
