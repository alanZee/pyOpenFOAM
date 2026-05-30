"""Tests for WallShearStressEnhanced4.

Tests cover:
- Quadrant analysis
- Roughness correction
- Turbulence production estimation
- Custom parameters
"""

import pytest
import torch

from pyfoam.postprocessing.wall_shear_stress_enhanced_4 import (
    WallShearStressEnhanced4,
    QuadrantEvent,
    SpatialCorrelation,
)
from pyfoam.postprocessing.wall_shear_stress_enhanced_3 import WallShearStressEnhanced3


class TestWallShearStressEnhanced4:
    """Tests for WallShearStressEnhanced4."""

    def test_inherits_from_enhanced3(self):
        wss = WallShearStressEnhanced4("test", {
            "patches": ["wall"],
            "rho": 1.0,
            "mu": 1e-3,
        })
        assert isinstance(wss, WallShearStressEnhanced3)

    def test_default_params(self):
        wss = WallShearStressEnhanced4("test", {
            "patches": ["wall"],
            "rho": 1.0,
            "mu": 1e-3,
        })
        assert wss._quadrant_enabled is False
        assert wss._hole_level == pytest.approx(0.0)
        assert wss._roughness_height == pytest.approx(0.0)
        assert wss._spatial_corr_enabled is False
        assert wss._compute_turb_production is True

    def test_custom_params(self):
        wss = WallShearStressEnhanced4("test", {
            "patches": ["wall"],
            "rho": 1.0,
            "mu": 1e-3,
            "quadrantAnalysis": True,
            "holeLevel": 2.0,
            "roughnessHeight": 1e-4,
            "spatialCorrelation": True,
            "computeTurbProduction": False,
        })
        assert wss._quadrant_enabled is True
        assert wss._hole_level == pytest.approx(2.0)
        assert wss._roughness_height == pytest.approx(1e-4)
        assert wss._spatial_corr_enabled is True
        assert wss._compute_turb_production is False

    def test_roughness_height_property(self):
        wss = WallShearStressEnhanced4("test", {
            "patches": ["wall"],
            "rho": 1.0,
            "mu": 1e-3,
            "roughnessHeight": 5e-5,
        })
        assert wss.roughness_height == pytest.approx(5e-5)

    def test_quadrant_event_dataclass(self):
        qe = QuadrantEvent(
            patch_name="wall",
            time=1.0,
            Q1_fraction=0.1,
            Q2_fraction=0.3,
            Q3_fraction=0.1,
            Q4_fraction=0.5,
        )
        assert qe.patch_name == "wall"
        assert qe.Q4_fraction == pytest.approx(0.5)

    def test_spatial_correlation_dataclass(self):
        sc = SpatialCorrelation(
            patch_name="wall",
            time=1.0,
            integral_length=0.05,
        )
        assert sc.integral_length == pytest.approx(0.05)

    def test_empty_quadrant_events(self):
        wss = WallShearStressEnhanced4("test", {
            "patches": ["wall"],
            "rho": 1.0,
            "mu": 1e-3,
        })
        assert wss.quadrant_events == []
        assert wss.get_latest_quadrant("wall") is None

    def test_empty_spatial_correlations(self):
        wss = WallShearStressEnhanced4("test", {
            "patches": ["wall"],
            "rho": 1.0,
            "mu": 1e-3,
        })
        assert wss.spatial_correlations == []
        assert wss.get_latest_spatial_correlation("wall") is None

    def test_empty_turb_production(self):
        wss = WallShearStressEnhanced4("test", {
            "patches": ["wall"],
            "rho": 1.0,
            "mu": 1e-3,
        })
        assert wss.turb_production == []
