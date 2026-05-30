"""Tests for ForcesEnhanced4.

Tests cover:
- Unsteady force statistics
- Aeroacoustic source computation
- Custom parameters
- Inheritance
"""

import pytest
import torch

from pyfoam.postprocessing.forces_enhanced_4 import (
    ForcesEnhanced4,
    UnsteadyForceStats,
    AeroacousticSource,
)
from pyfoam.postprocessing.forces_enhanced_3 import ForcesEnhanced3


class TestForcesEnhanced4:
    """Tests for ForcesEnhanced4."""

    def test_inherits_from_enhanced3(self):
        forces = ForcesEnhanced4("test", {
            "patches": ["wall"],
            "rhoInf": 1.225,
        })
        assert isinstance(forces, ForcesEnhanced3)

    def test_default_params(self):
        forces = ForcesEnhanced4("test", {
            "patches": ["wall"],
            "rhoInf": 1.225,
        })
        assert forces._compute_unsteady is True
        assert forces._compute_aeroacoustic is False
        assert forces._unsteady_window == 100
        assert forces._aa_ref_dist == pytest.approx(1.0)
        assert forces._c0 == pytest.approx(343.0)

    def test_custom_params(self):
        forces = ForcesEnhanced4("test", {
            "patches": ["wall"],
            "rhoInf": 1.225,
            "computeUnsteadyStats": False,
            "computeAeroacousticSources": True,
            "unsteadyWindowSize": 50,
            "aeroacousticRefDistance": 5.0,
            "speedOfSound": 1500.0,
        })
        assert forces._compute_unsteady is False
        assert forces._compute_aeroacoustic is True
        assert forces._unsteady_window == 50
        assert forces._aa_ref_dist == pytest.approx(5.0)
        assert forces._c0 == pytest.approx(1500.0)

    def test_unsteady_stats_dataclass(self):
        us = UnsteadyForceStats(
            time=1.0,
            drag_rms=0.5,
            lift_rms=0.3,
            drag_pp=2.0,
            lift_pp=1.5,
        )
        assert us.time == pytest.approx(1.0)
        assert us.n_samples == 0

    def test_aeroacoustic_source_dataclass(self):
        aa = AeroacousticSource(
            time=1.0,
            source_strength=100.0,
            source_power=1e-3,
            spl_estimate=60.0,
        )
        assert aa.time == pytest.approx(1.0)
        assert aa.spl_estimate == pytest.approx(60.0)

    def test_empty_unsteady_stats(self):
        forces = ForcesEnhanced4("test", {
            "patches": ["wall"],
            "rhoInf": 1.225,
        })
        assert forces.unsteady_stats == []
        assert forces.get_latest_unsteady() is None

    def test_empty_aeroacoustic(self):
        forces = ForcesEnhanced4("test", {
            "patches": ["wall"],
            "rhoInf": 1.225,
        })
        assert forces.aeroacoustic_sources == []
        assert forces.get_latest_aeroacoustic() is None
