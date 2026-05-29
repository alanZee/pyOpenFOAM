"""Tests for ProbesEnhanced2.

Tests cover:
- Probe group management
- Cross-spectral analysis
- Probe health monitoring
- Inheritance from ProbesEnhanced
"""

import math

import pytest
import torch

from pyfoam.postprocessing.probes_enhanced import ProbesEnhanced
from pyfoam.postprocessing.probes_enhanced_2 import (
    ProbesEnhanced2,
    CrossSpectrumResult,
    ProbeGroup,
)


class TestProbesEnhanced2:
    """Tests for ProbesEnhanced2."""

    def test_inherits_from_enhanced(self):
        probes = ProbesEnhanced2("test", {"fields": ["p"]})
        assert isinstance(probes, ProbesEnhanced)

    def test_probe_group_parsing(self):
        probes = ProbesEnhanced2("test", {
            "probeGroups": [
                {"name": "inlet", "locations": [[0.1, 0.5, 0.5]], "fields": ["p"]},
                {"name": "outlet", "locations": [[0.9, 0.5, 0.5]], "fields": ["p", "U"]},
            ],
        })
        assert len(probes.probe_groups) == 2
        assert probes.probe_groups[0].name == "inlet"
        assert probes.probe_groups[1].name == "outlet"

    def test_no_probe_groups(self):
        probes = ProbesEnhanced2("test", {"fields": ["p"]})
        assert len(probes.probe_groups) == 0

    def test_health_check_default(self):
        probes = ProbesEnhanced2("test", {"fields": ["p"]})
        assert probes._health_check is True

    def test_check_probe_health_no_data(self):
        probes = ProbesEnhanced2("test", {"fields": ["p"]})
        health = probes.check_probe_health()
        assert health["total_probes"] == 0
        assert health["healthy_probes"] == 0

    def test_cross_spectrum_insufficient_data(self, fv_mesh, sample_fields):
        probes = ProbesEnhanced2("test", {
            "fields": ["p"],
            "probeLocations": [[0.5, 0.5, 0.5], [0.3, 0.3, 0.3]],
            "computeCrossSpectrum": True,
        })
        probes.initialise(fv_mesh, sample_fields)
        probes.execute(0.0)

        # Only 1 data point, not enough for cross-spectrum
        result = probes.compute_cross_spectrum("p", 0, 1, dt=1.0)
        assert result is None

    def test_cross_spectrum_computation(self, fv_mesh, sample_fields):
        probes = ProbesEnhanced2("test", {
            "fields": ["p"],
            "probeLocations": [[0.5, 0.5, 0.5], [0.3, 0.3, 0.3]],
            "computeCrossSpectrum": True,
        })
        probes.initialise(fv_mesh, sample_fields)

        # Populate data
        dt = 0.01
        for i in range(50):
            t = i * dt
            probes._results["p"][0].append(math.sin(2 * math.pi * 10 * t))
            probes._results["p"][1].append(math.sin(2 * math.pi * 10 * t + 0.5))
            probes._times.append(t)

        result = probes.compute_cross_spectrum("p", 0, 1, dt=dt)
        assert result is not None
        assert isinstance(result, CrossSpectrumResult)
        assert result.coherence.shape == result.frequencies.shape
        assert (result.coherence >= 0).all()
        assert (result.coherence <= 1).all()

    def test_cross_spectrum_cached(self, fv_mesh, sample_fields):
        probes = ProbesEnhanced2("test", {
            "fields": ["p"],
            "probeLocations": [[0.5, 0.5, 0.5], [0.3, 0.3, 0.3]],
        })
        probes.initialise(fv_mesh, sample_fields)

        dt = 0.01
        for i in range(50):
            probes._results["p"][0].append(math.sin(i * dt))
            probes._results["p"][1].append(math.cos(i * dt))
            probes._times.append(i * dt)

        r1 = probes.compute_cross_spectrum("p", 0, 1, dt=dt)
        r2 = probes.compute_cross_spectrum("p", 0, 1, dt=dt)
        assert r1 is r2  # Cached

    def test_health_report(self, fv_mesh, sample_fields):
        probes = ProbesEnhanced2("test", {
            "fields": ["p"],
            "probeLocations": [[0.5, 0.5, 0.5]],
        })
        probes.initialise(fv_mesh, sample_fields)
        probes.execute(0.0)

        health = probes.check_probe_health()
        assert "total_probes" in health
        assert "healthy_probes" in health

    def test_out_of_bounds_probes_empty(self):
        probes = ProbesEnhanced2("test", {"fields": ["p"]})
        assert len(probes.out_of_bounds_probes) == 0


from tests.unit.postprocessing.conftest import fv_mesh, sample_fields
