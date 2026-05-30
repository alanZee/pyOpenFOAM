"""Tests for ProbesEnhanced6.

Tests cover:
- POD analysis
- Lagrangian tracking
- Custom parameters
- Inheritance
"""

import pytest
import torch

from pyfoam.postprocessing.probes_enhanced_6 import (
    ProbesEnhanced6,
    PODResult,
    LagrangianTrack,
)
from pyfoam.postprocessing.probes_enhanced_5 import ProbesEnhanced5


class TestProbesEnhanced6:
    """Tests for ProbesEnhanced6."""

    def test_inherits_from_enhanced5(self):
        probes = ProbesEnhanced6("test", {})
        assert isinstance(probes, ProbesEnhanced5)

    def test_default_params(self):
        probes = ProbesEnhanced6("test", {})
        assert probes._pod_enabled is False
        assert probes._n_modes == 10
        assert probes._lagrangian_enabled is False
        assert probes._sparse_recovery is False
        assert probes._pod_update_interval == 50

    def test_custom_params(self):
        probes = ProbesEnhanced6("test", {
            "podAnalysis": True,
            "nModes": 20,
            "lagrangianTracking": True,
            "sparseRecovery": True,
            "podUpdateInterval": 100,
        })
        assert probes._pod_enabled is True
        assert probes._n_modes == 20
        assert probes._lagrangian_enabled is True
        assert probes._sparse_recovery is True
        assert probes._pod_update_interval == 100

    def test_pod_result_dataclass(self):
        pod = PODResult()
        assert pod.modes is None
        assert pod.n_modes == 0
        assert pod.time == 0.0

    def test_lagrangian_track_dataclass(self):
        track = LagrangianTrack(probe_index=0)
        assert track.probe_index == 0
        assert track.positions == []
        assert track.times == []

    def test_empty_pod_results(self):
        probes = ProbesEnhanced6("test", {})
        assert probes.pod_results == {}

    def test_empty_lagrangian_tracks(self):
        probes = ProbesEnhanced6("test", {})
        assert probes.lagrangian_tracks == []

    def test_get_pod_modes_empty(self):
        probes = ProbesEnhanced6("test", {})
        assert probes.get_pod_modes("p") is None

    def test_compute_pod_no_data(self):
        probes = ProbesEnhanced6("test", {"podAnalysis": True})
        result = probes.compute_pod("nonexistent_field")
        assert result is None
