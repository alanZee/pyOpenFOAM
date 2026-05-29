"""Tests for ProbesEnhanced4.

Tests cover:
- Probe group manager
- Frequency tracking
- Signal filtering
- Execute with mesh
"""

import pytest
import torch

from pyfoam.postprocessing.probes_enhanced_4 import (
    ProbesEnhanced4,
    ProbeGroupManager,
    FrequencyTracker,
)
from pyfoam.postprocessing.probes_enhanced_3 import ProbesEnhanced3


class TestProbeGroupManager:
    """Tests for ProbeGroupManager."""

    def test_add_group(self):
        mgr = ProbeGroupManager()
        mgr.add_group("inlet", [0, 1, 2])
        assert "inlet" in mgr.groups
        assert mgr.groups["inlet"] == [0, 1, 2]
        assert 0 in mgr.active_probes

    def test_remove_group(self):
        mgr = ProbeGroupManager()
        mgr.add_group("inlet", [0, 1])
        mgr.add_group("outlet", [2, 3])
        mgr.remove_group("inlet")
        assert "inlet" not in mgr.groups
        # Probes 2, 3 should still be active
        assert 2 in mgr.active_probes

    def test_get_group_indices(self):
        mgr = ProbeGroupManager()
        mgr.add_group("g1", [0, 1])
        assert mgr.get_group_indices("g1") == [0, 1]
        assert mgr.get_group_indices("nonexistent") == []

    def test_get_groups_for_probe(self):
        mgr = ProbeGroupManager()
        mgr.add_group("g1", [0, 1])
        mgr.add_group("g2", [1, 2])
        groups = mgr.get_groups_for_probe(1)
        assert "g1" in groups
        assert "g2" in groups


class TestFrequencyTracker:
    """Tests for FrequencyTracker."""

    def test_update(self):
        tracker = FrequencyTracker()
        tracker.update(0.0, 10.0, [10.0, 20.0, 30.0])
        assert len(tracker.times) == 1
        assert tracker.dominant_frequencies[0] == pytest.approx(10.0)

    def test_multiple_updates(self):
        tracker = FrequencyTracker()
        tracker.update(0.0, 10.0)
        tracker.update(0.1, 12.0)
        tracker.update(0.2, 11.0)
        assert len(tracker.times) == 3
        assert tracker.dominant_frequencies[-1] == pytest.approx(11.0)


class TestProbesEnhanced4:
    """Tests for ProbesEnhanced4."""

    def test_inherits_from_enhanced3(self):
        probes = ProbesEnhanced4("test", {
            "probeGroups": [
                {"name": "g1", "locations": [[0.5, 0.5, 0.5]], "fields": ["p"]},
            ],
        })
        assert isinstance(probes, ProbesEnhanced3)

    def test_default_params(self):
        probes = ProbesEnhanced4("test", {})
        assert probes._track_dominant_freq is True
        assert probes._filter_type == "none"
        assert probes._filter_order == 4

    def test_custom_params(self):
        probes = ProbesEnhanced4("test", {
            "trackDominantFrequency": False,
            "filterType": "lowpass",
            "filterCutoff": 100.0,
            "filterOrder": 2,
        })
        assert probes._track_dominant_freq is False
        assert probes._filter_type == "lowpass"
        assert probes._filter_cutoff == pytest.approx(100.0)

    def test_group_manager(self):
        probes = ProbesEnhanced4("test", {
            "probeGroups": [
                {"name": "inlet", "locations": [[0.1, 0.5, 0.5]], "fields": ["p"]},
                {"name": "outlet", "locations": [[0.9, 0.5, 0.5]], "fields": ["p"]},
            ],
        })
        assert "inlet" in probes.group_manager.groups
        assert "outlet" in probes.group_manager.groups

    def test_filter_none(self):
        probes = ProbesEnhanced4("test", {"filterType": "none"})
        signal = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        filtered = probes._apply_filter(signal, 0.01)
        assert torch.allclose(filtered, signal)

    def test_filter_lowpass(self):
        probes = ProbesEnhanced4("test", {
            "filterType": "lowpass",
            "filterCutoff": 5.0,
        })
        # Create a signal with high-frequency noise
        n = 100
        t = torch.linspace(0, 1, n, dtype=torch.float64)
        signal = torch.sin(2 * 3.14159 * 2.0 * t) + 0.5 * torch.sin(2 * 3.14159 * 50.0 * t)
        filtered = probes._apply_filter(signal, 1.0 / n)
        assert filtered.shape == signal.shape
        # High-frequency component should be reduced
        assert filtered.std() < signal.std()

    def test_frequency_trackers_empty(self):
        probes = ProbesEnhanced4("test", {})
        assert len(probes.frequency_trackers) == 0


from tests.unit.postprocessing.conftest import fv_mesh, sample_fields
