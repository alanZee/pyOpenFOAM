"""Tests for ProcessorPatchEnhanced6 -- v6 enhanced processor patches."""

import pytest
import torch

from pyfoam.parallel.processor_patch_enhanced_6 import (
    OverlappedPatch6,
    EnhancedHaloExchange6,
    WeightedInterpolation,
    BandwidthStats,
)
from pyfoam.parallel.processor_patch_enhanced_5 import (
    CoarsenablePatch5,
    EnhancedHaloExchange5,
)


class TestBandwidthStats:
    """Test BandwidthStats dataclass."""

    def test_defaults(self):
        stats = BandwidthStats()
        assert stats.total_bytes == 0
        assert stats.n_messages == 0


class TestWeightedInterpolation:
    """Test WeightedInterpolation."""

    def test_default_power(self):
        interp = WeightedInterpolation()
        assert interp.power == 2.0

    def test_custom_power(self):
        interp = WeightedInterpolation(power=3.0)
        assert interp.power == 3.0

    def test_compute_weights(self):
        interp = WeightedInterpolation(power=2.0)
        sources = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=torch.float64)
        targets = torch.tensor([
            [0.5, 0.0, 0.0],
        ], dtype=torch.float64)
        weights = interp.compute_weights(sources, targets)
        assert weights.shape == (1, 2)
        # Weights should sum to 1
        assert weights.sum().item() == pytest.approx(1.0)
        # Symmetric: equal distance -> equal weights
        assert weights[0, 0].item() == pytest.approx(weights[0, 1].item())

    def test_interpolate(self):
        interp = WeightedInterpolation()
        values = torch.tensor([1.0, 3.0], dtype=torch.float64)
        weights = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
        result = interp.interpolate(values, weights)
        assert result.item() == pytest.approx(2.0)


class TestOverlappedPatch6:
    """Test OverlappedPatch6 dataclass."""

    def test_defaults(self):
        patch = OverlappedPatch6(name="test", neighbour_rank=1)
        assert patch.n_overlap_layers == 1

    def test_extend_overlap(self):
        patch = OverlappedPatch6(name="test", neighbour_rank=1)
        patch.extend_overlap(2)
        assert patch.n_overlap_layers == 3

    def test_effective_overlap_size_no_cells(self):
        patch = OverlappedPatch6(name="test", neighbour_rank=1)
        assert patch.effective_overlap_size == 0

    def test_effective_overlap_size_with_cells(self):
        patch = OverlappedPatch6(
            name="test",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0, 1, 2]),
            n_overlap_layers=2,
        )
        assert patch.effective_overlap_size == 6  # 3 cells * 2 layers


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v5(self):
        assert issubclass(EnhancedHaloExchange6, EnhancedHaloExchange5)

    def test_patch_inherits(self):
        assert issubclass(OverlappedPatch6, CoarsenablePatch5)


class TestBandwidthEstimation:
    """Test bandwidth estimation."""

    def test_estimate_bandwidth(self):
        patch = OverlappedPatch6(
            name="proc0To1",
            neighbour_rank=1,
            local_ghost_cells=torch.arange(100),
        )
        halo = EnhancedHaloExchange6([patch], bandwidth_gbps=10.0)
        field = torch.randn(1000)
        stats = halo.estimate_bandwidth_stats(field)
        assert stats.total_bytes > 0
        assert stats.n_messages == 1
        assert stats.estimated_time > 0


class TestAdaptiveCompression:
    """Test adaptive compression selection."""

    def test_select_compression_level(self):
        patch = OverlappedPatch6(
            name="proc0To1",
            neighbour_rank=1,
            local_ghost_cells=torch.arange(10),
        )
        halo = EnhancedHaloExchange6([patch])
        # Uniform field -> high compression ratio -> level 1
        uniform = torch.ones(100)
        level = halo.select_compression_level(uniform)
        assert level == 1

    def test_repr(self):
        patch = OverlappedPatch6(name="proc0To1", neighbour_rank=1)
        halo = EnhancedHaloExchange6([patch], bandwidth_gbps=25.0)
        r = repr(halo)
        assert "EnhancedHaloExchange6" in r
        assert "25.0Gbps" in r
