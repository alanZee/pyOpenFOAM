"""Tests for ProcessorPatchEnhanced5 — v5 enhanced processor patches."""

import pytest
import torch

from pyfoam.parallel.processor_patch_enhanced_5 import (
    CoarsenablePatch5,
    EnhancedHaloExchange5,
    CompressionStats,
)
from pyfoam.parallel.processor_patch_enhanced_4 import (
    NonConformalPatch4,
    EnhancedHaloExchange4,
)


class TestCompressionStats:
    """Test CompressionStats dataclass."""

    def test_defaults(self):
        cs = CompressionStats()
        assert cs.ratio == 1.0
        assert cs.original_size == 0


class TestCoarsenablePatch:
    """Test CoarsenablePatch5."""

    def test_creation(self):
        patch = CoarsenablePatch5(
            name="proc0To1",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0, 1, 2, 3]),
            remote_cells=torch.tensor([4, 5, 6, 7]),
        )
        assert patch.coarsening_level == 0
        assert patch.effective_n_cells == 4

    def test_coarsen(self):
        patch = CoarsenablePatch5(
            name="proc0To1",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0, 1, 2, 3]),
            remote_cells=torch.tensor([4, 5, 6, 7]),
        )
        patch.coarsen(ratio=2)
        assert patch.coarsening_level == 1
        assert patch.effective_n_cells == 2

    def test_coarsen_not_divisible(self):
        patch = CoarsenablePatch5(
            name="proc0To1",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0, 1, 2]),
            remote_cells=torch.tensor([3, 4, 5]),
        )
        with pytest.raises(ValueError, match="not divisible"):
            patch.coarsen(ratio=2)

    def test_coarsen_no_ghost_cells(self):
        patch = CoarsenablePatch5(
            name="proc0To1",
            neighbour_rank=1,
            local_ghost_cells=torch.zeros(0, dtype=torch.long),
            remote_cells=torch.zeros(0, dtype=torch.long),
        )
        with pytest.raises(ValueError, match="empty ghost cell"):
            patch.coarsen(ratio=2)

    def test_inherits_v4(self):
        assert issubclass(CoarsenablePatch5, NonConformalPatch4)


class TestEnhancedHaloExchange5:
    """Test EnhancedHaloExchange5."""

    def test_rle_compress_decompress(self):
        data = torch.tensor([1.0, 1.0, 2.0, 3.0, 3.0, 3.0], dtype=torch.float64)
        values, counts = EnhancedHaloExchange5.compress_rle(data)
        restored = EnhancedHaloExchange5.decompress_rle(values, counts)
        assert torch.allclose(restored, data)

    def test_rle_empty(self):
        data = torch.zeros(0, dtype=torch.float64)
        values, counts = EnhancedHaloExchange5.compress_rle(data)
        assert values.numel() == 0
        assert counts.numel() == 0

    def test_rle_constant(self):
        data = torch.full((100,), 42.0, dtype=torch.float64)
        values, counts = EnhancedHaloExchange5.compress_rle(data)
        assert values.numel() == 1
        assert counts[0].item() == 100

    def test_compression_stats(self):
        patch = CoarsenablePatch5(
            name="proc0To1",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0, 1]),
            remote_cells=torch.tensor([2, 3]),
        )
        halo = EnhancedHaloExchange5([patch])
        field = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0], dtype=torch.float64)
        stats = halo.compute_compression_stats(field)
        assert stats.ratio > 1.0  # should compress

    def test_exchange_compressed(self):
        patch = CoarsenablePatch5(
            name="proc0To1",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0, 1]),
            remote_cells=torch.tensor([2, 3]),
        )
        halo = EnhancedHaloExchange5([patch])
        field = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float64)
        result = halo.exchange_compressed(field, compression_level=1)
        assert result.shape == field.shape

    def test_exchange_coarsened(self):
        patch = CoarsenablePatch5(
            name="proc0To1",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0, 1, 2, 3]),
            remote_cells=torch.tensor([4, 5, 6, 7]),
        )
        halo = EnhancedHaloExchange5([patch])
        field = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                            dtype=torch.float64)
        result = halo.exchange_coarsened(field, coarsening_level=1)
        assert result.shape == field.shape

    def test_inherits_v4(self):
        assert issubclass(EnhancedHaloExchange5, EnhancedHaloExchange4)

    def test_repr(self):
        patch = CoarsenablePatch5(
            name="proc0To1",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0, 1, 2, 3]),
            remote_cells=torch.tensor([4, 5, 6, 7]),
        )
        halo = EnhancedHaloExchange5([patch])
        r = repr(halo)
        assert "EnhancedHaloExchange5" in r
