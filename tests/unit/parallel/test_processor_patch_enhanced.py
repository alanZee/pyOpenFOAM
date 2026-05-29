"""Tests for processor_patch_enhanced — non-conformal processor patches."""

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.parallel.processor_patch import ProcessorPatch
from pyfoam.parallel.processor_patch_enhanced import (
    NonConformalPatch,
    EnhancedHaloExchange,
)


# ---------------------------------------------------------------------------
# NonConformalPatch tests
# ---------------------------------------------------------------------------


class TestNonConformalPatch:
    """Test NonConformalPatch dataclass."""

    def test_creation_basic(self):
        """Create a basic non-conformal patch."""
        patch = NonConformalPatch(
            name="ncPatch0",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0, 1], dtype=INDEX_DTYPE),
            remote_cells=torch.tensor([2, 3], dtype=INDEX_DTYPE),
        )
        assert patch.name == "ncPatch0"
        assert patch.neighbour_rank == 1
        assert patch.n_ghost_cells == 2
        assert patch.n_send_cells == 2
        assert patch.is_non_conformal is False  # No weights

    def test_creation_with_weights(self):
        """Create a non-conformal patch with interpolation weights."""
        weights = torch.tensor([
            [0.6, 0.4],
            [0.3, 0.7],
        ], dtype=torch.float64)
        src_indices = torch.tensor([
            [0, 1],
            [1, 2],
        ], dtype=INDEX_DTYPE)

        patch = NonConformalPatch(
            name="ncPatch1",
            neighbour_rank=2,
            local_ghost_cells=torch.tensor([0, 1], dtype=INDEX_DTYPE),
            remote_cells=torch.tensor([3, 4, 5], dtype=INDEX_DTYPE),
            local_weights=weights,
            local_src_indices=src_indices,
        )

        assert patch.is_non_conformal is True
        assert patch.n_interp_sources == 2
        assert patch.local_weights is not None
        assert patch.local_weights.shape == (2, 2)

    def test_inherits_processor_patch(self):
        """NonConformalPatch is a subclass of ProcessorPatch."""
        patch = NonConformalPatch(
            name="test",
            neighbour_rank=0,
            local_ghost_cells=torch.tensor([], dtype=INDEX_DTYPE),
            remote_cells=torch.tensor([], dtype=INDEX_DTYPE),
        )
        assert isinstance(patch, ProcessorPatch)


# ---------------------------------------------------------------------------
# EnhancedHaloExchange tests
# ---------------------------------------------------------------------------


class TestEnhancedHaloExchange:
    """Test EnhancedHaloExchange."""

    def test_creation_conformal_only(self):
        """Create with only conformal patches."""
        patch = ProcessorPatch(
            name="proc0",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([4, 5], dtype=INDEX_DTYPE),
            remote_cells=torch.tensor([0, 1], dtype=INDEX_DTYPE),
        )
        halo = EnhancedHaloExchange([patch])

        assert halo.n_non_conformal_patches == 0
        assert halo.rank == 0  # Serial fallback

    def test_creation_mixed(self):
        """Create with conformal and non-conformal patches."""
        conformal = ProcessorPatch(
            name="proc0",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([4, 5], dtype=INDEX_DTYPE),
            remote_cells=torch.tensor([0, 1], dtype=INDEX_DTYPE),
        )
        nc = NonConformalPatch(
            name="ncProc0",
            neighbour_rank=2,
            local_ghost_cells=torch.tensor([6, 7], dtype=INDEX_DTYPE),
            remote_cells=torch.tensor([2, 3], dtype=INDEX_DTYPE),
            local_weights=torch.tensor([[1.0], [1.0]], dtype=torch.float64),
            local_src_indices=torch.tensor([[0], [1]], dtype=INDEX_DTYPE),
        )
        halo = EnhancedHaloExchange([conformal, nc])

        assert halo.n_non_conformal_patches == 1

    def test_exchange_conformal(self):
        """Conformal exchange works through base class."""
        patch = ProcessorPatch(
            name="proc0",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([4, 5], dtype=INDEX_DTYPE),
            remote_cells=torch.tensor([0, 1], dtype=INDEX_DTYPE),
        )
        halo = EnhancedHaloExchange([patch])

        field = torch.arange(6, dtype=torch.float64)
        result = halo.exchange(field, all_fields={1: torch.arange(6, dtype=torch.float64) * 10.0})

        # Ghost cells should be updated from the neighbour field
        assert result.shape == (6,)

    def test_exchange_non_conformal_serial(self):
        """Non-conformal exchange in serial mode."""
        nc = NonConformalPatch(
            name="ncProc0",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([2, 3], dtype=INDEX_DTYPE),
            remote_cells=torch.tensor([0, 1], dtype=INDEX_DTYPE),
            local_weights=torch.tensor([[1.0], [1.0]], dtype=torch.float64),
            local_src_indices=torch.tensor([[0], [1]], dtype=INDEX_DTYPE),
        )
        halo = EnhancedHaloExchange([nc])

        field = torch.tensor([10.0, 20.0, 0.0, 0.0])
        nbr_field = torch.tensor([100.0, 200.0])
        result = halo.exchange(field, all_fields={1: nbr_field})

        assert result[2] == 100.0
        assert result[3] == 200.0

    def test_exchange_non_conformal_weighted(self):
        """Non-conformal exchange with weighted interpolation."""
        nc = NonConformalPatch(
            name="ncProc0",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([4], dtype=INDEX_DTYPE),
            remote_cells=torch.tensor([0], dtype=INDEX_DTYPE),
            local_weights=torch.tensor([[0.3, 0.7]], dtype=torch.float64),
            local_src_indices=torch.tensor([[0, 1]], dtype=INDEX_DTYPE),
        )
        halo = EnhancedHaloExchange([nc])

        field = torch.tensor([10.0, 20.0, 30.0, 40.0, 0.0])
        nbr_field = torch.tensor([100.0, 200.0])
        result = halo.exchange(field, all_fields={1: nbr_field})

        # 0.3 * 100 + 0.7 * 200 = 170
        assert result[4] == pytest.approx(170.0)

    def test_exchange_multi_field(self):
        """Batch exchange for multiple fields."""
        patch = ProcessorPatch(
            name="proc0",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([2], dtype=INDEX_DTYPE),
            remote_cells=torch.tensor([0], dtype=INDEX_DTYPE),
        )
        halo = EnhancedHaloExchange([patch])

        fields = {
            "p": torch.tensor([101325.0, 101300.0, 0.0]),
            "U": torch.tensor([1.0, 0.5, 0.0]),
        }
        nbr_fields = {
            "p": torch.tensor([101325.0, 101300.0, 0.0]),
            "U": torch.tensor([2.0, 1.0, 0.0]),
        }

        result = halo.exchange_multi_field(fields, all_fields={1: nbr_fields})

        assert "p" in result
        assert "U" in result
        assert result["p"].shape == (3,)
        assert result["U"].shape == (3,)

    def test_empty_patches(self):
        """Exchange with no patches returns field unchanged."""
        halo = EnhancedHaloExchange([])
        field = torch.tensor([1.0, 2.0, 3.0])
        result = halo.exchange(field)
        assert torch.equal(result, field)
