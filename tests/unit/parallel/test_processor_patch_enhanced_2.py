"""Tests for processor_patch_enhanced_2 — v2 enhanced processor patches."""

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.parallel.processor_patch import ProcessorPatch
from pyfoam.parallel.processor_patch_enhanced import NonConformalPatch
from pyfoam.parallel.processor_patch_enhanced_2 import (
    NonConformalPatch2,
    EnhancedHaloExchange2,
    FaceCentreInterpolator,
)


# ---------------------------------------------------------------------------
# FaceCentreInterpolator tests
# ---------------------------------------------------------------------------


class TestFaceCentreInterpolator:
    """Test face-centre interpolation weight computation."""

    def test_compute_weights_shape(self):
        """Output shapes are correct."""
        interp = FaceCentreInterpolator(n_nearest=2)
        local = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        remote = torch.tensor([[0.1, 0.0, 0.0], [0.5, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=torch.float64)

        weights, indices = interp.compute_weights(local, remote)
        assert weights.shape == (2, 2)
        assert indices.shape == (2, 2)

    def test_weights_sum_to_one(self):
        """Each row of weights sums to 1."""
        interp = FaceCentreInterpolator(n_nearest=3)
        local = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        remote = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)

        weights, _ = interp.compute_weights(local, remote)
        assert weights.sum(dim=1).item() == pytest.approx(1.0, abs=1e-10)

    def test_nearest_dominates(self):
        """Closest point gets highest weight."""
        interp = FaceCentreInterpolator(n_nearest=2)
        local = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        remote = torch.tensor([[0.1, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=torch.float64)

        weights, indices = interp.compute_weights(local, remote)
        # Closest is index 0
        assert indices[0, 0].item() == 0
        # Weight of closest should be higher
        assert weights[0, 0].item() > weights[0, 1].item()

    def test_n_nearest_capped(self):
        """n_nearest is capped by number of remote faces."""
        interp = FaceCentreInterpolator(n_nearest=10)
        local = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        remote = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64)

        weights, indices = interp.compute_weights(local, remote)
        assert weights.shape == (1, 2)  # Capped to n_remote


# ---------------------------------------------------------------------------
# NonConformalPatch2 tests
# ---------------------------------------------------------------------------


class TestNonConformalPatch2:
    """Test NonConformalPatch2 dataclass."""

    def test_inherits_nc_patch(self):
        """NonConformalPatch2 extends NonConformalPatch."""
        assert issubclass(NonConformalPatch2, NonConformalPatch)

    def test_creation_with_face_centres(self):
        """Create with face centres."""
        lfc = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        rfc = torch.tensor([[0.1, 0.0, 0.0], [0.5, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=torch.float64)

        patch = NonConformalPatch2(
            name="ncPatch2",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0, 1], dtype=INDEX_DTYPE),
            remote_cells=torch.tensor([0, 1, 2], dtype=INDEX_DTYPE),
            local_face_centres=lfc,
            remote_face_centres=rfc,
        )
        assert patch.local_face_centres is not None
        assert patch.remote_face_centres is not None

    def test_auto_compute_weights(self):
        """auto_compute_weights computes weights from face centres."""
        lfc = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        rfc = torch.tensor([[0.1, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)

        patch = NonConformalPatch2(
            name="test",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0], dtype=INDEX_DTYPE),
            remote_cells=torch.tensor([0, 1], dtype=INDEX_DTYPE),
            local_face_centres=lfc,
            remote_face_centres=rfc,
        )
        patch.auto_compute_weights(n_nearest=2)

        assert patch.local_weights is not None
        assert patch.local_src_indices is not None
        assert patch.is_non_conformal

    def test_auto_compute_without_centres_raises(self):
        """ValueError when face centres not set."""
        patch = NonConformalPatch2(
            name="test",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0], dtype=INDEX_DTYPE),
            remote_cells=torch.tensor([0], dtype=INDEX_DTYPE),
        )
        with pytest.raises(ValueError, match="Face centres"):
            patch.auto_compute_weights()


# ---------------------------------------------------------------------------
# EnhancedHaloExchange2 tests
# ---------------------------------------------------------------------------


class TestEnhancedHaloExchange2:
    """Test EnhancedHaloExchange2."""

    def test_inherits_enhanced_halo(self):
        """EnhancedHaloExchange2 extends EnhancedHaloExchange."""
        from pyfoam.parallel.processor_patch_enhanced import EnhancedHaloExchange
        assert issubclass(EnhancedHaloExchange2, EnhancedHaloExchange)

    def test_exchange_compressed_first_call(self):
        """First compressed call does full exchange."""
        patch = ProcessorPatch(
            name="proc0",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([2, 3], dtype=INDEX_DTYPE),
            remote_cells=torch.tensor([0, 1], dtype=INDEX_DTYPE),
        )
        halo = EnhancedHaloExchange2([patch])
        field = torch.tensor([10.0, 20.0, 0.0, 0.0])
        nbr = torch.tensor([100.0, 200.0])

        result = halo.exchange_compressed(field, all_fields={1: nbr})
        assert result[2] == 100.0
        assert result[3] == 200.0

    def test_field_cache(self):
        """Field cache stores exchanged values."""
        patch = ProcessorPatch(
            name="proc0",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([2], dtype=INDEX_DTYPE),
            remote_cells=torch.tensor([0], dtype=INDEX_DTYPE),
        )
        halo = EnhancedHaloExchange2([patch])
        field = torch.tensor([1.0, 2.0, 0.0])
        nbr = torch.tensor([10.0])

        halo.exchange_compressed(field, all_fields={1: nbr})
        assert len(halo.field_cache) == 1

    def test_clear_cache(self):
        """clear_cache empties the cache."""
        patch = ProcessorPatch(
            name="proc0",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([2], dtype=INDEX_DTYPE),
            remote_cells=torch.tensor([0], dtype=INDEX_DTYPE),
        )
        halo = EnhancedHaloExchange2([patch])
        field = torch.tensor([1.0, 2.0, 0.0])
        nbr = torch.tensor([10.0])

        halo.exchange_compressed(field, all_fields={1: nbr})
        halo.clear_cache()
        assert len(halo.field_cache) == 0

    def test_exchange_multi_field_v2(self):
        """Batch exchange with v2 interface."""
        patch = ProcessorPatch(
            name="proc0",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([2], dtype=INDEX_DTYPE),
            remote_cells=torch.tensor([0], dtype=INDEX_DTYPE),
        )
        halo = EnhancedHaloExchange2([patch])

        fields = {
            "p": torch.tensor([101325.0, 101300.0, 0.0]),
            "U": torch.tensor([1.0, 0.5, 0.0]),
        }
        nbr_fields = {
            "p": torch.tensor([101325.0]),
            "U": torch.tensor([2.0]),
        }

        result = halo.exchange_multi_field_v2(fields, all_fields={1: nbr_fields})
        assert "p" in result
        assert "U" in result
