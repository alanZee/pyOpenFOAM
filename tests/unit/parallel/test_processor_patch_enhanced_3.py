"""Tests for ProcessorPatchEnhanced3 — v3 enhanced processor patches."""

import pytest
import torch

from pyfoam.parallel.processor_patch_enhanced_3 import (
    NonConformalPatch3,
    EnhancedHaloExchange3,
    AdaptiveInterpolator,
)
from pyfoam.parallel.processor_patch_enhanced_2 import (
    NonConformalPatch2,
    EnhancedHaloExchange2,
)


class TestAdaptiveInterpolator:
    """Test AdaptiveInterpolator."""

    def test_compute_weights_basic(self):
        """Basic weight computation."""
        interp = AdaptiveInterpolator(min_nearest=1, max_nearest=3)
        local = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.float64)
        remote = torch.tensor([[0.1, 0, 0], [0.9, 0, 0], [2, 0, 0]], dtype=torch.float64)

        weights, indices = interp.compute_weights(local, remote)
        assert weights.shape[0] == 2
        assert indices.shape[0] == 2
        # Weights should sum to 1 for each local point
        assert torch.allclose(weights.sum(dim=1), torch.ones(2, dtype=torch.float64), atol=1e-10)

    def test_few_remote_faces(self):
        """When few remote faces, uses all."""
        interp = AdaptiveInterpolator(min_nearest=1, max_nearest=5)
        local = torch.tensor([[0, 0, 0]], dtype=torch.float64)
        remote = torch.tensor([[1, 0, 0], [2, 0, 0]], dtype=torch.float64)

        weights, indices = interp.compute_weights(local, remote)
        # Should use 2 nearest (fewer than max)
        assert weights.shape[1] == 2


class TestInheritance:
    """Test class hierarchy."""

    def test_patch_inherits(self):
        assert issubclass(NonConformalPatch3, NonConformalPatch2)

    def test_halo_inherits(self):
        assert issubclass(EnhancedHaloExchange3, EnhancedHaloExchange2)


class TestNonConformalPatch3:
    """Test NonConformalPatch3."""

    def test_creation_defaults(self):
        """Default values for new fields."""
        patch = NonConformalPatch3(
            name="test",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0]),
            remote_cells=torch.tensor([0]),
        )
        assert patch.n_layers == 1
        assert patch.layer_indices is None

    def test_creation_with_layers(self):
        """Multi-layer ghost cell setup."""
        patch = NonConformalPatch3(
            name="test",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0, 1]),
            remote_cells=torch.tensor([2, 3]),
            n_layers=2,
            layer_indices=[
                torch.tensor([0, 1]),
                torch.tensor([2, 3]),
            ],
        )
        assert patch.n_layers == 2
        assert len(patch.layer_indices) == 2

    def test_adaptive_weights(self):
        """Adaptive weight computation."""
        patch = NonConformalPatch3(
            name="test",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0, 1]),
            remote_cells=torch.tensor([2, 3]),
            local_face_centres=torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.float64),
            remote_face_centres=torch.tensor([[0.1, 0, 0], [0.9, 0, 0]], dtype=torch.float64),
        )
        patch.auto_compute_weights_adaptive(min_nearest=1, max_nearest=2)
        assert patch.local_weights is not None
        assert patch.local_src_indices is not None

    def test_adaptive_weights_no_centres_raises(self):
        """ValueError when face centres not set."""
        patch = NonConformalPatch3(
            name="test",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0]),
            remote_cells=torch.tensor([0]),
        )
        with pytest.raises(ValueError, match="Face centres"):
            patch.auto_compute_weights_adaptive()


class TestEnhancedHaloExchange3:
    """Test EnhancedHaloExchange3."""

    def test_multilayer_single_layer(self):
        """Single-layer exchange works like standard exchange."""
        local_idx = torch.tensor([0, 1])
        remote_idx = torch.tensor([4, 5])
        weights = torch.tensor([[1.0], [1.0]])
        src_idx = torch.tensor([[0], [1]])

        patch = NonConformalPatch3(
            name="proc0To1",
            neighbour_rank=1,
            local_ghost_cells=local_idx,
            remote_cells=remote_idx,
            local_weights=weights,
            local_src_indices=src_idx,
            n_layers=1,
        )

        halo = EnhancedHaloExchange3([patch])
        field = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=torch.float64)
        all_fields = {
            0: field,
            1: torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0, 600.0], dtype=torch.float64),
        }

        result = halo.exchange_multilayer(field, n_layers=1, all_fields=all_fields)
        assert result.shape == field.shape

    def test_detect_overlaps(self):
        """Overlap detection finds common indices."""
        halo = EnhancedHaloExchange3([])
        ghost1 = torch.tensor([0, 1, 2, 3])
        ghost2 = torch.tensor([2, 3, 4, 5])

        overlaps = halo.detect_overlaps([ghost1, ghost2])
        assert len(overlaps) == 1
        assert torch.allclose(overlaps[0], torch.tensor([2, 3]))

    def test_detect_no_overlaps(self):
        """No overlaps returns empty list."""
        halo = EnhancedHaloExchange3([])
        ghost1 = torch.tensor([0, 1])
        ghost2 = torch.tensor([2, 3])

        overlaps = halo.detect_overlaps([ghost1, ghost2])
        assert len(overlaps) == 0

    def test_repr(self):
        halo = EnhancedHaloExchange3([])
        r = repr(halo)
        assert "EnhancedHaloExchange3" in r
