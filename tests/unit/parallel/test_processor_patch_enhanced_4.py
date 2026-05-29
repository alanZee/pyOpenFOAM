"""Tests for ProcessorPatchEnhanced4 — v4 enhanced processor patches."""

import pytest
import torch

from pyfoam.parallel.processor_patch_enhanced_4 import (
    NonConformalPatch4,
    EnhancedHaloExchange4,
    HigherOrderInterpolator,
)
from pyfoam.parallel.processor_patch_enhanced_3 import (
    NonConformalPatch3,
    EnhancedHaloExchange3,
)


class TestHigherOrderInterpolator:
    """Test HigherOrderInterpolator."""

    def test_compute_weights_basic(self):
        """Basic weight computation."""
        interp = HigherOrderInterpolator(power=2.0, n_nearest=2)
        local = torch.tensor([[0, 0, 0]], dtype=torch.float64)
        remote = torch.tensor([[1, 0, 0], [2, 0, 0]], dtype=torch.float64)

        weights, indices = interp.compute_weights(local, remote)
        assert weights.shape == (1, 2)
        assert indices.shape == (1, 2)
        assert torch.allclose(weights.sum(dim=1), torch.ones(1, dtype=torch.float64), atol=1e-10)

    def test_nearest_gets_highest_weight(self):
        """Nearest point gets the highest weight."""
        interp = HigherOrderInterpolator(power=2.0, n_nearest=2)
        local = torch.tensor([[0, 0, 0]], dtype=torch.float64)
        remote = torch.tensor([[0.1, 0, 0], [10.0, 0, 0]], dtype=torch.float64)

        weights, indices = interp.compute_weights(local, remote)
        # Nearest (0.1) should have higher weight than far (10.0)
        assert weights[0, 0].item() > weights[0, 1].item()

    def test_higher_power_more_localised(self):
        """Higher power concentrates weight on nearest."""
        local = torch.tensor([[0, 0, 0]], dtype=torch.float64)
        remote = torch.tensor([[1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=torch.float64)

        interp1 = HigherOrderInterpolator(power=1.0, n_nearest=3)
        w1, _ = interp1.compute_weights(local, remote)

        interp3 = HigherOrderInterpolator(power=3.0, n_nearest=3)
        w3, _ = interp3.compute_weights(local, remote)

        # Higher power should give more weight to nearest
        assert w3[0, 0].item() > w1[0, 0].item()


class TestInheritance:
    """Test class hierarchy."""

    def test_patch_inherits(self):
        assert issubclass(NonConformalPatch4, NonConformalPatch3)

    def test_halo_inherits(self):
        assert issubclass(EnhancedHaloExchange4, EnhancedHaloExchange3)


class TestNonConformalPatch4:
    """Test NonConformalPatch4."""

    def test_creation_defaults(self):
        """Default values for new fields."""
        patch = NonConformalPatch4(
            name="test",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0]),
            remote_cells=torch.tensor([0]),
        )
        assert patch.interpolation_order == 2.0

    def test_higher_order_weights(self):
        """Higher-order weight computation."""
        patch = NonConformalPatch4(
            name="test",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0, 1]),
            remote_cells=torch.tensor([2, 3]),
            local_face_centres=torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.float64),
            remote_face_centres=torch.tensor([[0.1, 0, 0], [0.9, 0, 0]], dtype=torch.float64),
        )
        patch.auto_compute_weights_higher_order(order=3.0, n_nearest=2)
        assert patch.local_weights is not None
        assert patch.interpolation_order == 3.0

    def test_higher_order_no_centres_raises(self):
        """ValueError when face centres not set."""
        patch = NonConformalPatch4(
            name="test",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0]),
            remote_cells=torch.tensor([0]),
        )
        with pytest.raises(ValueError, match="Face centres"):
            patch.auto_compute_weights_higher_order()


class TestEnhancedHaloExchange4:
    """Test EnhancedHaloExchange4."""

    def test_exchange_basic(self):
        """Basic exchange works."""
        weights = torch.tensor([[1.0], [1.0]])
        src_idx = torch.tensor([[0], [1]])
        patch = NonConformalPatch4(
            name="proc0To1",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0, 1]),
            remote_cells=torch.tensor([4, 5]),
            local_weights=weights,
            local_src_indices=src_idx,
        )
        halo = EnhancedHaloExchange4([patch])
        field = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=torch.float64)
        all_fields = {
            0: field,
            1: torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0, 600.0], dtype=torch.float64),
        }
        result = halo.exchange_conservative(field, all_fields)
        assert result.shape == field.shape

    def test_exchange_conservative_with_areas(self):
        """Conservative exchange with cell areas."""
        weights = torch.tensor([[1.0], [1.0]])
        src_idx = torch.tensor([[0], [1]])
        patch = NonConformalPatch4(
            name="proc0To1",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0, 1]),
            remote_cells=torch.tensor([4, 5]),
            local_weights=weights,
            local_src_indices=src_idx,
        )
        halo = EnhancedHaloExchange4([patch])
        field = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=torch.float64)
        areas = torch.ones(6, dtype=torch.float64)
        all_fields = {
            0: field,
            1: field.clone(),
        }
        result = halo.exchange_conservative(field, all_fields, areas=areas)
        assert result.shape == field.shape

    def test_periodic_exchange(self):
        """Periodic exchange adds offset."""
        weights = torch.tensor([[1.0], [1.0]])
        src_idx = torch.tensor([[0], [1]])
        patch = NonConformalPatch4(
            name="proc0To1",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0, 1]),
            remote_cells=torch.tensor([4, 5]),
            local_weights=weights,
            local_src_indices=src_idx,
        )
        halo = EnhancedHaloExchange4([patch])
        field = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=torch.float64)
        all_fields = {
            0: field,
            1: field.clone(),
        }
        result = halo.exchange_periodic(field, 5.0, all_fields)
        # Ghost cells should have the offset added
        assert result[0].item() != field[0].item() or result[1].item() != field[1].item()

    def test_repr(self):
        halo = EnhancedHaloExchange4([])
        r = repr(halo)
        assert "EnhancedHaloExchange4" in r
