"""Tests for ProcessorPatchEnhanced9 -- v9 enhanced processor patches."""

import pytest
import torch

from pyfoam.parallel.processor_patch_enhanced_8 import EnhancedHaloExchange8
from pyfoam.parallel.processor_patch_enhanced_9 import (
    TopologyAwarePatch9,
    EnhancedHaloExchange9,
    TopologyRoutingConfig,
    CoalescingConfig,
    CheckpointConfig,
    _CoalescingTracker,
    _CheckpointManager,
)


class TestTopologyRoutingConfig:
    """Test TopologyRoutingConfig dataclass."""

    def test_defaults(self):
        cfg = TopologyRoutingConfig()
        assert cfg.max_hops == 3
        assert cfg.enable_multi_hop is True


class TestCoalescingConfig:
    """Test CoalescingConfig dataclass."""

    def test_defaults(self):
        cfg = CoalescingConfig()
        assert cfg.max_coalesce_size == 4 * 1024 * 1024
        assert cfg.enable_adaptive is True


class TestCheckpointConfig:
    """Test CheckpointConfig dataclass."""

    def test_defaults(self):
        cfg = CheckpointConfig()
        assert cfg.enable_checkpoint is False
        assert cfg.checkpoint_interval == 100


class TestTopologyAwarePatch9:
    """Test TopologyAwarePatch9 dataclass."""

    def test_defaults(self):
        patch = TopologyAwarePatch9(
            name="proc0To1",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0]),
            remote_cells=torch.tensor([0]),
        )
        assert patch.hop_count == 1
        assert patch.is_multi_hop is False

    def test_route_through(self):
        patch = TopologyAwarePatch9(
            name="proc0To1",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0]),
            remote_cells=torch.tensor([0]),
        )
        patch.route_through(2)
        assert patch.hop_count == 2
        assert patch.is_multi_hop is True
        assert patch.intermediate_ranks == [2]


class TestCoalescingTracker:
    """Test coalescing tracker."""

    def test_small_message_coalesced(self):
        cfg = CoalescingConfig(max_coalesce_size=1024)
        tracker = _CoalescingTracker(cfg)
        assert tracker.should_coalesce(512) is True
        assert tracker.coalesced_count == 1

    def test_large_message_not_coalesced(self):
        cfg = CoalescingConfig(max_coalesce_size=1024)
        tracker = _CoalescingTracker(cfg)
        assert tracker.should_coalesce(2048) is False
        assert tracker.coalesced_count == 0

    def test_coalescing_ratio(self):
        cfg = CoalescingConfig(max_coalesce_size=1024)
        tracker = _CoalescingTracker(cfg)
        tracker.should_coalesce(512)   # coalesced
        tracker.should_coalesce(2048)  # not coalesced
        assert abs(tracker.coalescing_ratio - 0.5) < 1e-10


class TestCheckpointManager:
    """Test checkpoint manager."""

    def test_disabled_checkpoint(self):
        cfg = CheckpointConfig(enable_checkpoint=False)
        mgr = _CheckpointManager(cfg)
        assert mgr.should_checkpoint() is False

    def test_enabled_checkpoint(self):
        cfg = CheckpointConfig(enable_checkpoint=True, checkpoint_interval=2)
        mgr = _CheckpointManager(cfg)
        assert mgr.should_checkpoint() is False  # count=1
        assert mgr.should_checkpoint() is True   # count=2
        assert mgr.n_checkpoints == 0  # No save yet

    def test_save_and_restore(self):
        cfg = CheckpointConfig(enable_checkpoint=True, checkpoint_interval=1)
        mgr = _CheckpointManager(cfg)
        state = {"p": torch.tensor([1.0, 2.0]), "U": torch.tensor([3.0, 4.0])}
        mgr.save_checkpoint(state)
        restored = mgr.get_latest_checkpoint()
        assert restored is not None
        assert torch.allclose(restored["p"], state["p"])

    def test_no_checkpoint_returns_none(self):
        cfg = CheckpointConfig()
        mgr = _CheckpointManager(cfg)
        assert mgr.get_latest_checkpoint() is None


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v8(self):
        assert issubclass(EnhancedHaloExchange9, EnhancedHaloExchange8)


class TestEnhancedHaloExchange9:
    """Test EnhancedHaloExchange9."""

    def test_default_config(self):
        halo = EnhancedHaloExchange9([])
        assert halo.coalesced_count == 0
        assert halo.checkpoint_count == 0

    def test_repr(self):
        halo = EnhancedHaloExchange9([])
        r = repr(halo)
        assert "EnhancedHaloExchange9" in r
