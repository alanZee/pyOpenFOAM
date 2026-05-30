"""Tests for ProcessorPatchEnhanced8 -- v8 enhanced processor patches."""

import pytest
import torch

from pyfoam.parallel.processor_patch_enhanced_7 import EnhancedHaloExchange7
from pyfoam.parallel.processor_patch_enhanced_8 import (
    SparseAwarePatch8,
    EnhancedHaloExchange8,
    BatchedExchangeConfig,
    LatencyProfile,
    FaultToleranceConfig,
    _LatencyTracker,
)


class TestBatchedExchangeConfig:
    """Test BatchedExchangeConfig dataclass."""

    def test_defaults(self):
        cfg = BatchedExchangeConfig()
        assert cfg.max_batch_size == 16
        assert cfg.use_compression is True


class TestFaultToleranceConfig:
    """Test FaultToleranceConfig dataclass."""

    def test_defaults(self):
        cfg = FaultToleranceConfig()
        assert cfg.max_retries == 3
        assert cfg.fallback_to_serial is True


class TestLatencyProfile:
    """Test LatencyProfile dataclass."""

    def test_defaults(self):
        profile = LatencyProfile()
        assert profile.mean_latency_ms == 0.0
        assert profile.n_samples == 0


class TestSparseAwarePatch:
    """Test SparseAwarePatch8."""

    def test_sparsity_full(self):
        patch = SparseAwarePatch8(
            name="test",
            neighbour_rank=1,
        )
        field = torch.zeros(100)
        sparsity = patch.compute_sparsity(field)
        assert sparsity == pytest.approx(1.0)

    def test_sparsity_none(self):
        patch = SparseAwarePatch8(name="test", neighbour_rank=1)
        field = torch.ones(100)
        sparsity = patch.compute_sparsity(field)
        assert sparsity == pytest.approx(0.0)

    def test_sparse_transfer_decision(self):
        patch = SparseAwarePatch8(
            name="test",
            neighbour_rank=1,
            sparsity_threshold=0.5,
        )
        patch.compute_sparsity(torch.zeros(100))
        assert patch.use_sparse_transfer() is True

        patch.compute_sparsity(torch.ones(100))
        assert patch.use_sparse_transfer() is False


class TestLatencyTracker:
    """Test _LatencyTracker."""

    def test_empty_profile(self):
        tracker = _LatencyTracker()
        profile = tracker.profile()
        assert profile.n_samples == 0

    def test_recording(self):
        tracker = _LatencyTracker()
        tracker.record(1.0)
        tracker.record(2.0)
        profile = tracker.profile()
        assert profile.n_samples == 2
        assert profile.mean_latency_ms == pytest.approx(1.5)

    def test_reset(self):
        tracker = _LatencyTracker()
        tracker.record(1.0)
        tracker.reset()
        profile = tracker.profile()
        assert profile.n_samples == 0


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v7(self):
        assert issubclass(EnhancedHaloExchange8, EnhancedHaloExchange7)


class TestBatchedExchange:
    """Test batched exchange."""

    def test_exchange_count(self):
        halo = EnhancedHaloExchange8(patches=[])
        # With no patches, exchange_adaptive just returns field
        fields = {"p": torch.randn(10), "U_x": torch.randn(10)}
        results = halo.exchange_batched(fields)
        assert "p" in results
        assert "U_x" in results
        assert halo.exchange_count == 2


class TestRetryExchange:
    """Test fault-tolerant exchange."""

    def test_exchange_no_patches(self):
        halo = EnhancedHaloExchange8(patches=[])
        field = torch.randn(10)
        result = halo.exchange_with_retry(field)
        assert result.shape == field.shape
        assert halo.retry_count == 0


class TestRepr:
    """Test string representations."""

    def test_repr(self):
        halo = EnhancedHaloExchange8(patches=[])
        r = repr(halo)
        assert "EnhancedHaloExchange8" in r
