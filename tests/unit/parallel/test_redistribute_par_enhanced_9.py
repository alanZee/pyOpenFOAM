"""Tests for RedistributeParEnhanced9 -- v9 enhanced redistribution."""

import pytest
import torch

from pyfoam.parallel.redistribute_par_enhanced_8 import RedistributeParEnhanced8
from pyfoam.parallel.redistribute_par_enhanced_9 import (
    RedistributeParEnhanced9,
    V9RedistributeResult,
    BandwidthScheduleConfig,
    OnlineCostPrediction,
    PartitionQualityMetrics,
)


class TestBandwidthScheduleConfig:
    """Test BandwidthScheduleConfig dataclass."""

    def test_defaults(self):
        cfg = BandwidthScheduleConfig()
        assert cfg.bandwidth_gbps == 10.0
        assert cfg.overlap_computation is True


class TestOnlineCostPrediction:
    """Test OnlineCostPrediction dataclass."""

    def test_defaults(self):
        pred = OnlineCostPrediction()
        assert pred.predicted_cost == 0.0
        assert pred.confidence == 0.0


class TestPartitionQualityMetrics:
    """Test PartitionQualityMetrics dataclass."""

    def test_defaults(self):
        m = PartitionQualityMetrics()
        assert m.load_imbalance == 0.0
        assert m.overall_quality == 0.0


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v8(self):
        assert issubclass(RedistributeParEnhanced9, RedistributeParEnhanced8)


class TestGPUMapping:
    """Test GPU-accelerated cell mapping."""

    def test_mapping_shape(self):
        centres = torch.randn(100, 3, dtype=torch.float64)
        mapping = RedistributeParEnhanced9.compute_gpu_mapping(centres, n_procs=4)
        assert mapping.shape == (100,)
        assert mapping.min() >= 0
        assert mapping.max() < 4


class TestCostPrediction:
    """Test migration cost prediction."""

    def test_prediction(self):
        pred = RedistributeParEnhanced9.predict_migration_cost(
            n_cells_to_migrate=1000,
            n_fields=2,
            bandwidth_gbps=10.0,
        )
        assert pred.predicted_time > 0
        assert pred.n_cells_to_migrate == 1000
        assert pred.confidence > 0

    def test_zero_migration(self):
        pred = RedistributeParEnhanced9.predict_migration_cost(
            n_cells_to_migrate=0,
            n_fields=2,
        )
        assert pred.predicted_time >= 0
        assert pred.confidence == 0.0


class TestPartitionQuality:
    """Test partition quality evaluation."""

    def test_perfect_balance(self):
        mapping = torch.zeros(100, dtype=torch.long)
        mapping[:50] = 0
        mapping[50:] = 1
        costs = torch.ones(100, dtype=torch.float64)
        quality = RedistributeParEnhanced9.evaluate_partition_quality(
            mapping, costs, n_procs=2
        )
        assert quality.load_imbalance == 0.0
        assert quality.overall_quality > 0.5

    def test_imbalanced(self):
        mapping = torch.zeros(100, dtype=torch.long)
        mapping[:90] = 0
        mapping[90:] = 1
        costs = torch.ones(100, dtype=torch.float64)
        quality = RedistributeParEnhanced9.evaluate_partition_quality(
            mapping, costs, n_procs=2
        )
        assert quality.load_imbalance > 0


class TestV9Redistribution:
    """Test v9 redistribution method."""

    def test_returns_v9_result(self, tmp_path):
        proc0 = tmp_path / "processor0"
        proc0.mkdir()
        redist = RedistributeParEnhanced9(tmp_path, target_n_procs=2)
        result = redist.redistribute_v9()
        assert isinstance(result, V9RedistributeResult)


class TestRepr:
    """Test string representations."""

    def test_repr(self, tmp_path):
        redist = RedistributeParEnhanced9(tmp_path, target_n_procs=4)
        r = repr(redist)
        assert "RedistributeParEnhanced9" in r
