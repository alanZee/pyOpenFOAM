"""Tests for RedistributeParEnhanced8 -- v8 enhanced redistribution."""

import pytest
import torch

from pyfoam.parallel.redistribute_par_enhanced_7 import RedistributeParEnhanced7
from pyfoam.parallel.redistribute_par_enhanced_8 import (
    RedistributeParEnhanced8,
    V8RedistributeResult,
    MultiObjectiveConfig,
    PartitionFingerprint,
    IncrementalPlan,
)


class TestMultiObjectiveConfig:
    """Test MultiObjectiveConfig dataclass."""

    def test_defaults(self):
        cfg = MultiObjectiveConfig()
        assert cfg.weight_balance == 0.4
        assert cfg.weight_communication == 0.3
        assert cfg.weight_locality == 0.3


class TestPartitionFingerprint:
    """Test PartitionFingerprint dataclass."""

    def test_defaults(self):
        fp = PartitionFingerprint()
        assert fp.hash_value == ""
        assert fp.n_cells == 0


class TestIncrementalPlan:
    """Test IncrementalPlan dataclass."""

    def test_defaults(self):
        plan = IncrementalPlan()
        assert plan.n_migrated == 0
        assert plan.savings_ratio == 0.0


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v7(self):
        assert issubclass(RedistributeParEnhanced8, RedistributeParEnhanced7)


class TestFingerprinting:
    """Test partition fingerprinting."""

    def test_fingerprint_deterministic(self):
        mapping = torch.tensor([0, 0, 1, 1, 0])
        fp1 = RedistributeParEnhanced8.compute_fingerprint(mapping, 2)
        fp2 = RedistributeParEnhanced8.compute_fingerprint(mapping, 2)
        assert fp1.hash_value == fp2.hash_value
        assert fp1.n_cells == 5

    def test_fingerprint_different_mapping(self):
        mapping1 = torch.tensor([0, 0, 1, 1])
        mapping2 = torch.tensor([0, 1, 0, 1])
        fp1 = RedistributeParEnhanced8.compute_fingerprint(mapping1, 2)
        fp2 = RedistributeParEnhanced8.compute_fingerprint(mapping2, 2)
        assert fp1.hash_value != fp2.hash_value


class TestIncrementalPlanComputation:
    """Test incremental plan computation."""

    def test_no_migration_needed(self):
        mapping = torch.tensor([0, 0, 1, 1])
        plan = RedistributeParEnhanced8.compute_incremental_plan(mapping, mapping)
        assert plan.n_migrated == 0
        assert plan.savings_ratio == pytest.approx(1.0)

    def test_partial_migration(self):
        old = torch.tensor([0, 0, 1, 1])
        new = torch.tensor([0, 1, 1, 1])
        plan = RedistributeParEnhanced8.compute_incremental_plan(old, new)
        assert plan.n_migrated == 1
        assert plan.savings_ratio == pytest.approx(0.75)

    def test_full_migration(self):
        old = torch.tensor([0, 0, 1, 1])
        new = torch.tensor([1, 1, 0, 0])
        plan = RedistributeParEnhanced8.compute_incremental_plan(old, new)
        assert plan.n_migrated == 4
        assert plan.savings_ratio == pytest.approx(0.0)


class TestMultiObjectiveScoring:
    """Test multi-objective scoring."""

    def test_perfect_score(self):
        cfg = MultiObjectiveConfig()
        score = RedistributeParEnhanced8.compute_multi_objective_score(
            0.0, 0.0, 1.0, cfg
        )
        assert score == pytest.approx(0.0)

    def test_worst_score(self):
        cfg = MultiObjectiveConfig()
        score = RedistributeParEnhanced8.compute_multi_objective_score(
            2.0, 2.0, 0.0, cfg
        )
        assert score > 0.0


class TestV8Redistribution:
    """Test v8 redistribution method."""

    def test_returns_v8_result(self):
        redist = RedistributeParEnhanced8("/tmp/test_nonexist", target_n_procs=4)
        result = redist.redistribute_v8()
        assert isinstance(result, V8RedistributeResult)

    def test_incremental_flag(self):
        redist = RedistributeParEnhanced8("/tmp/test_nonexist", target_n_procs=4)
        old = torch.tensor([0, 0, 1, 1])
        new = torch.tensor([0, 1, 1, 1])
        redist._previous_mapping = old
        result = redist.redistribute_v8(
            current_mapping=new,
            incremental=True,
        )
        assert result.migration_savings > 0


class TestRepr:
    """Test string representations."""

    def test_repr(self):
        redist = RedistributeParEnhanced8("/tmp/test", 4)
        r = repr(redist)
        assert "RedistributeParEnhanced8" in r
