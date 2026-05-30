"""Tests for RedistributeParEnhanced5 — v5 enhanced redistribution."""

import pytest
import torch

from pyfoam.parallel.redistribute_par_enhanced_5 import (
    RedistributeParEnhanced5,
    V5RedistributeResult,
    MigrationPlan,
    CostEstimator,
)
from pyfoam.parallel.redistribute_par_enhanced_4 import RedistributeParEnhanced4


class TestCostEstimator:
    """Test CostEstimator."""

    def test_default_costs(self):
        est = CostEstimator()
        costs = est.estimate_costs()
        assert costs.numel() >= 1

    def test_solver_type(self):
        est = CostEstimator(solver_type="compressible")
        assert est.solver_type == "compressible"

    def test_volume_based(self):
        est = CostEstimator()
        volumes = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        costs = est.estimate_costs(cell_volumes=volumes)
        assert costs.shape == (3,)
        # Costs should scale with volume
        assert costs[2].item() > costs[0].item()

    def test_face_based(self):
        est = CostEstimator(base_cost=2.0)
        n_faces = torch.tensor([4, 6, 8], dtype=torch.float64)
        costs = est.estimate_costs(n_faces_per_cell=n_faces)
        assert costs.shape == (3,)

    def test_unknown_solver(self):
        est = CostEstimator(solver_type="unknown")
        assert est._multiplier == 1.0


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v4(self):
        assert issubclass(RedistributeParEnhanced5, RedistributeParEnhanced4)


class TestMigrationPlan:
    """Test migration planning."""

    def test_no_migrations(self, tmp_path):
        case_dir = str(tmp_path / "case")
        (tmp_path / "case").mkdir()
        redist = RedistributeParEnhanced5(case_dir, target_n_procs=2)

        old = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        new = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        plan = redist.compute_migration_plan(old, new)
        assert plan.n_migrations == 0

    def test_some_migrations(self, tmp_path):
        case_dir = str(tmp_path / "case")
        (tmp_path / "case").mkdir()
        redist = RedistributeParEnhanced5(case_dir, target_n_procs=2)

        old = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        new = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        plan = redist.compute_migration_plan(old, new)
        assert plan.n_migrations == 2  # cells 1 and 2 migrate


class TestRepartitionDecision:
    """Test repartitioning decision logic."""

    def test_recommended_when_imbalanced(self, tmp_path):
        case_dir = str(tmp_path / "case")
        (tmp_path / "case").mkdir()
        redist = RedistributeParEnhanced5(case_dir, target_n_procs=2)

        old = torch.tensor([0, 0, 0, 0], dtype=torch.long)
        new = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        costs = torch.ones(4, dtype=torch.float64)

        recommended, speedup = redist.should_repartition(
            old, new, costs, migration_threshold=1.0,
        )
        assert recommended is True
        assert speedup > 1.0

    def test_not_recommended_when_too_many_migrations(self, tmp_path):
        case_dir = str(tmp_path / "case")
        (tmp_path / "case").mkdir()
        redist = RedistributeParEnhanced5(case_dir, target_n_procs=2)

        old = torch.tensor([0, 0, 0, 0], dtype=torch.long)
        new = torch.tensor([1, 1, 1, 1], dtype=torch.long)
        costs = torch.ones(4, dtype=torch.float64)

        recommended, _ = redist.should_repartition(
            old, new, costs, migration_threshold=0.1,
        )
        assert recommended is False


class TestMigrationPlanDataclass:
    """Test MigrationPlan dataclass."""

    def test_creation(self):
        plan = MigrationPlan(
            cell_indices=torch.tensor([0, 2]),
            source_proc=torch.tensor([0, 1]),
            target_proc=torch.tensor([1, 0]),
            n_migrations=2,
        )
        assert plan.n_migrations == 2


class TestV5Result:
    """Test V5RedistributeResult dataclass."""

    def test_defaults(self):
        result = V5RedistributeResult(base=None)
        assert result.n_migrations == 0
        assert result.estimated_speedup == 1.0
        assert result.repartition_recommended is True


class TestRepr:
    """Test __repr__."""

    def test_repr(self, tmp_path):
        case_dir = str(tmp_path / "case")
        (tmp_path / "case").mkdir()
        redist = RedistributeParEnhanced5(case_dir, target_n_procs=4)
        r = repr(redist)
        assert "RedistributeParEnhanced5" in r
        assert "n_procs=4" in r
