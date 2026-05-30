"""Tests for RedistributeParEnhanced10 -- v10 enhanced redistribution."""

import pytest
import torch

from pyfoam.parallel.redistribute_par_enhanced_9 import RedistributeParEnhanced9
from pyfoam.parallel.redistribute_par_enhanced_10 import (
    RedistributeParEnhanced10,
    V10RedistributeResult,
    ParetoConfig,
    MultiLevelConfig,
    StabilityMetrics,
    _LoadPredictor,
)


class TestParetoConfig:
    def test_defaults(self):
        cfg = ParetoConfig()
        assert cfg.n_objectives == 3
        assert cfg.balance_weight == 0.4


class TestMultiLevelConfig:
    def test_defaults(self):
        cfg = MultiLevelConfig()
        assert cfg.n_levels == 3
        assert cfg.coarsening_ratio == 0.5


class TestStabilityMetrics:
    def test_defaults(self):
        m = StabilityMetrics()
        assert m.stability_score == 1.0
        assert m.cells_moved == 0


class TestInheritance:
    def test_inherits_v9(self):
        assert issubclass(RedistributeParEnhanced10, RedistributeParEnhanced9)


class TestLoadPredictor:
    def test_first_prediction(self):
        pred = _LoadPredictor()
        val = pred.predict(100.0)
        assert val == 100.0

    def test_smoothing(self):
        pred = _LoadPredictor(alpha=0.5)
        pred.predict(100.0)
        val = pred.predict(200.0)
        assert val == 150.0  # 0.5 * 200 + 0.5 * 100

    def test_reset(self):
        pred = _LoadPredictor()
        pred.predict(100.0)
        pred.reset()
        assert pred._prev_prediction is None


class TestStabilityComputation:
    def test_identical_mapping(self):
        old = torch.zeros(100, dtype=torch.long)
        new = torch.zeros(100, dtype=torch.long)
        stability = RedistributeParEnhanced10.compute_stability(old, new)
        assert stability.stability_score == 1.0
        assert stability.cells_moved == 0

    def test_all_moved(self):
        old = torch.zeros(100, dtype=torch.long)
        new = torch.ones(100, dtype=torch.long)
        stability = RedistributeParEnhanced10.compute_stability(old, new)
        assert stability.stability_score == 0.0
        assert stability.cells_moved == 100


class TestV10Redistribution:
    def test_returns_v10_result(self, tmp_path):
        proc0 = tmp_path / "processor0"
        proc0.mkdir()
        redist = RedistributeParEnhanced10(tmp_path, target_n_procs=2)
        result = redist.redistribute_v10()
        assert isinstance(result, V10RedistributeResult)

    def test_with_stability(self, tmp_path):
        proc0 = tmp_path / "processor0"
        proc0.mkdir()
        redist = RedistributeParEnhanced10(tmp_path, target_n_procs=2)
        centres = torch.randn(50, 3, dtype=torch.float64)
        old_mapping = torch.zeros(50, dtype=torch.long)
        result = redist.redistribute_v10(
            cell_centres=centres,
            current_mapping=old_mapping,
        )
        assert result.stability is not None
        assert result.stability.stability_score >= 0

    def test_with_pareto(self, tmp_path):
        proc0 = tmp_path / "processor0"
        proc0.mkdir()
        redist = RedistributeParEnhanced10(tmp_path, target_n_procs=2)
        result = redist.redistribute_v10(pareto_optimise=True)
        assert result.pareto_front_size > 0

    def test_with_load_prediction(self, tmp_path):
        proc0 = tmp_path / "processor0"
        proc0.mkdir()
        redist = RedistributeParEnhanced10(tmp_path, target_n_procs=2)
        costs = torch.ones(100, dtype=torch.float64)
        result = redist.redistribute_v10(
            cell_costs=costs,
            predict_load=True,
        )
        assert result.predicted_load_imbalance >= 0


class TestRepr:
    def test_repr(self, tmp_path):
        redist = RedistributeParEnhanced10(tmp_path, target_n_procs=4)
        r = repr(redist)
        assert "RedistributeParEnhanced10" in r
