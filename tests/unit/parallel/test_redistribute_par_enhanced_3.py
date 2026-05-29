"""Tests for RedistributeParEnhanced3 — v3 enhanced redistribution."""

import pytest
import torch

from pyfoam.parallel.redistribute_par_enhanced_3 import (
    RedistributeParEnhanced3,
    SpatialDecompositionStrategy,
    V3RedistributeResult,
)
from pyfoam.parallel.redistribute_par_enhanced_2 import RedistributeParEnhanced2


class TestSpatialDecompositionStrategy:
    """Test SpatialDecompositionStrategy constants."""

    def test_all_strategies(self):
        strategies = SpatialDecompositionStrategy.all_strategies()
        assert "spatial_rcb" in strategies
        assert "spatial_rib" in strategies
        assert "spatial_scatter" in strategies

    def test_strategy_constants(self):
        assert SpatialDecompositionStrategy.RCB == "spatial_rcb"
        assert SpatialDecompositionStrategy.RIB == "spatial_rib"
        assert SpatialDecompositionStrategy.SCATTER == "spatial_scatter"


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v2(self):
        assert issubclass(RedistributeParEnhanced3, RedistributeParEnhanced2)


class TestRCBPartitioning:
    """Test Recursive Coordinate Bisection."""

    def test_rcb_basic(self, tmp_path):
        """RCB produces a valid mapping."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced3(case_dir, target_n_procs=2)

        # 8 cells in a line along x
        centres = torch.tensor([
            [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],
            [4, 0, 0], [5, 0, 0], [6, 0, 0], [7, 0, 0],
        ], dtype=torch.float64)
        redist.set_cell_centres(centres)

        mapping = redist.compute_rcb_mapping(8)
        assert mapping.shape == (8,)
        assert mapping.min() >= 0
        assert mapping.max() < 2

    def test_rcb_no_centres_fallback(self, tmp_path):
        """RCB without centres falls back to round-robin."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced3(case_dir, target_n_procs=2)
        mapping = redist.compute_rcb_mapping(8)
        assert mapping.shape == (8,)

    def test_rcb_four_procs(self, tmp_path):
        """RCB with 4 processors."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced3(case_dir, target_n_procs=4)

        centres = torch.tensor([
            [i * 0.1, j * 0.1, k * 0.1]
            for k in range(2)
            for j in range(2)
            for i in range(2)
        ], dtype=torch.float64)
        redist.set_cell_centres(centres)

        mapping = redist.compute_rcb_mapping(8)
        assert mapping.shape == (8,)
        assert mapping.min() >= 0
        assert mapping.max() < 4


class TestRIBPartitioning:
    """Test Recursive Inertial Bisection."""

    def test_rib_basic(self, tmp_path):
        """RIB produces a valid mapping."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced3(case_dir, target_n_procs=2)

        centres = torch.tensor([
            [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],
            [0, 1, 0], [1, 1, 0], [2, 1, 0], [3, 1, 0],
        ], dtype=torch.float64)
        redist.set_cell_centres(centres)

        mapping = redist.compute_rib_mapping(8)
        assert mapping.shape == (8,)
        assert mapping.min() >= 0
        assert mapping.max() < 2

    def test_rib_no_centres_fallback(self, tmp_path):
        """RIB without centres falls back to round-robin."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced3(case_dir, target_n_procs=2)
        mapping = redist.compute_rib_mapping(8)
        assert mapping.shape == (8,)


class TestScatterMapping:
    """Test spatial scatter decomposition."""

    def test_scatter_basic(self, tmp_path):
        """Scatter produces a valid mapping."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced3(case_dir, target_n_procs=2)

        centres = torch.tensor([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        ], dtype=torch.float64)
        redist.set_cell_centres(centres)

        mapping = redist.compute_scatter_mapping(4)
        assert mapping.shape == (4,)
        assert mapping.min() >= 0
        assert mapping.max() < 2

    def test_scatter_no_centres_fallback(self, tmp_path):
        """Scatter without centres falls back to round-robin."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced3(case_dir, target_n_procs=2)
        mapping = redist.compute_scatter_mapping(4)
        assert mapping.shape == (4,)


class TestZonePreservation:
    """Test zone-aware redistribution."""

    def test_preserve_zones_no_zones(self, tmp_path):
        """No zone map: nothing preserved."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced3(case_dir, target_n_procs=2)

        mapping = torch.tensor([0, 0, 1, 1])
        result, n_preserved = redist.preserve_zones(mapping)
        assert n_preserved == 0

    def test_preserve_zones_consolidate(self, tmp_path):
        """Zone with dominant processor gets consolidated."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced3(case_dir, target_n_procs=2)

        # Zone "A" has cells 0, 1, 2. 3 out of 4 on proc 0
        mapping = torch.tensor([0, 0, 0, 1])
        redist.set_zone_map({"A": torch.tensor([0, 1, 2, 3])})

        result, n_preserved = redist.preserve_zones(mapping)
        assert n_preserved == 1
        # All zone cells should be on proc 0
        assert (result[torch.tensor([0, 1, 2, 3])] == 0).all()

    def test_repr(self, tmp_path):
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced3(case_dir, target_n_procs=4)
        r = repr(redist)
        assert "RedistributeParEnhanced3" in r
        assert "n_procs=4" in r
