"""Tests for RedistributeParEnhanced — enhanced redistribution strategies."""

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.parallel.redistribute_par_enhanced import (
    RedistributeParEnhanced,
    BalancingStrategy,
    PartitionDiagnostics,
    EnhancedRedistributeResult,
)
from pyfoam.parallel.redistribute_par import RedistributeResult

from tests.unit.parallel.conftest import make_8cell_fv_mesh


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------


class TestBalancingStrategy:
    """Test BalancingStrategy constants."""

    def test_all_strategies(self):
        """All strategy names are listed."""
        strategies = BalancingStrategy.all_strategies()
        assert "round_robin" in strategies
        assert "greedy" in strategies
        assert "spatial" in strategies
        assert "random" in strategies

    def test_strategy_constants(self):
        """Strategy constants are strings."""
        assert BalancingStrategy.ROUND_ROBIN == "round_robin"
        assert BalancingStrategy.GREEDY == "greedy"
        assert BalancingStrategy.SPATIAL == "spatial"
        assert BalancingStrategy.RANDOM == "random"


# ---------------------------------------------------------------------------
# Mapping tests (no case_dir needed)
# ---------------------------------------------------------------------------


class TestMappingStrategies:
    """Test various mapping strategies."""

    def test_round_robin(self, tmp_path):
        """Round-robin mapping distributes evenly."""
        case_dir = str(tmp_path / "empty_case")
        # Create minimal structure
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced(case_dir, target_n_procs=4)
        mapping = redist.compute_balanced_mapping(100, strategy="round_robin")

        assert mapping.shape == (100,)
        counts = torch.bincount(mapping, minlength=4)
        assert (counts == 25).all()

    def test_random_mapping(self, tmp_path):
        """Random mapping is reproducible with same seed."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced(case_dir, target_n_procs=2)
        m1 = redist.compute_random_mapping(50, seed=42)
        m2 = redist.compute_random_mapping(50, seed=42)

        assert torch.equal(m1, m2)

    def test_random_mapping_different_seeds(self, tmp_path):
        """Different seeds produce different mappings."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced(case_dir, target_n_procs=2)
        m1 = redist.compute_random_mapping(50, seed=42)
        m2 = redist.compute_random_mapping(50, seed=99)

        assert not torch.equal(m1, m2)

    def test_greedy_with_weights(self, tmp_path):
        """Greedy mapping respects cell weights."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced(case_dir, target_n_procs=2)
        weights = torch.zeros(10, dtype=torch.float64)
        weights[0] = 100.0  # One heavy cell

        mapping = redist.compute_greedy_mapping(10, cell_weights=weights)
        assert mapping.shape == (10,)
        # Heavy cell assigned to one proc
        heavy_proc = mapping[0]
        # Other cells should prefer the other proc
        other_count = (mapping[1:] == heavy_proc).sum().item()
        assert other_count <= 5  # Lighter cells distributed to the other proc

    def test_greedy_without_weights(self, tmp_path):
        """Greedy without weights falls back to round-robin."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced(case_dir, target_n_procs=2)
        mapping = redist.compute_greedy_mapping(10, cell_weights=None)
        rr = redist.compute_cell_mapping(10)
        assert torch.equal(mapping, rr)

    def test_spatial_mapping_fallback(self, tmp_path):
        """Spatial mapping falls back to round-robin without centres."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced(case_dir, target_n_procs=2)
        mapping = redist.compute_spatial_mapping(10)
        rr = redist.compute_cell_mapping(10)
        assert torch.equal(mapping, rr)

    def test_spatial_mapping_with_centres(self, tmp_path):
        """Spatial mapping partitions based on cell centres."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced(case_dir, target_n_procs=2)

        # Create 10 cells with distinct x coordinates
        centres = torch.zeros(10, 3, dtype=torch.float64)
        centres[:, 0] = torch.arange(10, dtype=torch.float64)
        redist.set_cell_centres(centres)

        mapping = redist.compute_spatial_mapping(10)
        assert mapping.shape == (10,)
        # Cells should be split by x coordinate
        counts = torch.bincount(mapping, minlength=2)
        assert counts.sum() == 10

    def test_unknown_strategy_raises(self, tmp_path):
        """ValueError for unknown strategy."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced(case_dir, target_n_procs=2)
        with pytest.raises(ValueError, match="Unknown strategy"):
            redist.compute_balanced_mapping(10, strategy="invalid")

    def test_set_cell_centres_validates(self, tmp_path):
        """set_cell_centres validates shape."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced(case_dir, target_n_procs=2)
        with pytest.raises(ValueError, match="shape"):
            redist.set_cell_centres(torch.zeros(10))


# ---------------------------------------------------------------------------
# Diagnostics tests
# ---------------------------------------------------------------------------


class TestPartitionDiagnostics:
    """Test partition diagnostics."""

    def test_compute_diagnostics_balanced(self, tmp_path):
        """Diagnostics for a perfectly balanced partition."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced(case_dir, target_n_procs=4)
        mapping = torch.arange(100, dtype=INDEX_DTYPE) % 4
        diag = redist.compute_diagnostics(mapping)

        assert isinstance(diag, PartitionDiagnostics)
        assert diag.imbalance_ratio == pytest.approx(1.0)
        assert diag.n_empty == 0
        assert diag.min_cells_per_proc == 25
        assert diag.max_cells_per_proc == 25

    def test_compute_diagnostics_imbalanced(self, tmp_path):
        """Diagnostics for an imbalanced partition."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced(case_dir, target_n_procs=2)
        mapping = torch.zeros(100, dtype=INDEX_DTYPE)  # All on proc 0
        diag = redist.compute_diagnostics(mapping)

        assert diag.imbalance_ratio > 1.0
        assert diag.n_empty == 1
        assert diag.min_cells_per_proc == 0
        assert diag.max_cells_per_proc == 100


# ---------------------------------------------------------------------------
# Enhanced redistribution tests
# ---------------------------------------------------------------------------


class TestEnhancedRedistribute:
    """Test enhanced redistribution (requires parallel case)."""

    def _create_parallel_case(self, tmpdir):
        """Create a parallel case for redistribution tests."""
        from pyfoam.parallel.decomposition import Decomposition
        from pyfoam.parallel.parallel_io import ParallelWriter

        mesh = make_8cell_fv_mesh()
        decomp = Decomposition(mesh, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        case_dir = str(tmpdir / "case")
        writer = ParallelWriter(case_dir, n_processors=2)
        writer.write_mesh(subdomains)
        global_field = torch.arange(8, dtype=torch.float64) * 10.0
        writer.write_field("p", global_field, subdomains, time=0)

        return case_dir

    def test_redistribute_enhanced_round_robin(self, tmp_path):
        """Enhanced redistribution with round-robin."""
        case_dir = self._create_parallel_case(tmp_path)
        out_dir = tmp_path / "redistributed"

        redist = RedistributeParEnhanced(case_dir, target_n_procs=4)
        result = redist.redistribute_enhanced(
            output_dir=out_dir,
            strategy=BalancingStrategy.ROUND_ROBIN,
        )

        assert isinstance(result, EnhancedRedistributeResult)
        assert result.strategy == "round_robin"
        assert result.diagnostics.imbalance_ratio >= 1.0

    def test_redistribute_enhanced_result_fields(self, tmp_path):
        """Enhanced result has base and diagnostics."""
        case_dir = self._create_parallel_case(tmp_path)
        out_dir = tmp_path / "redistributed"

        redist = RedistributeParEnhanced(case_dir, target_n_procs=2)
        result = redist.redistribute_enhanced(output_dir=out_dir)

        assert isinstance(result.base, RedistributeResult)
        assert result.base.n_cells == 8
