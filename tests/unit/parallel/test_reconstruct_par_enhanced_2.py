"""Tests for ReconstructParEnhanced2 — v2 enhanced parallel reconstruction."""

import pytest
import torch

from pyfoam.parallel.reconstruct_par_enhanced_2 import (
    ReconstructParEnhanced2,
    V2ReconstructResult,
    FieldMergeStats,
    MergeStrategy,
)
from pyfoam.parallel.reconstruct_par_enhanced import (
    ReconstructParEnhanced,
    EnhancedReconstructResult,
)

from tests.unit.parallel.conftest import make_8cell_fv_mesh


# ---------------------------------------------------------------------------
# MergeStrategy tests
# ---------------------------------------------------------------------------


class TestMergeStrategy:
    """Test MergeStrategy constants."""

    def test_all_strategies(self):
        strategies = MergeStrategy.all_strategies()
        assert "first_occurrence" in strategies
        assert "volume_weighted" in strategies
        assert "last_occurrence" in strategies

    def test_strategy_constants(self):
        assert MergeStrategy.FIRST_OCCURRENCE == "first_occurrence"
        assert MergeStrategy.VOLUME_WEIGHTED == "volume_weighted"
        assert MergeStrategy.LAST_OCCURRENCE == "last_occurrence"


# ---------------------------------------------------------------------------
# FieldMergeStats tests
# ---------------------------------------------------------------------------


class TestFieldMergeStats:
    """Test FieldMergeStats dataclass."""

    def test_creation(self):
        stat = FieldMergeStats(field_name="p", n_values=100, n_overlaps=5, max_overlap=2)
        assert stat.field_name == "p"
        assert stat.n_values == 100
        assert stat.n_overlaps == 5
        assert stat.max_overlap == 2

    def test_defaults(self):
        stat = FieldMergeStats(field_name="U")
        assert stat.n_values == 0
        assert stat.n_overlaps == 0
        assert stat.max_overlap == 0


# ---------------------------------------------------------------------------
# Inheritance tests
# ---------------------------------------------------------------------------


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_enhanced(self):
        """ReconstructParEnhanced2 extends ReconstructParEnhanced."""
        assert issubclass(ReconstructParEnhanced2, ReconstructParEnhanced)


# ---------------------------------------------------------------------------
# Volume-weighted merge tests
# ---------------------------------------------------------------------------


class TestVolumeWeightedMerge:
    """Test volume-weighted field merging."""

    def test_merge_single_proc(self, tmp_path):
        """Single processor: values used directly."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        recon = ReconstructParEnhanced2(case_dir)

        proc_fields = {0: torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float64)}
        proc_map = {0: torch.tensor([0, 1, 2, 3])}

        result = recon.merge_field_weighted(proc_fields, proc_map, n_global_cells=4)
        assert torch.allclose(result, torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float64))

    def test_merge_two_procs_no_overlap(self, tmp_path):
        """Two processors, no overlapping cells."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        recon = ReconstructParEnhanced2(case_dir)

        proc_fields = {
            0: torch.tensor([10.0, 20.0], dtype=torch.float64),
            1: torch.tensor([30.0, 40.0], dtype=torch.float64),
        }
        proc_map = {
            0: torch.tensor([0, 1]),
            1: torch.tensor([2, 3]),
        }

        result = recon.merge_field_weighted(proc_fields, proc_map, n_global_cells=4)
        assert torch.allclose(result, torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float64))

    def test_merge_with_overlap_uniform_weight(self, tmp_path):
        """Overlapping cells averaged with uniform weights."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        recon = ReconstructParEnhanced2(case_dir)

        proc_fields = {
            0: torch.tensor([10.0, 20.0]),
            1: torch.tensor([20.0, 40.0]),
        }
        proc_map = {
            0: torch.tensor([0, 1]),
            1: torch.tensor([1, 2]),
        }
        # Cell 1 appears on both procs: (20 + 20) / 2 = 20

        result = recon.merge_field_weighted(proc_fields, proc_map, n_global_cells=3)
        assert result[0].item() == pytest.approx(10.0)
        assert result[1].item() == pytest.approx(20.0)
        assert result[2].item() == pytest.approx(40.0)

    def test_merge_with_volumes(self, tmp_path):
        """Volume-weighted merging gives different averages."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        recon = ReconstructParEnhanced2(case_dir)

        # Cell 1: proc 0 has volume 3, proc 1 has volume 1
        recon.set_cell_volumes({
            0: torch.tensor([1.0, 3.0]),
            1: torch.tensor([1.0, 1.0]),
        })

        proc_fields = {
            0: torch.tensor([10.0, 100.0]),
            1: torch.tensor([200.0, 40.0]),
        }
        proc_map = {
            0: torch.tensor([0, 1]),
            1: torch.tensor([1, 2]),
        }
        # Cell 1: (100*3 + 200*1) / (3+1) = 500/4 = 125

        result = recon.merge_field_weighted(proc_fields, proc_map, n_global_cells=3)
        assert result[0].item() == pytest.approx(10.0)
        assert result[1].item() == pytest.approx(125.0)
        assert result[2].item() == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# V2 reconstruction tests
# ---------------------------------------------------------------------------


class TestV2Reconstruction:
    """Test v2 reconstruction features."""

    def test_reconstruct_case_v2_invalid_strategy(self, tmp_path):
        """ValueError for unknown merge strategy."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        recon = ReconstructParEnhanced2(case_dir)
        with pytest.raises(ValueError, match="Unknown merge strategy"):
            recon.reconstruct_case_v2(merge_strategy="invalid")

    def test_boundary_patch_info_empty_skip(self, tmp_path):
        """Boundary info returns empty dict for non-existent case."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        recon = ReconstructParEnhanced2(case_dir)
        # No processor dirs means discover() would fail,
        # but get_boundary_patch_info handles empty case gracefully
        # by setting _processor_dirs to empty before calling discover
        recon._processor_dirs = []
        info = recon.get_boundary_patch_info()
        assert isinstance(info, dict)
        assert len(info) == 0
