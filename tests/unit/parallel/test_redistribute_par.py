"""Tests for RedistributePar — redistribute parallel case across processors."""

import os
import shutil
import tempfile

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.parallel.redistribute_par import RedistributePar, RedistributeResult
from pyfoam.parallel.decomposition import Decomposition
from pyfoam.parallel.parallel_io import ParallelWriter

from tests.unit.parallel.conftest import make_8cell_fv_mesh


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _create_parallel_case(tmpdir: str, n_procs: int = 2):
    """Create a parallel case with processor directories."""
    mesh = make_8cell_fv_mesh()
    decomp = Decomposition(mesh, n_processors=n_procs, method="simple")
    subdomains = decomp.decompose()

    writer = ParallelWriter(tmpdir, n_processors=n_procs)
    writer.write_mesh(subdomains)

    global_field = torch.arange(8, dtype=torch.float64) * 10.0
    writer.write_field("p", global_field, subdomains, time=0)

    return tmpdir, global_field, subdomains


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestRedistributeValidation:
    """Test input validation."""

    def test_zero_target_raises(self, tmp_path):
        with pytest.raises(ValueError, match="target_n_procs must be >= 1"):
            RedistributePar(str(tmp_path), target_n_procs=0)

    def test_negative_target_raises(self, tmp_path):
        with pytest.raises(ValueError, match="target_n_procs must be >= 1"):
            RedistributePar(str(tmp_path), target_n_procs=-1)


# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------


class TestRedistributeDiscovery:
    """Test processor discovery."""

    def test_discover_processors(self, tmp_path):
        case_dir, _, _ = _create_parallel_case(str(tmp_path / "case"))
        redist = RedistributePar(case_dir, target_n_procs=4)
        n_procs, time_steps = redist.discover()
        assert n_procs == 2
        assert "0" in time_steps

    def test_no_processors_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        redist = RedistributePar(str(empty), target_n_procs=2)
        with pytest.raises(FileNotFoundError, match="No processor"):
            redist.discover()

    def test_properties(self, tmp_path):
        case_dir, _, _ = _create_parallel_case(str(tmp_path / "case"))
        redist = RedistributePar(case_dir, target_n_procs=4)
        assert redist.target_n_procs == 4
        assert redist.case_dir == tmp_path / "case"


# ---------------------------------------------------------------------------
# Cell mapping tests
# ---------------------------------------------------------------------------


class TestCellMapping:
    """Test cell-to-processor mapping."""

    def test_round_robin_mapping(self, tmp_path):
        """Round-robin mapping distributes cells evenly."""
        case_dir, _, _ = _create_parallel_case(str(tmp_path / "case"))
        redist = RedistributePar(case_dir, target_n_procs=4)
        mapping = redist.compute_cell_mapping(8)

        assert mapping.shape == (8,)
        assert mapping.dtype == INDEX_DTYPE
        # Each processor gets 2 cells
        for p in range(4):
            assert (mapping == p).sum().item() == 2

    def test_round_robin_2_procs(self, tmp_path):
        """Round-robin with 2 processors."""
        case_dir, _, _ = _create_parallel_case(str(tmp_path / "case"))
        redist = RedistributePar(case_dir, target_n_procs=2)
        mapping = redist.compute_cell_mapping(8)

        assert (mapping == 0).sum().item() == 4
        assert (mapping == 1).sum().item() == 4

    def test_load_balanced_equal_weights(self, tmp_path):
        """Load-balanced mapping with equal weights."""
        case_dir, _, _ = _create_parallel_case(str(tmp_path / "case"))
        redist = RedistributePar(case_dir, target_n_procs=3)
        mapping = redist.compute_load_balanced_mapping(8)

        assert mapping.shape == (8,)
        # All processors should have cells
        for p in range(3):
            assert (mapping == p).sum().item() >= 2

    def test_load_balanced_custom_weights(self, tmp_path):
        """Load-balanced mapping with custom weights."""
        case_dir, _, _ = _create_parallel_case(str(tmp_path / "case"))
        redist = RedistributePar(case_dir, target_n_procs=2)
        weights = torch.tensor([10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        mapping = redist.compute_load_balanced_mapping(8, cell_weights=weights)

        assert mapping.shape == (8,)
        # Heaviest cell should be on processor 0
        assert mapping[0].item() == 0


# ---------------------------------------------------------------------------
# Redistribution tests
# ---------------------------------------------------------------------------


class TestRedistribution:
    """Test full redistribution workflow."""

    def test_redistribute_to_4(self, tmp_path):
        """Redistribute from 2 to 4 processors."""
        case_dir, _, _ = _create_parallel_case(str(tmp_path / "case"))
        out_dir = str(tmp_path / "redistributed")

        redist = RedistributePar(case_dir, target_n_procs=4)
        result = redist.redistribute(output_dir=out_dir, field_names=["p"])

        assert isinstance(result, RedistributeResult)
        assert result.source_n_procs == 2
        assert result.target_n_procs == 4
        assert result.n_cells == 8

    def test_redistribute_creates_dirs(self, tmp_path):
        """Redistribute creates target processor directories."""
        case_dir, _, _ = _create_parallel_case(str(tmp_path / "case"))
        out_dir = tmp_path / "redistributed"

        redist = RedistributePar(case_dir, target_n_procs=4)
        redist.redistribute(output_dir=str(out_dir), field_names=["p"])

        for i in range(4):
            assert (out_dir / f"processor{i}").exists()

    def test_redistribute_imbalance_ratio(self, tmp_path):
        """Imbalance ratio is reasonable for round-robin."""
        case_dir, _, _ = _create_parallel_case(str(tmp_path / "case"))
        out_dir = str(tmp_path / "redistributed")

        redist = RedistributePar(case_dir, target_n_procs=4)
        result = redist.redistribute(output_dir=out_dir, field_names=["p"])

        assert result.imbalance_ratio >= 1.0
        assert result.imbalance_ratio <= 2.0  # Should be close to 1.0

    def test_redistribute_in_place(self, tmp_path):
        """Redistribute in place (no output_dir)."""
        case_dir, _, _ = _create_parallel_case(str(tmp_path / "case"))

        redist = RedistributePar(case_dir, target_n_procs=4)
        result = redist.redistribute(field_names=["p"])

        assert result.n_cells == 8
