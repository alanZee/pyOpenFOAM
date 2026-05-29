"""Tests for ReconstructPar — reconstruct parallel case from processor directories."""

import os
import shutil
import tempfile

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.parallel.reconstruct_par import ReconstructPar, ReconstructResult
from pyfoam.parallel.decomposition import Decomposition
from pyfoam.parallel.parallel_io import ParallelWriter

from tests.unit.parallel.conftest import make_8cell_fv_mesh


# ---------------------------------------------------------------------------
# Helper: create a test parallel case
# ---------------------------------------------------------------------------


def _create_parallel_case(tmpdir: str, n_procs: int = 2):
    """Create a parallel case with processor directories.

    Returns (case_dir, global_field, subdomains).
    """
    mesh = make_8cell_fv_mesh()
    decomp = Decomposition(mesh, n_processors=n_procs, method="simple")
    subdomains = decomp.decompose()

    writer = ParallelWriter(tmpdir, n_processors=n_procs)
    writer.write_mesh(subdomains)

    global_field = torch.arange(8, dtype=torch.float64) * 10.0
    writer.write_field("p", global_field, subdomains, time=0)

    return tmpdir, global_field, subdomains


# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------


class TestReconstructDiscovery:
    """Test processor directory discovery."""

    def test_discover_processors(self, tmp_path):
        """Discover processor directories."""
        case_dir, _, _ = _create_parallel_case(str(tmp_path / "case"))
        recon = ReconstructPar(case_dir)
        n = recon.discover()
        assert n == 2

    def test_discover_time_steps(self, tmp_path):
        """Discover time steps."""
        case_dir, _, _ = _create_parallel_case(str(tmp_path / "case"))
        recon = ReconstructPar(case_dir)
        recon.discover()
        assert "0" in recon.time_steps

    def test_no_processors_raises(self, tmp_path):
        """FileNotFoundError when no processor directories exist."""
        empty = tmp_path / "empty"
        empty.mkdir()
        recon = ReconstructPar(str(empty))
        with pytest.raises(FileNotFoundError, match="No processor"):
            recon.discover()

    def test_n_processors_property(self, tmp_path):
        """n_processors property after discovery."""
        case_dir, _, _ = _create_parallel_case(str(tmp_path / "case"))
        recon = ReconstructPar(case_dir)
        recon.discover()
        assert recon.n_processors == 2


# ---------------------------------------------------------------------------
# Mesh reconstruction tests
# ---------------------------------------------------------------------------


class TestReconstructMesh:
    """Test mesh reconstruction."""

    def test_reconstruct_mesh_returns_dict(self, tmp_path):
        """reconstruct_mesh returns expected dict keys."""
        case_dir, _, _ = _create_parallel_case(str(tmp_path / "case"))
        recon = ReconstructPar(case_dir)
        recon.discover()
        mesh = recon.reconstruct_mesh()

        assert "points" in mesh
        assert "faces" in mesh
        assert "owner" in mesh
        assert "neighbour" in mesh
        assert "boundary" in mesh
        assert "n_cells" in mesh

    def test_reconstruct_total_cells(self, tmp_path):
        """Reconstructed mesh has correct total cell count."""
        case_dir, _, _ = _create_parallel_case(str(tmp_path / "case"))
        recon = ReconstructPar(case_dir)
        recon.discover()
        mesh = recon.reconstruct_mesh()

        assert mesh["n_cells"] == 8

    def test_reconstruct_points_shape(self, tmp_path):
        """Reconstructed points have correct shape."""
        case_dir, _, _ = _create_parallel_case(str(tmp_path / "case"))
        recon = ReconstructPar(case_dir)
        recon.discover()
        mesh = recon.reconstruct_mesh()

        assert mesh["points"].dim() == 2
        assert mesh["points"].shape[1] == 3

    def test_reconstruct_owner_range(self, tmp_path):
        """Owner indices are in valid range."""
        case_dir, _, _ = _create_parallel_case(str(tmp_path / "case"))
        recon = ReconstructPar(case_dir)
        recon.discover()
        mesh = recon.reconstruct_mesh()

        assert mesh["owner"].min() >= 0
        assert mesh["owner"].max() < mesh["n_cells"]

    def test_reconstruct_neighbour_range(self, tmp_path):
        """Neighbour indices are in valid range."""
        case_dir, _, _ = _create_parallel_case(str(tmp_path / "case"))
        recon = ReconstructPar(case_dir)
        recon.discover()
        mesh = recon.reconstruct_mesh()

        if mesh["neighbour"].numel() > 0:
            assert mesh["neighbour"].min() >= 0
            assert mesh["neighbour"].max() < mesh["n_cells"]

    def test_reconstruct_boundary_exists(self, tmp_path):
        """Boundary patches are reconstructed."""
        case_dir, _, _ = _create_parallel_case(str(tmp_path / "case"))
        recon = ReconstructPar(case_dir)
        recon.discover()
        mesh = recon.reconstruct_mesh()

        assert len(mesh["boundary"]) > 0
        for patch in mesh["boundary"]:
            assert "name" in patch
            assert "nFaces" in patch
            assert "startFace" in patch


# ---------------------------------------------------------------------------
# Field reconstruction tests
# ---------------------------------------------------------------------------


class TestReconstructFields:
    """Test field reconstruction."""

    def test_reconstruct_field(self, tmp_path):
        """Reconstruct a single field."""
        case_dir, global_field, _ = _create_parallel_case(str(tmp_path / "case"))
        recon = ReconstructPar(case_dir)
        recon.discover()
        fields = recon.reconstruct_fields(time="0", field_names=["p"])

        assert "p" in fields
        assert fields["p"].shape[0] == 8

    def test_reconstruct_field_values(self, tmp_path):
        """Reconstructed field values match the global field."""
        case_dir, global_field, _ = _create_parallel_case(str(tmp_path / "case"))
        recon = ReconstructPar(case_dir)
        recon.discover()
        fields = recon.reconstruct_fields(time="0", field_names=["p"])

        # Values may be in a different order (processor decomposition order),
        # but the set of values should match
        expected_vals = set(global_field.tolist())
        actual_vals = set(fields["p"].tolist())
        assert actual_vals == expected_vals

    def test_reconstruct_auto_detect_fields(self, tmp_path):
        """Auto-detect field names from processor0."""
        case_dir, _, _ = _create_parallel_case(str(tmp_path / "case"))
        recon = ReconstructPar(case_dir)
        recon.discover()
        fields = recon.reconstruct_fields(time="0")

        assert "p" in fields

    def test_reconstruct_nonexistent_time_raises(self, tmp_path):
        """FileNotFoundError for nonexistent time step."""
        case_dir, _, _ = _create_parallel_case(str(tmp_path / "case"))
        recon = ReconstructPar(case_dir)
        recon.discover()
        with pytest.raises(FileNotFoundError):
            recon.reconstruct_fields(time="999")


# ---------------------------------------------------------------------------
# Full case reconstruction tests
# ---------------------------------------------------------------------------


class TestReconstructCase:
    """Test full case reconstruction."""

    def test_reconstruct_case(self, tmp_path):
        """Reconstruct full case with output."""
        case_dir, _, _ = _create_parallel_case(str(tmp_path / "case"))
        out_dir = tmp_path / "reconstructed"

        recon = ReconstructPar(case_dir)
        result = recon.reconstruct_case(output_dir=out_dir)

        assert isinstance(result, ReconstructResult)
        assert result.n_processors == 2
        assert result.n_global_cells == 8
        assert result.n_time_steps == 1

    def test_reconstruct_case_creates_mesh(self, tmp_path):
        """Reconstructed case has mesh files."""
        case_dir, _, _ = _create_parallel_case(str(tmp_path / "case"))
        out_dir = tmp_path / "reconstructed"

        recon = ReconstructPar(case_dir)
        recon.reconstruct_case(output_dir=out_dir)

        mesh_dir = out_dir / "constant" / "polyMesh"
        assert (mesh_dir / "points").exists()
        assert (mesh_dir / "faces").exists()
        assert (mesh_dir / "owner").exists()
        assert (mesh_dir / "neighbour").exists()
        assert (mesh_dir / "boundary").exists()

    def test_reconstruct_case_creates_fields(self, tmp_path):
        """Reconstructed case has field files."""
        case_dir, _, _ = _create_parallel_case(str(tmp_path / "case"))
        out_dir = tmp_path / "reconstructed"

        recon = ReconstructPar(case_dir)
        recon.reconstruct_case(output_dir=out_dir)

        assert (out_dir / "0" / "p").exists()
