"""Tests for ReconstructParEnhanced — zone-aware parallel reconstruction."""

import os
import shutil
import tempfile

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.parallel.reconstruct_par_enhanced import (
    ReconstructParEnhanced,
    ZoneInfo,
    EnhancedReconstructResult,
)
from pyfoam.parallel.reconstruct_par import ReconstructResult

from tests.unit.parallel.conftest import make_8cell_fv_mesh


# ---------------------------------------------------------------------------
# Helper: create a parallel case with zone data
# ---------------------------------------------------------------------------


def _create_case_with_zones(tmpdir: str, n_procs: int = 2):
    """Create a parallel case with processor directories and zone files.

    Returns (case_dir, zone_names).
    """
    from pyfoam.parallel.decomposition import Decomposition
    from pyfoam.parallel.parallel_io import ParallelWriter

    mesh = make_8cell_fv_mesh()
    decomp = Decomposition(mesh, n_processors=n_procs, method="simple")
    subdomains = decomp.decompose()

    writer = ParallelWriter(tmpdir, n_processors=n_procs)
    writer.write_mesh(subdomains)

    global_field = torch.arange(8, dtype=torch.float64) * 10.0
    writer.write_field("p", global_field, subdomains, time=0)

    # Write cell zone files manually
    for proc_idx in range(n_procs):
        zones_dir = os.path.join(tmpdir, f"processor{proc_idx}", "constant", "polyMesh", "zones")
        os.makedirs(zones_dir, exist_ok=True)

        # cellZones file with a "rotor" zone
        cell_zones_path = os.path.join(zones_dir, "cellZones")
        with open(cell_zones_path, "w") as f:
            f.write("FoamFile\n{ version 2.0; format ascii; class labelListList; ")
            f.write('location "constant/polyMesh/zones"; object cellZones; }\n\n')
            f.write("1\n(\n")  # 1 zone
            # Write zone: name + { n (indices) }
            n_cells = subdomains[proc_idx].n_cells if hasattr(subdomains[proc_idx], 'n_cells') else 4
            local_cells = list(range(min(2, n_cells)))  # first 2 cells in each proc
            f.write("rotor\n{\n")
            f.write(f"    {len(local_cells)}\n")
            f.write("    (\n")
            for c in local_cells:
                f.write(f"        {c}\n")
            f.write("    )\n")
            f.write("}\n")

    return tmpdir, {"cellZones": ["rotor"], "faceZones": []}


# ---------------------------------------------------------------------------
# Zone discovery tests
# ---------------------------------------------------------------------------


class TestZoneDiscovery:
    """Test zone discovery."""

    def test_discover_zones(self, tmp_path):
        """Discover cell zones from processor directories."""
        case_dir, zone_names = _create_case_with_zones(str(tmp_path / "case"))
        recon = ReconstructParEnhanced(case_dir)
        recon.discover()
        discovered = recon.discover_zones()

        assert "cellZones" in discovered
        assert "faceZones" in discovered
        assert "rotor" in discovered["cellZones"]

    def test_discover_no_zones(self, tmp_path):
        """Empty result when no zone files exist."""
        from pyfoam.parallel.decomposition import Decomposition
        from pyfoam.parallel.parallel_io import ParallelWriter

        mesh = make_8cell_fv_mesh()
        decomp = Decomposition(mesh, n_processors=2, method="simple")
        subdomains = decomp.decompose()
        case_dir = str(tmp_path / "case")
        writer = ParallelWriter(case_dir, n_processors=2)
        writer.write_mesh(subdomains)

        recon = ReconstructParEnhanced(case_dir)
        recon.discover()
        discovered = recon.discover_zones()

        assert discovered["cellZones"] == []
        assert discovered["faceZones"] == []


# ---------------------------------------------------------------------------
# Zone reconstruction tests
# ---------------------------------------------------------------------------


class TestZoneReconstruction:
    """Test zone reconstruction."""

    def test_reconstruct_zones(self, tmp_path):
        """Reconstruct zones from processor directories."""
        case_dir, _ = _create_case_with_zones(str(tmp_path / "case"))
        recon = ReconstructParEnhanced(case_dir)
        recon.discover()
        zones = recon.reconstruct_zones()

        assert len(zones) >= 1
        assert any(z.name == "rotor" for z in zones)
        assert any(z.zone_type == "cellZone" for z in zones)

    def test_zone_info_dataclass(self):
        """ZoneInfo stores correct data."""
        zi = ZoneInfo(name="test", zone_type="cellZone", n_entries=10)
        assert zi.name == "test"
        assert zi.zone_type == "cellZone"
        assert zi.n_entries == 10

    def test_cell_zones_property(self, tmp_path):
        """cell_zones property after reconstruction."""
        case_dir, _ = _create_case_with_zones(str(tmp_path / "case"))
        recon = ReconstructParEnhanced(case_dir)
        recon.discover()
        recon.reconstruct_zones()

        assert "rotor" in recon.cell_zones
        assert isinstance(recon.cell_zones["rotor"], list)

    def test_face_zones_property(self, tmp_path):
        """face_zones property after reconstruction."""
        case_dir, _ = _create_case_with_zones(str(tmp_path / "case"))
        recon = ReconstructParEnhanced(case_dir)
        recon.discover()
        recon.reconstruct_zones()

        assert isinstance(recon.face_zones, dict)


# ---------------------------------------------------------------------------
# Zone-aware field reconstruction tests
# ---------------------------------------------------------------------------


class TestZoneFieldReconstruction:
    """Test zone-aware field reconstruction."""

    def test_reconstruct_zone_fields(self, tmp_path):
        """Reconstruct fields restricted to a zone."""
        case_dir, _ = _create_case_with_zones(str(tmp_path / "case"))
        recon = ReconstructParEnhanced(case_dir)
        recon.discover()
        recon.reconstruct_zones()

        fields = recon.reconstruct_zone_fields(time="0", zone_name="rotor", field_names=["p"])
        assert "p" in fields

    def test_reconstruct_zone_fields_no_zone(self, tmp_path):
        """Reconstruct all fields when zone_name is None."""
        case_dir, _ = _create_case_with_zones(str(tmp_path / "case"))
        recon = ReconstructParEnhanced(case_dir)
        recon.discover()

        fields = recon.reconstruct_zone_fields(time="0", zone_name=None, field_names=["p"])
        assert "p" in fields
        assert fields["p"].shape[0] == 8

    def test_reconstruct_zone_fields_missing_zone(self, tmp_path):
        """Returns full fields for a missing zone (with warning)."""
        case_dir, _ = _create_case_with_zones(str(tmp_path / "case"))
        recon = ReconstructParEnhanced(case_dir)
        recon.discover()

        fields = recon.reconstruct_zone_fields(time="0", zone_name="nonexistent", field_names=["p"])
        assert "p" in fields


# ---------------------------------------------------------------------------
# Enhanced case reconstruction tests
# ---------------------------------------------------------------------------


class TestEnhancedCaseReconstruction:
    """Test full enhanced case reconstruction."""

    def test_reconstruct_case_enhanced(self, tmp_path):
        """Reconstruct full case including zones."""
        case_dir, _ = _create_case_with_zones(str(tmp_path / "case"))
        out_dir = tmp_path / "reconstructed"

        recon = ReconstructParEnhanced(case_dir)
        result = recon.reconstruct_case_enhanced(output_dir=out_dir)

        assert isinstance(result, EnhancedReconstructResult)
        assert isinstance(result.base, ReconstructResult)
        assert result.base.n_global_cells == 8
        assert len(result.zones) >= 1
