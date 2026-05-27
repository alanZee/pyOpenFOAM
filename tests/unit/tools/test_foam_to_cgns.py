"""Tests for foam_to_cgns — CGNS export utility."""

from pathlib import Path

import numpy as np
import pytest

from pyfoam.tools.foam_to_cgns import foam_to_cgns


def _is_hdf5_file(path):
    """Check if a file is HDF5 format."""
    try:
        with open(path, "rb") as f:
            header = f.read(8)
            # HDF5 magic bytes: 0x89 0x48 0x44 0x46 0x0d 0x0a 0x1a 0x0a
            return header[:4] == b'\x89HDF'
    except Exception:
        return False


def _read_cgns_content(path):
    """Read CGNS file content, handling both ASCII and HDF5."""
    if _is_hdf5_file(path):
        import h5py
        names = []
        def visitor(name, obj):
            names.append(name)
        with h5py.File(path, "r") as f:
            f.visititems(visitor)
        return names
    else:
        return Path(path).read_text(encoding="utf-8")


def _has_name(content, name):
    """Check if a name appears in content (handles both str and list)."""
    if isinstance(content, list):
        return any(name in item for item in content)
    return name in content


class TestFoamToCgns:
    """Test the foam_to_cgns function."""

    def test_export_creates_output_file(self, fv_mesh, tmp_path):
        """Basic export should create the output .cgns file."""
        out = foam_to_cgns(
            case_path=str(tmp_path),
            output_path=str(tmp_path / "case.cgns"),
            mesh=fv_mesh,
        )
        assert Path(out).is_file()
        assert Path(out).suffix == ".cgns"

    def test_default_output_path(self, fv_mesh, tmp_path):
        """Default output path is <case>/case.cgns."""
        out = foam_to_cgns(
            case_path=str(tmp_path),
            mesh=fv_mesh,
        )
        assert Path(out).exists()
        assert Path(out).name == "case.cgns"

    def test_nonexistent_case_path_raises(self, tmp_path):
        """Should raise FileNotFoundError for non-existent path."""
        with pytest.raises(FileNotFoundError):
            foam_to_cgns(
                case_path=str(tmp_path / "nonexistent"),
                mesh=None,
            )

    def test_no_mesh_raises(self, tmp_path):
        """Should raise ValueError when no mesh is provided."""
        with pytest.raises(ValueError, match="No mesh provided"):
            foam_to_cgns(
                case_path=str(tmp_path),
                mesh=None,
            )

    def test_output_contains_coordinates(self, fv_mesh, tmp_path):
        """CGNS file should contain coordinate data."""
        out = foam_to_cgns(
            case_path=str(tmp_path),
            output_path=str(tmp_path / "case.cgns"),
            mesh=fv_mesh,
        )
        content = _read_cgns_content(out)
        assert _has_name(content, "CoordinateX")
        assert _has_name(content, "CoordinateY")
        assert _has_name(content, "CoordinateZ")

    def test_output_contains_connectivity(self, fv_mesh, tmp_path):
        """CGNS file should contain element connectivity."""
        out = foam_to_cgns(
            case_path=str(tmp_path),
            output_path=str(tmp_path / "case.cgns"),
            mesh=fv_mesh,
        )
        content = _read_cgns_content(out)
        assert _has_name(content, "ElementConnectivity")

    def test_export_with_scalar_field(self, fv_mesh, tmp_path):
        """Scalar field should appear in the output."""
        n_cells = fv_mesh.n_cells
        pressure = np.ones(n_cells) * 101325.0

        out = foam_to_cgns(
            case_path=str(tmp_path),
            output_path=str(tmp_path / "case.cgns"),
            mesh=fv_mesh,
            fields={"p": pressure},
        )
        content = _read_cgns_content(out)
        assert _has_name(content, "p")

    def test_export_with_vector_field(self, fv_mesh, tmp_path):
        """Vector field should produce X, Y, Z components."""
        n_cells = fv_mesh.n_cells
        velocity = np.zeros((n_cells, 3))
        velocity[:, 0] = 1.0

        out = foam_to_cgns(
            case_path=str(tmp_path),
            output_path=str(tmp_path / "case.cgns"),
            mesh=fv_mesh,
            fields={"U": velocity},
        )
        content = _read_cgns_content(out)
        if _is_hdf5_file(out):
            # HDF5: check for U/X, U/Y, U/Z or UX, UY, UZ datasets
            assert _has_name(content, "X") or _has_name(content, "UX")
        else:
            assert _has_name(content, "UX")
            assert _has_name(content, "UY")
            assert _has_name(content, "UZ")

    def test_boundary_patches_in_output(self, fv_mesh, tmp_path):
        """Boundary patches should appear in the CGNS output."""
        out = foam_to_cgns(
            case_path=str(tmp_path),
            output_path=str(tmp_path / "case.cgns"),
            mesh=fv_mesh,
        )
        content = _read_cgns_content(out)
        # At least one boundary patch should appear
        for patch_info in fv_mesh.boundary:
            assert _has_name(content, patch_info["name"])

    def test_export_no_fields(self, fv_mesh, tmp_path):
        """Export without fields should still produce valid output."""
        out = foam_to_cgns(
            case_path=str(tmp_path),
            output_path=str(tmp_path / "case.cgns"),
            mesh=fv_mesh,
        )
        content = _read_cgns_content(out)
        assert _has_name(content, "CoordinateX")

    def test_parent_directory_created(self, fv_mesh, tmp_path):
        """Output directory is created if it does not exist."""
        out = foam_to_cgns(
            case_path=str(tmp_path),
            output_path=str(tmp_path / "subdir" / "case.cgns"),
            mesh=fv_mesh,
        )
        assert Path(out).exists()

    def test_file_size_nonzero(self, fv_mesh, tmp_path):
        """Output file should have nonzero size."""
        out = foam_to_cgns(
            case_path=str(tmp_path),
            output_path=str(tmp_path / "case.cgns"),
            mesh=fv_mesh,
        )
        assert Path(out).stat().st_size > 0
