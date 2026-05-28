"""Tests for foam_to_openfoam version conversion tool."""

from pathlib import Path

import pytest

from pyfoam.tools.foam_to_openfoam import foam_to_openfoam


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _write_foam_file(path: Path, class_name: str, object_name: str, n=5):
    """Write a minimal OpenFOAM ASCII field file."""
    hdr = (
        "FoamFile\n{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        f"    class       {class_name};\n"
        f"    object      {object_name};\n"
        "}\n"
    )
    if "Scalar" in class_name:
        vals = "\n".join(f"{float(i)}" for i in range(n))
        body = (
            f"\ndimensions      [0 2 -2 0 0 0 0];\n\n"
            f"internalField   nonuniform List<scalar> {n}\n(\n{vals}\n);\n\n"
            f"boundaryField\n{{\n"
            f"    inlet\n    {{\n"
            f"        type        fixedValue;\n"
            f"        value       uniform 0;\n"
            f"    }}\n}}\n"
        )
    else:
        vecs = "\n".join(f"({i} {i+1} {i+2})" for i in range(n))
        body = (
            f"\ndimensions      [0 1 -1 0 0 0 0];\n\n"
            f"internalField   nonuniform List<vector> {n}\n(\n{vecs}\n);\n\n"
            f"boundaryField\n{{\n"
            f"    inlet\n    {{\n"
            f"        type        fixedValue;\n"
            f"        value       uniform (0 0 0);\n"
            f"    }}\n}}\n"
        )
    path.write_text(hdr + body, encoding="latin-1")


def _write_control_dict(path: Path):
    """Write a minimal controlDict file."""
    content = (
        "FoamFile\n{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        "    class       dictionary;\n"
        "    object      controlDict;\n"
        "}\n\n"
        "application     simpleFoam;\n\n"
        "startFrom       startTime;\n\n"
        "startTime       0;\n\n"
        "stopAt          endTime;\n\n"
        "endTime         100;\n\n"
        "deltaT          0.001;\n\n"
        "writeControl    timeStep;\n\n"
        "writeInterval   10;\n\n"
        "functions\n{\n"
        "    // none\n"
        "}\n"
    )
    path.write_text(content, encoding="latin-1")


def _write_transport_properties(path: Path):
    """Write a minimal transportProperties file."""
    content = (
        "FoamFile\n{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        "    class       dictionary;\n"
        "    object      transportProperties;\n"
        "}\n\n"
        "nu              nu [ 0 2 -1 0 0 0 0 ] 1e-05;\n"
    )
    path.write_text(content, encoding="latin-1")


def _create_source_case(tmp_path: Path) -> Path:
    """Create a minimal OpenFOAM source case."""
    src = tmp_path / "source_case"
    src.mkdir()

    # Time directories
    t0 = src / "0"
    t0.mkdir()
    _write_foam_file(t0 / "p", "volScalarField", "p")
    _write_foam_file(t0 / "U", "volVectorField", "U")

    t100 = src / "100"
    t100.mkdir()
    _write_foam_file(t100 / "p", "volScalarField", "p", 3)

    # constant directory
    const = src / "constant"
    const.mkdir()
    _write_transport_properties(const / "transportProperties")

    # polyMesh (empty dir)
    mesh = const / "polyMesh"
    mesh.mkdir()

    # system directory
    sys = src / "system"
    sys.mkdir()
    _write_control_dict(sys / "controlDict")

    return src


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestFoamToOpenfoam:
    """Test the foam_to_openfoam conversion tool."""

    def test_basic_conversion(self, tmp_path):
        """Convert a source case to a target case."""
        src = _create_source_case(tmp_path)
        tgt = tmp_path / "target_case"

        result = foam_to_openfoam(src, tgt, source_version="v2312", target_version="v2512")

        assert result["converted"] >= 0
        assert result["source_version"] == "v2312"
        assert result["target_version"] == "v2512"
        assert tgt.is_dir()

    def test_conversion_marker_inserted(self, tmp_path):
        """Converted files contain the conversion marker comment."""
        src = _create_source_case(tmp_path)
        tgt = tmp_path / "target_case"

        foam_to_openfoam(src, tgt, source_version="v2312", target_version="v2512")

        p_text = (tgt / "0" / "p").read_text(encoding="latin-1")
        assert "Converted from v2312 to v2512" in p_text

    def test_mesh_copied(self, tmp_path):
        """polyMesh directory is copied when copy_mesh=True."""
        src = _create_source_case(tmp_path)
        tgt = tmp_path / "target_case"

        foam_to_openfoam(src, tgt, copy_mesh=True)
        assert (tgt / "constant" / "polyMesh").is_dir()

    def test_mesh_not_copied(self, tmp_path):
        """polyMesh directory is NOT copied when copy_mesh=False."""
        src = _create_source_case(tmp_path)
        tgt = tmp_path / "target_case"

        foam_to_openfoam(src, tgt, copy_mesh=False)
        assert not (tgt / "constant" / "polyMesh").is_dir()

    def test_nonexistent_source_raises(self, tmp_path):
        """Non-existent source directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Source case not found"):
            foam_to_openfoam(tmp_path / "nonexistent", tmp_path / "target")

    def test_same_source_target_raises(self, tmp_path):
        """Same source and target raises ValueError."""
        src = _create_source_case(tmp_path)
        with pytest.raises(ValueError, match="must be different"):
            foam_to_openfoam(src, src)

    def test_invalid_version_raises(self, tmp_path):
        """Invalid version string raises ValueError."""
        src = _create_source_case(tmp_path)
        tgt = tmp_path / "target"
        with pytest.raises(ValueError, match="Unknown OpenFOAM version"):
            foam_to_openfoam(src, tgt, source_version="v9999")

    def test_skip_existing_without_overwrite(self, tmp_path):
        """Existing target files are skipped when overwrite=False."""
        src = _create_source_case(tmp_path)
        tgt = tmp_path / "target_case"

        # First conversion
        foam_to_openfoam(src, tgt, source_version="v2312", target_version="v2512")

        # Second conversion (files already exist)
        result = foam_to_openfoam(src, tgt, source_version="v2312", target_version="v2512")
        assert result["skipped"] > 0

    def test_system_files_converted(self, tmp_path):
        """System directory files are converted."""
        src = _create_source_case(tmp_path)
        tgt = tmp_path / "target_case"

        result = foam_to_openfoam(src, tgt, source_version="v2312", target_version="v2512")
        assert (tgt / "system" / "controlDict").exists()

    def test_time_dirs_converted(self, tmp_path):
        """All time directories are processed."""
        src = _create_source_case(tmp_path)
        tgt = tmp_path / "target_case"

        result = foam_to_openfoam(src, tgt, source_version="v2312", target_version="v2512")
        # At least 0/ and 100/ should have files
        assert (tgt / "0" / "p").exists()
        assert (tgt / "0" / "U").exists()
        assert (tgt / "100" / "p").exists()

    def test_no_foamfile_files_skipped(self, tmp_path):
        """Files without FoamFile header are copied as-is."""
        src = _create_source_case(tmp_path)
        # Write a non-OpenFOAM file
        (src / "0" / "readme.txt").write_text("not an OpenFOAM file", encoding="utf-8")
        tgt = tmp_path / "target_case"

        foam_to_openfoam(src, tgt, source_version="v2312", target_version="v2512")
        assert (tgt / "0" / "readme.txt").exists()
