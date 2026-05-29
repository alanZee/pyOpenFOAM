"""Tests for ensight_io — EnSight Gold format I/O."""

import numpy as np
import pytest
import torch

from pyfoam.io.ensight_io import (
    EnSightCase,
    EnSightGeometry,
    EnSightPart,
    EnSightVariable,
    read_ensight_case,
    read_ensight_geometry,
    read_ensight_variable,
    write_ensight_case,
    write_ensight_geometry,
    write_ensight_variable,
)


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------


class TestEnSightDataClasses:
    """Test EnSight data classes."""

    def test_part_creation(self):
        """Create EnSightPart."""
        conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
        part = EnSightPart(part_id=1, description="fluid", element_type="hexa8",
                           connectivity=conn)
        assert part.part_id == 1
        assert part.element_type == "hexa8"
        assert part.connectivity.shape == (1, 8)

    def test_geometry_creation(self):
        """Create EnSightGeometry."""
        coords = np.zeros((8, 3))
        part = EnSightPart(part_id=1, description="test", element_type="hexa8",
                           connectivity=np.zeros((1, 8), dtype=np.int32))
        geo = EnSightGeometry(title=["test"], node_coords=coords, parts=[part])
        assert len(geo.parts) == 1

    def test_variable_creation(self):
        """Create EnSightVariable."""
        var = EnSightVariable(description="pressure", variable_type="scalar",
                              part_id=1, values=np.array([1.0, 2.0, 3.0]))
        assert var.variable_type == "scalar"
        assert var.values.shape == (3,)

    def test_case_creation(self):
        """Create EnSightCase."""
        case = EnSightCase(title="test case", geometry_file="case.geo",
                           variables={"p": "case.scl"})
        assert case.geometry_file == "case.geo"
        assert "p" in case.variables


# ---------------------------------------------------------------------------
# Geometry write/read tests
# ---------------------------------------------------------------------------


class TestGeometryIO:
    """Test geometry file write/read roundtrip."""

    def test_write_geometry_creates_file(self, tmp_path):
        """Write geometry file is created."""
        coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                           [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
                          dtype=np.float64)
        conn = np.array([[0, 1, 5, 4]], dtype=np.int32)
        part = EnSightPart(part_id=1, description="block", element_type="quad4",
                           connectivity=conn)

        path = tmp_path / "test.geo"
        write_ensight_geometry(path, coords, [part])
        assert path.exists()

    def test_write_read_roundtrip(self, tmp_path):
        """Write then read geometry file."""
        coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float64)
        conn = np.array([[0, 1, 2]], dtype=np.int32)
        part = EnSightPart(part_id=1, description="tri", element_type="tria3",
                           connectivity=conn)

        path = tmp_path / "test.geo"
        write_ensight_geometry(path, coords, [part])

        geo = read_ensight_geometry(path)
        assert geo.node_coords.shape == (3, 3)
        assert len(geo.parts) == 1
        assert geo.parts[0].element_type == "tria3"

    def test_read_nonexistent_raises(self, tmp_path):
        """FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            read_ensight_geometry(tmp_path / "nonexistent.geo")

    def test_write_read_multiple_parts(self, tmp_path):
        """Write/read multiple parts."""
        coords = np.random.rand(20, 3)
        conn1 = np.array([[0, 1, 2]], dtype=np.int32)
        conn2 = np.array([[3, 4, 5, 6]], dtype=np.int32)
        parts = [
            EnSightPart(part_id=1, description="tri", element_type="tria3",
                        connectivity=conn1),
            EnSightPart(part_id=2, description="quad", element_type="quad4",
                        connectivity=conn2),
        ]

        path = tmp_path / "multi.geo"
        write_ensight_geometry(path, coords, parts)

        geo = read_ensight_geometry(path)
        assert len(geo.parts) == 2
        assert geo.parts[0].element_type == "tria3"
        assert geo.parts[1].element_type == "quad4"


# ---------------------------------------------------------------------------
# Variable write/read tests
# ---------------------------------------------------------------------------


class TestVariableIO:
    """Test variable file write/read."""

    def test_write_scalar_variable(self, tmp_path):
        """Write scalar variable file."""
        var = EnSightVariable(description="pressure", variable_type="scalar",
                              part_id=1, values=np.array([101325.0, 101300.0]))
        path = tmp_path / "p.scl"
        write_ensight_variable(path, [var], description="pressure")
        assert path.exists()

    def test_write_vector_variable(self, tmp_path):
        """Write vector variable file."""
        var = EnSightVariable(description="velocity", variable_type="vector",
                              part_id=1,
                              values=np.array([[1.0, 0.0, 0.0], [0.5, 0.1, 0.0]]))
        path = tmp_path / "U.vel"
        write_ensight_variable(path, [var], description="velocity")
        assert path.exists()

    def test_read_scalar_variable(self, tmp_path):
        """Read scalar variable."""
        values = np.array([101325.0, 101300.0, 101350.0])
        var = EnSightVariable(description="pressure", variable_type="scalar",
                              part_id=1, values=values)
        path = tmp_path / "p.scl"
        write_ensight_variable(path, [var], description="pressure")

        result = read_ensight_variable(path)
        assert len(result) >= 1
        assert result[0].part_id == 1

    def test_read_nonexistent_raises(self, tmp_path):
        """FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            read_ensight_variable(tmp_path / "nonexistent.scl")


# ---------------------------------------------------------------------------
# Case file tests
# ---------------------------------------------------------------------------


class TestCaseFileIO:
    """Test case file write/read."""

    def test_write_case_file(self, tmp_path):
        """Write case file."""
        case = EnSightCase(
            title="test",
            geometry_file="case.geo",
            variables={"p": "case.scl", "U": "case.vel"},
            time_values=[0.0, 0.001, 0.002],
        )
        path = tmp_path / "test.case"
        write_ensight_case(path, case)
        assert path.exists()

    def test_read_case_file(self, tmp_path):
        """Read case file."""
        case = EnSightCase(
            title="test",
            geometry_file="case.geo",
            variables={"p": "case.scl"},
        )
        path = tmp_path / "test.case"
        write_ensight_case(path, case)

        result = read_ensight_case(path)
        assert result.geometry_file == "case.geo"

    def test_read_nonexistent_raises(self, tmp_path):
        """FileNotFoundError for missing case file."""
        with pytest.raises(FileNotFoundError):
            read_ensight_case(tmp_path / "nonexistent.case")
