"""Tests for field file reading and writing."""

import pytest
import torch
import numpy as np
from pathlib import Path

from pyfoam.io.field_io import (
    BoundaryField,
    BoundaryPatch,
    FieldData,
    parse_boundary_field,
    parse_internal_field,
    read_dimensions,
    read_field,
    write_field,
)
from pyfoam.io.foam_file import FoamFileHeader, FileFormat


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ascii_scalar_field(tmp_path):
    """Create a sample ASCII scalar field file."""
    content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 1.0;

boundaryField
{
    inlet
    {
        type        fixedValue;
        value       uniform 0;
    }
    outlet
    {
        type        zeroGradient;
    }
}
"""
    path = tmp_path / "p"
    path.write_text(content)
    return path


@pytest.fixture
def ascii_vector_field(tmp_path):
    """Create a sample ASCII vector field file."""
    content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (1 0 0);

boundaryField
{
    inlet
    {
        type        fixedValue;
        value       uniform (1 0 0);
    }
}
"""
    path = tmp_path / "U"
    path.write_text(content)
    return path


@pytest.fixture
def nonuniform_scalar_field(tmp_path):
    """Create a sample nonuniform scalar field file."""
    content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}

dimensions      [0 2 -2 0 0 0 0];

internalField   nonuniform List<scalar> 5
(
    1.0
    2.0
    3.0
    4.0
    5.0
);

boundaryField
{
}
"""
    path = tmp_path / "p_nonuniform"
    path.write_text(content)
    return path


@pytest.fixture
def nonuniform_vector_field(tmp_path):
    """Create a sample nonuniform vector field file."""
    content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}

dimensions      [0 1 -1 0 0 0 0];

internalField   nonuniform List<vector> 3
(
    (1 0 0)
    (0 1 0)
    (0 0 1)
);

boundaryField
{
}
"""
    path = tmp_path / "U_nonuniform"
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# read_dimensions
# ---------------------------------------------------------------------------


class TestReadDimensions:
    def test_parse_dimensions(self):
        """Parse dimensions vector."""
        content = "dimensions      [0 2 -2 0 0 0 0];"
        dims = read_dimensions(content)
        assert dims == [0.0, 2.0, -2.0, 0.0, 0.0, 0.0, 0.0]

    def test_parse_dimensions_negative(self):
        """Parse dimensions with negative values."""
        content = "dimensions      [1 -2 0 0 0 0 0];"
        dims = read_dimensions(content)
        assert dims == [1.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def test_no_dimensions_raises(self):
        """ValueError when no dimensions found."""
        with pytest.raises(ValueError, match="No 'dimensions'"):
            read_dimensions("no dimensions here")

    def test_wrong_count_raises(self):
        """ValueError when wrong number of dimensions."""
        content = "dimensions      [0 2 -2];"
        with pytest.raises(ValueError, match="7 values"):
            read_dimensions(content)


# ---------------------------------------------------------------------------
# parse_internal_field
# ---------------------------------------------------------------------------


class TestParseInternalField:
    def test_uniform_scalar(self):
        """Parse uniform scalar field."""
        content = "internalField   uniform 1.0;"
        value, is_uniform = parse_internal_field(content, "scalar")
        assert is_uniform is True
        assert value == 1.0

    def test_uniform_vector(self):
        """Parse uniform vector field."""
        content = "internalField   uniform (1 0 0);"
        value, is_uniform = parse_internal_field(content, "vector")
        assert is_uniform is True
        assert value == (1.0, 0.0, 0.0)

    def test_nonuniform_scalar(self, nonuniform_scalar_field):
        """Parse nonuniform scalar field."""
        content = nonuniform_scalar_field.read_text()
        # Remove header
        from pyfoam.io.foam_file import split_header_body
        _, body = split_header_body(content)
        value, is_uniform = parse_internal_field(body, "scalar")
        assert is_uniform is False
        assert isinstance(value, torch.Tensor)
        assert value.shape == (5,)
        assert torch.allclose(value, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64))

    def test_nonuniform_vector(self, nonuniform_vector_field):
        """Parse nonuniform vector field."""
        content = nonuniform_vector_field.read_text()
        from pyfoam.io.foam_file import split_header_body
        _, body = split_header_body(content)
        value, is_uniform = parse_internal_field(body, "vector")
        assert is_uniform is False
        assert isinstance(value, torch.Tensor)
        assert value.shape == (3, 3)
        expected = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float64)
        assert torch.allclose(value, expected)

    def test_no_internal_field_raises(self):
        """ValueError when no internalField found."""
        with pytest.raises(ValueError):
            parse_internal_field("no field here", "scalar")


# ---------------------------------------------------------------------------
# parse_boundary_field
# ---------------------------------------------------------------------------


class TestParseBoundaryField:
    def test_parse_boundary(self):
        """Parse boundary field with patches."""
        content = """
boundaryField
{
    inlet
    {
        type        fixedValue;
        value       uniform 0;
    }
    outlet
    {
        type        zeroGradient;
    }
}
"""
        bf = parse_boundary_field(content)
        assert len(bf) == 2
        assert "inlet" in bf
        assert "outlet" in bf
        assert bf["inlet"].patch_type == "fixedValue"
        assert bf["outlet"].patch_type == "zeroGradient"

    def test_no_boundary_field(self):
        """Empty boundary field when section missing."""
        bf = parse_boundary_field("no boundary here")
        assert len(bf) == 0


# ---------------------------------------------------------------------------
# BoundaryField
# ---------------------------------------------------------------------------


class TestBoundaryField:
    def test_getitem(self):
        """Get patch by name."""
        patches = [
            BoundaryPatch("inlet", "fixedValue"),
            BoundaryPatch("outlet", "zeroGradient"),
        ]
        bf = BoundaryField(patches)
        assert bf["inlet"].name == "inlet"

    def test_getitem_not_found(self):
        """KeyError for missing patch."""
        bf = BoundaryField([])
        with pytest.raises(KeyError):
            bf["missing"]

    def test_contains(self):
        """Check patch existence."""
        patches = [BoundaryPatch("inlet", "fixedValue")]
        bf = BoundaryField(patches)
        assert "inlet" in bf
        assert "outlet" not in bf

    def test_len(self):
        """Length of boundary field."""
        patches = [
            BoundaryPatch("a", "fixedValue"),
            BoundaryPatch("b", "zeroGradient"),
        ]
        bf = BoundaryField(patches)
        assert len(bf) == 2

    def test_iter(self):
        """Iterate over patches."""
        patches = [
            BoundaryPatch("a", "fixedValue"),
            BoundaryPatch("b", "zeroGradient"),
        ]
        bf = BoundaryField(patches)
        names = [p.name for p in bf]
        assert names == ["a", "b"]


# ---------------------------------------------------------------------------
# FieldData
# ---------------------------------------------------------------------------


class TestFieldData:
    def test_init(self):
        """FieldData initialization."""
        header = FoamFileHeader(class_name="volScalarField")
        fd = FieldData(
            header=header,
            dimensions=[0, 2, -2, 0, 0, 0, 0],
            internal_field=1.0,
            boundary_field=BoundaryField(),
            is_uniform=True,
            scalar_type="scalar",
        )
        assert fd.is_uniform is True
        assert fd.scalar_type == "scalar"

    def test_repr(self):
        """repr includes key info."""
        header = FoamFileHeader(class_name="volScalarField")
        fd = FieldData(
            header=header,
            dimensions=[0, 2, -2, 0, 0, 0, 0],
            internal_field=1.0,
            boundary_field=BoundaryField(),
        )
        r = repr(fd)
        assert "volScalarField" in r


# ---------------------------------------------------------------------------
# read_field
# ---------------------------------------------------------------------------


class TestReadField:
    def test_read_uniform_scalar(self, ascii_scalar_field):
        """Read uniform scalar field."""
        fd = read_field(ascii_scalar_field)
        assert fd.is_uniform is True
        assert fd.internal_field == 1.0
        assert fd.scalar_type == "scalar"
        assert fd.dimensions == [0.0, 2.0, -2.0, 0.0, 0.0, 0.0, 0.0]

    def test_read_uniform_vector(self, ascii_vector_field):
        """Read uniform vector field."""
        fd = read_field(ascii_vector_field)
        assert fd.is_uniform is True
        assert fd.internal_field == (1.0, 0.0, 0.0)
        assert fd.scalar_type == "vector"

    def test_read_nonuniform_scalar(self, nonuniform_scalar_field):
        """Read nonuniform scalar field."""
        fd = read_field(nonuniform_scalar_field)
        assert fd.is_uniform is False
        assert isinstance(fd.internal_field, torch.Tensor)
        assert fd.internal_field.shape == (5,)

    def test_read_nonuniform_vector(self, nonuniform_vector_field):
        """Read nonuniform vector field."""
        fd = read_field(nonuniform_vector_field)
        assert fd.is_uniform is False
        assert isinstance(fd.internal_field, torch.Tensor)
        assert fd.internal_field.shape == (3, 3)

    def test_read_boundary_patches(self, ascii_scalar_field):
        """Read boundary patches."""
        fd = read_field(ascii_scalar_field)
        assert len(fd.boundary_field) == 2
        assert "inlet" in fd.boundary_field
        assert "outlet" in fd.boundary_field


# ---------------------------------------------------------------------------
# write_field
# ---------------------------------------------------------------------------


class TestWriteField:
    def test_write_uniform_scalar(self, tmp_path):
        """Write uniform scalar field."""
        header = FoamFileHeader(
            class_name="volScalarField",
            object="p",
        )
        fd = FieldData(
            header=header,
            dimensions=[0, 2, -2, 0, 0, 0, 0],
            internal_field=1.0,
            boundary_field=BoundaryField(),
            is_uniform=True,
            scalar_type="scalar",
        )
        path = tmp_path / "p"
        write_field(path, fd)

        # Read back
        fd2 = read_field(path)
        assert fd2.is_uniform is True
        assert fd2.internal_field == 1.0

    def test_write_uniform_vector(self, tmp_path):
        """Write uniform vector field."""
        header = FoamFileHeader(
            class_name="volVectorField",
            object="U",
        )
        fd = FieldData(
            header=header,
            dimensions=[0, 1, -1, 0, 0, 0, 0],
            internal_field=(1.0, 0.0, 0.0),
            boundary_field=BoundaryField(),
            is_uniform=True,
            scalar_type="vector",
        )
        path = tmp_path / "U"
        write_field(path, fd)

        fd2 = read_field(path)
        assert fd2.is_uniform is True
        assert fd2.internal_field == (1.0, 0.0, 0.0)

    def test_write_nonuniform_scalar(self, tmp_path):
        """Write nonuniform scalar field."""
        header = FoamFileHeader(
            class_name="volScalarField",
            object="p",
        )
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        fd = FieldData(
            header=header,
            dimensions=[0, 2, -2, 0, 0, 0, 0],
            internal_field=values,
            boundary_field=BoundaryField(),
            is_uniform=False,
            scalar_type="scalar",
        )
        path = tmp_path / "p"
        write_field(path, fd)

        fd2 = read_field(path)
        assert fd2.is_uniform is False
        assert torch.allclose(fd2.internal_field, values)

    def test_write_with_boundary(self, tmp_path):
        """Write field with boundary patches."""
        header = FoamFileHeader(
            class_name="volScalarField",
            object="p",
        )
        patches = [
            BoundaryPatch("inlet", "fixedValue", value=0.0),
            BoundaryPatch("outlet", "zeroGradient"),
        ]
        fd = FieldData(
            header=header,
            dimensions=[0, 2, -2, 0, 0, 0, 0],
            internal_field=1.0,
            boundary_field=BoundaryField(patches),
            is_uniform=True,
            scalar_type="scalar",
        )
        path = tmp_path / "p"
        write_field(path, fd)

        fd2 = read_field(path)
        assert len(fd2.boundary_field) == 2

    def test_roundtrip_preserves_data(self, ascii_scalar_field):
        """Read -> write -> read produces identical data."""
        fd1 = read_field(ascii_scalar_field)

        # Write to new location
        path = ascii_scalar_field.parent / "p_copy"
        write_field(path, fd1, overwrite=True)

        fd2 = read_field(path)
        assert fd1.is_uniform == fd2.is_uniform
        assert fd1.internal_field == fd2.internal_field
        assert fd1.dimensions == fd2.dimensions
