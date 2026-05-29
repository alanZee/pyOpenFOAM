"""Tests for FoamFormatConverter — ASCII/binary format conversion."""

import pytest
from pathlib import Path

from pyfoam.io.foam_file import FileFormat, FoamFileHeader, write_foam_file, read_foam_file
from pyfoam.io.foam_format_convert import (
    FoamFormatConverter,
    convert_file,
    convert_directory,
)


# ---------------------------------------------------------------------------
# Helper: create test OpenFOAM files
# ---------------------------------------------------------------------------


def _create_test_field(path: Path, format_str: str = "ascii"):
    """Create a test volScalarField file."""
    header = FoamFileHeader(
        version="2.0",
        format=FileFormat(format_str),
        class_name="volScalarField",
        object="p",
    )
    body = (
        "dimensions      [0 2 -2 0 0 0 0];\n\n"
        "internalField nonuniform 3\n(\n101325\n101300\n101275\n);\n\n"
        "boundaryField\n{\n    inlet\n    {\n        type fixedValue;\n        value uniform 0;\n    }\n}\n"
    )
    write_foam_file(path, header, body)
    return path


# ---------------------------------------------------------------------------
# File format detection
# ---------------------------------------------------------------------------


class TestFormatDetection:
    """Test file format detection."""

    def test_detect_ascii(self, tmp_path):
        """Detect ASCII format."""
        path = _create_test_field(tmp_path / "p", "ascii")
        converter = FoamFormatConverter()
        fmt = converter.detect_file_format(path)
        assert fmt == FileFormat.ASCII

    def test_detect_binary(self, tmp_path):
        """Detect binary format."""
        path = _create_test_field(tmp_path / "p", "binary")
        converter = FoamFormatConverter()
        fmt = converter.detect_file_format(path)
        assert fmt == FileFormat.BINARY


# ---------------------------------------------------------------------------
# Single file conversion
# ---------------------------------------------------------------------------


class TestConvertFile:
    """Test single file conversion."""

    def test_ascii_to_binary_same_format(self, tmp_path):
        """Convert ASCII to binary (header change only)."""
        src = _create_test_field(tmp_path / "p", "ascii")
        dst = tmp_path / "p_binary"

        converter = FoamFormatConverter()
        result = converter.convert_file(src, dst, target_format="binary")
        assert result == FileFormat.BINARY

        # Verify the output header says binary
        header, _ = read_foam_file(dst)
        assert header.format == FileFormat.BINARY

    def test_binary_to_ascii(self, tmp_path):
        """Convert binary to ASCII."""
        src = _create_test_field(tmp_path / "p", "binary")
        dst = tmp_path / "p_ascii"

        converter = FoamFormatConverter()
        result = converter.convert_file(src, dst, target_format="ascii")
        assert result == FileFormat.ASCII

    def test_same_format_copies(self, tmp_path):
        """Same format just copies the file."""
        src = _create_test_field(tmp_path / "p", "ascii")
        dst = tmp_path / "p_copy"

        converter = FoamFormatConverter()
        converter.convert_file(src, dst, target_format="ascii")

        src_content = src.read_text()
        dst_content = dst.read_text()
        assert src_content == dst_content

    def test_nonexistent_source_raises(self, tmp_path):
        """FileNotFoundError for missing source file."""
        converter = FoamFormatConverter()
        with pytest.raises(FileNotFoundError):
            converter.convert_file(
                tmp_path / "missing", tmp_path / "out", "ascii"
            )

    def test_creates_parent_dirs(self, tmp_path):
        """Output file creates parent directories."""
        src = _create_test_field(tmp_path / "p", "ascii")
        dst = tmp_path / "sub" / "dir" / "p"

        converter = FoamFormatConverter()
        converter.convert_file(src, dst, "ascii")
        assert dst.exists()


# ---------------------------------------------------------------------------
# Directory conversion
# ---------------------------------------------------------------------------


class TestConvertDirectory:
    """Test directory-level conversion."""

    def test_convert_directory(self, tmp_path):
        """Convert all files in a directory."""
        src_dir = tmp_path / "src"
        dst_dir = tmp_path / "dst"
        src_dir.mkdir()

        _create_test_field(src_dir / "p", "ascii")
        _create_test_field(src_dir / "U", "ascii")

        converter = FoamFormatConverter()
        count = converter.convert_directory(src_dir, dst_dir, "binary")

        assert count == 2
        assert (dst_dir / "p").exists()
        assert (dst_dir / "U").exists()

    def test_convert_directory_recursive(self, tmp_path):
        """Convert files recursively."""
        src_dir = tmp_path / "src"
        sub_dir = src_dir / "0"
        sub_dir.mkdir(parents=True)

        _create_test_field(sub_dir / "p", "ascii")

        dst_dir = tmp_path / "dst"
        converter = FoamFormatConverter()
        count = converter.convert_directory(src_dir, dst_dir, "binary", recursive=True)

        assert count == 1
        assert (dst_dir / "0" / "p").exists()

    def test_convert_empty_directory(self, tmp_path):
        """Convert empty directory returns 0."""
        src_dir = tmp_path / "empty"
        src_dir.mkdir()

        converter = FoamFormatConverter()
        count = converter.convert_directory(src_dir, tmp_path / "out", "binary")
        assert count == 0

    def test_non_recursive(self, tmp_path):
        """Non-recursive conversion."""
        src_dir = tmp_path / "src"
        sub_dir = src_dir / "sub"
        sub_dir.mkdir()

        _create_test_field(src_dir / "p", "ascii")
        _create_test_field(sub_dir / "U", "ascii")

        converter = FoamFormatConverter()
        count = converter.convert_directory(
            src_dir, tmp_path / "dst", "binary", recursive=False
        )

        # Only p should be converted (U is in a subdirectory)
        assert count == 1


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """Test module-level functions."""

    def test_convert_file_function(self, tmp_path):
        """convert_file convenience function."""
        src = _create_test_field(tmp_path / "p", "ascii")
        dst = tmp_path / "p_out"

        result = convert_file(src, dst, "binary")
        assert result == FileFormat.BINARY

    def test_convert_directory_function(self, tmp_path):
        """convert_directory convenience function."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        _create_test_field(src_dir / "p", "ascii")

        count = convert_directory(src_dir, tmp_path / "dst", "binary")
        assert count == 1
