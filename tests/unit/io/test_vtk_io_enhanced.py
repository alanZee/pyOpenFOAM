"""Tests for vtk_io_enhanced — binary VTK and multi-block VTK output."""

import os
import struct
import base64
import tempfile

import numpy as np
import pytest
import torch

from pyfoam.io.vtk_io_enhanced import (
    write_vtk_binary,
    write_vtm_multiblock,
    VTKBlock,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _simple_mesh():
    """Create a simple 1-cell hex mesh for testing."""
    points = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=torch.float64)

    faces = [
        np.array([0, 1, 5, 4]),
        np.array([2, 3, 7, 6]),
        np.array([0, 1, 2, 3]),
        np.array([4, 5, 6, 7]),
        np.array([0, 4, 7, 3]),
        np.array([1, 2, 6, 5]),
    ]

    owner = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.int64)
    neighbour = torch.tensor([], dtype=torch.int64)

    return points, faces, owner, neighbour, 1


# ---------------------------------------------------------------------------
# Binary VTK tests
# ---------------------------------------------------------------------------


class TestWriteVTKBinary:
    """Test binary VTK writing."""

    def test_write_creates_file(self, tmp_path):
        """Binary VTK file is created."""
        points, faces, owner, neighbour, n_cells = _simple_mesh()
        out = tmp_path / "test.vtk"

        write_vtk_binary(out, points, faces, owner, neighbour, n_cells)
        assert out.exists()

    def test_write_header(self, tmp_path):
        """Binary VTK file has correct header."""
        points, faces, owner, neighbour, n_cells = _simple_mesh()
        out = tmp_path / "test.vtk"

        write_vtk_binary(out, points, faces, owner, neighbour, n_cells)

        with open(out, "rb") as f:
            lines = []
            for _ in range(4):
                lines.append(f.readline())
            assert b"vtk DataFile Version 3.0" in lines[0]
            assert b"BINARY" in lines[2]
            assert b"UNSTRUCTURED_GRID" in lines[3]

    def test_write_with_cell_data(self, tmp_path):
        """Binary VTK with cell data."""
        points, faces, owner, neighbour, n_cells = _simple_mesh()
        cell_data = {"p": torch.tensor([101325.0]), "T": torch.tensor([300.0])}
        out = tmp_path / "test_data.vtk"

        write_vtk_binary(out, points, faces, owner, neighbour, n_cells,
                         cell_data=cell_data)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_write_with_point_data(self, tmp_path):
        """Binary VTK with point data."""
        points, faces, owner, neighbour, n_cells = _simple_mesh()
        point_data = {"U": torch.randn(8, 3, dtype=torch.float64)}
        out = tmp_path / "test_pdata.vtk"

        write_vtk_binary(out, points, faces, owner, neighbour, n_cells,
                         point_data=point_data)
        assert out.exists()

    def test_write_vector_cell_data(self, tmp_path):
        """Binary VTK with vector cell data."""
        points, faces, owner, neighbour, n_cells = _simple_mesh()
        cell_data = {"U": torch.tensor([[1.0, 0.0, 0.0]])}
        out = tmp_path / "test_vec.vtk"

        write_vtk_binary(out, points, faces, owner, neighbour, n_cells,
                         cell_data=cell_data)
        assert out.exists()

    def test_write_empty_cell_data(self, tmp_path):
        """Binary VTK without cell data."""
        points, faces, owner, neighbour, n_cells = _simple_mesh()
        out = tmp_path / "test_empty.vtk"

        write_vtk_binary(out, points, faces, owner, neighbour, n_cells)
        content = out.read_bytes()
        assert b"CELL_DATA" not in content


# ---------------------------------------------------------------------------
# Multi-block VTK tests
# ---------------------------------------------------------------------------


class TestWriteVTMMultiblock:
    """Test multi-block VTK writing."""

    def _make_block(self, name="block0"):
        """Create a simple VTKBlock."""
        points, faces, owner, neighbour, n_cells = _simple_mesh()
        return VTKBlock(
            name=name,
            points=points,
            faces=faces,
            owner=owner,
            neighbour=neighbour,
            n_cells=n_cells,
        )

    def test_write_creates_vtm(self, tmp_path):
        """VTM manifest file is created."""
        blocks = [self._make_block("b0"), self._make_block("b1")]
        out = tmp_path / "output.vtm"

        write_vtm_multiblock(out, blocks, binary_arrays=False)
        assert out.exists()

    def test_write_creates_block_files(self, tmp_path):
        """Individual VTU block files are created."""
        blocks = [self._make_block("b0")]
        out = tmp_path / "output.vtm"

        write_vtm_multiblock(out, blocks, binary_arrays=False)
        block_dir = tmp_path / "output_blocks"
        assert block_dir.exists()
        assert (block_dir / "block_0000.vtu").exists()

    def test_vtm_references_blocks(self, tmp_path):
        """VTM file references block files."""
        blocks = [self._make_block("b0"), self._make_block("b1")]
        out = tmp_path / "output.vtm"

        write_vtm_multiblock(out, blocks, binary_arrays=False)
        content = out.read_text()
        assert "vtkMultiBlockDataSet" in content
        assert "block_0000.vtu" in content
        assert "block_0001.vtu" in content
        assert "b0" in content
        assert "b1" in content

    def test_write_with_binary_arrays(self, tmp_path):
        """Multi-block with base64 binary arrays."""
        blocks = [self._make_block()]
        out = tmp_path / "output.vtm"

        write_vtm_multiblock(out, blocks, binary_arrays=True)
        block_dir = tmp_path / "output_blocks"
        vtu = (block_dir / "block_0000.vtu").read_text()
        assert 'format="binary"' in vtu

    def test_write_with_cell_data(self, tmp_path):
        """Multi-block with cell data."""
        block = VTKBlock(
            name="data_block",
            points=torch.randn(8, 3, dtype=torch.float64),
            faces=[np.array([0, 1, 5, 4])],
            owner=torch.tensor([0], dtype=torch.int64),
            neighbour=torch.tensor([], dtype=torch.int64),
            n_cells=1,
            cell_data={"p": torch.tensor([1.0])},
        )

        out = tmp_path / "output.vtm"
        write_vtm_multiblock(out, [block], binary_arrays=False)
        assert out.exists()

    def test_vtk_block_dataclass(self):
        """VTKBlock stores data correctly."""
        points, faces, owner, neighbour, n_cells = _simple_mesh()
        block = VTKBlock(
            name="test",
            points=points,
            faces=faces,
            owner=owner,
            neighbour=neighbour,
            n_cells=n_cells,
        )

        assert block.name == "test"
        assert block.n_cells == 1
        assert block.points.shape == (8, 3)
        assert block.cell_data == {}
        assert block.point_data == {}
