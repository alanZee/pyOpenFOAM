"""
Enhanced VTK I/O — binary VTK and multi-block VTK output.

Extends :mod:`pyfoam.io.vtk_io` with:

- Binary VTK legacy format (.vtk)
- Multi-block VTK XML format (.vtm / .vtmb)
- Efficient writing of large datasets via base64-encoded binary arrays

Usage::

    from pyfoam.io.vtk_io_enhanced import (
        write_vtk_binary,
        write_vtm_multiblock,
    )

    write_vtk_binary("output.vtk", points, faces, owner, neighbour,
                     n_cells, cell_data=cell_data)
    write_vtm_multiblock("output.vtm", blocks)

References
----------
- VTK file format specification
- VTK XML format specification
"""

from __future__ import annotations

import base64
import logging
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "write_vtk_binary",
    "write_vtm_multiblock",
    "VTKBlock",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class VTKBlock:
    """A block in a multi-block VTK dataset.

    Attributes:
        name: Block name.
        points: Vertex coordinates, shape ``(n_points, 3)``.
        faces: List of face vertex-index arrays.
        owner: Owner cell indices.
        neighbour: Neighbour cell indices.
        n_cells: Total number of cells.
        cell_data: Optional dict of cell data fields.
        point_data: Optional dict of point data fields.
    """

    def __init__(
        self,
        name: str,
        points: torch.Tensor,
        faces: List[np.ndarray],
        owner: torch.Tensor,
        neighbour: torch.Tensor,
        n_cells: int,
        cell_data: Optional[Dict[str, torch.Tensor]] = None,
        point_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        self.name = name
        self.points = points
        self.faces = faces
        self.owner = owner
        self.neighbour = neighbour
        self.n_cells = n_cells
        self.cell_data = cell_data or {}
        self.point_data = point_data or {}


# ---------------------------------------------------------------------------
# Binary VTK Legacy format
# ---------------------------------------------------------------------------


def write_vtk_binary(
    path: Union[str, Path],
    points: torch.Tensor,
    faces: List[np.ndarray],
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    n_cells: int,
    cell_data: Optional[Dict[str, torch.Tensor]] = None,
    point_data: Optional[Dict[str, torch.Tensor]] = None,
    title: str = "pyOpenFOAM data",
) -> None:
    """Write an unstructured grid in VTK legacy binary format.

    Binary format is more compact and faster to read/write than ASCII,
    especially for large datasets.

    Args:
        path: Output file path.
        points: Vertex coordinates, shape ``(n_points, 3)``.
        faces: List of face vertex-index arrays.
        owner: Owner cell indices.
        neighbour: Neighbour cell indices (internal faces only).
        n_cells: Total number of cells.
        cell_data: Optional dict of cell data fields.
        point_data: Optional dict of point data fields.
        title: VTK file title.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_points = points.shape[0]
    n_faces = len(faces)
    n_internal = neighbour.shape[0]

    # Build cell -> faces mapping
    cell_faces: Dict[int, List[int]] = {i: [] for i in range(n_cells)}
    for face_idx in range(n_faces):
        own = owner[face_idx].item()
        cell_faces[own].append(face_idx)
        if face_idx < n_internal:
            nei = neighbour[face_idx].item()
            cell_faces[nei].append(face_idx)

    pts = points.detach().cpu().numpy().astype(np.float64)

    with open(path, "wb") as f:
        # Header (ASCII)
        f.write(b"# vtk DataFile Version 3.0\n")
        f.write(f"{title}\n".encode())
        f.write(b"BINARY\n")
        f.write(b"DATASET UNSTRUCTURED_GRID\n")

        # Points
        f.write(f"POINTS {n_points} double\n".encode())
        f.write(pts.tobytes())

        # Cells
        total_entries = 0
        for cell_idx in range(n_cells):
            cf = cell_faces[cell_idx]
            total_entries += 1
            for face_idx in cf:
                n_verts = len(faces[face_idx])
                total_entries += 1 + n_verts

        f.write(f"CELLS {n_cells} {total_entries}\n".encode())
        for cell_idx in range(n_cells):
            cf = cell_faces[cell_idx]
            cell_data_list = [len(cf)]
            for face_idx in cf:
                verts = faces[face_idx]
                cell_data_list.append(len(verts))
                cell_data_list.extend(verts)
            arr = np.array(cell_data_list, dtype=np.int32)
            f.write(arr.tobytes())

        # Cell types
        f.write(f"CELL_TYPES {n_cells}\n".encode())
        types = np.full(n_cells, 42, dtype=np.int32)  # VTK_POLYHEDRON
        f.write(types.tobytes())

        # Cell data
        if cell_data:
            f.write(f"CELL_DATA {n_cells}\n".encode())
            for name, data in cell_data.items():
                _write_binary_data_field(f, name, data, "cell")

        # Point data
        if point_data:
            f.write(f"POINT_DATA {n_points}\n".encode())
            for name, data in point_data.items():
                _write_binary_data_field(f, name, data, "point")

    logger.info("Wrote binary VTK file: %s", path)


def _write_binary_data_field(
    f, name: str, data: torch.Tensor, location: str
) -> None:
    """Write a data field in VTK binary format."""
    data_np = data.detach().cpu().numpy().astype(np.float64)

    if data_np.ndim == 1:
        f.write(f"SCALARS {name} double 1\n".encode())
        f.write(b"LOOKUP_TABLE default\n")
        f.write(data_np.tobytes())
    elif data_np.ndim == 2 and data_np.shape[1] == 3:
        # Interleave for VTK: x0 y0 z0 x1 y1 z1 ...
        interleaved = data_np.flatten()
        f.write(f"VECTORS {name} double\n".encode())
        f.write(interleaved.tobytes())
    elif data_np.ndim == 3 and data_np.shape[1] == 3 and data_np.shape[2] == 3:
        f.write(f"TENSORS {name} double\n".encode())
        # VTK expects row-major 3x3 tensors
        for i in range(data_np.shape[0]):
            f.write(data_np[i].tobytes())


# ---------------------------------------------------------------------------
# Multi-block VTK XML format (.vtm / .vtmb)
# ---------------------------------------------------------------------------


def write_vtm_multiblock(
    path: Union[str, Path],
    blocks: List[VTKBlock],
    *,
    binary_arrays: bool = True,
) -> None:
    """Write a multi-block VTK dataset.

    Each block is written as a separate ``.vtu`` file, and a ``.vtm``
    manifest file references them all.  This is useful for decomposed
    cases where each processor is a separate block.

    Args:
        path: Output ``.vtm`` file path.
        blocks: List of :class:`VTKBlock` objects.
        binary_arrays: If True, use base64-encoded binary for data arrays.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset_dir = path.parent / (path.stem + "_blocks")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Write each block as a VTU file
    vtu_paths: List[Path] = []
    for i, block in enumerate(blocks):
        vtu_name = f"block_{i:04d}.vtu"
        vtu_path = dataset_dir / vtu_name
        _write_vtu_piece(vtu_path, block, binary_arrays=binary_arrays)
        vtu_paths.append(vtu_path)

    # Write the .vtm manifest
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="vtkMultiBlockDataSet" version="1.0"'
                ' byte_order="LittleEndian">\n')
        f.write("  <vtkMultiBlockDataSet>\n")
        for i, (block, vtu_path) in enumerate(zip(blocks, vtu_paths)):
            rel_path = vtu_path.relative_to(path.parent)
            f.write(f'    <Block index="{i}" name="{block.name}">\n')
            f.write(f'      <DataSet index="0" file="{rel_path}"/>\n')
            f.write("    </Block>\n")
        f.write("  </vtkMultiBlockDataSet>\n")
        f.write("</VTKFile>\n")

    logger.info("Wrote multi-block VTK: %s (%d blocks)", path, len(blocks))


def _write_vtu_piece(
    path: Path,
    block: VTKBlock,
    *,
    binary_arrays: bool = True,
) -> None:
    """Write a single VTU piece (block) to disk."""
    n_points = block.points.shape[0]
    n_cells = block.n_cells
    faces = block.faces
    owner = block.owner
    neighbour = block.neighbour
    n_internal = neighbour.shape[0]

    # Build cell -> faces mapping
    cell_faces: Dict[int, List[int]] = {i: [] for i in range(n_cells)}
    for face_idx in range(len(faces)):
        own = owner[face_idx].item()
        cell_faces[own].append(face_idx)
        if face_idx < n_internal:
            nei = neighbour[face_idx].item()
            cell_faces[nei].append(face_idx)

    pts = block.points.detach().cpu().numpy().astype(np.float64)

    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1"'
                ' byte_order="LittleEndian">\n')
        f.write("  <UnstructuredGrid>\n")
        f.write(f'    <Piece NumberOfPoints="{n_points}"'
                f' NumberOfCells="{n_cells}">\n')

        # Points
        f.write("      <Points>\n")
        if binary_arrays:
            _write_xml_binary_array(f, "        ", "Float64", pts, n_components=3)
        else:
            f.write('        <DataArray type="Float64" NumberOfComponents="3"'
                    ' format="ascii">\n')
            for i in range(n_points):
                f.write(f"          {pts[i, 0]:.10e} {pts[i, 1]:.10e}"
                        f" {pts[i, 2]:.10e}\n")
            f.write("        </DataArray>\n")
        f.write("      </Points>\n")

        # Cells
        f.write("      <Cells>\n")
        # Connectivity
        conn_list: List[int] = []
        offsets: List[int] = []
        offset = 0
        for cell_idx in range(n_cells):
            cell_verts = set()
            for face_idx in cell_faces[cell_idx]:
                for v in faces[face_idx]:
                    cell_verts.add(v)
            sorted_verts = sorted(cell_verts)
            conn_list.extend(sorted_verts)
            offset += len(sorted_verts)
            offsets.append(offset)

        conn_arr = np.array(conn_list, dtype=np.int32)
        off_arr = np.array(offsets, dtype=np.int32)
        types_arr = np.full(n_cells, 42, dtype=np.uint8)  # VTK_POLYHEDRON

        if binary_arrays:
            _write_xml_binary_array(f, "        ", "Int32", conn_arr, name="connectivity")
            _write_xml_binary_array(f, "        ", "Int32", off_arr, name="offsets")
            _write_xml_binary_array(f, "        ", "UInt8", types_arr, name="types")
        else:
            f.write('        <DataArray type="Int32" Name="connectivity"'
                    ' format="ascii">\n')
            for v in conn_list:
                f.write(f"          {v}\n")
            f.write("        </DataArray>\n")
            f.write('        <DataArray type="Int32" Name="offsets"'
                    ' format="ascii">\n')
            for o in offsets:
                f.write(f"          {o}\n")
            f.write("        </DataArray>\n")
            f.write('        <DataArray type="UInt8" Name="types"'
                    ' format="ascii">\n')
            for _ in range(n_cells):
                f.write("          42\n")
            f.write("        </DataArray>\n")

        f.write("      </Cells>\n")

        # Cell data
        if block.cell_data:
            f.write("      <CellData>\n")
            for name, data in block.cell_data.items():
                _write_xml_data_array(f, "        ", name, data, binary_arrays)
            f.write("      </CellData>\n")

        # Point data
        if block.point_data:
            f.write("      <PointData>\n")
            for name, data in block.point_data.items():
                _write_xml_data_array(f, "        ", name, data, binary_arrays)
            f.write("      </PointData>\n")

        f.write("    </Piece>\n")
        f.write("  </UnstructuredGrid>\n")
        f.write("</VTKFile>\n")


def _write_xml_data_array(
    f, indent: str, name: str, data: torch.Tensor, binary: bool
) -> None:
    """Write a data array in XML format (ascii or base64 binary)."""
    data_np = data.detach().cpu().numpy().astype(np.float64)

    if data_np.ndim == 1:
        if binary:
            _write_xml_binary_array(f, indent, "Float64", data_np, name=name)
        else:
            f.write(f'{indent}<DataArray type="Float64" Name="{name}"'
                    ' format="ascii">\n')
            for val in data_np:
                f.write(f"{indent}  {val:.10e}\n")
            f.write(f"{indent}</DataArray>\n")
    elif data_np.ndim == 2 and data_np.shape[1] == 3:
        if binary:
            _write_xml_binary_array(
                f, indent, "Float64", data_np, name=name, n_components=3,
            )
        else:
            f.write(f'{indent}<DataArray type="Float64" Name="{name}"'
                    ' NumberOfComponents="3" format="ascii">\n')
            for i in range(data_np.shape[0]):
                f.write(f"{indent}  {data_np[i, 0]:.10e}"
                        f" {data_np[i, 1]:.10e}"
                        f" {data_np[i, 2]:.10e}\n")
            f.write(f"{indent}</DataArray>\n")


def _write_xml_binary_array(
    f,
    indent: str,
    dtype_name: str,
    data: np.ndarray,
    *,
    name: str = "",
    n_components: int = 0,
) -> None:
    """Write a base64-encoded binary data array in VTU XML format."""
    # Format: header (1 uint32 = number of bytes) + raw data
    raw = data.tobytes()
    header = struct.pack("<I", len(raw))
    encoded = base64.b64encode(header + raw).decode("ascii")

    attr = f'type="{dtype_name}"'
    if name:
        attr += f' Name="{name}"'
    if n_components > 0:
        attr += f' NumberOfComponents="{n_components}"'
    attr += ' format="binary"'

    f.write(f"{indent}<DataArray {attr}>\n")
    # Break base64 into 76-char lines
    for i in range(0, len(encoded), 76):
        f.write(f"{indent}  {encoded[i:i+76]}\n")
    f.write(f"{indent}</DataArray>\n")
