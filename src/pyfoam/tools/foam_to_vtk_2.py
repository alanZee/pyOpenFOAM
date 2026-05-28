"""
foamToVTK enhanced — enhanced VTK export with polyhedra, multi-block,
and time series support.

Extends the basic :func:`foam_to_vtk` with:

- **Polyhedral cells**: Full VTK_POLYHEDRON support with face-stream
  encoding (the native VTK format for arbitrary polyhedra).
- **Multi-block output**: ``.vtm`` (VTK MultiBlock) files that group
  volume and surface data into a structured hierarchy.
- **Time series**: Writes a ``.pvd`` (ParaView Data) file that
  aggregates all time steps for one-click loading in ParaView.
- **Field selection**: Export specific fields by name.
- **Binary output**: Optional binary VTK format for smaller files.

Usage::

    from pyfoam.tools.foam_to_vtk_2 import foam_to_vtk_enhanced

    foam_to_vtk_enhanced(
        case_path="cavity",
        mesh=mesh,
        fields={"p": p_arr, "U": U_arr},
        time_range=[0, 1, 2],
        binary=False,
    )
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union
from xml.etree import ElementTree as ET

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["foam_to_vtk_enhanced"]

# VTK cell type constants
_VTK_TETRA = 10
_VTK_HEXAHEDRON = 12
_VTK_WEDGE = 13
_VTK_PYRAMID = 14
_VTK_POLYHEDRON = 42


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def foam_to_vtk_enhanced(
    case_path: Union[str, Path],
    time_range: Optional[Sequence[float]] = None,
    mesh: Optional["FvMesh"] = None,
    fields: Optional[Dict[str, np.ndarray]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    binary: bool = False,
    write_multiblock: bool = True,
    write_pvd: bool = True,
) -> Path:
    """Enhanced VTK export with polyhedra, multi-block, and time series.

    Parameters
    ----------
    case_path : str or Path
        Root of the OpenFOAM case directory.
    time_range : sequence of float, optional
        Subset of time values to export.  ``None`` exports all available.
    mesh : FvMesh, optional
        Pre-loaded mesh.
    fields : dict, optional
        ``{field_name: numpy_array}`` of per-cell field values.
    output_dir : str or Path, optional
        Directory for VTK output.  Defaults to ``<case_path>/VTK``.
    binary : bool
        If True, write binary VTK format.  Default: ASCII.
    write_multiblock : bool
        If True, write ``.vtm`` multi-block file per time step.  Default: True.
    write_pvd : bool
        If True, write ``timeseries.pvd`` for all time steps.  Default: True.

    Returns
    -------
    Path
        Path to the output directory.
    """
    case_dir = Path(case_path).resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    vtk_dir = Path(output_dir) if output_dir else case_dir / "VTK"
    os.makedirs(vtk_dir, exist_ok=True)

    # Determine time values
    if time_range is not None:
        times = sorted(float(t) for t in time_range)
    else:
        from pyfoam.tools.foam_list_times import foam_list_times

        times = foam_list_times(case_dir)
        if not times and mesh is not None:
            times = [0.0]

    if mesh is None:
        raise ValueError("No mesh provided.  Pass a mesh object directly.")

    # Pre-compute connectivity
    cell_verts, vtk_cell_types, face_streams = _compute_cell_vertices_and_types(mesh)

    # Extract boundary patches
    patch_data = _extract_boundary_patches(mesh)

    # Collect PVD entries: (time, vtu_path)
    pvd_entries: list[tuple[float, str]] = []

    for t in times:
        t_name = _format_time(t)

        # Write VTU (volume)
        vtu_filename = f"{t_name}.vtu"
        vtu_path = vtk_dir / vtu_filename
        _write_vtu_enhanced(
            vtu_path, mesh, cell_verts, vtk_cell_types,
            face_streams, fields, binary,
        )

        # Write VTP (surface patches)
        for patch_name, patch_faces, patch_owner_cells in patch_data:
            vtp_filename = f"{t_name}_{patch_name}.vtp"
            vtp_path = vtk_dir / vtp_filename
            _write_vtp_enhanced(
                vtp_path, mesh, patch_faces, patch_owner_cells, fields, binary,
            )

        # Write VTM (multi-block)
        if write_multiblock:
            vtm_filename = f"{t_name}.vtm"
            vtm_path = vtk_dir / vtm_filename
            _write_vtm(
                vtm_path, vtu_filename,
                [(f"{t_name}_{pn}.vtp", pn) for pn, _, _ in patch_data],
            )

        pvd_entries.append((t, vtu_filename))

    # Write PVD time series
    if write_pvd and len(pvd_entries) > 0:
        pvd_path = vtk_dir / "timeseries.pvd"
        _write_pvd(pvd_path, pvd_entries)

    return vtk_dir


# ---------------------------------------------------------------------------
# Cell-vertex connectivity with polyhedral face streams
# ---------------------------------------------------------------------------


def _compute_cell_vertices_and_types(mesh: "FvMesh"):
    """Build cell connectivity, VTK types, and polyhedral face streams.

    Returns
    -------
    cell_verts : list[list[int]]
        Point indices for each cell (sorted, unique).
    vtk_types : list[int]
        VTK cell type code for each cell.
    face_streams : list[list[int] | None]
        For polyhedral cells, the VTK face-stream encoding.
        For standard cells, None.
    """
    n_cells = mesh.n_cells
    faces = mesh.faces
    owner = mesh.owner.detach().cpu().numpy()
    neighbour = mesh.neighbour.detach().cpu().numpy()
    n_internal = mesh.n_internal_faces

    # Build cell -> list of face indices
    cell_to_faces: list[list[int]] = [[] for _ in range(n_cells)]
    cell_to_verts: list[set[int]] = [set() for _ in range(n_cells)]

    for fi, face in enumerate(faces):
        face_nodes = face.detach().cpu().numpy().tolist()
        c_own = int(owner[fi])
        cell_to_faces[c_own].append(fi)
        cell_to_verts[c_own].update(face_nodes)

        if fi < n_internal:
            c_nbr = int(neighbour[fi])
            cell_to_faces[c_nbr].append(fi)
            cell_to_verts[c_nbr].update(face_nodes)

    cell_verts = [sorted(verts) for verts in cell_to_verts]

    vtk_types = []
    face_streams = []
    for ci, verts in enumerate(cell_verts):
        nn = len(verts)
        if nn == 4:
            vtk_types.append(_VTK_TETRA)
            face_streams.append(None)
        elif nn == 5:
            vtk_types.append(_VTK_PYRAMID)
            face_streams.append(None)
        elif nn == 6:
            vtk_types.append(_VTK_WEDGE)
            face_streams.append(None)
        elif nn == 8:
            vtk_types.append(_VTK_HEXAHEDRON)
            face_streams.append(None)
        else:
            vtk_types.append(_VTK_POLYHEDRON)
            # Build face stream for polyhedral cell
            fs = _build_face_stream(mesh, cell_to_faces[ci])
            face_streams.append(fs)

    return cell_verts, vtk_types, face_streams


def _build_face_stream(mesh: "FvMesh", face_indices: list[int]) -> list[int]:
    """Build VTK polyhedral face-stream encoding.

    Format: nFaces, nPoints_face0, pt0, pt1, ..., nPoints_face1, ...
    """
    stream = [len(face_indices)]
    for fi in face_indices:
        face_nodes = mesh.faces[fi].detach().cpu().numpy().tolist()
        stream.append(len(face_nodes))
        stream.extend(face_nodes)
    return stream


# ---------------------------------------------------------------------------
# Boundary patch extraction
# ---------------------------------------------------------------------------


def _extract_boundary_patches(mesh: "FvMesh"):
    """Extract boundary face indices grouped by patch."""
    patches = []
    owner = mesh.owner.detach().cpu().numpy()
    n_internal = mesh.n_internal_faces

    for patch_info in mesh.boundary:
        name = patch_info["name"]
        start = patch_info["startFace"]
        n_faces = patch_info["nFaces"]

        face_indices = list(range(start, start + n_faces))
        owner_cells = [int(owner[fi]) for fi in face_indices]
        patches.append((name, face_indices, owner_cells))

    return patches


# ---------------------------------------------------------------------------
# VTU writing (enhanced)
# ---------------------------------------------------------------------------


def _write_vtu_enhanced(
    path: Path,
    mesh: "FvMesh",
    cell_verts: list[list[int]],
    vtk_types: list[int],
    face_streams: list[list[int] | None],
    fields: Optional[Dict[str, np.ndarray]] = None,
    binary: bool = False,
) -> None:
    """Write enhanced VTU with polyhedral support."""
    pts = mesh.points.detach().cpu().numpy()
    n_points = pts.shape[0]
    n_cells = len(cell_verts)

    # Build connectivity (standard cells) and face-streams (polyhedra)
    connectivity = []
    offsets = []
    running = 0
    has_polyhedra = any(fs is not None for fs in face_streams)

    for ci, verts in enumerate(cell_verts):
        connectivity.extend(verts)
        running += len(verts)
        offsets.append(running)

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write(
            '<VTKFile type="UnstructuredGrid" version="0.1" '
            'byte_order="LittleEndian">\n'
        )
        f.write("<UnstructuredGrid>\n")
        f.write(
            f'<Piece NumberOfPoints="{n_points}" '
            f'NumberOfCells="{n_cells}">\n'
        )

        # Points
        f.write("<Points>\n")
        if binary:
            _write_data_array_binary(f, "Points", pts.flatten(), n_components=3)
        else:
            _write_data_array_ascii(f, "Points", pts.flatten(), n_components=3)
        f.write("</Points>\n")

        # Cells
        f.write("<Cells>\n")
        if binary:
            _write_int_data_array_binary(
                f, "connectivity", np.array(connectivity, dtype=np.int64),
            )
            _write_int_data_array_binary(
                f, "offsets", np.array(offsets, dtype=np.int64),
            )
            _write_int_data_array_binary(
                f, "types", np.array(vtk_types, dtype=np.uint8),
            )
        else:
            _write_int_data_array_ascii(
                f, "connectivity", np.array(connectivity, dtype=np.int64),
            )
            _write_int_data_array_ascii(
                f, "offsets", np.array(offsets, dtype=np.int64),
            )
            _write_int_data_array_ascii(
                f, "types", np.array(vtk_types, dtype=np.uint8),
            )

        # Face-streams for polyhedral cells
        if has_polyhedra:
            fs_flat = []
            for fs in face_streams:
                if fs is not None:
                    fs_flat.extend(fs)
                else:
                    # Standard cell: empty face stream
                    fs_flat.append(0)
            if binary:
                _write_int_data_array_binary(
                    f, "faces", np.array(fs_flat, dtype=np.int64),
                )
            else:
                _write_int_data_array_ascii(
                    f, "faces", np.array(fs_flat, dtype=np.int64),
                )

        f.write("</Cells>\n")

        # Cell data
        if fields:
            f.write("<CellData>\n")
            for name, data in fields.items():
                if data.ndim == 1:
                    if binary:
                        _write_data_array_binary(f, name, data, n_components=1)
                    else:
                        _write_data_array_ascii(f, name, data, n_components=1)
                elif data.ndim == 2 and data.shape[1] == 3:
                    if binary:
                        _write_data_array_binary(f, name, data.flatten(), n_components=3)
                    else:
                        _write_data_array_ascii(f, name, data.flatten(), n_components=3)
            f.write("</CellData>\n")

        f.write("</Piece>\n")
        f.write("</UnstructuredGrid>\n")
        f.write("</VTKFile>\n")


# ---------------------------------------------------------------------------
# VTP writing (enhanced)
# ---------------------------------------------------------------------------


def _write_vtp_enhanced(
    path: Path,
    mesh: "FvMesh",
    face_indices: list[int],
    owner_cells: list[int],
    fields: Optional[Dict[str, np.ndarray]] = None,
    binary: bool = False,
) -> None:
    """Write enhanced VTP for a boundary patch."""
    pts = mesh.points.detach().cpu().numpy()
    faces = mesh.faces

    all_point_indices: set[int] = set()
    for fi in face_indices:
        face_nodes = faces[fi].detach().cpu().numpy().tolist()
        all_point_indices.update(face_nodes)

    sorted_points = sorted(all_point_indices)
    point_map = {old: new for new, old in enumerate(sorted_points)}

    sub_pts = pts[sorted_points]
    n_sub_points = len(sorted_points)

    polygons = []
    polygon_offsets = []
    running = 0
    for fi in face_indices:
        face_nodes = faces[fi].detach().cpu().numpy().tolist()
        remapped = [point_map[n] for n in face_nodes]
        polygons.extend(remapped)
        running += len(remapped)
        polygon_offsets.append(running)

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write(
            '<VTKFile type="PolyData" version="0.1" '
            'byte_order="LittleEndian">\n'
        )
        f.write("<PolyData>\n")
        n_polys = len(face_indices)
        f.write(
            f'<Piece NumberOfPoints="{n_sub_points}" '
            f'NumberOfPolys="{n_polys}">\n'
        )

        f.write("<Points>\n")
        if binary:
            _write_data_array_binary(f, "Points", sub_pts.flatten(), n_components=3)
        else:
            _write_data_array_ascii(f, "Points", sub_pts.flatten(), n_components=3)
        f.write("</Points>\n")

        f.write("<Polys>\n")
        if binary:
            _write_int_data_array_binary(
                f, "connectivity", np.array(polygons, dtype=np.int64),
            )
            _write_int_data_array_binary(
                f, "offsets", np.array(polygon_offsets, dtype=np.int64),
            )
        else:
            _write_int_data_array_ascii(
                f, "connectivity", np.array(polygons, dtype=np.int64),
            )
            _write_int_data_array_ascii(
                f, "offsets", np.array(polygon_offsets, dtype=np.int64),
            )
        f.write("</Polys>\n")

        if fields:
            f.write("<CellData>\n")
            for name, data in fields.items():
                if data.ndim == 1:
                    patch_data = data[owner_cells]
                    if binary:
                        _write_data_array_binary(f, name, patch_data, n_components=1)
                    else:
                        _write_data_array_ascii(f, name, patch_data, n_components=1)
                elif data.ndim == 2 and data.shape[1] == 3:
                    patch_data = data[owner_cells].flatten()
                    if binary:
                        _write_data_array_binary(f, name, patch_data, n_components=3)
                    else:
                        _write_data_array_ascii(f, name, patch_data, n_components=3)
            f.write("</CellData>\n")

        f.write("</Piece>\n")
        f.write("</PolyData>\n")
        f.write("</VTKFile>\n")


# ---------------------------------------------------------------------------
# VTM (multi-block) writing
# ---------------------------------------------------------------------------


def _write_vtm(
    path: Path,
    vtu_filename: str,
    vtp_entries: list[Tuple[str, str]],
) -> None:
    """Write a VTK MultiBlock (.vtm) file."""
    root = ET.Element("VTKFile", type="vtkMultiBlockDataSet", version="1.0")
    mb = ET.SubElement(root, "vtkMultiBlockDataSet")

    # Volume block
    block0 = ET.SubElement(mb, "Block", index="0", name="Volume")
    dataset0 = ET.SubElement(
        block0, "DataSet", index="0", file=vtu_filename,
    )

    # Surface block
    block1 = ET.SubElement(mb, "Block", index="1", name="Surface")
    for idx, (vtp_file, patch_name) in enumerate(vtp_entries):
        ET.SubElement(
            block1, "DataSet", index=str(idx),
            name=patch_name, file=vtp_file,
        )

    tree = ET.ElementTree(root)
    _indent_xml(root)
    tree.write(path, xml_declaration=True, encoding="unicode")


# ---------------------------------------------------------------------------
# PVD (time series) writing
# ---------------------------------------------------------------------------


def _write_pvd(
    path: Path,
    entries: list[tuple[float, str]],
) -> None:
    """Write a ParaView .pvd time series file."""
    root = ET.Element("VTKFile", type="Collection", version="0.1")
    collection = ET.SubElement(root, "Collection")

    for t, filename in entries:
        ET.SubElement(
            collection, "DataSet",
            timestep=f"{t:g}",
            group="",
            part="0",
            file=filename,
        )

    tree = ET.ElementTree(root)
    _indent_xml(root)
    tree.write(path, xml_declaration=True, encoding="unicode")


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------


def _write_data_array_ascii(
    f, name: str, data: np.ndarray, n_components: int = 1,
) -> None:
    """Write a VTK DataArray element (Float64, ASCII)."""
    f.write(
        f'<DataArray type="Float64" Name="{name}" '
        f'NumberOfComponents="{n_components}" format="ascii">\n'
    )
    for val in data:
        f.write(f"{val:18.10E}\n")
    f.write("</DataArray>\n")


def _write_int_data_array_ascii(f, name: str, data: np.ndarray) -> None:
    """Write a VTK integer DataArray element (ASCII)."""
    dtype_name = "UInt8" if data.dtype == np.uint8 else "Int64"
    f.write(
        f'<DataArray type="{dtype_name}" Name="{name}" format="ascii">\n'
    )
    for val in data:
        f.write(f"{int(val)}\n")
    f.write("</DataArray>\n")


def _write_data_array_binary(
    f, name: str, data: np.ndarray, n_components: int = 1,
) -> None:
    """Write a VTK DataArray element (Float64, binary appended)."""
    import base64
    raw = data.astype(np.float64).tobytes()
    encoded = base64.b64encode(raw).decode("ascii")
    f.write(
        f'<DataArray type="Float64" Name="{name}" '
        f'NumberOfComponents="{n_components}" format="appended" '
        f'offset="0">\n'
    )
    f.write(f"  {encoded}\n")
    f.write("</DataArray>\n")


def _write_int_data_array_binary(f, name: str, data: np.ndarray) -> None:
    """Write a VTK integer DataArray element (binary appended)."""
    import base64
    dtype_name = "UInt8" if data.dtype == np.uint8 else "Int64"
    raw = data.tobytes()
    encoded = base64.b64encode(raw).decode("ascii")
    f.write(
        f'<DataArray type="{dtype_name}" Name="{name}" format="appended" '
        f'offset="0">\n'
    )
    f.write(f"  {encoded}\n")
    f.write("</DataArray>\n")


def _indent_xml(elem, level=0):
    """Add indentation to XML for readability."""
    indent = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_time(t: float) -> str:
    """Format a time value for use in file names."""
    if t == int(t):
        return str(int(t))
    return f"{t:g}"
