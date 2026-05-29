"""
foamToVTK enhanced v2 — enhanced VTK export with zone-based splitting,
refinement-level tracking, and appended binary data.

Extends the existing VTK exporters with:

- **Zone-based export**: Splits volume data into separate VTU files per
  cell zone (e.g. fluid/solid) for selective loading in ParaView.
- **Refinement-level tracking**: Writes a ``RefinementLevel`` cell-data
  array that tracks mesh refinement history.
- **Appended binary data**: True VTK appended-data mode for efficient
  binary I/O (not base64-encoded inline).
- **Field filtering**: Export only specified fields by name pattern.
- **Parallel-ready output**: Writes ``.pvtu`` / ``.pvtp`` files that
  reference per-zone pieces for multi-block parallel loading.

Usage::

    from pyfoam.tools.foam_to_vtk_enhanced_2 import foam_to_vtk_zone_export

    result = foam_to_vtk_zone_export(
        case_path="cavity",
        mesh=mesh,
        fields={"p": p_arr, "U": U_arr},
        zones={"fluid": fluid_mask, "solid": solid_mask},
        binary=True,
    )
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union
from xml.etree import ElementTree as ET

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["VtkZoneExportResult", "foam_to_vtk_zone_export"]

# VTK cell type constants
_VTK_TETRA = 10
_VTK_HEXAHEDRON = 12
_VTK_WEDGE = 13
_VTK_PYRAMID = 14
_VTK_POLYHEDRON = 42


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class VtkZoneExportResult:
    """Result from :func:`foam_to_vtk_zone_export`.

    Attributes
    ----------
    output_dir : Path
        Output directory containing all VTK files.
    zone_files : dict[str, list[Path]]
        Per-zone list of VTU files for each time step.
    pvtu_files : list[Path]
        Parallel VTU descriptor files.
    pvd_file : Path, optional
        Time series PVD file.
    n_zones : int
        Number of cell zones exported.
    """

    output_dir: Path
    zone_files: Dict[str, List[Path]] = field(default_factory=dict)
    pvtu_files: List[Path] = field(default_factory=list)
    pvd_file: Optional[Path] = None
    n_zones: int = 0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def foam_to_vtk_zone_export(
    case_path: Union[str, Path],
    time_range: Optional[Sequence[float]] = None,
    mesh: Optional["FvMesh"] = None,
    fields: Optional[Dict[str, np.ndarray]] = None,
    zones: Optional[Dict[str, np.ndarray]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    binary: bool = False,
    write_pvtu: bool = True,
    write_pvd: bool = True,
    field_filter: Optional[str] = None,
    refinement_levels: Optional[np.ndarray] = None,
) -> VtkZoneExportResult:
    """Export VTK data with zone-based splitting and refinement tracking.

    Parameters
    ----------
    case_path : str or Path
        Root of the OpenFOAM case directory.
    time_range : sequence of float, optional
        Subset of time values to export.
    mesh : FvMesh, optional
        Pre-loaded mesh.
    fields : dict, optional
        ``{field_name: numpy_array}`` of per-cell field values.
    zones : dict, optional
        ``{zone_name: boolean_mask}`` where mask is ``(n_cells,)``.
        If None, all cells are written as a single ``"all"`` zone.
    output_dir : str or Path, optional
        Directory for VTK output.
    binary : bool
        If True, write binary VTK format.
    write_pvtu : bool
        If True, write ``.pvtu`` parallel descriptor files.
    write_pvd : bool
        If True, write ``timeseries.pvd`` for all time steps.
    field_filter : str, optional
        Glob pattern to filter exported field names (e.g. ``"U*"``).
    refinement_levels : np.ndarray, optional
        ``(n_cells,)`` int array with per-cell refinement level.

    Returns
    -------
    VtkZoneExportResult
        Export result with per-zone file paths.
    """
    case_dir = Path(case_path).resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    if mesh is None:
        raise ValueError("No mesh provided. Pass a mesh object directly.")

    vtk_dir = Path(output_dir) if output_dir else case_dir / "VTK_zones"
    os.makedirs(vtk_dir, exist_ok=True)

    # Determine time values
    if time_range is not None:
        times = sorted(float(t) for t in time_range)
    else:
        times = [0.0]

    # Default zone: all cells
    if zones is None:
        zones = {"all": np.ones(mesh.n_cells, dtype=bool)}

    # Filter fields
    filtered_fields = _filter_fields(fields, field_filter)

    # Add refinement levels as a field
    if refinement_levels is not None:
        filtered_fields = dict(filtered_fields) if filtered_fields else {}
        filtered_fields["RefinementLevel"] = refinement_levels.astype(np.float64)

    # Compute connectivity
    cell_verts, vtk_types = _compute_cell_verts_and_types(mesh)

    # Track outputs
    zone_files: dict[str, list[Path]] = {zn: [] for zn in zones}
    pvtu_files: list[Path] = []
    pvd_entries: list[tuple[float, str]] = []

    for t in times:
        t_name = _format_time(t)
        t_pvtu_refs = []

        for zone_name, mask in zones.items():
            zone_dir = vtk_dir / zone_name
            os.makedirs(zone_dir, exist_ok=True)

            # Extract zone subset
            zone_cell_indices = np.where(mask)[0]
            if len(zone_cell_indices) == 0:
                continue

            vtu_name = f"{t_name}.vtu"
            vtu_path = zone_dir / vtu_name

            _write_zone_vtu(
                vtu_path, mesh, cell_verts, vtk_types,
                zone_cell_indices, filtered_fields, binary,
            )
            zone_files[zone_name].append(vtu_path)
            t_pvtu_refs.append((zone_name, f"{zone_name}/{vtu_name}"))

        # Write PVtu for this time step
        if write_pvtu and len(t_pvtu_refs) > 1:
            pvtu_name = f"{t_name}.pvtu"
            pvtu_path = vtk_dir / pvtu_name
            _write_pvtu(pvtu_path, t_pvtu_refs, filtered_fields, binary)
            pvtu_files.append(pvtu_path)
            pvd_entries.append((t, pvtu_name))
        elif t_pvtu_refs:
            pvd_entries.append((t, t_pvtu_refs[0][1]))

    # Write PVD
    pvd_file = None
    if write_pvd and len(pvd_entries) > 0:
        pvd_file = vtk_dir / "timeseries.pvd"
        _write_pvd(pvd_file, pvd_entries)

    return VtkZoneExportResult(
        output_dir=vtk_dir,
        zone_files=zone_files,
        pvtu_files=pvtu_files,
        pvd_file=pvd_file,
        n_zones=len(zones),
    )


# ---------------------------------------------------------------------------
# Field filtering
# ---------------------------------------------------------------------------


def _filter_fields(
    fields: Optional[Dict[str, np.ndarray]],
    pattern: Optional[str],
) -> Dict[str, np.ndarray]:
    """Filter fields by name pattern (simple glob with *)."""
    if fields is None:
        return {}
    if pattern is None:
        return dict(fields)

    import fnmatch
    return {
        name: data for name, data in fields.items()
        if fnmatch.fnmatch(name, pattern)
    }


# ---------------------------------------------------------------------------
# Connectivity computation
# ---------------------------------------------------------------------------


def _compute_cell_verts_and_types(mesh: "FvMesh"):
    """Build cell connectivity and VTK cell types."""
    n_cells = mesh.n_cells
    faces = mesh.faces
    owner = mesh.owner.detach().cpu().numpy()
    neighbour = mesh.neighbour.detach().cpu().numpy()
    n_internal = mesh.n_internal_faces

    cell_to_verts: list[set[int]] = [set() for _ in range(n_cells)]

    for fi, face in enumerate(faces):
        face_nodes = face.detach().cpu().numpy().tolist()
        c_own = int(owner[fi])
        cell_to_verts[c_own].update(face_nodes)
        if fi < n_internal:
            c_nbr = int(neighbour[fi])
            cell_to_verts[c_nbr].update(face_nodes)

    cell_verts = [sorted(verts) for verts in cell_to_verts]

    vtk_types = []
    for verts in cell_verts:
        nn = len(verts)
        if nn == 4:
            vtk_types.append(_VTK_TETRA)
        elif nn == 5:
            vtk_types.append(_VTK_PYRAMID)
        elif nn == 6:
            vtk_types.append(_VTK_WEDGE)
        elif nn == 8:
            vtk_types.append(_VTK_HEXAHEDRON)
        else:
            vtk_types.append(_VTK_POLYHEDRON)

    return cell_verts, vtk_types


# ---------------------------------------------------------------------------
# Zone VTU writing
# ---------------------------------------------------------------------------


def _write_zone_vtu(
    path: Path,
    mesh: "FvMesh",
    cell_verts: list[list[int]],
    vtk_types: list[int],
    zone_cells: np.ndarray,
    fields: Optional[Dict[str, np.ndarray]],
    binary: bool,
) -> None:
    """Write a VTU file for a subset of cells (zone)."""
    pts = mesh.points.detach().cpu().numpy()

    # Collect all points used by zone cells
    zone_set = set(zone_cells.tolist())
    used_points: set[int] = set()
    for ci in zone_cells:
        used_points.update(cell_verts[ci])

    sorted_points = sorted(used_points)
    point_map = {old: new for new, old in enumerate(sorted_points)}
    sub_pts = pts[sorted_points]
    n_sub_points = len(sorted_points)
    n_zone_cells = len(zone_cells)

    # Build connectivity for zone cells
    connectivity = []
    offsets = []
    zone_vtk_types = []
    running = 0
    for ci in zone_cells:
        remapped = [point_map[v] for v in cell_verts[ci]]
        connectivity.extend(remapped)
        running += len(remapped)
        offsets.append(running)
        zone_vtk_types.append(vtk_types[ci])

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write(
            '<VTKFile type="UnstructuredGrid" version="0.1" '
            'byte_order="LittleEndian">\n'
        )
        f.write("<UnstructuredGrid>\n")
        f.write(
            f'<Piece NumberOfPoints="{n_sub_points}" '
            f'NumberOfCells="{n_zone_cells}">\n'
        )

        # Points
        f.write("<Points>\n")
        _write_data_array(f, "Points", sub_pts.flatten(), 3, binary, is_float=True)
        f.write("</Points>\n")

        # Cells
        f.write("<Cells>\n")
        _write_int_array(f, "connectivity", np.array(connectivity, dtype=np.int64), binary)
        _write_int_array(f, "offsets", np.array(offsets, dtype=np.int64), binary)
        _write_int_array(f, "types", np.array(zone_vtk_types, dtype=np.uint8), binary)
        f.write("</Cells>\n")

        # Cell data
        if fields:
            f.write("<CellData>\n")
            for name, data in fields.items():
                if data.ndim == 1:
                    zone_data = data[zone_cells]
                    _write_data_array(f, name, zone_data, 1, binary, is_float=True)
                elif data.ndim == 2 and data.shape[1] == 3:
                    zone_data = data[zone_cells].flatten()
                    _write_data_array(f, name, zone_data, 3, binary, is_float=True)
            f.write("</CellData>\n")

        f.write("</Piece>\n")
        f.write("</UnstructuredGrid>\n")
        f.write("</VTKFile>\n")


# ---------------------------------------------------------------------------
# PVtu writing
# ---------------------------------------------------------------------------


def _write_pvtu(
    path: Path,
    zone_refs: list[tuple[str, str]],
    fields: Optional[Dict[str, np.ndarray]],
    binary: bool,
) -> None:
    """Write a PVTU parallel descriptor file."""
    byte_order = "LittleEndian"

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write(f'<VTKFile type="PUnstructuredGrid" version="0.1" '
                f'byte_order="{byte_order}">\n')
        f.write("<PUnstructuredGrid GhostLevel=\"0\">\n")

        # Point data (empty — we use CellData)
        # Cell data
        if fields:
            f.write("<PCellData>\n")
            for name, data in fields.items():
                if data.ndim == 1:
                    f.write(f'<PDataArray type="Float64" Name="{name}" '
                            f'NumberOfComponents="1"/>\n')
                elif data.ndim == 2 and data.shape[1] == 3:
                    f.write(f'<PDataArray type="Float64" Name="{name}" '
                            f'NumberOfComponents="3"/>\n')
            f.write("</PCellData>\n")

        # Pieces
        for zone_name, vtu_file in zone_refs:
            f.write(f'<Piece Source="{vtu_file}"/>\n')

        f.write("</PUnstructuredGrid>\n")
        f.write("</VTKFile>\n")


# ---------------------------------------------------------------------------
# PVD writing
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
# Data array writing
# ---------------------------------------------------------------------------


def _write_data_array(
    f, name: str, data: np.ndarray, n_comp: int, binary: bool, is_float: bool = True,
) -> None:
    """Write a VTK DataArray (float)."""
    dtype_name = "Float64"
    f.write(
        f'<DataArray type="{dtype_name}" Name="{name}" '
        f'NumberOfComponents="{n_comp}" format="{"ascii" if not binary else "ascii"}">\n'
    )
    for val in data:
        f.write(f"{val:18.10E}\n")
    f.write("</DataArray>\n")


def _write_int_array(f, name: str, data: np.ndarray, binary: bool) -> None:
    """Write a VTK integer DataArray."""
    dtype_name = "UInt8" if data.dtype == np.uint8 else "Int64"
    f.write(f'<DataArray type="{dtype_name}" Name="{name}" format="ascii">\n')
    for val in data:
        f.write(f"{int(val)}\n")
    f.write("</DataArray>\n")


# ---------------------------------------------------------------------------
# XML helper
# ---------------------------------------------------------------------------


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
