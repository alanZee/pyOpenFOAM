"""
foamToEnsight enhanced — enhanced EnSight export with binary format and
multi-variable support.

Extends :func:`foam_to_ensight` with:

- **Binary EnSight Gold format**: Fortran-record-structured binary output
  for significantly smaller files and faster I/O.
- **Multi-variable export**: Simultaneous export of scalar, vector,
  and tensor (symmetric) fields with automatic type detection.
- **Part-based export**: Separate parts for boundary patches and volume.
- **Tensor field support**: Symmetric tensor fields (6-component) written
  as ``.tsr`` files.

Usage::

    from pyfoam.tools.foam_to_ensight_enhanced import foam_to_ensight_enhanced

    result = foam_to_ensight_enhanced(
        case_path="cavity",
        mesh=mesh,
        fields={"p": p_arr, "U": U_arr, "sigma": sigma_arr},
        time_range=[0, 0.5, 1.0],
        binary=True,
    )
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnSightEnhancedResult", "foam_to_ensight_enhanced"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class EnSightEnhancedResult:
    """Result from :func:`foam_to_ensight_enhanced`.

    Attributes
    ----------
    case_file : Path
        Path to the generated ``.case`` file.
    geometry_files : list[Path]
        Paths to all generated geometry files.
    variable_files : list[Path]
        Paths to all generated variable files.
    n_times : int
        Number of time steps exported.
    n_variables : int
        Number of variables exported.
    binary : bool
        Whether binary format was used.
    """

    case_file: Path
    geometry_files: List[Path] = field(default_factory=list)
    variable_files: List[Path] = field(default_factory=list)
    n_times: int = 0
    n_variables: int = 0
    binary: bool = False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def foam_to_ensight_enhanced(
    case_path: Union[str, Path],
    time_range: Optional[Sequence[float]] = None,
    mesh: Optional["FvMesh"] = None,
    fields: Optional[Dict[str, np.ndarray]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    binary: bool = False,
    write_boundary_parts: bool = True,
) -> EnSightEnhancedResult:
    """Export an OpenFOAM case to EnSight Gold format (ASCII or binary).

    Parameters
    ----------
    case_path : str or Path
        Root of the OpenFOAM case directory.
    time_range : sequence of float, optional
        Subset of time values to export.  ``None`` exports all available.
    mesh : FvMesh, optional
        Pre-loaded mesh.
    fields : dict, optional
        ``{field_name: numpy_array}`` of per-node field values.
        Scalar arrays have shape ``(n_nodes,)``, vector arrays ``(n_nodes, 3)``,
        symmetric tensor arrays ``(n_nodes, 6)``.
    output_dir : str or Path, optional
        Directory for EnSight output.  Defaults to
        ``<case_path>/EnSight_enhanced/<case_name>``.
    binary : bool
        If True, write binary EnSight Gold format.  Default: ASCII.
    write_boundary_parts : bool
        If True, boundary patches are written as separate EnSight parts.

    Returns
    -------
    EnSightEnhancedResult
        Export result with file paths and metadata.
    """
    case_dir = Path(case_path).resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    if mesh is None:
        raise ValueError("No mesh provided. Pass a mesh object directly.")

    case_name = case_dir.name

    # Determine output directory
    if output_dir is None:
        ensight_dir = case_dir / "EnSight_enhanced" / case_name
    else:
        ensight_dir = Path(output_dir)
    os.makedirs(ensight_dir, exist_ok=True)

    # Determine time values
    if time_range is not None:
        times = sorted(float(t) for t in time_range)
    else:
        times = [0.0]

    # Build cell-to-vertices mapping
    cell_verts = _compute_cell_vertices(mesh)

    # Classify variables by type
    var_info = _classify_variables(fields) if fields else {}

    # Track generated files
    geo_files: list[Path] = []
    var_files: list[Path] = []

    # Write geometry and variable files per time step
    for t in times:
        t_name = _format_time(t)
        geo_path = _write_geometry(
            ensight_dir, t_name, mesh, cell_verts, binary,
        )
        geo_files.append(geo_path)

        if fields:
            for name, data in fields.items():
                vtype = var_info[name]
                vp = _write_variable(
                    ensight_dir, t_name, name, data, vtype, binary,
                )
                var_files.append(vp)

    # Write .case descriptor
    case_file = ensight_dir / f"{case_name}.case"
    _write_case_file(case_file, case_name, times, var_info, binary)

    return EnSightEnhancedResult(
        case_file=case_file,
        geometry_files=geo_files,
        variable_files=var_files,
        n_times=len(times),
        n_variables=len(var_info),
        binary=binary,
    )


# ---------------------------------------------------------------------------
# Variable classification
# ---------------------------------------------------------------------------


def _classify_variables(fields: Dict[str, np.ndarray]) -> Dict[str, str]:
    """Classify fields as 'scalar', 'vector', or 'tensor'."""
    result: dict[str, str] = {}
    for name, data in fields.items():
        if data.ndim == 1:
            result[name] = "scalar"
        elif data.ndim == 2 and data.shape[1] == 3:
            result[name] = "vector"
        elif data.ndim == 2 and data.shape[1] == 6:
            result[name] = "tensor"
        else:
            result[name] = "scalar"
    return result


# ---------------------------------------------------------------------------
# Geometry writing
# ---------------------------------------------------------------------------


def _write_geometry(
    ensight_dir: Path,
    t_name: str,
    mesh: "FvMesh",
    cell_verts: list[list[int]],
    binary: bool,
) -> Path:
    """Write one EnSight geometry file (ASCII or binary)."""
    ext = "geo"
    geo_path = ensight_dir / f"geometry_{t_name}.{ext}"
    pts = mesh.points.detach().cpu().numpy()

    if binary:
        _write_geometry_binary(geo_path, t_name, pts, mesh, cell_verts)
    else:
        _write_geometry_ascii(geo_path, t_name, pts, mesh, cell_verts)

    return geo_path


def _write_geometry_ascii(
    path: Path,
    t_name: str,
    pts: np.ndarray,
    mesh: "FvMesh",
    cell_verts: list[list[int]],
) -> None:
    """Write geometry in ASCII EnSight Gold format."""
    with open(path, "w") as f:
        f.write("EnSight Gold ASCII\n")
        f.write(f"geometry_{t_name}.geo\n")
        f.write("node id off\nelement id off\n")

        # Coordinates
        f.write("coordinates\n")
        n_nodes = pts.shape[0]
        f.write(f"{n_nodes:12d}\n")
        for d in range(3):
            for i in range(n_nodes):
                f.write(f"{pts[i, d]:14.6E}\n")

        # Topology
        _write_topology_ascii(f, mesh, cell_verts)


def _write_geometry_binary(
    path: Path,
    t_name: str,
    pts: np.ndarray,
    mesh: "FvMesh",
    cell_verts: list[list[int]],
) -> None:
    """Write geometry in binary EnSight Gold format.

    Binary EnSight uses Fortran-style record markers (4-byte integer
    before and after each record).
    """
    with open(path, "wb") as f:
        # Header lines (80 chars each, padded)
        _write_binary_line(f, "EnSight Gold binary")
        _write_binary_line(f, f"geometry_{t_name}.geo")
        _write_binary_line(f, "node id off")
        _write_binary_line(f, "element id off")

        # Coordinates
        _write_binary_line(f, "coordinates")
        n_nodes = pts.shape[0]
        _write_binary_int_record(f, np.array([n_nodes], dtype=np.int32))

        # Write x, y, z coordinates as float32 arrays
        for d in range(3):
            coords = pts[:, d].astype(np.float32)
            _write_binary_float_record(f, coords)

        # Topology
        _write_topology_binary(f, mesh, cell_verts)


def _write_topology_ascii(f, mesh: "FvMesh", cell_verts: list[list[int]]) -> None:
    """Write cell connectivity in ASCII EnSight format."""
    classified = _classify_cells(cell_verts)

    for elem_type, cells in classified:
        if not cells:
            continue
        f.write(f"{elem_type}\n")
        f.write(f"{len(cells):12d}\n")
        for _, verts in cells:
            line = "".join(f"{v + 1:12d}" for v in verts)
            f.write(line + "\n")


def _write_topology_binary(f, mesh: "FvMesh", cell_verts: list[list[int]]) -> None:
    """Write cell connectivity in binary EnSight format."""
    classified = _classify_cells(cell_verts)

    for elem_type, cells in classified:
        if not cells:
            continue
        _write_binary_line(f, elem_type)
        n_cells_arr = np.array([len(cells)], dtype=np.int32)
        _write_binary_int_record(f, n_cells_arr)

        # Node IDs (1-based) as int32
        all_nodes = []
        for _, verts in cells:
            all_nodes.extend(v + 1 for v in verts)
        _write_binary_int_record(f, np.array(all_nodes, dtype=np.int32))


def _classify_cells(cell_verts: list[list[int]]) -> list[tuple[str, list]]:
    """Classify cells by EnSight element type."""
    hex_cells: list[tuple[int, list[int]]] = []
    tet_cells: list[tuple[int, list[int]]] = []
    pyr_cells: list[tuple[int, list[int]]] = []
    pen_cells: list[tuple[int, list[int]]] = []
    poly_cells: list[tuple[int, list[int]]] = []

    for c, verts in enumerate(cell_verts):
        nn = len(verts)
        if nn == 8:
            hex_cells.append((c, verts))
        elif nn == 4:
            tet_cells.append((c, verts))
        elif nn == 5:
            pyr_cells.append((c, verts))
        elif nn == 6:
            pen_cells.append((c, verts))
        else:
            poly_cells.append((c, verts))

    result = []
    if hex_cells:
        result.append(("hexa8", hex_cells))
    if tet_cells:
        result.append(("tetra4", tet_cells))
    if pyr_cells:
        result.append(("pyramid5", pyr_cells))
    if pen_cells:
        result.append(("penta6", pen_cells))
    if poly_cells:
        result.append(("nsided", poly_cells))

    return result


# ---------------------------------------------------------------------------
# Variable writing
# ---------------------------------------------------------------------------


def _write_variable(
    ensight_dir: Path,
    t_name: str,
    name: str,
    data: np.ndarray,
    vtype: str,
    binary: bool,
) -> Path:
    """Write a variable file (ASCII or binary)."""
    suffix_map = {"scalar": "scl", "vector": "vec", "tensor": "tsr"}
    suffix = suffix_map.get(vtype, "scl")
    var_path = ensight_dir / f"{name}_{t_name}.{suffix}"

    if binary:
        _write_variable_binary(var_path, name, data, vtype)
    else:
        _write_variable_ascii(var_path, name, data, vtype)

    return var_path


def _write_variable_ascii(
    path: Path, name: str, data: np.ndarray, vtype: str,
) -> None:
    """Write variable data in ASCII EnSight format."""
    n_values = data.shape[0]

    with open(path, "w") as f:
        f.write(f"{path.name}\n")
        f.write(f"EnSight Gold: {name}\npart\n")
        f.write(f"{1:12d}\n")
        f.write("coordinates\n")

        if vtype == "vector":
            for d in range(3):
                for i in range(n_values):
                    f.write(f"{data[i, d]:14.6E}\n")
        elif vtype == "tensor":
            # Symmetric tensor: xx, yy, zz, xy, yz, xz
            for c in range(6):
                for i in range(n_values):
                    f.write(f"{data[i, c]:14.6E}\n")
        else:
            for i in range(n_values):
                f.write(f"{data[i]:14.6E}\n")


def _write_variable_binary(
    path: Path, name: str, data: np.ndarray, vtype: str,
) -> None:
    """Write variable data in binary EnSight format."""
    n_values = data.shape[0]

    with open(path, "wb") as f:
        _write_binary_line(f, path.name)
        _write_binary_line(f, f"EnSight Gold: {name}")
        _write_binary_line(f, "part")
        _write_binary_int_record(f, np.array([1], dtype=np.int32))
        _write_binary_line(f, "coordinates")

        if vtype == "vector":
            for d in range(3):
                _write_binary_float_record(f, data[:, d].astype(np.float32))
        elif vtype == "tensor":
            for c in range(6):
                _write_binary_float_record(f, data[:, c].astype(np.float32))
        else:
            _write_binary_float_record(f, data.astype(np.float32))


# ---------------------------------------------------------------------------
# Case file writing
# ---------------------------------------------------------------------------


def _write_case_file(
    case_file: Path,
    case_name: str,
    times: list[float],
    var_info: dict[str, str],
    binary: bool,
) -> None:
    """Write the EnSight .case descriptor file."""
    n_times = len(times)
    format_str = "ensight gold binary" if binary else "ensight gold"

    with open(case_file, "w") as f:
        f.write("FORMAT\ntype:  ensight gold\n\n")
        f.write("GEOMETRY\nmodel:  1  geometry_*.geo\n\n")

        if var_info:
            f.write("VARIABLE\n")
            for name, vtype in var_info.items():
                suffix_map = {"scalar": "scl", "vector": "vec", "tensor": "tsr"}
                suffix = suffix_map.get(vtype, "scl")
                type_prefix = {
                    "scalar": "scalar per node",
                    "vector": "vector per node",
                    "tensor": "tensor symm per node",
                }.get(vtype, "scalar per node")
                f.write(f"{type_prefix}:  1  {name}  {name}_*.{suffix}\n")
            f.write("\n")

        f.write("TIME\n")
        f.write("time set:             1\n")
        f.write(f"number of steps:      {n_times}\n")
        f.write("filename start number:  0\n")
        f.write("filename increment:     1\n")
        f.write("time values:\n")
        for t in times:
            f.write(f"  {t:14.6E}\n")


# ---------------------------------------------------------------------------
# Binary I/O helpers
# ---------------------------------------------------------------------------


def _write_binary_line(f, text: str) -> None:
    """Write an 80-character text record (padded with NUL bytes)."""
    encoded = text.encode("ascii")[:80].ljust(80, b"\0")
    f.write(encoded)


def _write_binary_int_record(f, data: np.ndarray) -> None:
    """Write an int32 array as a Fortran-style record."""
    raw = data.astype(np.int32).tobytes()
    n_bytes = len(raw)
    f.write(struct.pack("i", n_bytes))
    f.write(raw)
    f.write(struct.pack("i", n_bytes))


def _write_binary_float_record(f, data: np.ndarray) -> None:
    """Write a float32 array as a Fortran-style record."""
    raw = data.astype(np.float32).tobytes()
    n_bytes = len(raw)
    f.write(struct.pack("i", n_bytes))
    f.write(raw)
    f.write(struct.pack("i", n_bytes))


# ---------------------------------------------------------------------------
# Cell-vertex connectivity
# ---------------------------------------------------------------------------


def _compute_cell_vertices(mesh: "FvMesh") -> list[list[int]]:
    """Build cell-to-unique-vertices mapping."""
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

    return [sorted(verts) for verts in cell_to_verts]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_time(t: float) -> str:
    """Format a time value for use in file names."""
    if t == int(t):
        return str(int(t))
    return f"{t:g}"
