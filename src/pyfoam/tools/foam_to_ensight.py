"""
foamToEnsight — export an OpenFOAM case to EnSight Gold format.

Mirrors OpenFOAM's ``foamToEnsight`` utility.  Writes:

- A ``.case`` descriptor file referencing geometry and variable files.
- A ``.geo`` geometry file per time step (coordinates + cell connectivity).
- ``.scl`` files for scalar fields and ``.vec`` files for vector fields.

The output is ASCII EnSight Gold format.  Both standard element types
(HEXA8, TETRA4, PENTA6, PYRAMID5) and the general polyhedra format
(``nsided``) are supported.

Usage::

    from pyfoam.tools.foam_to_ensight import foam_to_ensight

    foam_to_ensight(case_path, time_range=[0, 0.5, 1.0])
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["foam_to_ensight"]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def foam_to_ensight(
    case_path: Union[str, Path],
    time_range: Optional[Sequence[float]] = None,
    mesh: Optional["FvMesh"] = None,
    fields: Optional[dict[str, np.ndarray]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Export an OpenFOAM case to EnSight Gold ASCII format.

    When *mesh* and *fields* are provided directly (e.g. from a running
    simulation), they are used directly.  Otherwise the function scans
    *case_path* for on-disk time directories and field files.

    Parameters
    ----------
    case_path : str or Path
        Root of the OpenFOAM case directory.
    time_range : sequence of float, optional
        Subset of time values to export.  ``None`` exports all available
        times (requires on-disk data).
    mesh : FvMesh, optional
        Pre-loaded mesh.  When given, the geometry is written from this
        object rather than read from disk.
    fields : dict, optional
        ``{field_name: numpy_array}`` of per-node field values to export
        alongside the geometry.  Scalar arrays have shape ``(n_nodes,)``,
        vector arrays ``(n_nodes, 3)``.
    output_dir : str or Path, optional
        Directory for EnSight output.  Defaults to
        ``<case_path>/EnSight/<case_name>``.

    Returns
    -------
    Path
        Path to the generated ``.case`` file.

    Raises
    ------
    FileNotFoundError
        If *case_path* does not exist.
    ValueError
        If neither *mesh* nor on-disk mesh data is available.
    """
    case_dir = Path(case_path).resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    case_name = case_dir.name

    # Determine output directory
    if output_dir is None:
        ensight_dir = case_dir / "EnSight" / case_name
    else:
        ensight_dir = Path(output_dir)
    os.makedirs(ensight_dir, exist_ok=True)

    # Determine time values
    if time_range is not None:
        times = sorted(float(t) for t in time_range)
    else:
        from pyfoam.tools.foam_list_times import foam_list_times
        times = foam_list_times(case_dir)
        if not times and mesh is not None:
            times = [0.0]

    # Build cell-to-vertices mapping (shared across time steps)
    cell_verts = None
    if mesh is not None:
        cell_verts = _compute_cell_vertices(mesh)
    elif times:
        # TODO: load mesh from disk if not provided
        raise ValueError(
            "No mesh provided and on-disk mesh loading is not yet supported. "
            "Pass a mesh object directly."
        )

    # Determine variable names from fields dict
    var_names: list[str] = list(fields.keys()) if fields else []

    # Write geometry and variable files per time step
    for t in times:
        t_name = _format_time(t)
        _write_geometry(ensight_dir, t_name, mesh, cell_verts)
        if fields:
            for name, data in fields.items():
                _write_variable(ensight_dir, t_name, name, data)

    # Write .case descriptor
    case_file = ensight_dir / f"{case_name}.case"
    _write_case_file(case_file, case_name, times, var_names, ensight_dir)

    return case_file


# ---------------------------------------------------------------------------
# Geometry writing
# ---------------------------------------------------------------------------


def _write_geometry(
    ensight_dir: Path,
    t_name: str,
    mesh: "FvMesh",
    cell_verts: list[list[int]],
) -> None:
    """Write one EnSight geometry file."""
    geo_path = ensight_dir / f"geometry_{t_name}.geo"
    pts = mesh.points.detach().cpu().numpy()

    with open(geo_path, "w") as f:
        # Header
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
        _write_topology(f, mesh, cell_verts)


def _write_topology(
    f,
    mesh: "FvMesh",
    cell_verts: list[list[int]],
) -> None:
    """Write cell connectivity in EnSight format.

    Standard element types (HEXA8, TETRA4, PENTA6, PYRAMID5) are used
    when the node count matches.  Polyhedral cells (≠ 4/5/6/8 nodes)
    are written using the general ``nsided`` format.
    """
    n_cells = len(cell_verts)

    # Classify cells by node count
    hex_cells = []     # 8-node
    tet_cells = []     # 4-node
    pyr_cells = []     # 5-node
    pen_cells = []     # 6-node
    poly_cells = []    # other

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

    # Write standard element types
    if hex_cells:
        f.write("hexa8\n")
        f.write(f"{len(hex_cells):12d}\n")
        for _, verts in hex_cells:
            line = "".join(f"{v + 1:12d}" for v in verts)
            f.write(line + "\n")

    if tet_cells:
        f.write("tetra4\n")
        f.write(f"{len(tet_cells):12d}\n")
        for _, verts in tet_cells:
            line = "".join(f"{v + 1:12d}" for v in verts)
            f.write(line + "\n")

    if pyr_cells:
        f.write("pyramid5\n")
        f.write(f"{len(pyr_cells):12d}\n")
        for _, verts in pyr_cells:
            line = "".join(f"{v + 1:12d}" for v in verts)
            f.write(line + "\n")

    if pen_cells:
        f.write("penta6\n")
        f.write(f"{len(pen_cells):12d}\n")
        for _, verts in pen_cells:
            line = "".join(f"{v + 1:12d}" for v in verts)
            f.write(line + "\n")

    # General polyhedra (nsided format)
    if poly_cells:
        f.write("nsided\n")
        f.write(f"{len(poly_cells):12d}\n")

        # Nodes per element
        for _, verts in poly_cells:
            f.write(f"{len(verts):12d}\n")

        # All node ids concatenated
        for _, verts in poly_cells:
            line = "".join(f"{v + 1:12d}" for v in verts)
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Cell-vertex connectivity
# ---------------------------------------------------------------------------


def _compute_cell_vertices(mesh: "FvMesh") -> list[list[int]]:
    """Build cell-to-unique-vertices mapping.

    Iterates over all faces and collects the point indices for each
    owner and neighbour cell.  Duplicate vertices are removed while
    preserving insertion order.
    """
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

    # Convert sets to sorted lists for deterministic output
    return [sorted(verts) for verts in cell_to_verts]


# ---------------------------------------------------------------------------
# Variable writing
# ---------------------------------------------------------------------------


def _write_variable(
    ensight_dir: Path,
    t_name: str,
    name: str,
    data: np.ndarray,
) -> None:
    """Write a scalar or vector variable file."""
    is_vector = data.ndim == 2 and data.shape[1] == 3
    suffix = "vec" if is_vector else "scl"
    var_path = ensight_dir / f"{name}_{t_name}.{suffix}"

    n_values = data.shape[0]

    with open(var_path, "w") as f:
        f.write(f"{name}_{t_name}.{suffix}\n")
        f.write(f"EnSight Gold: {name}\npart\n")
        f.write(f"{1:12d}\n")
        f.write("coordinates\n")

        if is_vector:
            for d in range(3):
                for i in range(n_values):
                    f.write(f"{data[i, d]:14.6E}\n")
        else:
            for i in range(n_values):
                f.write(f"{data[i]:14.6E}\n")


# ---------------------------------------------------------------------------
# Case file writing
# ---------------------------------------------------------------------------


def _write_case_file(
    case_file: Path,
    case_name: str,
    times: list[float],
    var_names: list[str],
    ensight_dir: Path,
) -> None:
    """Write the EnSight .case descriptor file."""
    n_times = len(times)

    with open(case_file, "w") as f:
        f.write("FORMAT\ntype:  ensight gold\n\n")
        f.write(f"GEOMETRY\nmodel:  1  geometry_*.geo\n\n")

        if var_names:
            f.write("VARIABLE\n")
            for name in var_names:
                # Infer dimensionality from the first time file
                vec_path = ensight_dir / f"{name}_{_format_time(times[0])}.vec"
                if vec_path.exists():
                    f.write(f"vector per node:  1  {name}  {name}_*.vec\n")
                else:
                    f.write(f"scalar per node:  1  {name}  {name}_*.scl\n")
            f.write("\n")

        f.write("TIME\n")
        f.write(f"time set:             1\n")
        f.write(f"number of steps:      {n_times}\n")
        f.write(f"filename start number:  0\n")
        f.write(f"filename increment:     1\n")
        f.write("time values:\n")
        for t in times:
            f.write(f"  {t:14.6E}\n")


# ---------------------------------------------------------------------------
# On-disk field reading
# ---------------------------------------------------------------------------


def _read_on_disk_scalar(file_path: Path, n_nodes: int) -> np.ndarray:
    """Read a plain-text OpenFOAM scalar field file."""
    from pyfoam.io.dictionary import parse_dict_file

    d = parse_dict_file(file_path)
    n_internal = int(d["nInternalFaces"]) if "nInternalFaces" in d else 0
    # For boundary fields, try "boundaryField" sub-dict
    # Fall back to reading raw values
    content = file_path.read_text(encoding="utf-8", errors="replace")
    values = _extract_foam_values(content)
    if len(values) < n_nodes:
        values = np.concatenate([values, np.zeros(n_nodes - len(values))])
    return values[:n_nodes]


def _read_on_disk_vector(file_path: Path, n_nodes: int) -> np.ndarray:
    """Read a plain-text OpenFOAM vector field file."""
    content = file_path.read_text(encoding="utf-8", errors="replace")
    # Each vector is (x y z) — extract all floating-point triples
    import re
    raw = re.findall(r"\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)", content)
    vecs = np.array([[float(x), float(y), float(z)] for x, y, z in raw])
    if len(vecs) < n_nodes:
        pad = np.zeros((n_nodes - len(vecs), 3))
        vecs = np.concatenate([vecs, pad])
    return vecs[:n_nodes]


def _extract_foam_values(content: str) -> np.ndarray:
    """Extract numeric values from an OpenFOAM field file body."""
    import re
    # Remove FoamFile header block
    body = re.sub(r"FoamFile\s*\{[^}]*\}", "", content, flags=re.DOTALL)
    # Find all numeric tokens
    tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", body)
    return np.array([float(t) for t in tokens])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_time(t: float) -> str:
    """Format a time value for use in file names."""
    if t == int(t):
        return str(int(t))
    return f"{t:g}"
