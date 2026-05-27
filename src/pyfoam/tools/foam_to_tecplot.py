"""
foamToTecplot — export an OpenFOAM case to Tecplot ASCII format.

Mirrors OpenFOAM's ``foamToTecplot`` utility.  Writes Tecplot ``.dat``
files that can be loaded by Tecplot 360 or ParaView (via the Tecplot
reader).

Tecplot ASCII format (``.dat``) overview:

- Title line
- ``VARIABLES = "X", "Y", "Z", "U", "V", "W", "P"`` header
- ``ZONE T="..." N=<n_points> E=<n_cells> ET=BRICK F=FEPOINT`` block
- Node data (coordinates + field values) one line per node
- Element connectivity (1-based indices)

Supported element types: BRICK (hex), TETRADE (tet), BRICK mapped from
wedge/pyramid via degenerate-node collapsing.

Usage::

    from pyfoam.tools.foam_to_tecplot import foam_to_tecplot

    foam_to_tecplot(case_path, mesh=mesh, fields=fields, time_range=[0, 1])
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["foam_to_tecplot"]

# Tecplot element types
_TEC_BRICK = "BRICK"       # 8-node hexahedron
_TEC_TETRA = "TETRADE"     # 4-node tetrahedron (Tecplot calls it TETRADE)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def foam_to_tecplot(
    case_path: Union[str, Path],
    time_range: Optional[Sequence[float]] = None,
    mesh: Optional["FvMesh"] = None,
    fields: Optional[Dict[str, np.ndarray]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Export an OpenFOAM case to Tecplot ASCII (.dat) format.

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
        Pre-loaded mesh.  When given, geometry and connectivity are
        extracted from this object rather than read from disk.
    fields : dict, optional
        ``{field_name: numpy_array}`` of per-cell or per-node field
        values to export.  Scalar arrays have shape ``(n,)``, vector
        arrays ``(n, 3)``.
    output_dir : str or Path, optional
        Directory for Tecplot output.  Defaults to ``<case_path>/Tecplot``.

    Returns
    -------
    Path
        Path to the output directory containing ``.dat`` files.

    Raises
    ------
    FileNotFoundError
        If *case_path* does not exist.
    ValueError
        If no mesh is available (neither provided nor on-disk).
    """
    case_dir = Path(case_path).resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    # Determine output directory
    if output_dir is None:
        tec_dir = case_dir / "Tecplot"
    else:
        tec_dir = Path(output_dir)
    os.makedirs(tec_dir, exist_ok=True)

    # Determine time values
    if time_range is not None:
        times = sorted(float(t) for t in time_range)
    else:
        from pyfoam.tools.foam_list_times import foam_list_times

        times = foam_list_times(case_dir)
        if not times and mesh is not None:
            times = [0.0]

    if mesh is None:
        raise ValueError(
            "No mesh provided.  Pass a mesh object directly."
        )

    # Pre-compute connectivity and cell types
    cell_verts, tec_types, connectivity = _compute_connectivity(mesh)

    # Determine variable names
    var_names: list[str] = _build_variable_names(fields)

    # Write one .dat file per time step
    for t in times:
        t_name = _format_time(t)
        dat_path = tec_dir / f"{t_name}.dat"
        _write_tecplot(
            dat_path, mesh, cell_verts, tec_types, connectivity,
            fields, var_names, t_name,
        )

    return tec_dir


# ---------------------------------------------------------------------------
# Connectivity computation
# ---------------------------------------------------------------------------


def _compute_connectivity(mesh: "FvMesh"):
    """Build cell-to-unique-vertices mapping, Tecplot types, and
    connectivity arrays.

    For Tecplot, BRICK cells need exactly 8 nodes.  Tetrahedra (4 nodes)
    are written as TETRADE.  Wedges (6 nodes) and pyramids (5 nodes)
    are converted to degenerate BRICKs by repeating nodes.

    Returns
    -------
    cell_verts : list[list[int]]
        Point indices for each cell.
    tec_types : list[str]
        Tecplot element type string for each cell.
    connectivity : list[list[int]]
        Node indices for each cell in Tecplot order (always 8 for BRICK,
        4 for TETRADE).
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

    cell_verts = [sorted(verts) for verts in cell_to_verts]

    tec_types = []
    connectivity = []
    for verts in cell_verts:
        nn = len(verts)
        if nn == 4:
            tec_types.append(_EC_TETRA)
            connectivity.append(verts)
        elif nn == 8:
            tec_types.append(_EC_BRICK)
            connectivity.append(verts)
        elif nn == 5:
            # Pyramid → degenerate BRICK (repeat last node 3 times)
            tec_types.append(_EC_BRICK)
            connectivity.append(verts + [verts[-1]] * 3)
        elif nn == 6:
            # Wedge/prism → degenerate BRICK (repeat last 2 nodes)
            tec_types.append(_EC_BRICK)
            connectivity.append(verts + [verts[-2], verts[-1]])
        else:
            # General polyhedron: pad or truncate to 8 nodes
            tec_types.append(_EC_BRICK)
            if nn < 8:
                padded = verts + [verts[-1]] * (8 - nn)
            else:
                padded = verts[:8]
            connectivity.append(padded)

    return cell_verts, tec_types, connectivity


# Constants for internal use
_EC_BRICK = "BRICK"
_EC_TETRA = "TETRADE"


# ---------------------------------------------------------------------------
# Variable name building
# ---------------------------------------------------------------------------


def _build_variable_names(fields: Optional[Dict[str, np.ndarray]]) -> List[str]:
    """Build the list of variable names for the Tecplot header.

    Always includes ``X``, ``Y``, ``Z``.  For each field:
    - Scalar: one variable name
    - Vector ``(n, 3)``: three names ``<name>X``, ``<name>Y``, ``<name>Z``
    """
    names = ["X", "Y", "Z"]
    if fields:
        for field_name, data in fields.items():
            if data.ndim == 2 and data.shape[1] == 3:
                names.extend([f"{field_name}X", f"{field_name}Y", f"{field_name}Z"])
            else:
                names.append(field_name)
    return names


# ---------------------------------------------------------------------------
# Tecplot file writing
# ---------------------------------------------------------------------------


def _write_tecplot(
    path: Path,
    mesh: "FvMesh",
    cell_verts: list[list[int]],
    tec_types: list[str],
    connectivity: list[list[int]],
    fields: Optional[Dict[str, np.ndarray]],
    var_names: List[str],
    t_name: str,
) -> None:
    """Write a complete Tecplot .dat file."""
    pts = mesh.points.detach().cpu().numpy()
    n_points = pts.shape[0]
    n_cells = len(cell_verts)

    # Determine the dominant element type for the zone
    # If all cells are TETRADE, use TETRADE; otherwise BRICK
    has_mixed = len(set(tec_types)) > 1
    zone_et = "BRICK" if any(t == "BRICK" for t in tec_types) else "TETRADE"

    with open(path, "w", encoding="utf-8") as f:
        # Title
        f.write(f'Title = "OpenFOAM case export t={t_name}"\n')

        # Variables
        var_str = ", ".join(f'"{v}"' for v in var_names)
        f.write(f"VARIABLES = {var_str}\n")

        # Zone header
        f.write(
            f'ZONE T="t={t_name}" '
            f"N={n_points} E={n_cells} "
            f"ET={zone_et} F=FEPOINT\n"
        )

        # Node data: X Y Z + field values
        n_fields = len(fields) if fields else 0
        for i in range(n_points):
            parts = [f"{pts[i, 0]:18.10E}", f"{pts[i, 1]:18.10E}", f"{pts[i, 2]:18.10E}"]
            if fields:
                for name, data in fields.items():
                    is_nodal = data.shape[0] == n_points
                    if not is_nodal:
                        continue  # skip cell-based fields in node section
                    if data.ndim == 2 and data.shape[1] == 3:
                        parts.extend([
                            f"{data[i, 0]:18.10E}",
                            f"{data[i, 1]:18.10E}",
                            f"{data[i, 2]:18.10E}",
                        ])
                    else:
                        parts.append(f"{data[i]:18.10E}")
            f.write(" ".join(parts) + "\n")

        # Element connectivity (1-based indices)
        for conn in connectivity:
            line = " ".join(str(v + 1) for v in conn)
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_time(t: float) -> str:
    """Format a time value for use in file names."""
    if t == int(t):
        return str(int(t))
    return f"{t:g}"
