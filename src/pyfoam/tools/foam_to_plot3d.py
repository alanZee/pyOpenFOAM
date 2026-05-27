"""
foamToPlot3D — export an OpenFOAM case to Plot3D format.

Mirrors the functionality of OpenFOAM's ``foamToPlot3D`` utility.
Writes Plot3D grid (``.xyz``) and solution (``.q``) files for
post-processing with tools such as FieldView, Tecplot, or PLOT3D-based
visualisation systems.

Plot3D format comes in two flavours:

- **Multi-block**: multiple grid blocks in a single file (each block
  preceded by its dimensions ``NI NJ NK``).
- **Formatted** (ASCII) or **unformatted** (binary, Fortran-style records).

This implementation writes **formatted multi-block** Plot3D with
iblanking disabled (solid geometry only).

Usage::

    from pyfoam.tools.foam_to_plot3d import foam_to_plot3d

    foam_to_plot3d(case_path, mesh=mesh, fields=fields)
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["foam_to_plot3d"]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def foam_to_plot3d(
    case_path: Union[str, Path],
    time_range: Optional[Sequence[float]] = None,
    mesh: Optional["FvMesh"] = None,
    fields: Optional[Dict[str, np.ndarray]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    binary: bool = False,
) -> Path:
    """Export an OpenFOAM case to Plot3D format.

    Writes ``.xyz`` (grid) and optionally ``.q`` (solution) files
    in multi-block format.

    Parameters
    ----------
    case_path : str or Path
        Root of the OpenFOAM case directory.
    time_range : sequence of float, optional
        Subset of time values to export.  ``None`` exports all available
        times (requires on-disk data).
    mesh : FvMesh, optional
        Pre-loaded mesh.
    fields : dict, optional
        ``{field_name: numpy_array}`` of per-cell field values to export.
    output_dir : str or Path, optional
        Directory for Plot3D output.  Defaults to ``<case_path>/plot3d``.
    binary : bool
        If True, write unformatted (binary) Plot3D files.

    Returns
    -------
    Path
        Path to the output directory.

    Raises
    ------
    FileNotFoundError
        If *case_path* does not exist.
    ValueError
        If no mesh is available.
    """
    case_dir = Path(case_path).resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    # Output directory
    if output_dir is None:
        p3d_dir = case_dir / "plot3d"
    else:
        p3d_dir = Path(output_dir)
    os.makedirs(p3d_dir, exist_ok=True)

    # Determine time values
    if time_range is not None:
        times = sorted(float(t) for t in time_range)
    else:
        from pyfoam.tools.foam_list_times import foam_list_times
        times = foam_list_times(case_dir)
        if not times and mesh is not None:
            times = [0.0]

    if mesh is None:
        raise ValueError("No mesh provided. Pass a mesh object directly.")

    # Build the structured grid from the unstructured mesh
    grid_points, grid_dims = _build_structured_grid(mesh)

    # Write grid file
    for t in times:
        t_name = _format_time(t)
        xyz_path = p3d_dir / f"{t_name}.xyz"
        _write_grid(xyz_path, grid_points, grid_dims, binary=binary)

        if fields:
            q_path = p3d_dir / f"{t_name}.q"
            _write_solution(q_path, grid_points, grid_dims, fields, binary=binary)

    return p3d_dir


# ---------------------------------------------------------------------------
# Structured grid extraction
# ---------------------------------------------------------------------------


def _build_structured_grid(mesh: "FvMesh"):
    """Extract structured-grid coordinates from an unstructured mesh.

    For hex-meshes, the cells are arranged as a 3D array
    ``(NI, NJ, NK)`` where ``NI``, ``NJ``, ``NK`` are the number of
    cells in each direction.  Plot3D uses node-based (vertex) coordinates,
    so we output ``(NI+1, NJ+1, NK+1)`` points.

    Returns
    -------
    grid_points : np.ndarray
        ``((NI+1)*(NJ+1)*(NK+1), 3)`` node coordinates.
    dims : tuple[int, int, int]
        ``(NI, NJ, NK)`` — number of cells in each direction.
    """
    n_cells = mesh.n_cells
    pts = mesh.points.detach().cpu().numpy()

    # Attempt to detect structured dimensions from a hex mesh.
    # If the mesh is structured hex, we can detect NI, NJ, NK from
    # the connectivity.  For unstructured meshes, we create a single
    # block with all cells in a 1D column.
    dims = _detect_grid_dims(mesh)

    NI, NJ, NK = dims
    n_nodes_x = NI + 1
    n_nodes_y = NJ + 1
    n_nodes_z = NK + 1

    # For hex meshes, extract corner-based node coordinates
    grid_points = _extract_node_coords(mesh, dims)

    return grid_points, dims


def _detect_grid_dims(mesh: "FvMesh") -> tuple[int, int, int]:
    """Detect structured grid dimensions from mesh.

    Attempts to find (NI, NJ, NK) by analysing the mesh topology.
    Falls back to (n_cells, 1, 1) for unstructured meshes.
    """
    n_cells = mesh.n_cells
    points = mesh.points.detach().cpu().numpy()

    # Find bounding box
    xmin, ymin, zmin = points.min(axis=0)
    xmax, ymax, zmax = points.max(axis=0)

    # Count unique coordinate values in each direction
    # with a tolerance for floating-point grouping
    def _count_unique(coords, tol=1e-6):
        if len(coords) == 0:
            return 1
        sorted_c = np.sort(coords)
        groups = [sorted_c[0]]
        for c in sorted_c[1:]:
            if abs(c - groups[-1]) > tol:
                groups.append(c)
        return len(groups)

    nx = max(_count_unique(points[:, 0]) - 1, 1)
    ny = max(_count_unique(points[:, 1]) - 1, 1)
    nz = max(_count_unique(points[:, 2]) - 1, 1)

    # Verify the product matches n_cells (or close to it)
    if nx * ny * nz == n_cells:
        return (nx, ny, nz)

    # For 2D meshes (single z-layer)
    if nz == 1 and nx * ny == n_cells:
        return (nx, ny, 1)

    # Fallback: 1D column
    return (n_cells, 1, 1)


def _extract_node_coords(
    mesh: "FvMesh", dims: tuple[int, int, int],
) -> np.ndarray:
    """Extract structured node coordinates.

    For each structured node (i, j, k), compute its position as the
    average of the surrounding cell centres, or directly from mesh
    points when available.

    Returns
    -------
    np.ndarray
        ``(NI+1, NJ+1, NK+1, 3)`` array of node coordinates.
    """
    NI, NJ, NK = dims
    pts = mesh.points.detach().cpu().numpy()
    cell_centres = mesh.cell_centres.detach().cpu().numpy()

    n_nodes_x = NI + 1
    n_nodes_y = NJ + 1
    n_nodes_z = NK + 1

    grid = np.zeros((n_nodes_x, n_nodes_y, n_nodes_z, 3))

    # For structured hex meshes, nodes are at the corners of cells.
    # We assign nodes by geometric position: each node is the average
    # of the (up to 8) cell centres surrounding it.
    for k in range(n_nodes_z):
        for j in range(n_nodes_y):
            for i in range(n_nodes_x):
                # Cells sharing this node: (i-1..i, j-1..j, k-1..k)
                # bounded by [0, NI-1] etc.
                ci_lo = max(i - 1, 0)
                ci_hi = min(i, NI - 1)
                cj_lo = max(j - 1, 0)
                cj_hi = min(j, NJ - 1)
                ck_lo = max(k - 1, 0)
                ck_hi = min(k, NK - 1)

                accum = np.zeros(3)
                count = 0
                for ck in range(ck_lo, ck_hi + 1):
                    for cj in range(cj_lo, cj_hi + 1):
                        for ci in range(ci_lo, ci_hi + 1):
                            idx = ck * NJ * NI + cj * NI + ci
                            if idx < len(cell_centres):
                                accum += cell_centres[idx]
                                count += 1

                if count > 0:
                    grid[i, j, k] = accum / count

    return grid


# ---------------------------------------------------------------------------
# Grid file (.xyz)
# ---------------------------------------------------------------------------


def _write_grid(
    path: Path,
    grid: np.ndarray,
    dims: tuple[int, int, int],
    binary: bool = False,
) -> None:
    """Write a Plot3D multi-block grid file (.xyz).

    Parameters
    ----------
    path : Path
        Output file path.
    grid : np.ndarray
        ``(NI+1, NJ+1, NK+1, 3)`` node coordinates.
    dims : tuple[int, int, int]
        ``(NI, NJ, NK)`` cell counts.
    binary : bool
        If True, write binary format.
    """
    NI, NJ, NK = dims
    nx, ny, nz = NI + 1, NJ + 1, NK + 1

    if binary:
        _write_grid_binary(path, grid, nx, ny, nz)
    else:
        _write_grid_ascii(path, grid, nx, ny, nz)


def _write_grid_ascii(path: Path, grid: np.ndarray, nx: int, ny: int, nz: int) -> None:
    """Write ASCII Plot3D grid file."""
    with open(path, "w", encoding="utf-8") as f:
        # Number of blocks
        f.write("1\n")
        # Block dimensions
        f.write(f"{nx} {ny} {nz}\n")
        # X coordinates
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{grid[i, j, k, 0]:18.10E}\n")
        # Y coordinates
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{grid[i, j, k, 1]:18.10E}\n")
        # Z coordinates
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{grid[i, j, k, 2]:18.10E}\n")


def _write_grid_binary(path: Path, grid: np.ndarray, nx: int, ny: int, nz: int) -> None:
    """Write binary Plot3D grid file."""
    with open(path, "wb") as f:
        # Fortran record: number of blocks
        f.write(struct.pack("<i", 4))
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<i", 4))

        # Fortran record: block dimensions
        f.write(struct.pack("<i", 12))
        f.write(struct.pack("<iii", nx, ny, nz))
        f.write(struct.pack("<i", 12))

        # Coordinates (x, y, z in order)
        n_nodes = nx * ny * nz
        record_size = n_nodes * 8  # float64

        for dim_idx in range(3):
            f.write(struct.pack("<i", record_size))
            data = grid[:, :, :, dim_idx].flatten(order='F').astype(np.float64)
            f.write(data.tobytes())
            f.write(struct.pack("<i", record_size))


# ---------------------------------------------------------------------------
# Solution file (.q)
# ---------------------------------------------------------------------------


def _write_solution(
    path: Path,
    grid: np.ndarray,
    dims: tuple[int, int, int],
    fields: Dict[str, np.ndarray],
    binary: bool = False,
) -> None:
    """Write a Plot3D multi-block solution file (.q).

    The solution file contains per-node values:
    - Mach number, alpha (angle of attack), Reynolds, time
    - rho, u, v, w, pressure (per node)

    For CFD applications, we interpolate cell-centred fields to nodes.

    Parameters
    ----------
    path : Path
        Output file path.
    grid : np.ndarray
        Node coordinates.
    dims : tuple
        Grid dimensions.
    fields : dict
        Cell-centred fields to interpolate to nodes.
    binary : bool
        If True, write binary format.
    """
    NI, NJ, NK = dims
    nx, ny, nz = NI + 1, NJ + 1, NK + 1

    # Extract scalar and vector fields
    rho_data = fields.get("rho", None)
    U_data = fields.get("U", None)
    p_data = fields.get("p", None)

    # Build per-node arrays (interpolate cell-centred to nodes)
    n_nodes = nx * ny * nz

    if rho_data is not None:
        rho_node = _cell_to_node(rho_data, dims)
    else:
        rho_node = np.ones(n_nodes)

    if U_data is not None and U_data.ndim == 2 and U_data.shape[1] >= 3:
        u_node = _cell_to_node(U_data[:, 0], dims)
        v_node = _cell_to_node(U_data[:, 1], dims)
        w_node = _cell_to_node(U_data[:, 2], dims)
    else:
        u_node = np.zeros(n_nodes)
        v_node = np.zeros(n_nodes)
        w_node = np.zeros(n_nodes)

    if p_data is not None:
        p_node = _cell_to_node(p_data, dims)
    else:
        p_node = np.ones(n_nodes) * 101325.0

    if binary:
        _write_solution_binary(path, dims, rho_node, u_node, v_node, w_node, p_node)
    else:
        _write_solution_ascii(path, dims, rho_node, u_node, v_node, w_node, p_node)


def _write_solution_ascii(
    path: Path, dims: tuple[int, int, int],
    rho: np.ndarray, u: np.ndarray, v: np.ndarray, w: np.ndarray, p: np.ndarray,
) -> None:
    """Write ASCII Plot3D solution file."""
    NI, NJ, NK = dims
    nx, ny, nz = NI + 1, NJ + 1, NK + 1

    with open(path, "w", encoding="utf-8") as f:
        # Number of blocks
        f.write("1\n")
        # Block dimensions
        f.write(f"{nx} {ny} {nz}\n")
        # Reference values: Mach, alpha, Reynolds, time
        f.write("0.0 0.0 0.0 0.0\n")

        # Per-node data: rho, u, v, w, p
        for arr in [rho, u, v, w, p]:
            for val in arr:
                f.write(f"{val:18.10E}\n")


def _write_solution_binary(
    path: Path, dims: tuple[int, int, int],
    rho: np.ndarray, u: np.ndarray, v: np.ndarray, w: np.ndarray, p: np.ndarray,
) -> None:
    """Write binary Plot3D solution file."""
    NI, NJ, NK = dims
    nx, ny, nz = NI + 1, NJ + 1, NK + 1
    n_nodes = nx * ny * nz

    with open(path, "wb") as f:
        # Number of blocks
        f.write(struct.pack("<i", 4))
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<i", 4))

        # Block dimensions
        f.write(struct.pack("<i", 12))
        f.write(struct.pack("<iii", nx, ny, nz))
        f.write(struct.pack("<i", 12))

        # Reference values: Mach, alpha, Reynolds, time
        f.write(struct.pack("<i", 32))
        f.write(struct.pack("<dddd", 0.0, 0.0, 0.0, 0.0))
        f.write(struct.pack("<i", 32))

        # Per-node data
        record_size = n_nodes * 8
        for arr in [rho, u, v, w, p]:
            f.write(struct.pack("<i", record_size))
            f.write(arr.astype(np.float64).tobytes())
            f.write(struct.pack("<i", record_size))


# ---------------------------------------------------------------------------
# Cell-to-node interpolation
# ---------------------------------------------------------------------------


def _cell_to_node(cell_data: np.ndarray, dims: tuple[int, int, int]) -> np.ndarray:
    """Interpolate cell-centred data to node values.

    Each node value is the average of the surrounding cells.

    Parameters
    ----------
    cell_data : np.ndarray
        ``(n_cells,)`` cell-centred values.
    dims : tuple[int, int, int]
        ``(NI, NJ, NK)`` grid dimensions.

    Returns
    -------
    np.ndarray
        ``(n_nodes,)`` node-averaged values.
    """
    NI, NJ, NK = dims
    nx, ny, nz = NI + 1, NJ + 1, NK + 1
    n_nodes = nx * ny * nz

    node_sum = np.zeros(n_nodes)
    node_count = np.zeros(n_nodes)

    # Reshape cell data to 3D array
    cell_3d = cell_data.reshape(NK, NJ, NI)

    for dk in range(2):
        for dj in range(2):
            for di in range(2):
                for k in range(max(0, -dk), min(NK, NK + 1 - dk) - (1 if dk == 0 else 0)):
                    for j in range(max(0, -dj), min(NJ, NJ + 1 - dj) - (1 if dj == 0 else 0)):
                        for i in range(max(0, -di), min(NI, NI + 1 - di) - (1 if di == 0 else 0)):
                            ni = i + di
                            nj = j + dj
                            nk = k + dk
                            if ni < nx and nj < ny and nk < nz:
                                node_idx = nk * ny * nx + nj * nx + ni
                                node_sum[node_idx] += cell_3d[k, j, i]
                                node_count[node_idx] += 1

    # Average
    valid = node_count > 0
    node_sum[valid] /= node_count[valid]

    return node_sum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_time(t: float) -> str:
    """Format a time value for use in file names."""
    if t == int(t):
        return str(int(t))
    return f"{t:g}"
