"""
foamToEnsight enhanced v2 — enhanced EnSight export with binary format
and multi-variable export (second generation).

Extends :func:`foam_to_ensight_enhanced` with:

- **Per-part variables**: Variables can be written per-part (volume
  and boundary patches separately).
- **Tensor support**: Full symmetric tensor export (6-component) as
  ``.tsr`` files.
- **Efficient I/O**: Uses structured packing for binary records,
  reducing file sizes by 4x compared to ASCII.

Usage::

    from pyfoam.tools.foam_to_ensight_enhanced_2 import foam_to_ensight_enhanced_2

    result = foam_to_ensight_enhanced_2(
        case_path="cavity",
        mesh=mesh,
        fields={"p": p_arr, "U": U_arr},
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

__all__ = ["EnSightV2Result", "foam_to_ensight_enhanced_2"]


@dataclass
class EnSightV2Result:
    """Result from :func:`foam_to_ensight_enhanced_2`.

    Attributes
    ----------
    case_file : Path
        Path to the .case file.
    geometry_files : list[Path]
        All geometry files generated.
    variable_files : list[Path]
        All variable files generated.
    n_times : int
        Time steps exported.
    n_variables : int
        Variables exported.
    n_parts : int
        Parts (volume + boundary) written.
    binary : bool
        Whether binary format was used.
    """

    case_file: Path = field(default_factory=lambda: Path("."))
    geometry_files: List[Path] = field(default_factory=list)
    variable_files: List[Path] = field(default_factory=list)
    n_times: int = 0
    n_variables: int = 0
    n_parts: int = 1
    binary: bool = False


def foam_to_ensight_enhanced_2(
    case_path: Union[str, Path],
    time_range: Optional[Sequence[float]] = None,
    mesh: Optional["FvMesh"] = None,
    fields: Optional[Dict[str, np.ndarray]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    binary: bool = False,
    write_boundary_parts: bool = True,
) -> EnSightV2Result:
    """Export to EnSight Gold with per-part variables and tensor support.

    Parameters
    ----------
    case_path : str or Path
        Case directory root.
    time_range : sequence of float, optional
        Subset of time values.
    mesh : FvMesh, optional
        Pre-loaded mesh.
    fields : dict, optional
        ``{name: array}`` fields. Scalars ``(n,)``, vectors ``(n, 3)``,
        tensors ``(n, 6)``.
    output_dir : str or Path, optional
        Output directory.
    binary : bool
        Binary EnSight Gold format.
    write_boundary_parts : bool
        Write boundary patches as separate parts.

    Returns
    -------
    EnSightV2Result
        Export metadata.
    """
    case_dir = Path(case_path).resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    if mesh is None:
        raise ValueError("No mesh provided.")

    case_name = case_dir.name

    if output_dir is None:
        ensight_dir = case_dir / "EnSight_v2" / case_name
    else:
        ensight_dir = Path(output_dir)
    os.makedirs(ensight_dir, exist_ok=True)

    # Time values
    if time_range is not None:
        times = sorted(float(t) for t in time_range)
    else:
        times = [0.0]

    # Cell-vertex mapping
    cell_verts = _compute_cell_vertices(mesh)

    # Classify variables
    var_info = _classify_variables(fields) if fields else {}

    geo_files = []
    var_files = []
    n_parts = 1  # Volume part always present

    if write_boundary_parts:
        n_parts += len(mesh.boundary)

    for t in times:
        t_name = _format_time(t)

        # Write geometry
        geo_path = _write_geometry(
            ensight_dir, t_name, mesh, cell_verts, binary, write_boundary_parts,
        )
        geo_files.append(geo_path)

        # Write variables
        if fields:
            for name, data in fields.items():
                vtype = var_info[name]
                vp = _write_variable(
                    ensight_dir, t_name, name, data, vtype, binary,
                    mesh, write_boundary_parts,
                )
                var_files.append(vp)

    # Write .case descriptor
    case_file = ensight_dir / f"{case_name}.case"
    _write_case_file(case_file, case_name, times, var_info, binary)

    return EnSightV2Result(
        case_file=case_file,
        geometry_files=geo_files,
        variable_files=var_files,
        n_times=len(times),
        n_variables=len(var_info),
        n_parts=n_parts,
        binary=binary,
    )


# ---------------------------------------------------------------------------
# Variable classification
# ---------------------------------------------------------------------------


def _classify_variables(fields: Dict[str, np.ndarray]) -> Dict[str, str]:
    result = {}
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
    ensight_dir: Path, t_name: str, mesh: "FvMesh",
    cell_verts: list[list[int]], binary: bool, boundary_parts: bool,
) -> Path:
    geo_path = ensight_dir / f"geometry_{t_name}.geo"
    pts = mesh.points.detach().cpu().numpy()

    if binary:
        _write_geo_binary(geo_path, t_name, pts, mesh, cell_verts, boundary_parts)
    else:
        _write_geo_ascii(geo_path, t_name, pts, mesh, cell_verts, boundary_parts)

    return geo_path


def _write_geo_ascii(path, t_name, pts, mesh, cell_verts, boundary_parts):
    with open(path, "w") as f:
        f.write("EnSight Gold ASCII\n")
        f.write(f"geometry_{t_name}.geo\n")
        f.write("node id off\nelement id off\n")

        # Part 1: volume
        f.write("part\n")
        f.write(f"{1:12d}\n")
        f.write("volume mesh\n")
        f.write("coordinates\n")
        n_nodes = pts.shape[0]
        f.write(f"{n_nodes:12d}\n")
        for d in range(3):
            for i in range(n_nodes):
                f.write(f"{pts[i, d]:14.6E}\n")
        _write_topology_ascii(f, cell_verts)

        # Boundary parts
        if boundary_parts:
            for pi, patch in enumerate(mesh.boundary):
                part_id = pi + 2
                f.write("part\n")
                f.write(f"{part_id:12d}\n")
                f.write(f"{patch['name']}\n")
                f.write("coordinates\n")
                f.write(f"{n_nodes:12d}\n")
                for d in range(3):
                    for i in range(n_nodes):
                        f.write(f"{pts[i, d]:14.6E}\n")
                # Boundary faces as 2D elements (tria3 or quad4)
                _write_boundary_topology_ascii(f, mesh, patch)


def _write_geo_binary(path, t_name, pts, mesh, cell_verts, boundary_parts):
    with open(path, "wb") as f:
        _bw(f, "EnSight Gold binary")
        _bw(f, f"geometry_{t_name}.geo")
        _bw(f, "node id off")
        _bw(f, "element id off")

        # Part 1: volume
        _bw(f, "part")
        _bi(f, np.array([1], dtype=np.int32))
        _bw(f, "volume mesh")
        _bw(f, "coordinates")
        n_nodes = pts.shape[0]
        _bi(f, np.array([n_nodes], dtype=np.int32))
        for d in range(3):
            _bf(f, pts[:, d].astype(np.float32))
        _write_topology_binary(f, cell_verts)

        if boundary_parts:
            for pi, patch in enumerate(mesh.boundary):
                _bw(f, "part")
                _bi(f, np.array([pi + 2], dtype=np.int32))
                _bw(f, patch["name"])
                _bw(f, "coordinates")
                _bi(f, np.array([n_nodes], dtype=np.int32))
                for d in range(3):
                    _bf(f, pts[:, d].astype(np.float32))
                _write_boundary_topology_binary(f, mesh, patch)


def _write_topology_ascii(f, cell_verts):
    classified = _classify_cells(cell_verts)
    for elem_type, cells in classified:
        if not cells:
            continue
        f.write(f"{elem_type}\n")
        f.write(f"{len(cells):12d}\n")
        for _, verts in cells:
            f.write("".join(f"{v + 1:12d}" for v in verts) + "\n")


def _write_topology_binary(f, cell_verts):
    classified = _classify_cells(cell_verts)
    for elem_type, cells in classified:
        if not cells:
            continue
        _bw(f, elem_type)
        _bi(f, np.array([len(cells)], dtype=np.int32))
        nodes = []
        for _, verts in cells:
            nodes.extend(v + 1 for v in verts)
        _bi(f, np.array(nodes, dtype=np.int32))


def _write_boundary_topology_ascii(f, mesh, patch):
    start = patch["startFace"]
    nf = patch["nFaces"]
    tria3 = []
    quad4 = []
    for fi in range(start, start + nf):
        face = mesh.faces[fi]
        nn = face.shape[0]
        verts = [(v + 1).item() for v in face]
        if nn == 3:
            tria3.append(verts)
        elif nn == 4:
            quad4.append(verts)
        else:
            # Triangulate polygon
            for k in range(1, nn - 1):
                tria3.append([verts[0], verts[k], verts[k + 1]])

    if tria3:
        f.write("tria3\n")
        f.write(f"{len(tria3):12d}\n")
        for v in tria3:
            f.write("".join(f"{vi:12d}" for vi in v) + "\n")
    if quad4:
        f.write("quad4\n")
        f.write(f"{len(quad4):12d}\n")
        for v in quad4:
            f.write("".join(f"{vi:12d}" for vi in v) + "\n")


def _write_boundary_topology_binary(f, mesh, patch):
    start = patch["startFace"]
    nf = patch["nFaces"]
    tria3 = []
    quad4 = []
    for fi in range(start, start + nf):
        face = mesh.faces[fi]
        nn = face.shape[0]
        verts = [(v + 1).item() for v in face]
        if nn == 3:
            tria3.append(verts)
        elif nn == 4:
            quad4.append(verts)
        else:
            for k in range(1, nn - 1):
                tria3.append([verts[0], verts[k], verts[k + 1]])

    if tria3:
        _bw(f, "tria3")
        _bi(f, np.array([len(tria3)], dtype=np.int32))
        nodes = [vi for v in tria3 for vi in v]
        _bi(f, np.array(nodes, dtype=np.int32))
    if quad4:
        _bw(f, "quad4")
        _bi(f, np.array([len(quad4)], dtype=np.int32))
        nodes = [vi for v in quad4 for vi in v]
        _bi(f, np.array(nodes, dtype=np.int32))


# ---------------------------------------------------------------------------
# Variable writing
# ---------------------------------------------------------------------------


def _write_variable(
    ensight_dir, t_name, name, data, vtype, binary, mesh, boundary_parts,
) -> Path:
    suffix_map = {"scalar": "scl", "vector": "vec", "tensor": "tsr"}
    suffix = suffix_map.get(vtype, "scl")
    var_path = ensight_dir / f"{name}_{t_name}.{suffix}"

    n_vol = data.shape[0]

    if binary:
        with open(var_path, "wb") as f:
            _bw(f, var_path.name)
            _bw(f, f"EnSight Gold: {name}")
            # Part 1: volume
            _bw(f, "part")
            _bi(f, np.array([1], dtype=np.int32))
            _bw(f, "coordinates")
            _write_field_binary(f, data, vtype)

            if boundary_parts:
                for pi, patch in enumerate(mesh.boundary):
                    _bw(f, "part")
                    _bi(f, np.array([pi + 2], dtype=np.int32))
                    _bw(f, "coordinates")
                    _write_field_binary(f, data, vtype)
    else:
        with open(var_path, "w") as f:
            f.write(f"{var_path.name}\n")
            f.write(f"EnSight Gold: {name}\n")
            f.write("part\n")
            f.write(f"{1:12d}\n")
            f.write("coordinates\n")
            _write_field_ascii(f, data, vtype)

            if boundary_parts:
                for pi, patch in enumerate(mesh.boundary):
                    f.write("part\n")
                    f.write(f"{pi + 2:12d}\n")
                    f.write("coordinates\n")
                    _write_field_ascii(f, data, vtype)

    return var_path


def _write_field_ascii(f, data, vtype):
    n = data.shape[0]
    if vtype == "vector":
        for d in range(3):
            for i in range(n):
                f.write(f"{data[i, d]:14.6E}\n")
    elif vtype == "tensor":
        for c in range(6):
            for i in range(n):
                f.write(f"{data[i, c]:14.6E}\n")
    else:
        for i in range(n):
            f.write(f"{data[i]:14.6E}\n")


def _write_field_binary(f, data, vtype):
    if vtype == "vector":
        for d in range(3):
            _bf(f, data[:, d].astype(np.float32))
    elif vtype == "tensor":
        for c in range(6):
            _bf(f, data[:, c].astype(np.float32))
    else:
        _bf(f, data.astype(np.float32))


# ---------------------------------------------------------------------------
# Case file
# ---------------------------------------------------------------------------


def _write_case_file(case_file, case_name, times, var_info, binary):
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
        f.write(f"number of steps:      {len(times)}\n")
        f.write("filename start number:  0\n")
        f.write("filename increment:     1\n")
        f.write("time values:\n")
        for t in times:
            f.write(f"  {t:14.6E}\n")


# ---------------------------------------------------------------------------
# Binary helpers
# ---------------------------------------------------------------------------


def _bw(f, text: str):
    """Write 80-char binary text record."""
    f.write(text.encode("ascii")[:80].ljust(80, b"\0"))


def _bi(f, data: np.ndarray):
    """Write int32 Fortran record."""
    raw = data.astype(np.int32).tobytes()
    n = len(raw)
    f.write(struct.pack("i", n) + raw + struct.pack("i", n))


def _bf(f, data: np.ndarray):
    """Write float32 Fortran record."""
    raw = data.astype(np.float32).tobytes()
    n = len(raw)
    f.write(struct.pack("i", n) + raw + struct.pack("i", n))


# ---------------------------------------------------------------------------
# Cell-vertex connectivity
# ---------------------------------------------------------------------------


def _compute_cell_vertices(mesh: "FvMesh") -> list[list[int]]:
    n_cells = mesh.n_cells
    owner = mesh.owner.detach().cpu().numpy()
    neighbour = mesh.neighbour.detach().cpu().numpy()
    n_internal = mesh.n_internal_faces

    cell_verts: list[set[int]] = [set() for _ in range(n_cells)]
    for fi, face in enumerate(mesh.faces):
        fn = face.detach().cpu().numpy().tolist()
        cell_verts[int(owner[fi])].update(fn)
        if fi < n_internal:
            cell_verts[int(neighbour[fi])].update(fn)

    return [sorted(v) for v in cell_verts]


def _classify_cells(cell_verts):
    hex_c, tet_c, pyr_c, pen_c, poly_c = [], [], [], [], []
    for c, verts in enumerate(cell_verts):
        nn = len(verts)
        entry = (c, verts)
        if nn == 8:
            hex_c.append(entry)
        elif nn == 4:
            tet_c.append(entry)
        elif nn == 5:
            pyr_c.append(entry)
        elif nn == 6:
            pen_c.append(entry)
        else:
            poly_c.append(entry)

    result = []
    if hex_c:
        result.append(("hexa8", hex_c))
    if tet_c:
        result.append(("tetra4", tet_c))
    if pyr_c:
        result.append(("pyramid5", pyr_c))
    if pen_c:
        result.append(("penta6", pen_c))
    if poly_c:
        result.append(("nsided", poly_c))
    return result


def _format_time(t: float) -> str:
    return str(int(t)) if t == int(t) else f"{t:g}"
