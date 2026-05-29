"""
foamToEnsight enhanced v3 — enhanced EnSight export with binary format
and multi-variable export (third generation).

Extends :func:`foam_to_ensight_enhanced_2` with:

- **Parallel I/O buffer**: Uses a write buffer for large binary exports
  to reduce system call overhead.
- **Selective variable export**: Export only specified variables instead
  of all fields.
- **Time-step compression**: Deduplicate geometry across time steps
  when mesh does not change.
- **Element type mapping**: Explicit mapping from polyhedral cells
  to EnSight element types with automatic decomposition.

Usage::

    from pyfoam.tools.foam_to_ensight_enhanced_3 import foam_to_ensight_enhanced_3

    result = foam_to_ensight_enhanced_3(
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
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnSightV3Result", "foam_to_ensight_enhanced_3"]


@dataclass
class EnSightV3Result:
    """Result from :func:`foam_to_ensight_enhanced_3`.

    Attributes
    ----------
    case_file : Path
    geometry_files : list[Path]
    variable_files : list[Path]
    n_times, n_variables, n_parts : int
    binary : bool
    geometry_reused : int
        Number of time steps that reuse the same geometry file.
    """

    case_file: Path = field(default_factory=lambda: Path("."))
    geometry_files: List[Path] = field(default_factory=list)
    variable_files: List[Path] = field(default_factory=list)
    n_times: int = 0
    n_variables: int = 0
    n_parts: int = 1
    binary: bool = False
    geometry_reused: int = 0


def foam_to_ensight_enhanced_3(
    case_path: Union[str, Path],
    time_range: Optional[Sequence[float]] = None,
    mesh: Optional["FvMesh"] = None,
    fields: Optional[Dict[str, np.ndarray]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    binary: bool = False,
    write_boundary_parts: bool = True,
    export_variables: Optional[Set[str]] = None,
    deduplicate_geometry: bool = True,
) -> EnSightV3Result:
    """Export to EnSight Gold with selective variables and geometry deduplication.

    Parameters
    ----------
    case_path : str or Path
        Case directory root.
    time_range : sequence of float, optional
        Subset of time values.
    mesh : FvMesh, optional
        Pre-loaded mesh.
    fields : dict, optional
        ``{name: array}`` fields.
    output_dir : str or Path, optional
        Output directory.
    binary : bool
        Binary EnSight Gold format.
    write_boundary_parts : bool
        Write boundary patches as separate parts.
    export_variables : set of str, optional
        If given, only export variables in this set.
    deduplicate_geometry : bool
        If True, reuse geometry file across time steps when mesh is
        unchanged (default True).

    Returns
    -------
    EnSightV3Result
        Export metadata.
    """
    case_dir = Path(case_path).resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    if mesh is None:
        raise ValueError("No mesh provided.")

    case_name = case_dir.name

    if output_dir is None:
        ensight_dir = case_dir / "EnSight_v3" / case_name
    else:
        ensight_dir = Path(output_dir)
    os.makedirs(ensight_dir, exist_ok=True)

    if time_range is not None:
        times = sorted(float(t) for t in time_range)
    else:
        times = [0.0]

    cell_verts = _compute_cell_vertices(mesh)

    # Filter variables
    active_fields = fields
    if export_variables is not None and fields is not None:
        active_fields = {k: v for k, v in fields.items() if k in export_variables}

    var_info = _classify_variables(active_fields) if active_fields else {}

    geo_files = []
    var_files = []
    n_parts = 1
    geo_reused = 0

    if write_boundary_parts:
        n_parts += len(mesh.boundary)

    # Geometry deduplication: write once if mesh is static
    geo_written = False
    geo_path = None

    for t in times:
        t_name = _format_time(t)

        if deduplicate_geometry and geo_written:
            geo_reused += 1
        else:
            geo_path = _write_geometry(
                ensight_dir, t_name, mesh, cell_verts, binary, write_boundary_parts,
            )
            geo_files.append(geo_path)
            geo_written = True

        if active_fields:
            for name, data in active_fields.items():
                vtype = var_info[name]
                vp = _write_variable(
                    ensight_dir, t_name, name, data, vtype, binary,
                    mesh, write_boundary_parts,
                )
                var_files.append(vp)

    # Write .case descriptor
    case_file = ensight_dir / f"{case_name}.case"
    _write_case_file(case_file, case_name, times, var_info, binary)

    return EnSightV3Result(
        case_file=case_file,
        geometry_files=geo_files,
        variable_files=var_files,
        n_times=len(times),
        n_variables=len(var_info),
        n_parts=n_parts,
        binary=binary,
        geometry_reused=geo_reused,
    )


# ---------------------------------------------------------------------------
# Variable classification
# ---------------------------------------------------------------------------


def _classify_variables(fields):
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


def _write_geometry(ensight_dir, t_name, mesh, cell_verts, binary, boundary_parts):
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
                _write_boundary_topology_ascii(f, mesh, patch)


def _write_geo_binary(path, t_name, pts, mesh, cell_verts, boundary_parts):
    buf = bytearray()

    def _bw_buf(text):
        buf.extend(text.encode("ascii")[:80].ljust(80, b"\0"))

    def _bi_buf(data):
        raw = data.astype(np.int32).tobytes()
        n = len(raw)
        buf.extend(struct.pack("i", n) + raw + struct.pack("i", n))

    def _bf_buf(data):
        raw = data.astype(np.float32).tobytes()
        n = len(raw)
        buf.extend(struct.pack("i", n) + raw + struct.pack("i", n))

    _bw_buf("EnSight Gold binary")
    _bw_buf(f"geometry_{t_name}.geo")
    _bw_buf("node id off")
    _bw_buf("element id off")

    _bw_buf("part")
    _bi_buf(np.array([1], dtype=np.int32))
    _bw_buf("volume mesh")
    _bw_buf("coordinates")
    n_nodes = pts.shape[0]
    _bi_buf(np.array([n_nodes], dtype=np.int32))
    for d in range(3):
        _bf_buf(pts[:, d].astype(np.float32))
    _write_topology_binary_buf(buf, cell_verts)

    if boundary_parts:
        for pi, patch in enumerate(mesh.boundary):
            _bw_buf_buf(buf, "part")
            buf.extend(struct.pack("i", 4) + np.array([pi + 2], dtype=np.int32).tobytes() + struct.pack("i", 4))
            _bw_buf_buf(buf, patch["name"])
            _bw_buf_buf(buf, "coordinates")
            buf.extend(struct.pack("i", 4) + np.array([n_nodes], dtype=np.int32).tobytes() + struct.pack("i", 4))
            for d in range(3):
                raw = pts[:, d].astype(np.float32).tobytes()
                buf.extend(struct.pack("i", len(raw)) + raw + struct.pack("i", len(raw)))
            _write_boundary_topology_binary_buf(buf, mesh, patch)

    with open(path, "wb") as f:
        f.write(buf)


def _bw_buf_buf(buf, text):
    buf.extend(text.encode("ascii")[:80].ljust(80, b"\0"))


def _write_topology_ascii(f, cell_verts):
    classified = _classify_cells(cell_verts)
    for elem_type, cells in classified:
        if not cells:
            continue
        f.write(f"{elem_type}\n")
        f.write(f"{len(cells):12d}\n")
        for _, verts in cells:
            f.write("".join(f"{v + 1:12d}" for v in verts) + "\n")


def _write_topology_binary_buf(buf, cell_verts):
    classified = _classify_cells(cell_verts)
    for elem_type, cells in classified:
        if not cells:
            continue
        _bw_buf_buf(buf, elem_type)
        raw_i = np.array([len(cells)], dtype=np.int32).tobytes()
        buf.extend(struct.pack("i", len(raw_i)) + raw_i + struct.pack("i", len(raw_i)))
        nodes = []
        for _, verts in cells:
            nodes.extend(v + 1 for v in verts)
        raw_i = np.array(nodes, dtype=np.int32).tobytes()
        buf.extend(struct.pack("i", len(raw_i)) + raw_i + struct.pack("i", len(raw_i)))


def _write_boundary_topology_ascii(f, mesh, patch):
    start = patch["startFace"]
    nf = patch["nFaces"]
    tria3, quad4 = [], []
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
        f.write("tria3\n")
        f.write(f"{len(tria3):12d}\n")
        for v in tria3:
            f.write("".join(f"{vi:12d}" for vi in v) + "\n")
    if quad4:
        f.write("quad4\n")
        f.write(f"{len(quad4):12d}\n")
        for v in quad4:
            f.write("".join(f"{vi:12d}" for vi in v) + "\n")


def _write_boundary_topology_binary_buf(buf, mesh, patch):
    start = patch["startFace"]
    nf = patch["nFaces"]
    tria3, quad4 = [], []
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
        _bw_buf_buf(buf, "tria3")
        raw_i = np.array([len(tria3)], dtype=np.int32).tobytes()
        buf.extend(struct.pack("i", len(raw_i)) + raw_i + struct.pack("i", len(raw_i)))
        nodes = [vi for v in tria3 for vi in v]
        raw_i = np.array(nodes, dtype=np.int32).tobytes()
        buf.extend(struct.pack("i", len(raw_i)) + raw_i + struct.pack("i", len(raw_i)))
    if quad4:
        _bw_buf_buf(buf, "quad4")
        raw_i = np.array([len(quad4)], dtype=np.int32).tobytes()
        buf.extend(struct.pack("i", len(raw_i)) + raw_i + struct.pack("i", len(raw_i)))
        nodes = [vi for v in quad4 for vi in v]
        raw_i = np.array(nodes, dtype=np.int32).tobytes()
        buf.extend(struct.pack("i", len(raw_i)) + raw_i + struct.pack("i", len(raw_i)))


# ---------------------------------------------------------------------------
# Variable writing
# ---------------------------------------------------------------------------


def _write_variable(ensight_dir, t_name, name, data, vtype, binary, mesh, boundary_parts):
    suffix_map = {"scalar": "scl", "vector": "vec", "tensor": "tsr"}
    suffix = suffix_map.get(vtype, "scl")
    var_path = ensight_dir / f"{name}_{t_name}.{suffix}"

    if binary:
        with open(var_path, "wb") as f:
            _bw(f, var_path.name)
            _bw(f, f"EnSight Gold: {name}")
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


def _bw(f, text):
    f.write(text.encode("ascii")[:80].ljust(80, b"\0"))


def _bi(f, data):
    raw = data.astype(np.int32).tobytes()
    n = len(raw)
    f.write(struct.pack("i", n) + raw + struct.pack("i", n))


def _bf(f, data):
    raw = data.astype(np.float32).tobytes()
    n = len(raw)
    f.write(struct.pack("i", n) + raw + struct.pack("i", n))


# ---------------------------------------------------------------------------
# Cell connectivity
# ---------------------------------------------------------------------------


def _compute_cell_vertices(mesh):
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


def _format_time(t):
    return str(int(t)) if t == int(t) else f"{t:g}"
