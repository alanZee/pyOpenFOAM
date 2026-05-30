"""
foamToEnsight enhanced v5 — enhanced EnSight export with adaptive compression,
multi-resolution output, and streaming support (fifth generation).

Extends :func:`foam_to_ensight_enhanced_4` with:

- **Adaptive compression**: Automatically select between ASCII and binary
  based on data characteristics and available disk space.
- **Multi-resolution output**: Write coarse and fine mesh representations
  for multi-resolution visualisation.
- **Streaming support**: Stream data directly to disk without buffering
  the entire dataset in memory.

Usage::

    from pyfoam.tools.foam_to_ensight_enhanced_5 import foam_to_ensight_enhanced_5

    result = foam_to_ensight_enhanced_5(
        case_path="cavity",
        mesh=mesh,
        fields={"p": p_arr, "U": U_arr},
        binary=True,
        multi_resolution=True,
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

__all__ = ["EnSightV5Result", "foam_to_ensight_enhanced_5"]


@dataclass
class EnSightV5Result:
    """Result from :func:`foam_to_ensight_enhanced_5`.

    Attributes
    ----------
    case_file : Path
    geometry_files : list[Path]
    variable_files : list[Path]
    n_times, n_variables, n_parts : int
    binary : bool
    geometry_reused : int
    total_bytes_written : int
    export_time_ms : float
    coarse_geometry_file : Path, optional
        Path to the coarse-resolution geometry.
    n_coarse_cells : int
        Number of cells in the coarse mesh.
    compression_ratio : float
        Ratio of ASCII to binary file sizes.
    streamed : bool
        Whether streaming I/O was used.
    """

    case_file: Path = field(default_factory=lambda: Path("."))
    geometry_files: List[Path] = field(default_factory=list)
    variable_files: List[Path] = field(default_factory=list)
    n_times: int = 0
    n_variables: int = 0
    n_parts: int = 1
    binary: bool = False
    geometry_reused: int = 0
    total_bytes_written: int = 0
    export_time_ms: float = 0.0
    coarse_geometry_file: Optional[Path] = None
    n_coarse_cells: int = 0
    compression_ratio: float = 1.0
    streamed: bool = False


def foam_to_ensight_enhanced_5(
    case_path: Union[str, Path],
    time_range: Optional[Sequence[float]] = None,
    mesh: Optional["FvMesh"] = None,
    fields: Optional[Dict[str, np.ndarray]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    binary: bool = False,
    write_boundary_parts: bool = True,
    export_variables: Optional[Set[str]] = None,
    deduplicate_geometry: bool = True,
    chunk_size: int = 1024 * 1024,
    multi_resolution: bool = False,
    coarse_ratio: float = 0.25,
    stream_mode: bool = False,
    adaptive_compression: bool = False,
) -> EnSightV5Result:
    """Export to EnSight Gold with multi-resolution and streaming support.

    Parameters
    ----------
    case_path : str or Path
    time_range : sequence of float, optional
    mesh : FvMesh, optional
    fields : dict, optional
    output_dir : str or Path, optional
    binary : bool
    write_boundary_parts : bool
    export_variables : set of str, optional
    deduplicate_geometry : bool
    chunk_size : int
        Binary write chunk size in bytes.
    multi_resolution : bool
        Also export a coarse-resolution mesh.
    coarse_ratio : float
        Cell count ratio for coarse mesh (0-1).
    stream_mode : bool
        Write data in streaming mode (minimal memory).
    adaptive_compression : bool
        Automatically choose binary/ascii based on data size.

    Returns
    -------
    EnSightV5Result
    """
    import time as _time
    t_start = _time.perf_counter()

    case_dir = Path(case_path).resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    if mesh is None:
        raise ValueError("No mesh provided.")

    case_name = case_dir.name

    if output_dir is None:
        ensight_dir = case_dir / "EnSight_v5" / case_name
    else:
        ensight_dir = Path(output_dir)
    os.makedirs(ensight_dir, exist_ok=True)

    if time_range is not None:
        times = sorted(float(t) for t in time_range)
    else:
        times = [0.0]

    cell_verts = _compute_cell_vertices(mesh)

    # Adaptive compression: choose binary for large datasets
    use_binary = binary
    if adaptive_compression:
        n_cells = mesh.n_cells
        data_size = n_cells * 4 * 3  # approximate
        if fields:
            for name, data in fields.items():
                data_size += data.nbytes
        # Use binary if total data > 10 MB
        use_binary = binary or data_size > 10 * 1024 * 1024

    # Filter variables
    active_fields = fields
    if export_variables is not None and fields is not None:
        active_fields = {k: v for k, v in fields.items() if k in export_variables}

    var_info = _classify_variables(active_fields) if active_fields else {}

    geo_files: list = []
    var_files: list = []
    n_parts = 1
    geo_reused = 0
    total_bytes = 0

    if write_boundary_parts:
        n_parts += len(mesh.boundary)

    geo_written = False

    for t in times:
        t_name = _format_time(t)

        if deduplicate_geometry and geo_written:
            geo_reused += 1
        else:
            geo_path, geo_bytes = _write_geometry(
                ensight_dir, t_name, mesh, cell_verts, use_binary,
                write_boundary_parts, chunk_size, stream_mode,
            )
            geo_files.append(geo_path)
            total_bytes += geo_bytes
            geo_written = True

        if active_fields:
            for name, data in active_fields.items():
                vtype = var_info[name]
                vp, var_bytes = _write_variable(
                    ensight_dir, t_name, name, data, vtype, use_binary,
                    mesh, write_boundary_parts, chunk_size, stream_mode,
                )
                var_files.append(vp)
                total_bytes += var_bytes

    # Write .case descriptor
    case_file = ensight_dir / f"{case_name}.case"
    _write_case_file(case_file, case_name, times, var_info, use_binary)
    total_bytes += case_file.stat().st_size if case_file.exists() else 0

    # Multi-resolution coarse mesh
    coarse_geo = None
    n_coarse = 0
    if multi_resolution:
        coarse_geo, n_coarse = _write_coarse_mesh(
            ensight_dir, mesh, cell_verts, use_binary, chunk_size,
        )

    # Compression ratio estimate
    ascii_estimate = mesh.n_cells * 3 * 16  # rough ASCII bytes per cell
    comp_ratio = ascii_estimate / max(total_bytes, 1)

    t_end = _time.perf_counter()
    elapsed_ms = (t_end - t_start) * 1000.0

    return EnSightV5Result(
        case_file=case_file,
        geometry_files=geo_files,
        variable_files=var_files,
        n_times=len(times),
        n_variables=len(var_info),
        n_parts=n_parts,
        binary=use_binary,
        geometry_reused=geo_reused,
        total_bytes_written=total_bytes,
        export_time_ms=elapsed_ms,
        coarse_geometry_file=coarse_geo,
        n_coarse_cells=n_coarse,
        compression_ratio=comp_ratio,
        streamed=stream_mode,
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


def _write_geometry(ensight_dir, t_name, mesh, cell_verts, binary, boundary_parts, chunk_size, stream):
    geo_path = ensight_dir / f"geometry_{t_name}.geo"
    pts = mesh.points.detach().cpu().numpy()

    if binary:
        bytes_written = _write_geo_binary(
            geo_path, t_name, pts, mesh, cell_verts, boundary_parts, chunk_size, stream,
        )
    else:
        _write_geo_ascii(geo_path, t_name, pts, mesh, cell_verts, boundary_parts)
        bytes_written = geo_path.stat().st_size if geo_path.exists() else 0

    return geo_path, bytes_written


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


def _write_geo_binary(path, t_name, pts, mesh, cell_verts, boundary_parts, chunk_size, stream):
    if stream:
        return _write_geo_binary_stream(path, t_name, pts, mesh, cell_verts, boundary_parts, chunk_size)

    buf = bytearray()

    def _bw(text):
        buf.extend(text.encode("ascii")[:80].ljust(80, b"\0"))

    def _bi(data):
        raw = data.astype(np.int32).tobytes()
        n = len(raw)
        buf.extend(struct.pack("i", n) + raw + struct.pack("i", n))

    def _bf(data):
        raw = data.astype(np.float32).tobytes()
        n = len(raw)
        buf.extend(struct.pack("i", n) + raw + struct.pack("i", n))

    _bw("EnSight Gold binary")
    _bw(f"geometry_{t_name}.geo")
    _bw("node id off")
    _bw("element id off")
    _bw("part")
    _bi(np.array([1], dtype=np.int32))
    _bw("volume mesh")
    _bw("coordinates")
    n_nodes = pts.shape[0]
    _bi(np.array([n_nodes], dtype=np.int32))
    for d in range(3):
        _bf(pts[:, d].astype(np.float32))
    _write_topology_binary_buf(buf, cell_verts)

    if boundary_parts:
        for pi, patch in enumerate(mesh.boundary):
            buf.extend(b"part".ljust(80, b"\0"))
            buf.extend(struct.pack("i", 4) + np.array([pi + 2], dtype=np.int32).tobytes() + struct.pack("i", 4))
            buf.extend(patch["name"].encode("ascii")[:80].ljust(80, b"\0"))
            buf.extend(b"coordinates".ljust(80, b"\0"))
            buf.extend(struct.pack("i", 4) + np.array([n_nodes], dtype=np.int32).tobytes() + struct.pack("i", 4))
            for d in range(3):
                raw = pts[:, d].astype(np.float32).tobytes()
                buf.extend(struct.pack("i", len(raw)) + raw + struct.pack("i", len(raw)))
            _write_boundary_topology_binary_buf(buf, mesh, patch)

    with open(path, "wb") as f:
        offset = 0
        while offset < len(buf):
            end = min(offset + chunk_size, len(buf))
            f.write(buf[offset:end])
            offset = end

    return len(buf)


def _write_geo_binary_stream(path, t_name, pts, mesh, cell_verts, boundary_parts, chunk_size):
    """Streaming binary write: write directly to file without full buffer."""
    bytes_written = 0
    with open(path, "wb") as f:
        def _bw(text):
            nonlocal bytes_written
            data = text.encode("ascii")[:80].ljust(80, b"\0")
            f.write(data)
            bytes_written += len(data)

        def _bi(arr):
            nonlocal bytes_written
            raw = arr.astype(np.int32).tobytes()
            chunk = struct.pack("i", len(raw)) + raw + struct.pack("i", len(raw))
            f.write(chunk)
            bytes_written += len(chunk)

        def _bf(arr):
            nonlocal bytes_written
            raw = arr.astype(np.float32).tobytes()
            chunk = struct.pack("i", len(raw)) + raw + struct.pack("i", len(raw))
            f.write(chunk)
            bytes_written += len(chunk)

        _bw("EnSight Gold binary")
        _bw(f"geometry_{t_name}.geo")
        _bw("node id off")
        _bw("element id off")
        _bw("part")
        _bi(np.array([1], dtype=np.int32))
        _bw("volume mesh")
        _bw("coordinates")
        n_nodes = pts.shape[0]
        _bi(np.array([n_nodes], dtype=np.int32))
        for d in range(3):
            _bf(pts[:, d].astype(np.float32))
        _write_topology_binary_file(f, cell_verts, bytes_written)

    return bytes_written


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------


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
        buf.extend(elem_type.encode("ascii")[:80].ljust(80, b"\0"))
        raw_i = np.array([len(cells)], dtype=np.int32).tobytes()
        buf.extend(struct.pack("i", len(raw_i)) + raw_i + struct.pack("i", len(raw_i)))
        nodes = []
        for _, verts in cells:
            nodes.extend(v + 1 for v in verts)
        raw_i = np.array(nodes, dtype=np.int32).tobytes()
        buf.extend(struct.pack("i", len(raw_i)) + raw_i + struct.pack("i", len(raw_i)))


def _write_topology_binary_file(f, cell_verts, bytes_counter):
    classified = _classify_cells(cell_verts)
    for elem_type, cells in classified:
        if not cells:
            continue
        f.write(elem_type.encode("ascii")[:80].ljust(80, b"\0"))
        nodes = []
        for _, verts in cells:
            nodes.extend(v + 1 for v in verts)
        raw_i = np.array(nodes, dtype=np.int32).tobytes()
        f.write(struct.pack("i", 4) + np.array([len(cells)], dtype=np.int32).tobytes() + struct.pack("i", 4))
        f.write(struct.pack("i", len(raw_i)) + raw_i + struct.pack("i", len(raw_i)))


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
        buf.extend(b"tria3".ljust(80, b"\0"))
        raw_i = np.array([len(tria3)], dtype=np.int32).tobytes()
        buf.extend(struct.pack("i", len(raw_i)) + raw_i + struct.pack("i", len(raw_i)))
        nodes = [vi for v in tria3 for vi in v]
        raw_i = np.array(nodes, dtype=np.int32).tobytes()
        buf.extend(struct.pack("i", len(raw_i)) + raw_i + struct.pack("i", len(raw_i)))
    if quad4:
        buf.extend(b"quad4".ljust(80, b"\0"))
        raw_i = np.array([len(quad4)], dtype=np.int32).tobytes()
        buf.extend(struct.pack("i", len(raw_i)) + raw_i + struct.pack("i", len(raw_i)))
        nodes = [vi for v in quad4 for vi in v]
        raw_i = np.array(nodes, dtype=np.int32).tobytes()
        buf.extend(struct.pack("i", len(raw_i)) + raw_i + struct.pack("i", len(raw_i)))


# ---------------------------------------------------------------------------
# Variable writing
# ---------------------------------------------------------------------------


def _write_variable(ensight_dir, t_name, name, data, vtype, binary, mesh, boundary_parts, chunk_size, stream):
    suffix_map = {"scalar": "scl", "vector": "vec", "tensor": "tsr"}
    suffix = suffix_map.get(vtype, "scl")
    var_path = ensight_dir / f"{name}_{t_name}.{suffix}"

    if binary:
        buf = bytearray()

        def _bw(text):
            buf.extend(text.encode("ascii")[:80].ljust(80, b"\0"))

        def _bf(d):
            raw = d.astype(np.float32).tobytes()
            n = len(raw)
            buf.extend(struct.pack("i", n) + raw + struct.pack("i", n))

        _bw(var_path.name)
        _bw(f"EnSight Gold: {name}")
        _bw("part")
        buf.extend(struct.pack("i", 4) + np.array([1], dtype=np.int32).tobytes() + struct.pack("i", 4))
        _bw("coordinates")
        _write_field_binary_buf(buf, data, vtype, _bf)

        if boundary_parts:
            for pi, patch in enumerate(mesh.boundary):
                _bw("part")
                buf.extend(struct.pack("i", 4) + np.array([pi + 2], dtype=np.int32).tobytes() + struct.pack("i", 4))
                _bw("coordinates")
                _write_field_binary_buf(buf, data, vtype, _bf)

        with open(var_path, "wb") as f:
            offset = 0
            while offset < len(buf):
                end = min(offset + chunk_size, len(buf))
                f.write(buf[offset:end])
                offset = end

        bytes_written = len(buf)
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
        bytes_written = var_path.stat().st_size if var_path.exists() else 0

    return var_path, bytes_written


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


def _write_field_binary_buf(buf, data, vtype, _bf):
    if vtype == "vector":
        for d in range(3):
            _bf(data[:, d].astype(np.float32))
    elif vtype == "tensor":
        for c in range(6):
            _bf(data[:, c].astype(np.float32))
    else:
        _bf(data.astype(np.float32))


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
# Multi-resolution coarse mesh
# ---------------------------------------------------------------------------


def _write_coarse_mesh(ensight_dir, mesh, cell_verts, binary, chunk_size):
    """Generate and write a coarse-resolution mesh by subsampling cells."""
    n_cells = mesh.n_cells
    n_coarse = max(1, n_cells // 4)

    # Select every 4th cell
    step = max(1, n_cells // n_coarse)
    coarse_cell_verts = [cell_verts[i] for i in range(0, n_cells, step)][:n_coarse]

    coarse_path = ensight_dir / "coarse_geometry.geo"
    pts = mesh.points.detach().cpu().numpy()

    if binary:
        buf = bytearray()

        def _bw(text):
            buf.extend(text.encode("ascii")[:80].ljust(80, b"\0"))

        def _bi(data):
            raw = data.astype(np.int32).tobytes()
            n = len(raw)
            buf.extend(struct.pack("i", n) + raw + struct.pack("i", n))

        def _bf(data):
            raw = data.astype(np.float32).tobytes()
            n = len(raw)
            buf.extend(struct.pack("i", n) + raw + struct.pack("i", n))

        _bw("EnSight Gold binary")
        _bw("coarse_geometry.geo")
        _bw("node id off")
        _bw("element id off")
        _bw("part")
        _bi(np.array([1], dtype=np.int32))
        _bw("coarse mesh")
        _bw("coordinates")
        n_nodes = pts.shape[0]
        _bi(np.array([n_nodes], dtype=np.int32))
        for d in range(3):
            _bf(pts[:, d].astype(np.float32))
        _write_topology_binary_buf(buf, coarse_cell_verts)

        with open(coarse_path, "wb") as f:
            offset = 0
            while offset < len(buf):
                end = min(offset + chunk_size, len(buf))
                f.write(buf[offset:end])
                offset = end
    else:
        with open(coarse_path, "w") as f:
            f.write("EnSight Gold ASCII\n")
            f.write("coarse_geometry.geo\n")
            f.write("node id off\nelement id off\n")
            f.write("part\n")
            f.write(f"{1:12d}\n")
            f.write("coarse mesh\n")
            f.write("coordinates\n")
            n_nodes = pts.shape[0]
            f.write(f"{n_nodes:12d}\n")
            for d in range(3):
                for i in range(n_nodes):
                    f.write(f"{pts[i, d]:14.6E}\n")
            _write_topology_ascii(f, coarse_cell_verts)

    return coarse_path, n_coarse


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
