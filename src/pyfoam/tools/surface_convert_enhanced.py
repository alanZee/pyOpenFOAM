"""
surfaceConvert enhanced — enhanced surface format conversion with more
format support and better quality preservation.

Extends :func:`surface_convert` with:

- **PLY format**: Read and write Stanford PLY files (ASCII and binary).
- **OFF format**: Read and write Object File Format.
- **Point deduplication**: Optional merging of coincident vertices
  during conversion.
- **Normal recomputation**: Recompute normals from geometry instead of
  copying potentially inconsistent normals from the source.

Usage::

    from pyfoam.tools.surface_convert_enhanced import surface_convert_enhanced

    output = surface_convert_enhanced(
        "input.stl", "output.ply",
        deduplicate_points=True,
        recompute_normals=True,
    )
"""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Optional, Union

import numpy as np

__all__ = ["surface_convert_enhanced"]

# Supported formats (superset of base)
_FMTS = {"stl", "obj", "vtk", "ply", "off"}


def surface_convert_enhanced(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    output_format: Optional[str] = None,
    deduplicate_points: bool = False,
    deduplicate_tol: float = 1e-10,
    recompute_normals: bool = False,
) -> Path:
    """Convert surface mesh files with enhanced options.

    Parameters
    ----------
    input_path : str or Path
        Input surface file.
    output_path : str or Path
        Output surface file.
    output_format : str, optional
        Target format (``"stl"``, ``"obj"``, ``"vtk"``, ``"ply"``,
        ``"off"``).  Inferred from output extension when not given.
    deduplicate_points : bool
        If True, merge coincident vertices.
    deduplicate_tol : float
        Tolerance for point deduplication.
    recompute_normals : bool
        If True, recompute face normals from geometry.

    Returns
    -------
    Path
        Path to the output file.
    """
    ip = Path(input_path).resolve()
    if not ip.is_file():
        raise FileNotFoundError(f"Input file not found: {ip}")

    op = Path(output_path).resolve()
    inf = _detect_format(ip)
    outf = output_format.lower() if output_format else _detect_format(op)

    if outf not in _FMTS:
        raise ValueError(f"Unsupported format: {outf!r}. Supported: {sorted(_FMTS)}")

    v, n, f = _read_surface(ip, inf)

    # Optional deduplication
    if deduplicate_points:
        v, f = _deduplicate(v, f, deduplicate_tol)

    # Optional normal recomputation
    if recompute_normals:
        n = _compute_normals(v, f)

    op.parent.mkdir(parents=True, exist_ok=True)
    _write_surface(op, outf, v, n, f)
    return op


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def _detect_format(p: Path) -> str:
    """Detect surface format from file extension."""
    ext = p.suffix.lower()
    mapping = {
        ".stl": "stl", ".obj": "obj", ".vtk": "vtk",
        ".ply": "ply", ".off": "off",
    }
    if ext in mapping:
        return mapping[ext]
    raise ValueError(f"Cannot determine format from extension '{ext}'")


# ---------------------------------------------------------------------------
# Read dispatch
# ---------------------------------------------------------------------------


def _read_surface(p: Path, fmt: str):
    """Read a surface file and return (vertices, normals, faces)."""
    if fmt == "stl":
        return _read_stl(p)
    elif fmt == "obj":
        return _read_obj(p)
    elif fmt == "vtk":
        return _read_vtk(p)
    elif fmt == "ply":
        return _read_ply(p)
    elif fmt == "off":
        return _read_off(p)
    raise ValueError(f"Unsupported format: {fmt!r}")


# ---------------------------------------------------------------------------
# STL reader (reuses base)
# ---------------------------------------------------------------------------


def _read_stl(p: Path):
    """Read STL file (ASCII or binary)."""
    from pyfoam.tools.surface_convert import _rs
    return _rs(p, "stl")


# ---------------------------------------------------------------------------
# OBJ reader
# ---------------------------------------------------------------------------


def _read_obj(p: Path):
    """Read OBJ file."""
    from pyfoam.tools.surface_convert import _rs
    return _rs(p, "obj")


# ---------------------------------------------------------------------------
# VTK reader
# ---------------------------------------------------------------------------


def _read_vtk(p: Path):
    """Read VTK file."""
    from pyfoam.tools.surface_convert import _rs
    return _rs(p, "vtk")


# ---------------------------------------------------------------------------
# PLY reader
# ---------------------------------------------------------------------------


def _read_ply(p: Path):
    """Read PLY file (ASCII or binary little-endian)."""
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # Parse header
    n_verts = 0
    n_faces = 0
    fmt = "ascii"
    header_end = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("element vertex"):
            n_verts = int(stripped.split()[-1])
        elif stripped.startswith("element face"):
            n_faces = int(stripped.split()[-1])
        elif stripped.startswith("format"):
            fmt = stripped.split()[1]
        elif stripped == "end_header":
            header_end = i + 1
            break

    if fmt.startswith("ascii"):
        return _read_ply_ascii(lines, header_end, n_verts, n_faces)
    else:
        return _read_ply_binary(p, header_end, n_verts, n_faces)


def _read_ply_ascii(lines, start, n_verts, n_faces):
    """Read ASCII PLY data."""
    vl, nl, fl = [], [], []
    idx = start

    for _ in range(n_verts):
        if idx >= len(lines):
            break
        tk = lines[idx].strip().split()
        idx += 1
        if len(tk) >= 3:
            vl.append([float(tk[0]), float(tk[1]), float(tk[2])])

    for _ in range(n_faces):
        if idx >= len(lines):
            break
        tk = lines[idx].strip().split()
        idx += 1
        if len(tk) >= 4:
            nv = int(tk[0])
            fi = [int(tk[j + 1]) for j in range(nv)]
            for k in range(1, len(fi) - 1):
                fl.append([fi[0], fi[k], fi[k + 1]])

    v = np.array(vl, dtype=np.float64) if vl else np.empty((0, 3), dtype=np.float64)
    n = np.empty((0, 3), dtype=np.float64)
    f = np.array(fl, dtype=np.int32) if fl else np.empty((0, 3), dtype=np.int32)
    return v, n, f


def _read_ply_binary(p: Path, header_end_approx: int, n_verts: int, n_faces: int):
    """Read binary little-endian PLY."""
    raw = p.read_bytes()
    # Find end_header in raw bytes
    hdr_end = raw.find(b"end_header")
    if hdr_end < 0:
        raise ValueError("Invalid PLY: no end_header found")
    data_start = raw.index(b"\n", hdr_end) + 1

    offset = data_start
    vl = []
    for _ in range(n_verts):
        x, y, z = struct.unpack_from("<ddd", raw, offset)
        offset += 24
        vl.append([x, y, z])
        # Skip extra vertex properties (assume ~32 bytes per vertex if present)
        # For simplicity, only read x, y, z

    fl = []
    for _ in range(n_faces):
        nv = struct.unpack_from("<B", raw, offset)[0]
        offset += 1
        indices = []
        for _ in range(nv):
            idx = struct.unpack_from("<i", raw, offset)[0]
            offset += 4
            indices.append(idx)
        for k in range(1, len(indices) - 1):
            fl.append([indices[0], indices[k], indices[k + 1]])

    v = np.array(vl, dtype=np.float64) if vl else np.empty((0, 3), dtype=np.float64)
    n = np.empty((0, 3), dtype=np.float64)
    f = np.array(fl, dtype=np.int32) if fl else np.empty((0, 3), dtype=np.int32)
    return v, n, f


# ---------------------------------------------------------------------------
# OFF reader
# ---------------------------------------------------------------------------


def _read_off(p: Path):
    """Read OFF file."""
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    idx = 0

    # Skip comments and find header
    while idx < len(lines) and lines[idx].strip().startswith("#"):
        idx += 1

    header = lines[idx].strip().split()
    idx += 1

    if header[0] == "OFF":
        n_verts, n_faces, _n_edges = int(header[1]), int(header[2]), int(header[3])
    else:
        n_verts, n_faces, _n_edges = int(header[0]), int(header[1]), int(header[2])

    vl = []
    for _ in range(n_verts):
        tk = lines[idx].strip().split()
        idx += 1
        vl.append([float(tk[0]), float(tk[1]), float(tk[2])])

    fl = []
    for _ in range(n_faces):
        tk = lines[idx].strip().split()
        idx += 1
        nv = int(tk[0])
        fi = [int(tk[j + 1]) for j in range(nv)]
        for k in range(1, len(fi) - 1):
            fl.append([fi[0], fi[k], fi[k + 1]])

    v = np.array(vl, dtype=np.float64) if vl else np.empty((0, 3), dtype=np.float64)
    n = np.empty((0, 3), dtype=np.float64)
    f = np.array(fl, dtype=np.int32) if fl else np.empty((0, 3), dtype=np.int32)
    return v, n, f


# ---------------------------------------------------------------------------
# Write dispatch
# ---------------------------------------------------------------------------


def _write_surface(p: Path, fmt: str, v, n, f):
    """Write a surface file."""
    if fmt == "stl":
        _write_stl(p, v, n, f)
    elif fmt == "obj":
        _write_obj(p, v, n, f)
    elif fmt == "vtk":
        _write_vtk(p, v, n, f)
    elif fmt == "ply":
        _write_ply(p, v, n, f)
    elif fmt == "off":
        _write_off(p, v, n, f)


def _write_stl(p, v, n, f):
    """Write ASCII STL."""
    from pyfoam.tools.surface_convert import _ws
    _ws(p, "stl", v, n, f)


def _write_obj(p, v, n, f):
    """Write OBJ."""
    from pyfoam.tools.surface_convert import _ws
    _ws(p, "obj", v, n, f)


def _write_vtk(p, v, n, f):
    """Write VTK."""
    from pyfoam.tools.surface_convert import _ws
    _ws(p, "vtk", v, n, f)


def _write_ply(p: Path, v: np.ndarray, n: np.ndarray, f: np.ndarray):
    """Write ASCII PLY."""
    nv, nf = v.shape[0], f.shape[0]
    with open(p, "w") as fo:
        fo.write("ply\nformat ascii 1.0\n")
        fo.write(f"element vertex {nv}\n")
        fo.write("property double x\nproperty double y\nproperty double z\n")
        fo.write(f"element face {nf}\n")
        fo.write("property list uchar int vertex_indices\n")
        fo.write("end_header\n")
        for i in range(nv):
            fo.write(f"{v[i, 0]:.10e} {v[i, 1]:.10e} {v[i, 2]:.10e}\n")
        for fi in range(nf):
            fo.write(f"3 {f[fi, 0]} {f[fi, 1]} {f[fi, 2]}\n")


def _write_off(p: Path, v: np.ndarray, n: np.ndarray, f: np.ndarray):
    """Write OFF."""
    nv, nf = v.shape[0], f.shape[0]
    with open(p, "w") as fo:
        fo.write("OFF\n")
        fo.write(f"{nv} {nf} 0\n")
        for i in range(nv):
            fo.write(f"{v[i, 0]:.10e} {v[i, 1]:.10e} {v[i, 2]:.10e}\n")
        for fi in range(nf):
            fo.write(f"3 {f[fi, 0]} {f[fi, 1]} {f[fi, 2]}\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-face unit normals."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    safe = np.where(norms > 1e-30, norms, 1.0)
    return cross / safe


def _deduplicate(
    verts: np.ndarray, faces: np.ndarray, tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge coincident vertices and remap face indices."""
    n = verts.shape[0]
    if n == 0:
        return verts, faces

    # Simple hash-based deduplication
    mapping = np.arange(n, dtype=np.int32)
    cell_size = max(tol * 2, 1e-12)
    hash_table: dict[tuple, int] = {}

    for i in range(n):
        gx = int(np.floor(verts[i, 0] / cell_size))
        gy = int(np.floor(verts[i, 1] / cell_size))
        gz = int(np.floor(verts[i, 2] / cell_size))

        found = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    key = (gx + dx, gy + dy, gz + dz)
                    if key in hash_table:
                        j = hash_table[key]
                        if np.linalg.norm(verts[i] - verts[j]) < tol:
                            mapping[i] = mapping[j]
                            found = True
                            break
                if found:
                    break
            if found:
                break

        if not found:
            hash_table[(gx, gy, gz)] = i

    # Build unique vertex list
    unique_indices, inverse = np.unique(mapping, return_inverse=True)
    new_verts = verts[unique_indices]

    # Remap faces
    new_faces = inverse[faces].astype(np.int32)

    return new_verts, new_faces
