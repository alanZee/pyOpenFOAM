"""
surfaceConvert enhanced v3 — enhanced surface format conversion with 3MF
support and normal smoothing for better quality preservation (third generation).

Extends :func:`surface_convert_enhanced_2` with:

- **3MF read support**: Read 3D Manufacturing Format files (XML-based).
- **Normal smoothing**: Laplacian smoothing of vertex normals to reduce
  artefacts in converted meshes.
- **Conversion diagnostics**: Reports point deduplication ratio, normal
  change statistics, and manifold checks.

Usage::

    from pyfoam.tools.surface_convert_enhanced_3 import surface_convert_enhanced_3

    result = surface_convert_enhanced_3(
        "input.stl", "output.ply",
        deduplicate_points=True,
        smooth_normals=True,
    )
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

__all__ = ["ConvertEnhanced3Result", "surface_convert_enhanced_3"]

_FMTS = {"stl", "obj", "vtk", "ply", "off", "3mf"}


@dataclass
class ConvertEnhanced3Result:
    """Result from :func:`surface_convert_enhanced_3`.

    Attributes
    ----------
    output_path : Path
    n_vertices, n_faces : int
    mean_aspect_ratio : float
    n_degenerate : int
    dedup_ratio : float
        Fraction of points removed by deduplication.
    n_non_manifold_edges : int
    mean_normal_change : float
        Mean angle (degrees) between original and smoothed normals.
    """

    output_path: Path = Path(".")
    n_vertices: int = 0
    n_faces: int = 0
    mean_aspect_ratio: float = 0.0
    n_degenerate: int = 0
    dedup_ratio: float = 0.0
    n_non_manifold_edges: int = 0
    mean_normal_change: float = 0.0


def surface_convert_enhanced_3(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    output_format: Optional[str] = None,
    deduplicate_points: bool = False,
    deduplicate_tol: float = 1e-10,
    recompute_normals: bool = False,
    smooth_normals: bool = False,
    smooth_iterations: int = 1,
    scale: float = 1.0,
    translate: Optional[Tuple[float, float, float]] = None,
    rotate_axis: Optional[Tuple[float, float, float]] = None,
    rotate_angle: float = 0.0,
    quality_report: bool = False,
) -> ConvertEnhanced3Result:
    """Convert surface mesh files with 3MF support and normal smoothing.

    Parameters
    ----------
    input_path, output_path : str or Path
    output_format : str, optional
    deduplicate_points : bool
    deduplicate_tol : float
    recompute_normals : bool
    smooth_normals : bool
        Apply Laplacian smoothing to vertex normals.
    smooth_iterations : int
        Number of normal smoothing passes.
    scale, translate, rotate_axis, rotate_angle
        Coordinate transforms.
    quality_report : bool

    Returns
    -------
    ConvertEnhanced3Result
        Conversion metadata with diagnostics.
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
    n_orig = v.shape[0]

    # Coordinate transforms
    if scale != 1.0:
        v = v * scale
    if translate is not None:
        v = v + np.asarray(translate, dtype=np.float64)
    if rotate_axis is not None and rotate_angle != 0.0:
        v = _rotate_points(v, rotate_axis, rotate_angle)

    if deduplicate_points:
        v, f = _deduplicate(v, f, deduplicate_tol)

    if recompute_normals:
        n = _compute_normals(v, f)

    # Normal smoothing
    mean_normal_change = 0.0
    if smooth_normals and n.shape[0] > 0:
        n_orig_normals = n.copy()
        for _ in range(smooth_iterations):
            n = _smooth_normals(v, f, n)
        # Compute mean change
        dots = np.clip(np.sum(n * n_orig_normals, axis=1), -1.0, 1.0)
        changes = np.degrees(np.arccos(dots))
        mean_normal_change = float(np.mean(changes))

    op.parent.mkdir(parents=True, exist_ok=True)
    if outf == "3mf":
        _write_3mf(op, v, f)
    else:
        _write_surface(op, outf, v, n, f)

    dedup_ratio = (n_orig - v.shape[0]) / n_orig if n_orig > 0 else 0.0

    result = ConvertEnhanced3Result(
        output_path=op,
        n_vertices=v.shape[0],
        n_faces=f.shape[0],
        dedup_ratio=dedup_ratio,
        mean_normal_change=mean_normal_change,
    )

    if quality_report and f.shape[0] > 0:
        ar = _compute_aspect_ratios(v, f)
        result.mean_aspect_ratio = float(np.mean(ar))
        areas = _compute_areas(v, f)
        result.n_degenerate = int((areas < 1e-30).sum())
        result.n_non_manifold_edges = _count_non_manifold_edges(f)

    return result


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------


def _rotate_points(
    verts: np.ndarray,
    axis: Tuple[float, float, float],
    angle_deg: float,
) -> np.ndarray:
    ax = np.asarray(axis, dtype=np.float64)
    ax_norm = np.linalg.norm(ax)
    if ax_norm < 1e-30:
        return verts
    ax = ax / ax_norm

    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    K = np.array([
        [0, -ax[2], ax[1]],
        [ax[2], 0, -ax[0]],
        [-ax[1], ax[0], 0],
    ], dtype=np.float64)
    R = cos_t * np.eye(3) + sin_t * K + (1 - cos_t) * np.outer(ax, ax)

    return (R @ verts.T).T


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def _detect_format(p: Path) -> str:
    ext = p.suffix.lower()
    mapping = {
        ".stl": "stl", ".obj": "obj", ".vtk": "vtk",
        ".ply": "ply", ".off": "off", ".3mf": "3mf",
    }
    if ext in mapping:
        return mapping[ext]
    raise ValueError(f"Cannot determine format from extension '{ext}'")


# ---------------------------------------------------------------------------
# Read dispatch
# ---------------------------------------------------------------------------


def _read_surface(p: Path, fmt: str):
    if fmt == "stl":
        return _read_stl(p)
    elif fmt == "3mf":
        return _read_3mf(p)
    elif fmt in ("obj", "vtk"):
        from pyfoam.tools.surface_convert import _rs
        return _rs(p, fmt)
    elif fmt == "ply":
        return _read_ply(p)
    elif fmt == "off":
        return _read_off(p)
    raise ValueError(f"Unsupported format: {fmt!r}")


def _read_stl(p: Path):
    """Read STL file (binary or ASCII)."""
    raw = p.read_bytes()
    if raw[:5] == b"solid" and b"facet" in raw[:1000]:
        from pyfoam.tools.surface_convert import _rs
        return _rs(p, "stl")
    return _read_stl_binary(p)


def _read_stl_binary(p: Path):
    raw = p.read_bytes()
    if len(raw) < 84:
        raise ValueError("Binary STL too short")
    n_tri = struct.unpack_from("<I", raw, 80)[0]
    vl, fl = [], []
    offset = 84
    for i in range(n_tri):
        if offset + 50 > len(raw):
            break
        offset += 12  # skip normal
        base = len(vl)
        for _ in range(3):
            x, y, z = struct.unpack_from("<fff", raw, offset)
            vl.append([x, y, z])
            offset += 12
        fl.append([base, base + 1, base + 2])
        offset += 2

    v = np.array(vl, dtype=np.float64) if vl else np.empty((0, 3), dtype=np.float64)
    n = np.empty((0, 3), dtype=np.float64)
    f = np.array(fl, dtype=np.int32) if fl else np.empty((0, 3), dtype=np.int32)
    return v, n, f


def _read_3mf(p: Path):
    """Read 3MF file (XML-based)."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(p))
    root = tree.getroot()

    # Namespace handling
    ns = ""
    for elem in root.iter():
        if elem.tag.startswith("{"):
            ns = elem.tag.split("}")[0] + "}"
            break

    vl: list = []
    fl_list: list = []

    for mesh_elem in root.iter(f"{ns}mesh"):
        verts_elem = mesh_elem.find(f"{ns}vertices")
        tris_elem = mesh_elem.find(f"{ns}triangles")
        if verts_elem is None or tris_elem is None:
            continue

        base = len(vl)
        for v_elem in verts_elem.iter(f"{ns}vertex"):
            x = float(v_elem.get("x", "0"))
            y = float(v_elem.get("y", "0"))
            z = float(v_elem.get("z", "0"))
            vl.append([x, y, z])

        for t_elem in tris_elem.iter(f"{ns}triangle"):
            v0 = int(t_elem.get("v1", "0")) + base
            v1 = int(t_elem.get("v2", "0")) + base
            v2 = int(t_elem.get("v3", "0")) + base
            fl_list.append([v0, v1, v2])

    v = np.array(vl, dtype=np.float64) if vl else np.empty((0, 3), dtype=np.float64)
    n = np.empty((0, 3), dtype=np.float64)
    f = np.array(fl_list, dtype=np.int32) if fl_list else np.empty((0, 3), dtype=np.int32)
    return v, n, f


def _read_ply(p: Path):
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    n_verts, n_faces = 0, 0
    fmt_str = "ascii"
    header_end = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("element vertex"):
            n_verts = int(stripped.split()[-1])
        elif stripped.startswith("element face"):
            n_faces = int(stripped.split()[-1])
        elif stripped.startswith("format"):
            fmt_str = stripped.split()[1]
        elif stripped == "end_header":
            header_end = i + 1
            break

    vl, fl = [], []
    if fmt_str.startswith("ascii"):
        idx = header_end
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


def _read_off(p: Path):
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    idx = 0
    while idx < len(lines) and lines[idx].strip().startswith("#"):
        idx += 1

    header = lines[idx].strip().split()
    idx += 1

    if header[0] == "OFF":
        n_verts, n_faces = int(header[1]), int(header[2])
    else:
        n_verts, n_faces = int(header[0]), int(header[1])

    vl = []
    for _ in range(n_verts):
        tk = lines[idx].strip().split()
        idx += 1
        vl.append([float(tk[0]), float(tk[1]), float(tk[2])])

    fl_list = []
    for _ in range(n_faces):
        tk = lines[idx].strip().split()
        idx += 1
        nv = int(tk[0])
        fi = [int(tk[j + 1]) for j in range(nv)]
        for k in range(1, len(fi) - 1):
            fl_list.append([fi[0], fi[k], fi[k + 1]])

    v = np.array(vl, dtype=np.float64) if vl else np.empty((0, 3), dtype=np.float64)
    n = np.empty((0, 3), dtype=np.float64)
    f = np.array(fl_list, dtype=np.int32) if fl_list else np.empty((0, 3), dtype=np.int32)
    return v, n, f


# ---------------------------------------------------------------------------
# Write dispatch
# ---------------------------------------------------------------------------


def _write_surface(p: Path, fmt: str, v, n, f):
    if fmt in ("stl", "obj", "vtk"):
        from pyfoam.tools.surface_convert import _ws
        _ws(p, fmt, v, n, f)
    elif fmt == "ply":
        _write_ply(p, v, n, f)
    elif fmt == "off":
        _write_off(p, v, n, f)
    elif fmt == "3mf":
        _write_3mf(p, v, f)


def _write_3mf(p: Path, v, f):
    """Write 3MF file (XML-based)."""
    p.parent.mkdir(parents=True, exist_ok=True)
    nv, nf = v.shape[0], f.shape[0]
    with open(p, "w", encoding="utf-8") as fo:
        fo.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        fo.write('<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n')
        fo.write('  <resources>\n')
        fo.write('    <object id="1" type="model">\n')
        fo.write('      <mesh>\n')
        fo.write(f'        <vertices>\n')
        for i in range(nv):
            fo.write(f'          <vertex x="{v[i,0]:.10e}" y="{v[i,1]:.10e}" z="{v[i,2]:.10e}"/>\n')
        fo.write(f'        </vertices>\n')
        fo.write(f'        <triangles>\n')
        for fi in range(nf):
            fo.write(f'          <triangle v1="{f[fi,0]}" v2="{f[fi,1]}" v3="{f[fi,2]}"/>\n')
        fo.write(f'        </triangles>\n')
        fo.write('      </mesh>\n')
        fo.write('    </object>\n')
        fo.write('  </resources>\n')
        fo.write('  <build>\n')
        fo.write('    <item objectid="1"/>\n')
        fo.write('  </build>\n')
        fo.write('</model>\n')


def _write_ply(p: Path, v, n, f):
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


def _write_off(p: Path, v, n, f):
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
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    safe = np.where(norms > 1e-30, norms, 1.0)
    return cross / safe


def _compute_aspect_ratios(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    longest = np.maximum(np.maximum(e0, e1), e2)
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    safe_longest = np.where(longest > 1e-30, longest, 1.0)
    shortest_alt = 2.0 * areas / safe_longest
    safe_alt = np.where(shortest_alt > 1e-30, shortest_alt, 1e-30)
    return longest / safe_alt


def _compute_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def _deduplicate(verts: np.ndarray, faces: np.ndarray, tol: float):
    n = verts.shape[0]
    if n == 0:
        return verts, faces

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

    unique_indices, inverse = np.unique(mapping, return_inverse=True)
    new_verts = verts[unique_indices]
    new_faces = inverse[faces].astype(np.int32)
    return new_verts, new_faces


def _smooth_normals(verts: np.ndarray, faces: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Laplace-smooth face normals by averaging over face neighbours."""
    n_faces = faces.shape[0]
    # Build face adjacency via shared edges
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi in range(n_faces):
        tri = faces[fi]
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            key = (min(int(tri[a]), int(tri[b])), max(int(tri[a]), int(tri[b])))
            edge_faces.setdefault(key, []).append(fi)

    new_normals = normals.copy()
    for fi in range(n_faces):
        tri = faces[fi]
        neighbours: list[int] = []
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            key = (min(int(tri[a]), int(tri[b])), max(int(tri[a]), int(tri[b])))
            for adj_fi in edge_faces.get(key, []):
                if adj_fi != fi:
                    neighbours.append(adj_fi)
        if neighbours:
            avg = normals[fi].copy()
            for ni in neighbours:
                avg += normals[ni]
            avg /= (len(neighbours) + 1)
            norm_mag = np.linalg.norm(avg)
            if norm_mag > 1e-30:
                new_normals[fi] = avg / norm_mag

    return new_normals


def _count_non_manifold_edges(faces: np.ndarray) -> int:
    """Count edges shared by more than 2 faces."""
    edge_faces: dict[tuple[int, int], int] = {}
    for fi in range(faces.shape[0]):
        tri = faces[fi]
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            key = (min(int(tri[a]), int(tri[b])), max(int(tri[a]), int(tri[b])))
            edge_faces[key] = edge_faces.get(key, 0) + 1
    return sum(1 for count in edge_faces.values() if count > 2)
