"""
surfaceSplitByPatch — split a surface file into per-patch files.

Mirrors OpenFOAM's ``surfaceSplitByPatch`` utility.  Takes a surface
file that contains multiple ``solid`` blocks (e.g. a multi-solid STL)
and writes each block as a separate surface file.

For files without explicit patch information, the surface can be
pre-processed with :func:`surface_auto_patch` first.

Usage::

    from pyfoam.tools.surface_split_by_patch import surface_split_by_patch

    result = surface_split_by_patch("multi.stl", output_dir="patches/")
    print(result.output_files)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

__all__ = ["SurfaceSplitResult", "surface_split_by_patch"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SurfaceSplitResult:
    """Structured result from :func:`surface_split_by_patch`.

    Attributes
    ----------
    n_patches : int
        Number of patches extracted.
    patch_names : list[str]
        Names of each patch (from solid block names or numeric IDs).
    output_files : list[Path]
        Paths to the written per-patch surface files.
    patch_face_counts : list[int]
        Number of faces in each patch.
    """

    n_patches: int = 0
    patch_names: list[str] = field(default_factory=list)
    output_files: list = field(default_factory=list)
    patch_face_counts: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def surface_split_by_patch(
    surface_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    output_format: str = "stl",
) -> SurfaceSplitResult:
    """Split a multi-patch surface file into per-patch files.

    Parameters
    ----------
    surface_path : str or Path
        Path to the input surface file (STL with multiple solid blocks,
        or VTK with multiple POLYGONS sections).
    output_dir : str or Path, optional
        Directory for output files.  Defaults to
        ``<input_dir>/<input_stem>_patches/``.
    output_format : str
        Output surface format: ``"stl"`` (default), ``"obj"``, or ``"vtk"``.

    Returns
    -------
    SurfaceSplitResult
        Information about the extracted patches.

    Raises
    ------
    FileNotFoundError
        If *surface_path* does not exist.
    ValueError
        If the file contains no patches to split.
    """
    p = Path(surface_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Surface file not found: {p}")

    ext = p.suffix.lower()

    # Parse patches from file
    if ext == ".stl":
        patches = _parse_stl_patches(p)
    elif ext == ".obj":
        # OBJ files don't have native patch blocks — treat entire file as one
        patches = _parse_obj_as_single(p)
    elif ext == ".vtk":
        patches = _parse_vtk_as_single(p)
    else:
        raise ValueError(f"Unsupported input format: {ext}")

    if not patches:
        raise ValueError(f"No patches found in {p.name}")

    # Determine output directory
    if output_dir is None:
        out_dir = p.parent / f"{p.stem}_patches"
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = output_format.lower()
    ext_map = {"stl": ".stl", "obj": ".obj", "vtk": ".vtk"}
    if fmt not in ext_map:
        raise ValueError(f"Unsupported output format: {fmt}")
    out_ext = ext_map[fmt]

    result = SurfaceSplitResult()
    result.n_patches = len(patches)
    result.patch_names = []
    result.output_files = []
    result.patch_face_counts = []

    for name, verts, norms, facs in patches:
        safe_name = re.sub(r"[^\w\-.]", "_", name)
        out_path = out_dir / f"{safe_name}{out_ext}"
        _write_patch(out_path, fmt, verts, norms, facs)
        result.patch_names.append(name)
        result.output_files.append(out_path)
        result.patch_face_counts.append(facs.shape[0])

    return result


# ---------------------------------------------------------------------------
# STL multi-solid parser
# ---------------------------------------------------------------------------


def _parse_stl_patches(
    path: Path,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Parse an ASCII STL with multiple solid blocks.

    Returns list of (name, vertices, normals, faces) tuples.
    """
    text = path.read_text(encoding="utf-8", errors="replace")

    # Check if binary
    if not text.lstrip().startswith("solid") or "facet" not in text:
        # Binary STL — single patch
        from pyfoam.tools.surface_convert import _rstlb

        verts, norms, facs = _rstlb(path)
        return [(path.stem, verts, norms, facs)]

    # Parse multi-solid ASCII STL
    solid_pattern = re.compile(
        r"solid\s+(\S.*?)\s*\n(.*?)endsolid", re.DOTALL | re.MULTILINE
    )
    facet_pattern = re.compile(
        r"facet\s+normal\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)"
        r"\s+outer\s+loop(.*?)endfacet",
        re.DOTALL,
    )
    vertex_pattern = re.compile(
        r"vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)"
    )

    patches = []
    for m in solid_pattern.finditer(text):
        name = m.group(1).strip()
        body = m.group(2)

        vl, nl, fl = [], [], []
        for fm in facet_pattern.finditer(body):
            nl.append([float(fm.group(i)) for i in (1, 2, 3)])
            verts_match = vertex_pattern.findall(fm.group(4))
            if len(verts_match) != 3:
                continue
            fi = []
            for vx, vy, vz in verts_match:
                idx = len(vl)
                vl.append([float(vx), float(vy), float(vz)])
                fi.append(idx)
            fl.append(fi)

        if not fl:
            continue

        verts = np.array(vl, dtype=np.float64)
        norms = np.array(nl, dtype=np.float64)
        facs = np.array(fl, dtype=np.int32)
        patches.append((name, verts, norms, facs))

    return patches


def _parse_obj_as_single(
    path: Path,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Parse an OBJ file as a single patch."""
    from pyfoam.tools.surface_convert import _robj

    verts, norms, facs = _robj(path)
    return [(path.stem, verts, norms, facs)]


def _parse_vtk_as_single(
    path: Path,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Parse a VTK file as a single patch."""
    from pyfoam.tools.surface_convert import _rvtk

    verts, norms, facs = _rvtk(path)
    return [(path.stem, verts, norms, facs)]


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------


def _write_patch(
    path: Path,
    fmt: str,
    verts: np.ndarray,
    norms: np.ndarray,
    facs: np.ndarray,
) -> None:
    """Write a single patch to a surface file."""
    from pyfoam.tools.surface_convert import _wstl, _wobj, _wvtk

    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "stl":
        _wstl(path, verts, norms, facs)
    elif fmt == "obj":
        _wobj(path, verts, norms, facs)
    elif fmt == "vtk":
        _wvtk(path, verts, norms, facs)
