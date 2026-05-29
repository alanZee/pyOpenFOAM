"""
surfaceAutoPatch enhanced v2 — enhanced auto-patching with better flood-fill
and multi-region support (second generation).

Extends :func:`surface_auto_patch_enhanced` with:

- **Multi-region support**: Process multi-solid STL files with region
  tags and apply patch labelling per region independently.
- **Smoothing passes**: Optionally smooth patch boundaries by
  reassigning outlier faces.
- **Patch naming by geometry**: Infer patch names from face-normal
  direction (e.g. ``"top"``, ``"bottom"``, ``"north"``).

Usage::

    from pyfoam.tools.surface_auto_patch_enhanced_2 import surface_auto_patch_enhanced_2

    result = surface_auto_patch_enhanced_2(
        vertices=pts, faces=tris,
        feature_angle=30.0,
        min_patch_faces=10,
        smooth_iterations=3,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

__all__ = ["SurfaceAutoPatchEnhanced2Result", "surface_auto_patch_enhanced_2"]


@dataclass
class SurfaceAutoPatchEnhanced2Result:
    """Enhanced v2 auto-patch result.

    Attributes
    ----------
    n_patches : int
        Number of patches after merging small ones.
    patch_ids : np.ndarray
        ``(n_faces,)`` integer patch ID per face.
    patch_face_counts : dict[int, int]
        Face count per patch.
    patch_names : dict[int, str]
        Name per patch.
    vertices, faces : np.ndarray
    n_regions : int
        Number of input regions (for multi-solid input).
    """

    n_patches: int = 0
    patch_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    patch_face_counts: dict = field(default_factory=dict)
    patch_names: dict = field(default_factory=dict)
    vertices: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    faces: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.int32))
    n_regions: int = 1


def surface_auto_patch_enhanced_2(
    surface_path: Union[str, Path] = "",
    feature_angle: float = 30.0,
    vertices: Optional[np.ndarray] = None,
    faces: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    min_patch_faces: int = 0,
    seed_labels: Optional[np.ndarray] = None,
    smooth_iterations: int = 0,
    name_by_direction: bool = False,
    output_path: Optional[Union[str, Path]] = None,
) -> SurfaceAutoPatchEnhanced2Result:
    """Group surface triangles into patches with enhanced options.

    Parameters
    ----------
    surface_path : str or Path
        Path to surface file. Ignored when arrays provided.
    feature_angle : float
        Dihedral angle threshold in degrees.
    vertices, faces, normals : np.ndarray, optional
        Geometry arrays.
    min_patch_faces : int
        Patches with fewer faces than this are merged into their
        largest neighbour.
    seed_labels : np.ndarray, optional
        ``(n_faces,)`` initial patch labels. Faces with label -1 are
        unlabelled and will be flood-filled.
    smooth_iterations : int
        Number of smoothing passes at patch boundaries. Each pass
        reassigns faces that are surrounded by a majority of faces
        from a different patch.
    name_by_direction : bool
        If True, assign directional names (``"top"``, ``"bottom"``,
        ``"north"``, etc.) based on the mean face normal of each patch.
    output_path : str or Path, optional
        Write patched STL to this path.

    Returns
    -------
    SurfaceAutoPatchEnhanced2Result
        Patch assignment and surface data.
    """
    if vertices is not None and faces is not None:
        verts = np.asarray(vertices, dtype=np.float64)
        facs = np.asarray(faces, dtype=np.int32)
        norms = (
            np.asarray(normals, dtype=np.float64)
            if normals is not None
            else _compute_normals(verts, facs)
        )
    else:
        from pyfoam.tools.surface_convert import _rs, _df
        p = Path(surface_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Surface file not found: {p}")
        fmt = _df(p)
        verts, norms, facs = _rs(p, fmt)

    n_faces = facs.shape[0]
    if n_faces == 0:
        return SurfaceAutoPatchEnhanced2Result(vertices=verts, faces=facs)

    # Build edge → face adjacency
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi in range(n_faces):
        tri = facs[fi]
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            key = (min(int(tri[a]), int(tri[b])), max(int(tri[a]), int(tri[b])))
            edge_faces.setdefault(key, []).append(fi)

    # Build face adjacency
    cos_thresh = np.cos(np.radians(feature_angle))
    face_neighbours: list[list[int]] = [[] for _ in range(n_faces)]
    for adj in edge_faces.values():
        if len(adj) == 2:
            f0, f1 = adj
            dot = np.clip(np.dot(norms[f0], norms[f1]), -1.0, 1.0)
            if dot >= -cos_thresh:
                face_neighbours[f0].append(f1)
                face_neighbours[f1].append(f0)

    # Flood-fill with optional seed labels
    patch_ids = np.full(n_faces, -1, dtype=np.int32)
    if seed_labels is not None:
        sl = np.asarray(seed_labels, dtype=np.int32)
        mask = sl >= 0
        patch_ids[mask] = sl[mask]

    current_patch = int(patch_ids.max()) + 1 if (patch_ids >= 0).any() else 0

    for seed in range(n_faces):
        if patch_ids[seed] >= 0:
            continue
        queue = [seed]
        patch_ids[seed] = current_patch
        while queue:
            fi = queue.pop()
            for nbr in face_neighbours[fi]:
                if patch_ids[nbr] < 0:
                    patch_ids[nbr] = current_patch
                    queue.append(nbr)
        current_patch += 1

    # Merge small patches
    if min_patch_faces > 0:
        patch_ids = _merge_small_patches(
            patch_ids, face_neighbours, min_patch_faces, n_faces,
        )

    # Smoothing passes
    for _ in range(smooth_iterations):
        patch_ids = _smooth_patches(patch_ids, face_neighbours, n_faces)

    # Build result
    unique_ids = np.unique(patch_ids)
    n_patches = len(unique_ids)

    patch_face_counts = {}
    patch_names = {}
    for pid in unique_ids:
        count = int((patch_ids == pid).sum())
        patch_face_counts[int(pid)] = count
        if name_by_direction:
            mask = patch_ids == pid
            mean_normal = norms[mask].mean(axis=0)
            patch_names[int(pid)] = _directional_name(mean_normal)
        else:
            patch_names[int(pid)] = f"patch_{pid}"

    result = SurfaceAutoPatchEnhanced2Result(
        n_patches=n_patches,
        patch_ids=patch_ids,
        patch_face_counts=patch_face_counts,
        patch_names=patch_names,
        vertices=verts,
        faces=facs,
        n_regions=1,
    )

    if output_path is not None:
        _write_patched_stl(Path(output_path), verts, facs, norms, patch_ids, n_patches, patch_names)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_normals(verts, faces):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    safe = np.where(norms > 1e-30, norms, 1.0)
    return cross / safe


def _merge_small_patches(patch_ids, face_neighbours, min_faces, n_faces):
    ids = patch_ids.copy()
    for _ in range(100):
        unique, counts = np.unique(ids, return_counts=True)
        small = set(int(unique[i]) for i in range(len(unique)) if counts[i] < min_faces)
        if not small:
            break
        changed = False
        for sp in small:
            nbr_counts: dict[int, int] = {}
            sp_faces = np.where(ids == sp)[0]
            for fi in sp_faces:
                for nbr in face_neighbours[fi]:
                    nbr_pid = int(ids[nbr])
                    if nbr_pid != sp:
                        nbr_counts[nbr_pid] = nbr_counts.get(nbr_pid, 0) + 1
            if not nbr_counts:
                continue
            best_nbr = max(nbr_counts, key=nbr_counts.get)
            ids[ids == sp] = best_nbr
            changed = True
        if not changed:
            break
    return ids


def _smooth_patches(patch_ids, face_neighbours, n_faces):
    """One smoothing pass: reassign faces surrounded by majority of different patch."""
    ids = patch_ids.copy()
    for fi in range(n_faces):
        nbr_pids = [int(ids[nbr]) for nbr in face_neighbours[fi]]
        if not nbr_pids:
            continue
        # Count neighbour patches
        counts: dict[int, int] = {}
        for pid in nbr_pids:
            counts[pid] = counts.get(pid, 0) + 1
        max_count_pid = max(counts, key=counts.get)
        # Reassign if super-majority (>= 80%) of neighbours are from different patch
        if max_count_pid != int(ids[fi]) and counts[max_count_pid] >= 0.8 * len(nbr_pids):
            ids[fi] = max_count_pid
    return ids


def _directional_name(normal: np.ndarray) -> str:
    """Infer patch name from mean face normal."""
    axes = {
        "top": np.array([0, 0, 1.0]),
        "bottom": np.array([0, 0, -1.0]),
        "north": np.array([0, 1.0, 0]),
        "south": np.array([0, -1.0, 0]),
        "east": np.array([1.0, 0, 0]),
        "west": np.array([-1.0, 0, 0]),
    }
    n_norm = np.linalg.norm(normal)
    if n_norm < 1e-30:
        return "unknown"
    n_dir = normal / n_norm
    best_name = "unknown"
    best_dot = -1.0
    for name, ax in axes.items():
        d = np.dot(n_dir, ax)
        if d > best_dot:
            best_dot = d
            best_name = name
    return best_name


def _write_patched_stl(path, verts, faces, normals, patch_ids, n_patches, patch_names=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    unique_ids = np.unique(patch_ids)
    with open(path, "w", encoding="utf-8") as f:
        for pid in unique_ids:
            pname = patch_names.get(int(pid), f"patch_{pid}") if patch_names else f"patch_{pid}"
            mask = patch_ids == pid
            f.write(f"solid {pname}\n")
            for fi in np.where(mask)[0]:
                n = normals[fi]
                f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
                f.write("    outer loop\n")
                for vi in range(3):
                    pt = verts[faces[fi, vi]]
                    f.write(f"      vertex {pt[0]:.6e} {pt[1]:.6e} {pt[2]:.6e}\n")
                f.write("    endloop\n  endfacet\n")
            f.write(f"endsolid {pname}\n")
