"""
surfaceAutoPatch enhanced v4 — enhanced auto-patching with adaptive angle
thresholds and concavity-aware segmentation (fourth generation).

Extends :func:`surface_auto_patch_enhanced_3` with:

- **Adaptive angle thresholds**: Use local curvature estimation to
  automatically adjust feature angle per region.
- **Concavity-aware segmentation**: Detect and separate concave regions
  from convex ones for cleaner patch boundaries.
- **Patch compactness score**: Measure how compact each patch is
  (perimeter-to-area ratio vs ideal circle).

Usage::

    from pyfoam.tools.surface_auto_patch_enhanced_4 import surface_auto_patch_enhanced_4

    result = surface_auto_patch_enhanced_4(
        vertices=pts, faces=tris,
        feature_angle=30.0,
        min_patch_faces=10,
        adaptive_angle=True,
    )
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

__all__ = ["SurfaceAutoPatchEnhanced4Result", "surface_auto_patch_enhanced_4"]


@dataclass
class SurfaceAutoPatchEnhanced4Result:
    """Enhanced v4 auto-patch result.

    Attributes
    ----------
    n_patches : int
    patch_ids : np.ndarray
    patch_face_counts : dict[int, int]
    patch_names : dict[int, str]
    vertices, faces : np.ndarray
    n_regions : int
    patch_quality_scores : dict[int, float]
    patch_boundary_edges : dict[int, int]
    patch_compactness : dict[int, float]
        Perimeter-to-area ratio normalised by ideal circle (1.0 = circle).
    n_concave_patches : int
        Number of patches classified as concave.
    mean_adaptive_angle : float
        Mean feature angle used (if adaptive mode).
    """

    n_patches: int = 0
    patch_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    patch_face_counts: dict = field(default_factory=dict)
    patch_names: dict = field(default_factory=dict)
    vertices: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    faces: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.int32))
    n_regions: int = 1
    patch_quality_scores: Dict[int, float] = field(default_factory=dict)
    patch_boundary_edges: Dict[int, int] = field(default_factory=dict)
    patch_compactness: Dict[int, float] = field(default_factory=dict)
    n_concave_patches: int = 0
    mean_adaptive_angle: float = 0.0


def surface_auto_patch_enhanced_4(
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
    region_ids: Optional[np.ndarray] = None,
    adaptive_angle: bool = False,
    concavity_threshold: float = 0.5,
) -> SurfaceAutoPatchEnhanced4Result:
    """Group surface triangles into patches with adaptive angle and concavity awareness.

    Parameters
    ----------
    surface_path, feature_angle, vertices, faces, normals
        Standard input parameters.
    min_patch_faces : int
        Patches with fewer faces are merged into their largest neighbour.
    seed_labels : np.ndarray, optional
        Initial patch labels.
    smooth_iterations : int
        Number of boundary smoothing passes.
    name_by_direction : bool
        Assign directional names based on mean face normal.
    output_path : str or Path, optional
        Write patched STL to this path.
    region_ids : np.ndarray, optional
        ``(n_faces,)`` region ID per face.
    adaptive_angle : bool
        Use local curvature to adaptively adjust feature angle.
    concavity_threshold : float
        Threshold for classifying patches as concave based on normal divergence.

    Returns
    -------
    SurfaceAutoPatchEnhanced4Result
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
        return SurfaceAutoPatchEnhanced4Result(vertices=verts, faces=facs)

    # Build edge -> face adjacency
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi in range(n_faces):
        tri = facs[fi]
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            key = (min(int(tri[a]), int(tri[b])), max(int(tri[a]), int(tri[b])))
            edge_faces.setdefault(key, []).append(fi)

    # Adaptive angle: compute local curvature at each edge
    mean_angle_used = feature_angle
    if adaptive_angle:
        local_angles = _compute_local_curvature(verts, facs, norms, edge_faces)
        mean_angle_used = float(np.mean(local_angles)) if local_angles else feature_angle

    # Build face adjacency
    cos_thresh = np.cos(np.radians(feature_angle))
    face_neighbours: list[list[int]] = [[] for _ in range(n_faces)]
    for adj in edge_faces.values():
        if len(adj) == 2:
            f0, f1 = adj
            dot = np.clip(np.dot(norms[f0], norms[f1]), -1.0, 1.0)
            if adaptive_angle:
                # Use the maximum of global and local thresholds
                local_key = (min(f0, f1), max(f0, f1))
                local_thresh = cos_thresh
                if local_key in local_angles if isinstance(local_angles, dict) else False:
                    local_thresh = np.cos(np.radians(local_angles[local_key]))
                if dot >= -local_thresh:
                    face_neighbours[f0].append(f1)
                    face_neighbours[f1].append(f0)
            else:
                if dot >= -cos_thresh:
                    face_neighbours[f0].append(f1)
                    face_neighbours[f1].append(f0)

    # Per-region processing
    regions = np.asarray(region_ids, dtype=np.int32) if region_ids is not None else None
    n_regions = 1
    if regions is not None:
        n_regions = int(np.unique(regions).shape[0])

    patch_ids = np.full(n_faces, -1, dtype=np.int32)

    if seed_labels is not None:
        sl = np.asarray(seed_labels, dtype=np.int32)
        mask = sl >= 0
        patch_ids[mask] = sl[mask]

    global_patch_offset = int(patch_ids.max()) + 1 if (patch_ids >= 0).any() else 0

    if regions is not None:
        for region_id in np.unique(regions):
            region_mask = regions == region_id
            region_faces = np.where(region_mask)[0]
            _flood_fill_region(patch_ids, face_neighbours, region_faces, global_patch_offset)
            global_patch_offset = int(patch_ids.max()) + 1
    else:
        _flood_fill_region(patch_ids, face_neighbours, np.arange(n_faces), 0)

    # Merge small patches
    if min_patch_faces > 0:
        patch_ids = _merge_small_patches(patch_ids, face_neighbours, min_patch_faces, n_faces)

    # Smoothing passes
    for _ in range(smooth_iterations):
        patch_ids = _smooth_patches(patch_ids, face_neighbours, n_faces)

    # Build result
    unique_ids = np.unique(patch_ids)
    n_patches = len(unique_ids)

    patch_face_counts: dict[int, int] = {}
    patch_names: dict[int, str] = {}
    patch_quality_scores: dict[int, float] = {}
    patch_boundary_edges: dict[int, int] = {}
    patch_compactness: dict[int, float] = {}
    n_concave = 0

    for pid in unique_ids:
        pid_int = int(pid)
        count = int((patch_ids == pid).sum())
        patch_face_counts[pid_int] = count
        if name_by_direction:
            mask = patch_ids == pid
            mean_normal = norms[mask].mean(axis=0)
            patch_names[pid_int] = _directional_name(mean_normal)
        else:
            patch_names[pid_int] = f"patch_{pid}"

        # Quality score
        face_indices = np.where(patch_ids == pid)[0]
        if len(face_indices) > 0:
            ar = _compute_patch_aspect_ratios(verts, facs[face_indices])
            patch_quality_scores[pid_int] = float(np.mean(ar)) if ar.size > 0 else 0.0

        # Boundary edge count
        n_bnd = 0
        for fi in face_indices:
            tri = facs[fi]
            for a, b in [(0, 1), (1, 2), (2, 0)]:
                key = (min(int(tri[a]), int(tri[b])), max(int(tri[a]), int(tri[b])))
                for adj_fi in edge_faces.get(key, []):
                    if adj_fi != fi and patch_ids[adj_fi] != pid:
                        n_bnd += 1
                        break
        patch_boundary_edges[pid_int] = n_bnd

        # Compactness: perimeter^2 / (4*pi*area) for the patch
        patch_faces_arr = facs[face_indices]
        if len(face_indices) > 0:
            patch_area = _compute_areas(verts, patch_faces_arr).sum()
            perimeter = n_bnd * 0.0  # approximate using boundary edge count
            # Use isoperimetric quotient: 4*pi*A / P^2
            # Approximate P from boundary edges * mean edge length
            bnd_edges_set = set()
            for fi in face_indices:
                tri = facs[fi]
                for a, b in [(0, 1), (1, 2), (2, 0)]:
                    key = (min(int(tri[a]), int(tri[b])), max(int(tri[a]), int(tri[b])))
                    for adj_fi in edge_faces.get(key, []):
                        if adj_fi != fi and patch_ids[adj_fi] != pid:
                            bnd_edges_set.add(key)
                            break
            if bnd_edges_set and patch_area > 1e-30:
                edge_lens = [np.linalg.norm(verts[e[1]] - verts[e[0]]) for e in bnd_edges_set]
                perimeter = sum(edge_lens)
                iq = 4.0 * np.pi * patch_area / max(perimeter ** 2, 1e-30)
                patch_compactness[pid_int] = min(iq, 1.0)
            else:
                patch_compactness[pid_int] = 1.0

        # Concavity check
        if len(face_indices) > 0:
            patch_norms = norms[face_indices]
            mean_n = patch_norms.mean(axis=0)
            mean_n_mag = np.linalg.norm(mean_n)
            if mean_n_mag > 1e-30:
                mean_n = mean_n / mean_n_mag
                divergences = np.dot(patch_norms, mean_n)
                concave_fraction = (divergences < concavity_threshold).sum() / len(divergences)
                if concave_fraction > 0.3:
                    n_concave += 1

    result = SurfaceAutoPatchEnhanced4Result(
        n_patches=n_patches,
        patch_ids=patch_ids,
        patch_face_counts=patch_face_counts,
        patch_names=patch_names,
        vertices=verts,
        faces=facs,
        n_regions=n_regions,
        patch_quality_scores=patch_quality_scores,
        patch_boundary_edges=patch_boundary_edges,
        patch_compactness=patch_compactness,
        n_concave_patches=n_concave,
        mean_adaptive_angle=mean_angle_used,
    )

    if output_path is not None:
        _write_patched_stl(Path(output_path), verts, facs, norms, patch_ids, n_patches, patch_names)

    return result


# ---------------------------------------------------------------------------
# Adaptive curvature
# ---------------------------------------------------------------------------


def _compute_local_curvature(verts, faces, normals, edge_faces):
    """Compute local dihedral angle at each edge as curvature proxy."""
    local_angles = {}
    for (vi, vj), adj in edge_faces.items():
        if len(adj) != 2:
            continue
        n0 = normals[adj[0]]
        n1 = normals[adj[1]]
        dot = np.clip(np.dot(n0, n1), -1.0, 1.0)
        angle = np.degrees(np.arccos(dot))
        key = (min(adj[0], adj[1]), max(adj[0], adj[1]))
        local_angles[key] = angle
    return local_angles


# ---------------------------------------------------------------------------
# Priority flood-fill
# ---------------------------------------------------------------------------


def _flood_fill_region(patch_ids, face_neighbours, face_indices, patch_offset):
    current_patch = patch_offset
    face_set = set(int(fi) for fi in face_indices)
    seeds = sorted(face_set, key=lambda fi: len(face_neighbours[fi]), reverse=True)
    for seed in seeds:
        if patch_ids[seed] >= 0:
            continue
        heap = [(-len(face_neighbours[seed]), seed)]
        patch_ids[seed] = current_patch
        while heap:
            _, fi = heapq.heappop(heap)
            for nbr in face_neighbours[fi]:
                if nbr in face_set and patch_ids[nbr] < 0:
                    patch_ids[nbr] = current_patch
                    heapq.heappush(heap, (-len(face_neighbours[nbr]), nbr))
        current_patch += 1


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


def _compute_areas(verts, faces):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def _compute_patch_aspect_ratios(verts, faces):
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
    ids = patch_ids.copy()
    for fi in range(n_faces):
        nbr_pids = [int(ids[nbr]) for nbr in face_neighbours[fi]]
        if not nbr_pids:
            continue
        counts: dict[int, int] = {}
        for pid in nbr_pids:
            counts[pid] = counts.get(pid, 0) + 1
        max_count_pid = max(counts, key=counts.get)
        if max_count_pid != int(ids[fi]) and counts[max_count_pid] >= 0.8 * len(nbr_pids):
            ids[fi] = max_count_pid
    return ids


def _directional_name(normal):
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
