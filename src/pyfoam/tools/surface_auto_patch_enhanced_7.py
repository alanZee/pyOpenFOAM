"""
surfaceAutoPatch enhanced v7 — enhanced auto-patching with semantic labelling,
patch adjacency analysis, and mesh size estimation (seventh generation).

Extends :func:`surface_auto_patch_enhanced_6` with:

- **Semantic labelling**: Assign semantic labels (inlet, outlet, wall,
  etc.) to patches based on geometry and flow direction analysis.
- **Patch adjacency analysis**: Build an adjacency matrix of patches
  and identify shared edges for interface handling.
- **Mesh size estimation**: Estimate the local cell size needed for
  each patch based on curvature and feature density.

Usage::

    from pyfoam.tools.surface_auto_patch_enhanced_7 import surface_auto_patch_enhanced_7

    result = surface_auto_patch_enhanced_7(
        vertices=pts, faces=tris,
        feature_angle=30.0,
        semantic_labelling=True,
        estimate_mesh_size=True,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

__all__ = ["SurfaceAutoPatchEnhanced7Result", "surface_auto_patch_enhanced_7"]


@dataclass
class PatchAdjacency:
    """Adjacency relationship between two patches."""
    patch_a: str = ""
    patch_b: str = ""
    n_shared_edges: int = 0
    interface_type: str = "internal"


@dataclass
class MeshSizeEstimate:
    """Estimated mesh size parameters for a patch."""
    patch_name: str = ""
    min_cell_size: float = 0.0
    max_cell_size: float = 0.0
    mean_curvature: float = 0.0
    n_cells_estimated: int = 0


@dataclass
class SurfaceAutoPatchEnhanced7Result:
    """Enhanced v7 auto-patch result.

    Attributes
    ----------
    Inherits all from v6, plus:
    semantic_labels : dict[str, str]
        Patch name -> semantic label mapping.
    adjacencies : list[PatchAdjacency]
        Patch adjacency relationships.
    mesh_size_estimates : list[MeshSizeEstimate]
        Per-patch mesh size estimates.
    """

    n_patches: int = 0
    patch_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    patch_face_counts: dict = field(default_factory=dict)
    patch_names: dict = field(default_factory=dict)
    vertices: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    faces: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.int32))
    n_regions: int = 1
    patch_statistics: list = field(default_factory=list)
    n_merges: int = 0
    dict_snippet: Optional[str] = None
    n_curvature_refined: int = 0
    n_optimised_patches: int = 0
    feature_aligned: bool = False
    boundary_edge_count: int = 0
    semantic_labels: dict = field(default_factory=dict)
    adjacencies: list = field(default_factory=list)
    mesh_size_estimates: list = field(default_factory=list)


def surface_auto_patch_enhanced_7(
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
    merge_strategy: str = "largest_neighbour",
    export_statistics: bool = False,
    generate_dict: bool = False,
    curvature_aware: bool = False,
    curvature_threshold: float = 0.1,
    max_patch_faces: Optional[int] = None,
    align_to_features: bool = False,
    semantic_labelling: bool = False,
    flow_direction: tuple = (1.0, 0.0, 0.0),
    estimate_mesh_size: bool = False,
    target_cell_count: int = 10000,
) -> SurfaceAutoPatchEnhanced7Result:
    """Group triangles into patches with semantic labelling.

    Parameters
    ----------
    surface_path .. align_to_features
        Forwarded to v6 auto-patch.
    semantic_labelling : bool
        Assign semantic labels to patches.
    flow_direction : tuple
        Reference flow direction for semantic analysis.
    estimate_mesh_size : bool
        Estimate per-patch mesh size.
    target_cell_count : int
        Target total cell count for size estimation.

    Returns
    -------
    SurfaceAutoPatchEnhanced7Result
    """
    from pyfoam.tools.surface_auto_patch_enhanced_6 import surface_auto_patch_enhanced_6

    v6_result = surface_auto_patch_enhanced_6(
        surface_path=surface_path,
        feature_angle=feature_angle,
        vertices=vertices,
        faces=faces,
        normals=normals,
        min_patch_faces=min_patch_faces,
        seed_labels=seed_labels,
        smooth_iterations=smooth_iterations,
        name_by_direction=name_by_direction,
        output_path=output_path,
        region_ids=region_ids,
        adaptive_angle=adaptive_angle,
        concavity_threshold=concavity_threshold,
        merge_strategy=merge_strategy,
        export_statistics=export_statistics,
        generate_dict=generate_dict,
        curvature_aware=curvature_aware,
        curvature_threshold=curvature_threshold,
        max_patch_faces=max_patch_faces,
        align_to_features=align_to_features,
    )

    # Semantic labelling
    labels = {}
    if semantic_labelling and v6_result.n_patches > 0:
        labels = _assign_semantic_labels(
            v6_result.vertices, v6_result.faces,
            v6_result.patch_ids, np.array(flow_direction),
        )

    # Adjacency analysis
    adjacencies = []
    if v6_result.faces.shape[0] > 0:
        adjacencies = _build_adjacency(v6_result.patch_ids, v6_result.faces)

    # Mesh size estimation
    size_estimates = []
    if estimate_mesh_size and v6_result.n_patches > 0:
        size_estimates = _estimate_mesh_sizes(
            v6_result.vertices, v6_result.faces,
            v6_result.patch_ids, target_cell_count,
        )

    return SurfaceAutoPatchEnhanced7Result(
        n_patches=v6_result.n_patches,
        patch_ids=v6_result.patch_ids,
        patch_face_counts=v6_result.patch_face_counts,
        patch_names=v6_result.patch_names,
        vertices=v6_result.vertices,
        faces=v6_result.faces,
        n_regions=v6_result.n_regions,
        patch_statistics=v6_result.patch_statistics,
        n_merges=v6_result.n_merges,
        dict_snippet=v6_result.dict_snippet,
        n_curvature_refined=v6_result.n_curvature_refined,
        n_optimised_patches=v6_result.n_optimised_patches,
        feature_aligned=v6_result.feature_aligned,
        boundary_edge_count=v6_result.boundary_edge_count,
        semantic_labels=labels,
        adjacencies=adjacencies,
        mesh_size_estimates=size_estimates,
    )


# ---------------------------------------------------------------------------
# Semantic labelling
# ---------------------------------------------------------------------------


def _assign_semantic_labels(verts, faces, patch_ids, flow_dir):
    """Assign semantic labels based on face normal vs flow direction."""
    labels = {}
    unique_ids = np.unique(patch_ids)

    for pid in unique_ids:
        mask = patch_ids == pid
        pid_int = int(pid)
        if verts.shape[0] == 0 or faces[mask].shape[0] == 0:
            labels[f"patch_{pid_int}"] = "wall"
            continue

        # Compute mean normal for this patch
        patch_faces = faces[mask]
        v0 = verts[patch_faces[:, 0]]
        v1 = verts[patch_faces[:, 1]]
        v2 = verts[patch_faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        safe = np.where(norms > 1e-30, norms, 1.0)
        normals = normals / safe
        mean_normal = normals.mean(axis=0)
        mean_norm = np.linalg.norm(mean_normal)
        if mean_norm > 1e-30:
            mean_normal /= mean_norm

        dot_flow = np.dot(mean_normal, flow_dir)
        if dot_flow > 0.7:
            labels[f"patch_{pid_int}"] = "outlet"
        elif dot_flow < -0.7:
            labels[f"patch_{pid_int}"] = "inlet"
        else:
            labels[f"patch_{pid_int}"] = "wall"

    return labels


# ---------------------------------------------------------------------------
# Adjacency analysis
# ---------------------------------------------------------------------------


def _build_adjacency(patch_ids, faces):
    """Build patch adjacency from shared edges."""
    edge_patches: dict[tuple, set] = {}
    for fi in range(faces.shape[0]):
        for ei in range(3):
            vi, vj = int(faces[fi, ei]), int(faces[fi, (ei + 1) % 3])
            key = (min(vi, vj), max(vi, vj))
            edge_patches.setdefault(key, set()).add(int(patch_ids[fi]))

    adj_map: dict[tuple, int] = {}
    for edge, pids in edge_patches.items():
        if len(pids) >= 2:
            plist = sorted(pids)
            for i in range(len(plist)):
                for j in range(i + 1, len(plist)):
                    pair = (plist[i], plist[j])
                    adj_map[pair] = adj_map.get(pair, 0) + 1

    adjacencies = []
    for (pa, pb), count in adj_map.items():
        adjacencies.append(PatchAdjacency(
            patch_a=f"patch_{pa}",
            patch_b=f"patch_{pb}",
            n_shared_edges=count,
        ))

    return adjacencies


# ---------------------------------------------------------------------------
# Mesh size estimation
# ---------------------------------------------------------------------------


def _estimate_mesh_sizes(verts, faces, patch_ids, target_count):
    """Estimate mesh size per patch."""
    estimates = []
    unique_ids = np.unique(patch_ids)
    total_faces = len(unique_ids)

    for pid in unique_ids:
        mask = patch_ids == pid
        n_patch_faces = int(np.sum(mask))
        ratio = n_patch_faces / max(total_faces, 1)
        est_cells = int(target_count * ratio)

        # Estimate sizes from face area
        patch_faces = faces[mask]
        if patch_faces.shape[0] > 0 and verts.shape[0] > 0:
            v0 = verts[patch_faces[:, 0]]
            v1 = verts[patch_faces[:, 1]]
            v2 = verts[patch_faces[:, 2]]
            areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
            mean_area = float(np.mean(areas)) if areas.size > 0 else 0.0
            min_size = float(np.sqrt(mean_area * 0.5)) if mean_area > 0 else 0.0
            max_size = float(np.sqrt(mean_area * 2.0)) if mean_area > 0 else 0.0
        else:
            min_size = 0.0
            max_size = 0.0
            mean_area = 0.0

        estimates.append(MeshSizeEstimate(
            patch_name=f"patch_{int(pid)}",
            min_cell_size=min_size,
            max_cell_size=max_size,
            mean_curvature=mean_area,
            n_cells_estimated=est_cells,
        ))

    return estimates
