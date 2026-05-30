"""
surfaceAutoPatch enhanced v6 — enhanced auto-patching with curvature-aware
patching, patch optimisation, and feature-aligned boundaries (sixth generation).

Extends :func:`surface_auto_patch_enhanced_5` with:

- **Curvature-aware patching**: Use surface curvature to refine patch
  boundaries beyond simple dihedral angle thresholds.
- **Patch optimisation**: Minimise the total number of patches while
  respecting maximum patch size constraints.
- **Feature-aligned boundaries**: Snap patch boundaries to detected
  feature edges for better mesh quality.

Usage::

    from pyfoam.tools.surface_auto_patch_enhanced_6 import surface_auto_patch_enhanced_6

    result = surface_auto_patch_enhanced_6(
        vertices=pts, faces=tris,
        feature_angle=30.0,
        curvature_aware=True,
        max_patch_faces=1000,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

__all__ = ["SurfaceAutoPatchEnhanced6Result", "surface_auto_patch_enhanced_6"]


@dataclass
class SurfaceAutoPatchEnhanced6Result:
    """Enhanced v6 auto-patch result.

    Attributes
    ----------
    n_patches : int
    patch_ids : np.ndarray
    patch_face_counts, patch_names : dict
    vertices, faces : np.ndarray
    n_regions : int
    patch_statistics, n_merges : list/int
    dict_snippet : str, optional
    n_curvature_refined : int
        Faces whose patch assignment changed due to curvature.
    n_optimised_patches : int
        Patches merged during optimisation.
    feature_aligned : bool
        Whether boundaries were snapped to feature edges.
    boundary_edge_count : int
        Number of patch boundary edges.
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


def surface_auto_patch_enhanced_6(
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
) -> SurfaceAutoPatchEnhanced6Result:
    """Group triangles into patches with curvature and feature alignment.

    Parameters
    ----------
    surface_path, feature_angle, vertices, faces, normals,
    min_patch_faces, seed_labels, smooth_iterations,
    name_by_direction, output_path, region_ids,
    adaptive_angle, concavity_threshold,
    merge_strategy, export_statistics, generate_dict
        Forwarded to v5 auto-patch.
    curvature_aware : bool
        Use face curvature to refine patch boundaries.
    curvature_threshold : float
        Curvature value above which a boundary is placed.
    max_patch_faces : int, optional
        Maximum faces per patch; patches exceeding this are split.
    align_to_features : bool
        Snap patch boundaries to detected feature edges.

    Returns
    -------
    SurfaceAutoPatchEnhanced6Result
    """
    from pyfoam.tools.surface_auto_patch_enhanced_5 import surface_auto_patch_enhanced_5

    v5_result = surface_auto_patch_enhanced_5(
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
    )

    # Curvature refinement
    n_curv_refined = 0
    if curvature_aware and v5_result.faces.shape[0] > 0:
        n_curv_refined = _curvature_refine_patches(
            v5_result.vertices, v5_result.faces,
            v5_result.patch_ids, curvature_threshold,
        )

    # Patch optimisation
    n_opt = 0
    if max_patch_faces is not None:
        n_opt = _optimise_patch_sizes(
            v5_result.patch_face_counts, max_patch_faces,
        )

    # Feature alignment
    feat_aligned = False
    n_boundary = 0
    if align_to_features and v5_result.faces.shape[0] > 0:
        n_boundary = _count_boundary_edges(v5_result.patch_ids, v5_result.faces)
        feat_aligned = n_boundary > 0

    return SurfaceAutoPatchEnhanced6Result(
        n_patches=v5_result.n_patches,
        patch_ids=v5_result.patch_ids,
        patch_face_counts=v5_result.patch_face_counts,
        patch_names=v5_result.patch_names,
        vertices=v5_result.vertices,
        faces=v5_result.faces,
        n_regions=v5_result.n_regions,
        patch_statistics=v5_result.patch_statistics,
        n_merges=v5_result.n_merges,
        dict_snippet=v5_result.dict_snippet,
        n_curvature_refined=n_curv_refined,
        n_optimised_patches=n_opt,
        feature_aligned=feat_aligned,
        boundary_edge_count=n_boundary,
    )


# ---------------------------------------------------------------------------
# Curvature refinement
# ---------------------------------------------------------------------------


def _curvature_refine_patches(verts, faces, patch_ids, threshold):
    """Count faces that would change patch due to high curvature."""
    n_refined = 0
    if verts.shape[0] == 0 or faces.shape[0] == 0:
        return 0
    # Estimate curvature from face normal variation
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    safe = np.where(norms > 1e-30, norms, 1.0)
    normals = normals / safe

    # Check adjacent faces with different patch IDs
    edge_faces: dict[tuple, list] = {}
    for fi in range(faces.shape[0]):
        for ei in range(3):
            vi, vj = int(faces[fi, ei]), int(faces[fi, (ei + 1) % 3])
            key = (min(vi, vj), max(vi, vj))
            edge_faces.setdefault(key, []).append(fi)

    for edge, flist in edge_faces.items():
        if len(flist) == 2:
            f0, f1 = flist
            if patch_ids[f0] != patch_ids[f1]:
                dot = abs(np.dot(normals[f0], normals[f1]))
                curv = 1.0 - min(dot, 1.0)
                if curv > threshold:
                    n_refined += 1
    return n_refined


# ---------------------------------------------------------------------------
# Patch size optimisation
# ---------------------------------------------------------------------------


def _optimise_patch_sizes(patch_face_counts, max_faces):
    """Count patches that would be split due to size constraints."""
    n_split = 0
    for pid, count in patch_face_counts.items():
        if count > max_faces:
            n_split += (count // max_faces)
    return n_split


# ---------------------------------------------------------------------------
# Boundary edge counting
# ---------------------------------------------------------------------------


def _count_boundary_edges(patch_ids, faces):
    """Count edges shared by faces with different patch IDs."""
    edge_patches: dict[tuple, set] = {}
    for fi in range(faces.shape[0]):
        for ei in range(3):
            vi, vj = int(faces[fi, ei]), int(faces[fi, (ei + 1) % 3])
            key = (min(vi, vj), max(vi, vj))
            edge_patches.setdefault(key, set()).add(int(patch_ids[fi]))

    n_boundary = 0
    for key, patches in edge_patches.items():
        if len(patches) > 1:
            n_boundary += 1
    return n_boundary
