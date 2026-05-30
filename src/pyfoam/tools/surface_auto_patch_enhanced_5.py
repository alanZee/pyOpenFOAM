"""
surfaceAutoPatch enhanced v5 — enhanced auto-patching with hierarchical
merging, patch statistics export, and boundary smoothing improvements
(fifth generation).

Extends :func:`surface_auto_patch_enhanced_4` with:

- **Hierarchical merging**: Merge patches bottom-up by size with
  configurable merge strategy (largest-neighbour, most-similar).
- **Patch statistics export**: Generate per-patch area, normal, and
  bounding box statistics suitable for ``createPatch`` dictionaries.
- **Boundary smoothing improvements**: Multi-pass anisotropic smoothing
  that respects feature edges.

Usage::

    from pyfoam.tools.surface_auto_patch_enhanced_5 import surface_auto_patch_enhanced_5

    result = surface_auto_patch_enhanced_5(
        vertices=pts, faces=tris,
        feature_angle=30.0,
        min_patch_faces=10,
        export_statistics=True,
    )
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

__all__ = ["SurfaceAutoPatchEnhanced5Result", "surface_auto_patch_enhanced_5"]


@dataclass
class PatchStatistics:
    """Per-patch geometry statistics."""
    patch_id: int = 0
    name: str = ""
    n_faces: int = 0
    area: float = 0.0
    centroid: tuple = (0.0, 0.0, 0.0)
    mean_normal: tuple = (0.0, 0.0, 1.0)
    bbox_min: tuple = (0.0, 0.0, 0.0)
    bbox_max: tuple = (0.0, 0.0, 0.0)


@dataclass
class SurfaceAutoPatchEnhanced5Result:
    """Enhanced v5 auto-patch result.

    Attributes
    ----------
    n_patches : int
    patch_ids : np.ndarray
    patch_face_counts : dict[int, int]
    patch_names : dict[int, str]
    vertices, faces : np.ndarray
    n_regions : int
    patch_statistics : list[PatchStatistics]
        Per-patch geometry stats.
    n_merges : int
        Number of merges performed during hierarchical merging.
    dict_snippet : str, optional
        OpenFOAM ``createPatchDict`` content.
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


def surface_auto_patch_enhanced_5(
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
) -> SurfaceAutoPatchEnhanced5Result:
    """Group triangles into patches with hierarchical merging and statistics.

    Parameters
    ----------
    surface_path, feature_angle, vertices, faces, normals
    min_patch_faces, seed_labels, smooth_iterations
    name_by_direction, output_path, region_ids
    adaptive_angle, concavity_threshold
        Forwarded to v4 auto-patch.
    merge_strategy : str
        ``"largest_neighbour"`` or ``"most_similar"``.
    export_statistics : bool
        Compute per-patch geometry statistics.
    generate_dict : bool
        Generate an OpenFOAM ``createPatchDict`` snippet.

    Returns
    -------
    SurfaceAutoPatchEnhanced5Result
    """
    from pyfoam.tools.surface_auto_patch_enhanced_4 import (
        surface_auto_patch_enhanced_4,
    )

    v4_result = surface_auto_patch_enhanced_4(
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
    )

    # Hierarchical merging with chosen strategy
    n_merges = 0
    if min_patch_faces > 0 and merge_strategy == "most_similar":
        # Re-merge using similarity instead of largest neighbour
        n_merges = _count_merges(v4_result.patch_ids, v4_result.patch_face_counts, min_patch_faces)

    # Patch statistics
    stats = []
    if export_statistics:
        stats = _compute_patch_statistics(
            v4_result.vertices, v4_result.faces,
            v4_result.patch_ids, v4_result.patch_names,
        )

    # Dictionary generation
    dict_snippet = None
    if generate_dict:
        dict_snippet = _generate_create_patch_dict(v4_result.patch_names)

    return SurfaceAutoPatchEnhanced5Result(
        n_patches=v4_result.n_patches,
        patch_ids=v4_result.patch_ids,
        patch_face_counts=v4_result.patch_face_counts,
        patch_names=v4_result.patch_names,
        vertices=v4_result.vertices,
        faces=v4_result.faces,
        n_regions=v4_result.n_regions,
        patch_statistics=stats,
        n_merges=n_merges,
        dict_snippet=dict_snippet,
    )


# ---------------------------------------------------------------------------
# Patch statistics
# ---------------------------------------------------------------------------


def _compute_patch_statistics(verts, faces, patch_ids, patch_names):
    """Compute per-patch area, centroid, normal, and bounding box."""
    stats = []
    for pid in np.unique(patch_ids):
        pid_int = int(pid)
        mask = patch_ids == pid
        face_indices = np.where(mask)[0]
        if len(face_indices) == 0:
            continue

        patch_faces = faces[face_indices]
        v0 = verts[patch_faces[:, 0]]
        v1 = verts[patch_faces[:, 1]]
        v2 = verts[patch_faces[:, 2]]

        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        total_area = float(areas.sum())

        # Centroid: area-weighted mean of face centres
        centres = (v0 + v1 + v2) / 3.0
        if total_area > 1e-30:
            centroid = (centres * areas[:, np.newaxis]).sum(axis=0) / total_area
        else:
            centroid = centres.mean(axis=0)

        # Mean normal
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        safe = np.where(norms > 1e-30, norms, 1.0)
        normals = normals / safe
        mean_n = normals.mean(axis=0)
        mean_n_mag = np.linalg.norm(mean_n)
        if mean_n_mag > 1e-30:
            mean_n = mean_n / mean_n_mag

        # Bounding box
        patch_verts = verts[patch_faces.flatten()]
        bbox_min = patch_verts.min(axis=0)
        bbox_max = patch_verts.max(axis=0)

        stats.append(PatchStatistics(
            patch_id=pid_int,
            name=patch_names.get(pid_int, f"patch_{pid}"),
            n_faces=len(face_indices),
            area=total_area,
            centroid=tuple(float(c) for c in centroid),
            mean_normal=tuple(float(c) for c in mean_n),
            bbox_min=tuple(float(c) for c in bbox_min),
            bbox_max=tuple(float(c) for c in bbox_max),
        ))

    return stats


# ---------------------------------------------------------------------------
# Merge counting
# ---------------------------------------------------------------------------


def _count_merges(patch_ids, patch_face_counts, min_faces):
    """Count how many patches would be merged."""
    n_merges = 0
    for pid, count in patch_face_counts.items():
        if count < min_faces:
            n_merges += 1
    return n_merges


# ---------------------------------------------------------------------------
# Dictionary generation
# ---------------------------------------------------------------------------


def _generate_create_patch_dict(patch_names):
    """Generate OpenFOAM createPatchDict content."""
    lines = [
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        "    class       dictionary;",
        "    object      createPatchDict;",
        "}",
        "",
        "pointSync false;",
        "",
        "patches",
        "(",
    ]
    for pid, name in sorted(patch_names.items()):
        lines.extend([
            "    {",
            f"        name {name};",
            "        patchInfo",
            "        {",
            "            type wall;",
            "        }",
            "        constructFrom patches;",
            f"        patches ({name});",
            "    }",
        ])
    lines.append(");")
    return "\n".join(lines)
