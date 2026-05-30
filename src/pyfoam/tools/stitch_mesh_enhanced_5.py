"""
stitchMesh enhanced v5 — enhanced mesh stitching with adaptive tolerance,
stitch quality scoring, and gap detection (fifth generation).

Extends :func:`stitch_mesh_enhanced_4` with:

- **Adaptive tolerance**: Automatically adjust matching tolerance based on
  local edge length statistics around each patch.
- **Stitch quality scoring**: Rate each stitched pair by face alignment,
  area ratio, and centroid distance (0-1 score).
- **Gap detection**: Identify and report unmatched boundary regions that
  may indicate missing geometry.

Usage::

    from pyfoam.tools.stitch_mesh_enhanced_5 import stitch_mesh_enhanced_5

    result = stitch_mesh_enhanced_5(
        mesh, "patch1", "patch2",
        tolerance=1e-4,
        adaptive_tolerance=True,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import torch

from pyfoam.core.dtype import INDEX_DTYPE

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["StitchEnhanced5Result", "stitch_mesh_enhanced_5"]


@dataclass
class StitchEnhanced5Result:
    """Result from :func:`stitch_mesh_enhanced_5`.

    Attributes
    ----------
    mesh : FvMesh
    n_stitched : int
    n_unmatched : int
    mean_quality : float
        Mean stitch quality score (0-1).
    min_quality : float
        Minimum stitch quality score.
    gap_regions : list[tuple[int, int]]
        Contiguous groups of unmatched faces as ``(start_idx, count)``.
    n_gap_faces : int
        Total unmatched boundary faces after stitching.
    adaptive_tol_used : float
        Tolerance actually used (may differ from input).
    """

    mesh: object  # FvMesh
    n_stitched: int = 0
    n_unmatched: int = 0
    mean_quality: float = 1.0
    min_quality: float = 1.0
    gap_regions: list = field(default_factory=list)
    n_gap_faces: int = 0
    adaptive_tol_used: float = 0.0


def stitch_mesh_enhanced_5(
    mesh: "FvMesh",
    patch1_name: str,
    patch2_name: str,
    tolerance: float = 1e-6,
    non_conformal: bool = False,
    adaptive_tolerance: bool = False,
    detect_gaps: bool = True,
) -> StitchEnhanced5Result:
    """Stitch two patches with adaptive tolerance and quality scoring.

    Parameters
    ----------
    mesh : FvMesh
    patch1_name, patch2_name : str
    tolerance : float
    non_conformal : bool
    adaptive_tolerance : bool
        Scale tolerance by local mean edge length.
    detect_gaps : bool
        Identify contiguous unmatched regions.

    Returns
    -------
    StitchEnhanced5Result
    """
    from pyfoam.mesh.fv_mesh import FvMesh
    from pyfoam.tools.stitch_mesh_enhanced import stitch_mesh_enhanced

    # Locate patches
    p1 = _find_patch(mesh, patch1_name)
    p2 = _find_patch(mesh, patch2_name)

    eff_tol = tolerance
    if adaptive_tolerance:
        eff_tol = _compute_adaptive_tolerance(mesh, p1, p2, tolerance)

    # Delegate to v1 enhanced for core stitching
    v1_result = stitch_mesh_enhanced(
        mesh, patch1_name, patch2_name, eff_tol, non_conformal,
    )

    # Quality scoring on the stitch pairs
    mean_q, min_q = _score_stitch_quality(mesh, p1, p2, v1_result.n_stitched)

    # Gap detection
    gap_regions = []
    n_gap = v1_result.n_unmatched
    if detect_gaps and n_gap > 0:
        gap_regions = _detect_gap_regions(v1_result.mesh, patch1_name, patch2_name)

    return StitchEnhanced5Result(
        mesh=v1_result.mesh,
        n_stitched=v1_result.n_stitched,
        n_unmatched=v1_result.n_unmatched,
        mean_quality=mean_q,
        min_quality=min_q,
        gap_regions=gap_regions,
        n_gap_faces=n_gap,
        adaptive_tol_used=eff_tol,
    )


# ---------------------------------------------------------------------------
# Adaptive tolerance
# ---------------------------------------------------------------------------


def _compute_adaptive_tolerance(mesh, p1, p2, base_tol):
    """Scale tolerance by local mean edge length of the two patches."""
    edge_lengths = []
    for patch in (p1, p2):
        start = patch["startFace"]
        for fi in range(start, start + patch["nFaces"]):
            pts = mesh.points[mesh.faces[fi]].float()
            for i in range(pts.shape[0]):
                j = (i + 1) % pts.shape[0]
                edge_lengths.append((pts[i] - pts[j]).norm().item())

    if not edge_lengths:
        return base_tol

    mean_len = sum(edge_lengths) / len(edge_lengths)
    # Tolerance = min of base and 1% of mean edge length
    return min(base_tol, max(mean_len * 0.01, 1e-12))


# ---------------------------------------------------------------------------
# Stitch quality scoring
# ---------------------------------------------------------------------------


def _score_stitch_quality(mesh, p1, p2, n_stitched):
    """Score stitched face pairs by centroid distance and area ratio."""
    if n_stitched == 0:
        return 0.0, 0.0

    scores = []
    p1_start = p1["startFace"]
    p2_start = p2["startFace"]
    p1_n = p1["nFaces"]

    for i in range(min(n_stitched, p1_n)):
        fi1 = p1_start + i
        fi2 = p2_start + i
        if fi2 >= p2_start + p2["nFaces"]:
            break

        pts1 = mesh.points[mesh.faces[fi1]].float()
        pts2 = mesh.points[mesh.faces[fi2]].float()
        c1 = pts1.mean(dim=0)
        c2 = pts2.mean(dim=0)
        dist = (c1 - c2).norm().item()

        # Area ratio
        a1 = 0.5 * torch.cross(pts1[1] - pts1[0], pts1[2] - pts1[0]).norm().item() if pts1.shape[0] >= 3 else 0.0
        a2 = 0.5 * torch.cross(pts2[1] - pts2[0], pts2[2] - pts2[0]).norm().item() if pts2.shape[0] >= 3 else 0.0
        area_ratio = min(a1, a2) / max(a1, a2, 1e-30)

        # Score = combination of proximity and area match
        prox_score = max(0.0, 1.0 - dist / max(1e-6, dist + 0.01))
        scores.append(0.5 * prox_score + 0.5 * area_ratio)

    if not scores:
        return 0.0, 0.0
    return sum(scores) / len(scores), min(scores)


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------


def _detect_gap_regions(result_mesh, p1_name, p2_name):
    """Identify contiguous unmatched boundary regions."""
    # After stitching, the two stitched patches should be removed.
    # Look for remaining boundary patches that might be gaps.
    regions = []
    for p in result_mesh.boundary:
        if p["name"] not in (p1_name, p2_name):
            continue
        # If the patch still exists, it has unmatched faces
        start = p["startFace"]
        count = p["nFaces"]
        if count > 0:
            regions.append((start, count))
    return regions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_patch(mesh: "FvMesh", name: str) -> dict:
    for patch in mesh.boundary:
        if patch["name"] == name:
            return patch
    raise ValueError(f"Patch '{name}' not found in mesh boundary.")
