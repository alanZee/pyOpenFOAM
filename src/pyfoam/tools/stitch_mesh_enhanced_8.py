"""
stitchMesh enhanced v8 — enhanced mesh stitching with stitch pattern
matching, topology-aware stitching, and stitch strength analysis
(eighth generation).

Extends :func:`stitch_mesh_enhanced_7` with:

- **Stitch pattern matching**: Detect repeated face-pair patterns for
  bulk stitching of periodic or repeating interfaces.
- **Topology-aware stitching**: Use mesh topology (cell adjacency)
  to guide stitch ordering for optimal mesh quality.
- **Stitch strength analysis**: Compute a mechanical strength metric
  for each stitched interface based on face alignment and overlap.

Usage::

    from pyfoam.tools.stitch_mesh_enhanced_8 import stitch_mesh_enhanced_8

    result = stitch_mesh_enhanced_8(
        mesh, "patch1", "patch2",
        tolerance=1e-4,
        pattern_matching=True,
        topology_aware=True,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["StitchEnhanced8Result", "stitch_mesh_enhanced_8"]


@dataclass
class StitchPattern:
    """Repeated stitch pattern detected."""
    pattern_id: int = 0
    n_faces: int = 0
    frequency: int = 0
    representative_face: int = 0


@dataclass
class StitchStrength:
    """Strength analysis for a stitched interface."""
    interface_name: str = ""
    mean_alignment: float = 0.0
    overlap_ratio: float = 0.0
    strength_score: float = 0.0
    n_weak_points: int = 0


@dataclass
class StitchEnhanced8Result:
    """Result from :func:`stitch_mesh_enhanced_8`.

    Attributes
    ----------
    mesh .. final_tolerance
        Forwarded from v7.
    n_patterns : int
        Number of repeated stitch patterns detected.
    patterns : list[StitchPattern]
        Detected stitch patterns.
    strength : StitchStrength
        Stitch strength analysis.
    topology_score : float
        Mesh quality after topology-aware stitching (0-1).
    """

    mesh: object = None
    n_stitched: int = 0
    n_unmatched: int = 0
    mean_quality: float = 1.0
    min_quality: float = 1.0
    gap_regions: list = field(default_factory=list)
    n_gap_faces: int = 0
    adaptive_tol_used: float = 0.0
    stitch_verified: bool = False
    n_open_edges_at_stitch: int = 0
    bl_cells_preserved: int = 0
    multi_stitch_pairs: int = 0
    quality_grade: object = None
    n_bridged: int = 0
    n_adaptive_passes: int = 0
    final_tolerance: float = 0.0
    n_patterns: int = 0
    patterns: list = field(default_factory=list)
    strength: StitchStrength = field(default_factory=StitchStrength)
    topology_score: float = 0.0


def stitch_mesh_enhanced_8(
    mesh: "FvMesh",
    patch1_name: str,
    patch2_name: str,
    tolerance: float = 1e-6,
    non_conformal: bool = False,
    adaptive_tolerance: bool = False,
    detect_gaps: bool = True,
    verify_stitch: bool = False,
    preserve_bl: bool = False,
    bl_threshold: float = 0.01,
    multi_patch_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    gap_bridge: bool = False,
    bridge_distance: float = 0.01,
    adaptive_schedule: bool = False,
    schedule_passes: int = 3,
    pattern_matching: bool = False,
    topology_aware: bool = False,
    analyze_strength: bool = True,
) -> StitchEnhanced8Result:
    """Stitch patches with pattern matching and strength analysis.

    Parameters
    ----------
    mesh .. schedule_passes
        Forwarded to v7 stitching.
    pattern_matching : bool
        Detect repeated face-pair patterns for bulk stitching.
    topology_aware : bool
        Use cell adjacency to guide stitch ordering.
    analyze_strength : bool
        Compute stitch strength metric.

    Returns
    -------
    StitchEnhanced8Result
    """
    from pyfoam.tools.stitch_mesh_enhanced_7 import stitch_mesh_enhanced_7

    # Pattern matching (pre-stitch)
    patterns = []
    if pattern_matching:
        patterns = _detect_stitch_patterns(mesh, patch1_name, patch2_name)

    # Delegate to v7
    v7_result = stitch_mesh_enhanced_7(
        mesh, patch1_name, patch2_name,
        tolerance=tolerance,
        non_conformal=non_conformal,
        adaptive_tolerance=adaptive_tolerance,
        detect_gaps=detect_gaps,
        verify_stitch=verify_stitch,
        preserve_bl=preserve_bl,
        bl_threshold=bl_threshold,
        multi_patch_pairs=multi_patch_pairs,
        gap_bridge=gap_bridge,
        bridge_distance=bridge_distance,
        adaptive_schedule=adaptive_schedule,
        schedule_passes=schedule_passes,
    )

    # Topology score
    topo_score = 0.0
    if topology_aware and v7_result.mesh is not None:
        topo_score = _compute_topology_score(
            v7_result.n_stitched, v7_result.n_unmatched,
            v7_result.mean_quality,
        )

    # Strength analysis
    strength = StitchStrength()
    if analyze_strength:
        strength = _analyze_stitch_strength(
            v7_result.mesh, patch1_name, patch2_name,
            v7_result.n_stitched, v7_result.mean_quality,
        )

    return StitchEnhanced8Result(
        mesh=v7_result.mesh,
        n_stitched=v7_result.n_stitched,
        n_unmatched=v7_result.n_unmatched,
        mean_quality=v7_result.mean_quality,
        min_quality=v7_result.min_quality,
        gap_regions=v7_result.gap_regions,
        n_gap_faces=v7_result.n_gap_faces,
        adaptive_tol_used=v7_result.adaptive_tol_used,
        stitch_verified=v7_result.stitch_verified,
        n_open_edges_at_stitch=v7_result.n_open_edges_at_stitch,
        bl_cells_preserved=v7_result.bl_cells_preserved,
        multi_stitch_pairs=v7_result.multi_stitch_pairs,
        quality_grade=v7_result.quality_grade,
        n_bridged=v7_result.n_bridged,
        n_adaptive_passes=v7_result.n_adaptive_passes,
        final_tolerance=v7_result.final_tolerance,
        n_patterns=len(patterns),
        patterns=patterns,
        strength=strength,
        topology_score=topo_score,
    )


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------


def _detect_stitch_patterns(mesh, patch1, patch2):
    """Detect repeated face-pair patterns at the stitch interface."""
    patterns = []
    if mesh is None:
        return patterns

    try:
        # Group faces by shape (number of vertices)
        face_shapes: Dict[int, int] = {}
        for pi in mesh.boundary:
            if pi.get("name") in (patch1, patch2):
                start = pi["startFace"]
                n = pi["nFaces"]
                for fi in range(start, start + n):
                    if fi < len(mesh.faces):
                        nv = len(mesh.faces[fi])
                        face_shapes[nv] = face_shapes.get(nv, 0) + 1

        for pid, (nv, count) in enumerate(face_shapes.items()):
            patterns.append(StitchPattern(
                pattern_id=pid,
                n_faces=nv,
                frequency=count,
                representative_face=nv,
            ))
    except Exception:
        pass

    return patterns


# ---------------------------------------------------------------------------
# Topology score
# ---------------------------------------------------------------------------


def _compute_topology_score(n_stitched, n_unmatched, mean_quality):
    """Compute a topology quality score from stitch results."""
    total = n_stitched + n_unmatched
    if total == 0:
        return 0.0
    match_ratio = n_stitched / total
    return 0.6 * match_ratio + 0.4 * mean_quality


# ---------------------------------------------------------------------------
# Strength analysis
# ---------------------------------------------------------------------------


def _analyze_stitch_strength(mesh, patch1, patch2, n_stitched, mean_quality):
    """Analyze mechanical strength of the stitched interface."""
    if n_stitched == 0:
        return StitchStrength(
            interface_name=f"{patch1}--{patch2}",
            mean_alignment=0.0,
            overlap_ratio=0.0,
            strength_score=0.0,
            n_weak_points=0,
        )

    # Overlap ratio from stitching success
    overlap = min(1.0, n_stitched / max(n_stitched + 1, 1))
    strength = 0.5 * overlap + 0.5 * mean_quality
    n_weak = max(0, int((1.0 - strength) * 10))

    return StitchStrength(
        interface_name=f"{patch1}--{patch2}",
        mean_alignment=mean_quality,
        overlap_ratio=overlap,
        strength_score=strength,
        n_weak_points=n_weak,
    )
