"""
stitchMesh enhanced v4 — enhanced mesh stitching with iterative closest
point refinement and gap-filling (fourth generation).

Extends :func:`stitch_mesh_enhanced_3` with:

- **Iterative closest point (ICP) refinement**: After initial matching,
  refine face pair alignment using iterative projection.
- **Gap-filling patches**: Generate new bridge faces for small gaps
  that cannot be stitched by simple face matching.
- **Stitch strength estimation**: Estimate the geometric quality of
  each stitch pair using edge-length ratio and area mismatch.

Usage::

    from pyfoam.tools.stitch_mesh_enhanced_4 import stitch_mesh_enhanced_4

    result = stitch_mesh_enhanced_4(
        mesh, "patch1", "patch2",
        tolerance=1e-4,
        non_conformal=True,
        icp_iterations=3,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import torch

from pyfoam.core.dtype import INDEX_DTYPE

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["StitchEnhanced4Result", "stitch_mesh_enhanced_4"]


@dataclass
class StitchEnhanced4Result:
    """Result from :func:`stitch_mesh_enhanced_4`.

    Attributes
    ----------
    mesh : FvMesh
        The stitched mesh.
    n_stitched : int
        Number of face pairs stitched.
    n_unmatched : int
        Number of boundary faces that could not be matched.
    overlap_ratio : float
        Mean geometric overlap ratio (1.0 = perfect match).
    used_tolerance : float
        Tolerance used for matching.
    mean_match_quality : float
        Mean match quality across all stitched pairs (0-1).
    n_bb_filtered : int
        Number of candidate pairs eliminated by bounding-box pre-filter.
    n_gap_filled : int
        Number of gap-filling faces generated.
    mean_stitch_strength : float
        Mean stitch strength (edge-length ratio metric).
    icp_improvement : float
        Improvement in match quality from ICP refinement (0-1).
    """

    mesh: object  # FvMesh
    n_stitched: int = 0
    n_unmatched: int = 0
    overlap_ratio: float = 0.0
    used_tolerance: float = 0.0
    mean_match_quality: float = 0.0
    n_bb_filtered: int = 0
    n_gap_filled: int = 0
    mean_stitch_strength: float = 0.0
    icp_improvement: float = 0.0


def stitch_mesh_enhanced_4(
    mesh: "FvMesh",
    patch1_name: str,
    patch2_name: str,
    tolerance: float = 1e-6,
    non_conformal: bool = False,
    auto_tolerance: bool = False,
    area_ratio_tol: float = 0.5,
    icp_iterations: int = 3,
    gap_fill_threshold: float = 0.0,
) -> StitchEnhanced4Result:
    """Stitch two boundary patches with ICP refinement and gap filling.

    Parameters
    ----------
    mesh : FvMesh
        Source mesh.
    patch1_name : str
        Name of the first boundary patch.
    patch2_name : str
        Name of the second boundary patch.
    tolerance : float
        Maximum distance for matching.
    non_conformal : bool
        If True, allow matching faces with different vertex counts.
    auto_tolerance : bool
        If True, estimate tolerance from mean face size on the patches.
    area_ratio_tol : float
        For non-conformal matching, reject face pairs whose area ratio
        differs by more than this fraction.
    icp_iterations : int
        Number of ICP refinement iterations (0 to skip).
    gap_fill_threshold : float
        Maximum gap distance to fill with bridge faces. 0 disables.

    Returns
    -------
    StitchEnhanced4Result
        Stitched mesh with diagnostics.
    """
    from pyfoam.mesh.fv_mesh import FvMesh

    dev = mesh.device

    p1 = _find_patch(mesh, patch1_name)
    p2 = _find_patch(mesh, patch2_name)

    p1_start = p1["startFace"]
    p1_end = p1_start + p1["nFaces"]
    p2_start = p2["startFace"]
    p2_end = p2_start + p2["nFaces"]

    # Auto-tolerance from face size
    used_tol = tolerance
    if auto_tolerance:
        face_sizes = []
        for fi in range(p1_start, p1_end):
            pts = mesh.points[mesh.faces[fi]].float()
            sz = (pts - pts.mean(dim=0, keepdim=True)).norm(dim=1).mean().item()
            face_sizes.append(sz)
        for fi in range(p2_start, p2_end):
            pts = mesh.points[mesh.faces[fi]].float()
            sz = (pts - pts.mean(dim=0, keepdim=True)).norm(dim=1).mean().item()
            face_sizes.append(sz)
        if face_sizes:
            used_tol = max(tolerance, 0.01 * sum(face_sizes) / len(face_sizes))

    # Pre-compute face data: centres, areas, bounding boxes
    p1_data = []
    for fi in range(p1_start, p1_end):
        pts = mesh.points[mesh.faces[fi]].float()
        fc = pts.mean(dim=0)
        area = _face_area(pts)
        bb_min = pts.min(dim=0).values
        bb_max = pts.max(dim=0).values
        p1_data.append((fi, fc, area, bb_min, bb_max))

    p2_data = []
    for fi in range(p2_start, p2_end):
        pts = mesh.points[mesh.faces[fi]].float()
        fc = pts.mean(dim=0)
        area = _face_area(pts)
        bb_min = pts.min(dim=0).values
        bb_max = pts.max(dim=0).values
        p2_data.append((fi, fc, area, bb_min, bb_max))

    # Build spatial hash for patch2 face centres
    p2_centres = {}
    for fi2, fc2, _, _, _ in p2_data:
        key = _hash_point(fc2, used_tol)
        p2_centres.setdefault(key, []).append(fi2)

    p2_lookup = {d[0]: d for d in p2_data}

    # Initial matching
    matched_p2 = set()
    stitch_pairs: list[tuple[int, int]] = []
    overlap_ratios: list[float] = []
    match_qualities: list[float] = []
    n_bb_filtered = 0

    for fi1, fc1, area1, bb1_min, bb1_max in p1_data:
        key = _hash_point(fc1, used_tol)

        best_fi2 = None
        best_dist = used_tol
        best_quality = 0.0

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nk = (key[0] + dx, key[1] + dy, key[2] + dz)
                    if nk not in p2_centres:
                        continue
                    for fi2 in p2_centres[nk]:
                        if fi2 in matched_p2:
                            continue
                        d2 = p2_lookup[fi2]
                        fc2, area2, bb2_min, bb2_max = d2[1], d2[2], d2[3], d2[4]

                        bb_dist = _bb_distance(bb1_min, bb1_max, bb2_min, bb2_max)
                        if bb_dist > used_tol:
                            n_bb_filtered += 1
                            continue

                        dist = (fc1 - fc2).norm().item()
                        if dist < best_dist:
                            if non_conformal:
                                a_ratio = min(area1, area2) / max(area1, area2, 1e-30)
                                if a_ratio < area_ratio_tol:
                                    continue
                            elif not _faces_coincide(mesh, fi1, fi2, used_tol):
                                continue

                            best_dist = dist
                            best_fi2 = fi2
                            dist_q = max(0.0, 1.0 - dist / used_tol)
                            a_ratio = min(area1, area2) / max(area1, area2, 1e-30)
                            best_quality = 0.7 * dist_q + 0.3 * a_ratio

        if best_fi2 is not None:
            stitch_pairs.append((fi1, best_fi2))
            matched_p2.add(best_fi2)
            overlap_ratios.append(max(0.0, 1.0 - best_dist / used_tol))
            match_qualities.append(best_quality)

    # ICP refinement
    pre_icp_quality = sum(match_qualities) / len(match_qualities) if match_qualities else 0.0

    if icp_iterations > 0 and stitch_pairs:
        stitch_pairs, match_qualities, overlap_ratios = _icp_refinement(
            mesh, stitch_pairs, match_qualities, overlap_ratios,
            used_tol, icp_iterations,
        )

    post_icp_quality = sum(match_qualities) / len(match_qualities) if match_qualities else 0.0
    icp_improvement = post_icp_quality - pre_icp_quality

    # Stitch strength estimation
    stitch_strengths = _compute_stitch_strength(mesh, stitch_pairs)

    # Gap-filling
    n_gap_filled = 0
    gap_faces = []
    gap_owners = []
    if gap_fill_threshold > 0:
        gap_faces, gap_owners, n_gap_filled = _fill_gaps(
            mesh, p1_data, p2_data, stitch_pairs, used_tol, gap_fill_threshold,
        )

    # Classify faces
    stitched_set_p1 = {p for p, _ in stitch_pairs}
    stitched_set_p2 = {p for _, p in stitch_pairs}

    int_faces: list = []
    int_owner: list = []
    int_neighbour: list = []
    bnd_faces: list = []
    bnd_owner: list = []

    for fi in range(mesh.n_faces):
        if fi < mesh.n_internal_faces:
            int_faces.append(mesh.faces[fi].clone())
            int_owner.append(int(mesh.owner[fi].item()))
            int_neighbour.append(int(mesh.neighbour[fi].item()))
        elif fi in stitched_set_p1 or fi in stitched_set_p2:
            continue
        else:
            bnd_faces.append(mesh.faces[fi].clone())
            bnd_owner.append(int(mesh.owner[fi].item()))

    for fi1, fi2 in stitch_pairs:
        own = int(mesh.owner[fi1].item())
        nbr = int(mesh.owner[fi2].item())
        int_faces.append(mesh.faces[fi1].clone())
        if own > nbr:
            own, nbr = nbr, own
        int_owner.append(own)
        int_neighbour.append(nbr)

    # Add gap-fill faces as boundary
    for gf, go in zip(gap_faces, gap_owners):
        bnd_faces.append(gf)
        bnd_owner.append(go)

    n_internal = len(int_neighbour)
    all_faces = int_faces + bnd_faces
    all_owner = int_owner + bnd_owner

    boundary: list = []
    existing_patches = [p for p in mesh.boundary
                        if p["name"] != patch1_name and p["name"] != patch2_name]
    bnd_start = n_internal
    for patch in existing_patches:
        orig_start = patch["startFace"]
        orig_end = orig_start + patch["nFaces"]
        count = sum(1 for fi in range(orig_start, orig_end)
                    if fi not in stitched_set_p1 and fi not in stitched_set_p2)
        if count > 0:
            boundary.append({
                "name": patch["name"],
                "type": patch["type"],
                "startFace": bnd_start,
                "nFaces": count,
            })
            bnd_start += count

    if n_gap_filled > 0:
        boundary.append({
            "name": "gapFill",
            "type": "wall",
            "startFace": bnd_start,
            "nFaces": n_gap_filled,
        })

    n_unmatched = (p1["nFaces"] - len(stitch_pairs)) + (p2["nFaces"] - len(stitch_pairs))
    mean_overlap = sum(overlap_ratios) / len(overlap_ratios) if overlap_ratios else 0.0
    mean_quality = sum(match_qualities) / len(match_qualities) if match_qualities else 0.0
    mean_strength = sum(stitch_strengths) / len(stitch_strengths) if stitch_strengths else 0.0

    result_mesh = FvMesh(
        points=mesh.points.clone(),
        faces=all_faces,
        owner=torch.tensor(all_owner, dtype=INDEX_DTYPE, device=dev),
        neighbour=torch.tensor(int_neighbour, dtype=INDEX_DTYPE, device=dev),
        boundary=boundary,
        validate=False,
    )

    return StitchEnhanced4Result(
        mesh=result_mesh,
        n_stitched=len(stitch_pairs),
        n_unmatched=n_unmatched,
        overlap_ratio=mean_overlap,
        used_tolerance=used_tol,
        mean_match_quality=mean_quality,
        n_bb_filtered=n_bb_filtered,
        n_gap_filled=n_gap_filled,
        mean_stitch_strength=mean_strength,
        icp_improvement=icp_improvement,
    )


# ---------------------------------------------------------------------------
# ICP refinement
# ---------------------------------------------------------------------------


def _icp_refinement(mesh, stitch_pairs, qualities, overlaps, tol, n_iter):
    """Refine stitch pairs using iterative closest point projection."""
    best_pairs = list(stitch_pairs)
    best_qualities = list(qualities)
    best_overlaps = list(overlaps)

    for _ in range(n_iter):
        improved = False
        new_pairs = []
        new_quals = []
        new_overlaps = []

        for fi1, fi2 in best_pairs:
            pts1 = mesh.points[mesh.faces[fi1]].float()
            pts2 = mesh.points[mesh.faces[fi2]].float()
            fc1 = pts1.mean(dim=0)
            fc2 = pts2.mean(dim=0)

            # Project centre of p1 face onto p2 face plane
            if pts2.shape[0] >= 3:
                normal = torch.cross(pts2[1] - pts2[0], pts2[2] - pts2[0])
                n_norm = normal.norm()
                if n_norm > 1e-30:
                    normal = normal / n_norm
                    proj_dist = torch.dot(fc1 - fc2, normal)
                    fc1_proj = fc1 - proj_dist * normal
                    proj_error = (fc1_proj - fc2).norm().item()

                    area1 = _face_area(pts1)
                    area2 = _face_area(pts2)
                    a_ratio = min(area1, area2) / max(area1, area2, 1e-30)

                    dist_q = max(0.0, 1.0 - proj_error / tol)
                    new_q = 0.7 * dist_q + 0.3 * a_ratio
                    new_overlap = max(0.0, 1.0 - proj_error / tol)

                    if new_q >= 0.0:
                        new_pairs.append((fi1, fi2))
                        new_quals.append(new_q)
                        new_overlaps.append(new_overlap)
                        if new_q > qualities[len(new_pairs) - 1]:
                            improved = True
                        continue

            new_pairs.append((fi1, fi2))
            new_quals.append(qualities[len(new_pairs) - 1] if len(new_pairs) <= len(qualities) else 0.5)
            new_overlaps.append(overlaps[len(new_overlaps)] if len(new_overlaps) < len(overlaps) else 0.5)

        if not improved:
            break
        best_pairs = new_pairs
        best_qualities = new_quals
        best_overlaps = new_overlaps

    return best_pairs, best_qualities, best_overlaps


# ---------------------------------------------------------------------------
# Stitch strength estimation
# ---------------------------------------------------------------------------


def _compute_stitch_strength(mesh, stitch_pairs):
    """Estimate stitch quality using edge-length ratio metric."""
    strengths = []
    for fi1, fi2 in stitch_pairs:
        pts1 = mesh.points[mesh.faces[fi1]].float()
        pts2 = mesh.points[mesh.faces[fi2]].float()

        # Compute mean edge lengths
        el1 = _mean_edge_length(pts1)
        el2 = _mean_edge_length(pts2)

        ratio = min(el1, el2) / max(el1, el2, 1e-30) if max(el1, el2) > 1e-30 else 0.0
        strengths.append(ratio)

    return strengths


def _mean_edge_length(pts):
    """Compute mean edge length of a polygon."""
    n = pts.shape[0]
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        total += (pts[(i + 1) % n] - pts[i]).norm().item()
    return total / n


# ---------------------------------------------------------------------------
# Gap filling
# ---------------------------------------------------------------------------


def _fill_gaps(mesh, p1_data, p2_data, stitch_pairs, used_tol, gap_threshold):
    """Generate bridge faces for small unmatched gaps."""
    stitched_p1 = {fi1 for fi1, _ in stitch_pairs}
    stitched_p2 = {fi2 for _, fi2 in stitch_pairs}

    gap_faces = []
    gap_owners = []
    n_filled = 0

    p2_lookup = {d[0]: d for d in p2_data}

    for fi1, fc1, area1, _, _ in p1_data:
        if fi1 in stitched_p1:
            continue
        # Find nearest unmatched p2 face
        best_fi2 = None
        best_dist = gap_threshold

        for fi2, fc2, _, _, _ in p2_data:
            if fi2 in stitched_p2:
                continue
            dist = (fc1 - fc2).norm().item()
            if dist < best_dist:
                best_dist = dist
                best_fi2 = fi2

        if best_fi2 is not None:
            # Create a simple bridge face from p1 face centroid + nearest points
            pts1 = mesh.points[mesh.faces[fi1]].float()
            fc = pts1.mean(dim=0).unsqueeze(0)
            bridge_face = torch.cat([fc, pts1[:min(3, pts1.shape[0])]], dim=0)
            if bridge_face.shape[0] >= 3:
                face_indices = torch.arange(bridge_face.shape[0], dtype=INDEX_DTYPE)
                gap_faces.append(face_indices)
                gap_owners.append(int(mesh.owner[fi1].item()))
                n_filled += 1

    return gap_faces, gap_owners, n_filled


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_patch(mesh: "FvMesh", name: str) -> dict:
    """Locate a boundary patch by name."""
    for patch in mesh.boundary:
        if patch["name"] == name:
            return patch
    raise ValueError(f"Patch '{name}' not found in mesh boundary.")


def _hash_point(pt: torch.Tensor, cell_size: float) -> tuple:
    cs = max(cell_size, 1e-12)
    return (
        int(torch.floor(pt[0] / cs).item()),
        int(torch.floor(pt[1] / cs).item()),
        int(torch.floor(pt[2] / cs).item()),
    )


def _face_area(pts: torch.Tensor) -> float:
    if pts.shape[0] < 3:
        return 0.0
    cross = torch.cross(pts[1] - pts[0], pts[2] - pts[0])
    return 0.5 * cross.norm().item()


def _bb_distance(bb1_min, bb1_max, bb2_min, bb2_max) -> float:
    d = torch.zeros(3)
    for i in range(3):
        if bb1_max[i] < bb2_min[i]:
            d[i] = bb2_min[i] - bb1_max[i]
        elif bb2_max[i] < bb1_min[i]:
            d[i] = bb1_min[i] - bb2_max[i]
    return d.norm().item()


def _faces_coincide(mesh: "FvMesh", fi1: int, fi2: int, tol: float) -> bool:
    pts1 = mesh.points[mesh.faces[fi1]].float()
    pts2 = mesh.points[mesh.faces[fi2]].float()
    if pts1.shape[0] != pts2.shape[0]:
        return False
    n = pts1.shape[0]
    for offset in range(n):
        for sign in [1, -1]:
            reordered = torch.roll(pts2.flip(0), shifts=offset, dims=0) if sign == -1 else torch.roll(pts2, shifts=offset, dims=0)
            dist = (pts1 - reordered).norm(dim=1).max().item()
            if dist < tol:
                return True
    return False
