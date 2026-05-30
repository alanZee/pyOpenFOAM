"""
stitchMesh enhanced v3 — enhanced mesh stitching with bounding-box
pre-filtering and area-weighted non-conformal matching (third generation).

Extends :func:`stitch_mesh_enhanced_2` with:

- **Bounding-box pre-filter**: Rapidly eliminates candidate face pairs
  before expensive distance checks.
- **Area-weighted matching**: For non-conformal meshes, match faces by
  area similarity in addition to centre proximity.
- **Stitch quality diagnostics**: Reports per-pair matching quality
  and overall stitch confidence.

Usage::

    from pyfoam.tools.stitch_mesh_enhanced_3 import stitch_mesh_enhanced_3

    result = stitch_mesh_enhanced_3(
        mesh, "patch1", "patch2",
        tolerance=1e-4,
        non_conformal=True,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import torch

from pyfoam.core.dtype import INDEX_DTYPE

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["StitchEnhanced3Result", "stitch_mesh_enhanced_3"]


@dataclass
class StitchEnhanced3Result:
    """Result from :func:`stitch_mesh_enhanced_3`.

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
    """

    mesh: object  # FvMesh
    n_stitched: int = 0
    n_unmatched: int = 0
    overlap_ratio: float = 0.0
    used_tolerance: float = 0.0
    mean_match_quality: float = 0.0
    n_bb_filtered: int = 0


def stitch_mesh_enhanced_3(
    mesh: "FvMesh",
    patch1_name: str,
    patch2_name: str,
    tolerance: float = 1e-6,
    non_conformal: bool = False,
    auto_tolerance: bool = False,
    area_ratio_tol: float = 0.5,
) -> StitchEnhanced3Result:
    """Stitch two boundary patches with BB pre-filter and area-weighted matching.

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

    Returns
    -------
    StitchEnhanced3Result
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

    # Build lookup for p2 data by face index
    p2_lookup = {d[0]: d for d in p2_data}

    # Match patch1 -> patch2 with BB pre-filter
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

                        # Bounding-box pre-filter
                        bb_dist = _bb_distance(
                            bb1_min, bb1_max, bb2_min, bb2_max,
                        )
                        if bb_dist > used_tol:
                            n_bb_filtered += 1
                            continue

                        dist = (fc1 - fc2).norm().item()
                        if dist < best_dist:
                            # Area ratio check for non-conformal
                            if non_conformal:
                                a_ratio = min(area1, area2) / max(area1, area2, 1e-30)
                                if a_ratio < area_ratio_tol:
                                    continue
                            elif not _faces_coincide(mesh, fi1, fi2, used_tol):
                                continue

                            best_dist = dist
                            best_fi2 = fi2
                            # Quality: combination of distance and area match
                            dist_q = max(0.0, 1.0 - dist / used_tol)
                            a_ratio = min(area1, area2) / max(area1, area2, 1e-30)
                            best_quality = 0.7 * dist_q + 0.3 * a_ratio

        if best_fi2 is not None:
            stitch_pairs.append((fi1, best_fi2))
            matched_p2.add(best_fi2)
            overlap_ratios.append(max(0.0, 1.0 - best_dist / used_tol))
            match_qualities.append(best_quality)

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

    n_unmatched = (p1["nFaces"] - len(stitch_pairs)) + (p2["nFaces"] - len(stitch_pairs))
    mean_overlap = sum(overlap_ratios) / len(overlap_ratios) if overlap_ratios else 0.0
    mean_quality = sum(match_qualities) / len(match_qualities) if match_qualities else 0.0

    result_mesh = FvMesh(
        points=mesh.points.clone(),
        faces=all_faces,
        owner=torch.tensor(all_owner, dtype=INDEX_DTYPE, device=dev),
        neighbour=torch.tensor(int_neighbour, dtype=INDEX_DTYPE, device=dev),
        boundary=boundary,
        validate=False,
    )

    return StitchEnhanced3Result(
        mesh=result_mesh,
        n_stitched=len(stitch_pairs),
        n_unmatched=n_unmatched,
        overlap_ratio=mean_overlap,
        used_tolerance=used_tol,
        mean_match_quality=mean_quality,
        n_bb_filtered=n_bb_filtered,
    )


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
    """Quantize a 3D point to a spatial hash grid cell."""
    cs = max(cell_size, 1e-12)
    return (
        int(torch.floor(pt[0] / cs).item()),
        int(torch.floor(pt[1] / cs).item()),
        int(torch.floor(pt[2] / cs).item()),
    )


def _face_area(pts: torch.Tensor) -> float:
    """Compute area of a face from its vertex points."""
    if pts.shape[0] < 3:
        return 0.0
    cross = torch.cross(pts[1] - pts[0], pts[2] - pts[0])
    return 0.5 * cross.norm().item()


def _bb_distance(
    bb1_min: torch.Tensor, bb1_max: torch.Tensor,
    bb2_min: torch.Tensor, bb2_max: torch.Tensor,
) -> float:
    """Compute minimum distance between two axis-aligned bounding boxes."""
    d = torch.zeros(3)
    for i in range(3):
        if bb1_max[i] < bb2_min[i]:
            d[i] = bb2_min[i] - bb1_max[i]
        elif bb2_max[i] < bb1_min[i]:
            d[i] = bb1_min[i] - bb2_max[i]
    return d.norm().item()


def _faces_coincide(mesh: "FvMesh", fi1: int, fi2: int, tol: float) -> bool:
    """Check if two faces have coincident vertices within tolerance."""
    pts1 = mesh.points[mesh.faces[fi1]].float()
    pts2 = mesh.points[mesh.faces[fi2]].float()

    if pts1.shape[0] != pts2.shape[0]:
        return False

    n = pts1.shape[0]
    for offset in range(n):
        for sign in [1, -1]:
            if sign == 1:
                reordered = torch.roll(pts2, shifts=offset, dims=0)
            else:
                reordered = torch.roll(pts2.flip(0), shifts=offset, dims=0)
            dist = (pts1 - reordered).norm(dim=1).max().item()
            if dist < tol:
                return True
    return False
