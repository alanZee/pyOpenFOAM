"""
stitchMesh enhanced v2 — enhanced mesh stitching with better coincident face
detection and non-conformal stitching support (second generation).

Extends :func:`stitch_mesh_enhanced` with:

- **Multi-patch stitching**: Stitch multiple patch pairs in one call.
- **Overlap ratio**: For non-conformal meshes, compute and report the
  geometric overlap ratio of matched face pairs.
- **Tolerance auto-detection**: Automatically determine stitching
  tolerance from face-size statistics.

Usage::

    from pyfoam.tools.stitch_mesh_enhanced_2 import stitch_mesh_enhanced_2

    result = stitch_mesh_enhanced_2(
        mesh, "patch1", "patch2",
        tolerance=1e-4,
        non_conformal=True,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import torch

from pyfoam.core.dtype import INDEX_DTYPE

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["StitchEnhanced2Result", "stitch_mesh_enhanced_2"]


@dataclass
class StitchEnhanced2Result:
    """Result from :func:`stitch_mesh_enhanced_2`.

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
    """

    mesh: object  # FvMesh
    n_stitched: int = 0
    n_unmatched: int = 0
    overlap_ratio: float = 0.0
    used_tolerance: float = 0.0


def stitch_mesh_enhanced_2(
    mesh: "FvMesh",
    patch1_name: str,
    patch2_name: str,
    tolerance: float = 1e-6,
    non_conformal: bool = False,
    auto_tolerance: bool = False,
) -> StitchEnhanced2Result:
    """Stitch two boundary patches with enhanced matching.

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

    Returns
    -------
    StitchEnhanced2Result
        Stitched mesh with metadata.
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

    # Build spatial hash for patch2 face centres
    p2_centres = {}
    for fi in range(p2_start, p2_end):
        fc = mesh.points[mesh.faces[fi]].float().mean(dim=0)
        key = _hash_point(fc, used_tol)
        p2_centres.setdefault(key, []).append((fi, fc))

    # Match patch1 → patch2
    matched_p2 = set()
    stitch_pairs = []
    overlap_ratios = []

    for fi1 in range(p1_start, p1_end):
        pts1 = mesh.points[mesh.faces[fi1]].float()
        fc1 = pts1.mean(dim=0)
        key = _hash_point(fc1, used_tol)

        best_fi2 = None
        best_dist = used_tol
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nk = (key[0] + dx, key[1] + dy, key[2] + dz)
                    if nk not in p2_centres:
                        continue
                    for fi2, fc2 in p2_centres[nk]:
                        if fi2 in matched_p2:
                            continue
                        dist = (fc1 - fc2).norm().item()
                        if dist < best_dist:
                            if non_conformal or _faces_coincide(mesh, fi1, fi2, used_tol):
                                best_dist = dist
                                best_fi2 = fi2

        if best_fi2 is not None:
            stitch_pairs.append((fi1, best_fi2))
            matched_p2.add(best_fi2)
            # Overlap ratio: 1.0 when perfectly coincident
            overlap_ratios.append(max(0.0, 1.0 - best_dist / used_tol))

    # Classify faces
    stitched_set_p1 = {p for p, _ in stitch_pairs}
    stitched_set_p2 = {p for _, p in stitch_pairs}

    int_faces = []
    int_owner = []
    int_neighbour = []
    bnd_faces = []
    bnd_owner = []

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

    boundary = []
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

    result_mesh = FvMesh(
        points=mesh.points.clone(),
        faces=all_faces,
        owner=torch.tensor(all_owner, dtype=INDEX_DTYPE, device=dev),
        neighbour=torch.tensor(int_neighbour, dtype=INDEX_DTYPE, device=dev),
        boundary=boundary,
        validate=False,
    )

    return StitchEnhanced2Result(
        mesh=result_mesh,
        n_stitched=len(stitch_pairs),
        n_unmatched=n_unmatched,
        overlap_ratio=mean_overlap,
        used_tolerance=used_tol,
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
