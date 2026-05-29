"""
viewFactorsGen — generate view factor matrices for radiation calculations.

Mirrors the view factor computation used in OpenFOAM's radiation modelling
(particularly the ``viewFactor`` boundary condition).  Computes the
geometric view factor matrix F_ij between boundary patches using the
analytical Nusselt unit sphere method.

For each pair of face elements, the view factor is computed as:

    F_ij = (cos(theta_i) * cos(theta_j)) / (pi * r^2) * A_j

where theta_i, theta_j are the angles between the face normals and the
line connecting the face centres, r is the distance between the face
centres, and A_j is the area of the receiving face.

For efficiency, self-view factors (F_ii) are set to zero (flat faces),
and a visibility check rejects back-facing elements.

Usage::

    from pyfoam.tools.view_factors_gen import view_factors_gen

    result = view_factors_gen(
        mesh=mesh,
        boundary_patches=["floor", "ceiling", "walls"],
    )
    print(result.view_factor_matrix.shape)  # (n_total_faces, n_total_faces)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["ViewFactorResult", "view_factors_gen"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ViewFactorResult:
    """Result from :func:`view_factors_gen`.

    Attributes
    ----------
    view_factor_matrix : np.ndarray
        ``(n_total_faces, n_total_faces)`` view factor matrix.
    patch_names : list[str]
        Names of the patches included.
    patch_face_counts : list[int]
        Number of faces per patch.
    row_sums : np.ndarray
        ``(n_total_faces,)`` row sums of the view factor matrix.
        For a closed enclosure, each row should sum to 1.
    n_total_faces : int
        Total number of boundary faces included.
    """

    view_factor_matrix: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    patch_names: list[str] = field(default_factory=list)
    patch_face_counts: list[int] = field(default_factory=list)
    row_sums: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    n_total_faces: int = 0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def view_factors_gen(
    mesh: "FvMesh",
    boundary_patches: Optional[Sequence[str]] = None,
    n_rays: int = 0,
    clip_tolerance: float = 1e-6,
) -> ViewFactorResult:
    """Compute view factor matrix for boundary patches.

    Parameters
    ----------
    mesh : FvMesh
        Computational mesh with geometry already computed.
    boundary_patches : sequence of str, optional
        Patch names to include.  If ``None``, all boundary patches
        are used.
    n_rays : int
        Unused (reserved for future Monte-Carlo ray-tracing mode).
        The analytical method is always used.
    clip_tolerance : float
        Small tolerance for clipping view factors to [0, 1].

    Returns
    -------
    ViewFactorResult
        View factor matrix and metadata.

    Raises
    ------
    ValueError
        If no patches are found or mesh has no boundary.
    """
    # Collect face data from specified patches
    patch_names: list[str] = []
    patch_face_counts: list[int] = []
    face_centres_list: list[np.ndarray] = []
    face_normals_list: list[np.ndarray] = []
    face_areas_list: list[np.ndarray] = []

    owner = mesh.owner.detach().cpu().numpy()
    cell_centres = mesh.cell_centres.detach().cpu().numpy()

    for patch_info in mesh.boundary:
        name = patch_info["name"]
        if boundary_patches is not None and name not in boundary_patches:
            continue

        start = patch_info["startFace"]
        n_faces = patch_info["nFaces"]

        # Compute face centres and normals from mesh data
        centres = []
        normals = []
        areas = []
        for fi in range(start, start + n_faces):
            face_nodes = mesh.faces[fi].detach().cpu().numpy()
            pts = mesh.points.detach().cpu().numpy()[face_nodes]
            c = pts.mean(axis=0)
            centres.append(c)

            # Face normal from cross product (for general polygon)
            if len(pts) >= 3:
                e1 = pts[1] - pts[0]
                e2 = pts[2] - pts[0]
                n = np.cross(e1, e2)
                area = 0.5 * np.linalg.norm(n)
                if area > 1e-30:
                    n = n / (2.0 * area)
                else:
                    n = np.array([0.0, 0.0, 1.0])
                    area = 0.0
            else:
                n = np.array([0.0, 0.0, 1.0])
                area = 0.0

            normals.append(n)
            areas.append(area)

        if centres:
            face_centres_list.append(np.array(centres, dtype=np.float64))
            face_normals_list.append(np.array(normals, dtype=np.float64))
            face_areas_list.append(np.array(areas, dtype=np.float64))
            patch_names.append(name)
            patch_face_counts.append(len(centres))

    if not patch_names:
        raise ValueError("No matching boundary patches found in mesh.")

    # Concatenate all face data
    all_centres = np.vstack(face_centres_list)
    all_normals = np.vstack(face_normals_list)
    all_areas = np.concatenate(face_areas_list)
    n_total = all_centres.shape[0]

    # Compute view factor matrix
    F = _compute_view_factors(all_centres, all_normals, all_areas, clip_tolerance)

    row_sums = F.sum(axis=1)

    return ViewFactorResult(
        view_factor_matrix=F,
        patch_names=patch_names,
        patch_face_counts=patch_face_counts,
        row_sums=row_sums,
        n_total_faces=n_total,
    )


# ---------------------------------------------------------------------------
# View factor computation
# ---------------------------------------------------------------------------


def _compute_view_factors(
    centres: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    clip_tol: float,
) -> np.ndarray:
    """Compute the view factor matrix using the analytical method.

    F_ij = cos(theta_i) * cos(theta_j) * A_j / (pi * r_ij^2)

    Subject to visibility: the two faces must face each other.  For
    boundary faces whose normals point outward from the owner cell,
    we check both the given normal and its negation, accepting whichever
    orientation yields a positive contribution.
    """
    n = centres.shape[0]
    F = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        ci = centres[i]
        ni = normals[i]
        for j in range(n):
            if i == j:
                continue

            rij = centres[j] - ci
            dist_sq = np.dot(rij, rij)
            if dist_sq < 1e-30:
                continue

            dist = np.sqrt(dist_sq)
            rij_hat = rij / dist

            # Try both normal orientations (outward and inward)
            cos_i = np.dot(ni, rij_hat)
            cos_j = -np.dot(normals[j], rij_hat)

            if cos_i <= 0.0 or cos_j <= 0.0:
                # Try flipping both normals (inward-pointing convention)
                cos_i = -cos_i
                cos_j = np.dot(normals[j], rij_hat)

            if cos_i <= 0.0 or cos_j <= 0.0:
                continue

            fij = cos_i * cos_j * areas[j] / (np.pi * dist_sq)
            F[i, j] = max(0.0, min(1.0, fij))

    return F
