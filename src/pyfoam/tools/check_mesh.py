"""
checkMesh — validate mesh quality metrics.

Mirrors OpenFOAM's ``checkMesh`` utility.  Computes per-face and per-cell
quality indicators and returns a structured result with pass/fail status.

Metrics
-------
- **Non-orthogonality** (degrees): angle between face normal and
  owner–neighbour cell-centre vector.  Threshold: 70° warn, 85° fail.
- **Skewness**: normalised distance from face centre to the intersection
  of the cell-connection line with the face plane.  Threshold: 2 warn, 4 fail.
- **Aspect ratio**: approximated as max/min distance from cell centre to
  adjacent face centres.  Threshold: 1000 warn, 10 000 fail.
- **Volume ratio**: max/min volume ratio across internal faces.
  Threshold: 100 warn, 1000 fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["CheckMeshResult", "check_mesh"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class CheckMeshResult:
    """Structured result from :func:`check_mesh`.

    Attributes
    ----------
    passed : bool
        ``True`` if all metrics are within fail thresholds.
    n_cells : int
        Number of cells in the mesh.
    n_internal_faces : int
        Number of internal faces.
    n_boundary_faces : int
        Number of boundary faces.
    min_non_orthogonality : float
        Minimum non-orthogonality angle (degrees) across internal faces.
    max_non_orthogonality : float
        Maximum non-orthogonality angle (degrees) across internal faces.
    average_non_orthogonality : float
        Mean non-orthogonality angle (degrees) across internal faces.
    n_warn_non_orthogonality : int
        Number of internal faces exceeding the warning threshold (70 deg).
    n_fail_non_orthogonality : int
        Number of internal faces exceeding the failure threshold (85 deg).
    max_skewness : float
        Maximum face skewness across internal faces.
    average_skewness : float
        Mean face skewness across internal faces.
    n_warn_skewness : int
        Number of internal faces exceeding the skewness warning threshold (2).
    n_fail_skewness : int
        Number of internal faces exceeding the skewness failure threshold (4).
    max_aspect_ratio : float
        Maximum cell aspect ratio.
    average_aspect_ratio : float
        Mean cell aspect ratio.
    max_volume_ratio : float
        Maximum volume ratio between adjacent cells across internal faces.
    min_cell_volume : float
        Smallest cell volume.
    max_cell_volume : float
        Largest cell volume.
    warnings : list[str]
        Human-readable warning messages.
    errors : list[str]
        Human-readable error messages.
    """

    passed: bool = True

    n_cells: int = 0
    n_internal_faces: int = 0
    n_boundary_faces: int = 0

    # Non-orthogonality
    min_non_orthogonality: float = 0.0
    max_non_orthogonality: float = 0.0
    average_non_orthogonality: float = 0.0
    n_warn_non_orthogonality: int = 0
    n_fail_non_orthogonality: int = 0

    # Skewness
    max_skewness: float = 0.0
    average_skewness: float = 0.0
    n_warn_skewness: int = 0
    n_fail_skewness: int = 0

    # Aspect ratio
    max_aspect_ratio: float = 0.0
    average_aspect_ratio: float = 0.0

    # Volume ratio
    max_volume_ratio: float = 0.0

    # Cell volumes
    min_cell_volume: float = 0.0
    max_cell_volume: float = 0.0

    # Diagnostics
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Thresholds (matching OpenFOAM defaults)
# ---------------------------------------------------------------------------

_NON_ORTHO_WARN = 70.0   # degrees
_NON_ORTHO_FAIL = 85.0   # degrees
_SKEWNESS_WARN = 2.0
_SKEWNESS_FAIL = 4.0
_ASPECT_RATIO_WARN = 1000.0
_ASPECT_RATIO_FAIL = 10000.0
_VOLUME_RATIO_WARN = 100.0
_VOLUME_RATIO_FAIL = 1000.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def check_mesh(mesh: "FvMesh") -> CheckMeshResult:
    """Validate mesh quality for an FvMesh.

    Computes geometric quality metrics and compares them against standard
    thresholds.  The mesh must have geometry already computed (call
    ``mesh.compute_geometry()`` first or access a geometry property).

    Args:
        mesh: Finite volume mesh with geometry computed.

    Returns:
        :class:`CheckMeshResult` with all quality metrics and diagnostics.
    """
    result = CheckMeshResult()

    n_cells = mesh.n_cells
    n_internal = mesh.n_internal_faces
    n_faces = mesh.n_faces

    result.n_cells = n_cells
    result.n_internal_faces = n_internal
    result.n_boundary_faces = n_faces - n_internal

    # Early exit for degenerate meshes
    if n_internal == 0 and n_cells == 0:
        result.passed = True
        result.warnings.append("Mesh has no cells or internal faces.")
        return result

    # Access geometry (triggers lazy computation)
    cell_centres = mesh.cell_centres
    cell_volumes = mesh.cell_volumes
    face_centres = mesh.face_centres
    face_area_vectors = mesh.face_areas
    owner = mesh.owner
    neighbour = mesh.neighbour

    # --- Cell volumes ---
    if n_cells > 0:
        vol_np = cell_volumes.detach().cpu().numpy()
        result.min_cell_volume = float(vol_np.min())
        result.max_cell_volume = float(vol_np.max())

        if result.min_cell_volume <= 0.0:
            result.passed = False
            result.errors.append(
                f"Found {int((vol_np <= 0).sum())} cell(s) with non-positive volume."
            )

    # --- Internal face metrics ---
    if n_internal > 0:
        _check_internal_faces(mesh, result, cell_centres, cell_volumes,
                              face_centres, face_area_vectors, owner, neighbour)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_internal_faces(
    mesh: "FvMesh",
    result: CheckMeshResult,
    cell_centres: torch.Tensor,
    cell_volumes: torch.Tensor,
    face_centres: torch.Tensor,
    face_area_vectors: torch.Tensor,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
) -> None:
    """Compute all internal-face quality metrics."""
    n_internal = mesh.n_internal_faces

    int_fc = face_centres[:n_internal]
    int_fav = face_area_vectors[:n_internal]
    own_idx = owner[:n_internal]
    nbr_idx = neighbour[:n_internal]

    # --- Non-orthogonality ---
    non_ortho = _compute_non_orthogonality(
        cell_centres, int_fc, int_fav, own_idx, nbr_idx
    )
    non_ortho_deg = non_ortho.detach().cpu().numpy()

    result.min_non_orthogonality = float(non_ortho_deg.min())
    result.max_non_orthogonality = float(non_ortho_deg.max())
    result.average_non_orthogonality = float(non_ortho_deg.mean())
    result.n_warn_non_orthogonality = int((non_ortho_deg > _NON_ORTHO_WARN).sum())
    result.n_fail_non_orthogonality = int((non_ortho_deg > _NON_ORTHO_FAIL).sum())

    if result.n_fail_non_orthogonality > 0:
        result.passed = False
        result.errors.append(
            f"{result.n_fail_non_orthogonality} face(s) exceed non-orthogonality "
            f"threshold of {_NON_ORTHO_FAIL}°."
        )
    if result.n_warn_non_orthogonality > 0:
        result.warnings.append(
            f"{result.n_warn_non_orthogonality} face(s) exceed non-orthogonality "
            f"warning threshold of {_NON_ORTHO_WARN}°."
        )

    # --- Skewness ---
    skewness = _compute_skewness(
        cell_centres, int_fc, int_fav, own_idx, nbr_idx
    )
    skew_np = skewness.detach().cpu().numpy()

    result.max_skewness = float(skew_np.max())
    result.average_skewness = float(skew_np.mean())
    result.n_warn_skewness = int((skew_np > _SKEWNESS_WARN).sum())
    result.n_fail_skewness = int((skew_np > _SKEWNESS_FAIL).sum())

    if result.n_fail_skewness > 0:
        result.passed = False
        result.errors.append(
            f"{result.n_fail_skewness} face(s) exceed skewness "
            f"threshold of {_SKEWNESS_FAIL}."
        )
    if result.n_warn_skewness > 0:
        result.warnings.append(
            f"{result.n_warn_skewness} face(s) exceed skewness "
            f"warning threshold of {_SKEWNESS_WARN}."
        )

    # --- Volume ratio ---
    vol_ratio = _compute_volume_ratio(cell_volumes, own_idx, nbr_idx)
    vr_np = vol_ratio.detach().cpu().numpy()

    result.max_volume_ratio = float(vr_np.max())
    if result.max_volume_ratio > _VOLUME_RATIO_FAIL:
        result.passed = False
        result.errors.append(
            f"Max volume ratio {result.max_volume_ratio:.1f} exceeds "
            f"threshold of {_VOLUME_RATIO_FAIL}."
        )
    if result.max_volume_ratio > _VOLUME_RATIO_WARN:
        result.warnings.append(
            f"Max volume ratio {result.max_volume_ratio:.1f} exceeds "
            f"warning threshold of {_VOLUME_RATIO_WARN}."
        )

    # --- Aspect ratio (per-cell metric) ---
    _check_aspect_ratio(mesh, result, cell_centres, face_centres, owner)


def _compute_non_orthogonality(
    cell_centres: torch.Tensor,
    face_centres: torch.Tensor,
    face_area_vectors: torch.Tensor,
    own_idx: torch.Tensor,
    nbr_idx: torch.Tensor,
) -> torch.Tensor:
    """Non-orthogonality angle (degrees) for internal faces.

    The angle is between the face normal and the vector connecting the
    owner and neighbour cell centres.
    """
    d = cell_centres[nbr_idx] - cell_centres[own_idx]
    d_mag = d.norm(dim=1)
    safe_d_mag = torch.where(d_mag > 1e-30, d_mag, torch.ones_like(d_mag))
    d_hat = d / safe_d_mag.unsqueeze(1)

    face_mag = face_area_vectors.norm(dim=1, keepdim=True)
    safe_face_mag = torch.where(face_mag > 1e-30, face_mag, torch.ones_like(face_mag))
    n_hat = face_area_vectors / safe_face_mag

    # cos(theta) = |d_hat . n_hat|
    cos_theta = (d_hat * n_hat).sum(dim=1).abs()
    cos_theta = cos_theta.clamp(0.0, 1.0)
    return torch.rad2deg(torch.acos(cos_theta))


def _compute_skewness(
    cell_centres: torch.Tensor,
    face_centres: torch.Tensor,
    face_area_vectors: torch.Tensor,
    own_idx: torch.Tensor,
    nbr_idx: torch.Tensor,
) -> torch.Tensor:
    """Face skewness for internal faces.

    Skewness = |e_parallel| / |d|

    where e_parallel is the component of (face_centre - mid-point of
    owner and neighbour centres) projected onto the cell-connection
    direction, and d is the distance between cell centres.
    """
    cc_own = cell_centres[own_idx]
    cc_nbr = cell_centres[nbr_idx]
    mid = 0.5 * (cc_own + cc_nbr)

    d = cc_nbr - cc_own
    d_mag = d.norm(dim=1, keepdim=True)
    safe_d_mag = torch.where(d_mag > 1e-30, d_mag, torch.ones_like(d_mag))
    d_hat = d / safe_d_mag

    # e = face_centre - midpoint of cell centres
    e = face_centres - mid
    # Project e onto d direction
    e_parallel = (e * d_hat).sum(dim=1, keepdim=True)
    # Skewness: |e_parallel| / |d|
    skew = e_parallel.abs() / safe_d_mag
    return skew.squeeze(1)


def _compute_volume_ratio(
    cell_volumes: torch.Tensor,
    own_idx: torch.Tensor,
    nbr_idx: torch.Tensor,
) -> torch.Tensor:
    """Volume ratio (max/min) between adjacent cells across internal faces."""
    vol_own = cell_volumes[own_idx]
    vol_nbr = cell_volumes[nbr_idx]
    max_vol = torch.maximum(vol_own, vol_nbr)
    min_vol = torch.minimum(vol_own, vol_nbr)
    safe_min = torch.where(min_vol > 1e-30, min_vol, torch.ones_like(min_vol))
    return max_vol / safe_min


def _check_aspect_ratio(
    mesh: "FvMesh",
    result: CheckMeshResult,
    cell_centres: torch.Tensor,
    face_centres: torch.Tensor,
    owner: torch.Tensor,
) -> None:
    """Approximate cell aspect ratio and update *result*.

    Aspect ratio for each cell is estimated as max/min distance from the
    cell centre to the centres of its faces.
    """
    from pyfoam.mesh.topology import build_cell_to_faces

    n_cells = mesh.n_cells
    cell_faces = build_cell_to_faces(owner, mesh.neighbour, n_cells, mesh.n_internal_faces)

    aspect_ratios = []
    for cell in range(n_cells):
        cf = cell_faces[cell]
        if len(cf) == 0:
            continue
        fc = face_centres[cf]
        cc = cell_centres[cell]
        dists = (fc - cc.unsqueeze(0)).norm(dim=1)
        d_min = dists.min().item()
        d_max = dists.max().item()
        if d_min > 1e-30:
            aspect_ratios.append(d_max / d_min)
        else:
            aspect_ratios.append(float("inf"))

    if not aspect_ratios:
        return

    import numpy as np
    ar_np = np.array(aspect_ratios)
    result.max_aspect_ratio = float(ar_np.max()) if np.isfinite(ar_np.max()) else float("inf")
    finite_ar = ar_np[np.isfinite(ar_np)]
    result.average_aspect_ratio = float(finite_ar.mean()) if len(finite_ar) > 0 else 0.0

    n_warn = int((ar_np > _ASPECT_RATIO_WARN).sum())
    n_fail = int((ar_np > _ASPECT_RATIO_FAIL).sum())

    if n_fail > 0:
        result.passed = False
        result.errors.append(
            f"{n_fail} cell(s) exceed aspect ratio threshold of {_ASPECT_RATIO_FAIL}."
        )
    if n_warn > 0:
        result.warnings.append(
            f"{n_warn} cell(s) exceed aspect ratio warning threshold of {_ASPECT_RATIO_WARN}."
        )
