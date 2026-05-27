"""
Enhanced mesh quality check — per-cell detailed metrics.

Provides :func:`check_mesh_quality` which extends the basic
:func:`~pyfoam.tools.check_mesh.check_mesh` with per-cell quality
metrics: face orthogonality, cell convexity, volume non-orthogonality,
and a per-cell quality score.

Metrics (per cell):
- **Orthogonality**: average non-orthogonality of internal faces
  adjacent to the cell.
- **Skewness**: maximum face skewness of internal faces of the cell.
- **Volume ratio**: ratio of largest to smallest adjacent cell volume.
- **Convexity**: check that the cell-centre-to-face-centre vector
  points outward relative to the face normal.
- **Quality score**: combined scalar metric in [0, 1] (1 = perfect).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["QualityReport", "check_mesh_quality"]


# ---------------------------------------------------------------------------
# Per-cell result container
# ---------------------------------------------------------------------------


@dataclass
class CellQuality:
    """Quality metrics for a single cell.

    Attributes
    ----------
    cell_id : int
        Cell index.
    n_faces : int
        Number of faces.
    non_orthogonality : float
        Average non-orthogonality (degrees) of internal faces.
    max_skewness : float
        Maximum skewness of adjacent internal faces.
    volume_ratio : float
        Volume ratio with adjacent cells (max/min).
    convexity_violated : bool
        ``True`` if any face-centre-to-cell-centre vector points
        inward relative to the face normal.
    quality_score : float
        Combined quality metric in [0, 1].  1 = perfect cell.
    """

    cell_id: int = 0
    n_faces: int = 0
    non_orthogonality: float = 0.0
    max_skewness: float = 0.0
    volume_ratio: float = 1.0
    convexity_violated: bool = False
    quality_score: float = 1.0


# ---------------------------------------------------------------------------
# Overall report
# ---------------------------------------------------------------------------


@dataclass
class QualityReport:
    """Detailed mesh quality report with per-cell metrics.

    Attributes
    ----------
    n_cells : int
        Number of cells.
    n_internal_faces : int
        Number of internal faces.
    cell_qualities : list[CellQuality]
        Per-cell quality metrics.
    min_quality_score : float
        Minimum quality score across all cells.
    max_quality_score : float
        Maximum quality score across all cells.
    mean_quality_score : float
        Mean quality score across all cells.
    n_convexity_violations : int
        Number of cells with convexity violations.
    n_poor_quality : int
        Number of cells with quality score < 0.5.
    warnings : list[str]
        Human-readable warning messages.
    errors : list[str]
        Human-readable error messages.
    passed : bool
        ``True`` if no errors.
    """

    n_cells: int = 0
    n_internal_faces: int = 0
    cell_qualities: list[CellQuality] = field(default_factory=list)

    min_quality_score: float = 1.0
    max_quality_score: float = 1.0
    mean_quality_score: float = 1.0

    n_convexity_violations: int = 0
    n_poor_quality: int = 0

    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    passed: bool = True


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_NON_ORTHO_WARN = 70.0
_NON_ORTHO_FAIL = 85.0
_SKEWNESS_WARN = 2.0
_SKEWNESS_FAIL = 4.0
_QUALITY_POOR = 0.5
_QUALITY_FAIL = 0.1


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def check_mesh_quality(mesh: "FvMesh") -> QualityReport:
    """Compute detailed per-cell quality metrics for a mesh.

    Returns a :class:`QualityReport` containing per-cell metrics
    including non-orthogonality, skewness, volume ratio, convexity,
    and a combined quality score.

    Parameters
    ----------
    mesh : FvMesh
        Finite volume mesh with geometry computed.

    Returns
    -------
    QualityReport
        Detailed quality report with per-cell metrics.
    """
    report = QualityReport()

    n_cells = mesh.n_cells
    n_internal = mesh.n_internal_faces

    report.n_cells = n_cells
    report.n_internal_faces = n_internal

    if n_cells == 0:
        report.warnings.append("Mesh has no cells.")
        return report

    # Access geometry
    cell_centres = mesh.cell_centres
    cell_volumes = mesh.cell_volumes
    face_centres = mesh.face_centres
    face_area_vectors = mesh.face_areas
    owner = mesh.owner
    neighbour = mesh.neighbour

    # Build cell-to-face adjacency for internal faces
    cell_faces = _build_cell_internal_faces(owner, neighbour, n_cells, n_internal)

    # Compute per-face non-orthogonality and skewness
    face_non_ortho = torch.zeros(n_internal, dtype=cell_centres.dtype)
    face_skewness = torch.zeros(n_internal, dtype=cell_centres.dtype)

    if n_internal > 0:
        face_non_ortho = _compute_non_orthogonality(
            cell_centres, face_centres[:n_internal],
            face_area_vectors[:n_internal],
            owner[:n_internal], neighbour[:n_internal],
        )
        face_skewness = _compute_skewness(
            cell_centres, face_centres[:n_internal],
            face_area_vectors[:n_internal],
            owner[:n_internal], neighbour[:n_internal],
        )

    # Compute volume ratios
    face_vol_ratio = torch.ones(n_internal, dtype=cell_centres.dtype)
    if n_internal > 0:
        face_vol_ratio = _compute_volume_ratio(
            cell_volumes, owner[:n_internal], neighbour[:n_internal],
        )

    # Compute convexity
    face_convexity = torch.zeros(n_internal, dtype=torch.bool)
    if n_internal > 0:
        face_convexity = _check_convexity(
            cell_centres, face_centres[:n_internal],
            face_area_vectors[:n_internal], owner[:n_internal],
            neighbour[:n_internal],
        )

    # Build per-cell metrics
    no_np = face_non_ortho.detach().cpu().numpy()
    sk_np = face_skewness.detach().cpu().numpy()
    vr_np = face_vol_ratio.detach().cpu().numpy()
    cv_np = face_convexity.detach().cpu().numpy()

    quality_scores = []

    for cell in range(n_cells):
        cf = cell_faces[cell]
        cq = CellQuality(cell_id=cell, n_faces=len(cf))

        if len(cf) > 0:
            cq.non_orthogonality = float(no_np[cf].mean())
            cq.max_skewness = float(sk_np[cf].max())
            cq.volume_ratio = float(vr_np[cf].max())
            cq.convexity_violated = bool(cv_np[cf].any())

        # Combined quality score (1 = perfect, 0 = terrible)
        score = _compute_quality_score(cq)
        cq.quality_score = score
        quality_scores.append(score)

        report.cell_qualities.append(cq)

    # Aggregate statistics
    scores = np.array(quality_scores)
    report.min_quality_score = float(scores.min())
    report.max_quality_score = float(scores.max())
    report.mean_quality_score = float(scores.mean())

    report.n_convexity_violations = sum(
        1 for cq in report.cell_qualities if cq.convexity_violated
    )
    report.n_poor_quality = int((scores < _QUALITY_POOR).sum())

    # Warnings and errors
    if report.n_convexity_violations > 0:
        report.passed = False
        report.errors.append(
            f"{report.n_convexity_violations} cell(s) have convexity violations."
        )

    if report.min_quality_score < _QUALITY_FAIL:
        report.passed = False
        report.errors.append(
            f"Minimum quality score {report.min_quality_score:.4f} < {_QUALITY_FAIL}."
        )

    if report.n_poor_quality > 0:
        report.warnings.append(
            f"{report.n_poor_quality} cell(s) have quality score < {_QUALITY_POOR}."
        )

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_cell_internal_faces(
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    n_cells: int,
    n_internal: int,
) -> list[list[int]]:
    """Build list of internal face indices for each cell."""
    cell_faces: list[list[int]] = [[] for _ in range(n_cells)]

    own = owner[:n_internal].cpu().tolist()
    nbr = neighbour[:n_internal].cpu().tolist()

    for i in range(n_internal):
        cell_faces[own[i]].append(i)
        cell_faces[nbr[i]].append(i)

    return cell_faces


def _compute_non_orthogonality(
    cell_centres: torch.Tensor,
    face_centres: torch.Tensor,
    face_area_vectors: torch.Tensor,
    own_idx: torch.Tensor,
    nbr_idx: torch.Tensor,
) -> torch.Tensor:
    """Non-orthogonality angle (degrees) for internal faces."""
    d = cell_centres[nbr_idx] - cell_centres[own_idx]
    d_mag = d.norm(dim=1)
    safe_d_mag = torch.where(d_mag > 1e-30, d_mag, torch.ones_like(d_mag))
    d_hat = d / safe_d_mag.unsqueeze(1)

    face_mag = face_area_vectors.norm(dim=1, keepdim=True)
    safe_face_mag = torch.where(face_mag > 1e-30, face_mag, torch.ones_like(face_mag))
    n_hat = face_area_vectors / safe_face_mag

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
    """Face skewness for internal faces."""
    cc_own = cell_centres[own_idx]
    cc_nbr = cell_centres[nbr_idx]
    mid = 0.5 * (cc_own + cc_nbr)

    d = cc_nbr - cc_own
    d_mag = d.norm(dim=1, keepdim=True)
    safe_d_mag = torch.where(d_mag > 1e-30, d_mag, torch.ones_like(d_mag))
    d_hat = d / safe_d_mag

    e = face_centres - mid
    e_parallel = (e * d_hat).sum(dim=1, keepdim=True)
    skew = e_parallel.abs() / safe_d_mag
    return skew.squeeze(1)


def _compute_volume_ratio(
    cell_volumes: torch.Tensor,
    own_idx: torch.Tensor,
    nbr_idx: torch.Tensor,
) -> torch.Tensor:
    """Volume ratio (max/min) between adjacent cells."""
    vol_own = cell_volumes[own_idx]
    vol_nbr = cell_volumes[nbr_idx]
    max_vol = torch.maximum(vol_own, vol_nbr)
    min_vol = torch.minimum(vol_own, vol_nbr)
    safe_min = torch.where(min_vol > 1e-30, min_vol, torch.ones_like(min_vol))
    return max_vol / safe_min


def _check_convexity(
    cell_centres: torch.Tensor,
    face_centres: torch.Tensor,
    face_area_vectors: torch.Tensor,
    owner_idx: torch.Tensor,
    neighbour_idx: torch.Tensor,
) -> torch.Tensor:
    """Check convexity condition at internal faces.

    A mesh is locally convex at a face if the owner and neighbour cell
    centres lie on opposite sides of the face plane.  The check computes:

        sign_o = sign((cc_owner - fc) . n)
        sign_n = sign((cc_neighbour - fc) . n)

    If sign_o == sign_n, both cells are on the same side of the face,
    which violates convexity.

    Parameters
    ----------
    owner_idx : torch.Tensor
        Owner cell indices for each internal face.
    neighbour_idx : torch.Tensor
        Neighbour cell indices for each internal face.

    Returns
    -------
    torch.Tensor
        Boolean tensor: ``True`` for faces where convexity is violated.
    """
    d_own = cell_centres[owner_idx] - face_centres
    d_nbr = cell_centres[neighbour_idx] - face_centres
    dot_own = (d_own * face_area_vectors).sum(dim=1)
    dot_nbr = (d_nbr * face_area_vectors).sum(dim=1)
    return (dot_own * dot_nbr) > 0.0  # same side → violation


def _compute_quality_score(cq: CellQuality) -> float:
    """Compute a combined quality score in [0, 1].

    The score is a weighted geometric mean of sub-scores:
    - Orthogonality score: 1 - non_ortho / 90
    - Skewness score: max(0, 1 - skewness / 4)
    - Volume ratio score: 1 / (1 + log(vol_ratio))
    - Convexity score: 0.5 if violated, 1.0 otherwise
    """
    # Orthogonality: perfect at 0 deg, worst at 90 deg
    ortho_score = max(0.0, 1.0 - cq.non_orthogonality / 90.0)

    # Skewness: perfect at 0, worst at 4
    skew_score = max(0.0, 1.0 - cq.max_skewness / 4.0)

    # Volume ratio: perfect at 1
    import math
    vr = max(cq.volume_ratio, 1.0)
    vr_score = 1.0 / (1.0 + math.log(vr)) if vr > 0 else 0.0

    # Convexity: 1.0 if OK, 0.5 if violated
    conv_score = 0.5 if cq.convexity_violated else 1.0

    # Weighted geometric mean
    weights = [0.3, 0.3, 0.2, 0.2]
    scores = [ortho_score, skew_score, vr_score, conv_score]

    log_score = sum(
        w * math.log(max(s, 1e-10)) for w, s in zip(weights, scores)
    )
    quality = math.exp(log_score)

    return max(0.0, min(1.0, quality))
