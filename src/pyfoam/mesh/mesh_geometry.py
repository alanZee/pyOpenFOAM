"""
Geometric calculations for finite volume meshes.

All functions operate on PyTorch tensors and respect the global device/dtype
configuration from :mod:`pyfoam.core`.  They implement the standard OpenFOAM
geometric conventions:

- Face area vector = outward-pointing normal with magnitude equal to face area.
- Cell volume via tetrahedral decomposition from cell centre to each face.
- Cell centre = volume-weighted average of sub-tetrahedron centroids.
- Face weights for linear interpolation: w = |d_N| / (|d_P| + |d_N|).
- Delta coefficients: 1/|d · n| where d connects cell centres, n is face normal.
"""

from __future__ import annotations

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "compute_face_centres",
    "compute_face_area_vectors",
    "compute_cell_volumes_and_centres",
    "compute_face_weights",
    "compute_delta_coefficients",
]


# ---------------------------------------------------------------------------
# Face geometry
# ---------------------------------------------------------------------------


def compute_face_centres(
    points: torch.Tensor,
    faces: list[torch.Tensor],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute the geometric centre of each face.

    The face centre is the arithmetic mean of the face's vertex positions.

    Args:
        points: ``(n_points, 3)`` vertex positions.
        faces: List of ``(n_vertices_in_face,)`` int tensors, each containing
            point indices for that face.
        device: Target device.
        dtype: Target dtype.

    Returns:
        ``(n_faces, 3)`` face centre positions.
    """
    device = device or get_device()
    dtype = dtype or get_default_dtype()
    n_faces = len(faces)
    centres = torch.zeros(n_faces, 3, dtype=dtype, device=device)
    for i, face_pts in enumerate(faces):
        verts = points[face_pts.to(device=device)]
        centres[i] = verts.mean(dim=0)
    return centres


def compute_face_area_vectors(
    points: torch.Tensor,
    faces: list[torch.Tensor],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute the area vector (normal × area) for each face.

    For a general polygon with vertices :math:`v_0, \\ldots, v_{n-1}`:

    1. Compute the face centroid :math:`c = \\frac{1}{n}\\sum v_i`.
    2. Sum over edges: :math:`\\vec{A} = \\frac{1}{2}\\sum_i (v_i - c) \\times (v_{i+1} - c)`.

    The resulting vector has magnitude equal to the face area and direction
    along the face normal.  The sign follows the right-hand rule with respect
    to the vertex ordering (OpenFOAM uses counter-clockwise when viewed from
    the owner side).

    Args:
        points: ``(n_points, 3)`` vertex positions.
        faces: List of face-vertex index tensors.
        device: Target device.
        dtype: Target dtype.

    Returns:
        ``(n_faces, 3)`` area vectors.
    """
    device = device or get_device()
    dtype = dtype or get_default_dtype()
    n_faces = len(faces)
    area_vecs = torch.zeros(n_faces, 3, dtype=dtype, device=device)

    for i, face_pts in enumerate(faces):
        verts = points[face_pts.to(device=device)].to(dtype=dtype)
        n_v = verts.shape[0]
        if n_v < 3:
            continue
        centroid = verts.mean(dim=0)
        # Fan triangulation from centroid
        for j in range(n_v):
            e0 = verts[j] - centroid
            e1 = verts[(j + 1) % n_v] - centroid
            area_vecs[i] += torch.linalg.cross(e0, e1)
        area_vecs[i] *= 0.5
    return area_vecs


# ---------------------------------------------------------------------------
# Cell geometry
# ---------------------------------------------------------------------------


def _tet_volume_signed(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    """Signed volume of tetrahedron (a, b, c, d).

    V = (1/6) * (b-a) · ((c-a) × (d-a))
    """
    return torch.dot(b - a, torch.linalg.cross(c - a, d - a)) / 6.0


def _tet_centroid(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    """Centroid of tetrahedron (a, b, c, d)."""
    return (a + b + c + d) / 4.0


def compute_cell_volumes_and_centres(
    points: torch.Tensor,
    faces: list[torch.Tensor],
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    n_cells: int,
    n_internal_faces: int,
    face_centres: torch.Tensor | None = None,
    face_area_vectors: torch.Tensor | None = None,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cell volumes and cell centres.

    Algorithm:
    1. Build cell → face connectivity.
    2. For each cell, compute a reference point (mean of the cell's vertex
       positions).  For convex cells this is guaranteed to be inside the cell
       and not on any face plane, avoiding degenerate tetrahedra.
    3. For each face, triangulate from face centre.
    4. For each triangle (ref, fc, v_j, v_{j+1}), compute signed tet volume.
    5. Cell volume = |Σ signed tet volumes|.
    6. Cell centre = Σ(vol_tet * centroid_tet) / Σ(vol_tet).

    Args:
        points: ``(n_points, 3)`` vertex positions.
        faces: List of face-vertex index tensors.
        owner: ``(n_faces,)`` owner cell per face.
        neighbour: ``(n_internal_faces,)`` neighbour cell per internal face.
        n_cells: Total number of cells.
        n_internal_faces: Number of internal faces.
        face_centres: Pre-computed face centres (optional, computed if None).
        face_area_vectors: Pre-computed area vectors (optional, not used here
            but accepted for API consistency).
        device: Target device.
        dtype: Target dtype.

    Returns:
        ``(volumes, centres)`` — ``(n_cells,)`` and ``(n_cells, 3)`` tensors.
    """
    device = device or get_device()
    dtype = dtype or get_default_dtype()

    if face_centres is None:
        face_centres = compute_face_centres(points, faces, device=device, dtype=dtype)

    face_centres = face_centres.to(device=device, dtype=dtype)

    # Build cell → face list
    n_faces_total = len(faces)
    cell_faces: list[list[int]] = [[] for _ in range(n_cells)]
    for f in range(n_faces_total):
        cell_faces[owner[f].item()].append(f)
    for f in range(n_internal_faces):
        cell_faces[neighbour[f].item()].append(f)

    volumes = torch.zeros(n_cells, dtype=dtype, device=device)
    centres = torch.zeros(n_cells, 3, dtype=dtype, device=device)

    for cell in range(n_cells):
        cf_list = cell_faces[cell]
        if not cf_list:
            continue

        # Collect all unique vertex indices for this cell
        all_pt_indices: list[int] = []
        for fi in cf_list:
            all_pt_indices.extend(faces[fi].tolist())
        unique_pts = list(set(all_pt_indices))

        # Reference point: mean of cell vertices (inside cell for convex cells)
        ref = points[unique_pts].to(dtype=dtype).mean(dim=0)

        total_vol = torch.tensor(0.0, dtype=dtype, device=device)
        weighted_centre = torch.zeros(3, dtype=dtype, device=device)

        for fi in cf_list:
            face_pts = faces[fi]
            verts = points[face_pts.to(device=device)].to(dtype=dtype)
            n_v = verts.shape[0]
            if n_v < 3:
                continue

            # For internal faces, the normal points from owner to neighbour.
            # If this cell is the neighbour, the normal points inward, so we
            # must reverse the vertex ordering to get an outward normal.
            is_neighbour_face = (fi < n_internal_faces and
                                 neighbour[fi].item() == cell)
            if is_neighbour_face:
                verts = verts.flip(0)

            fc = face_centres[fi]
            # Fan triangulation from face centre
            for j in range(n_v):
                # Tet: ref, fc, v_j, v_{j+1}
                vol = _tet_volume_signed(ref, fc, verts[j], verts[(j + 1) % n_v])
                if vol.abs() < 1e-30:
                    continue
                cent = _tet_centroid(ref, fc, verts[j], verts[(j + 1) % n_v])
                total_vol += vol
                weighted_centre += vol * cent

        if total_vol.abs() > 1e-30:
            volumes[cell] = total_vol.abs()
            centres[cell] = weighted_centre / total_vol
        else:
            # Degenerate cell — fall back to mean of face centres
            fc_all = face_centres[cf_list]
            volumes[cell] = torch.tensor(0.0, dtype=dtype, device=device)
            centres[cell] = fc_all.mean(dim=0)

    return volumes, centres


# ---------------------------------------------------------------------------
# Interpolation weights
# ---------------------------------------------------------------------------


def compute_face_weights(
    cell_centres: torch.Tensor,
    face_centres: torch.Tensor,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    n_internal_faces: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute face interpolation weights for linear interpolation.

    For internal faces the weight is:

    .. math::

        w = \\frac{|d_N|}{|d_P| + |d_N|}

    where :math:`d_P` is the vector from the owner cell centre to the face
    centre and :math:`d_N` is the vector from the neighbour cell centre to
    the face centre.

    Boundary faces receive weight ``1.0`` (fully owner-based).

    Args:
        cell_centres: ``(n_cells, 3)`` cell centre positions.
        face_centres: ``(n_faces, 3)`` face centre positions.
        owner: ``(n_faces,)`` owner cell per face.
        neighbour: ``(n_internal_faces,)`` neighbour cell per internal face.
        n_internal_faces: Number of internal faces.
        device: Target device.
        dtype: Target dtype.

    Returns:
        ``(n_faces,)`` interpolation weights in ``[0, 1]``.
    """
    device = device or get_device()
    dtype = dtype or get_default_dtype()

    cell_centres = cell_centres.to(device=device, dtype=dtype)
    face_centres = face_centres.to(device=device, dtype=dtype)
    owner = owner.to(device=device)
    neighbour = neighbour.to(device=device)

    n_faces = owner.shape[0]
    weights = torch.ones(n_faces, dtype=dtype, device=device)

    if n_internal_faces == 0:
        return weights

    # Owner cell centres for internal faces
    owner_centres = cell_centres[owner[:n_internal_faces]]
    # Neighbour cell centres
    nbr_centres = cell_centres[neighbour[:n_internal_faces]]
    # Face centres for internal faces
    int_fc = face_centres[:n_internal_faces]

    d_P = (int_fc - owner_centres).norm(dim=1)  # |face_centre - owner_centre|
    d_N = (int_fc - nbr_centres).norm(dim=1)  # |face_centre - neighbour_centre|

    denom = d_P + d_N
    # Avoid division by zero
    safe_denom = torch.where(denom > 1e-30, denom, torch.ones_like(denom))
    weights[:n_internal_faces] = d_N / safe_denom

    return weights


def compute_delta_coefficients(
    cell_centres: torch.Tensor,
    face_centres: torch.Tensor,
    face_area_vectors: torch.Tensor,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    n_internal_faces: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute delta coefficients for diffusion discretisation.

    The delta coefficient for an internal face is:

    .. math::

        \\delta = \\frac{1}{|\\vec{d} \\cdot \\hat{n}|}

    where :math:`\\vec{d}` is the vector from the owner to the neighbour
    cell centre and :math:`\\hat{n}` is the unit face normal.

    Boundary faces receive ``0.0`` (not used in the same way).

    Args:
        cell_centres: ``(n_cells, 3)`` cell centre positions.
        face_centres: ``(n_faces, 3)`` face centre positions.
        face_area_vectors: ``(n_faces, 3)`` area vectors (normal × area).
        owner: ``(n_faces,)`` owner cell per face.
        neighbour: ``(n_internal_faces,)`` neighbour cell per internal face.
        n_internal_faces: Number of internal faces.
        device: Target device.
        dtype: Target dtype.

    Returns:
        ``(n_faces,)`` delta coefficients.  Internal faces have positive
        values; boundary faces are ``0.0``.
    """
    device = device or get_device()
    dtype = dtype or get_default_dtype()

    cell_centres = cell_centres.to(device=device, dtype=dtype)
    face_area_vectors = face_area_vectors.to(device=device, dtype=dtype)
    owner = owner.to(device=device)
    neighbour = neighbour.to(device=device)

    n_faces = owner.shape[0]
    delta = torch.zeros(n_faces, dtype=dtype, device=device)

    if n_internal_faces == 0:
        return delta

    # d = neighbour_centre - owner_centre
    d = cell_centres[neighbour[:n_internal_faces]] - cell_centres[owner[:n_internal_faces]]
    # Face normal (unit vector)
    area_vecs = face_area_vectors[:n_internal_faces]
    area_mag = area_vecs.norm(dim=1, keepdim=True)
    safe_area_mag = torch.where(area_mag > 1e-30, area_mag, torch.ones_like(area_mag))
    n_hat = area_vecs / safe_area_mag

    # d · n_hat
    d_dot_n = (d * n_hat).sum(dim=1).abs()
    safe_d_dot_n = torch.where(d_dot_n > 1e-30, d_dot_n, torch.ones_like(d_dot_n))
    delta[:n_internal_faces] = 1.0 / safe_d_dot_n

    return delta
