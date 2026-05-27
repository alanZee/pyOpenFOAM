"""
Gradient schemes — compute cell-centre gradients from cell-centre values.

Provides the abstract base class and concrete gradient reconstruction
schemes used by ``fvm.grad`` / ``fvc.grad`` operators.

Schemes
-------
GaussLinearGrad
    Gauss theorem with linear face interpolation (default).
LeastSquaresGrad
    Least-squares gradient reconstruction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from pyfoam.core.backend import gather, scatter_add
from pyfoam.discretisation.weights import compute_centre_weights

__all__ = [
    "GradScheme",
    "GaussLinearGrad",
    "LeastSquaresGrad",
    "resolve_grad_scheme",
]


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class GradScheme(ABC):
    """Abstract base for gradient reconstruction schemes.

    A gradient scheme computes cell-centre gradient vectors from
    cell-centre scalar values and mesh geometry.

    All schemes return ``(n_cells, 3)`` gradient tensors.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    """

    def __init__(self, mesh) -> None:
        self._mesh = mesh

    @property
    def mesh(self):
        """The finite volume mesh."""
        return self._mesh

    @abstractmethod
    def compute_grad(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute cell-centre gradient of *phi*.

        Args:
            phi: ``(n_cells,)`` scalar field values.

        Returns:
            ``(n_cells, 3)`` gradient vector at each cell.
        """

    def __call__(self, phi: torch.Tensor) -> torch.Tensor:
        """Callable interface — delegates to :meth:`compute_grad`."""
        return self.compute_grad(phi)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mesh={self._mesh})"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_GRAD_REGISTRY: dict[str, type[GradScheme]] = {}


def _register_grad(name: str):
    """Class decorator that registers a gradient scheme by *name*."""
    def decorator(cls: type[GradScheme]) -> type[GradScheme]:
        _GRAD_REGISTRY[name] = cls
        return cls
    return decorator


def resolve_grad_scheme(scheme_name: str, mesh) -> GradScheme:
    """Create a gradient scheme instance from a name string.

    Supports ``"Gauss <name>"`` prefix (stripped) and bare names.

    Args:
        scheme_name: Scheme name (e.g. ``"Gauss linear"``, ``"leastSquares"``).
        mesh: The ``FvMesh``.

    Returns:
        A :class:`GradScheme` instance.

    Raises:
        ValueError: If *scheme_name* is not registered.
    """
    name = scheme_name.strip()
    if name.startswith("Gauss "):
        name = name[6:].strip()

    if name not in _GRAD_REGISTRY:
        raise ValueError(
            f"Unknown grad scheme '{scheme_name}'. "
            f"Available: {list(_GRAD_REGISTRY.keys())}"
        )
    return _GRAD_REGISTRY[name](mesh)


# ---------------------------------------------------------------------------
# Least-squares gradient reconstruction
# ---------------------------------------------------------------------------


@_register_grad("leastSquares")
class LeastSquaresGrad(GradScheme):
    """Least-squares gradient reconstruction (overlapping-matrix form).

    For each cell *P*, the gradient :math:`\\nabla\\phi_P` is found by
    minimising:

    .. math::

        \\min_{\\nabla\\phi_P}
        \\sum_{N} w_N^2
        \\bigl[ (\\phi_N - \\phi_P)
              - \\nabla\\phi_P \\cdot (\\mathbf{x}_N - \\mathbf{x}_P) \\bigr]^2

    where the sum runs over all face-connected neighbours (and boundary
    ghost cells derived from boundary face geometry), and

    .. math::

        w_N = \\frac{1}{|\\mathbf{x}_N - \\mathbf{x}_P|}

    The resulting normal equations are:

    .. math::

        (A^\\top W A)\\, \\nabla\\phi = A^\\top W\\, \\Delta\\phi

    where *A* is the direction matrix, *W* is the diagonal weight matrix,
    and *Dphi* is the value difference vector.  The cell-local coefficient
    matrix :math:`M = A^\\top W A` (shape ``[3, 3]``) and RHS vector
    :math:`r = A^\\top W\\, Dphi` (shape ``[3]``) are assembled into
    cell-indexed scatter buffers, then the gradient is solved per cell
    via Cramer's rule (no dense batched ``solve`` needed).

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    """

    def __init__(self, mesh) -> None:
        super().__init__(mesh)

        device = mesh.device
        dtype = mesh.dtype
        cc = mesh.cell_centres.to(device=device, dtype=dtype)
        fc = mesh.face_centres.to(device=device, dtype=dtype)
        owner = mesh.owner.to(device=device)
        neighbour = mesh.neighbour.to(device=device)
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces
        n_cells = mesh.n_cells

        # -- Build (direction, weight) pairs for each (cell, neighbour) ----
        directions = []  # (neighbour_dir)  shape (K, 3)
        weights = []     # (weight)         shape (K,)
        cell_idx = []    # (cell_index)     shape (K,)
        nb_val_idx = []  # (neighbour_cell_idx)  shape (K,)
        has_boundary_neighbour = [False] * n_cells

        # Internal face pairs
        if n_internal > 0:
            int_own = owner[:n_internal]
            int_nei = neighbour[:n_internal]
            cc_P = cc[int_own]  # (n_int, 3)
            cc_N = cc[int_nei]  # (n_int, 3)

            d = cc_N - cc_P  # (n_int, 3)
            dist = d.norm(dim=1).clamp(min=1e-30)  # (n_int,)
            w = 1.0 / dist  # (n_int,)

            # owner side: neighbour is int_nei
            cell_idx.append(int_own)
            nb_val_idx.append(int_nei)
            directions.append(d)
            weights.append(w)

            # neighbour side: neighbour is int_own
            cell_idx.append(int_nei)
            nb_val_idx.append(int_own)
            directions.append(-d)
            weights.append(w)

            has_boundary_neighbour = [False] * n_cells

        # Boundary ghost cells
        if n_faces > n_internal:
            bnd_own = owner[n_internal:]
            bnd_fc = fc[n_internal:]
            cc_bnd_P = cc[bnd_own]  # (n_bnd, 3)

            d_bnd = bnd_fc - cc_bnd_P  # (n_bnd, 3)
            dist_bnd = d_bnd.norm(dim=1).clamp(min=1e-30)  # (n_bnd,)
            w_bnd = 1.0 / dist_bnd  # (n_bnd,)

            cell_idx.append(bnd_own)
            # Boundary ghost cell index: encode as -(i + 1) to distinguish
            # from valid interior cell indices (0..n_cells-1).
            nb_val_idx.append(
                -(torch.arange(n_faces - n_internal, device=device) + 1)
            )
            directions.append(d_bnd)
            weights.append(w_bnd)

            for c in bnd_own.tolist():
                has_boundary_neighbour[c] = True

        cell_idx = torch.cat(cell_idx)  # (K,)
        nb_val_idx = torch.cat(nb_val_idx)  # (K,)
        directions = torch.cat(directions)  # (K, 3)
        weights = torch.cat(weights)  # (K,)

        # -- Assemble normal equations per cell ----
        # M_cell[i] = sum_k  w_k^2 * d_k[i] * d_k[j]   (3 x 3 per cell)
        # r_cell[i] = sum_k  w_k^2 * dphi_k * d_k[i]    (3 per cell)
        w2 = weights * weights  # (K,)
        wd = w2.unsqueeze(-1) * directions  # (K, 3)

        M_buf = torch.zeros(n_cells, 3, 3, dtype=dtype, device=device)
        # M[k, i, j] = wd[k, i] * directions[k, j]
        for i in range(3):
            for j in range(3):
                M_buf[:, i, j] = scatter_add(
                    wd[:, i] * directions[:, j], cell_idx, n_cells
                )

        # Store assembly metadata for compute_grad
        self._cell_idx = cell_idx
        self._nb_val_idx = nb_val_idx
        self._wd = wd
        self._M_buf = M_buf
        self._n_cells = n_cells
        self._n_faces = n_faces
        self._n_internal = n_internal
        self._owner = owner
        self._has_boundary_neighbour = has_boundary_neighbour

    def compute_grad(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute gradient via least-squares reconstruction.

        Args:
            phi: ``(n_cells,)`` scalar field values.

        Returns:
            ``(n_cells, 3)`` gradient vector at each cell.
        """
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_cells = self._n_cells
        n_faces = self._n_faces
        n_internal = self._n_internal

        phi = phi.to(device=device, dtype=dtype)

        # Build phi_neighbour values (interior + boundary ghost)
        phi_nb = gather(phi, self._nb_val_idx.clamp(min=0))
        if n_faces > n_internal:
            bnd_mask = self._nb_val_idx < 0
            if bnd_mask.any():
                bnd_own_phi = gather(phi, self._owner[n_internal:])
                # Map negative indices back to the boundary entry range
                bnd_own_full = torch.zeros_like(phi_nb)
                bnd_positions = torch.where(bnd_mask)[0]
                bnd_own_full[bnd_positions] = bnd_own_phi
                phi_nb = torch.where(bnd_mask, bnd_own_full, phi_nb)

        dphi = phi_nb - gather(phi, self._cell_idx)  # (K,)

        # RHS: r_cell[i] = sum_k  wd[k, i] * dphi[k]
        rhs = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        for i in range(3):
            rhs[:, i] = scatter_add(self._wd[:, i] * dphi, self._cell_idx, n_cells)

        # Solve per cell via Cramer's rule (M is 3x3 symmetric positive-definite)
        grad_phi = _solve_3x3_cramer(self._M_buf, rhs)

        # Fallback: cells with no LSQ neighbours (isolated) → zero gradient
        grad_phi[torch.isnan(grad_phi)] = 0.0

        return grad_phi


# ---------------------------------------------------------------------------
# Gauss-linear gradient
# ---------------------------------------------------------------------------


@_register_grad("linear")
class GaussLinearGrad(GradScheme):
    """Gauss theorem with linear face interpolation (default scheme).

    For each cell *P*:

    .. math::

        \\nabla\\phi_P =
        \\frac{1}{V_P} \\sum_f \\phi_f \\, \\mathbf{S}_f

    where :math:`\\phi_f` is linearly interpolated from owner and
    neighbour cell values using distance-based weights.

    Boundary face values use the owner cell value (zero-gradient
    approximation).

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    """

    def __init__(self, mesh) -> None:
        super().__init__(mesh)
        device = mesh.device
        dtype = mesh.dtype

        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces

        # Pre-compute interpolation weights
        w_all = compute_centre_weights(
            mesh.cell_centres,
            mesh.face_centres,
            mesh.owner,
            mesh.neighbour,
            n_internal,
            n_faces,
            device=device,
            dtype=dtype,
        )
        self._w = w_all[:n_internal]  # (n_internal,)
        self._face_areas = mesh.face_areas.to(device=device, dtype=dtype)
        self._cell_volumes = mesh.cell_volumes.to(device=device, dtype=dtype)

    def compute_grad(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute gradient via Gauss theorem with linear interpolation.

        Args:
            phi: ``(n_cells,)`` scalar field values.

        Returns:
            ``(n_cells, 3)`` gradient vector at each cell.
        """
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces

        phi = phi.to(device=device, dtype=dtype)

        # Linear interpolation to internal faces
        int_own = mesh.owner[:n_internal]
        int_nei = mesh.neighbour[:n_internal]

        phi_P = gather(phi, int_own)
        phi_N = gather(phi, int_nei)
        w = self._w

        phi_face = torch.zeros(n_faces, dtype=dtype, device=device)
        phi_face[:n_internal] = w * phi_P + (1.0 - w) * phi_N

        # Boundary faces: owner value (zero-gradient BC)
        if n_faces > n_internal:
            phi_face[n_internal:] = gather(phi, mesh.owner[n_internal:])

        # Gauss theorem: grad = (1/V) * sum_f phi_f * S_f
        face_contrib = phi_face.unsqueeze(-1) * self._face_areas  # (n_faces, 3)

        grad_phi = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad_phi.index_add_(0, int_own, face_contrib[:n_internal])
        grad_phi.index_add_(0, int_nei, -face_contrib[:n_internal])
        if n_faces > n_internal:
            grad_phi.index_add_(0, mesh.owner[n_internal:], face_contrib[n_internal:])

        V = self._cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        return grad_phi / V


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solve_3x3_cramer(
    A: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Solve *Ax = b* for 3x3 matrices via Cramer's rule.

    Args:
        A: ``(n, 3, 3)`` coefficient matrices.
        b: ``(n, 3)`` right-hand side vectors.

    Returns:
        ``(n, 3)`` solution vectors.
    """
    # Extract matrix components
    a00, a01, a02 = A[:, 0, 0], A[:, 0, 1], A[:, 0, 2]
    a10, a11, a12 = A[:, 1, 0], A[:, 1, 1], A[:, 1, 2]
    a20, a21, a22 = A[:, 2, 0], A[:, 2, 1], A[:, 2, 2]

    # det(A)
    det = (
        a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
    )
    safe_det = torch.where(det.abs() > 1e-30, det, torch.ones_like(det))

    b0, b1, b2 = b[:, 0], b[:, 1], b[:, 2]

    # x0 = det([b, col1, col2]) / det(A)
    x0 = (
        b0 * (a11 * a22 - a12 * a21)
        - a01 * (b1 * a22 - a12 * b2)
        + a02 * (b1 * a21 - a11 * b2)
    ) / safe_det

    # x1 = det([col0, b, col2]) / det(A)
    x1 = (
        a00 * (b1 * a22 - a12 * b2)
        - b0 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * b2 - b1 * a20)
    ) / safe_det

    # x2 = det([col0, col1, b]) / det(A)
    x2 = (
        a00 * (a11 * b2 - b1 * a21)
        - a01 * (a10 * b2 - b1 * a20)
        + b0 * (a10 * a21 - a11 * a20)
    ) / safe_det

    # Zero out solutions for singular cells (det ≈ 0)
    singular = det.abs() <= 1e-30
    x0 = torch.where(singular, torch.zeros_like(x0), x0)
    x1 = torch.where(singular, torch.zeros_like(x1), x1)
    x2 = torch.where(singular, torch.zeros_like(x2), x2)

    return torch.stack([x0, x1, x2], dim=1)
