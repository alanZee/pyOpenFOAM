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
FourthGrad
    Fourth-order gradient using extended stencil (higher-order Gauss).
CellLimitedGrad
    Cell-limited gradient to prevent overshoots / undershoots.
FaceLimitedGrad
    Face-limited gradient — limits per face rather than per cell.
GaussLinearCorrectedGrad
    Gauss linear with non-orthogonal correction.
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
    "FourthGrad",
    "FourthGrad2",
    "CellLimitedGrad",
    "CellLimitedGrad2",
    "FaceLimitedGrad",
    "FaceLimitedGrad2",
    "FourthGrad3",
    "FourthGrad4",
    "CellLimitedGrad3",
    "CellLimitedGrad4",
    "FaceLimitedGrad3",
    "FaceLimitedGrad4",
    "FourthGrad5",
    "CellLimitedGrad5",
    "FaceLimitedGrad5",
    "GaussLinearCorrectedGrad",
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
# Fourth-order gradient (higher-order Gauss)
# ---------------------------------------------------------------------------


@_register_grad("fourth")
class FourthGrad(GradScheme):
    r"""Fourth-order gradient using extended stencil.

    Improves on the standard Gauss linear scheme by adding a
    face-curvature correction.  For each cell *P*:

    .. math::

        \nabla\phi_P =
        \frac{1}{V_P} \sum_f \phi_f \, \mathbf{S}_f
        + \frac{1}{V_P} \sum_f
          \left[ \frac{1}{12}\,(\nabla\phi_f - (\nabla\phi_f \cdot \hat{n}_f)\hat{n}_f)
          \cdot \mathbf{S}_f \right]

    The correction accounts for the variation of *phi* across the face,
    providing fourth-order accuracy on structured meshes.

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

        # Pre-compute interpolation weights (same as GaussLinearGrad)
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
        self._w = w_all[:n_internal]
        self._face_areas = mesh.face_areas.to(device=device, dtype=dtype)
        self._cell_volumes = mesh.cell_volumes.to(device=device, dtype=dtype)

    def compute_grad(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute gradient via higher-order Gauss theorem.

        First pass: standard Gauss-linear to get a provisional gradient.
        Second pass: use the provisional gradient to add face-curvature
        correction for fourth-order accuracy.
        """
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces

        phi = phi.to(device=device, dtype=dtype)

        # --- First pass: standard Gauss-linear gradient ---
        int_own = mesh.owner[:n_internal]
        int_nei = mesh.neighbour[:n_internal]

        phi_P = gather(phi, int_own)
        phi_N = gather(phi, int_nei)
        w = self._w

        phi_face = torch.zeros(n_faces, dtype=dtype, device=device)
        phi_face[:n_internal] = w * phi_P + (1.0 - w) * phi_N
        if n_faces > n_internal:
            phi_face[n_internal:] = gather(phi, mesh.owner[n_internal:])

        face_contrib = phi_face.unsqueeze(-1) * self._face_areas
        grad_phi = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad_phi.index_add_(0, int_own, face_contrib[:n_internal])
        grad_phi.index_add_(0, int_nei, -face_contrib[:n_internal])
        if n_faces > n_internal:
            grad_phi.index_add_(
                0, mesh.owner[n_internal:], face_contrib[n_internal:],
            )

        V = self._cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        grad_phi = grad_phi / V

        # --- Second pass: face-curvature correction ---
        if n_internal > 0:
            # Interpolate gradient to internal faces
            grad_P = grad_phi[int_own]  # (n_int, 3)
            grad_N = grad_phi[int_nei]
            wt = w.unsqueeze(-1)
            grad_face = wt * grad_P + (1.0 - wt) * grad_N  # (n_int, 3)

            # Face normal
            S = self._face_areas[:n_internal]  # (n_int, 3)
            S_mag = S.norm(dim=1, keepdim=True).clamp(min=1e-30)
            n_hat = S / S_mag

            # Remove normal component: tangential gradient
            grad_normal = (grad_face * n_hat).sum(dim=1, keepdim=True)
            grad_tangential = grad_face - grad_normal * n_hat

            # Fourth-order correction: (1/12) * grad_tangential * S
            correction = (1.0 / 12.0) * grad_tangential * S

            corr = torch.zeros(n_cells, 3, dtype=dtype, device=device)
            corr.index_add_(0, int_own, correction)
            corr.index_add_(0, int_nei, -correction)

            grad_phi = grad_phi + corr / V

        return grad_phi


# ---------------------------------------------------------------------------
# Cell-limited gradient
# ---------------------------------------------------------------------------


@_register_grad("cellLimited")
class CellLimitedGrad(GradScheme):
    r"""Cell-limited gradient to prevent overshoots.

    Computes an unlimited gradient (default: Gauss linear), then limits
    it so that face values extrapolated from the cell centre do not
    exceed the cell's neighbours' min/max:

    .. math::

        \nabla\phi_P^{\text{lim}} = \alpha_P \, \nabla\phi_P

    where :math:`\alpha_P \in [0, 1]` is the minimum limiter coefficient
    across all faces of cell *P*, ensuring:

    .. math::

        \phi_P + \nabla\phi_P \cdot (\mathbf{x}_f - \mathbf{x}_P)
        \in [\phi_{\min},\; \phi_{\max}]

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    base_scheme : type[GradScheme], optional
        The unlimited gradient scheme class.  Default is
        :class:`GaussLinearGrad`.
    """

    def __init__(self, mesh, base_scheme=None) -> None:
        super().__init__(mesh)
        if base_scheme is None:
            base_scheme = GaussLinearGrad
        self._base = base_scheme(mesh)

    def compute_grad(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute cell-limited gradient."""
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces

        phi = phi.to(device=device, dtype=dtype)

        # Unlimited gradient from base scheme
        grad_unlimited = self._base.compute_grad(phi)  # (n_cells, 3)

        if n_internal == 0:
            return grad_unlimited

        cc = mesh.cell_centres.to(device=device, dtype=dtype)
        fa = mesh.face_centres.to(device=device, dtype=dtype)

        int_own = mesh.owner[:n_internal]
        int_nei = mesh.neighbour[:n_internal]

        # Compute face-extrapolated values from unlimited gradient
        # delta_Pf = x_f - x_P for owner side
        delta_P = fa[:n_internal] - cc[int_own]  # (n_int, 3)
        delta_N = fa[:n_internal] - cc[int_nei]  # (n_int, 3)

        grad_P = grad_unlimited[int_own]  # (n_int, 3)
        grad_N = grad_unlimited[int_nei]

        phi_P = gather(phi, int_own)
        phi_N = gather(phi, int_nei)

        # Extrapolated face values
        phi_ext_P = phi_P + (grad_P * delta_P).sum(dim=1)  # (n_int,)
        phi_ext_N = phi_N + (grad_N * delta_N).sum(dim=1)

        # Min/max at each face (from owner and neighbour cell values)
        phi_face_min = torch.min(phi_P, phi_N)  # (n_int,)
        phi_face_max = torch.max(phi_P, phi_N)

        # Limiter per face (owner side and neighbour side)
        eps = 1e-30

        # Owner limiter: how much can we extrapolate from P toward face?
        diff_P = phi_ext_P - phi_P  # extrapolated - cell-centre
        max_allow_P = phi_face_max - phi_P
        min_allow_P = phi_face_min - phi_P

        alpha_P = torch.ones(n_internal, dtype=dtype, device=device)
        pos_mask = diff_P > eps
        neg_mask = diff_P < -eps
        alpha_P = torch.where(
            pos_mask,
            torch.clamp(max_allow_P / (diff_P + eps), 0.0, 1.0),
            alpha_P,
        )
        alpha_P = torch.where(
            neg_mask,
            torch.clamp(min_allow_P / (diff_P - eps), 0.0, 1.0),
            alpha_P,
        )

        # Neighbour limiter
        diff_N = phi_ext_N - phi_N
        max_allow_N = phi_face_max - phi_N
        min_allow_N = phi_face_min - phi_N

        alpha_N = torch.ones(n_internal, dtype=dtype, device=device)
        pos_mask = diff_N > eps
        neg_mask = diff_N < -eps
        alpha_N = torch.where(
            pos_mask,
            torch.clamp(max_allow_N / (diff_N + eps), 0.0, 1.0),
            alpha_N,
        )
        alpha_N = torch.where(
            neg_mask,
            torch.clamp(min_allow_N / (diff_N - eps), 0.0, 1.0),
            alpha_N,
        )

        # Take minimum alpha across all faces for each cell
        cell_alpha = torch.ones(n_cells, dtype=dtype, device=device)
        # Owner side
        cell_alpha.scatter_reduce_(
            0, int_own, alpha_P, reduce="amin", include_self=True,
        )
        # Neighbour side
        cell_alpha.scatter_reduce_(
            0, int_nei, alpha_N, reduce="amin", include_self=True,
        )

        return cell_alpha.unsqueeze(-1) * grad_unlimited


# ---------------------------------------------------------------------------
# Face-limited gradient
# ---------------------------------------------------------------------------


@_register_grad("faceLimited")
class FaceLimitedGrad(GradScheme):
    r"""Face-limited gradient — limits per face rather than per cell.

    Similar to :class:`CellLimitedGrad` but applies the limiter at each
    face independently, producing a face-limited gradient that can be
    more accurate than the cell-limited version on meshes with mixed
    cell quality.

    The algorithm:

    1. Compute unlimited gradient (default Gauss linear).
    2. For each face, compute a limiter that prevents the extrapolated
       face value from exceeding the neighbour min/max.
    3. Scatter the limited gradient contribution back to cells.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    base_scheme : type[GradScheme], optional
        The unlimited gradient scheme class.  Default is
        :class:`GaussLinearGrad`.
    """

    def __init__(self, mesh, base_scheme=None) -> None:
        super().__init__(mesh)
        if base_scheme is None:
            base_scheme = GaussLinearGrad
        self._base = base_scheme(mesh)

        device = mesh.device
        dtype = mesh.dtype
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces

        # Pre-compute interpolation weights for face-limited correction
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
        self._w = w_all[:n_internal]
        self._face_areas = mesh.face_areas.to(device=device, dtype=dtype)
        self._cell_volumes = mesh.cell_volumes.to(device=device, dtype=dtype)

    def compute_grad(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute face-limited gradient."""
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces

        phi = phi.to(device=device, dtype=dtype)

        # Unlimited gradient from base scheme
        grad_unlimited = self._base.compute_grad(phi)  # (n_cells, 3)

        if n_internal == 0:
            return grad_unlimited

        cc = mesh.cell_centres.to(device=device, dtype=dtype)
        fa = mesh.face_centres.to(device=device, dtype=dtype)
        int_own = mesh.owner[:n_internal]
        int_nei = mesh.neighbour[:n_internal]

        # Extrapolate face values from each side using unlimited gradient
        delta_P = fa[:n_internal] - cc[int_own]
        delta_N = fa[:n_internal] - cc[int_nei]

        grad_P = grad_unlimited[int_own]
        grad_N = grad_unlimited[int_nei]
        phi_P = gather(phi, int_own)
        phi_N = gather(phi, int_nei)

        phi_ext_P = phi_P + (grad_P * delta_P).sum(dim=1)
        phi_ext_N = phi_N + (grad_N * delta_N).sum(dim=1)

        phi_face_min = torch.min(phi_P, phi_N)
        phi_face_max = torch.max(phi_P, phi_N)

        eps = 1e-30

        # Face limiter for owner side
        diff_P = phi_ext_P - phi_P
        alpha_P = torch.ones(n_internal, dtype=dtype, device=device)
        pos_mask = diff_P > eps
        neg_mask = diff_P < -eps
        alpha_P = torch.where(
            pos_mask,
            torch.clamp((phi_face_max - phi_P) / (diff_P + eps), 0.0, 1.0),
            alpha_P,
        )
        alpha_P = torch.where(
            neg_mask,
            torch.clamp((phi_face_min - phi_P) / (diff_P - eps), 0.0, 1.0),
            alpha_P,
        )

        # Face limiter for neighbour side
        diff_N = phi_ext_N - phi_N
        alpha_N = torch.ones(n_internal, dtype=dtype, device=device)
        pos_mask = diff_N > eps
        neg_mask = diff_N < -eps
        alpha_N = torch.where(
            pos_mask,
            torch.clamp((phi_face_max - phi_N) / (diff_N + eps), 0.0, 1.0),
            alpha_N,
        )
        alpha_N = torch.where(
            neg_mask,
            torch.clamp((phi_face_min - phi_N) / (diff_N - eps), 0.0, 1.0),
            alpha_N,
        )

        # Per-face limited gradient: interpolate limited grad to face
        grad_limited_P = alpha_P.unsqueeze(-1) * grad_P
        grad_limited_N = alpha_N.unsqueeze(-1) * grad_N

        # Use the average limited gradient at the face
        w = self._w.unsqueeze(-1)
        grad_face_lim = w * grad_limited_P + (1.0 - w) * grad_limited_N

        # Gauss integration with limited face gradient
        # grad_P = (1/V) * sum_f (phi_f * S_f)
        # where phi_f = w*phi_P + (1-w)*phi_N (standard linear interp)
        # plus correction from limited gradient at face
        phi_face = w.squeeze(-1) * phi_P + (1.0 - w.squeeze(-1)) * phi_N

        face_contrib = phi_face.unsqueeze(-1) * self._face_areas[:n_internal]
        grad_phi = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad_phi.index_add_(0, int_own, face_contrib)
        grad_phi.index_add_(0, int_nei, -face_contrib)

        # Add boundary contributions
        if n_faces > n_internal:
            bnd_own = mesh.owner[n_internal:]
            phi_bnd = gather(phi, bnd_own)
            bnd_contrib = phi_bnd.unsqueeze(-1) * self._face_areas[n_internal:]
            grad_phi.index_add_(0, bnd_own, bnd_contrib)

        V = self._cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        grad_phi = grad_phi / V

        # Blend with unlimited gradient using the average limiting factor
        # For cells where all faces are unlimited, alpha ~ 1 (no change)
        alpha_cell = torch.ones(n_cells, dtype=dtype, device=device)
        alpha_cell.scatter_reduce_(
            0, int_own, alpha_P, reduce="amin", include_self=True,
        )
        alpha_cell.scatter_reduce_(
            0, int_nei, alpha_N, reduce="amin", include_self=True,
        )

        return alpha_cell.unsqueeze(-1) * grad_phi + \
            (1.0 - alpha_cell.unsqueeze(-1)) * grad_unlimited


# ---------------------------------------------------------------------------
# Gauss-linear corrected gradient (non-orthogonal correction)
# ---------------------------------------------------------------------------


@_register_grad("linear corrected")
class GaussLinearCorrectedGrad(GradScheme):
    r"""Gauss linear with non-orthogonal correction.

    Extends :class:`GaussLinearGrad` by adding a non-orthogonal
    correction at each face.  For each internal face *f*:

    .. math::

        \phi_f = w\,\phi_P + (1 - w)\,\phi_N
        + \mathbf{k}_f \cdot \nabla\phi_f

    where :math:`\mathbf{k}_f` is the minimum-correction vector:

    .. math::

        \mathbf{k}_f = \hat{\mathbf{n}}_f
        - \hat{\mathbf{d}}_f \, (\hat{\mathbf{d}}_f \cdot \hat{\mathbf{n}}_f)

    This scheme is more accurate than plain Gauss linear on
    non-orthogonal meshes.

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

        # Interpolation weights
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
        self._w = w_all[:n_internal]
        self._face_areas = mesh.face_areas.to(device=device, dtype=dtype)
        self._cell_volumes = mesh.cell_volumes.to(device=device, dtype=dtype)

        # Pre-compute correction vectors (minimum correction approach)
        if n_internal > 0:
            cc = mesh.cell_centres.to(device=device, dtype=dtype)
            fa = mesh.face_areas.to(device=device, dtype=dtype)

            d = cc[mesh.neighbour[:n_internal]] - cc[mesh.owner[:n_internal]]
            d_mag = d.norm(dim=1, keepdim=True)
            d_hat = d / d_mag.clamp(min=1e-30)

            n_hat = fa[:n_internal] / fa[:n_internal].norm(
                dim=1, keepdim=True,
            ).clamp(min=1e-30)

            d_dot_n = (d_hat * n_hat).sum(dim=1, keepdim=True)
            self._k = n_hat - d_hat * d_dot_n  # (n_int, 3)
        else:
            self._k = torch.zeros(0, 3, dtype=dtype, device=device)

    def compute_grad(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute corrected Gauss linear gradient (iterative)."""
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces

        phi = phi.to(device=device, dtype=dtype)

        int_own = mesh.owner[:n_internal]
        int_nei = mesh.neighbour[:n_internal]

        # --- Pass 1: standard Gauss linear (provisional gradient) ---
        phi_P = gather(phi, int_own)
        phi_N = gather(phi, int_nei)
        w = self._w

        phi_face = torch.zeros(n_faces, dtype=dtype, device=device)
        phi_face[:n_internal] = w * phi_P + (1.0 - w) * phi_N
        if n_faces > n_internal:
            phi_face[n_internal:] = gather(phi, mesh.owner[n_internal:])

        face_contrib = phi_face.unsqueeze(-1) * self._face_areas
        grad_phi = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad_phi.index_add_(0, int_own, face_contrib[:n_internal])
        grad_phi.index_add_(0, int_nei, -face_contrib[:n_internal])
        if n_faces > n_internal:
            grad_phi.index_add_(
                0, mesh.owner[n_internal:], face_contrib[n_internal:],
            )
        V = self._cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        grad_phi = grad_phi / V

        # --- Pass 2: corrected face values using provisional gradient ---
        if n_internal > 0:
            # Interpolate gradient to internal faces
            grad_P = grad_phi[int_own]
            grad_N = grad_phi[int_nei]
            wt = w.unsqueeze(-1)
            grad_face = wt * grad_P + (1.0 - wt) * grad_N

            # Non-orthogonal correction: k . grad_f
            correction = (self._k * grad_face).sum(dim=1)  # (n_int,)
            phi_face_corr = phi_face.clone()
            phi_face_corr[:n_internal] = phi_face_corr[:n_internal] + correction

            # Re-integrate with corrected face values
            face_contrib_corr = phi_face_corr.unsqueeze(-1) * self._face_areas
            grad_phi_corr = torch.zeros(
                n_cells, 3, dtype=dtype, device=device,
            )
            grad_phi_corr.index_add_(
                0, int_own, face_contrib_corr[:n_internal],
            )
            grad_phi_corr.index_add_(
                0, int_nei, -face_contrib_corr[:n_internal],
            )
            if n_faces > n_internal:
                grad_phi_corr.index_add_(
                    0,
                    mesh.owner[n_internal:],
                    face_contrib_corr[n_internal:],
                )

            grad_phi = grad_phi_corr / V

        return grad_phi


# ---------------------------------------------------------------------------
# Fourth-order gradient v2 (tangential correction)
# ---------------------------------------------------------------------------


@_register_grad("fourth2")
class FourthGrad2(GradScheme):
    r"""Fourth-order gradient v2 with enhanced tangential correction.

    Improves on :class:`FourthGrad` by using a tangential gradient
    correction with face-averaged gradient, providing better accuracy
    on non-uniform structured meshes.

    For each cell *P*:

    .. math::

        \nabla\phi_P =
        \frac{1}{V_P} \sum_f \phi_f \, \mathbf{S}_f
        + \frac{1}{V_P} \sum_f
          \left[ \frac{1}{6}\,
          (\nabla\phi_P + \nabla\phi_N)_t \cdot \mathbf{S}_f \right]

    where the subscript *t* denotes the tangential component (perpendicular
    to the face normal).

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
        self._w = w_all[:n_internal]
        self._face_areas = mesh.face_areas.to(device=device, dtype=dtype)
        self._cell_volumes = mesh.cell_volumes.to(device=device, dtype=dtype)

    def compute_grad(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute v2 higher-order Gauss gradient with enhanced tangential correction."""
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces

        phi = phi.to(device=device, dtype=dtype)

        int_own = mesh.owner[:n_internal]
        int_nei = mesh.neighbour[:n_internal]

        # 第一步：标准 Gauss 线性梯度
        phi_P = gather(phi, int_own)
        phi_N = gather(phi, int_nei)
        w = self._w

        phi_face = torch.zeros(n_faces, dtype=dtype, device=device)
        phi_face[:n_internal] = w * phi_P + (1.0 - w) * phi_N
        if n_faces > n_internal:
            phi_face[n_internal:] = gather(phi, mesh.owner[n_internal:])

        face_contrib = phi_face.unsqueeze(-1) * self._face_areas
        grad_phi = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad_phi.index_add_(0, int_own, face_contrib[:n_internal])
        grad_phi.index_add_(0, int_nei, -face_contrib[:n_internal])
        if n_faces > n_internal:
            grad_phi.index_add_(
                0, mesh.owner[n_internal:], face_contrib[n_internal:],
            )

        V = self._cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        grad_phi = grad_phi / V

        # 第二步：增强切向修正 (1/6 系数，使用面平均梯度)
        if n_internal > 0:
            grad_P = grad_phi[int_own]
            grad_N = grad_phi[int_nei]
            # v2：使用算术平均而非距离加权
            grad_face = 0.5 * (grad_P + grad_N)

            S = self._face_areas[:n_internal]
            S_mag = S.norm(dim=1, keepdim=True).clamp(min=1e-30)
            n_hat = S / S_mag

            grad_normal = (grad_face * n_hat).sum(dim=1, keepdim=True)
            grad_tangential = grad_face - grad_normal * n_hat

            # v2：使用 1/6 系数
            correction = (1.0 / 6.0) * grad_tangential * S

            corr = torch.zeros(n_cells, 3, dtype=dtype, device=device)
            corr.index_add_(0, int_own, correction)
            corr.index_add_(0, int_nei, -correction)

            grad_phi = grad_phi + corr / V

        return grad_phi


# ---------------------------------------------------------------------------
# Cell-limited gradient v2 (face-smoothed limiter)
# ---------------------------------------------------------------------------


@_register_grad("cellLimited2")
class CellLimitedGrad2(GradScheme):
    r"""Cell-limited gradient v2 with face-smoothed limiter.

    Improves on :class:`CellLimitedGrad` by applying a smoothing pass
    to the cell limiter coefficients, reducing the effect of isolated
    limiters that can cause stepped solutions:

    .. math::

        \alpha_P^{\text{smooth}} = \frac{\alpha_P + \sum_N \alpha_N}{1 + n_N}

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    base_scheme : type[GradScheme], optional
        The unlimited gradient scheme class.  Default is
        :class:`GaussLinearGrad`.
    """

    def __init__(self, mesh, base_scheme=None) -> None:
        super().__init__(mesh)
        if base_scheme is None:
            base_scheme = GaussLinearGrad
        self._base = base_scheme(mesh)

    def compute_grad(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute v2 cell-limited gradient with smoothed limiter."""
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces

        phi = phi.to(device=device, dtype=dtype)

        grad_unlimited = self._base.compute_grad(phi)

        if n_internal == 0:
            return grad_unlimited

        cc = mesh.cell_centres.to(device=device, dtype=dtype)
        fa = mesh.face_centres.to(device=device, dtype=dtype)

        int_own = mesh.owner[:n_internal]
        int_nei = mesh.neighbour[:n_internal]

        delta_P = fa[:n_internal] - cc[int_own]
        delta_N = fa[:n_internal] - cc[int_nei]

        grad_P = grad_unlimited[int_own]
        grad_N = grad_unlimited[int_nei]

        phi_P = gather(phi, int_own)
        phi_N = gather(phi, int_nei)

        phi_ext_P = phi_P + (grad_P * delta_P).sum(dim=1)
        phi_ext_N = phi_N + (grad_N * delta_N).sum(dim=1)

        phi_face_min = torch.min(phi_P, phi_N)
        phi_face_max = torch.max(phi_P, phi_N)

        eps = 1e-30

        # 所有者限制器
        diff_P = phi_ext_P - phi_P
        max_allow_P = phi_face_max - phi_P
        min_allow_P = phi_face_min - phi_P

        alpha_P = torch.ones(n_internal, dtype=dtype, device=device)
        pos_mask = diff_P > eps
        neg_mask = diff_P < -eps
        alpha_P = torch.where(
            pos_mask,
            torch.clamp(max_allow_P / (diff_P + eps), 0.0, 1.0),
            alpha_P,
        )
        alpha_P = torch.where(
            neg_mask,
            torch.clamp(min_allow_P / (diff_P - eps), 0.0, 1.0),
            alpha_P,
        )

        # 邻居限制器
        diff_N = phi_ext_N - phi_N
        max_allow_N = phi_face_max - phi_N
        min_allow_N = phi_face_min - phi_N

        alpha_N = torch.ones(n_internal, dtype=dtype, device=device)
        pos_mask = diff_N > eps
        neg_mask = diff_N < -eps
        alpha_N = torch.where(
            pos_mask,
            torch.clamp(max_allow_N / (diff_N + eps), 0.0, 1.0),
            alpha_N,
        )
        alpha_N = torch.where(
            neg_mask,
            torch.clamp(min_allow_N / (diff_N - eps), 0.0, 1.0),
            alpha_N,
        )

        # v2 改进：计算每个单元的最小限制器
        cell_alpha = torch.ones(n_cells, dtype=dtype, device=device)
        cell_alpha.scatter_reduce_(
            0, int_own, alpha_P, reduce="amin", include_self=True,
        )
        cell_alpha.scatter_reduce_(
            0, int_nei, alpha_N, reduce="amin", include_self=True,
        )

        # v2：平滑限制器系数（与邻居平均）
        cell_count = torch.ones(n_cells, dtype=dtype, device=device)
        alpha_sum = cell_alpha.clone()
        # 所有者→邻居
        alpha_sum.scatter_add_(0, int_own, cell_alpha[int_nei])
        alpha_sum.scatter_add_(0, int_nei, cell_alpha[int_own])
        cell_count.scatter_add_(
            0, int_own, torch.ones(n_internal, dtype=dtype, device=device),
        )
        cell_count.scatter_add_(
            0, int_nei, torch.ones(n_internal, dtype=dtype, device=device),
        )
        alpha_smooth = (alpha_sum / cell_count.clamp(min=1)).clamp(0.0, 1.0)

        return alpha_smooth.unsqueeze(-1) * grad_unlimited


# ---------------------------------------------------------------------------
# Face-limited gradient v2 (higher blending)
# ---------------------------------------------------------------------------


@_register_grad("faceLimited2")
class FaceLimitedGrad2(GradScheme):
    r"""Face-limited gradient v2 with higher blending ratio.

    Improves on :class:`FaceLimitedGrad` by using a 70/30 blend between
    the limited and unlimited gradients instead of the alpha-weighted
    average, providing better accuracy while maintaining boundedness.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    base_scheme : type[GradScheme], optional
        The unlimited gradient scheme class.  Default is
        :class:`GaussLinearGrad`.
    blend_ratio : float, optional
        Ratio of unlimited gradient to blend in.  Default is 0.3.
    """

    def __init__(self, mesh, base_scheme=None, blend_ratio: float = 0.3) -> None:
        super().__init__(mesh)
        if base_scheme is None:
            base_scheme = GaussLinearGrad
        self._base = base_scheme(mesh)
        self._blend_ratio = blend_ratio

        device = mesh.device
        dtype = mesh.dtype
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces

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
        self._w = w_all[:n_internal]
        self._face_areas = mesh.face_areas.to(device=device, dtype=dtype)
        self._cell_volumes = mesh.cell_volumes.to(device=device, dtype=dtype)

    def compute_grad(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute v2 face-limited gradient with higher blending ratio."""
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces

        phi = phi.to(device=device, dtype=dtype)

        grad_unlimited = self._base.compute_grad(phi)

        if n_internal == 0:
            return grad_unlimited

        cc = mesh.cell_centres.to(device=device, dtype=dtype)
        fa = mesh.face_centres.to(device=device, dtype=dtype)
        int_own = mesh.owner[:n_internal]
        int_nei = mesh.neighbour[:n_internal]

        delta_P = fa[:n_internal] - cc[int_own]
        delta_N = fa[:n_internal] - cc[int_nei]

        grad_P = grad_unlimited[int_own]
        grad_N = grad_unlimited[int_nei]
        phi_P = gather(phi, int_own)
        phi_N = gather(phi, int_nei)

        phi_ext_P = phi_P + (grad_P * delta_P).sum(dim=1)
        phi_ext_N = phi_N + (grad_N * delta_N).sum(dim=1)

        phi_face_min = torch.min(phi_P, phi_N)
        phi_face_max = torch.max(phi_P, phi_N)

        eps = 1e-30

        # 面限制器
        diff_P = phi_ext_P - phi_P
        alpha_P = torch.ones(n_internal, dtype=dtype, device=device)
        pos_mask = diff_P > eps
        neg_mask = diff_P < -eps
        alpha_P = torch.where(
            pos_mask,
            torch.clamp((phi_face_max - phi_P) / (diff_P + eps), 0.0, 1.0),
            alpha_P,
        )
        alpha_P = torch.where(
            neg_mask,
            torch.clamp((phi_face_min - phi_P) / (diff_P - eps), 0.0, 1.0),
            alpha_P,
        )

        diff_N = phi_ext_N - phi_N
        alpha_N = torch.ones(n_internal, dtype=dtype, device=device)
        pos_mask = diff_N > eps
        neg_mask = diff_N < -eps
        alpha_N = torch.where(
            pos_mask,
            torch.clamp((phi_face_max - phi_N) / (diff_N + eps), 0.0, 1.0),
            alpha_N,
        )
        alpha_N = torch.where(
            neg_mask,
            torch.clamp((phi_face_min - phi_N) / (diff_N - eps), 0.0, 1.0),
            alpha_N,
        )

        grad_limited_P = alpha_P.unsqueeze(-1) * grad_P
        grad_limited_N = alpha_N.unsqueeze(-1) * grad_N

        w = self._w.unsqueeze(-1)
        grad_face_lim = w * grad_limited_P + (1.0 - w) * grad_limited_N

        phi_face = w.squeeze(-1) * phi_P + (1.0 - w.squeeze(-1)) * phi_N

        face_contrib = phi_face.unsqueeze(-1) * self._face_areas[:n_internal]
        grad_phi = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad_phi.index_add_(0, int_own, face_contrib)
        grad_phi.index_add_(0, int_nei, -face_contrib)

        if n_faces > n_internal:
            bnd_own = mesh.owner[n_internal:]
            phi_bnd = gather(phi, bnd_own)
            bnd_contrib = phi_bnd.unsqueeze(-1) * self._face_areas[n_internal:]
            grad_phi.index_add_(0, bnd_own, bnd_contrib)

        V = self._cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        grad_phi = grad_phi / V

        # v2 改进：固定混合比率而非 alpha 加权
        alpha_cell = torch.ones(n_cells, dtype=dtype, device=device)
        alpha_cell.scatter_reduce_(
            0, int_own, alpha_P, reduce="amin", include_self=True,
        )
        alpha_cell.scatter_reduce_(
            0, int_nei, alpha_N, reduce="amin", include_self=True,
        )

        # 检查哪些单元需要限制
        needs_limiting = alpha_cell < 1.0 - 1e-10
        blend = self._blend_ratio

        result = grad_phi.clone()
        # 对需要限制的单元使用固定混合
        limited_grad = alpha_cell.unsqueeze(-1) * grad_phi
        result = torch.where(
            needs_limiting.unsqueeze(-1),
            (1.0 - blend) * limited_grad + blend * grad_unlimited,
            result,
        )

        return result


# ---------------------------------------------------------------------------
# Fourth-order gradient v3 (weighted tangential correction)
# ---------------------------------------------------------------------------


@_register_grad("fourth3")
class FourthGrad3(GradScheme):
    r"""Fourth-order gradient v3 with weighted tangential correction.

    Improves on :class:`FourthGrad2` by using a weighted average of the
    owner and neighbour cell gradients at the face (distance-weighted
    instead of arithmetic average), providing better accuracy on
    non-uniform structured meshes:

    .. math::

        \nabla\phi_P =
        \frac{1}{V_P} \sum_f \phi_f \, \mathbf{S}_f
        + \frac{1}{V_P} \sum_f
          \left[ \frac{1}{8}\,
          (w\,\nabla\phi_P + (1-w)\,\nabla\phi_N)_t \cdot \mathbf{S}_f \right]

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
        self._w = w_all[:n_internal]
        self._face_areas = mesh.face_areas.to(device=device, dtype=dtype)
        self._cell_volumes = mesh.cell_volumes.to(device=device, dtype=dtype)

    def compute_grad(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute v3 higher-order Gauss gradient with weighted tangential correction."""
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces

        phi = phi.to(device=device, dtype=dtype)

        int_own = mesh.owner[:n_internal]
        int_nei = mesh.neighbour[:n_internal]

        phi_P = gather(phi, int_own)
        phi_N = gather(phi, int_nei)
        w = self._w

        phi_face = torch.zeros(n_faces, dtype=dtype, device=device)
        phi_face[:n_internal] = w * phi_P + (1.0 - w) * phi_N
        if n_faces > n_internal:
            phi_face[n_internal:] = gather(phi, mesh.owner[n_internal:])

        face_contrib = phi_face.unsqueeze(-1) * self._face_areas
        grad_phi = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad_phi.index_add_(0, int_own, face_contrib[:n_internal])
        grad_phi.index_add_(0, int_nei, -face_contrib[:n_internal])
        if n_faces > n_internal:
            grad_phi.index_add_(
                0, mesh.owner[n_internal:], face_contrib[n_internal:],
            )

        V = self._cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        grad_phi = grad_phi / V

        # v3: 加权切向修正 (1/8 系数，距离加权平均梯度)
        if n_internal > 0:
            grad_P = grad_phi[int_own]
            grad_N = grad_phi[int_nei]
            wt = w.unsqueeze(-1)
            # v3: 使用距离加权平均
            grad_face = wt * grad_P + (1.0 - wt) * grad_N

            S = self._face_areas[:n_internal]
            S_mag = S.norm(dim=1, keepdim=True).clamp(min=1e-30)
            n_hat = S / S_mag

            grad_normal = (grad_face * n_hat).sum(dim=1, keepdim=True)
            grad_tangential = grad_face - grad_normal * n_hat

            # v3: 使用 1/8 系数
            correction = (1.0 / 8.0) * grad_tangential * S

            corr = torch.zeros(n_cells, 3, dtype=dtype, device=device)
            corr.index_add_(0, int_own, correction)
            corr.index_add_(0, int_nei, -correction)

            grad_phi = grad_phi + corr / V

        return grad_phi


# ---------------------------------------------------------------------------
# Cell-limited gradient v3 (neighbor-aware limiting)
# ---------------------------------------------------------------------------


@_register_grad("cellLimited3")
class CellLimitedGrad3(GradScheme):
    r"""Cell-limited gradient v3 with neighbor-aware limiting.

    Improves on :class:`CellLimitedGrad2` by incorporating neighbour
    cell gradient information into the limiter calculation, providing
    smoother limiting across cell interfaces:

    .. math::

        \alpha_P^{\text{v3}} = \min(\alpha_P, \; 0.5(\alpha_P + \alpha_{N,\text{avg}}))

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    base_scheme : type[GradScheme], optional
        The unlimited gradient scheme class.  Default is
        :class:`GaussLinearGrad`.
    """

    def __init__(self, mesh, base_scheme=None) -> None:
        super().__init__(mesh)
        if base_scheme is None:
            base_scheme = GaussLinearGrad
        self._base = base_scheme(mesh)

    def compute_grad(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute v3 cell-limited gradient with neighbor-aware limiting."""
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces

        phi = phi.to(device=device, dtype=dtype)

        grad_unlimited = self._base.compute_grad(phi)

        if n_internal == 0:
            return grad_unlimited

        cc = mesh.cell_centres.to(device=device, dtype=dtype)
        fa = mesh.face_centres.to(device=device, dtype=dtype)

        int_own = mesh.owner[:n_internal]
        int_nei = mesh.neighbour[:n_internal]

        delta_P = fa[:n_internal] - cc[int_own]
        delta_N = fa[:n_internal] - cc[int_nei]

        grad_P = grad_unlimited[int_own]
        grad_N = grad_unlimited[int_nei]

        phi_P = gather(phi, int_own)
        phi_N = gather(phi, int_nei)

        phi_ext_P = phi_P + (grad_P * delta_P).sum(dim=1)
        phi_ext_N = phi_N + (grad_N * delta_N).sum(dim=1)

        phi_face_min = torch.min(phi_P, phi_N)
        phi_face_max = torch.max(phi_P, phi_N)

        eps = 1e-30

        diff_P = phi_ext_P - phi_P
        max_allow_P = phi_face_max - phi_P
        min_allow_P = phi_face_min - phi_P

        alpha_P = torch.ones(n_internal, dtype=dtype, device=device)
        pos_mask = diff_P > eps
        neg_mask = diff_P < -eps
        alpha_P = torch.where(
            pos_mask,
            torch.clamp(max_allow_P / (diff_P + eps), 0.0, 1.0),
            alpha_P,
        )
        alpha_P = torch.where(
            neg_mask,
            torch.clamp(min_allow_P / (diff_P - eps), 0.0, 1.0),
            alpha_P,
        )

        diff_N = phi_ext_N - phi_N
        max_allow_N = phi_face_max - phi_N
        min_allow_N = phi_face_min - phi_N

        alpha_N = torch.ones(n_internal, dtype=dtype, device=device)
        pos_mask = diff_N > eps
        neg_mask = diff_N < -eps
        alpha_N = torch.where(
            pos_mask,
            torch.clamp(max_allow_N / (diff_N + eps), 0.0, 1.0),
            alpha_N,
        )
        alpha_N = torch.where(
            neg_mask,
            torch.clamp(min_allow_N / (diff_N - eps), 0.0, 1.0),
            alpha_N,
        )

        # 基础限制器
        cell_alpha = torch.ones(n_cells, dtype=dtype, device=device)
        cell_alpha.scatter_reduce_(
            0, int_own, alpha_P, reduce="amin", include_self=True,
        )
        cell_alpha.scatter_reduce_(
            0, int_nei, alpha_N, reduce="amin", include_self=True,
        )

        # v3: 邻居感知平滑
        cell_count = torch.ones(n_cells, dtype=dtype, device=device)
        alpha_sum = cell_alpha.clone()
        alpha_sum.scatter_add_(0, int_own, cell_alpha[int_nei])
        alpha_sum.scatter_add_(0, int_nei, cell_alpha[int_own])
        cell_count.scatter_add_(
            0, int_own, torch.ones(n_internal, dtype=dtype, device=device),
        )
        cell_count.scatter_add_(
            0, int_nei, torch.ones(n_internal, dtype=dtype, device=device),
        )
        alpha_avg = (alpha_sum / cell_count.clamp(min=1)).clamp(0.0, 1.0)

        # v3: 取原始限制器和平滑限制器的最小值
        alpha_v3 = torch.min(cell_alpha, 0.5 * (cell_alpha + alpha_avg))

        return alpha_v3.unsqueeze(-1) * grad_unlimited


# ---------------------------------------------------------------------------
# Face-limited gradient v3 (per-face adaptive limiting)
# ---------------------------------------------------------------------------


@_register_grad("faceLimited3")
class FaceLimitedGrad3(GradScheme):
    r"""Face-limited gradient v3 with per-face adaptive limiting.

    Improves on :class:`FaceLimitedGrad2` by adaptively adjusting the
    blend ratio per face based on the local limiter value, providing
    more correction where possible and less where needed:

    .. math::

        \text{blend}_f = \text{blend\_base} \cdot \alpha_f

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    base_scheme : type[GradScheme], optional
        The unlimited gradient scheme class.  Default is
        :class:`GaussLinearGrad`.
    blend_base : float, optional
        Base blending ratio.  Default is 0.4.
    """

    def __init__(self, mesh, base_scheme=None, blend_base: float = 0.4) -> None:
        super().__init__(mesh)
        if base_scheme is None:
            base_scheme = GaussLinearGrad
        self._base = base_scheme(mesh)
        self._blend_base = blend_base

        device = mesh.device
        dtype = mesh.dtype
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces

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
        self._w = w_all[:n_internal]
        self._face_areas = mesh.face_areas.to(device=device, dtype=dtype)
        self._cell_volumes = mesh.cell_volumes.to(device=device, dtype=dtype)

    def compute_grad(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute v3 face-limited gradient with adaptive per-face blending."""
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces

        phi = phi.to(device=device, dtype=dtype)

        grad_unlimited = self._base.compute_grad(phi)

        if n_internal == 0:
            return grad_unlimited

        cc = mesh.cell_centres.to(device=device, dtype=dtype)
        fa = mesh.face_centres.to(device=device, dtype=dtype)
        int_own = mesh.owner[:n_internal]
        int_nei = mesh.neighbour[:n_internal]

        delta_P = fa[:n_internal] - cc[int_own]
        delta_N = fa[:n_internal] - cc[int_nei]

        grad_P = grad_unlimited[int_own]
        grad_N = grad_unlimited[int_nei]
        phi_P = gather(phi, int_own)
        phi_N = gather(phi, int_nei)

        phi_ext_P = phi_P + (grad_P * delta_P).sum(dim=1)
        phi_ext_N = phi_N + (grad_N * delta_N).sum(dim=1)

        phi_face_min = torch.min(phi_P, phi_N)
        phi_face_max = torch.max(phi_P, phi_N)

        eps = 1e-30

        diff_P = phi_ext_P - phi_P
        alpha_P = torch.ones(n_internal, dtype=dtype, device=device)
        pos_mask = diff_P > eps
        neg_mask = diff_P < -eps
        alpha_P = torch.where(
            pos_mask,
            torch.clamp((phi_face_max - phi_P) / (diff_P + eps), 0.0, 1.0),
            alpha_P,
        )
        alpha_P = torch.where(
            neg_mask,
            torch.clamp((phi_face_min - phi_P) / (diff_P - eps), 0.0, 1.0),
            alpha_P,
        )

        diff_N = phi_ext_N - phi_N
        alpha_N = torch.ones(n_internal, dtype=dtype, device=device)
        pos_mask = diff_N > eps
        neg_mask = diff_N < -eps
        alpha_N = torch.where(
            pos_mask,
            torch.clamp((phi_face_max - phi_N) / (diff_N + eps), 0.0, 1.0),
            alpha_N,
        )
        alpha_N = torch.where(
            neg_mask,
            torch.clamp((phi_face_min - phi_N) / (diff_N - eps), 0.0, 1.0),
            alpha_N,
        )

        alpha_face = torch.min(alpha_P, alpha_N)

        w = self._w.unsqueeze(-1)
        phi_face = w.squeeze(-1) * phi_P + (1.0 - w.squeeze(-1)) * phi_N

        face_contrib = phi_face.unsqueeze(-1) * self._face_areas[:n_internal]
        grad_phi = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad_phi.index_add_(0, int_own, face_contrib)
        grad_phi.index_add_(0, int_nei, -face_contrib)

        if n_faces > n_internal:
            bnd_own = mesh.owner[n_internal:]
            phi_bnd = gather(phi, bnd_own)
            bnd_contrib = phi_bnd.unsqueeze(-1) * self._face_areas[n_internal:]
            grad_phi.index_add_(0, bnd_own, bnd_contrib)

        V = self._cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        grad_phi = grad_phi / V

        # v3: 自适应混合比率
        alpha_cell = torch.ones(n_cells, dtype=dtype, device=device)
        alpha_cell.scatter_reduce_(
            0, int_own, alpha_P, reduce="amin", include_self=True,
        )
        alpha_cell.scatter_reduce_(
            0, int_nei, alpha_N, reduce="amin", include_self=True,
        )

        needs_limiting = alpha_cell < 1.0 - 1e-10
        # v3: 混合比率随限制器值自适应调整
        adaptive_blend = self._blend_base * alpha_cell
        limited_grad = alpha_cell.unsqueeze(-1) * grad_phi
        result = grad_phi.clone()
        result = torch.where(
            needs_limiting.unsqueeze(-1),
            (1.0 - adaptive_blend.unsqueeze(-1)) * limited_grad
            + adaptive_blend.unsqueeze(-1) * grad_unlimited,
            result,
        )

        return result


# ---------------------------------------------------------------------------
# Fourth-order gradient v4 (iterative tangential correction)
# ---------------------------------------------------------------------------


@_register_grad("fourth4")
class FourthGrad4(GradScheme):
    r"""Fourth-order gradient v4 with iterative tangential correction.

    Improves on :class:`FourthGrad3` by applying two passes of the
    tangential correction, using the result of the first correction
    to refine the face gradient estimate for the second pass:

    .. math::

        \nabla\phi_P =
        \frac{1}{V_P} \sum_f \phi_f \, \mathbf{S}_f
        + \frac{1}{V_P} \sum_f
          \left[ \frac{1}{10}\,
          (w\,\nabla\phi_P^{(1)} + (1-w)\,\nabla\phi_N^{(1)})_t \cdot \mathbf{S}_f \right]

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
        self._w = w_all[:n_internal]
        self._face_areas = mesh.face_areas.to(device=device, dtype=dtype)
        self._cell_volumes = mesh.cell_volumes.to(device=device, dtype=dtype)

    def compute_grad(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute v4 higher-order Gauss gradient with iterative tangential correction."""
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces

        phi = phi.to(device=device, dtype=dtype)

        int_own = mesh.owner[:n_internal]
        int_nei = mesh.neighbour[:n_internal]

        phi_P = gather(phi, int_own)
        phi_N = gather(phi, int_nei)
        w = self._w

        phi_face = torch.zeros(n_faces, dtype=dtype, device=device)
        phi_face[:n_internal] = w * phi_P + (1.0 - w) * phi_N
        if n_faces > n_internal:
            phi_face[n_internal:] = gather(phi, mesh.owner[n_internal:])

        face_contrib = phi_face.unsqueeze(-1) * self._face_areas
        grad_phi = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad_phi.index_add_(0, int_own, face_contrib[:n_internal])
        grad_phi.index_add_(0, int_nei, -face_contrib[:n_internal])
        if n_faces > n_internal:
            grad_phi.index_add_(
                0, mesh.owner[n_internal:], face_contrib[n_internal:],
            )

        V = self._cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        grad_phi = grad_phi / V

        # v4: 两轮迭代切向修正
        if n_internal > 0:
            S = self._face_areas[:n_internal]
            S_mag = S.norm(dim=1, keepdim=True).clamp(min=1e-30)
            n_hat = S / S_mag
            wt = w.unsqueeze(-1)

            for _pass in range(2):
                grad_P = grad_phi[int_own]
                grad_N = grad_phi[int_nei]
                grad_face = wt * grad_P + (1.0 - wt) * grad_N

                grad_normal = (grad_face * n_hat).sum(dim=1, keepdim=True)
                grad_tangential = grad_face - grad_normal * n_hat

                # v4: 使用 1/10 系数
                correction = (1.0 / 10.0) * grad_tangential * S

                corr = torch.zeros(n_cells, 3, dtype=dtype, device=device)
                corr.index_add_(0, int_own, correction)
                corr.index_add_(0, int_nei, -correction)

                grad_phi = grad_phi + corr / V

        return grad_phi


# ---------------------------------------------------------------------------
# Cell-limited gradient v4 (sigmoid-smoothed limiter)
# ---------------------------------------------------------------------------


@_register_grad("cellLimited4")
class CellLimitedGrad4(GradScheme):
    r"""Cell-limited gradient v4 with sigmoid-smoothed limiter.

    Improves on :class:`CellLimitedGrad3` by applying a sigmoid smoothing
    to the cell limiter coefficients, providing infinitely smooth
    transitions between limited and unlimited regions:

    .. math::

        \alpha_P^{\text{smooth}} = 1 / (1 + \exp(-k \, (\alpha_P - \alpha_0)))

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    base_scheme : type[GradScheme], optional
        The unlimited gradient scheme class.  Default is
        :class:`GaussLinearGrad`.
    steepness : float, optional
        Steepness of the sigmoid smoothing.  Default is 10.0.
    """

    def __init__(self, mesh, base_scheme=None, steepness: float = 10.0) -> None:
        super().__init__(mesh)
        if base_scheme is None:
            base_scheme = GaussLinearGrad
        self._base = base_scheme(mesh)
        self._steepness = steepness

    def compute_grad(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute v4 cell-limited gradient with sigmoid-smoothed limiter."""
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces

        phi = phi.to(device=device, dtype=dtype)

        grad_unlimited = self._base.compute_grad(phi)

        if n_internal == 0:
            return grad_unlimited

        cc = mesh.cell_centres.to(device=device, dtype=dtype)
        fa = mesh.face_centres.to(device=device, dtype=dtype)

        int_own = mesh.owner[:n_internal]
        int_nei = mesh.neighbour[:n_internal]

        delta_P = fa[:n_internal] - cc[int_own]
        delta_N = fa[:n_internal] - cc[int_nei]

        grad_P = grad_unlimited[int_own]
        grad_N = grad_unlimited[int_nei]

        phi_P = gather(phi, int_own)
        phi_N = gather(phi, int_nei)

        phi_ext_P = phi_P + (grad_P * delta_P).sum(dim=1)
        phi_ext_N = phi_N + (grad_N * delta_N).sum(dim=1)

        phi_face_min = torch.min(phi_P, phi_N)
        phi_face_max = torch.max(phi_P, phi_N)

        eps = 1e-30

        diff_P = phi_ext_P - phi_P
        max_allow_P = phi_face_max - phi_P
        min_allow_P = phi_face_min - phi_P

        alpha_P = torch.ones(n_internal, dtype=dtype, device=device)
        pos_mask = diff_P > eps
        neg_mask = diff_P < -eps
        alpha_P = torch.where(
            pos_mask,
            torch.clamp(max_allow_P / (diff_P + eps), 0.0, 1.0),
            alpha_P,
        )
        alpha_P = torch.where(
            neg_mask,
            torch.clamp(min_allow_P / (diff_P - eps), 0.0, 1.0),
            alpha_P,
        )

        diff_N = phi_ext_N - phi_N
        max_allow_N = phi_face_max - phi_N
        min_allow_N = phi_face_min - phi_N

        alpha_N = torch.ones(n_internal, dtype=dtype, device=device)
        pos_mask = diff_N > eps
        neg_mask = diff_N < -eps
        alpha_N = torch.where(
            pos_mask,
            torch.clamp(max_allow_N / (diff_N + eps), 0.0, 1.0),
            alpha_N,
        )
        alpha_N = torch.where(
            neg_mask,
            torch.clamp(min_allow_N / (diff_N - eps), 0.0, 1.0),
            alpha_N,
        )

        cell_alpha = torch.ones(n_cells, dtype=dtype, device=device)
        cell_alpha.scatter_reduce_(
            0, int_own, alpha_P, reduce="amin", include_self=True,
        )
        cell_alpha.scatter_reduce_(
            0, int_nei, alpha_N, reduce="amin", include_self=True,
        )

        # v4: sigmoid 平滑限制器系数
        # 将 alpha 值从 [0,1] 平滑映射到更均匀的分布
        k = self._steepness
        alpha_smooth = torch.sigmoid(k * (cell_alpha - 0.5))
        # 重新映射到 [min_alpha, 1] 范围
        min_alpha = cell_alpha.min()
        alpha_v4 = min_alpha + (1.0 - min_alpha) * alpha_smooth

        return alpha_v4.unsqueeze(-1) * grad_unlimited


# ---------------------------------------------------------------------------
# Face-limited gradient v4 (distance-weighted correction)
# ---------------------------------------------------------------------------


@_register_grad("faceLimited4")
class FaceLimitedGrad4(GradScheme):
    r"""Face-limited gradient v4 with distance-weighted correction.

    Improves on :class:`FaceLimitedGrad3` by using distance-weighted
    correction factors that give more correction weight to faces
    closer to the cell centre:

    .. math::

        \text{weight}_f = 1 / (1 + |d_f| / d_{\text{avg}})

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    base_scheme : type[GradScheme], optional
        The unlimited gradient scheme class.  Default is
        :class:`GaussLinearGrad`.
    blend_base : float, optional
        Base blending ratio.  Default is 0.5.
    """

    def __init__(self, mesh, base_scheme=None, blend_base: float = 0.5) -> None:
        super().__init__(mesh)
        if base_scheme is None:
            base_scheme = GaussLinearGrad
        self._base = base_scheme(mesh)
        self._blend_base = blend_base

        device = mesh.device
        dtype = mesh.dtype
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces

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
        self._w = w_all[:n_internal]
        self._face_areas = mesh.face_areas.to(device=device, dtype=dtype)
        self._cell_volumes = mesh.cell_volumes.to(device=device, dtype=dtype)

        # v4: 预计算面距离权重
        if n_internal > 0:
            cc = mesh.cell_centres
            fc = mesh.face_centres[:n_internal]
            cc_own = cc[mesh.owner[:n_internal]]
            cc_nei = cc[mesh.neighbour[:n_internal]]
            d_P = (fc - cc_own).norm(dim=1)
            d_N = (fc - cc_nei).norm(dim=1)
            d_avg = (d_P + d_N) / 2.0
            safe_avg = d_avg.clamp(min=1e-30)
            self._dist_weight_P = 1.0 / (1.0 + d_P / safe_avg)
            self._dist_weight_N = 1.0 / (1.0 + d_N / safe_avg)
        else:
            self._dist_weight_P = torch.zeros(0, dtype=dtype, device=device)
            self._dist_weight_N = torch.zeros(0, dtype=dtype, device=device)

    def compute_grad(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute v4 face-limited gradient with distance-weighted correction."""
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces

        phi = phi.to(device=device, dtype=dtype)

        grad_unlimited = self._base.compute_grad(phi)

        if n_internal == 0:
            return grad_unlimited

        cc = mesh.cell_centres.to(device=device, dtype=dtype)
        fa = mesh.face_centres.to(device=device, dtype=dtype)
        int_own = mesh.owner[:n_internal]
        int_nei = mesh.neighbour[:n_internal]

        delta_P = fa[:n_internal] - cc[int_own]
        delta_N = fa[:n_internal] - cc[int_nei]

        grad_P = grad_unlimited[int_own]
        grad_N = grad_unlimited[int_nei]
        phi_P = gather(phi, int_own)
        phi_N = gather(phi, int_nei)

        phi_ext_P = phi_P + (grad_P * delta_P).sum(dim=1)
        phi_ext_N = phi_N + (grad_N * delta_N).sum(dim=1)

        phi_face_min = torch.min(phi_P, phi_N)
        phi_face_max = torch.max(phi_P, phi_N)

        eps = 1e-30

        diff_P = phi_ext_P - phi_P
        alpha_P = torch.ones(n_internal, dtype=dtype, device=device)
        pos_mask = diff_P > eps
        neg_mask = diff_P < -eps
        alpha_P = torch.where(
            pos_mask,
            torch.clamp((phi_face_max - phi_P) / (diff_P + eps), 0.0, 1.0),
            alpha_P,
        )
        alpha_P = torch.where(
            neg_mask,
            torch.clamp((phi_face_min - phi_P) / (diff_P - eps), 0.0, 1.0),
            alpha_P,
        )

        diff_N = phi_ext_N - phi_N
        alpha_N = torch.ones(n_internal, dtype=dtype, device=device)
        pos_mask = diff_N > eps
        neg_mask = diff_N < -eps
        alpha_N = torch.where(
            pos_mask,
            torch.clamp((phi_face_max - phi_N) / (diff_N + eps), 0.0, 1.0),
            alpha_N,
        )
        alpha_N = torch.where(
            neg_mask,
            torch.clamp((phi_face_min - phi_N) / (diff_N - eps), 0.0, 1.0),
            alpha_N,
        )

        # v4: 距离加权限制器
        alpha_face = (
            self._dist_weight_P * alpha_P + self._dist_weight_N * alpha_N
        ) / (self._dist_weight_P + self._dist_weight_N).clamp(min=1e-30)

        w = self._w.unsqueeze(-1)
        phi_face = w.squeeze(-1) * phi_P + (1.0 - w.squeeze(-1)) * phi_N

        face_contrib = phi_face.unsqueeze(-1) * self._face_areas[:n_internal]
        grad_phi = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad_phi.index_add_(0, int_own, face_contrib)
        grad_phi.index_add_(0, int_nei, -face_contrib)

        if n_faces > n_internal:
            bnd_own = mesh.owner[n_internal:]
            phi_bnd = gather(phi, bnd_own)
            bnd_contrib = phi_bnd.unsqueeze(-1) * self._face_areas[n_internal:]
            grad_phi.index_add_(0, bnd_own, bnd_contrib)

        V = self._cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        grad_phi = grad_phi / V

        alpha_cell = torch.ones(n_cells, dtype=dtype, device=device)
        alpha_cell.scatter_reduce_(
            0, int_own, alpha_P, reduce="amin", include_self=True,
        )
        alpha_cell.scatter_reduce_(
            0, int_nei, alpha_N, reduce="amin", include_self=True,
        )

        needs_limiting = alpha_cell < 1.0 - 1e-10
        blend = self._blend_base * alpha_face.min()
        limited_grad = alpha_cell.unsqueeze(-1) * grad_phi
        result = grad_phi.clone()
        result = torch.where(
            needs_limiting.unsqueeze(-1),
            (1.0 - blend) * limited_grad + blend * grad_unlimited,
            result,
        )

        return result


# ---------------------------------------------------------------------------
# Fourth-order gradient v5 (multi-pass weighted tangential correction)
# ---------------------------------------------------------------------------


@_register_grad("fourth5")
class FourthGrad5(GradScheme):
    r"""Fourth-order gradient v5 with multi-pass weighted tangential correction.

    Improves on :class:`FourthGrad4` by using a three-pass tangential
    correction with progressively refined weights, providing better
    accuracy on highly non-uniform meshes:

    .. math::

        \nabla\phi_P =
        \frac{1}{V_P} \sum_f \phi_f \, \mathbf{S}_f
        + \frac{1}{V_P} \sum_f
          \left[ \frac{1}{12}\,
          (w\,\nabla\phi_P + (1-w)\,\nabla\phi_N)_t \cdot \mathbf{S}_f \right]

    with three correction passes using 1/12, 1/14, and 1/16 coefficients.

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
        self._w = w_all[:n_internal]
        self._face_areas = mesh.face_areas.to(device=device, dtype=dtype)
        self._cell_volumes = mesh.cell_volumes.to(device=device, dtype=dtype)

    def compute_grad(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute v5 higher-order Gauss gradient with multi-pass tangential correction."""
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces

        phi = phi.to(device=device, dtype=dtype)

        int_own = mesh.owner[:n_internal]
        int_nei = mesh.neighbour[:n_internal]

        phi_P = gather(phi, int_own)
        phi_N = gather(phi, int_nei)
        w = self._w

        phi_face = torch.zeros(n_faces, dtype=dtype, device=device)
        phi_face[:n_internal] = w * phi_P + (1.0 - w) * phi_N
        if n_faces > n_internal:
            phi_face[n_internal:] = gather(phi, mesh.owner[n_internal:])

        face_contrib = phi_face.unsqueeze(-1) * self._face_areas
        grad_phi = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad_phi.index_add_(0, int_own, face_contrib[:n_internal])
        grad_phi.index_add_(0, int_nei, -face_contrib[:n_internal])
        if n_faces > n_internal:
            grad_phi.index_add_(
                0, mesh.owner[n_internal:], face_contrib[n_internal:],
            )

        V = self._cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        grad_phi = grad_phi / V

        # v5: 三轮迭代切向修正，使用递减系数
        if n_internal > 0:
            S = self._face_areas[:n_internal]
            S_mag = S.norm(dim=1, keepdim=True).clamp(min=1e-30)
            n_hat = S / S_mag
            wt = w.unsqueeze(-1)

            # 三轮系数: 1/12, 1/14, 1/16
            for coeff in (1.0 / 12.0, 1.0 / 14.0, 1.0 / 16.0):
                grad_P = grad_phi[int_own]
                grad_N = grad_phi[int_nei]
                grad_face = wt * grad_P + (1.0 - wt) * grad_N

                grad_normal = (grad_face * n_hat).sum(dim=1, keepdim=True)
                grad_tangential = grad_face - grad_normal * n_hat

                correction = coeff * grad_tangential * S

                corr = torch.zeros(n_cells, 3, dtype=dtype, device=device)
                corr.index_add_(0, int_own, correction)
                corr.index_add_(0, int_nei, -correction)

                grad_phi = grad_phi + corr / V

        return grad_phi


# ---------------------------------------------------------------------------
# Cell-limited gradient v5 (exponential-smoothed limiter)
# ---------------------------------------------------------------------------


@_register_grad("cellLimited5")
class CellLimitedGrad5(GradScheme):
    r"""Cell-limited gradient v5 with exponential-smoothed limiter.

    Improves on :class:`CellLimitedGrad4` by using an exponential smoothing
    function instead of sigmoid, providing a different smoothing characteristic
    that works better for gradient-dominated flows:

    .. math::

        \alpha_P^{\text{smooth}} = 1 - \exp(-k \, \alpha_P)

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    base_scheme : type[GradScheme], optional
        The unlimited gradient scheme class.  Default is
        :class:`GaussLinearGrad`.
    smoothing : float, optional
        Smoothing exponent.  Default is 3.0.
    """

    def __init__(self, mesh, base_scheme=None, smoothing: float = 3.0) -> None:
        super().__init__(mesh)
        if base_scheme is None:
            base_scheme = GaussLinearGrad
        self._base = base_scheme(mesh)
        self._smoothing = smoothing

    def compute_grad(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute v5 cell-limited gradient with exponential-smoothed limiter."""
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces

        phi = phi.to(device=device, dtype=dtype)

        grad_unlimited = self._base.compute_grad(phi)

        if n_internal == 0:
            return grad_unlimited

        cc = mesh.cell_centres.to(device=device, dtype=dtype)
        fa = mesh.face_centres.to(device=device, dtype=dtype)

        int_own = mesh.owner[:n_internal]
        int_nei = mesh.neighbour[:n_internal]

        delta_P = fa[:n_internal] - cc[int_own]
        delta_N = fa[:n_internal] - cc[int_nei]

        grad_P = grad_unlimited[int_own]
        grad_N = grad_unlimited[int_nei]

        phi_P = gather(phi, int_own)
        phi_N = gather(phi, int_nei)

        phi_ext_P = phi_P + (grad_P * delta_P).sum(dim=1)
        phi_ext_N = phi_N + (grad_N * delta_N).sum(dim=1)

        phi_face_min = torch.min(phi_P, phi_N)
        phi_face_max = torch.max(phi_P, phi_N)

        eps = 1e-30

        diff_P = phi_ext_P - phi_P
        max_allow_P = phi_face_max - phi_P
        min_allow_P = phi_face_min - phi_P

        alpha_P = torch.ones(n_internal, dtype=dtype, device=device)
        pos_mask = diff_P > eps
        neg_mask = diff_P < -eps
        alpha_P = torch.where(
            pos_mask,
            torch.clamp(max_allow_P / (diff_P + eps), 0.0, 1.0),
            alpha_P,
        )
        alpha_P = torch.where(
            neg_mask,
            torch.clamp(min_allow_P / (diff_P - eps), 0.0, 1.0),
            alpha_P,
        )

        diff_N = phi_ext_N - phi_N
        max_allow_N = phi_face_max - phi_N
        min_allow_N = phi_face_min - phi_N

        alpha_N = torch.ones(n_internal, dtype=dtype, device=device)
        pos_mask = diff_N > eps
        neg_mask = diff_N < -eps
        alpha_N = torch.where(
            pos_mask,
            torch.clamp(max_allow_N / (diff_N + eps), 0.0, 1.0),
            alpha_N,
        )
        alpha_N = torch.where(
            neg_mask,
            torch.clamp(min_allow_N / (diff_N - eps), 0.0, 1.0),
            alpha_N,
        )

        cell_alpha = torch.ones(n_cells, dtype=dtype, device=device)
        cell_alpha.scatter_reduce_(
            0, int_own, alpha_P, reduce="amin", include_self=True,
        )
        cell_alpha.scatter_reduce_(
            0, int_nei, alpha_N, reduce="amin", include_self=True,
        )

        # v5: 指数平滑限制器
        k = self._smoothing
        alpha_smooth = 1.0 - torch.exp(-k * cell_alpha)
        # 重新映射到 [min_alpha, 1] 范围
        min_alpha = cell_alpha.min()
        alpha_v5 = min_alpha + (1.0 - min_alpha) * alpha_smooth

        return alpha_v5.unsqueeze(-1) * grad_unlimited


# ---------------------------------------------------------------------------
# Face-limited gradient v5 (harmonic mean correction)
# ---------------------------------------------------------------------------


@_register_grad("faceLimited5")
class FaceLimitedGrad5(GradScheme):
    r"""Face-limited gradient v5 with harmonic mean correction.

    Improves on :class:`FaceLimitedGrad4` by using harmonic mean blending
    of the limiting factors instead of arithmetic mean, providing better
    behaviour on meshes with strongly varying cell sizes:

    .. math::

        \alpha_f = \frac{2}{1/\alpha_P + 1/\alpha_N}

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    base_scheme : type[GradScheme], optional
        The unlimited gradient scheme class.  Default is
        :class:`GaussLinearGrad`.
    blend_base : float, optional
        Base blending ratio.  Default is 0.5.
    """

    def __init__(self, mesh, base_scheme=None, blend_base: float = 0.5) -> None:
        super().__init__(mesh)
        if base_scheme is None:
            base_scheme = GaussLinearGrad
        self._base = base_scheme(mesh)
        self._blend_base = blend_base

        device = mesh.device
        dtype = mesh.dtype
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces

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
        self._w = w_all[:n_internal]
        self._face_areas = mesh.face_areas.to(device=device, dtype=dtype)
        self._cell_volumes = mesh.cell_volumes.to(device=device, dtype=dtype)

    def compute_grad(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute v5 face-limited gradient with harmonic mean correction."""
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces

        phi = phi.to(device=device, dtype=dtype)

        grad_unlimited = self._base.compute_grad(phi)

        if n_internal == 0:
            return grad_unlimited

        cc = mesh.cell_centres.to(device=device, dtype=dtype)
        fa = mesh.face_centres.to(device=device, dtype=dtype)
        int_own = mesh.owner[:n_internal]
        int_nei = mesh.neighbour[:n_internal]

        delta_P = fa[:n_internal] - cc[int_own]
        delta_N = fa[:n_internal] - cc[int_nei]

        grad_P = grad_unlimited[int_own]
        grad_N = grad_unlimited[int_nei]
        phi_P = gather(phi, int_own)
        phi_N = gather(phi, int_nei)

        phi_ext_P = phi_P + (grad_P * delta_P).sum(dim=1)
        phi_ext_N = phi_N + (grad_N * delta_N).sum(dim=1)

        phi_face_min = torch.min(phi_P, phi_N)
        phi_face_max = torch.max(phi_P, phi_N)

        eps = 1e-30

        diff_P = phi_ext_P - phi_P
        alpha_P = torch.ones(n_internal, dtype=dtype, device=device)
        pos_mask = diff_P > eps
        neg_mask = diff_P < -eps
        alpha_P = torch.where(
            pos_mask,
            torch.clamp((phi_face_max - phi_P) / (diff_P + eps), 0.0, 1.0),
            alpha_P,
        )
        alpha_P = torch.where(
            neg_mask,
            torch.clamp((phi_face_min - phi_P) / (diff_P - eps), 0.0, 1.0),
            alpha_P,
        )

        diff_N = phi_ext_N - phi_N
        alpha_N = torch.ones(n_internal, dtype=dtype, device=device)
        pos_mask = diff_N > eps
        neg_mask = diff_N < -eps
        alpha_N = torch.where(
            pos_mask,
            torch.clamp((phi_face_max - phi_N) / (diff_N + eps), 0.0, 1.0),
            alpha_N,
        )
        alpha_N = torch.where(
            neg_mask,
            torch.clamp((phi_face_min - phi_N) / (diff_N - eps), 0.0, 1.0),
            alpha_N,
        )

        # v5: 调和均值限制器
        safe_P = alpha_P.clamp(min=eps)
        safe_N = alpha_N.clamp(min=eps)
        alpha_face = 2.0 / (1.0 / safe_P + 1.0 / safe_N)

        w = self._w.unsqueeze(-1)
        phi_face = w.squeeze(-1) * phi_P + (1.0 - w.squeeze(-1)) * phi_N

        face_contrib = phi_face.unsqueeze(-1) * self._face_areas[:n_internal]
        grad_phi = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad_phi.index_add_(0, int_own, face_contrib)
        grad_phi.index_add_(0, int_nei, -face_contrib)

        if n_faces > n_internal:
            bnd_own = mesh.owner[n_internal:]
            phi_bnd = gather(phi, bnd_own)
            bnd_contrib = phi_bnd.unsqueeze(-1) * self._face_areas[n_internal:]
            grad_phi.index_add_(0, bnd_own, bnd_contrib)

        V = self._cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        grad_phi = grad_phi / V

        alpha_cell = torch.ones(n_cells, dtype=dtype, device=device)
        alpha_cell.scatter_reduce_(
            0, int_own, alpha_P, reduce="amin", include_self=True,
        )
        alpha_cell.scatter_reduce_(
            0, int_nei, alpha_N, reduce="amin", include_self=True,
        )

        needs_limiting = alpha_cell < 1.0 - 1e-10
        blend = self._blend_base * alpha_face.min()
        limited_grad = alpha_cell.unsqueeze(-1) * grad_phi
        result = grad_phi.clone()
        result = torch.where(
            needs_limiting.unsqueeze(-1),
            (1.0 - blend) * limited_grad + blend * grad_unlimited,
            result,
        )

        return result


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
