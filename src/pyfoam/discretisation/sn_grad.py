"""
Surface-normal gradient schemes for finite volume discretisation.

Provides the :class:`SnGradScheme` abstract base class and concrete
implementations for computing the face-normal gradient used in
Laplacian discretisation.

Schemes
-------
UncorrectedSnGrad
    Simple ``(phi_N - phi_P) * delta`` — exact for orthogonal meshes.
CorrectedSnGrad
    Full non-orthogonal correction using interpolated cell gradient.
LimitedSnGrad
    Limited non-orthogonal correction with coefficient *k* in [0, 1].
OrthogonalSnGrad
    Simple orthogonal snGrad using |S|/(d . S) form — fast path for
    orthogonal meshes.
OverRelaxedSnGrad
    Over-relaxed correction: divides by (d_hat . n_hat) to obtain the
    full normal gradient, better for non-orthogonal meshes.
BoundedSnGrad
    Bounded snGrad: bounds the face-normal gradient to prevent overshoots
    beyond the owner/neighbour cell values.

The surface-normal gradient controls how the Laplacian operator
``nabla cdot (D nabla phi)`` is discretised:

.. math::

    \\nabla \\cdot (D \\nabla \\phi)
    = \\frac{1}{V_P} \\sum_f D_f \\, \\text{snGrad}_f \\, |S_f|
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from pyfoam.core.backend import gather

__all__ = [
    "SnGradScheme",
    "UncorrectedSnGrad",
    "CorrectedSnGrad",
    "LimitedSnGrad",
    "OrthogonalSnGrad",
    "OverRelaxedSnGrad",
    "BoundedSnGrad",
    "sn_grad_from_name",
]


# ---------------------------------------------------------------------------
# Scheme registry
# ---------------------------------------------------------------------------

_SN_GRAD_REGISTRY: dict[str, type["SnGradScheme"]] = {}


def sn_grad_from_name(
    name: str,
    mesh: Any,
    **kwargs,
) -> "SnGradScheme":
    """Create an snGrad scheme from a registry name.

    Args:
        name: Scheme name (e.g. ``"uncorrected"``, ``"corrected"``,
            ``"limited"``).
        mesh: The ``FvMesh``.
        **kwargs: Additional arguments forwarded to the scheme constructor
            (e.g. ``k_coeff`` for ``"limited"``).

    Returns:
        An :class:`SnGradScheme` instance.

    Raises:
        ValueError: If *name* is not in the registry.
    """
    if name not in _SN_GRAD_REGISTRY:
        raise ValueError(
            f"Unknown snGrad scheme '{name}'. "
            f"Available: {list(_SN_GRAD_REGISTRY.keys())}"
        )
    return _SN_GRAD_REGISTRY[name](mesh, **kwargs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_correction_vectors(mesh: Any) -> torch.Tensor:
    r"""Compute non-orthogonal correction vectors for internal faces.

    Uses the *minimum correction* approach:

    .. math::

        \vec{k}_f = \hat{n}_f - \hat{d}_f \, (\hat{d}_f \cdot \hat{n}_f)

    where :math:`\hat{n}_f` is the unit face normal and
    :math:`\hat{d}_f` is the unit vector from owner to neighbour cell
    centre.

    Args:
        mesh: The ``FvMesh``.

    Returns:
        ``(n_internal_faces, 3)`` correction vectors.
    """
    device = mesh.device
    dtype = mesh.dtype
    n_internal = mesh.n_internal_faces

    if n_internal == 0:
        return torch.zeros(0, 3, dtype=dtype, device=device)

    cc = mesh.cell_centres
    fa = mesh.face_areas

    # d = x_N - x_P
    d = cc[mesh.neighbour[:n_internal]] - cc[mesh.owner[:n_internal]]
    d_mag = d.norm(dim=1, keepdim=True)
    d_hat = d / d_mag.clamp(min=1e-30)

    # Unit face normal
    n_hat = fa[:n_internal] / fa[:n_internal].norm(dim=1, keepdim=True).clamp(
        min=1e-30
    )

    # k = n_hat - d_hat * (d_hat . n_hat)
    d_dot_n = (d_hat * n_hat).sum(dim=1, keepdim=True)
    return n_hat - d_hat * d_dot_n


def _compute_cell_gradient(mesh: Any, phi: torch.Tensor) -> torch.Tensor:
    """Compute cell-centre gradient via the Gauss theorem.

    Same algorithm as ``fvc.grad`` but inlined to avoid a circular
    import with :mod:`pyfoam.discretisation.operators`.

    Args:
        mesh: The ``FvMesh``.
        phi: ``(n_cells,)`` scalar field.

    Returns:
        ``(n_cells, 3)`` gradient vectors.
    """
    device = mesh.device
    dtype = mesh.dtype
    n_cells = mesh.n_cells
    n_internal = mesh.n_internal_faces
    n_faces = mesh.n_faces
    fa = mesh.face_areas
    V = mesh.cell_volumes

    phi = phi.to(device=device, dtype=dtype)

    # Linear interpolation to faces (inlined for 1-D)
    w = mesh.face_weights  # (n_faces,)
    phi_P = gather(phi, mesh.owner[:n_internal])
    phi_N = gather(phi, mesh.neighbour)
    phi_face = torch.zeros(n_faces, dtype=dtype, device=device)
    phi_face[:n_internal] = w[:n_internal] * phi_P + (1.0 - w[:n_internal]) * phi_N
    if n_faces > n_internal:
        phi_face[n_internal:] = gather(phi, mesh.owner[n_internal:])

    # Gauss: grad_P = (1/V) * sum_f phi_f * S_f
    face_contrib = phi_face.unsqueeze(-1) * fa  # (n_faces, 3)
    grad_phi = torch.zeros(n_cells, 3, dtype=dtype, device=device)
    grad_phi.index_add_(0, mesh.owner[:n_internal], face_contrib[:n_internal])
    grad_phi.index_add_(
        0, mesh.neighbour, -face_contrib[:n_internal],
    )
    if n_faces > n_internal:
        grad_phi.index_add_(
            0, mesh.owner[n_internal:], face_contrib[n_internal:],
        )

    vol = V.unsqueeze(-1).clamp(min=1e-30)
    return grad_phi / vol


def _interpolate_vector_to_faces(
    mesh: Any, vec: torch.Tensor,
) -> torch.Tensor:
    """Linearly interpolate a ``(n_cells, 3)`` field to faces.

    Uses the pre-computed :pyattr:`FvMesh.face_weights`.

    Args:
        mesh: The ``FvMesh``.
        vec: ``(n_cells, 3)`` cell-centre values.

    Returns:
        ``(n_faces, 3)`` face values.
    """
    device = mesh.device
    dtype = mesh.dtype
    n_faces = mesh.n_faces
    n_internal = mesh.n_internal_faces

    w = mesh.face_weights  # (n_faces,)
    result = torch.zeros(n_faces, 3, dtype=dtype, device=device)

    if n_internal > 0:
        vec_P = vec[mesh.owner[:n_internal]]   # (n_internal, 3)
        vec_N = vec[mesh.neighbour[:n_internal]]
        wt = w[:n_internal].unsqueeze(-1)      # (n_internal, 1)
        result[:n_internal] = wt * vec_P + (1.0 - wt) * vec_N

    if n_faces > n_internal:
        result[n_internal:] = vec[mesh.owner[n_internal:]]

    return result


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class SnGradScheme(ABC):
    """Abstract base for surface-normal gradient schemes.

    A snGrad scheme computes the gradient of a
    :class:`~pyfoam.core.geometric_field.volScalarField` in the
    direction of the face normal at each face.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    """

    def __init__(self, mesh: Any) -> None:
        self._mesh = mesh

    @property
    def mesh(self) -> Any:
        """The finite volume mesh."""
        return self._mesh

    @abstractmethod
    def sn_grad(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute surface-normal gradient at each face.

        Args:
            phi: ``(n_cells,)`` cell-centre scalar values.

        Returns:
            ``(n_faces,)`` tensor.  Internal faces contain the snGrad
            values; boundary faces are **zero** (boundary conditions
            contribute separately).
        """

    def __call__(self, phi: torch.Tensor) -> torch.Tensor:
        """Callable interface — delegates to :meth:`sn_grad`."""
        return self.sn_grad(phi)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mesh={self._mesh})"


# ---------------------------------------------------------------------------
# Uncorrected scheme
# ---------------------------------------------------------------------------


class UncorrectedSnGrad(SnGradScheme):
    """Uncorrected surface-normal gradient.

    For each internal face *f*:

    .. math::

        \\text{snGrad}_f = \\delta_f \\,(\\phi_N - \\phi_P)

    where :math:`\\delta_f = 1/(\\vec{d} \\cdot \\hat{n}_f)` is the
    delta coefficient.

    This is exact for orthogonal meshes and provides a zeroth-order
    approximation for non-orthogonal meshes.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    """

    def sn_grad(self, phi: torch.Tensor) -> torch.Tensor:
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_faces = mesh.n_faces
        n_internal = mesh.n_internal_faces
        delta = mesh.delta_coefficients

        phi = phi.to(device=device, dtype=dtype)
        result = torch.zeros(n_faces, dtype=dtype, device=device)

        if n_internal == 0:
            return result

        phi_P = gather(phi, mesh.owner[:n_internal])
        phi_N = gather(phi, mesh.neighbour)
        result[:n_internal] = delta[:n_internal] * (phi_N - phi_P)
        return result


_SN_GRAD_REGISTRY["uncorrected"] = UncorrectedSnGrad


# ---------------------------------------------------------------------------
# Corrected scheme
# ---------------------------------------------------------------------------


class CorrectedSnGrad(SnGradScheme):
    """Fully corrected surface-normal gradient.

    For each internal face *f*:

    .. math::

        \\text{snGrad}_f = \\delta_f \\,(\\phi_N - \\phi_P)
        + \\vec{k}_f \\cdot \\nabla \\phi_f

    where :math:`\\vec{k}_f` is the non-orthogonal correction vector
    (minimum-correction approach) and :math:`\\nabla \\phi_f` is the
    linearly interpolated cell gradient at the face.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    """

    def __init__(self, mesh: Any) -> None:
        super().__init__(mesh)
        self._k = _compute_correction_vectors(mesh)

    def sn_grad(self, phi: torch.Tensor) -> torch.Tensor:
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_faces = mesh.n_faces
        n_internal = mesh.n_internal_faces
        delta = mesh.delta_coefficients

        phi = phi.to(device=device, dtype=dtype)
        result = torch.zeros(n_faces, dtype=dtype, device=device)

        if n_internal == 0:
            return result

        # --- Uncorrected part ---
        phi_P = gather(phi, mesh.owner[:n_internal])
        phi_N = gather(phi, mesh.neighbour)
        result[:n_internal] = delta[:n_internal] * (phi_N - phi_P)

        # --- Non-orthogonal correction ---
        grad_phi = _compute_cell_gradient(mesh, phi)          # (n_cells, 3)
        grad_face = _interpolate_vector_to_faces(mesh, grad_phi)  # (n_faces, 3)

        # correction = k . grad_f
        correction = (self._k * grad_face[:n_internal]).sum(dim=1)
        result[:n_internal] = result[:n_internal] + correction

        return result


_SN_GRAD_REGISTRY["corrected"] = CorrectedSnGrad


# ---------------------------------------------------------------------------
# Limited scheme
# ---------------------------------------------------------------------------


class LimitedSnGrad(SnGradScheme):
    """Limited surface-normal gradient.

    For each internal face *f*:

    .. math::

        \\text{snGrad}_f = \\delta_f \\,(\\phi_N - \\phi_P)
        + k_{\\text{coeff}} \\, \\vec{k}_f \\cdot \\nabla \\phi_f

    where :math:`k_{\\text{coeff}} \\in [0, 1]` limits the
    non-orthogonal correction:

    - ``k_coeff = 0`` — equivalent to :class:`UncorrectedSnGrad`.
    - ``k_coeff = 1`` — equivalent to :class:`CorrectedSnGrad`.
    - ``0 < k_coeff < 1`` — partial correction (improves stability
      on highly non-orthogonal meshes).

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    k_coeff : float
        Limiting coefficient in [0, 1].  Default is 0.5.
    """

    def __init__(self, mesh: Any, k_coeff: float = 0.5) -> None:
        super().__init__(mesh)
        if not 0.0 <= k_coeff <= 1.0:
            raise ValueError(
                f"k_coeff must be in [0, 1], got {k_coeff}"
            )
        self._k_coeff = k_coeff
        self._k = _compute_correction_vectors(mesh)

    @property
    def k_coeff(self) -> float:
        """The limiting coefficient."""
        return self._k_coeff

    def sn_grad(self, phi: torch.Tensor) -> torch.Tensor:
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_faces = mesh.n_faces
        n_internal = mesh.n_internal_faces
        delta = mesh.delta_coefficients

        phi = phi.to(device=device, dtype=dtype)
        result = torch.zeros(n_faces, dtype=dtype, device=device)

        if n_internal == 0:
            return result

        # --- Uncorrected part ---
        phi_P = gather(phi, mesh.owner[:n_internal])
        phi_N = gather(phi, mesh.neighbour)
        result[:n_internal] = delta[:n_internal] * (phi_N - phi_P)

        # --- Limited non-orthogonal correction ---
        if self._k_coeff > 0.0:
            grad_phi = _compute_cell_gradient(mesh, phi)
            grad_face = _interpolate_vector_to_faces(mesh, grad_phi)
            correction = (self._k * grad_face[:n_internal]).sum(dim=1)
            result[:n_internal] = result[:n_internal] + self._k_coeff * correction

        return result

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mesh={self._mesh}, "
            f"k_coeff={self._k_coeff})"
        )


_SN_GRAD_REGISTRY["limited"] = LimitedSnGrad


# ---------------------------------------------------------------------------
# Orthogonal scheme
# ---------------------------------------------------------------------------


class OrthogonalSnGrad(SnGradScheme):
    """Simple orthogonal surface-normal gradient.

    For each internal face *f*:

    .. math::

        \\text{snGrad}_f = \\frac{\\phi_N - \\phi_P}{|\\mathbf{d}_f|}

    where :math:`|\\mathbf{d}_f|` is the distance between the owner and
    neighbour cell centres projected onto the face normal direction.

    This is equivalent to the uncorrected scheme but uses
    ``|S|/(d . S)`` as the delta coefficient — exact for orthogonal
    meshes with no non-orthogonal correction.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    """

    def sn_grad(self, phi: torch.Tensor) -> torch.Tensor:
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_faces = mesh.n_faces
        n_internal = mesh.n_internal_faces

        phi = phi.to(device=device, dtype=dtype)
        result = torch.zeros(n_faces, dtype=dtype, device=device)

        if n_internal == 0:
            return result

        cc = mesh.cell_centres.to(device=device, dtype=dtype)
        fa = mesh.face_areas.to(device=device, dtype=dtype)

        # d = x_N - x_P
        d = cc[mesh.neighbour[:n_internal]] - cc[mesh.owner[:n_internal]]

        # |d . S| / |S|^2 gives 1/(d_hat . n_hat) * 1/|d|
        # For orthogonal: d . S = |d| * |S| (since d || n)
        # snGrad = (phi_N - phi_P) * |S| / (d . S)
        S = fa[:n_internal]  # (n_int, 3)
        d_dot_S = (d * S).sum(dim=1)  # (n_int,)
        S_mag = S.norm(dim=1)  # (n_int,)

        # delta = |S| / (d . S)  →  snGrad = delta * (phi_N - phi_P)
        delta = S_mag / d_dot_S.clamp(min=1e-30)

        phi_P = gather(phi, mesh.owner[:n_internal])
        phi_N = gather(phi, mesh.neighbour)
        result[:n_internal] = delta * (phi_N - phi_P)

        return result


_SN_GRAD_REGISTRY["orthogonal"] = OrthogonalSnGrad


# ---------------------------------------------------------------------------
# Over-relaxed scheme
# ---------------------------------------------------------------------------


class OverRelaxedSnGrad(SnGradScheme):
    r"""Over-relaxed surface-normal gradient.

    For each internal face *f*:

    .. math::

        \text{snGrad}_f = \frac{\phi_N - \phi_P}
        {|\mathbf{d}_f| \, (\hat{\mathbf{d}}_f \cdot \hat{\mathbf{n}}_f)}

    This divides by the cosine of the angle between the cell-centre
    displacement and the face normal, effectively "over-relaxing" the
    gradient for non-orthogonal meshes.  The result is the full
    face-normal component of the gradient rather than the component
    along *d*.

    On perfectly orthogonal meshes, :math:`\hat{\mathbf{d}} \cdot
    \hat{\mathbf{n}} = 1` and this reduces to the uncorrected scheme.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    """

    def sn_grad(self, phi: torch.Tensor) -> torch.Tensor:
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_faces = mesh.n_faces
        n_internal = mesh.n_internal_faces

        phi = phi.to(device=device, dtype=dtype)
        result = torch.zeros(n_faces, dtype=dtype, device=device)

        if n_internal == 0:
            return result

        cc = mesh.cell_centres.to(device=device, dtype=dtype)
        fa = mesh.face_areas.to(device=device, dtype=dtype)

        # d = x_N - x_P
        d = cc[mesh.neighbour[:n_internal]] - cc[mesh.owner[:n_internal]]
        d_mag = d.norm(dim=1)  # (n_int,)
        d_hat = d / d_mag.clamp(min=1e-30)

        # Unit face normal
        n_hat = fa[:n_internal] / fa[:n_internal].norm(dim=1, keepdim=True).clamp(
            min=1e-30,
        )

        # d_hat . n_hat (cosine of angle between d and n)
        cos_angle = (d_hat * n_hat).sum(dim=1)  # (n_int,)

        # Over-relaxed delta: 1 / (|d| * cos(angle))
        # For orthogonal: cos(angle) = 1, so delta = 1/|d| = delta_coeff
        # For non-orthogonal: delta is larger → over-relaxed
        delta = 1.0 / (d_mag * cos_angle.clamp(min=1e-30))

        phi_P = gather(phi, mesh.owner[:n_internal])
        phi_N = gather(phi, mesh.neighbour)
        result[:n_internal] = delta * (phi_N - phi_P)

        return result


_SN_GRAD_REGISTRY["overRelaxed"] = OverRelaxedSnGrad


# ---------------------------------------------------------------------------
# Bounded scheme
# ---------------------------------------------------------------------------


class BoundedSnGrad(SnGradScheme):
    r"""Bounded surface-normal gradient.

    Computes the snGrad using the fully corrected scheme, then bounds
    the result to prevent overshoots / undershoots.  The bounding
    ensures that the face-normal gradient does not imply face values
    outside the range defined by the owner and neighbour cell values.

    For each internal face *f*:

    1. Compute the uncorrected snGrad:
       ``snGrad_unc = delta * (phi_N - phi_P)``

    2. Compute the full corrected snGrad (with non-orthogonal
       correction).

    3. Bound: ``snGrad_bounded = clamp(snGrad_full, 0, snGrad_unc)``
       or vice versa, ensuring the correction does not change the
       sign of the uncorrected gradient.

    This is useful for maintaining boundedness in transport equations
    on non-orthogonal meshes.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    """

    def __init__(self, mesh: Any) -> None:
        super().__init__(mesh)
        self._k = _compute_correction_vectors(mesh)

    def sn_grad(self, phi: torch.Tensor) -> torch.Tensor:
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_faces = mesh.n_faces
        n_internal = mesh.n_internal_faces
        delta = mesh.delta_coefficients

        phi = phi.to(device=device, dtype=dtype)
        result = torch.zeros(n_faces, dtype=dtype, device=device)

        if n_internal == 0:
            return result

        # --- Uncorrected part ---
        phi_P = gather(phi, mesh.owner[:n_internal])
        phi_N = gather(phi, mesh.neighbour)
        sn_grad_unc = delta[:n_internal] * (phi_N - phi_P)

        # --- Non-orthogonal correction ---
        grad_phi = _compute_cell_gradient(mesh, phi)
        grad_face = _interpolate_vector_to_faces(mesh, grad_phi)
        correction = (self._k * grad_face[:n_internal]).sum(dim=1)

        sn_grad_full = sn_grad_unc + correction

        # --- Bounding ---
        # Bound the full gradient between 0 and uncorrected (or vice versa)
        lo = torch.min(sn_grad_unc, torch.zeros_like(sn_grad_unc))
        hi = torch.max(sn_grad_unc, torch.zeros_like(sn_grad_unc))

        result[:n_internal] = torch.clamp(sn_grad_full, lo, hi)

        return result


_SN_GRAD_REGISTRY["bounded"] = BoundedSnGrad
