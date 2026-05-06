"""
Limited-linear (TVD) interpolation scheme.

Combines linear interpolation with a flux limiter to prevent overshoots
while maintaining second-order accuracy in smooth regions.

The limiter adjusts the linear interpolation toward upwind:

    φ_f = φ_upwind + ψ(r) * (φ_f_linear - φ_upwind)

where r is the ratio of consecutive gradients and ψ(r) is the limiter function.
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.weights import compute_centre_weights

__all__ = ["LimitedLinearInterpolation"]


def _van_leer_limiter(r: torch.Tensor) -> torch.Tensor:
    """Van Leer flux limiter: ψ(r) = (r + |r|) / (1 + |r|)."""
    abs_r = r.abs()
    return (r + abs_r) / (1.0 + abs_r)


def _minmod_limiter(r: torch.Tensor) -> torch.Tensor:
    """Minmod flux limiter: ψ(r) = max(0, min(1, r))."""
    return torch.clamp(r, 0.0, 1.0)


def _superbee_limiter(r: torch.Tensor) -> torch.Tensor:
    """Superbee flux limiter: ψ(r) = max(0, min(2r, 1), min(r, 2))."""
    return torch.max(
        torch.zeros_like(r),
        torch.min(2.0 * r, torch.ones_like(r)),
    ).max(torch.min(r, 2.0 * torch.ones_like(r)))


_LIMITERS = {
    "vanLeer": _van_leer_limiter,
    "minmod": _minmod_limiter,
    "superbee": _superbee_limiter,
}


class LimitedLinearInterpolation(InterpolationScheme):
    """TVD limited-linear interpolation scheme.

    Applies a flux limiter to the linear interpolation to prevent
    overshoots while preserving second-order accuracy in smooth regions.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    limiter : str
        Limiter function name.  One of ``"vanLeer"``, ``"minmod"``, ``"superbee"``.
        Default is ``"vanLeer"``.
    """

    def __init__(self, mesh, limiter: str = "vanLeer") -> None:
        super().__init__(mesh)
        if limiter not in _LIMITERS:
            raise ValueError(
                f"Unknown limiter '{limiter}'. "
                f"Choose from: {list(_LIMITERS.keys())}"
            )
        self._limiter_name = limiter
        self._limiter_fn = _LIMITERS[limiter]
        self._weights = compute_centre_weights(
            mesh.cell_centres,
            mesh.face_centres,
            mesh.owner,
            mesh.neighbour,
            mesh.n_internal_faces,
            mesh.n_faces,
            device=mesh.device,
            dtype=mesh.dtype,
        )

    @property
    def limiter_name(self) -> str:
        """Name of the active limiter."""
        return self._limiter_name

    def interpolate(
        self,
        phi: torch.Tensor,
        face_flux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Limited-linear interpolation of cell values to faces.

        Args:
            phi: ``(n_cells,)`` cell-centre values (must be 1-D).
            face_flux: ``(n_faces,)`` volumetric flux.  Required.

        Returns:
            ``(n_faces,)`` face values.

        Raises:
            ValueError: If *phi* is not 1-D or *face_flux* is None.
        """
        if phi.dim() != 1:
            raise ValueError(
                f"Expected 1-D input tensor, got {phi.dim()}-D. "
                f"Interpolation operates on scalar fields only."
            )
        if face_flux is None:
            raise ValueError(
                "LimitedLinearInterpolation requires 'face_flux'."
            )

        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_faces = mesh.n_faces
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour

        phi = phi.to(device=device, dtype=dtype)
        face_flux = face_flux.to(device=device, dtype=dtype)
        face_values = torch.zeros(n_faces, dtype=dtype, device=device)

        if n_internal == 0:
            face_values = gather(phi, owner)
            return face_values

        phi_P = gather(phi, owner[:n_internal])
        phi_N = gather(phi, neighbour[:n_internal])

        # Upwind values
        int_flux = face_flux[:n_internal]
        is_positive = int_flux >= 0.0
        phi_upwind = torch.where(is_positive, phi_P, phi_N)

        # Linear interpolation
        w = self._weights[:n_internal]
        phi_linear = w * phi_P + (1.0 - w) * phi_N

        # For TVD limiting, we need the ratio r of consecutive gradients.
        # For a 2-cell mesh, we don't have enough stencil points,
        # so we fall back to linear interpolation.
        if mesh.n_cells <= 2:
            face_values[:n_internal] = phi_linear
        else:
            # Compute r = (φ_upwind - φ_2upwind) / (φ_downwind - φ_upwind)
            # Simplified: use gradient-based estimate
            grad_phi = self._compute_cell_gradients(phi)
            grad_P = gather(grad_phi, owner[:n_internal])
            grad_N = gather(grad_phi, neighbour[:n_internal])

            cc_P = gather(mesh.cell_centres, owner[:n_internal])
            cc_N = gather(mesh.cell_centres, neighbour[:n_internal])
            fc = mesh.face_centres[:n_internal]

            d_upwind = torch.where(
                is_positive.unsqueeze(-1),
                fc - cc_P,
                fc - cc_N,
            )
            grad_upwind = torch.where(
                is_positive.unsqueeze(-1), grad_P, grad_N
            )

            phi_face_grad = phi_upwind + (grad_upwind * d_upwind).sum(dim=1)

            denom = phi_linear - phi_upwind
            safe_denom = torch.where(
                denom.abs() > 1e-30, denom, torch.ones_like(denom) * 1e-30
            )
            r = (phi_face_grad - phi_upwind) / safe_denom
            psi = self._limiter_fn(r)

            face_values[:n_internal] = (
                phi_upwind + psi * (phi_linear - phi_upwind)
            )

        # Boundary faces: use owner values
        if n_faces > n_internal:
            face_values[n_internal:] = gather(phi, owner[n_internal:])

        return face_values

    def _compute_cell_gradients(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute cell-centre gradients using Gauss theorem."""
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour

        phi_face = torch.zeros(mesh.n_faces, dtype=dtype, device=device)
        if n_internal > 0:
            phi_P = gather(phi, owner[:n_internal])
            phi_N = gather(phi, neighbour[:n_internal])
            phi_face[:n_internal] = 0.5 * (phi_P + phi_N)
        if mesh.n_faces > n_internal:
            phi_face[n_internal:] = gather(phi, owner[n_internal:])

        face_contrib = phi_face.unsqueeze(-1) * mesh.face_areas
        grad = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        int_own = owner[:n_internal]
        int_nei = neighbour[:n_internal]
        grad.index_add_(0, int_own, face_contrib[:n_internal])
        grad.index_add_(0, int_nei, -face_contrib[:n_internal])
        if mesh.n_faces > n_internal:
            grad.index_add_(0, owner[n_internal:], face_contrib[n_internal:])

        V = mesh.cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        return grad / V
