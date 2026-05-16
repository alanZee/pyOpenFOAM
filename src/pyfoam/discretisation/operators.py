"""
Finite volume discretisation operators — fvm (implicit) and fvc (explicit).

Provides the ``fvm`` and ``fvc`` namespaces mirroring OpenFOAM's API:

- ``fvm.grad(phi)`` — implicit gradient (returns FvMatrix)
- ``fvm.div(phi, U)`` — implicit divergence (returns FvMatrix)
- ``fvm.laplacian(D, phi)`` — implicit Laplacian (returns FvMatrix)
- ``fvc.grad(phi)`` — explicit gradient (returns volVectorField)
- ``fvc.div(phi, U)`` — explicit divergence (returns volScalarField)
- ``fvc.laplacian(D, phi)`` — explicit Laplacian (returns volScalarField)
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.fv_matrix import FvMatrix

from pyfoam.discretisation.interpolation import InterpolationScheme, LinearInterpolation
from pyfoam.discretisation.schemes.upwind import UpwindInterpolation
from pyfoam.discretisation.schemes.linear_upwind import LinearUpwindInterpolation
from pyfoam.discretisation.schemes.limited_linear import LimitedLinearInterpolation
from pyfoam.discretisation.weights import compute_centre_weights

__all__ = ["fvm", "fvc"]


# ---------------------------------------------------------------------------
# Scheme registry
# ---------------------------------------------------------------------------

_SCHEME_REGISTRY: dict[str, type[InterpolationScheme]] = {
    "linear": LinearInterpolation,
    "upwind": UpwindInterpolation,
    "linearUpwind": LinearUpwindInterpolation,
    "limitedLinear": LimitedLinearInterpolation,
}


def _resolve_scheme(
    scheme_name: str,
    mesh: Any = None,
    **kwargs,
) -> InterpolationScheme:
    """Create an interpolation scheme from a name.

    Args:
        scheme_name: Scheme name (e.g. ``"Gauss linear"``, ``"upwind"``).
        mesh: The ``FvMesh`` (required for scheme construction).
        **kwargs: Additional arguments passed to the scheme constructor.

    Returns:
        An :class:`InterpolationScheme` instance.
    """
    name = scheme_name.strip()
    if name.startswith("Gauss "):
        name = name[6:].strip()

    if name not in _SCHEME_REGISTRY:
        raise ValueError(
            f"Unknown scheme '{scheme_name}'. "
            f"Available: {list(_SCHEME_REGISTRY.keys())}"
        )

    scheme_cls = _SCHEME_REGISTRY[name]

    if name == "limitedLinear":
        limiter = kwargs.get("limiter", "vanLeer")
        return scheme_cls(mesh, limiter=limiter)
    else:
        return scheme_cls(mesh)


# ---------------------------------------------------------------------------
# Helper: create FvMatrix with internal-face-only addressing
# ---------------------------------------------------------------------------

def _make_fvmatrix(mesh: Any, device=None, dtype=None) -> FvMatrix:
    """Create an FvMatrix using internal-face owner/neighbour arrays."""
    return FvMatrix(
        mesh.n_cells,
        mesh.owner[:mesh.n_internal_faces],
        mesh.neighbour,
        device=device or mesh.device,
        dtype=dtype or mesh.dtype,
    )


# ---------------------------------------------------------------------------
# fvm — Implicit operators (return FvMatrix)
# ---------------------------------------------------------------------------


class _FvmNamespace:
    """Implicit finite volume operators."""

    @staticmethod
    def ddt(
        coeff: float,
        phi: Any,
        dt: float,
        *,
        mesh: Any = None,
    ) -> FvMatrix:
        """Discretise the time derivative operator ∂φ/∂t (implicit Euler).

        Returns an FvMatrix where:
        - diagonal: coeff * V / dt
        - source: coeff * V * phi_old / dt

        Args:
            coeff: Coefficient (e.g. ρ or 1).
            phi: Current field values ``(n_cells,)`` or ``(n_cells, 3)``.
            dt: Time step size.
            mesh: The ``FvMesh``.

        Returns:
            :class:`FvMatrix` with time derivative coefficients.
        """
        if hasattr(phi, "mesh"):
            mesh = phi.mesh
            phi_data = phi.internal_field
        elif mesh is not None:
            phi_data = phi if isinstance(phi, torch.Tensor) else torch.tensor(phi)
        else:
            raise ValueError("mesh is required when phi is not a GeometricField")

        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        cell_volumes = mesh.cell_volumes

        phi_data = phi_data.to(device=device, dtype=dtype)

        # Create FvMatrix with zero off-diagonals (no face contributions)
        mat = _make_fvmatrix(mesh)

        # Time derivative: coeff * V / dt on diagonal
        diag = coeff * cell_volumes / dt
        mat.diag = diag

        # Source: coeff * V * phi_old / dt
        if phi_data.dim() == 1:
            mat.source = coeff * cell_volumes * phi_data / dt
        else:
            # For vector fields, sum over components
            mat.source = coeff * cell_volumes * phi_data.sum(dim=-1) / dt

        return mat

    @staticmethod
    def grad(
        phi: Any,
        scheme: str = "Gauss linear",
        *,
        mesh: Any = None,
    ) -> FvMatrix:
        """Discretise the gradient operator ∇φ (implicit).

        Args:
            phi: A ``volScalarField`` or ``(n_cells,)`` tensor.
            scheme: Discretisation scheme name.
            mesh: The ``FvMesh``.  Required if *phi* is a raw tensor.

        Returns:
            :class:`~pyfoam.core.fv_matrix.FvMatrix` with gradient coefficients.
        """
        if hasattr(phi, "mesh"):
            mesh = phi.mesh
            phi_data = phi.internal_field
        elif mesh is not None:
            phi_data = phi if isinstance(phi, torch.Tensor) else torch.tensor(phi)
        else:
            raise ValueError("mesh is required when phi is not a GeometricField")

        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces
        face_areas = mesh.face_areas
        cell_volumes = mesh.cell_volumes

        phi_data = phi_data.to(device=device, dtype=dtype)

        mat = _make_fvmatrix(mesh)

        S_mag = face_areas[:n_internal].norm(dim=1)

        weights = compute_centre_weights(
            mesh.cell_centres, mesh.face_centres,
            mesh.owner, mesh.neighbour, n_internal, n_faces,
            device=device, dtype=dtype,
        )
        w = weights[:n_internal]

        # Internal face owner/neighbour (for FvMatrix)
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        V_P = gather(cell_volumes, int_owner)
        V_N = gather(cell_volumes, int_neigh)

        lower_coeff = S_mag * (1.0 - w) / V_P
        upper_coeff = -S_mag * w / V_N

        mat.lower = lower_coeff
        mat.upper = upper_coeff

        diag = torch.zeros(n_cells, dtype=dtype, device=device)
        diag = diag + scatter_add(-S_mag * w / V_P, int_owner, n_cells)
        diag = diag + scatter_add(S_mag * (1.0 - w) / V_N, int_neigh, n_cells)

        if n_faces > n_internal:
            bnd_areas = face_areas[n_internal:]
            bnd_S_mag = bnd_areas.norm(dim=1) if bnd_areas.dim() > 1 else bnd_areas.abs()
            bnd_V = gather(cell_volumes, mesh.owner[n_internal:])
            diag = diag + scatter_add(bnd_S_mag / bnd_V, mesh.owner[n_internal:], n_cells)

        mat.diag = diag
        return mat

    @staticmethod
    def div(
        phi: Any,
        U: Any,
        scheme: str = "Gauss linear",
        *,
        mesh: Any = None,
    ) -> FvMatrix:
        """Discretise the divergence operator ∇·(φU) (implicit).

        Args:
            phi: Face flux ``(n_faces,)`` tensor or surfaceScalarField.
            U: A ``volScalarField`` or ``(n_cells,)`` tensor.
            scheme: Discretisation scheme name.
            mesh: The ``FvMesh``.  Required if fields are raw tensors.

        Returns:
            :class:`~pyfoam.core.fv_matrix.FvMatrix` with divergence coefficients.
        """
        if hasattr(U, "mesh"):
            mesh = U.mesh
            U_data = U.internal_field
        elif mesh is not None:
            U_data = U if isinstance(U, torch.Tensor) else torch.tensor(U)
        else:
            raise ValueError("mesh is required when U is not a GeometricField")

        if hasattr(phi, "internal_field"):
            phi_face = phi.internal_field
        else:
            phi_face = phi if isinstance(phi, torch.Tensor) else torch.tensor(phi)

        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        cell_volumes = mesh.cell_volumes

        U_data = U_data.to(device=device, dtype=dtype)
        phi_face = phi_face.to(device=device, dtype=dtype)

        mat = _make_fvmatrix(mesh)

        interp = _resolve_scheme(scheme, mesh=mesh)
        U_face = interp.interpolate(U_data, phi_face)

        flux = phi_face[:n_internal]
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        V_P = gather(cell_volumes, int_owner)
        V_N = gather(cell_volumes, int_neigh)

        is_positive = flux >= 0.0
        flux_pos = torch.where(is_positive, flux, torch.zeros_like(flux))
        flux_neg = torch.where(~is_positive, flux, torch.zeros_like(flux))

        mat.lower = flux_neg / V_P
        mat.upper = flux_pos / V_N

        diag = torch.zeros(n_cells, dtype=dtype, device=device)
        diag = diag + scatter_add(-flux_pos / V_P, int_owner, n_cells)
        diag = diag + scatter_add(flux_neg.abs() / V_N, int_neigh, n_cells)
        mat.diag = diag

        # Deferred correction for higher-order schemes
        if scheme not in ("Gauss upwind",):
            upwind_interp = UpwindInterpolation(mesh)
            U_upwind = upwind_interp.interpolate(U_data, phi_face)
            correction = (U_face[:n_internal] - U_upwind[:n_internal]) * flux
            source_corr = torch.zeros(n_cells, dtype=dtype, device=device)
            source_corr = source_corr + scatter_add(
                -correction / V_P, int_owner, n_cells
            )
            source_corr = source_corr + scatter_add(
                correction / V_N, int_neigh, n_cells
            )
            mat.source = mat.source + source_corr

        return mat

    @staticmethod
    def laplacian(
        D: Any,
        phi: Any,
        scheme: str = "Gauss linear corrected",
        *,
        mesh: Any = None,
    ) -> FvMatrix:
        """Discretise the Laplacian operator ∇·(D∇φ) (implicit).

        Args:
            D: Diffusion coefficient (scalar, tensor, or field).
            phi: A ``volScalarField`` or ``(n_cells,)`` tensor.
            scheme: Discretisation scheme name.
            mesh: The ``FvMesh``.  Required if fields are raw tensors.

        Returns:
            :class:`~pyfoam.core.fv_matrix.FvMatrix` with Laplacian coefficients.
        """
        if hasattr(phi, "mesh"):
            mesh = phi.mesh
            phi_data = phi.internal_field
        elif mesh is not None:
            phi_data = phi if isinstance(phi, torch.Tensor) else torch.tensor(phi)
        else:
            raise ValueError("mesh is required when phi is not a GeometricField")

        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces
        cell_volumes = mesh.cell_volumes
        delta_coeffs = mesh.delta_coefficients
        face_areas = mesh.face_areas

        phi_data = phi_data.to(device=device, dtype=dtype)

        # Resolve D
        if isinstance(D, (int, float)):
            D_face = torch.full(
                (n_faces,), float(D), dtype=dtype, device=device
            )
        elif hasattr(D, "internal_field"):
            D_data = D.internal_field.to(device=device, dtype=dtype)
            interp = LinearInterpolation(mesh)
            D_face = interp.interpolate(D_data)
        else:
            D_data = D.to(device=device, dtype=dtype) if isinstance(D, torch.Tensor) else torch.tensor(D, dtype=dtype, device=device)
            if D_data.dim() == 0:
                D_face = D_data.expand(n_faces)
            elif D_data.shape[0] == n_cells:
                interp = LinearInterpolation(mesh)
                D_face = interp.interpolate(D_data)
            else:
                D_face = D_data

        mat = _make_fvmatrix(mesh)

        S_mag = face_areas[:n_internal].norm(dim=1)
        delta = delta_coeffs[:n_internal]
        D_int = D_face[:n_internal]
        face_coeff = D_int * S_mag * delta

        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        V_P = gather(cell_volumes, int_owner)
        V_N = gather(cell_volumes, int_neigh)

        mat.lower = -face_coeff / V_P
        mat.upper = -face_coeff / V_N

        diag = torch.zeros(n_cells, dtype=dtype, device=device)
        diag = diag + scatter_add(face_coeff / V_P, int_owner, n_cells)
        diag = diag + scatter_add(face_coeff / V_N, int_neigh, n_cells)

        if n_faces > n_internal:
            bnd_areas = face_areas[n_internal:]
            bnd_S_mag = bnd_areas.norm(dim=1) if bnd_areas.dim() > 1 else bnd_areas.abs()
            bnd_delta = delta_coeffs[n_internal:]
            bnd_D = D_face[n_internal:]
            bnd_coeff = bnd_D * bnd_S_mag * bnd_delta
            bnd_V = gather(cell_volumes, mesh.owner[n_internal:])
            diag = diag + scatter_add(bnd_coeff / bnd_V, mesh.owner[n_internal:], n_cells)

        mat.diag = diag
        return mat


# ---------------------------------------------------------------------------
# fvc — Explicit operators (return field tensors)
# ---------------------------------------------------------------------------


class _FvcNamespace:
    """Explicit finite volume operators."""

    @staticmethod
    def grad(
        phi: Any,
        scheme: str = "Gauss linear",
        *,
        mesh: Any = None,
    ) -> torch.Tensor:
        """Explicit gradient ∇φ using Gauss theorem.

        Returns:
            ``(n_cells, 3)`` tensor — the gradient vector at each cell.
        """
        if hasattr(phi, "mesh"):
            mesh = phi.mesh
            phi_data = phi.internal_field
        elif mesh is not None:
            phi_data = phi if isinstance(phi, torch.Tensor) else torch.tensor(phi)
        else:
            raise ValueError("mesh is required when phi is not a GeometricField")

        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces
        face_areas = mesh.face_areas
        cell_volumes = mesh.cell_volumes

        phi_data = phi_data.to(device=device, dtype=dtype)

        interp = _resolve_scheme(scheme, mesh=mesh)
        phi_face = interp.interpolate(phi_data)

        face_contrib = phi_face.unsqueeze(-1) * face_areas

        grad_phi = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        int_own = mesh.owner[:n_internal]
        int_nei = mesh.neighbour
        grad_phi.index_add_(0, int_own, face_contrib[:n_internal])
        grad_phi.index_add_(0, int_nei, -face_contrib[:n_internal])
        if n_faces > n_internal:
            grad_phi.index_add_(0, mesh.owner[n_internal:], face_contrib[n_internal:])

        V = cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        return grad_phi / V

    @staticmethod
    def div(
        phi: Any,
        U: Any,
        scheme: str = "Gauss linear",
        *,
        mesh: Any = None,
    ) -> torch.Tensor:
        """Explicit divergence ∇·(φU) using Gauss theorem.

        Returns:
            ``(n_cells,)`` tensor — the divergence at each cell.
        """
        if hasattr(U, "mesh"):
            mesh = U.mesh
            U_data = U.internal_field
        elif mesh is not None:
            U_data = U if isinstance(U, torch.Tensor) else torch.tensor(U)
        else:
            raise ValueError("mesh is required when U is not a GeometricField")

        if hasattr(phi, "internal_field"):
            phi_face = phi.internal_field
        else:
            phi_face = phi if isinstance(phi, torch.Tensor) else torch.tensor(phi)

        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        cell_volumes = mesh.cell_volumes

        U_data = U_data.to(device=device, dtype=dtype)
        phi_face = phi_face.to(device=device, dtype=dtype)

        interp = _resolve_scheme(scheme, mesh=mesh)
        U_face = interp.interpolate(U_data, phi_face)

        flux = phi_face[:n_internal] * U_face[:n_internal]

        div_phi = torch.zeros(n_cells, dtype=dtype, device=device)
        div_phi = div_phi + scatter_add(flux, mesh.owner[:n_internal], n_cells)
        div_phi = div_phi + scatter_add(-flux, mesh.neighbour, n_cells)

        if mesh.n_faces > n_internal:
            bnd_flux = phi_face[n_internal:] * U_face[n_internal:]
            div_phi = div_phi + scatter_add(bnd_flux, mesh.owner[n_internal:], n_cells)

        V = cell_volumes.clamp(min=1e-30)
        return div_phi / V

    @staticmethod
    def laplacian(
        D: Any,
        phi: Any,
        scheme: str = "Gauss linear corrected",
        *,
        mesh: Any = None,
    ) -> torch.Tensor:
        """Explicit Laplacian ∇·(D∇φ) using Gauss theorem.

        Returns:
            ``(n_cells,)`` tensor — the Laplacian at each cell.
        """
        if hasattr(phi, "mesh"):
            mesh = phi.mesh
            phi_data = phi.internal_field
        elif mesh is not None:
            phi_data = phi if isinstance(phi, torch.Tensor) else torch.tensor(phi)
        else:
            raise ValueError("mesh is required when phi is not a GeometricField")

        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces
        cell_volumes = mesh.cell_volumes
        delta_coeffs = mesh.delta_coefficients
        face_areas = mesh.face_areas

        phi_data = phi_data.to(device=device, dtype=dtype)

        if isinstance(D, (int, float)):
            D_face = torch.full(
                (n_faces,), float(D), dtype=dtype, device=device
            )
        elif hasattr(D, "internal_field"):
            D_data = D.internal_field.to(device=device, dtype=dtype)
            interp = LinearInterpolation(mesh)
            D_face = interp.interpolate(D_data)
        else:
            D_data = D.to(device=device, dtype=dtype) if isinstance(D, torch.Tensor) else torch.tensor(D, dtype=dtype, device=device)
            if D_data.dim() == 0:
                D_face = D_data.expand(n_faces)
            elif D_data.shape[0] == n_cells:
                interp = LinearInterpolation(mesh)
                D_face = interp.interpolate(D_data)
            else:
                D_face = D_data

        S_mag = face_areas[:n_internal].norm(dim=1)
        delta = delta_coeffs[:n_internal]
        D_int = D_face[:n_internal]

        phi_P = gather(phi_data, mesh.owner[:n_internal])
        phi_N = gather(phi_data, mesh.neighbour)

        face_flux = D_int * (phi_N - phi_P) * S_mag * delta

        lap = torch.zeros(n_cells, dtype=dtype, device=device)
        lap = lap + scatter_add(face_flux, mesh.owner[:n_internal], n_cells)
        lap = lap + scatter_add(-face_flux, mesh.neighbour, n_cells)

        V = cell_volumes.clamp(min=1e-30)
        return lap / V


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

fvm = _FvmNamespace()
fvc = _FvcNamespace()
