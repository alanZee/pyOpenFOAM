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
from pyfoam.discretisation.schemes.harmonic import HarmonicInterpolation
from pyfoam.discretisation.schemes.mid_point import MidPointInterpolation
from pyfoam.discretisation.schemes.lust import LUSTInterpolation
from pyfoam.discretisation.schemes.van_leer import VanLeerInterpolation
from pyfoam.discretisation.schemes.gamma import GammaInterpolation
from pyfoam.discretisation.schemes.interface_compression import InterfaceCompressionInterpolation
from pyfoam.discretisation.schemes.muscl import MUSCLInterpolation
from pyfoam.discretisation.schemes.central import CentralInterpolation
from pyfoam.discretisation.schemes.sfcd import SFCDInterpolation
from pyfoam.discretisation.schemes.cubic import CubicInterpolation
from pyfoam.discretisation.schemes.linear_fit import LinearFitInterpolation
from pyfoam.discretisation.schemes.filtered_linear import FilteredLinearInterpolation
from pyfoam.discretisation.schemes.blended import BlendedInterpolation
from pyfoam.discretisation.schemes.linear_fit_2 import LinearFit2Interpolation
from pyfoam.discretisation.schemes.cubic_upwind import CubicUpwindInterpolation
from pyfoam.discretisation.schemes.ami_interpolation import AMIInterpolation
from pyfoam.discretisation.schemes.linear_upwind_fit import LinearUpwindFitInterpolation
from pyfoam.discretisation.schemes.upwind_fit import UpwindFitInterpolation
from pyfoam.discretisation.schemes.cubic_upwind_fit import CubicUpwindFitInterpolation
from pyfoam.discretisation.schemes.filtered_linear_2 import FilteredLinear2Interpolation
from pyfoam.discretisation.schemes.filtered_linear_v import FilteredLinearVInterpolation
from pyfoam.discretisation.schemes.van_leer_v import VanLeerVInterpolation
from pyfoam.discretisation.schemes.muscl_v import MUSCLVInterpolation
from pyfoam.discretisation.schemes.gamma_v import GammaVInterpolation
from pyfoam.discretisation.schemes.clipped_linear import ClippedLinearInterpolation
from pyfoam.discretisation.schemes.corrected_linear import CorrectedLinearInterpolation
from pyfoam.discretisation.schemes.linear_upwind_fit_2 import LinearUpwindFit2Interpolation
from pyfoam.discretisation.schemes.upwind_fit_2 import UpwindFit2Interpolation
from pyfoam.discretisation.schemes.cubic_upwind_fit_2 import CubicUpwindFit2Interpolation
from pyfoam.discretisation.schemes.filtered_linear_3 import FilteredLinear3Interpolation
from pyfoam.discretisation.schemes.van_leer_v_2 import VanLeerV2Interpolation
from pyfoam.discretisation.schemes.muscl_v_2 import MUSCLV2Interpolation
from pyfoam.discretisation.schemes.gamma_v_2 import GammaV2Interpolation
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
    "harmonic": HarmonicInterpolation,
    "midPoint": MidPointInterpolation,
    "LUST": LUSTInterpolation,
    "vanLeer": VanLeerInterpolation,
    "gamma": GammaInterpolation,
    "interfaceCompression": InterfaceCompressionInterpolation,
    "MUSCL": MUSCLInterpolation,
    "central": CentralInterpolation,
    "SFCD": SFCDInterpolation,
    "cubic": CubicInterpolation,
    "linearFit": LinearFitInterpolation,
    "filteredLinear": FilteredLinearInterpolation,
    "blended": BlendedInterpolation,
    "linearFit2": LinearFit2Interpolation,
    "cubicUpwind": CubicUpwindInterpolation,
    "AMI": AMIInterpolation,
    "linearUpwindFit": LinearUpwindFitInterpolation,
    "upwindFit": UpwindFitInterpolation,
    "cubicUpwindFit": CubicUpwindFitInterpolation,
    "filteredLinear2": FilteredLinear2Interpolation,
    "filteredLinearV": FilteredLinearVInterpolation,
    "vanLeerV": VanLeerVInterpolation,
    "MUSCLV": MUSCLVInterpolation,
    "gammaV": GammaVInterpolation,
    "clippedLinear": ClippedLinearInterpolation,
    "correctedLinear": CorrectedLinearInterpolation,
    "linearUpwindFit2": LinearUpwindFit2Interpolation,
    "upwindFit2": UpwindFit2Interpolation,
    "cubicUpwindFit2": CubicUpwindFit2Interpolation,
    "filteredLinear3": FilteredLinear3Interpolation,
    "vanLeerV2": VanLeerV2Interpolation,
    "MUSCLV2": MUSCLV2Interpolation,
    "gammaV2": GammaV2Interpolation,
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
    elif name == "interfaceCompression":
        beta = kwargs.get("beta", 1.0)
        return scheme_cls(mesh, beta=beta)
    elif name == "blended":
        scheme1_name = kwargs.get("scheme1")
        scheme2_name = kwargs.get("scheme2")
        alpha = kwargs.get("alpha", 0.5)
        if scheme1_name is None or scheme2_name is None:
            raise ValueError(
                "BlendedInterpolation requires 'scheme1' and 'scheme2' kwargs."
            )
        scheme1 = _resolve_scheme(scheme1_name, mesh=mesh)
        scheme2 = _resolve_scheme(scheme2_name, mesh=mesh)
        return scheme_cls(mesh, scheme1=scheme1, scheme2=scheme2, alpha=alpha)
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

    @staticmethod
    def div(
        U_or_phi: Any,
        U: Any = None,
        scheme: str = "Gauss linear",
        *,
        mesh: Any = None,
    ) -> torch.Tensor:
        """Explicit divergence operator.

        Overloaded:
        - ``fvc.div(phi, U)`` — divergence of flux-weighted field ∇·(φU)
        - ``fvc.div(U, mesh=mesh)`` — divergence of vector field ∇·U

        When called with a single positional argument (vector field),
        computes ∇·U = Σ_i ∂U_i/∂x_i using the Gauss theorem.

        Returns:
            ``(n_cells,)`` tensor — the divergence at each cell.
        """
        # Distinguish the two call signatures
        if U is None:
            # Single argument: div of vector field
            return _FvcNamespace._div_vector(U_or_phi, scheme=scheme, mesh=mesh)
        else:
            # Two arguments: div of flux-weighted field (original API)
            return _FvcNamespace._div_flux(U_or_phi, U, scheme=scheme, mesh=mesh)

    @staticmethod
    def _div_flux(
        phi: Any,
        U: Any,
        scheme: str = "Gauss linear",
        *,
        mesh: Any = None,
    ) -> torch.Tensor:
        """Original div(phi, U) implementation."""
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
    def _div_vector(
        U: Any,
        scheme: str = "Gauss linear",
        *,
        mesh: Any = None,
    ) -> torch.Tensor:
        """Divergence of a vector field: ∇·U.

        Uses the Gauss theorem: ∇·U ≈ (1/V) Σ_f (U_f · S_f).

        Returns:
            ``(n_cells,)`` tensor.
        """
        if hasattr(U, "mesh"):
            mesh = U.mesh
            U_data = U.internal_field
        elif mesh is not None:
            U_data = U if isinstance(U, torch.Tensor) else torch.tensor(U)
        else:
            raise ValueError("mesh is required when U is a raw tensor")

        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces
        face_areas = mesh.face_areas
        cell_volumes = mesh.cell_volumes

        U_data = U_data.to(device=device, dtype=dtype)

        # Interpolate each component to faces (linear)
        interp = _resolve_scheme(scheme, mesh=mesh)

        div_U = torch.zeros(n_cells, dtype=dtype, device=device)

        for comp in range(U_data.shape[1] if U_data.dim() > 1 else 1):
            if U_data.dim() > 1:
                u_comp = U_data[:, comp]
            else:
                u_comp = U_data

            u_face = interp.interpolate(u_comp)
            # Flux: u_face * S_face_component
            if face_areas.dim() > 1:
                flux = u_face * face_areas[:, comp] if U_data.dim() > 1 else (
                    u_face * face_areas[:, 0]  # scalar: use x-component of area
                )
            else:
                flux = u_face * face_areas

            div_U = div_U + scatter_add(
                flux[:n_internal], mesh.owner[:n_internal], n_cells
            )
            div_U = div_U + scatter_add(
                -flux[:n_internal], mesh.neighbour, n_cells
            )
            if n_faces > n_internal:
                div_U = div_U + scatter_add(
                    flux[n_internal:], mesh.owner[n_internal:], n_cells
                )

        V = cell_volumes.clamp(min=1e-30)
        return div_U / V

    @staticmethod
    def curl(
        U: Any,
        scheme: str = "Gauss linear",
        *,
        mesh: Any = None,
    ) -> torch.Tensor:
        """Explicit curl of a vector field: ∇ × U.

        Uses the Gauss theorem: (∇ × U)_i ≈ (1/V) Σ_f (S_f × U_f)_i.

        Returns:
            ``(n_cells, 3)`` tensor — the curl vector at each cell.
        """
        if hasattr(U, "mesh"):
            mesh = U.mesh
            U_data = U.internal_field
        elif mesh is not None:
            U_data = U if isinstance(U, torch.Tensor) else torch.tensor(U)
        else:
            raise ValueError("mesh is required when U is a raw tensor")

        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces
        face_areas = mesh.face_areas
        cell_volumes = mesh.cell_volumes

        U_data = U_data.to(device=device, dtype=dtype)

        # Interpolate each component to faces
        interp = _resolve_scheme(scheme, mesh=mesh)

        # Interpolate vector to faces: (n_faces, 3)
        U_face = torch.zeros(n_faces, 3, dtype=dtype, device=device)
        for comp in range(3):
            U_face[:, comp] = interp.interpolate(U_data[:, comp])

        # Compute S_f × U_f for each face
        # face_areas: (n_faces, 3), U_face: (n_faces, 3)
        cross_face = torch.cross(face_areas, U_face, dim=1)  # (n_faces, 3)

        # Accumulate: owner gets +S×U, neighbour gets -S×U
        curl_U = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        curl_U.index_add_(0, mesh.owner[:n_internal], cross_face[:n_internal])
        curl_U.index_add_(0, mesh.neighbour, -cross_face[:n_internal])
        if n_faces > n_internal:
            curl_U.index_add_(0, mesh.owner[n_internal:], cross_face[n_internal:])

        V = cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        return curl_U / V


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

fvm = _FvmNamespace()
fvc = _FvcNamespace()
