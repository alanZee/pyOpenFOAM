"""
Interface compression for Volume of Fluid (VOF) methods.

Implements the interface compression technique from OpenFOAM, which
adds an artificial compressive velocity to sharpen the diffuse
interface in VOF simulations.

The compressive velocity is directed from cells with low alpha toward
cells with high alpha, effectively pushing the interface toward a
sharp step function.  The compression flux on each internal face is:

    phi_c = C_alpha * max(|phi|) * (alpha_P - alpha_N)

where:
    - C_alpha is the compression coefficient (0 = off, 1 = full)
    - max(|phi|) is the maximum face flux magnitude
    - alpha_P, alpha_N are owner/neighbour volume fractions

The compressive velocity at each face is then:

    U_c = phi_c * nf / |Sf|

where nf is the face normal vector and |Sf| is the face area.

Usage::

    from pyfoam.multiphase.interface_compression import InterfaceCompression

    comp = InterfaceCompression(mesh, C_alpha=1.0)
    U_compressive = comp.compute_compressive_velocity(alpha, phi)
    phi_compressive = comp.compute_compression_flux(alpha, phi)
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["InterfaceCompression"]

logger = logging.getLogger(__name__)


class InterfaceCompression:
    """Interface compression model for VOF simulations.

    Computes compressive fluxes and velocities that sharpen the
    volume fraction interface.  This is the Python equivalent of
    OpenFOAM's ``interfaceCompression`` scheme.

    Parameters
    ----------
    mesh : Any
        The finite volume mesh.  Must provide ``n_cells``,
        ``n_internal_faces``, ``owner``, ``neighbour``,
        ``face_areas``, ``face_normals`` (or equivalent).
    C_alpha : float
        Compression coefficient.  0 = no compression (pure upwind),
        1 = full compression.  Typical range: 0.5 to 2.0.
        Default 1.0.
    alpha_min : float
        Minimum volume fraction for clamping.  Default 0.0.
    alpha_max : float
        Maximum volume fraction for clamping.  Default 1.0.

    Examples::

        comp = InterfaceCompression(mesh, C_alpha=1.0)
        phi_c = comp.compute_compression_flux(alpha, phi)
        U_c = comp.compute_compressive_velocity(alpha, phi)
    """

    def __init__(
        self,
        mesh: Any,
        C_alpha: float = 1.0,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
    ) -> None:
        self._mesh = mesh
        self._C_alpha = C_alpha
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._device = get_device()
        self._dtype = get_default_dtype()

    @property
    def C_alpha(self) -> float:
        """Return the compression coefficient."""
        return self._C_alpha

    @C_alpha.setter
    def C_alpha(self, value: float) -> None:
        self._C_alpha = float(value)

    @property
    def alpha_min(self) -> float:
        """Return the minimum volume fraction bound."""
        return self._alpha_min

    @property
    def alpha_max(self) -> float:
        """Return the maximum volume fraction bound."""
        return self._alpha_max

    def compute_compression_flux(
        self,
        alpha: torch.Tensor,
        phi: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the compression flux on all internal faces.

        The compression flux drives the interface toward a sharp step:

            phi_c = C_alpha * max(|phi|) * (alpha_P - alpha_N)

        Positive phi_c pushes alpha from low- to high-alpha cells.

        Args:
            alpha: Volume fraction ``(n_cells,)``.
            phi: Face flux ``(n_faces,)``.

        Returns:
            ``(n_internal_faces,)`` compression flux on internal faces.
        """
        mesh = self._mesh
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        alpha_P = gather(alpha, int_owner)
        alpha_N = gather(alpha, int_neigh)

        flux = phi[:n_internal]

        # Maximum face flux magnitude (global for consistency with OpenFOAM)
        phi_max = flux.abs().max().clamp(min=1e-30)

        # Compression flux: C_alpha * phi_max * (alpha_P - alpha_N)
        phi_c = self._C_alpha * phi_max * (alpha_P - alpha_N)

        return phi_c

    def _get_face_geometry(self, n_internal: int):
        """Extract face normals and areas from the mesh.

        Handles both FvMesh (scalar face_areas + face_normals) and
        SimpleMesh (vector face_areas, no face_normals) conventions.

        Returns (normals, areas) both of shape ``(n_internal, 3)``
        and ``(n_internal,)`` respectively.
        """
        mesh = self._mesh

        # Get face areas
        if hasattr(mesh, 'face_areas'):
            raw_areas = mesh.face_areas[:n_internal]
        else:
            return None, None

        raw_areas = raw_areas.to(device=self._device, dtype=self._dtype)

        # If face_areas is a scalar (n_faces,), derive normals separately
        if raw_areas.ndim == 1:
            # Scalar areas — need face_normals attribute
            if hasattr(mesh, 'face_normals'):
                normals = mesh.face_normals[:n_internal].to(
                    device=self._device, dtype=self._dtype
                )
                areas = raw_areas
            else:
                return None, None
        else:
            # Vector face_areas (n_faces, 3) — derive normals
            areas = raw_areas.norm(dim=-1).clamp(min=1e-30)
            normals = raw_areas / areas.unsqueeze(-1)

        return normals, areas

    def compute_compressive_velocity(
        self,
        alpha: torch.Tensor,
        phi: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the compressive cell-centre velocity field.

        The compressive velocity at each cell is obtained by averaging
        the face compression fluxes, then converting to a velocity:

            U_c[c] = (1/V[c]) * sum_f (phi_c * nf / |Sf|)

        where the sum is over faces adjacent to cell c.

        Args:
            alpha: Volume fraction ``(n_cells,)``.
            phi: Face flux ``(n_faces,)``.

        Returns:
            ``(n_cells, 3)`` compressive velocity field.
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        phi_c = self.compute_compression_flux(alpha, phi)

        normals, areas = self._get_face_geometry(n_internal)

        if normals is None or areas is None:
            # No geometry available — return zero
            return torch.zeros(
                n_cells, 3, dtype=self._dtype, device=self._device
            )

        # U_c_face = phi_c * n_hat / |Sf|
        U_c_face = (phi_c / areas.clamp(min=1e-30)).unsqueeze(-1) * normals  # (n_int, 3)

        # Scatter to owner and neighbour cells
        U_c = torch.zeros(n_cells, 3, dtype=self._dtype, device=self._device)
        U_c.scatter_add_(0, int_owner.unsqueeze(-1).expand_as(U_c_face), U_c_face)
        U_c.scatter_add_(0, int_neigh.unsqueeze(-1).expand_as(U_c_face), -U_c_face)

        # Divide by cell volume
        cell_volumes = mesh.cell_volumes.to(
            device=self._device, dtype=self._dtype
        ).clamp(min=1e-30)
        U_c = U_c / cell_volumes.unsqueeze(-1)

        return U_c

    def apply_compression(
        self,
        alpha: torch.Tensor,
        phi: torch.Tensor,
        delta_t: float,
    ) -> torch.Tensor:
        """Apply interface compression to the volume fraction field.

        This performs an explicit Euler update using the compression
        flux only (no advection), with optional clamping.

        Args:
            alpha: Current volume fraction ``(n_cells,)``.
            phi: Face flux ``(n_faces,)``.
            delta_t: Time step size (s).

        Returns:
            Updated volume fraction ``(n_cells,)``.
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        phi_c = self.compute_compression_flux(alpha, phi)

        # Compute divergence of compression flux
        div_phi_c = torch.zeros(
            n_cells, dtype=self._dtype, device=self._device
        )
        div_phi_c = div_phi_c + scatter_add(phi_c, int_owner, n_cells)
        div_phi_c = div_phi_c + scatter_add(-phi_c, int_neigh, n_cells)

        # Boundary faces: no compression flux (alpha boundary handled by BCs)
        V = mesh.cell_volumes.to(
            device=self._device, dtype=self._dtype
        ).clamp(min=1e-30)

        # Update: alpha_new = alpha - dt * div(phi_c) / V
        alpha_new = alpha - delta_t * div_phi_c / V
        alpha_new = alpha_new.clamp(min=self._alpha_min, max=self._alpha_max)

        return alpha_new

    def __repr__(self) -> str:
        return (
            f"InterfaceCompression(n_cells={self._mesh.n_cells}, "
            f"C_alpha={self._C_alpha})"
        )
