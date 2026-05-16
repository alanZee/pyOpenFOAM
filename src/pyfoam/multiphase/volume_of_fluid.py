"""
Volume of Fluid (VOF) advection for multiphase flows.

Implements the VOF method for interface tracking between two immiscible
fluids. The volume fraction α ∈ [0, 1] indicates the proportion of the
second fluid in each cell:

- α = 0: cell contains only fluid 1
- α = 1: cell contains only fluid 2
- 0 < α < 1: cell contains an interface

The transport equation with interface compression:

    ∂α/∂t + ∇·(Uα) + ∇·(U_r α(1-α)) = 0

where U_r is a compression velocity that sharpens the interface.

Includes MULES (Multidimensional Universal Limiter with Explicit
Solution) limiting to ensure α ∈ [0, 1] while preserving conservation.

Usage::

    from pyfoam.multiphase.volume_of_fluid import VOFAdvection

    vof = VOFAdvection(mesh, alpha, phi, U, C_alpha=1.0)
    alpha = vof.advance(delta_t)
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["VOFAdvection"]

logger = logging.getLogger(__name__)


class VOFAdvection:
    """Volume of Fluid advection with interface compression.

    Solves the VOF transport equation:

        ∂α/∂t + ∇·(Uα) + ∇·(U_r α(1-α)) = 0

    The compression term ∇·(U_r α(1-α)) sharpens the interface by
    applying an artificial velocity U_r normal to the interface.

    Parameters
    ----------
    mesh : Any
        The finite volume mesh.
    alpha : torch.Tensor
        Initial volume fraction ``(n_cells,)``.
    phi : torch.Tensor
        Face flux ``(n_faces,)``.
    U : torch.Tensor
        Cell-centre velocity ``(n_cells, 3)``.
    C_alpha : float
        Compression coefficient (0 = no compression, 1 = full).
        Default 1.0.
    alpha_min : float
        Minimum volume fraction for clamping. Default 0.0.
    alpha_max : float
        Maximum volume fraction for clamping. Default 1.0.

    Examples::

        vof = VOFAdvection(mesh, alpha, phi, U, C_alpha=1.0)
        for t in range(n_steps):
            alpha = vof.advance(delta_t)
    """

    def __init__(
        self,
        mesh: Any,
        alpha: torch.Tensor,
        phi: torch.Tensor,
        U: torch.Tensor,
        C_alpha: float = 1.0,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        use_mules: bool = True,
        mules_iterations: int = 3,
    ) -> None:
        self._mesh = mesh
        self._device = get_device()
        self._dtype = get_default_dtype()

        self._alpha = alpha.to(device=self._device, dtype=self._dtype)
        self._phi = phi.to(device=self._device, dtype=self._dtype)
        self._U = U.to(device=self._device, dtype=self._dtype)
        self._C_alpha = C_alpha
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._use_mules = use_mules
        self._mules_iterations = mules_iterations

        # Lazy-initialise MULES limiter
        self._mules = None

    @property
    def alpha(self) -> torch.Tensor:
        """Current volume fraction field ``(n_cells,)``."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: torch.Tensor) -> None:
        self._alpha = value.to(device=self._device, dtype=self._dtype)

    @property
    def phi(self) -> torch.Tensor:
        """Face flux ``(n_faces,)``."""
        return self._phi

    @phi.setter
    def phi(self, value: torch.Tensor) -> None:
        self._phi = value.to(device=self._device, dtype=self._dtype)

    @property
    def U(self) -> torch.Tensor:
        """Cell-centre velocity ``(n_cells, 3)``."""
        return self._U

    @U.setter
    def U(self, value: torch.Tensor) -> None:
        self._U = value.to(device=self._device, dtype=self._dtype)

    def advance(self, delta_t: float) -> torch.Tensor:
        """Advance volume fraction by one time step.

        Uses a semi-implicit Euler scheme with interface compression
        and optional MULES limiting for boundedness.

        Args:
            delta_t: Time step size (s).

        Returns:
            Updated volume fraction ``(n_cells,)``.
        """
        mesh = self._mesh
        alpha = self._alpha
        phi = self._phi
        C_alpha = self._C_alpha

        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour
        cell_volumes = mesh.cell_volumes

        int_owner = owner[:n_internal]
        int_neigh = neighbour

        # ----------------------------------------------------------
        # Step 1: Interpolate alpha to faces (upwind)
        # ----------------------------------------------------------
        flux = phi[:n_internal]
        is_positive = flux >= 0.0
        alpha_P = gather(alpha, int_owner)
        alpha_N = gather(alpha, int_neigh)
        alpha_face = torch.where(is_positive, alpha_P, alpha_N)

        # ----------------------------------------------------------
        # Step 2: Compute compression flux (OpenFOAM-style)
        # ----------------------------------------------------------
        # The compression velocity is directed from the cell with
        # lower alpha to higher alpha, i.e. normal to the interface.
        # phi_c = C_alpha * max(|phi|) * (alpha_P - alpha_N)
        # This is consistent with OpenFOAM's interfaceCompression scheme
        # where the compression flux is proportional to the alpha
        # difference and the maximum face flux magnitude.
        phi_max = flux.abs().max().clamp(min=1e-30)
        delta_alpha = alpha_P - alpha_N
        compression_flux = C_alpha * phi_max * delta_alpha

        # ----------------------------------------------------------
        # Step 3: Compute total alpha face flux
        # ----------------------------------------------------------
        alpha_flux = flux * alpha_face + compression_flux

        # ----------------------------------------------------------
        # Step 4: Apply MULES limiting (if enabled)
        # ----------------------------------------------------------
        if self._use_mules:
            if self._mules is None:
                from pyfoam.multiphase.mules import MULESLimiter
                self._mules = MULESLimiter(
                    mesh, n_iterations=self._mules_iterations
                )

            alpha_new = self._mules.limit(
                alpha, phi, alpha_flux, delta_t,
                alpha_min=self._alpha_min,
                alpha_max=self._alpha_max,
            )
        else:
            # No MULES: direct forward Euler + clamp
            div_alpha = torch.zeros(
                n_cells, dtype=self._dtype, device=self._device
            )
            div_alpha = div_alpha + scatter_add(
                alpha_flux, int_owner, n_cells
            )
            div_alpha = div_alpha + scatter_add(
                -alpha_flux, int_neigh, n_cells
            )

            # Boundary faces
            if mesh.n_faces > n_internal:
                bnd_flux = phi[n_internal:] * gather(
                    alpha, owner[n_internal:]
                )
                div_alpha = div_alpha + scatter_add(
                    bnd_flux, owner[n_internal:], n_cells
                )

            V = cell_volumes.clamp(min=1e-30)
            alpha_new = alpha - delta_t * div_alpha / V
            alpha_new = alpha_new.clamp(
                min=self._alpha_min, max=self._alpha_max
            )

        self._alpha = alpha_new
        return alpha_new

    def compute_interface_normal(self) -> torch.Tensor:
        """Compute the interface normal from volume fraction gradient.

        Returns:
            ``(n_cells, 3)`` — unit normal vector pointing from fluid 1
            to fluid 2.  Zero where there is no interface.
        """
        mesh = self._mesh
        alpha = self._alpha
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        # Gradient using Gauss theorem
        face_areas = mesh.face_areas[:n_internal]
        alpha_P = gather(alpha, int_owner)
        alpha_N = gather(alpha, int_neigh)
        alpha_mid = 0.5 * (alpha_P + alpha_N)

        face_contrib = alpha_mid.unsqueeze(-1) * face_areas

        grad_alpha = torch.zeros(
            n_cells, 3, dtype=self._dtype, device=self._device
        )
        grad_alpha.index_add_(0, int_owner, face_contrib)
        grad_alpha.index_add_(0, int_neigh, -face_contrib)

        V = mesh.cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        grad_alpha = grad_alpha / V

        # Normalise
        grad_mag = grad_alpha.norm(dim=1, keepdim=True).clamp(min=1e-30)
        n_hat = grad_alpha / grad_mag

        # Zero where there's no interface (alpha far from 0.5)
        has_interface = (alpha > 0.01) & (alpha < 0.99)
        n_hat = n_hat * has_interface.unsqueeze(-1).float()

        return n_hat

    def compute_curvature(self) -> torch.Tensor:
        """Compute interface curvature from the normal field.

        κ = -∇·n̂

        Returns:
            ``(n_cells,)`` — interface curvature.
        """
        mesh = self._mesh
        n_hat = self.compute_interface_normal()
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        face_areas = mesh.face_areas[:n_internal]

        n_P = n_hat[int_owner]
        n_N = n_hat[int_neigh]
        n_face = 0.5 * (n_P + n_N)

        # Flux of n through faces
        n_flux = (n_face * face_areas).sum(dim=1)

        div_n = torch.zeros(n_cells, dtype=self._dtype, device=self._device)
        div_n = div_n + scatter_add(n_flux, int_owner, n_cells)
        div_n = div_n + scatter_add(-n_flux, int_neigh, n_cells)

        V = mesh.cell_volumes.clamp(min=1e-30)
        return -div_n / V

    def __repr__(self) -> str:
        return (
            f"VOFAdvection(n_cells={self._mesh.n_cells}, "
            f"C_alpha={self._C_alpha}, "
            f"alpha_range=[{self._alpha.min():.3f}, "
            f"{self._alpha.max():.3f}])"
        )
