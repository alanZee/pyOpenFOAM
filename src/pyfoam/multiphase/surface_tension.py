"""
Continuum Surface Force (CSF) model for surface tension.

Implements the CSF model of Brackbill et al. (1992) for representing
surface tension as a volumetric force at fluid interfaces.

The surface tension force is:

    F_st = σ * κ * ∇α

where:
- σ is the surface tension coefficient
- κ = -∇·n̂ is the interface curvature
- n̂ = ∇α/|∇α| is the interface unit normal
- α is the volume fraction

Includes optional Laplacian smoothing of the alpha field to reduce
spurious currents (Lafaurie et al. 1994).

Usage::

    from pyfoam.multiphase.surface_tension import SurfaceTensionModel

    st = SurfaceTensionModel(sigma=0.07, mesh=mesh, n_smooth=1)
    F_st = st.compute_force(alpha)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["SurfaceTensionModel"]

logger = logging.getLogger(__name__)


class SurfaceTensionModel:
    """Continuum Surface Force (CSF) model for surface tension.

    Based on Brackbill et al. (1992), J. Comput. Phys. 100, 335-354.

    The model converts the surface tension force at an interface into
    a volume force that can be added to the momentum equation:

        F_st = σ * κ * ∇α

    where κ = -∇·(∇α/|∇α|) is the interface curvature.

    Parameters
    ----------
    sigma : float
        Surface tension coefficient (N/m).
    mesh : Any
        The finite volume mesh.
    n_smooth : int
        Number of Laplacian smoothing passes for alpha before computing
        normals.  Reduces spurious currents.  Default 1.

    Examples::

        st = SurfaceTensionModel(sigma=0.07, mesh=mesh)
        F_st = st.compute_force(alpha)  # (n_cells, 3)
    """

    def __init__(
        self,
        sigma: float,
        mesh: Any,
        n_smooth: int = 1,
    ) -> None:
        self._sigma = sigma
        self._mesh = mesh
        self._n_smooth = n_smooth
        self._device = get_device()
        self._dtype = get_default_dtype()

    @property
    def sigma(self) -> float:
        """Surface tension coefficient."""
        return self._sigma

    def compute_force(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute the CSF surface tension force per unit volume.

        Args:
            alpha: Volume fraction ``(n_cells,)``.

        Returns:
            ``(n_cells, 3)`` — surface tension force per unit volume.
        """
        if self._sigma == 0.0:
            return torch.zeros(
                self._mesh.n_cells, 3,
                dtype=self._dtype, device=self._device,
            )

        mesh = self._mesh

        # Step 1: Smooth alpha to reduce noise
        alpha_s = self._smooth_alpha(alpha)

        # Step 2: Compute gradient of smoothed alpha
        grad_alpha = self._compute_gradient(alpha_s)

        # Step 3: Compute interface normal n̂ = ∇α/|∇α|
        grad_mag = grad_alpha.norm(dim=1, keepdim=True).clamp(min=1e-30)
        n_hat = grad_alpha / grad_mag

        # Step 4: Compute curvature κ = -∇·n̂
        kappa = -self._compute_divergence_vector(n_hat)

        # Step 5: Apply interface mask (only near interface)
        has_interface = (alpha > 0.01) & (alpha < 0.99)
        kappa = kappa * has_interface.float()

        # Step 6: Surface tension force F = σ * κ * ∇α
        # Use original (unsmoothed) alpha gradient for the force
        grad_alpha_orig = self._compute_gradient(alpha)
        F_st = self._sigma * kappa.unsqueeze(-1) * grad_alpha_orig

        return F_st

    def compute_curvature(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute the interface curvature field.

        Args:
            alpha: Volume fraction ``(n_cells,)``.

        Returns:
            ``(n_cells,)`` — interface curvature κ.
        """
        alpha_s = self._smooth_alpha(alpha)
        grad_alpha = self._compute_gradient(alpha_s)
        grad_mag = grad_alpha.norm(dim=1, keepdim=True).clamp(min=1e-30)
        n_hat = grad_alpha / grad_mag

        kappa = -self._compute_divergence_vector(n_hat)

        has_interface = (alpha > 0.01) & (alpha < 0.99)
        kappa = kappa * has_interface.float()
        return kappa

    def _smooth_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        """Laplacian smoothing of alpha field.

        Applies n_smooth passes of:
            alpha_s[i] = alpha[i] + (1/6) * Σ_j (alpha[j] - alpha[i])
        (3D Jacobi smoothing factor; in 2D it is 1/4).
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        alpha_s = alpha.clone()

        for _ in range(self._n_smooth):
            alpha_P = gather(alpha_s, int_owner)
            alpha_N = gather(alpha_s, int_neigh)

            # Sum of (alpha_neigh - alpha_cell) for each cell
            delta_sum = torch.zeros(
                n_cells, dtype=self._dtype, device=self._device
            )
            delta_sum = delta_sum + scatter_add(
                alpha_N - alpha_P, int_owner, n_cells
            )
            delta_sum = delta_sum + scatter_add(
                alpha_P - alpha_N, int_neigh, n_cells
            )

            # Count neighbours per cell
            n_neigh = torch.zeros(
                n_cells, dtype=self._dtype, device=self._device
            )
            ones = torch.ones(n_internal, dtype=self._dtype, device=self._device)
            n_neigh = n_neigh + scatter_add(ones, int_owner, n_cells)
            n_neigh = n_neigh + scatter_add(ones, int_neigh, n_cells)
            n_neigh = n_neigh.clamp(min=1.0)

            alpha_s = alpha_s + 0.25 * delta_sum / n_neigh

        return alpha_s

    def _compute_gradient(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute gradient of scalar field using Gauss theorem."""
        mesh = self._mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        face_areas = mesh.face_areas[:n_internal]
        w = mesh.face_weights[:n_internal]

        phi_P = gather(phi, int_owner)
        phi_N = gather(phi, int_neigh)
        phi_face = w * phi_P + (1.0 - w) * phi_N

        face_contrib = phi_face.unsqueeze(-1) * face_areas

        grad = torch.zeros(
            n_cells, 3, dtype=self._dtype, device=self._device
        )
        grad.index_add_(0, int_owner, face_contrib)
        grad.index_add_(0, int_neigh, -face_contrib)

        V = mesh.cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        return grad / V

    def _compute_divergence_vector(self, vec: torch.Tensor) -> torch.Tensor:
        """Compute divergence of a vector field using Gauss theorem."""
        mesh = self._mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        face_areas = mesh.face_areas[:n_internal]

        vec_P = vec[int_owner]
        vec_N = vec[int_neigh]
        vec_face = 0.5 * (vec_P + vec_N)

        flux = (vec_face * face_areas).sum(dim=1)

        div = torch.zeros(
            n_cells, dtype=self._dtype, device=self._device
        )
        div = div + scatter_add(flux, int_owner, n_cells)
        div = div + scatter_add(-flux, int_neigh, n_cells)

        V = mesh.cell_volumes.clamp(min=1e-30)
        return div / V

    def __repr__(self) -> str:
        return (
            f"SurfaceTensionModel(sigma={self._sigma}, "
            f"n_smooth={self._n_smooth}, "
            f"n_cells={self._mesh.n_cells})"
        )
