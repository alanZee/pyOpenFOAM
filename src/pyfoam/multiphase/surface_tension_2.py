"""
Enhanced Continuum Surface Force (CSF) model for surface tension.

Extends the basic CSF model of Brackbill et al. (1992) with curvature
smoothing to reduce spurious currents near the interface.  The enhanced
model applies Laplacian smoothing to the curvature field itself, in
addition to the standard alpha smoothing.

The surface tension force is:

    F_st = sigma * kappa_smoothed * grad(alpha)

where kappa_smoothed is obtained by applying Laplacian smoothing to
the raw curvature field.  This two-stage smoothing (alpha + curvature)
produces more stable results than single-stage smoothing alone.

Registered in the RTS selection table as ``"CSF"``.

References
----------
- Brackbill, J.U., Kothe, D.B., Zemach, C. (1992).
  A continuum method for modeling surface tension.
  *J. Comput. Phys.*, 100(2), 335-354.
- Lafaurie, B., Nardone, C., Scardovelli, R., Zaleski, S., Zanetti, G.
  (1994). Modelling merging and fragmentation in multiphase flows with
  SURFER. *J. Comput. Phys.*, 113(1), 134-147.

Usage::

    from pyfoam.multiphase.surface_tension_2 import CSFSurfaceTension

    model = CSFSurfaceTension(sigma=0.07, mesh=mesh, n_alpha_smooth=2, n_curvature_smooth=3)
    F_st = model.compute_force(alpha)
    kappa = model.compute_curvature(alpha)
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["CSFSurfaceTension"]

logger = logging.getLogger(__name__)


class CSFSurfaceTension:
    """Enhanced Continuum Surface Force model with curvature smoothing.

    Two-stage smoothing approach:

    1. Smooth the volume fraction alpha (reduces gradient noise).
    2. Compute curvature from the smoothed alpha.
    3. Smooth the curvature field (reduces spurious currents).

    The smoothed curvature is then used in the CSF force formula:
    F_st = sigma * kappa_smoothed * grad(alpha).

    Parameters
    ----------
    sigma : float
        Surface tension coefficient (N/m).
    mesh : Any
        The finite volume mesh (must have ``n_cells``, ``owner``,
        ``neighbour``, ``face_areas``, ``face_weights``, ``cell_volumes``,
        ``n_internal_faces``).
    n_alpha_smooth : int
        Number of Laplacian smoothing passes on the alpha field.
        Default 1.
    n_curvature_smooth : int
        Number of Laplacian smoothing passes on the curvature field.
        Default 2.
    interface_threshold : float
        Band around alpha=0.5 where the interface is active.
        Cells with ``alpha_threshold < alpha < 1 - alpha_threshold``
        are considered interfacial.  Default 0.01.

    Examples::

        model = CSFSurfaceTension(sigma=0.07, mesh=mesh, n_curvature_smooth=3)
        F_st = model.compute_force(alpha)   # (n_cells, 3)
        kappa = model.compute_curvature(alpha)  # (n_cells,)
    """

    def __init__(
        self,
        sigma: float,
        mesh: Any,
        n_alpha_smooth: int = 1,
        n_curvature_smooth: int = 2,
        interface_threshold: float = 0.01,
    ) -> None:
        self._sigma = sigma
        self._mesh = mesh
        self._n_alpha_smooth = n_alpha_smooth
        self._n_curvature_smooth = n_curvature_smooth
        self._interface_threshold = interface_threshold
        self._device = get_device()
        self._dtype = get_default_dtype()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sigma(self) -> float:
        """Surface tension coefficient."""
        return self._sigma

    @sigma.setter
    def sigma(self, value: float) -> None:
        self._sigma = value

    @property
    def n_alpha_smooth(self) -> int:
        """Number of alpha smoothing passes."""
        return self._n_alpha_smooth

    @property
    def n_curvature_smooth(self) -> int:
        """Number of curvature smoothing passes."""
        return self._n_curvature_smooth

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_force(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute the enhanced CSF surface tension force per unit volume.

        Uses two-stage smoothing: alpha smoothing followed by curvature
        smoothing.

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

        # Stage 1: Smooth alpha
        alpha_s = self._smooth_scalar(alpha, self._n_alpha_smooth)

        # Stage 2: Compute gradient and curvature from smoothed alpha
        grad_alpha_s = self._compute_gradient(alpha_s)
        kappa_raw = self._compute_curvature_from_grad(grad_alpha_s)

        # Stage 3: Smooth curvature
        kappa_s = self._smooth_scalar(kappa_raw, self._n_curvature_smooth)

        # Stage 4: Apply interface mask
        has_interface = (
            (alpha > self._interface_threshold)
            & (alpha < 1.0 - self._interface_threshold)
        )
        kappa_s = kappa_s * has_interface.to(dtype=self._dtype)

        # Stage 5: Force = sigma * kappa_smoothed * grad(alpha_original)
        grad_alpha_orig = self._compute_gradient(alpha)
        F_st = self._sigma * kappa_s.unsqueeze(-1) * grad_alpha_orig

        return F_st

    def compute_curvature(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute the smoothed interface curvature field.

        Applies the same two-stage smoothing as :meth:`compute_force`.

        Args:
            alpha: Volume fraction ``(n_cells,)``.

        Returns:
            ``(n_cells,)`` — smoothed interface curvature.
        """
        alpha_s = self._smooth_scalar(alpha, self._n_alpha_smooth)
        grad_alpha_s = self._compute_gradient(alpha_s)
        kappa_raw = self._compute_curvature_from_grad(grad_alpha_s)
        kappa_s = self._smooth_scalar(kappa_raw, self._n_curvature_smooth)

        has_interface = (
            (alpha > self._interface_threshold)
            & (alpha < 1.0 - self._interface_threshold)
        )
        return kappa_s * has_interface.to(dtype=self._dtype)

    # ------------------------------------------------------------------
    # Internal: smoothing
    # ------------------------------------------------------------------

    def _smooth_scalar(
        self, phi: torch.Tensor, n_passes: int,
    ) -> torch.Tensor:
        """Laplacian smoothing of a scalar field.

        Applies *n_passes* Jacobi smoothing iterations:

            phi_s[i] = phi[i] + (1/4) * sum_j (phi[j] - phi[i])

        where the sum runs over neighbouring cells connected by
        internal faces.

        Args:
            phi: Scalar field ``(n_cells,)``.
            n_passes: Number of smoothing iterations.

        Returns:
            Smoothed scalar field ``(n_cells,)``.
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        phi_s = phi.clone()

        for _ in range(n_passes):
            phi_P = gather(phi_s, int_owner)
            phi_N = gather(phi_s, int_neigh)

            # Sum of (neighbour - cell) contributions
            delta_sum = torch.zeros(
                n_cells, dtype=self._dtype, device=self._device,
            )
            delta_sum = delta_sum + scatter_add(
                phi_N - phi_P, int_owner, n_cells,
            )
            delta_sum = delta_sum + scatter_add(
                phi_P - phi_N, int_neigh, n_cells,
            )

            # Count neighbours per cell
            n_neigh = torch.zeros(
                n_cells, dtype=self._dtype, device=self._device,
            )
            ones = torch.ones(
                n_internal, dtype=self._dtype, device=self._device,
            )
            n_neigh = n_neigh + scatter_add(ones, int_owner, n_cells)
            n_neigh = n_neigh + scatter_add(ones, int_neigh, n_cells)
            n_neigh = n_neigh.clamp(min=1.0)

            phi_s = phi_s + 0.25 * delta_sum / n_neigh

        return phi_s

    # ------------------------------------------------------------------
    # Internal: gradient and curvature
    # ------------------------------------------------------------------

    def _compute_gradient(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute gradient of a scalar field using Gauss theorem.

        Args:
            phi: Scalar field ``(n_cells,)``.

        Returns:
            ``(n_cells, 3)`` — gradient vector per cell.
        """
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
            n_cells, 3, dtype=self._dtype, device=self._device,
        )
        grad.index_add_(0, int_owner, face_contrib)
        grad.index_add_(0, int_neigh, -face_contrib)

        V = mesh.cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        return grad / V

    def _compute_curvature_from_grad(
        self, grad_alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Compute curvature from a precomputed gradient field.

        kappa = -div(n_hat), where n_hat = grad_alpha / |grad_alpha|.

        Args:
            grad_alpha: Gradient of alpha ``(n_cells, 3)``.

        Returns:
            ``(n_cells,)`` — curvature.
        """
        # Interface normal n_hat = grad_alpha / |grad_alpha|
        grad_mag = grad_alpha.norm(dim=1, keepdim=True).clamp(min=1e-30)
        n_hat = grad_alpha / grad_mag

        # Curvature = -div(n_hat)
        return -self._compute_divergence_vector(n_hat)

    def _compute_divergence_vector(self, vec: torch.Tensor) -> torch.Tensor:
        """Compute divergence of a vector field using Gauss theorem.

        Args:
            vec: Vector field ``(n_cells, 3)``.

        Returns:
            ``(n_cells,)`` — divergence.
        """
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
            n_cells, dtype=self._dtype, device=self._device,
        )
        div = div + scatter_add(flux, int_owner, n_cells)
        div = div + scatter_add(-flux, int_neigh, n_cells)

        V = mesh.cell_volumes.clamp(min=1e-30)
        return div / V

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CSFSurfaceTension(sigma={self._sigma}, "
            f"n_alpha_smooth={self._n_alpha_smooth}, "
            f"n_curvature_smooth={self._n_curvature_smooth}, "
            f"n_cells={self._mesh.n_cells})"
        )
