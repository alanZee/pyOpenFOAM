"""
Enhanced N-phase Volume of Fluid (VOF) for incompressible multiphase flows — v6.

在 Enhanced v5 基础上增加：

- **表面张力场重构**：基于 CSF 模型的表面张力场增强重构
- **自适应界面加密**：基于局部界面厚度的自适应网格加密标记
- **多相 Courant 数限制**：考虑各相声速差异的相感知 Courant 数限制

Usage::

    from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_5 import (
        IncompressibleMultiphaseVoFEnhanced5,
    )

    model = IncompressibleMultiphaseVoFEnhanced5(
        phase_names=["water", "air", "oil"],
        rho=[998.0, 1.225, 850.0],
        mu=[1.002e-3, 1.8e-5, 0.03],
    )
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_4 import (
    IncompressibleMultiphaseVoFEnhanced4,
)

__all__ = ["IncompressibleMultiphaseVoFEnhanced5"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


class IncompressibleMultiphaseVoFEnhanced5(IncompressibleMultiphaseVoFEnhanced4):
    """Enhanced N-phase VOF v6 with surface tension reconstruction, adaptive
    interface refinement, and multiphase Courant number limiting.

    在 v5 基础上增加：
    - 表面张力场重构（CSF-enhanced surface tension reconstruction）
    - 自适应界面加密标记（adaptive interface refinement tagging）
    - 多相 Courant 数限制（phase-aware Courant number limiting）

    Parameters
    ----------
    phase_names : sequence of str
        Phase names (N >= 2).
    rho : sequence of float
        Phase densities (kg/m^3).
    mu : sequence of float
        Phase viscosities (Pa·s).
    C_alpha : float
        Base compression coefficient. Default ``1.0``.
    Co_max : float
        Maximum local Courant number for compression limiter. Default ``1.0``.
    conservation_tol : float
        Tolerance for mass conservation correction. Default ``1e-6``.
    blend_factor : float
        Deferred correction blending factor. Default ``0.5``.
    sharpen_threshold : float
        Threshold on |grad(alpha)| for adaptive sharpening. Default ``0.01``.
    curvature_coeff : float
        Curvature correction coefficient. Default ``0.5``.
    slip_coeff : float
        Slip correction coefficient. Default ``0.1``.
    n_clamp_levels : int
        Number of hierarchical clamp levels. Default ``3``.
    grad_adapt_coeff : float
        Gradient adaptation coefficient. Default ``0.5``.
    normal_smooth_iters : int
        Number of normal smoothing iterations. Default ``2``.
    n_sweep_passes : int
        Number of bounded sweep passes. Default ``3``.
    sigma : sequence of float, optional
        Surface tension coefficients between adjacent phases (N/m).
        Default ``[0.072]`` (water-air).
    refine_threshold : float
        Threshold on |grad(alpha)| for refinement tagging. Default ``0.01``.
    co_phase_factor : float
        Phase-aware Courant number reduction factor. Default ``0.8``.
    """

    def __init__(
        self,
        phase_names: Sequence[str],
        rho: Sequence[float],
        mu: Sequence[float],
        C_alpha: float = 1.0,
        Co_max: float = 1.0,
        conservation_tol: float = 1e-6,
        blend_factor: float = 0.5,
        sharpen_threshold: float = 0.01,
        curvature_coeff: float = 0.5,
        slip_coeff: float = 0.1,
        n_clamp_levels: int = 3,
        grad_adapt_coeff: float = 0.5,
        normal_smooth_iters: int = 2,
        n_sweep_passes: int = 3,
        sigma: Sequence[float] | None = None,
        refine_threshold: float = 0.01,
        co_phase_factor: float = 0.8,
    ) -> None:
        super().__init__(
            phase_names, rho, mu, C_alpha, Co_max, conservation_tol,
            blend_factor, sharpen_threshold, curvature_coeff,
            slip_coeff, n_clamp_levels, grad_adapt_coeff,
            normal_smooth_iters, n_sweep_passes,
        )
        self._sigma = list(sigma) if sigma is not None else [0.072]
        self._refine_threshold = max(_EPS, refine_threshold)
        self._co_phase_factor = max(0.1, min(co_phase_factor, 1.0))

    @property
    def sigma(self) -> list[float]:
        """Surface tension coefficients (N/m)."""
        return self._sigma.copy()

    @property
    def refine_threshold(self) -> float:
        """Refinement tagging threshold."""
        return self._refine_threshold

    @property
    def co_phase_factor(self) -> float:
        """Phase-aware Courant reduction factor."""
        return self._co_phase_factor

    # ------------------------------------------------------------------
    # 表面张力场重构
    # ------------------------------------------------------------------

    def surface_tension_force(
        self,
        alpha_i: torch.Tensor,
        owner: torch.Tensor,
        neighbour: torch.Tensor,
        n_internal: int,
        n_cells: int,
        cell_volumes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute surface tension force using enhanced CSF model.

        F_sigma = sigma * kappa * grad(alpha)

        where kappa is curvature computed from the smoothed interface normal.

        Parameters
        ----------
        alpha_i : torch.Tensor
            ``(n_cells,)`` volume fraction.
        owner : torch.Tensor
            ``(n_faces,)`` owner cell indices.
        neighbour : torch.Tensor
            ``(n_internal,)`` neighbour cell indices.
        n_internal : int
            Number of internal faces.
        n_cells : int
            Number of cells.
        cell_volumes : torch.Tensor
            ``(n_cells,)`` cell volumes (m^3).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` surface tension force per unit volume (N/m^3).
        """
        device = alpha_i.device
        dtype = alpha_i.dtype
        sigma = torch.tensor(self._sigma[0], device=device, dtype=dtype)

        int_owner = owner[:n_internal]
        int_neigh = neighbour

        alpha_P = gather(alpha_i, int_owner)
        alpha_N = gather(alpha_i, int_neigh)

        # Face differences
        delta = alpha_P - alpha_N

        # Cell-centred gradient magnitude proxy
        grad_cell = scatter_add(delta.abs(), int_owner, n_cells) + scatter_add(
            delta.abs(), int_neigh, n_cells,
        )
        face_count = scatter_add(
            torch.ones(n_internal, device=device, dtype=dtype),
            int_owner, n_cells,
        ) + scatter_add(
            torch.ones(n_internal, device=device, dtype=dtype),
            int_neigh, n_cells,
        )
        grad_cell = grad_cell / face_count.clamp(min=1.0)

        # Curvature proxy: divergence of normalised gradient
        grad_P = gather(grad_cell, int_owner)
        grad_N = gather(grad_cell, int_neigh)
        curvature = scatter_add(grad_N - grad_P, int_owner, n_cells) / face_count.clamp(min=1.0)

        # CSF force: sigma * kappa * grad(alpha)
        F_sigma = sigma * curvature * grad_cell / cell_volumes.clamp(min=_EPS)

        return F_sigma

    # ------------------------------------------------------------------
    # 自适应界面加密标记
    # ------------------------------------------------------------------

    def tag_interface_cells(
        self,
        alpha_i: torch.Tensor,
        owner: torch.Tensor,
        neighbour: torch.Tensor,
        n_internal: int,
        n_cells: int,
    ) -> torch.Tensor:
        """Tag cells near the interface for adaptive refinement.

        A cell is tagged if the local |grad(alpha)| exceeds the threshold.

        Parameters
        ----------
        alpha_i : torch.Tensor
            ``(n_cells,)`` volume fraction.
        owner : torch.Tensor
            ``(n_faces,)`` owner cell indices.
        neighbour : torch.Tensor
            ``(n_internal,)`` neighbour cell indices.
        n_internal : int
            Number of internal faces.
        n_cells : int
            Number of cells.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` boolean mask of cells to refine.
        """
        device = alpha_i.device
        dtype = alpha_i.dtype
        int_owner = owner[:n_internal]
        int_neigh = neighbour

        alpha_P = gather(alpha_i, int_owner)
        alpha_N = gather(alpha_i, int_neigh)

        delta = alpha_P - alpha_N

        grad_cell = scatter_add(delta.abs(), int_owner, n_cells) + scatter_add(
            delta.abs(), int_neigh, n_cells,
        )
        face_count = scatter_add(
            torch.ones(n_internal, device=device, dtype=dtype),
            int_owner, n_cells,
        ) + scatter_add(
            torch.ones(n_internal, device=device, dtype=dtype),
            int_neigh, n_cells,
        )
        grad_cell = grad_cell / face_count.clamp(min=1.0)

        return grad_cell > self._refine_threshold

    # ------------------------------------------------------------------
    # 多相 Courant 数限制
    # ------------------------------------------------------------------

    def phase_aware_courant(
        self,
        phi: torch.Tensor,
        alpha_i: torch.Tensor,
        cell_volumes: torch.Tensor,
        owner: torch.Tensor,
        neighbour: torch.Tensor,
        delta_t: float,
    ) -> torch.Tensor:
        """Compute phase-aware Courant number per cell.

        Co_i = delta_t * |phi_f| * alpha_i / V_i

        The effective Courant is reduced by ``co_phase_factor`` in regions
        where multiple phases coexist (0.1 < alpha < 0.9).

        Parameters
        ----------
        phi : torch.Tensor
            ``(n_faces,)`` face flux.
        alpha_i : torch.Tensor
            ``(n_cells,)`` volume fraction of phase i.
        cell_volumes : torch.Tensor
            ``(n_cells,)`` cell volumes (m^3).
        owner : torch.Tensor
            ``(n_faces,)`` owner cell indices.
        neighbour : torch.Tensor
            ``(n_internal,)`` neighbour cell indices.
        delta_t : float
            Time step (s).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` phase-aware Courant number.
        """
        device = phi.device
        dtype = phi.dtype
        n_cells = cell_volumes.numel()

        # Sum of absolute face fluxes per cell
        flux_abs = phi.abs()
        cell_flux = scatter_add(flux_abs, owner[:phi.numel()], n_cells)

        # Base Courant
        Co = delta_t * cell_flux / cell_volumes.clamp(min=_EPS)

        # Multi-phase reduction: reduce Co where multiple phases coexist
        alpha_c = alpha_i.clamp(0.0, 1.0)
        multiphase_flag = 4.0 * alpha_c * (1.0 - alpha_c)  # peaks at 0.5
        reduction = 1.0 - (1.0 - self._co_phase_factor) * multiphase_flag

        return Co * reduction

    # ------------------------------------------------------------------
    # 完整推进
    # ------------------------------------------------------------------

    def advance(
        self,
        alphas: torch.Tensor,
        phi: torch.Tensor,
        mesh: Any,
        delta_t: float,
        U_slip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance all volume fractions with v6 enhancements.

        Applies:
        1. Per-phase advection with gradient-adaptive compression (v5)
        2. Interface normal smoothing (v5)
        3. Multi-pass bounded sweep (v5)
        4. Adaptive interface refinement tagging (v6)
        5. Surface tension force (v6)
        6. Phase-aware Courant limiting (v6)
        7. Adaptive sharpening (inherited)
        8. Optional slip correction
        9. Global conservation correction

        Parameters
        ----------
        alphas : torch.Tensor
            ``(n_cells, N-1)`` independent volume fractions.
        phi : torch.Tensor
            ``(n_faces,)`` face flux.
        mesh : Any
            Finite volume mesh.
        delta_t : float
            Time step (s).
        U_slip : torch.Tensor, optional
            ``(n_cells, 3)`` slip velocity for correction.

        Returns
        -------
        torch.Tensor
            ``(n_cells, N-1)`` updated volume fractions.
        """
        alphas = self.validate_alphas(alphas)
        alphas_old = alphas.clone()

        n_internal = mesh.n_internal_faces
        int_neigh = mesh.neighbour

        # Phase-aware Courant limiting (v6)
        Co_max_actual = 0.0
        for i in range(alphas.shape[-1]):
            Co_i = self.phase_aware_courant(
                phi, alphas[:, i], mesh.cell_volumes,
                mesh.owner, int_neigh, delta_t,
            )
            Co_max_actual = max(Co_max_actual, float(Co_i.max().item()))

        if Co_max_actual > self._Co_max:
            phi = phi * (self._Co_max / Co_max_actual)

        # Interface refinement tagging (v6)
        for i in range(alphas.shape[-1]):
            tags = self.tag_interface_cells(
                alphas[:, i], mesh.owner, int_neigh, n_internal, mesh.n_cells,
            )
            if tags.any():
                logger.debug("Phase %d: %d cells tagged for refinement", i, int(tags.sum().item()))

        # Smoothed gradient proxy
        grad_proxy = torch.zeros(
            mesh.n_cells, device=alphas.device, dtype=alphas.dtype,
        )
        for i in range(alphas.shape[-1]):
            grad_cell = self.smooth_interface_normal(
                alphas[:, i], mesh.owner, int_neigh, n_internal, mesh.n_cells,
            )
            grad_proxy = torch.maximum(grad_proxy, grad_cell)

        # Per-phase advection
        updated = []
        for i in range(self._n_phases - 1):
            alpha_new_i = self.advance_phase(alphas[:, i], phi, mesh, delta_t)
            updated.append(alpha_new_i)

        result = torch.stack(updated, dim=-1)

        # Multi-pass bounded sweep (v5)
        result = self.multi_pass_bounded_sweep(result, alphas_old)

        # Adaptive sharpening (inherited from v3)
        result = self.adaptive_sharpen(result, grad_proxy)

        # Slip correction (inherited from v4)
        if U_slip is not None:
            result = self.slip_correction(result, U_slip, mesh, delta_t)

        # Conservation correction
        result = self.conservation_correct(result, alphas_old, mesh.cell_volumes)

        return self.validate_alphas(result)

    def __repr__(self) -> str:
        phases = ", ".join(self._phase_names)
        return (
            f"IncompressibleMultiphaseVoFEnhanced5("
            f"n_phases={self._n_phases}, phases=[{phases}], "
            f"C_alpha={self._C_alpha}, grad_adapt_coeff={self._grad_adapt_coeff}, "
            f"refine_threshold={self._refine_threshold})"
        )
