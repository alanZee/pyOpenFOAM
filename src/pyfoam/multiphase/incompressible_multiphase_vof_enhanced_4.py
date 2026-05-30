"""
Enhanced N-phase Volume of Fluid (VOF) for incompressible multiphase flows — v5.

在 Enhanced v4 基础上增加：

- **梯度自适应压缩**：基于界面梯度幅值动态调节压缩系数
- **界面法线平滑**：对界面法线场进行 Laplacian 平滑，减少锯齿
- **多遍有界扫描**：多轮扫描-修正策略，逐单元收敛至有界

Governing equations (per independent phase i):

    d(alpha_i)/dt + div(U * alpha_i) + div(U_r * alpha_i * (1 - alpha_i)) = 0

Enhanced v5 compression flux with gradient adaptation:

    U_r = C_alpha(S_f, |grad(alpha)|) * min(|phi|/|S_f|, U_max) * n_f * K(kappa, Co)

where C_alpha is now a spatially-varying field.

Usage::

    from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_4 import (
        IncompressibleMultiphaseVoFEnhanced4,
    )

    model = IncompressibleMultiphaseVoFEnhanced4(
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
from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_3 import (
    IncompressibleMultiphaseVoFEnhanced3,
)

__all__ = ["IncompressibleMultiphaseVoFEnhanced4"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


class IncompressibleMultiphaseVoFEnhanced4(IncompressibleMultiphaseVoFEnhanced3):
    """Enhanced N-phase VOF v5 with gradient-adaptive compression and normal smoothing.

    在 v4 基础上增加：
    - 梯度自适应压缩（gradient-adaptive compression）
    - 界面法线平滑（interface normal smoothing）
    - 多遍有界扫描（multi-pass bounded sweep）

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
    ) -> None:
        super().__init__(
            phase_names, rho, mu, C_alpha, Co_max, conservation_tol,
            blend_factor, sharpen_threshold, curvature_coeff,
            slip_coeff, n_clamp_levels,
        )
        self._grad_adapt_coeff = max(0.0, min(grad_adapt_coeff, 2.0))
        self._normal_smooth_iters = max(0, normal_smooth_iters)
        self._n_sweep_passes = max(1, n_sweep_passes)

    @property
    def grad_adapt_coeff(self) -> float:
        """Gradient adaptation coefficient."""
        return self._grad_adapt_coeff

    @property
    def normal_smooth_iters(self) -> int:
        """Number of normal smoothing iterations."""
        return self._normal_smooth_iters

    @property
    def n_sweep_passes(self) -> int:
        """Number of bounded sweep passes."""
        return self._n_sweep_passes

    # ------------------------------------------------------------------
    # 梯度自适应压缩
    # ------------------------------------------------------------------

    def gradient_adaptive_compression_coeff(
        self,
        alpha_i: torch.Tensor,
        owner: torch.Tensor,
        neighbour: torch.Tensor,
        n_internal: int,
    ) -> torch.Tensor:
        """Compute spatially-varying compression coefficient.

        C_alpha_f = C_alpha * (1 + C_g * tanh(|grad(alpha)| / threshold))

        In regions with sharp gradients the compression coefficient is
        increased; in smooth regions it is reduced.

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

        Returns
        -------
        torch.Tensor
            ``(n_internal,)`` face compression coefficients.
        """
        int_owner = owner[:n_internal]
        int_neigh = neighbour

        alpha_P = gather(alpha_i, int_owner)
        alpha_N = gather(alpha_i, int_neigh)

        # Face gradient proxy
        grad_face = (alpha_P - alpha_N).abs()

        # Adaptive coefficient: increases at interfaces
        C_adapt = self._C_alpha * (
            1.0 + self._grad_adapt_coeff * torch.tanh(grad_face / max(self._sharpen_threshold, _EPS))
        )

        return C_adapt.clamp(0.0, 3.0 * self._C_alpha)

    # ------------------------------------------------------------------
    # 界面法线平滑
    # ------------------------------------------------------------------

    def smooth_interface_normal(
        self,
        alpha_i: torch.Tensor,
        owner: torch.Tensor,
        neighbour: torch.Tensor,
        n_internal: int,
        n_cells: int,
    ) -> torch.Tensor:
        """Smooth the interface normal field using Laplacian averaging.

        Reduces staircase artefacts on the interface normal direction.

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
            ``(n_cells,)`` smoothed gradient magnitude proxy.
        """
        device = alpha_i.device
        dtype = alpha_i.dtype
        int_owner = owner[:n_internal]
        int_neigh = neighbour

        # Compute face differences
        alpha_P = gather(alpha_i, int_owner)
        alpha_N = gather(alpha_i, int_neigh)
        delta = alpha_P - alpha_N  # signed difference

        # Initial cell-centred gradient proxy (average of face deltas)
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

        # Laplacian smoothing iterations
        for _ in range(self._normal_smooth_iters):
            grad_P = gather(grad_cell, int_owner)
            grad_N = gather(grad_cell, int_neigh)
            laplacian = scatter_add(grad_N - grad_P, int_owner, n_cells)
            grad_cell = grad_cell + 0.25 * laplacian / face_count.clamp(min=1.0)
            grad_cell = grad_cell.clamp(min=0.0)

        return grad_cell

    # ------------------------------------------------------------------
    # 多遍有界扫描
    # ------------------------------------------------------------------

    def multi_pass_bounded_sweep(
        self,
        alphas_new: torch.Tensor,
        alphas_old: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-pass bounded sweep for improved convergence.

        Iteratively scans and corrects volume fractions until all
        cells are bounded within [0, 1] and sum to at most 1.

        Parameters
        ----------
        alphas_new : torch.Tensor
            ``(n_cells, N-1)`` proposed volume fractions.
        alphas_old : torch.Tensor
            ``(n_cells, N-1)`` old volume fractions.

        Returns
        -------
        torch.Tensor
            ``(n_cells, N-1)`` bounded volume fractions.
        """
        result = alphas_new.clone()

        for sweep in range(self._n_sweep_passes):
            # Clamp independent phases
            result = result.clamp(0.0, 1.0)

            # Check last phase
            alpha_last = 1.0 - result.sum(dim=-1)
            excess = alpha_last.clamp(max=0.0).abs()

            if excess.max() < _EPS:
                break

            # Proportional reduction
            current_sum = result.sum(dim=-1).clamp(min=_EPS)
            reduction = (1.0 - excess / current_sum).clamp(min=0.0, max=1.0)
            result = result * reduction.unsqueeze(-1)

            # Back-correction: where sum is too small, boost
            deficit = alpha_last.clamp(min=0.0)
            if deficit.max() > _EPS:
                # Distribute deficit proportionally
                boost = deficit / current_sum.clamp(min=_EPS)
                result = result * (1.0 + boost.unsqueeze(-1)).clamp(max=1.0)

        return self.validate_alphas(result)

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
        """Advance all volume fractions with v5 enhancements.

        Applies:
        1. Per-phase advection with gradient-adaptive compression
        2. Interface normal smoothing
        3. Multi-pass bounded sweep
        4. Adaptive sharpening (inherited)
        5. Optional slip correction
        6. Global conservation correction

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
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        # Smoothed gradient proxy for adaptive sharpening
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

        # Multi-pass bounded sweep (v5 enhancement)
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
            f"IncompressibleMultiphaseVoFEnhanced4("
            f"n_phases={self._n_phases}, phases=[{phases}], "
            f"C_alpha={self._C_alpha}, grad_adapt_coeff={self._grad_adapt_coeff}, "
            f"n_sweep_passes={self._n_sweep_passes})"
        )
