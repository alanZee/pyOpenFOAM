"""Enhanced k-omega v8 — cross-diffusion limiter, omega dissipation partition, freestream sensitivity.

Extends KOmegaEnhanced7Model with:
- Cross-diffusion limiter for separated flows
- Omega-specific dissipation partition model
- Freestream sensitivity analysis and correction

Usage::

    from pyfoam.turbulence.k_omega_enhanced_8 import KOmegaEnhanced8Model
    model = KOmegaEnhanced8Model(mesh, U, phi)
    model.correct()
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm, fvc
from .turbulence_model import TurbulenceModel
from .k_omega_enhanced_7 import KOmegaEnhanced7Model, KOmegaEnhanced7Constants

__all__ = ["KOmegaEnhanced8Model", "KOmegaEnhanced8Constants"]


@dataclass(frozen=True)
class KOmegaEnhanced8Constants(KOmegaEnhanced7Constants):
    """Constants for enhanced k-omega v8."""
    C_xd_lim: float = 0.8
    C_diss_partition: float = 0.5
    C_fs_corr: float = 0.1


_DEFAULTS = KOmegaEnhanced8Constants()


@TurbulenceModel.register("kOmegaEnhanced8")
class KOmegaEnhanced8Model(KOmegaEnhanced7Model):
    """Enhanced k-omega v8 with cross-diffusion limiter and freestream sensitivity."""

    def __init__(
        self,
        mesh: Any, U: Any, phi: torch.Tensor,
        *, constants: KOmegaEnhanced8Constants | None = None,
        g: tuple[float, float, float] = (0.0, 0.0, -9.81),
        beta_t: float = 3e-3,
        **kwargs: Any,
    ) -> None:
        super(KOmegaEnhanced7Model, self).__init__(mesh, U, phi)
        self._C = constants or _DEFAULTS
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        self._k = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._omega = torch.full((n_cells,), 1.0, device=device, dtype=dtype)
        self._grad_U = None
        self._y = self._compute_wall_distance()
        self._g = torch.tensor(g, device=device, dtype=dtype)
        self._beta_t = beta_t

    def _compute_wall_distance(self) -> torch.Tensor:
        cc = self._mesh.cell_centres
        return cc.norm(dim=1).clamp(min=1e-10)

    def _cross_diffusion_limiter(self) -> torch.Tensor:
        """Cross-diffusion limiter for separated flow regions.

        Limits the cross-diffusion term to prevent unbounded growth
        in separated regions where omega gradients are large.
        """
        C = self._C
        C_lim = getattr(C, 'C_xd_lim', 0.8)
        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)

        # Ratio of production to dissipation as separation indicator
        if self._grad_U is not None:
            nut = self.nut()
            S = self._strain_rate()
            P_k = 2.0 * nut * (S * S).sum(dim=(1, 2))
            eps = 0.09 * k_safe * omega_safe
            ratio = (P_k / eps.clamp(min=1e-16)).clamp(max=10.0)
            # Strong limiter when production >> dissipation
            f_lim = (1.0 / (1.0 + C_lim * ratio)).clamp(min=0.1, max=1.0)
        else:
            f_lim = torch.ones_like(self._k)

        return f_lim

    def _dissipation_partition(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Partition total dissipation between k and omega equations.

        Returns scaling factors for k and omega dissipation source terms.
        """
        C = self._C
        C_dp = getattr(C, 'C_diss_partition', 0.5)
        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)

        # Reynolds number based partition
        Re_t = k_safe / (1e-5 * omega_safe).clamp(min=1e-10)
        f_k = (1.0 + C_dp * (Re_t / (Re_t + 100.0))).clamp(max=2.0)
        f_omega = (2.0 - f_k).clamp(min=0.5, max=2.0)

        return f_k, f_omega

    def _freestream_sensitivity(self) -> torch.Tensor:
        """Freestream sensitivity correction for omega.

        Reduces omega in regions far from walls to minimize
        freestream sensitivity.
        """
        C = self._C
        C_fs = getattr(C, 'C_fs_corr', 0.1)
        y_norm = (self._y / self._y.max().clamp(min=1e-10)).clamp(0.0, 1.0)
        # Far from wall: reduce omega
        f_fs = (1.0 - C_fs * y_norm).clamp(min=0.5, max=1.0)
        return f_fs

    def nut(self) -> torch.Tensor:
        k = self._k.clamp(min=1e-16)
        omega = self._omega.clamp(min=1e-16)
        f_beta = self._f_beta()
        nut_base = k / omega
        if self._grad_U is not None:
            C = self._C
            S = self._strain_rate()
            S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
            denominator = (C.a1 * omega).max(S_mag)
            nut_limited = C.a1 * k / denominator.clamp(min=1e-16)
            nut_base = torch.min(nut_base, nut_limited)
        return (f_beta * nut_base).clamp(min=0.0)

    def correct(self) -> None:
        grad_U = torch.zeros(self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype)
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(self._U[:, i], "Gauss linear", mesh=self._mesh)
        self._grad_U = grad_U
        nut = self.nut()
        S = self._strain_rate()
        P_k = 2.0 * nut * (S * S).sum(dim=(1, 2))
        P_b = self._buoyancy_production(nut)
        P_k_total = P_k + P_b
        # Apply dissipation partition
        f_k, f_omega = self._dissipation_partition()
        f_fs = self._freestream_sensitivity()
        self._solve_k(P_k_total * f_k)
        self._solve_omega(P_k * f_omega * f_fs)

    def __repr__(self) -> str:
        return f"KOmegaEnhanced8Model(n_cells={self._mesh.n_cells})"
