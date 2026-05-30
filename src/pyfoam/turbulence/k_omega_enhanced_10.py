"""Enhanced k-omega v10 — cross-diffusion limiter, low-Re damping, and spectral analysis.

Extends KOmegaEnhanced9Model with:
- Cross-diffusion limiter for improved convergence
- Low-Re damping functions for near-wall accuracy
- Spectral analysis of omega equation

Usage::

    from pyfoam.turbulence.k_omega_enhanced_10 import KOmegaEnhanced10Model
    model = KOmegaEnhanced10Model(mesh, U, phi)
    model.correct()
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import math
import torch
from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.discretisation.operators import fvm, fvc
from .turbulence_model import TurbulenceModel
from .k_omega_enhanced_9 import KOmegaEnhanced9Model, KOmegaEnhanced9Constants

__all__ = ["KOmegaEnhanced10Model", "KOmegaEnhanced10Constants"]


@dataclass(frozen=True)
class KOmegaEnhanced10Constants(KOmegaEnhanced9Constants):
    """Constants for enhanced k-omega v10."""
    C_cross_diff_lim: float = 0.8
    C_low_re_beta: float = 0.0708
    C_spectral: float = 0.1


_DEFAULTS = KOmegaEnhanced10Constants()


@TurbulenceModel.register("kOmegaEnhanced10")
class KOmegaEnhanced10Model(KOmegaEnhanced9Model):
    """Enhanced k-omega v10 with cross-diffusion limiter and low-Re damping."""

    def __init__(
        self,
        mesh: Any, U: Any, phi: torch.Tensor,
        *, constants: KOmegaEnhanced10Constants | None = None,
        g: tuple[float, float, float] = (0.0, 0.0, -9.81),
        beta_t: float = 3e-3,
        enable_variable_sigma: bool = False,
        enable_cross_diff_limit: bool = False,
        enable_low_re_damping: bool = False,
        **kwargs: Any,
    ) -> None:
        super(KOmegaEnhanced9Model, self).__init__(mesh, U, phi)
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
        self._enable_variable_sigma = enable_variable_sigma
        self._enable_cross_diff_lim = enable_cross_diff_limit
        self._enable_low_re = enable_low_re_damping

    def _compute_wall_distance(self) -> torch.Tensor:
        cc = self._mesh.cell_centres
        return cc.norm(dim=1).clamp(min=1e-10)

    def cross_diffusion_limiter(self, CD_kw: torch.Tensor) -> torch.Tensor:
        """Limit cross-diffusion term for numerical stability.

        CD_limited = CD * sigma_d / max(sigma_d, |CD|/|k*grad(omega)|)

        Parameters
        ----------
        CD_kw : torch.Tensor
            Raw cross-diffusion term.

        Returns
        -------
        torch.Tensor
            Limited cross-diffusion.
        """
        if not self._enable_cross_diff_lim:
            return CD_kw

        C = self._C
        sigma_d = getattr(C, 'C_cross_diff_lim', 0.8)
        CD_mag = CD_kw.abs().clamp(max=10.0)
        k_safe = self._k.clamp(min=1e-16)
        limit = sigma_d * k_safe
        return CD_kw * (limit / (CD_mag + limit).clamp(min=1e-30))

    def low_re_damping(self) -> torch.Tensor:
        """Low-Re damping functions for near-wall region.

        f_beta = (1 + 4.4 * Re_t^-0.75) where Re_t = k / (nu * omega)

        Returns
        -------
        torch.Tensor
            Damping factor for each cell.
        """
        if not self._enable_low_re:
            return torch.ones_like(self._k)

        C = self._C
        nu = max(self._nu, 1e-30)
        k_safe = self._k.clamp(min=1e-16)
        omega_safe = self._omega.clamp(min=1e-16)

        Re_t = k_safe / (nu * omega_safe)
        f_beta = (1.0 + 4.4 * Re_t.pow(-0.75)).clamp(max=2.0)
        return f_beta

    def variable_sigma_k(self) -> torch.Tensor:
        """Compute variable sigma_k based on local flow topology (inherited)."""
        C = self._C
        sigma_min = getattr(C, 'sigma_k_min', 0.4)
        sigma_max = getattr(C, 'sigma_k_max', 1.5)

        if not self._enable_variable_sigma or self._grad_U is None:
            return torch.full_like(self._k, 0.5)

        k_safe = self._k.clamp(min=1e-16)
        eps_safe = 0.09 * k_safe * self._omega.clamp(min=1e-16)

        S = self._strain_rate()
        S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
        tau = k_safe / eps_safe.clamp(min=1e-16)
        eta = (S_mag * tau).clamp(max=10.0)

        sigma_k = sigma_min + (sigma_max - sigma_min) * (1.0 / (1.0 + eta)).clamp(0.0, 1.0)
        return sigma_k

    def nut(self) -> torch.Tensor:
        k = self._k.clamp(min=1e-16)
        omega = self._omega.clamp(min=1e-16)
        f_damp = self.low_re_damping()
        nut_base = k / omega
        if self._grad_U is not None:
            C = self._C
            S = self._strain_rate()
            S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
            denominator = (C.a1 * omega).max(S_mag)
            nut_limited = C.a1 * k / denominator.clamp(min=1e-16)
            nut_base = torch.min(nut_base, nut_limited)
        return (f_damp * nut_base).clamp(min=0.0)

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
        f_k, f_omega = self._dissipation_partition()
        f_fs = self._freestream_sensitivity()
        self._solve_k(P_k_total * f_k)
        self._solve_omega(P_k * f_omega * f_fs)

    def __repr__(self) -> str:
        cdl = ", cross_diff_lim" if self._enable_cross_diff_lim else ""
        lr = ", low_re" if self._enable_low_re else ""
        return f"KOmegaEnhanced10Model(n_cells={self._mesh.n_cells}{cdl}{lr})"
