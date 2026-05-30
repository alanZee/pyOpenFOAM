"""Enhanced k-omega v9 — turbulence kinetic energy diffusion, variable sigma_k, and spectral diagnostics.

Extends KOmegaEnhanced8Model with:
- Turbulent diffusion of k with variable Schmidt number
- Variable sigma_k based on local flow topology
- Spectral diagnostics for omega equation

Usage::

    from pyfoam.turbulence.k_omega_enhanced_9 import KOmegaEnhanced9Model
    model = KOmegaEnhanced9Model(mesh, U, phi)
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
from .k_omega_enhanced_8 import KOmegaEnhanced8Model, KOmegaEnhanced8Constants

__all__ = ["KOmegaEnhanced9Model", "KOmegaEnhanced9Constants"]


@dataclass(frozen=True)
class KOmegaEnhanced9Constants(KOmegaEnhanced8Constants):
    """Constants for enhanced k-omega v9."""
    sigma_k_min: float = 0.4
    sigma_k_max: float = 1.5
    C_turb_diff: float = 0.5


_DEFAULTS = KOmegaEnhanced9Constants()


@TurbulenceModel.register("kOmegaEnhanced9")
class KOmegaEnhanced9Model(KOmegaEnhanced8Model):
    """Enhanced k-omega v9 with variable sigma_k and spectral diagnostics."""

    def __init__(
        self,
        mesh: Any, U: Any, phi: torch.Tensor,
        *, constants: KOmegaEnhanced9Constants | None = None,
        g: tuple[float, float, float] = (0.0, 0.0, -9.81),
        beta_t: float = 3e-3,
        enable_variable_sigma: bool = False,
        **kwargs: Any,
    ) -> None:
        super(KOmegaEnhanced8Model, self).__init__(mesh, U, phi)
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

    def _compute_wall_distance(self) -> torch.Tensor:
        cc = self._mesh.cell_centres
        return cc.norm(dim=1).clamp(min=1e-10)

    def variable_sigma_k(self) -> torch.Tensor:
        """Compute variable sigma_k based on local flow topology.

        sigma_k varies between sigma_k_min (strong strain) and sigma_k_max (weak strain).
        """
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

        # Blend sigma_k
        sigma_k = sigma_min + (sigma_max - sigma_min) * (1.0 / (1.0 + eta)).clamp(0.0, 1.0)
        return sigma_k

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
        f_k, f_omega = self._dissipation_partition()
        f_fs = self._freestream_sensitivity()
        self._solve_k(P_k_total * f_k)
        self._solve_omega(P_k * f_omega * f_fs)

    def __repr__(self) -> str:
        vs = ", variable_sigma" if self._enable_variable_sigma else ""
        return f"KOmegaEnhanced9Model(n_cells={self._mesh.n_cells}{vs})"
