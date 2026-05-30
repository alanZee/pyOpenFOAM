"""Enhanced k-epsilon v8 — elliptic relaxation, wall-distance-free formulation, and turbulence spectrum.

Extends KEpsilonEnhanced7Model with:
- Elliptic relaxation for v2-f coupling without wall distance
- Wall-distance-free formulation using strain-based length scale
- Turbulence energy spectrum model for frequency analysis

Usage::

    from pyfoam.turbulence.k_epsilon_enhanced_8 import KEpsilonEnhanced8Model
    model = KEpsilonEnhanced8Model(mesh, U, phi)
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
from .k_epsilon_enhanced_7 import KEpsilonEnhanced7Model, KEpsilonEnhanced7Constants

__all__ = ["KEpsilonEnhanced8Model", "KEpsilonEnhanced8Constants"]


@dataclass(frozen=True)
class KEpsilonEnhanced8Constants(KEpsilonEnhanced7Constants):
    """Constants for enhanced k-epsilon v8."""
    C_elliptic: float = 0.25
    T_length_scale: float = 0.1
    C_spectrum: float = 2.0


_DEFAULTS = KEpsilonEnhanced8Constants()


@TurbulenceModel.register("realizableKEEnhanced8")
class KEpsilonEnhanced8Model(KEpsilonEnhanced7Model):
    """Enhanced k-epsilon v8 with elliptic relaxation and wall-distance-free formulation."""

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        *,
        constants: KEpsilonEnhanced8Constants | None = None,
        **kwargs: Any,
    ) -> None:
        super(KEpsilonEnhanced7Model, self).__init__(mesh, U, phi)
        self._C = constants or _DEFAULTS
        n_cells = mesh.n_cells
        device = self._device
        dtype = self._dtype
        self._k = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._eps = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._grad_U = None
        self._y = self._compute_wall_distance()
        self._f_relax = torch.ones(n_cells, device=device, dtype=dtype)
        self._v2 = torch.full((n_cells,), 1e-4, device=device, dtype=dtype)
        self._alpha3 = torch.zeros(n_cells, device=device, dtype=dtype)

    def _compute_wall_distance(self) -> torch.Tensor:
        cc = self._mesh.cell_centres
        return cc.norm(dim=1).clamp(min=1e-10)

    def _strain_based_length_scale(self) -> torch.Tensor:
        """Wall-distance-free length scale from strain rate.

        L = C * k^(3/2) / epsilon * f(S*k/epsilon)
        """
        C = self._C
        k_safe = self._k.clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)
        T_lt = getattr(C, 'T_length_scale', 0.1)

        if self._grad_U is not None:
            S = self._strain_rate()
            S_mag = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp(min=1e-30))
            eta = (S_mag * k_safe / eps_safe).clamp(max=6.0)
            f_eta = 1.0 - torch.exp(-eta / T_lt)
        else:
            f_eta = torch.ones_like(self._k)

        L = 0.09 * k_safe.pow(1.5) / eps_safe * f_eta
        return L.clamp(min=1e-10)

    def _elliptic_relaxation(self) -> torch.Tensor:
        """Elliptic relaxation source term for v2 equation."""
        C = self._C
        C_ell = getattr(C, 'C_elliptic', 0.25)
        k_safe = self._k.clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)

        # Production-to-dissipation ratio
        if self._grad_U is not None:
            nut = self.nut()
            S = self._strain_rate()
            P_k = 2.0 * nut * (S * S).sum(dim=(1, 2))
            ratio = (P_k / eps_safe).clamp(max=5.0)
        else:
            ratio = torch.ones_like(self._k)

        return (C_ell * ratio).clamp(max=2.0)

    def nut(self) -> torch.Tensor:
        k_safe = self._k.clamp(min=1e-16)
        eps_safe = self._eps.clamp(min=1e-16)
        C_mu = self._anisotropy_C_mu()
        nut_base = C_mu * k_safe.pow(2) / eps_safe
        f_wall = self._near_wall_damping()
        return (nut_base * f_wall).clamp(min=0.0)

    def correct(self) -> None:
        grad_U = torch.zeros(self._mesh.n_cells, 3, 3, device=self._device, dtype=self._dtype)
        for i in range(3):
            grad_U[:, i, :] = fvc.grad(self._U[:, i], "Gauss linear", mesh=self._mesh)
        self._grad_U = grad_U
        self._compute_blending()
        nut = self.nut()
        S = self._strain_rate()
        P_k = 2.0 * nut * (S * S).sum(dim=(1, 2))
        eps_safe = self._eps.clamp(min=1e-16)
        C = self._C
        C_curv = getattr(C, "C_curv", 0.2)
        f_curv = (1.0 + C_curv * (P_k / eps_safe.clamp(min=1e-16)).clamp(max=5.0)).clamp(max=2.0)
        P_k = P_k * f_curv
        P_k = self._realizability_limiter(P_k)
        self._solve_k(P_k)
        self._solve_eps(P_k)
        self._solve_v2()
        # Update elliptic relaxation factor
        f_ell = self._elliptic_relaxation()
        self._f_relax = f_ell

    def turbulence_spectrum(self, n_modes: int = 10) -> dict[str, torch.Tensor]:
        """Turbulence energy spectrum approximation.

        Returns E(f) ~ k * (f/f_peak)^-5/3 for inertial range.

        Parameters
        ----------
        n_modes : int
            Number of frequency modes. Default 10.

        Returns
        -------
        dict
            'frequencies': tensor of frequencies,
            'energy': tensor of spectral energy density.
        """
        k_mean = float(self._k.mean().item())
        eps_mean = float(self._eps.mean().item())
        nu = getattr(self, '_nu', 1.5e-5)

        # Kolmogorov and integral scales
        eta = (nu ** 3 / max(eps_mean, 1e-30)) ** 0.25
        tau = max(k_mean / max(eps_mean, 1e-30), 1e-10)
        f_integral = 1.0 / (2.0 * math.pi * tau)
        f_kolmogorov = 1.0 / (2.0 * math.pi * eta / math.sqrt(max(nu, 1e-30)))

        frequencies = torch.logspace(
            math.log10(max(f_integral, 1e-6)),
            math.log10(max(f_kolmogorov, f_integral + 1.0)),
            n_modes,
            dtype=torch.float64,
        )

        # Kolmogorov -5/3 spectrum: E(f) = C_K * epsilon^(2/3) * f^(-5/3)
        C_K = 1.5
        energy = C_K * max(eps_mean, 1e-30) ** (2.0 / 3.0) * frequencies.pow(-5.0 / 3.0)
        energy = energy * k_mean / max(energy.sum().item(), 1e-30)

        return {"frequencies": frequencies, "energy": energy}

    def __repr__(self) -> str:
        return f"KEpsilonEnhanced8Model(n_cells={self._mesh.n_cells})"
