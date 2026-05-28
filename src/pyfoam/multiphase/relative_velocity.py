"""
Relative velocity models for Euler-Euler multiphase flows.

Provides algebraic slip models for computing the relative velocity between
phases in multiphase Euler-Euler formulations.  These models are used in
drift-flux and mixture models as closure for the slip velocity.

Based on:
- Manninen et al. (1996) — algebraic slip model
- Grace (1976) — drag correlation for bubbles and particles in liquids

Usage::

    from pyfoam.multiphase.relative_velocity import (
        ManninenRelativeVelocity,
        GraceRelativeVelocity,
    )

    model = ManninenRelativeVelocity(rho_d=1.225, rho_c=998.0, d=1e-3)
    U_slip = model.compute(alpha_d, U_mix)
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "RelativeVelocityModel",
    "ManninenRelativeVelocity",
    "GraceRelativeVelocity",
]

logger = logging.getLogger(__name__)


class RelativeVelocityModel(ABC):
    """Abstract base class for relative velocity (slip) models.

    Subclasses must implement :meth:`compute` which returns the slip
    velocity vector for a dispersed phase relative to the continuous phase.
    """

    @abstractmethod
    def compute(
        self,
        alpha_d: torch.Tensor,
        U_mix: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the relative (slip) velocity.

        Parameters
        ----------
        alpha_d : torch.Tensor
            Dispersed phase volume fraction ``(n_cells,)``.
        U_mix : torch.Tensor
            Mixture velocity ``(n_cells, 3)``.

        Returns
        -------
        torch.Tensor
            Slip velocity ``(n_cells, 3)`` — velocity of the dispersed
            phase relative to the continuous phase.
        """


class ManninenRelativeVelocity(RelativeVelocityModel):
    """Manninen et al. (1996) algebraic slip model.

    The slip velocity is computed from a balance of drag, buoyancy, and
    dispersion forces:

        U_slip = (rho_d - rho_c) * d^2 / (18 * mu_c * (1 + 2.5 * alpha_d))
                 * (g - dU_mix/dt)

    For the simplified algebraic form (neglecting virtual mass and
    turbulent dispersion):

        U_slip = tau_d * (rho_d - rho_c) / rho_d * (g - dU/dt)

    where tau_d is the particle relaxation time.

    Parameters
    ----------
    rho_d : float
        Dispersed phase density (kg/m^3).
    rho_c : float
        Continuous phase density (kg/m^3).
    d : float
        Particle/bubble diameter (m).
    mu_c : float
        Continuous phase dynamic viscosity (Pa·s).
    C_vm : float
        Virtual mass coefficient (default 0.5).
    """

    def __init__(
        self,
        rho_d: float,
        rho_c: float,
        d: float,
        mu_c: float = 1.002e-3,
        C_vm: float = 0.5,
    ) -> None:
        self.rho_d = rho_d
        self.rho_c = rho_c
        self.d = d
        self.mu_c = mu_c
        self.C_vm = C_vm

    @property
    def particle_relaxation_time(self) -> float:
        """Particle relaxation time: tau_d = rho_d * d^2 / (18 * mu_c)."""
        return self.rho_d * self.d ** 2 / (18.0 * self.mu_c)

    def compute(
        self,
        alpha_d: torch.Tensor,
        U_mix: torch.Tensor,
        g: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the Manninen algebraic slip velocity.

        Parameters
        ----------
        alpha_d : torch.Tensor
            Dispersed phase volume fraction ``(n_cells,)``.
        U_mix : torch.Tensor
            Mixture velocity ``(n_cells, 3)``.
        g : torch.Tensor, optional
            Gravity vector ``(3,)``.  Default: [0, 0, -9.81].

        Returns
        -------
        torch.Tensor
            Slip velocity ``(n_cells, 3)``.
        """
        device = U_mix.device
        dtype = U_mix.dtype
        n_cells = U_mix.shape[0]

        if g is None:
            g = torch.tensor([0.0, 0.0, -9.81], dtype=dtype, device=device)
        else:
            g = g.to(device=device, dtype=dtype)

        # Particle relaxation time (modified for volume fraction)
        tau_d = self.particle_relaxation_time
        alpha_safe = alpha_d.to(dtype=dtype).clamp(min=0.0, max=1.0 - 1e-6)

        # Effective relaxation time (Richardson-Zaki type correction)
        # tau_eff = tau_d / (1 - alpha_d) * (1 + 2.5 * alpha_d)
        tau_eff = tau_d * (1.0 + 2.5 * alpha_safe) / (1.0 - alpha_safe)

        # Density ratio effect
        drho = (self.rho_d - self.rho_c) / self.rho_d

        # Slip velocity: U_slip = tau_eff * drho * g
        # This is the algebraic equilibrium form
        U_slip = tau_eff.unsqueeze(-1) * drho * g.unsqueeze(0).expand(n_cells, -1)

        return U_slip


class GraceRelativeVelocity(RelativeVelocityModel):
    """Grace drag correlation for bubbles and particles.

    Uses the Grace (1976) drag correlation which accounts for the shape
    regime of bubbles/particles:

    For spherical particles (Re < 1):
        Cd = 24/Re

    For intermediate Re (1 < Re < 1000):
        Cd = 24/Re * (1 + 0.15 * Re^0.687)

    For distorted particles (high Re or high Eo):
        Cd = (4/3) * Eo / (Mo^-0.149 * (mu_c/mu_d)^-0.14)

    where Eo = Eotvos number, Mo = Morton number.

    Parameters
    ----------
    rho_d : float
        Dispersed phase density (kg/m^3).
    rho_c : float
        Continuous phase density (kg/m^3).
    d : float
        Particle/bubble diameter (m).
    mu_c : float
        Continuous phase dynamic viscosity (Pa·s).
    mu_d : float
        Dispersed phase dynamic viscosity (Pa·s).
    sigma : float
        Surface tension coefficient (N/m).
    """

    def __init__(
        self,
        rho_d: float,
        rho_c: float,
        d: float,
        mu_c: float = 1.002e-3,
        mu_d: float = 1.8e-5,
        sigma: float = 0.072,
    ) -> None:
        self.rho_d = rho_d
        self.rho_c = rho_c
        self.d = d
        self.mu_c = mu_c
        self.mu_d = mu_d
        self.sigma = sigma

    @property
    def eotvos_number(self) -> float:
        """Eotvos number: Eo = g * |rho_c - rho_d| * d^2 / sigma."""
        g = 9.81
        return g * abs(self.rho_c - self.rho_d) * self.d ** 2 / self.sigma

    @property
    def morton_number(self) -> float:
        """Morton number: Mo = g * mu_c^4 * |rho_c - rho_d| / (rho_c^2 * sigma^3)."""
        g = 9.81
        drho = abs(self.rho_c - self.rho_d)
        return g * self.mu_c ** 4 * drho / (self.rho_c ** 2 * self.sigma ** 3)

    def _grace_drag_coefficient(self, Re: torch.Tensor) -> torch.Tensor:
        """Compute Grace drag coefficient Cd(Re, Eo, Mo)."""
        Re_safe = Re.clamp(min=1e-10)
        Eo = self.eotvos_number
        Mo = self.morton_number

        # Schiller-Naumann for low Re
        Cd_low = 24.0 / Re_safe * (1.0 + 0.15 * Re_safe.pow(0.687))

        # Grace correlation for distorted regime
        if Mo > 1e-10:
            viscosity_ratio = self.mu_c / max(self.mu_d, 1e-20)
            Cd_distorted = (4.0 / 3.0) * Eo / (
                Mo ** (-0.149) * viscosity_ratio ** (-0.14)
            )
        else:
            Cd_distorted = 0.44  # Newton regime

        # Blending: use max of Schiller-Naumann and distorted
        Cd = torch.where(
            Cd_low < Cd_distorted,
            torch.full_like(Re_safe, Cd_distorted),
            Cd_low,
        )

        return Cd

    def compute(
        self,
        alpha_d: torch.Tensor,
        U_mix: torch.Tensor,
        g: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the Grace slip velocity.

        Parameters
        ----------
        alpha_d : torch.Tensor
            Dispersed phase volume fraction ``(n_cells,)``.
        U_mix : torch.Tensor
            Mixture velocity ``(n_cells, 3)``.
        g : torch.Tensor, optional
            Gravity vector ``(3,)``.  Default: [0, 0, -9.81].

        Returns
        -------
        torch.Tensor
            Slip velocity ``(n_cells, 3)``.
        """
        device = U_mix.device
        dtype = U_mix.dtype
        n_cells = U_mix.shape[0]

        if g is None:
            g = torch.tensor([0.0, 0.0, -9.81], dtype=dtype, device=device)
        else:
            g = g.to(device=device, dtype=dtype)

        # Estimate terminal velocity from buoyancy-drag balance
        # U_t^2 = 4/3 * d * |rho_d - rho_c| * g / (rho_c * Cd)
        # Use iterative approach: start with Stokes estimate, refine
        drho = abs(self.rho_d - self.rho_c)
        g_mag = g.norm().item() if g.dim() == 1 else 9.81

        # Stokes terminal velocity as initial estimate
        if self.mu_c > 0:
            U_stokes = drho * g_mag * self.d ** 2 / (18.0 * self.mu_c)
        else:
            U_stokes = 1e-3

        # Estimate Re and Cd iteratively
        Re_est = self.rho_c * U_stokes * self.d / self.mu_c
        Cd_est = self._grace_drag_coefficient(
            torch.tensor([Re_est], dtype=dtype, device=device),
        )

        Cd_val = Cd_est.mean().item()

        Cd_val = max(Cd_val, 1e-6)
        U_t = math.sqrt(4.0 / 3.0 * self.d * drho * g_mag / (self.rho_c * Cd_val))

        # Terminal velocity vector (opposing gravity)
        g_norm = g / max(g_mag, 1e-10)
        U_t_vec = -U_t * g_norm  # Terminal velocity opposes gravity

        # Volume fraction correction (hindered settling)
        alpha_safe = alpha_d.to(dtype=dtype).clamp(min=0.0, max=1.0 - 1e-6)
        hindered = (1.0 - alpha_safe).pow(2.0)  # Richardson-Zaki exponent ~2

        # Slip velocity
        U_slip = U_t_vec.unsqueeze(0).expand(n_cells, -1) * hindered.unsqueeze(-1)

        return U_slip
