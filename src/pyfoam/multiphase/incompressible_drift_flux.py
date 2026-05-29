"""
Incompressible drift-flux two-phase model.

Solves mixture momentum and a relative velocity equation for
incompressible two-phase flow. The model treats the mixture as a
single fluid with density and viscosity weighted by volume fraction,
and solves an algebraic or transport equation for the slip velocity
between the dispersed and continuous phases.

Governing equations:

    Mixture continuity:
        div(U_m) = 0

    Mixture momentum:
        d(rho_m U_m)/dt + div(rho_m U_m U_m)
            = -grad(p) + div(mu_m * grad(U_m)) + rho_m * g
              + div(turbulent dispersion stress)

    where:
        rho_m = alpha * rho_d + (1 - alpha) * rho_c
        mu_m  = alpha * mu_d + (1 - alpha) * mu_c

    Relative velocity (algebraic slip):
        U_slip = (rho_d - rho_c) * d^2 / (18 * mu_m) * g
                 * f(alpha)  [hindered settling correction]

The drift flux is then:
    J = alpha * (1 - alpha) * U_slip

Used in OpenFOAM's ``incompressibleDriftFluxFoam`` solver.

Parameters:
    rho_d, rho_c : phase densities (kg/m3)
    mu_d, mu_c   : phase dynamic viscosities (Pa·s)
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["IncompressibleDriftFlux"]

logger = logging.getLogger(__name__)


class IncompressibleDriftFlux:
    """Incompressible drift-flux mixture model.

    Computes mixture properties (density, viscosity), slip velocity,
    and drift flux for incompressible two-phase flow using the
    algebraic drift-flux approach.

    The slip velocity accounts for buoyancy, drag, and hindered
    settling via the Richardson-Zaki correlation:

        U_slip = V_t * (1 - alpha)^(n-1) * sign(rho_d - rho_c)

    where V_t is the terminal settling velocity and n is the
    Richardson-Zaki exponent.

    Parameters
    ----------
    alpha : torch.Tensor
        ``(n_cells,)`` dispersed-phase volume fraction.
    rho_d : float
        Dispersed-phase density (kg/m^3).
    rho_c : float
        Continuous-phase density (kg/m^3).
    mu_d : float
        Dispersed-phase dynamic viscosity (Pa·s).
    mu_c : float
        Continuous-phase dynamic viscosity (Pa·s).
    particle_diameter : float
        Particle/bubble diameter (m). Default: 1e-3.
    richardson_zaki_n : float
        Richardson-Zaki exponent. Default: 2.4 (intermediate Re).
    alpha_max : float
        Maximum packing volume fraction. Default: 0.63.
    """

    def __init__(
        self,
        rho_d: float = 1000.0,
        rho_c: float = 1.225,
        mu_d: float = 1.002e-3,
        mu_c: float = 1.8e-5,
        particle_diameter: float = 1e-3,
        richardson_zaki_n: float = 2.4,
        alpha_max: float = 0.63,
    ) -> None:
        self.rho_d = rho_d
        self.rho_c = rho_c
        self.mu_d = mu_d
        self.mu_c = mu_c
        self.particle_diameter = particle_diameter
        self.richardson_zaki_n = richardson_zaki_n
        self.alpha_max = alpha_max

    # ------------------------------------------------------------------
    # Mixture properties
    # ------------------------------------------------------------------

    def mixture_density(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute mixture density: rho_m = alpha * rho_d + (1 - alpha) * rho_c.

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` dispersed-phase volume fraction in [0, 1].

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` mixture density (kg/m^3).
        """
        alpha_c = alpha.clamp(0.0, 1.0)
        return alpha_c * self.rho_d + (1.0 - alpha_c) * self.rho_c

    def mixture_viscosity(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute mixture viscosity: mu_m = alpha * mu_d + (1 - alpha) * mu_c.

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` dispersed-phase volume fraction in [0, 1].

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` mixture dynamic viscosity (Pa·s).
        """
        alpha_c = alpha.clamp(0.0, 1.0)
        return alpha_c * self.mu_d + (1.0 - alpha_c) * self.mu_c

    # ------------------------------------------------------------------
    # Slip velocity
    # ------------------------------------------------------------------

    def compute_slip_velocity(
        self,
        alpha: torch.Tensor,
        gravity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute algebraic slip velocity with hindered settling.

        Uses the Richardson-Zaki correlation:

            U_slip = V_t * (1 - alpha)^(n-1) * gravity_dir

        where V_t = (rho_d - rho_c) * g * d^2 / (18 * mu_m) is the
        Stokes terminal velocity evaluated at the mixture viscosity.

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` dispersed-phase volume fraction.
        gravity : torch.Tensor, optional
            ``(3,)`` gravity vector. Default: [0, 0, -9.81].

        Returns
        -------
        torch.Tensor
            ``(n_cells, 3)`` slip velocity vector.
        """
        device = alpha.device
        dtype = alpha.dtype
        n_cells = alpha.shape[0]

        if gravity is None:
            gravity = torch.tensor([0.0, 0.0, -9.81], dtype=dtype, device=device)
        else:
            gravity = gravity.to(device=device, dtype=dtype)

        alpha_c = alpha.clamp(0.0, self.alpha_max)

        # Mixture viscosity
        mu_m = self.mixture_viscosity(alpha_c)

        # Stokes terminal velocity magnitude
        d_rho = self.rho_d - self.rho_c
        d = self.particle_diameter
        tau = d ** 2 / (18.0 * mu_m.clamp(min=1e-30))
        V_t = d_rho * tau  # signed: positive if rho_d > rho_c

        # Hindered settling correction: (1 - alpha)^(n - 1)
        n = self.richardson_zaki_n
        hindered = (1.0 - alpha_c).pow(n - 1.0)

        # Slip velocity vector
        slip_mag = V_t * hindered
        U_slip = slip_mag.unsqueeze(-1) * gravity.unsqueeze(0).expand(n_cells, -1)

        return U_slip

    # ------------------------------------------------------------------
    # Drift flux
    # ------------------------------------------------------------------

    def compute_drift_flux(
        self,
        alpha: torch.Tensor,
        U_slip: torch.Tensor,
    ) -> torch.Tensor:
        """Compute drift flux: J = alpha * (1 - alpha) * U_slip.

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` dispersed-phase volume fraction.
        U_slip : torch.Tensor
            ``(n_cells, 3)`` slip velocity.

        Returns
        -------
        torch.Tensor
            ``(n_cells, 3)`` drift flux vector.
        """
        alpha_c = alpha.clamp(0.0, self.alpha_max)
        factor = (alpha_c * (1.0 - alpha_c)).unsqueeze(-1)
        return factor * U_slip

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def solve_timestep(
        self,
        alpha: torch.Tensor,
        gravity: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Single-step solve: compute mixture properties, slip, and flux.

        Convenience method that returns all relevant quantities in one call.

        Parameters
        ----------
        alpha : torch.Tensor
            ``(n_cells,)`` dispersed-phase volume fraction.
        gravity : torch.Tensor, optional
            ``(3,)`` gravity vector. Default: [0, 0, -9.81].

        Returns
        -------
        dict
            Keys: ``rho_m``, ``mu_m``, ``U_slip``, ``J``.
        """
        rho_m = self.mixture_density(alpha)
        mu_m = self.mixture_viscosity(alpha)
        U_slip = self.compute_slip_velocity(alpha, gravity)
        J = self.compute_drift_flux(alpha, U_slip)
        return {"rho_m": rho_m, "mu_m": mu_m, "U_slip": U_slip, "J": J}

    def __repr__(self) -> str:
        return (
            f"IncompressibleDriftFlux(rho_d={self.rho_d}, rho_c={self.rho_c}, "
            f"mu_d={self.mu_d}, mu_c={self.mu_c}, "
            f"d={self.particle_diameter}, n={self.richardson_zaki_n})"
        )
