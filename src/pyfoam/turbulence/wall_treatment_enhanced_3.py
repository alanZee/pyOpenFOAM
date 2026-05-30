"""
Enhanced wall treatment model v3 with adaptive blending and heat transfer.

Extends :class:`~pyfoam.turbulence.wall_treatment_enhanced_2.EnhancedWallTreatment2`
and :class:`~pyfoam.turbulence.wall_treatment_enhanced_2.FourLayerWallTreatment`
with:

- Jayatilleke-like blending function for improved buffer layer
- Heat transfer wall function (Nusselt number correlation)
- Adaptive y+ switching with hysteresis

Usage::

    from pyfoam.turbulence.wall_treatment_enhanced_3 import EnhancedWallTreatment3, AdaptiveWallTreatment

    wt = EnhancedWallTreatment3(nu=1.5e-5)
    nut_wall = wt.compute_nut(k, y)
    h_wall = wt.compute_htc(k, y, T, T_wall)
"""

from __future__ import annotations

import logging
import math

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.turbulence.wall_treatment import WallTreatment
from pyfoam.turbulence.wall_treatment_enhanced_2 import (
    EnhancedWallTreatment2,
    FourLayerWallTreatment,
)

__all__ = [
    "EnhancedWallTreatment3",
    "AdaptiveWallTreatment",
]

logger = logging.getLogger(__name__)


@WallTreatment.register("enhanced3")
class EnhancedWallTreatment3(EnhancedWallTreatment2):
    """Enhanced wall treatment v3 with adaptive blending and heat transfer.

    Extends :class:`EnhancedWallTreatment2` with:

    - **Jayatilleke blending**: improved blending function from
      Jayatilleke (1969) for buffer layer transition:
      lambda = (1/Pr_t) * ln(y+) + C_jay
    - **Heat transfer wall function**: computes wall heat transfer
      coefficient from k, y, and temperature boundary conditions.
    - **Adaptive switching with hysteresis**: prevents oscillation in
      y+ regime switching by using different thresholds for on/off.

    Parameters
    ----------
    nu : float
        Molecular kinematic viscosity (m^2/s).
    kappa : float
        Von Karman constant.
    E : float
        Log-law wall constant.
    C_mu : float
        k-epsilon model constant.
    y_plus_low : float
        Upper bound of low-Re region.
    y_plus_high : float
        Lower bound of high-Re region.
    ks : float
        Equivalent sand-grain roughness height (m). Default 0.
    Pr : float
        Prandtl number. Default 0.71.
    Pr_t : float
        Turbulent Prandtl number. Default 0.85.
    hysteresis_width : float
        Width of y+ hysteresis band. Default 2.0.
    """

    def __init__(
        self,
        nu: float = 1.5e-5,
        kappa: float = 0.41,
        E: float = 9.8,
        C_mu: float = 0.09,
        y_plus_low: float = 5.0,
        y_plus_high: float = 30.0,
        ks: float = 0.0,
        Pr: float = 0.71,
        Pr_t: float = 0.85,
        hysteresis_width: float = 2.0,
    ) -> None:
        super().__init__(
            nu=nu, kappa=kappa, E=E, C_mu=C_mu,
            y_plus_low=y_plus_low, y_plus_high=y_plus_high, ks=ks,
        )
        self._Pr = Pr
        self._Pr_t = Pr_t
        self._hysteresis = hysteresis_width

    @property
    def Pr(self) -> float:
        """Molecular Prandtl number."""
        return self._Pr

    @property
    def Pr_t(self) -> float:
        """Turbulent Prandtl number."""
        return self._Pr_t

    # ------------------------------------------------------------------
    # Jayatilleke blending
    # ------------------------------------------------------------------

    def _jayatilleke_blending(self, y_plus: torch.Tensor) -> torch.Tensor:
        """Jayatilleke blending function for thermal wall function.

        P(y+) = 9.24 * ((Pr/Pr_t)^0.75 - 1) * (1 + 0.28 * exp(-0.007 * Pr/Pr_t))
                * (y+/E)^(1/4) * (1 - exp(-y+/11))^2

        Parameters
        ----------
        y_plus : torch.Tensor
            Dimensionless wall distance.

        Returns
        -------
        torch.Tensor
            Jayatilleke P function.
        """
        Pr_Pr_t = self._Pr / max(self._Pr_t, 1e-10)
        coeff = 9.24 * (Pr_Pr_t ** 0.75 - 1.0) * (1.0 + 0.28 * math.exp(-0.007 * Pr_Pr_t))
        y_E = y_plus / max(self.E, 1e-10)

        P = coeff * y_E.clamp(min=0.0).pow(0.25) * (1.0 - torch.exp(-y_plus / 11.0)).pow(2)
        return P.clamp(min=0.0)

    # ------------------------------------------------------------------
    # Heat transfer wall function
    # ------------------------------------------------------------------

    def compute_htc(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
        T_fluid: torch.Tensor,
        T_wall: torch.Tensor,
        rho: float = 1.2,
        Cp: float = 1005.0,
    ) -> torch.Tensor:
        """Compute wall heat transfer coefficient.

        htc = rho * Cp * u_tau / (Pr_t * (1/kappa * ln(y+) + P(y+)))

        Parameters
        ----------
        k : torch.Tensor
            Turbulent kinetic energy.
        y : torch.Tensor
            Wall-normal distance.
        T_fluid : torch.Tensor
            Fluid temperature near wall.
        T_wall : torch.Tensor
            Wall temperature.
        rho : float
            Fluid density (kg/m^3).
        Cp : float
            Specific heat (J/(kg*K)).

        Returns
        -------
        torch.Tensor
            Heat transfer coefficient (W/(m^2*K)).
        """
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        u_tau = self.compute_u_tau(k)
        y_plus = self.compute_y_plus(u_tau, y)
        P = self._jayatilleke_blending(y_plus)

        # T* = Pr_t * (1/kappa * ln(y+) + P)
        log_arg = y_plus.clamp(min=1.01)
        T_star = self._Pr_t * (torch.log(log_arg) / self.kappa + P)
        T_star = T_star.clamp(min=0.01)

        # htc = rho * Cp * u_tau / T_star
        htc = rho * Cp * u_tau / T_star

        return htc.clamp(min=0.0)

    # ------------------------------------------------------------------
    # Hysteresis-aware blending
    # ------------------------------------------------------------------

    def _blending_with_hysteresis(
        self,
        y_plus: torch.Tensor,
        direction: int = 1,
    ) -> torch.Tensor:
        """Blending factor with hysteresis.

        direction=1 (increasing y+): uses y_plus_high + hysteresis
        direction=-1 (decreasing y+): uses y_plus_high - hysteresis

        Parameters
        ----------
        y_plus : torch.Tensor
            Dimensionless wall distance.
        direction : int
            1 for increasing y+, -1 for decreasing.

        Returns
        -------
        torch.Tensor
            Blending factor in [0, 1].
        """
        if direction > 0:
            y_high = self.y_plus_high + self._hysteresis
        else:
            y_high = self.y_plus_high - self._hysteresis

        y_high = max(y_high, self.y_plus_low + 1.0)
        blend = 0.5 * (1.0 + torch.tanh(
            ((y_plus - 0.5 * (self.y_plus_low + y_high))
             / (0.15 * (y_high - self.y_plus_low)))
            .clamp(min=-10.0, max=10.0)
        ))
        return blend.clamp(min=0.0, max=1.0)

    def __repr__(self) -> str:
        rough = f", ks={self._ks}" if self._ks > 0 else ""
        return (
            f"EnhancedWallTreatment3(nu={self.nu}, Pr={self._Pr}, "
            f"y+=[{self.y_plus_low}, {self.y_plus_high}]{rough})"
        )


@WallTreatment.register("adaptive")
class AdaptiveWallTreatment(WallTreatment):
    """Adaptive wall treatment that automatically selects the best model.

    Monitors local y+ and selects between:
    - viscous sublayer: no wall function needed
    - buffer layer: enhanced blending
    - log-law: standard wall function

    Uses a simple state machine with hysteresis to prevent oscillation.

    Parameters
    ----------
    nu : float
        Kinematic viscosity (m^2/s).
    kappa : float
        Von Karman constant.
    E : float
        Log-law constant.
    C_mu : float
        k-epsilon model constant.
    """

    def __init__(
        self,
        nu: float = 1.5e-5,
        kappa: float = 0.41,
        E: float = 9.8,
        C_mu: float = 0.09,
    ) -> None:
        super().__init__(nu=nu, kappa=kappa, E=E, C_mu=C_mu)
        self._previous_state: torch.Tensor | None = None

    def _regime_state(
        self,
        y_plus: torch.Tensor,
    ) -> torch.Tensor:
        """Determine regime: 0=viscous, 1=buffer, 2=log-law.

        With hysteresis: if previous state was log-law, transition to buffer
        at y+ = 10 instead of 15.

        Parameters
        ----------
        y_plus : torch.Tensor
            Dimensionless wall distance.

        Returns
        -------
        torch.Tensor
            Integer regime state.
        """
        viscous_threshold = 5.0
        buffer_threshold = 15.0
        log_threshold = 30.0

        # Simple regime detection
        state = torch.zeros_like(y_plus, dtype=torch.long)
        state = torch.where(y_plus >= viscous_threshold, torch.ones_like(state), state)
        state = torch.where(y_plus >= log_threshold, 2 * torch.ones_like(state), state)

        # Hysteresis: if was in log-law, need y+ to drop below buffer to leave
        if self._previous_state is not None:
            was_log = self._previous_state == 2
            # If was log-law and y+ is in buffer zone, keep as log-law
            still_buffer = (y_plus >= buffer_threshold) & (y_plus < log_threshold) & was_log
            state = torch.where(still_buffer, 2 * torch.ones_like(state), state)

        self._previous_state = state.clone()
        return state

    def compute_nut(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Adaptive nut computation."""
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        u_tau = self.compute_u_tau(k)
        y_plus = self.compute_y_plus(u_tau, y)
        regime = self._regime_state(y_plus)

        nut_log = self.kappa * u_tau * y / torch.log(
            self.E * y_plus.clamp(min=1.01)
        )

        # Buffer blending
        bw = ((y_plus - 15.0) / 15.0).clamp(min=0.0, max=1.0)

        nut = torch.zeros_like(k)
        nut = torch.where(regime == 1, nut_log * 0.3 * bw, nut)
        nut = torch.where(regime == 2, nut_log, nut)

        return nut.clamp(min=0.0)

    def compute_omega(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Adaptive omega computation."""
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        u_tau = self.compute_u_tau(k)
        y_plus = self.compute_y_plus(u_tau, y)
        regime = self._regime_state(y_plus)

        beta_1 = 0.075
        omega_visc = 6.0 * self.nu / (beta_1 * y.pow(2).clamp(min=1e-20))
        omega_log = torch.sqrt(k.clamp(min=1e-16)) / (
            self.C_mu ** 0.25 * self.kappa * y.clamp(min=1e-10)
        )

        # Buffer blending for omega
        bw = ((y_plus - 15.0) / 15.0).clamp(min=0.0, max=1.0)

        omega = torch.where(regime == 0, omega_visc,
                 torch.where(regime == 1, (1.0 - bw) * omega_visc + bw * omega_log * 0.5,
                 omega_log))

        return omega.clamp(min=1e-10)

    def compute_k(
        self,
        u_tau: torch.Tensor,
    ) -> torch.Tensor:
        """k = u_tau^2 / sqrt(C_mu)."""
        device = get_device()
        dtype = get_default_dtype()
        u_tau = u_tau.to(device=device, dtype=dtype)
        return u_tau.pow(2) / math.sqrt(self.C_mu)

    def compute_epsilon(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Adaptive epsilon computation."""
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        u_tau = self.compute_u_tau(k)
        y_plus = self.compute_y_plus(u_tau, y)
        regime = self._regime_state(y_plus)

        eps_visc = 2.0 * self.nu * k.clamp(min=1e-16) / y.pow(2).clamp(min=1e-20)
        eps_log = (
            self.C_mu ** 0.75
            * k.clamp(min=1e-16).pow(1.5)
            / (self.kappa * y.clamp(min=1e-10))
        )

        # Buffer blending
        bw = ((y_plus - 15.0) / 15.0).clamp(min=0.0, max=1.0)

        eps = torch.where(regime == 0, eps_visc,
               torch.where(regime == 1, (1.0 - bw) * eps_visc + bw * eps_log * 0.3,
               eps_log))

        return eps.clamp(min=1e-10)

    def reset_state(self) -> None:
        """Reset hysteresis state."""
        self._previous_state = None

    def __repr__(self) -> str:
        return f"AdaptiveWallTreatment(nu={self.nu})"
