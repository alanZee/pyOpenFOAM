"""Enhanced wall treatment v5 with conjugate heat transfer and adaptive y+ tracking.

Extends EnhancedWallTreatment4 with:
- Conjugate heat transfer interface for solid-fluid coupling
- Adaptive y+ tracking with time-series analysis
- Automatic wall function transition based on y+ evolution

Usage::

    from pyfoam.turbulence.wall_treatment_enhanced_5 import EnhancedWallTreatment5
    wt = EnhancedWallTreatment5(nu=1.5e-5, Pr=0.71, k_solid=50.0)
"""

from __future__ import annotations
import logging
import math
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.turbulence.wall_treatment import WallTreatment
from pyfoam.turbulence.wall_treatment_enhanced_4 import (
    EnhancedWallTreatment4, CompressibleWallTreatment,
)

__all__ = ["EnhancedWallTreatment5", "ConjugateHeatTransfer"]

logger = logging.getLogger(__name__)


@WallTreatment.register("enhanced5")
class EnhancedWallTreatment5(EnhancedWallTreatment4):
    """Enhanced wall treatment v5 with conjugate HT and adaptive y+ tracking.

    Extends EnhancedWallTreatment4 with:

    - **Conjugate heat transfer**: interfaces with solid-side thermal model
      for coupled fluid-solid wall temperature prediction.
    - **Adaptive y+ tracking**: monitors y+ history and triggers wall function
      transitions when y+ drifts outside the optimal range.
    - **Time-series y+ prediction**: uses exponential moving average to
      predict next-step y+ for proactive adjustment.

    Parameters
    ----------
    nu : float
        Molecular kinematic viscosity (m^2/s).
    kappa, E, C_mu : float
        Standard wall function constants.
    y_plus_low, y_plus_high : float
        Low/high y+ region bounds.
    ks : float
        Equivalent sand-grain roughness height (m).
    Pr, Pr_t : float
        Prandtl numbers.
    hysteresis_width : float
        Width of y+ hysteresis band.
    Le : float
        Lewis number.
    van_driest_A : float
        Van Driest damping constant.
    k_solid : float
        Solid-side thermal conductivity (W/(m*K)). Default 50.0.
    solid_thickness : float
        Solid wall thickness (m). Default 0.001.
    y_plus_ema_alpha : float
        EMA smoothing factor for y+ tracking. Default 0.3.
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
        Le: float = 1.0,
        van_driest_A: float = 26.0,
        k_solid: float = 50.0,
        solid_thickness: float = 0.001,
        y_plus_ema_alpha: float = 0.3,
    ) -> None:
        super().__init__(
            nu=nu, kappa=kappa, E=E, C_mu=C_mu,
            y_plus_low=y_plus_low, y_plus_high=y_plus_high,
            ks=ks, Pr=Pr, Pr_t=Pr_t, hysteresis_width=hysteresis_width,
            Le=Le, van_driest_A=van_driest_A,
        )
        self._k_solid = max(k_solid, 0.01)
        self._solid_thickness = max(solid_thickness, 1e-6)
        self._ema_alpha = max(0.01, min(y_plus_ema_alpha, 1.0))
        self._y_plus_history: list[float] = []
        self._y_plus_ema: float = 0.0

    @property
    def k_solid(self) -> float:
        return self._k_solid

    @property
    def y_plus_ema(self) -> float:
        """Exponential moving average of y+."""
        return self._y_plus_ema

    def conjugate_wall_temperature(
        self,
        T_fluid: float,
        htc_fluid: float,
        T_solid_interior: float,
    ) -> float:
        """Compute conjugate wall temperature.

        T_wall = (htc * T_fluid + k_solid/thickness * T_solid)
                 / (htc + k_solid/thickness)
        """
        h_solid = self._k_solid / self._solid_thickness
        denom = htc_fluid + h_solid
        if abs(denom) < 1e-30:
            return 0.5 * (T_fluid + T_solid_interior)
        return (htc_fluid * T_fluid + h_solid * T_solid_interior) / denom

    def update_y_plus_tracking(self, y_plus_mean: float) -> None:
        """Update y+ tracking with new measurement."""
        self._y_plus_history.append(y_plus_mean)
        if len(self._y_plus_history) > 1000:
            self._y_plus_history = self._y_plus_history[-500:]
        self._y_plus_ema = (
            self._ema_alpha * y_plus_mean
            + (1.0 - self._ema_alpha) * self._y_plus_ema
        )

    def predict_y_plus(self) -> float:
        """Predict next y+ using EMA."""
        return self._y_plus_ema

    def should_switch_wall_function(self) -> str | None:
        """Check if wall function should be switched."""
        yp = self._y_plus_ema
        if yp < 1.0:
            return "lowRe"
        elif yp < 5.0:
            return "enhancedWallTreatment"
        elif yp < 30.0:
            return "standardWallFunction"
        elif yp > 200.0:
            return "scalableWallFunction"
        return None

    def __repr__(self) -> str:
        return (
            f"EnhancedWallTreatment5(nu={self.nu}, Pr={self._Pr}, "
            f"Le={self._Le}, k_solid={self._k_solid})"
        )


class ConjugateHeatTransfer:
    """Standalone conjugate heat transfer calculator.

    Parameters
    ----------
    k_fluid : float
        Fluid thermal conductivity (W/(m*K)). Default 0.026.
    k_solid : float
        Solid thermal conductivity (W/(m*K)). Default 50.0.
    thickness : float
        Wall thickness (m). Default 0.001.
    """

    def __init__(
        self,
        k_fluid: float = 0.026,
        k_solid: float = 50.0,
        thickness: float = 0.001,
    ) -> None:
        self._k_fluid = max(k_fluid, 1e-10)
        self._k_solid = max(k_solid, 1e-10)
        self._thickness = max(thickness, 1e-6)

    def overall_htc(self, h_fluid: float) -> float:
        """Compute overall heat transfer coefficient.

        U = 1 / (1/h_fluid + thickness/k_solid)
        """
        R_fluid = 1.0 / max(h_fluid, 1e-10)
        R_solid = self._thickness / self._k_solid
        return 1.0 / max(R_fluid + R_solid, 1e-30)

    def __repr__(self) -> str:
        return f"ConjugateHeatTransfer(k_fluid={self._k_fluid}, k_solid={self._k_solid})"
