"""
Enhanced MPPIC models v5.

Adds MinimumMassLimiter and MaximumTemperatureLimiter following OpenFOAM conventions.

- :class:`MinimumMassLimiter`       — minimum particle mass limiter
- :class:`MaximumTemperatureLimiter` — maximum temperature limiter
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.mppic_models import MPPICModel

__all__ = ["MinimumMassLimiter", "MaximumTemperatureLimiter"]


class MinimumMassLimiter(MPPICModel):
    """Minimum particle mass limiter for MPPIC.

    Prevents particles from becoming unphysically small by applying
    a packing stress that increases as the particle mass approaches
    the minimum threshold.  Particles below the threshold are removed.

    Parameters
    ----------
    min_mass : float
        Minimum particle mass (kg).  Default ``1e-15``.
    p0 : float
        Reference stress (Pa).  Default ``1000.0``.
    exponent : float
        Stress exponent near the limit.  Default ``2.0``.
    """

    def __init__(
        self,
        min_mass: float = 1e-15,
        p0: float = 1000.0,
        exponent: float = 2.0,
    ) -> None:
        self.min_mass = min_mass
        self.p0 = p0
        self.exponent = exponent

    def packing_stress(
        self,
        alpha: float,
        particle_density: float = 1000.0,
    ) -> float:
        """Compute mass-limiting stress."""
        if alpha <= 0.0:
            return 0.0
        return self.p0 * alpha ** self.exponent

    def should_remove(self, mass: float) -> bool:
        """Check if particle mass is below minimum threshold."""
        return mass < self.min_mass


class MaximumTemperatureLimiter(MPPICModel):
    """Maximum temperature limiter for MPPIC.

    Applies thermal resistance stress when particle temperature
    approaches the material maximum to prevent unphysical temperatures.

    Parameters
    ----------
    max_temperature : float
        Maximum allowed temperature (K).  Default ``2000.0``.
    p0 : float
        Reference stress (Pa).  Default ``1000.0``.
    onset_temperature : float
        Temperature at which the limiter activates (K).  Default ``1500.0``.
    """

    def __init__(
        self,
        max_temperature: float = 2000.0,
        p0: float = 1000.0,
        onset_temperature: float = 1500.0,
    ) -> None:
        if onset_temperature >= max_temperature:
            raise ValueError("onset_temperature must be < max_temperature")
        self.max_temperature = max_temperature
        self.p0 = p0
        self.onset_temperature = onset_temperature

    def packing_stress(
        self,
        alpha: float,
        particle_density: float = 1000.0,
    ) -> float:
        """Compute temperature-limiting stress."""
        if alpha <= 0.0:
            return 0.0
        return self.p0 * alpha

    def thermal_stress(
        self,
        temperature: float,
        alpha: float,
    ) -> float:
        """Compute thermal resistance stress."""
        if temperature <= self.onset_temperature:
            return 0.0

        T_ratio = (temperature - self.onset_temperature) / max(
            self.max_temperature - self.onset_temperature, 1.0
        )
        T_ratio = min(T_ratio, 1.0)

        return self.p0 * T_ratio ** 2 * max(alpha, 0.0)

    def limit_temperature(
        self,
        temperature: float,
        alpha: float,
    ) -> float:
        """Clamp temperature to maximum allowed value."""
        return min(temperature, self.max_temperature)
