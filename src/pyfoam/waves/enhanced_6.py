"""
Enhanced wave models v6 — active and passive wave absorption.

Extends :class:`~pyfoam.waves.enhanced_5.AbsorptionModel` with:

- :class:`ActiveAbsorption` — active wave absorption using measured surface elevation
- :class:`PassiveAbsorption` — passive absorption via numerical sponge layer

References:
    OpenFOAM ``waveModels::activeAbsorption``
    OpenFOAM ``waveModels::passiveAbsorption``
    Schaffer & Klopman (2000). "Review of multidirectional active wave absorption methods."

Usage::

    from pyfoam.waves.enhanced_6 import ActiveAbsorption, PassiveAbsorption

    model = ActiveAbsorption(zone_length=20.0, depth=10.0, gain=0.5)
    eta_abs, u_abs, w_abs = model.absorb(eta, u, w, x, x_zone_start=80.0)
"""

from __future__ import annotations

import math

import torch

from pyfoam.waves.enhanced_5 import AbsorptionModel

__all__ = ["ActiveAbsorption", "PassiveAbsorption"]


# ---------------------------------------------------------------------------
# ActiveAbsorption
# ---------------------------------------------------------------------------

@AbsorptionModel.register("active")
class ActiveAbsorption(AbsorptionModel):
    """Active wave absorption using measured surface elevation.

    Active absorption subtracts the incident wave component from the
    computed wave field within the relaxation zone by measuring the
    surface elevation and generating a cancelling wave.

    The absorption signal is:
        eta_abs = eta - gain * eta_measured * weight(s)

    where weight(s) is the relaxation function (0 at zone start, 1 at end).

    Parameters
    ----------
    zone_length : float
        Length of the absorption zone (m).
    depth : float
        Water depth d (m).
    gain : float
        Absorption gain factor in (0, 1] (default 1.0 = perfect absorption).
    """

    def __init__(
        self,
        zone_length: float,
        depth: float,
        *,
        gain: float = 1.0,
    ) -> None:
        super().__init__(zone_length, depth)
        self._gain = max(0.0, min(1.0, gain))

    @property
    def gain(self) -> float:
        """返回吸收增益系数。"""
        return self._gain

    def absorb(
        self,
        eta: torch.Tensor,
        u: torch.Tensor,
        w: torch.Tensor,
        x: torch.Tensor,
        x_zone_start: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply active absorption within the relaxation zone.

        Subtracts the measured surface elevation weighted by the
        relaxation function and absorption gain:

            weight = relaxation_weight(x, x_zone_start)
            eta_abs = eta - gain * eta * weight
            u_abs = u * (1 - gain * weight)
            w_abs = w * (1 - gain * weight)

        Parameters
        ----------
        eta : torch.Tensor
            Wave elevation field.
        u : torch.Tensor
            Horizontal velocity field.
        w : torch.Tensor
            Vertical velocity field.
        x : torch.Tensor
            Position of each point (m).
        x_zone_start : float
            x-coordinate where the absorption zone starts.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Absorbed (eta, u, w) fields.
        """
        wt = self.relaxation_weight(x, x_zone_start)
        factor = 1.0 - self._gain * wt

        eta_abs = eta * factor
        u_abs = u * factor
        w_abs = w * factor

        return eta_abs, u_abs, w_abs

    def __repr__(self) -> str:
        return (
            f"ActiveAbsorption(L_zone={self._zone_length}, "
            f"d={self._depth}, gain={self._gain})"
        )


# ---------------------------------------------------------------------------
# PassiveAbsorption
# ---------------------------------------------------------------------------

@AbsorptionModel.register("passive")
class PassiveAbsorption(AbsorptionModel):
    """Passive wave absorption via numerical sponge layer.

    Applies a damping coefficient that increases through the relaxation
    zone to suppress wave energy. Unlike active absorption, passive
    absorption does not require wave measurement.

    Damping profile:
        sigma(s) = sigma_max * s^damping_power

    where s is the normalized position in [0, 1] within the zone,
    sigma_max is the maximum damping coefficient, and damping_power
    controls the profile shape (default 2 = quadratic ramp).

    The damping is applied as:
        eta_abs = eta * exp(-sigma * dt)
        u_abs = u * exp(-sigma * dt)
        w_abs = w * exp(-sigma * dt)

    For steady-state evaluation (dt=1):
        eta_abs = eta * exp(-sigma)

    Parameters
    ----------
    zone_length : float
        Length of the absorption zone (m).
    depth : float
        Water depth d (m).
    sigma_max : float
        Maximum damping coefficient (default 5.0, 1/s).
    damping_power : float
        Power law exponent for damping ramp (default 2.0).
    """

    def __init__(
        self,
        zone_length: float,
        depth: float,
        *,
        sigma_max: float = 5.0,
        damping_power: float = 2.0,
    ) -> None:
        super().__init__(zone_length, depth)
        self._sigma_max = sigma_max
        self._damping_power = damping_power

    @property
    def sigma_max(self) -> float:
        """返回最大阻尼系数。"""
        return self._sigma_max

    @property
    def damping_power(self) -> float:
        """返回阻尼幂律指数。"""
        return self._damping_power

    def damping_coefficient(self, x: torch.Tensor, x_zone_start: float) -> torch.Tensor:
        """Compute position-dependent damping coefficient.

        sigma(x) = sigma_max * s^damping_power

        Parameters
        ----------
        x : torch.Tensor
            Position of each point.
        x_zone_start : float
            Start of the absorption zone.

        Returns
        -------
        torch.Tensor
            Damping coefficient at each position (1/s).
        """
        L = self._zone_length
        s = ((x - x_zone_start) / L).clamp(0.0, 1.0)
        return self._sigma_max * s.pow(self._damping_power)

    def absorb(
        self,
        eta: torch.Tensor,
        u: torch.Tensor,
        w: torch.Tensor,
        x: torch.Tensor,
        x_zone_start: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply passive absorption (sponge layer) within the relaxation zone.

        Applies exponential damping:
            damping = exp(-sigma(x))
            eta_abs = eta * damping
            u_abs = u * damping
            w_abs = w * damping

        Parameters
        ----------
        eta : torch.Tensor
            Wave elevation field.
        u : torch.Tensor
            Horizontal velocity field.
        w : torch.Tensor
            Vertical velocity field.
        x : torch.Tensor
            Position of each point (m).
        x_zone_start : float
            x-coordinate where the absorption zone starts.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Absorbed (eta, u, w) fields.
        """
        sigma = self.damping_coefficient(x, x_zone_start)
        damping = torch.exp(-sigma)

        eta_abs = eta * damping
        u_abs = u * damping
        w_abs = w * damping

        return eta_abs, u_abs, w_abs

    def __repr__(self) -> str:
        return (
            f"PassiveAbsorption(L_zone={self._zone_length}, "
            f"d={self._depth}, sigma_max={self._sigma_max})"
        )
