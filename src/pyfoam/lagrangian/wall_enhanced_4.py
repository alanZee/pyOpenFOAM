"""
Enhanced wall interaction models v4.

Adds ThermalWallInteraction and WetWallInteraction following OpenFOAM conventions.

- :class:`ThermalWallInteraction` — wall interaction with heat transfer
- :class:`WetWallInteraction`     — interaction with wall film
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.wall_interaction import WallInteractionModel, _normalize, _dot

__all__ = ["ThermalWallInteraction", "WetWallInteraction"]


class ThermalWallInteraction(WallInteractionModel):
    """Wall interaction with heat transfer model.

    Models the thermal interaction when a particle contacts a wall,
    including convective heating/cooling and film formation at high
    wall temperatures.

    Parameters
    ----------
    restitution : float
        Mechanical restitution coefficient.  Default ``0.7``.
    wall_temperature : float
        Wall surface temperature (K).  Default ``500.0``.
    heat_transfer_coefficient : float
        Particle-wall heat transfer coefficient (W/(m²*K)).
        Default ``1000.0``.
    """

    def __init__(
        self,
        restitution: float = 0.7,
        wall_temperature: float = 500.0,
        heat_transfer_coefficient: float = 1000.0,
    ) -> None:
        self.restitution = restitution
        self.wall_temperature = wall_temperature
        self.heat_transfer_coefficient = heat_transfer_coefficient

    def interact(
        self,
        velocity: list[float],
        wall_normal: list[float],
    ) -> dict:
        """Compute thermal wall interaction."""
        n = _normalize(wall_normal)
        v_n = _dot(velocity, n)

        if v_n >= 0:
            return {
                "velocity": list(velocity),
                "stuck": False,
                "heat_transferred": 0.0,
                "wall_temperature": self.wall_temperature,
            }

        e = self.restitution
        new_v = [velocity[i] - (1.0 + e) * v_n * n[i] for i in range(3)]

        # 热传递估算: Q = h * A * dT (A = pi * d^2, dT ~ T_wall - T_p)
        # 使用默认 d=1e-4, T_p=300
        A_contact = math.pi * (1e-4) ** 2
        dT = self.wall_temperature - 300.0
        Q = self.heat_transfer_coefficient * A_contact * dT

        return {
            "velocity": new_v,
            "stuck": False,
            "heat_transferred": Q,
            "wall_temperature": self.wall_temperature,
        }


class WetWallInteraction(WallInteractionModel):
    """Interaction with a wet wall (wall film).

    Models particle interaction with a liquid film on the wall:

    1. **Absorption**: particle merges with the film (stuck)
    2. **Splash through**: high-energy particles penetrate the film

    Parameters
    ----------
    film_thickness : float
        Wall film thickness (m).  Default ``1e-5``.
    we_absorption : float
        Critical We for absorption.  Default ``10.0``.
    restitution : float
        Restitution for splash-through.  Default ``0.3``.
    surface_tension : float
        Film surface tension (N/m).  Default ``0.072``.
    """

    def __init__(
        self,
        film_thickness: float = 1e-5,
        we_absorption: float = 10.0,
        restitution: float = 0.3,
        surface_tension: float = 0.072,
    ) -> None:
        self.film_thickness = film_thickness
        self.we_absorption = we_absorption
        self.restitution = restitution
        self.surface_tension = surface_tension

    def interact(
        self,
        velocity: list[float],
        wall_normal: list[float],
    ) -> dict:
        """Compute wet wall interaction."""
        n = _normalize(wall_normal)
        v_n = _dot(velocity, n)

        if v_n >= 0:
            return {
                "velocity": list(velocity),
                "stuck": False,
                "absorbed": False,
                "film_mass_added": 0.0,
            }

        v_impact = abs(v_n)
        We = 1000.0 * v_impact ** 2 * 1e-4 / max(self.surface_tension, 1e-15)

        if We < self.we_absorption:
            # 被薄膜吸收
            m_particle = (math.pi / 6.0) * (1e-4) ** 3 * 1000.0
            return {
                "velocity": [0.0, 0.0, 0.0],
                "stuck": True,
                "absorbed": True,
                "film_mass_added": m_particle,
            }

        # 穿透薄膜
        e = self.restitution
        new_v = [velocity[i] - (1.0 + e) * v_n * n[i] for i in range(3)]
        return {
            "velocity": new_v,
            "stuck": False,
            "absorbed": False,
            "film_mass_added": 0.0,
        }
