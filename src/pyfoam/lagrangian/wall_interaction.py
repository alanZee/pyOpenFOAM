"""
Wall interaction models for Lagrangian particle tracking.

Models the behaviour of particles when they collide with wall
boundaries.  These models determine whether a particle bounces off
the wall, sticks to it, or undergoes some other interaction.

Provides:

- :class:`WallInteractionModel` — abstract base
- :class:`ElasticBounce`       — elastic wall bounce with restitution
- :class:`Stick`               — particle sticks to wall
- :class:`SplashModel`         — particle splash on wall impact

Usage::

    from pyfoam.lagrangian.wall_interaction import ElasticBounce

    model = ElasticBounce(restitution=0.8)
    result = model.interact(
        velocity=[5.0, -3.0, 0.0],
        wall_normal=[0.0, 1.0, 0.0],
    )
    # result["velocity"] — post-interaction velocity
    # result["stuck"]    — False (particle bounced)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


__all__ = [
    "WallInteractionModel",
    "ElasticBounce",
    "Stick",
    "SplashModel",
]


# ======================================================================
# 辅助函数
# ======================================================================

def _dot(a: list[float], b: list[float]) -> float:
    """3-D dot product."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _normalize(v: list[float]) -> list[float]:
    """Return the unit vector of *v*.  Returns [0,0,0] for zero vector."""
    mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if mag < 1e-30:
        return [0.0, 0.0, 0.0]
    return [v[0] / mag, v[1] / mag, v[2] / mag]


# ======================================================================
# 抽象基类
# ======================================================================

class WallInteractionModel(ABC):
    """Abstract base for Lagrangian wall interaction models.

    Subclasses implement :meth:`interact`, which computes the
    post-interaction velocity and whether the particle has stuck
    to the wall.
    """

    @abstractmethod
    def interact(
        self,
        velocity: list[float],
        wall_normal: list[float],
    ) -> dict:
        """Compute post-wall-interaction state.

        Parameters
        ----------
        velocity : list[float]
            Particle velocity ``[u, v, w]`` before the wall interaction (m/s).
        wall_normal : list[float]
            Outward-pointing unit normal of the wall surface ``[nx, ny, nz]``.
            Points from the wall into the fluid domain.

        Returns
        -------
        dict
            ``{"velocity": list[float], "stuck": bool}`` — post-interaction
            velocity and whether the particle is stuck to the wall.
        """


# ======================================================================
# 弹性反弹
# ======================================================================

class ElasticBounce(WallInteractionModel):
    """Elastic wall bounce with a normal restitution coefficient.

    The velocity component normal to the wall is reflected and scaled
    by the restitution coefficient, while the tangential component is
    preserved:

    .. math::

        v'_n = -e \\, (v \\cdot \\hat{n})

        v' = v - (1 + e) \\, (v \\cdot \\hat{n}) \\, \\hat{n}

    where :math:`e \\in [0, 1]` is the normal restitution coefficient
    (``1`` = perfectly elastic, ``0`` = perfectly inelastic / normal
    component fully absorbed).

    Parameters
    ----------
    restitution : float
        Normal restitution coefficient ``e in [0, 1]``.  Default ``1.0``.
    """

    def __init__(self, restitution: float = 1.0) -> None:
        if not 0.0 <= restitution <= 1.0:
            raise ValueError(
                f"restitution must be in [0, 1], got {restitution}"
            )
        self.restitution = restitution

    def interact(
        self,
        velocity: list[float],
        wall_normal: list[float],
    ) -> dict:
        """Bounce off the wall with normal restitution.

        If the particle is moving away from the wall (normal component
        of velocity is non-negative), the velocity is returned unchanged.
        """
        n = _normalize(wall_normal)
        n_mag = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        if n_mag < 1e-15:
            return {"velocity": list(velocity), "stuck": False}

        v_n = _dot(velocity, n)

        # 粒子正在远离壁面，不处理
        if v_n >= 0:
            return {"velocity": list(velocity), "stuck": False}

        # 完全非弹性 (e=0): 法向分量归零
        # 部分弹性: v' = v - (1+e) * (v·n) * n
        e = self.restitution
        new_velocity = [
            velocity[0] - (1.0 + e) * v_n * n[0],
            velocity[1] - (1.0 + e) * v_n * n[1],
            velocity[2] - (1.0 + e) * v_n * n[2],
        ]

        return {"velocity": new_velocity, "stuck": False}

    def __repr__(self) -> str:
        return f"ElasticBounce(restitution={self.restitution})"


# ======================================================================
# 粘附
# ======================================================================

class Stick(WallInteractionModel):
    """Particle sticks to the wall.

    The particle velocity is set to zero and the particle is marked as
    stuck.  This models situations such as wet droplets impinging on a
    surface or particles depositing on a wall.

    A minimum approach speed can be specified; if the particle's
    normal-approach speed is below this threshold, the interaction is
    ignored (the particle is just passing near the wall, not actually
    hitting it).
    """

    def __init__(self, min_approach_speed: float = 0.0) -> None:
        if min_approach_speed < 0:
            raise ValueError(
                f"min_approach_speed must be non-negative, "
                f"got {min_approach_speed}"
            )
        self.min_approach_speed = min_approach_speed

    def interact(
        self,
        velocity: list[float],
        wall_normal: list[float],
    ) -> dict:
        """Stick to the wall — velocity set to zero.

        If the particle is moving away from the wall (normal component
        of velocity is non-negative), the velocity is returned unchanged.
        If *min_approach_speed* is set, the particle must be approaching
        faster than this threshold to stick.
        """
        n = _normalize(wall_normal)
        n_mag = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        if n_mag < 1e-15:
            return {"velocity": list(velocity), "stuck": False}

        v_n = _dot(velocity, n)

        # 粒子正在远离壁面
        if v_n >= 0:
            return {"velocity": list(velocity), "stuck": False}

        # 检查是否满足最小接近速度
        approach_speed = abs(v_n)
        if approach_speed < self.min_approach_speed:
            return {"velocity": list(velocity), "stuck": False}

        # 粘附: 速度归零
        return {"velocity": [0.0, 0.0, 0.0], "stuck": True}

    def __repr__(self) -> str:
        return f"Stick(min_approach_speed={self.min_approach_speed})"


# ======================================================================
# 飞溅模型
# ======================================================================

class SplashModel(WallInteractionModel):
    """Particle splash on wall impact.

    Models the outcome of a droplet hitting a wall based on the impact
    Weber number:

    .. math::

        We_{impact} = \\frac{\\rho_d \\, v_n^2 \\, d}{\\sigma}

    The interaction has three regimes:

    1. **Stick** (:math:`We_{impact} < We_{stick}`): particle adheres
       to the wall (velocity zeroed, ``stuck = True``).
    2. **Bounce** (:math:`We_{stick} \\le We_{impact} < We_{splash}`):
       particle reflects off the wall with normal restitution.
    3. **Splash** (:math:`We_{impact} \\ge We_{splash}`): particle
       shatters — the post-interaction velocity is reduced (energy
       absorbed by the splash) and a *fragment_diameter* is computed.

    Parameters
    ----------
    we_stick : float
        Critical Weber number below which the particle sticks.
        Default ``5.0``.
    we_splash : float
        Critical Weber number above which splash occurs.
        Default ``80.0``.
    restitution : float
        Normal restitution coefficient for the bounce regime.
        Default ``0.5``.
    splash_absorption : float
        Fraction of kinetic energy absorbed during splash (0–1).
        Default ``0.3``.
    fragment_ratio : float
        Ratio of fragment diameter to original diameter during splash.
        Default ``0.6``.
    surface_tension : float
        Liquid-gas surface tension (N/m) for Weber number calculation.
        Default ``0.072`` (water at 20°C).
    particle_density : float
        Droplet material density (kg/m³). Default ``1000.0``.
    diameter : float
        Particle diameter (m) for Weber number calculation.
        Default ``1e-4``.
    """

    def __init__(
        self,
        we_stick: float = 5.0,
        we_splash: float = 80.0,
        restitution: float = 0.5,
        splash_absorption: float = 0.3,
        fragment_ratio: float = 0.6,
        surface_tension: float = 0.072,
        particle_density: float = 1000.0,
        diameter: float = 1e-4,
    ) -> None:
        if we_stick < 0:
            raise ValueError(f"we_stick must be non-negative, got {we_stick}")
        if we_splash <= we_stick:
            raise ValueError(
                f"we_splash ({we_splash}) must be > we_stick ({we_stick})."
            )
        if not 0.0 <= restitution <= 1.0:
            raise ValueError(
                f"restitution must be in [0, 1], got {restitution}"
            )
        if not 0.0 <= splash_absorption <= 1.0:
            raise ValueError(
                f"splash_absorption must be in [0, 1], got {splash_absorption}"
            )
        if not 0.0 < fragment_ratio <= 1.0:
            raise ValueError(
                f"fragment_ratio must be in (0, 1], got {fragment_ratio}"
            )
        if surface_tension <= 0:
            raise ValueError(
                f"surface_tension must be positive, got {surface_tension}"
            )

        self.we_stick = we_stick
        self.we_splash = we_splash
        self.restitution = restitution
        self.splash_absorption = splash_absorption
        self.fragment_ratio = fragment_ratio
        self.surface_tension = surface_tension
        self.particle_density = particle_density
        self.diameter = diameter

    def interact(
        self,
        velocity: list[float],
        wall_normal: list[float],
    ) -> dict:
        """Compute post-wall-interaction state with splash.

        Parameters
        ----------
        velocity : list[float]
            Particle velocity ``[u, v, w]`` before interaction (m/s).
        wall_normal : list[float]
            Outward unit normal of the wall surface.

        Returns
        -------
        dict
            ``{"velocity": list[float], "stuck": bool, "splashed": bool,
            "fragment_diameter": float | None}``
        """
        n = _normalize(wall_normal)
        n_mag = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        if n_mag < 1e-15:
            return {
                "velocity": list(velocity),
                "stuck": False,
                "splashed": False,
                "fragment_diameter": None,
            }

        v_n = _dot(velocity, n)

        # 粒子正在远离壁面
        if v_n >= 0:
            return {
                "velocity": list(velocity),
                "stuck": False,
                "splashed": False,
                "fragment_diameter": None,
            }

        # 冲击法向速度
        v_impact = abs(v_n)

        # 冲击 Weber 数: We = rho_d * v_n^2 * d / sigma
        We = (
            self.particle_density * v_impact ** 2 * self.diameter
            / self.surface_tension
        )

        # 粘附区: We < we_stick
        if We < self.we_stick:
            return {
                "velocity": [0.0, 0.0, 0.0],
                "stuck": True,
                "splashed": False,
                "fragment_diameter": None,
            }

        # 反弹区: we_stick <= We < we_splash
        if We < self.we_splash:
            e = self.restitution
            new_velocity = [
                velocity[i] - (1.0 + e) * v_n * n[i]
                for i in range(3)
            ]
            return {
                "velocity": new_velocity,
                "stuck": False,
                "splashed": False,
                "fragment_diameter": None,
            }

        # 飞溅区: We >= we_splash
        # 动能被部分吸收，速度衰减
        factor = math.sqrt(1.0 - self.splash_absorption)
        new_velocity = [
            velocity[i] - (1.0 + factor) * v_n * n[i]
            for i in range(3)
        ]
        fragment_diameter = self.diameter * self.fragment_ratio

        return {
            "velocity": new_velocity,
            "stuck": False,
            "splashed": True,
            "fragment_diameter": fragment_diameter,
        }

    def __repr__(self) -> str:
        return (
            f"SplashModel(we_stick={self.we_stick}, "
            f"we_splash={self.we_splash})"
        )
