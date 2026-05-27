"""
Oxidation models for Lagrangian particle tracking.

Models the heterogeneous oxidation of solid particles (e.g. char
combustion in coal-fired furnaces).  These models reduce the particle
diameter and mass over time through surface reactions.

Provides:

- :class:`OxidationModel`   — abstract base
- :class:`NoOxidation`      — no oxidation
- :class:`FieldOxidation`   — simple Arrhenius oxidation model

Usage::

    from pyfoam.lagrangian.oxidation import FieldOxidation

    model = FieldOxidation(pre_exponential=1.0, activation_energy=8e4)
    dm = model.oxidise(
        dt=1e-4,
        diameter=1e-3,
        temperature=1500.0,
        oxygen_mass_fraction=0.23,
        fluid_density=0.5,
    )
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


__all__ = [
    "OxidationModel",
    "NoOxidation",
    "FieldOxidation",
]


# ======================================================================
# 抽象基类
# ======================================================================

class OxidationModel(ABC):
    """Abstract base for Lagrangian oxidation models.

    Subclasses implement :meth:`oxidise`, which returns the mass lost
    by a single particle during one time step due to oxidation.
    """

    @abstractmethod
    def oxidise(
        self,
        dt: float,
        diameter: float,
        temperature: float,
        oxygen_mass_fraction: float = 0.23,
        fluid_density: float = 1.0,
    ) -> float:
        """Compute mass loss due to oxidation (kg).

        Parameters
        ----------
        dt : float
            Time step (s).
        diameter : float
            Current particle diameter (m).
        temperature : float
            Particle surface temperature (K).
        oxygen_mass_fraction : float
            Local O₂ mass fraction in the carrier phase (dimensionless).
        fluid_density : float
            Carrier-phase density (kg/m³).

        Returns
        -------
        float
            Mass lost by the particle during this time step (kg). Always
            non-negative; zero means no oxidation.
        """


# ======================================================================
# 无氧化
# ======================================================================

class NoOxidation(OxidationModel):
    """No oxidation.

    Always returns zero mass loss.
    """

    def oxidise(
        self,
        dt: float,
        diameter: float,
        temperature: float,
        oxygen_mass_fraction: float = 0.23,
        fluid_density: float = 1.0,
    ) -> float:
        """Return zero mass loss."""
        return 0.0


# ======================================================================
# Arrhenius 氧化模型
# ======================================================================

# 通用气体常数 (J/(mol·K))
_UNIVERSAL_GAS_CONSTANT = 8.314


class FieldOxidation(OxidationModel):
    """Simple Arrhenius surface oxidation model.

    Models heterogeneous oxidation using an Arrhenius rate expression
    for the surface reaction rate:

    .. math::

        k_s = A \\exp\\left(-\\frac{E_a}{R T}\\right)

    The mass loss rate per particle is:

    .. math::

        \\dot{m} = \\pi d^2 \\, k_s \\, \\rho_f \\, Y_{O_2}

    where:

    - :math:`A` is the pre-exponential factor (m/s)
    - :math:`E_a` is the activation energy (J/mol)
    - :math:`R` is the universal gas constant
    - :math:`T` is the particle surface temperature (K)
    - :math:`d` is the particle diameter (m)
    - :math:`\\rho_f` is the carrier-phase density (kg/m³)
    - :math:`Y_{O_2}` is the local O₂ mass fraction

    Parameters
    ----------
    pre_exponential : float
        Pre-exponential factor :math:`A` (m/s).  Default ``1.0``.
    activation_energy : float
        Activation energy :math:`E_a` (J/mol).  Default ``8.0e4``
        (typical for char oxidation).
    """

    def __init__(
        self,
        pre_exponential: float = 1.0,
        activation_energy: float = 8.0e4,
    ) -> None:
        if pre_exponential < 0:
            raise ValueError(
                f"pre_exponential must be non-negative, got {pre_exponential}"
            )
        if activation_energy < 0:
            raise ValueError(
                f"activation_energy must be non-negative, got {activation_energy}"
            )
        self.pre_exponential = pre_exponential
        self.activation_energy = activation_energy

    def oxidise(
        self,
        dt: float,
        diameter: float,
        temperature: float,
        oxygen_mass_fraction: float = 0.23,
        fluid_density: float = 1.0,
    ) -> float:
        """Compute mass loss via Arrhenius surface oxidation.

        Returns ``0.0`` when the diameter is negligible, the oxygen
        concentration is negligible, or the temperature is too low for
        any meaningful reaction rate.
        """
        if diameter < 1e-15:
            return 0.0

        if oxygen_mass_fraction < 1e-15:
            return 0.0

        if temperature < 1e-15:
            return 0.0

        # Arrhenius 反应速率
        k_s = self.pre_exponential * math.exp(
            -self.activation_energy / (_UNIVERSAL_GAS_CONSTANT * temperature)
        )

        if k_s < 1e-30:
            return 0.0

        # 表面积
        surface_area = math.pi * diameter ** 2

        # 质量损失速率: dm/dt = pi * d^2 * k_s * rho_f * Y_O2
        mdot = surface_area * k_s * fluid_density * oxygen_mass_fraction

        # 限制质量损失不超过当前粒子质量（假设碳密度 2000 kg/m³）
        m_particle = (math.pi / 6.0) * diameter ** 3 * 2000.0
        dm = mdot * dt
        dm = min(dm, m_particle)

        return max(dm, 0.0)

    def __repr__(self) -> str:
        return (
            f"FieldOxidation(A={self.pre_exponential}, "
            f"Ea={self.activation_energy})"
        )
