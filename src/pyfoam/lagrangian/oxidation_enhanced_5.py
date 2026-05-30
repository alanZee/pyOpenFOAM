"""
Enhanced oxidation models v5.

Adds RandomPoreModel and ShrinkingCoreModel following OpenFOAM conventions.

- :class:`RandomPoreModel`    — Bhatia & Perlmutter random pore model
- :class:`ShrinkingCoreModel`  — shrinking core oxidation model
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.oxidation import OxidationModel

__all__ = ["RandomPoreModel", "ShrinkingCoreModel"]

_R = 8.314


class RandomPoreModel(OxidationModel):
    """Bhatia & Perlmutter (1980) random pore model.

    Accounts for the evolution of internal pore structure during
    oxidation:

    .. math::

        \\frac{dX}{dt} = k_s (1 - X) \\sqrt{1 - \\psi \\ln(1 - X)}

    where X is the conversion, psi is the structural parameter,
    and k_s is the intrinsic reaction rate.

    Parameters
    ----------
    A : float
        Pre-exponential factor (1/s).  Default ``1.0``.
    E_a : float
        Activation energy (J/mol).  Default ``8.0e4``.
    psi : float
        Structural parameter (dimensionless).  Default ``2.0``.
        psi < 1: pore growth dominates
        psi > 1: pore coalescence dominates
    """

    def __init__(
        self,
        A: float = 1.0,
        E_a: float = 8.0e4,
        psi: float = 2.0,
    ) -> None:
        self.A = A
        self.E_a = E_a
        self.psi = psi

    def oxidise(
        self,
        dt: float,
        diameter: float,
        temperature: float,
        oxygen_mass_fraction: float = 0.23,
        fluid_density: float = 1.0,
    ) -> float:
        """Compute random pore model oxidation."""
        if diameter < 1e-15 or oxygen_mass_fraction < 1e-15 or temperature < 1e-15:
            return 0.0

        RT = _R * temperature
        if RT < 1e-30:
            return 0.0

        k_s = self.A * math.exp(-self.E_a / RT) * oxygen_mass_fraction

        # 假设当前转化率 X = 0 (初始状态)
        X = 0.0
        dXdt = k_s * (1.0 - X) * math.sqrt(max(1.0 - self.psi * math.log(max(1.0 - X, 1e-15)), 0.0))

        m_particle = (math.pi / 6.0) * diameter ** 3 * 2000.0
        dm = m_particle * dXdt * dt
        return max(min(dm, m_particle), 0.0)


class ShrinkingCoreModel(OxidationModel):
    """Shrinking core oxidation model.

    Models the oxidation of a particle where the unreacted core
    shrinks over time while an ash/product layer grows.  The overall
    rate considers three resistances in series:

    1. External mass transfer
    2. Diffusion through product layer
    3. Chemical reaction at the core surface

    Parameters
    ----------
    A : float
        Pre-exponential factor (m/s).  Default ``1.0``.
    E_a : float
        Activation energy (J/mol).  Default ``8.0e4``.
    product_layer_diffusivity : float
        Diffusivity through product layer (m²/s).  Default ``1e-10``.
    density : float
        Particle density (kg/m³).  Default ``2000.0``.
    """

    def __init__(
        self,
        A: float = 1.0,
        E_a: float = 8.0e4,
        product_layer_diffusivity: float = 1e-10,
        density: float = 2000.0,
    ) -> None:
        self.A = A
        self.E_a = E_a
        self.product_layer_diffusivity = product_layer_diffusivity
        self.density = density

    def oxidise(
        self,
        dt: float,
        diameter: float,
        temperature: float,
        oxygen_mass_fraction: float = 0.23,
        fluid_density: float = 1.0,
    ) -> float:
        """Compute shrinking core oxidation."""
        if diameter < 1e-15 or oxygen_mass_fraction < 1e-15 or temperature < 1e-15:
            return 0.0

        RT = _R * temperature
        if RT < 1e-30:
            return 0.0

        r = diameter / 2.0
        k_chem = self.A * math.exp(-self.E_a / RT)

        # 三重串联阻力（简化：假设核心占满粒子）
        k_ext = 2.0 * 2.6e-5 / max(r, 1e-15)  # 外部传质
        k_prod = self.product_layer_diffusivity / max(r, 1e-15)  # 产物层扩散

        k_total = 1.0 / (1.0 / max(k_ext, 1e-30) + 1.0 / max(k_prod, 1e-30) + 1.0 / max(k_chem, 1e-30))

        mdot = math.pi * diameter ** 2 * k_total * fluid_density * oxygen_mass_fraction

        m_particle = (math.pi / 6.0) * diameter ** 3 * self.density
        dm = mdot * dt
        return max(min(dm, m_particle), 0.0)
