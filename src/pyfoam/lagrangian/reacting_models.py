"""
Reacting particle models for Lagrangian particle tracking.

模拟颗粒在高温环境中的化学反应过程，包括：
- 单相反应：颗粒与气相的异相反应（如煤粉燃烧）
- 多相反应：颗粒中多种组分参与的复杂反应

Provides:

- :class:`ReactingModel`       — abstract base
- :class:`SinglePhaseReacting` — single phase (heterogeneous) reaction
- :class:`MultiPhaseReacting`  — multi-component reaction model

Usage::

    from pyfoam.lagrangian.reacting_models import SinglePhaseReacting

    model = SinglePhaseReacting(
        A=1.0e3, activation_energy=1.2e5,
        heat_of_reaction=-2.0e7, diffusivity=2.6e-5,
    )
    result = model.react(
        dt=1e-4,
        diameter=1e-4,
        temperature=1500.0,
        fluid_temperature=2000.0,
        species_mass_fraction=0.8,
    )
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


__all__ = [
    "ReactingModel",
    "SinglePhaseReacting",
    "MultiPhaseReacting",
]


# ======================================================================
# 抽象基类
# ======================================================================

class ReactingModel(ABC):
    """Abstract base for Lagrangian reacting particle models.

    Subclasses implement :meth:`react`, which computes the mass loss,
    heat release, and post-reaction particle state during one time step.
    """

    @abstractmethod
    def react(
        self,
        dt: float,
        diameter: float,
        temperature: float,
        fluid_temperature: float,
        species_mass_fraction: float = 1.0,
    ) -> dict:
        """Compute reaction rates during one time step.

        Parameters
        ----------
        dt : float
            Time step (s).
        diameter : float
            Current particle diameter (m).
        temperature : float
            Particle temperature (K).
        fluid_temperature : float
            Local carrier-phase temperature (K).
        species_mass_fraction : float
            Mass fraction of the reactive species in the particle (0-1).

        Returns
        -------
        dict
            ``{"diameter": float, "mass_loss": float, "heat_release": float}``
            where mass_loss and heat_release are always non-negative.
        """


# ======================================================================
# 单相异相反应
# ======================================================================

_DEFAULT_A = 1.0e3       # 指前因子 (kg/(m2*s*Pa))
_DEFAULT_EA = 1.2e5      # 活化能 (J/mol)
_DEFAULT_N_ORDER = 0.5   # 反应级数
_DEFAULT_R_GAS = 8.314   # 气体常数 (J/(mol*K))
_DEFAULT_HOR = -2.0e7    # 反应热 (J/kg, 负值表示放热)
_DEFAULT_D = 2.6e-5      # 扩散系数 (m2/s)


class SinglePhaseReacting(ReactingModel):
    """Single-phase heterogeneous reaction model.

    Models the reaction of a solid/liquid particle with a gas-phase
    oxidiser using an Arrhenius-type reaction rate:

    .. math::

        k = A \\cdot T^n \\cdot \\exp(-E_a / R T)

    The mass loss rate is diffusion-limited or kinetically limited:

    .. math::

        \\dot{m} = \\pi d^2 \\min(k_{diff}, k_{kin})

    where the diffusion rate is:

    .. math::

        k_{diff} = Sh \\cdot D / d \\cdot \\rho_{ox}

    Parameters
    ----------
    A : float
        Pre-exponential factor. Default ``1e3``.
    activation_energy : float
        Activation energy (J/mol). Default ``1.2e5``.
    n_order : float
        Reaction order (temperature exponent). Default ``0.5``.
    heat_of_reaction : float
        Heat of reaction per unit mass (J/kg, negative for exothermic).
        Default ``-2e7``.
    diffusivity : float
        Diffusivity of oxidiser in carrier phase (m2/s). Default ``2.6e-5``.
    r_gas : float
        Universal gas constant (J/(mol*K)). Default ``8.314``.
    """

    def __init__(
        self,
        A: float = _DEFAULT_A,
        activation_energy: float = _DEFAULT_EA,
        n_order: float = _DEFAULT_N_ORDER,
        heat_of_reaction: float = _DEFAULT_HOR,
        diffusivity: float = _DEFAULT_D,
        r_gas: float = _DEFAULT_R_GAS,
    ) -> None:
        if A < 0.0:
            raise ValueError(f"A must be non-negative, got {A}")
        if activation_energy < 0.0:
            raise ValueError(
                f"activation_energy must be non-negative, got {activation_energy}"
            )
        if diffusivity < 0.0:
            raise ValueError(f"diffusivity must be non-negative, got {diffusivity}")
        if r_gas <= 0.0:
            raise ValueError(f"r_gas must be positive, got {r_gas}")

        self.A = A
        self.activation_energy = activation_energy
        self.n_order = n_order
        self.heat_of_reaction = heat_of_reaction
        self.diffusivity = diffusivity
        self.r_gas = r_gas

    def react(
        self,
        dt: float,
        diameter: float,
        temperature: float,
        fluid_temperature: float,
        species_mass_fraction: float = 1.0,
    ) -> dict:
        """Compute single-phase heterogeneous reaction.

        Returns zero mass loss when diameter or temperature is negligible,
        or when species mass fraction is zero.
        """
        if diameter < 1e-15 or temperature < 1.0:
            return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}

        if species_mass_fraction <= 0.0:
            return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}

        # 动力学速率: k_kin = A * T^n * exp(-Ea/RT)
        RT = self.r_gas * temperature
        if RT < 1e-30:
            return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}

        k_kin = self.A * (temperature ** self.n_order) * math.exp(
            -self.activation_energy / RT
        )

        # 扩散速率: k_diff = 2 * Sh * D / d
        Sh = 2.0  # 稀薄条件下的 Sherwood 数
        k_diff = 2.0 * Sh * self.diffusivity / max(diameter, 1e-30)

        # 总速率受最慢机制控制
        k_total = min(k_kin, k_diff)
        k_total = max(k_total, 0.0)

        # 质量损失: dm = pi * d^2 * k_total * dt * Y_species
        area = math.pi * diameter ** 2
        dm = area * k_total * dt * species_mass_fraction

        # 质量上限：不超过当前粒子质量
        m_particle = (math.pi / 6.0) * diameter ** 3 * 1000.0  # 假设密度
        dm = min(dm, m_particle)
        dm = max(dm, 0.0)

        # 热释放
        heat_release = dm * abs(self.heat_of_reaction)

        # 更新直径
        if dm > 0.0 and m_particle > 1e-30:
            mass_ratio = 1.0 - dm / m_particle
            mass_ratio = max(mass_ratio, 0.0)
            new_d = diameter * (mass_ratio ** (1.0 / 3.0))
        else:
            new_d = diameter

        return {
            "diameter": new_d,
            "mass_loss": dm,
            "heat_release": heat_release,
        }

    def arrhenius_rate(self, temperature: float) -> float:
        """Compute the Arrhenius reaction rate at a given temperature.

        Useful for diagnostics.
        """
        if temperature < 1.0:
            return 0.0
        RT = self.r_gas * temperature
        if RT < 1e-30:
            return 0.0
        return self.A * (temperature ** self.n_order) * math.exp(
            -self.activation_energy / RT
        )

    def __repr__(self) -> str:
        return (
            f"SinglePhaseReacting("
            f"A={self.A}, Ea={self.activation_energy}, "
            f"n={self.n_order})"
        )


# ======================================================================
# 多相反应模型
# ======================================================================

_DEFAULT_N_REACTIONS = 1
_DEFAULT_SPECIES_WEIGHTS = [1.0]


class MultiPhaseReacting(ReactingModel):
    """Multi-component reaction model for complex particle chemistry.

    Handles particles with multiple reactive species (e.g., coal with
    volatile matter and fixed carbon), each with its own reaction
    kinetics.  The total mass loss is the sum of individual species
    contributions:

    .. math::

        \\dot{m}_{total} = \\sum_i w_i \\cdot \\dot{m}_i

    where :math:`w_i` is the mass fraction of species *i* in the particle.

    Parameters
    ----------
    species : list[dict]
        List of species definitions, each containing:
        - ``"A"``: pre-exponential factor
        - ``"Ea"``: activation energy (J/mol)
        - ``"heat"``: heat of reaction (J/kg)
        - ``"weight"``: mass fraction weight (0-1)
    r_gas : float
        Universal gas constant. Default ``8.314``.
    """

    def __init__(
        self,
        species: list[dict] | None = None,
        r_gas: float = _DEFAULT_R_GAS,
    ) -> None:
        if r_gas <= 0.0:
            raise ValueError(f"r_gas must be positive, got {r_gas}")

        if species is None:
            # 默认：单碳反应（类比焦炭燃烧）
            species = [
                {"A": 1.0e3, "Ea": 1.2e5, "heat": -2.0e7, "weight": 1.0},
            ]

        for i, sp in enumerate(species):
            if "A" not in sp or "Ea" not in sp:
                raise ValueError(
                    f"Species {i} must have 'A' and 'Ea' keys"
                )

        self.species = species
        self.r_gas = r_gas

    def react(
        self,
        dt: float,
        diameter: float,
        temperature: float,
        fluid_temperature: float,
        species_mass_fraction: float = 1.0,
    ) -> dict:
        """Compute multi-component reaction rates.

        Returns the sum of mass loss and heat release from all species.
        """
        if diameter < 1e-15 or temperature < 1.0:
            return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}

        if species_mass_fraction <= 0.0:
            return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}

        m_particle = (math.pi / 6.0) * diameter ** 3 * 1000.0
        total_dm = 0.0
        total_heat = 0.0

        for sp in self.species:
            A = sp.get("A", 0.0)
            Ea = sp.get("Ea", 0.0)
            heat = sp.get("heat", 0.0)
            weight = sp.get("weight", 1.0)

            # Arrhenius 速率
            RT = self.r_gas * temperature
            if RT < 1e-30:
                continue

            k = A * math.exp(-Ea / RT)

            # 单个物种的质量损失
            dm_i = (
                math.pi * diameter ** 2 * k * dt
                * species_mass_fraction * weight
            )
            dm_i = max(dm_i, 0.0)

            total_dm += dm_i
            total_heat += dm_i * abs(heat)

        # 限制总量
        total_dm = min(total_dm, m_particle)

        # 更新直径
        if total_dm > 0.0 and m_particle > 1e-30:
            mass_ratio = max(1.0 - total_dm / m_particle, 0.0)
            new_d = diameter * (mass_ratio ** (1.0 / 3.0))
        else:
            new_d = diameter

        return {
            "diameter": new_d,
            "mass_loss": total_dm,
            "heat_release": total_heat,
        }

    def __repr__(self) -> str:
        n = len(self.species)
        return f"MultiPhaseReacting(n_species={n})"
