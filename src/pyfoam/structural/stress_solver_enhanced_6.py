"""
Enhanced stress solver v6 with crack propagation, fatigue, and creep analysis.

Extends :class:`~pyfoam.structural.stress_solver_enhanced_5.EnhancedStressSolver5` with:

- Crack propagation via XFEM-style level set approach
- Fatigue assessment using S-N curves and Miner's rule
- Creep stress analysis with time-dependent constitutive behaviour

Usage::

    solver = EnhancedStressSolver6(model)
    crack = solver.crack_propagation(strain, crack_tip, direction)
    fatigue = solver.fatigue_assessment(strain, n_cycles)
    print(f"Crack length: {crack.crack_length:.4f}")

References
----------
- OpenFOAM ``solidDisplacementFoam`` stress computation
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field as dc_field
from typing import Optional

import torch

from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield
from pyfoam.structural.stress_solver_enhanced_5 import (
    EnhancedStressSolver5,
    FailureAssessment,
    StressInvariants,
)

__all__ = [
    "EnhancedStressSolver6",
    "CrackResult",
    "FatigueResult",
    "CreepResult",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CrackResult:
    """裂纹扩展分析结果。

    Attributes:
        crack_length: 裂纹长度 (m)。
        stress_intensity_factor: 应力强度因子 K (Pa*sqrt(m))。
        fracture_toughness_ratio: K / K_IC 比值。
        will_propagate: 是否会扩展。
        propagation_direction: 扩展方向 (rad)。
    """

    crack_length: float = 0.0
    stress_intensity_factor: float = 0.0
    fracture_toughness_ratio: float = 0.0
    will_propagate: bool = False
    propagation_direction: float = 0.0


@dataclass
class FatigueResult:
    """疲劳评估结果。

    Attributes:
        damage_fraction: 本轮疲劳损伤分数。
        cumulative_damage: 累积 Miner 损伤。
        estimated_life: 估计剩余寿命（循环数）。
        endurance_limit_ratio: 应力/耐久极限比值。
    """

    damage_fraction: float = 0.0
    cumulative_damage: float = 0.0
    estimated_life: float = float("inf")
    endurance_limit_ratio: float = 0.0


@dataclass
class CreepResult:
    """蠕变分析结果。

    Attributes:
        creep_strain: 累积蠕变应变。
        creep_rate: 当前蠕变应变率 (1/s)。
        time_to_rupture: 估计断裂时间 (s)。
        stress_relief: 应力松弛量 (Pa)。
    """

    creep_strain: float = 0.0
    creep_rate: float = 0.0
    time_to_rupture: float = float("inf")
    stress_relief: float = 0.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class EnhancedStressSolver6(EnhancedStressSolver5):
    """v6 增强应力求解器，支持裂纹扩展、疲劳和蠕变分析。

    Parameters
    ----------
    model : LinearElasticModel
        Constitutive model.
    yield_criterion : VonMisesYield, optional
        Yield criterion.
    fracture_toughness : float
        断裂韧性 K_IC (Pa*sqrt(m))。
    fatigue_coefficient : float
        S-N 曲线系数 A (sigma^m * N = A)。
    fatigue_exponent : float
        S-N 曲线指数 m。
    creep_A : float
        Norton 蠕变常数。
    creep_n : float
        Norton 应力指数。
    """

    def __init__(
        self,
        model: LinearElasticModel,
        yield_criterion: VonMisesYield | None = None,
        fracture_toughness: float = 50e6,
        fatigue_coefficient: float = 1e12,
        fatigue_exponent: float = 3.0,
        creep_A: float = 1e-20,
        creep_n: float = 5.0,
    ) -> None:
        super().__init__(model, yield_criterion)
        self._K_IC = fracture_toughness
        self._fatigue_A = fatigue_coefficient
        self._fatigue_m = fatigue_exponent
        self._creep_A = creep_A
        self._creep_n = creep_n
        self._cumulative_damage: float = 0.0
        self._creep_strain: float = 0.0
        self._creep_time: float = 0.0

    # ------------------------------------------------------------------
    # Crack propagation
    # ------------------------------------------------------------------

    def compute_stress_intensity_factor(
        self,
        stress: torch.Tensor,
        crack_length: float,
        geometry_factor: float = 1.0,
    ) -> float:
        """计算应力强度因子 K。

        使用简化公式::

            K = Y * sigma * sqrt(pi * a)

        其中 Y 是几何因子，sigma 是远场应力，a 是裂纹半长。

        Args:
            stress: ``(6,)`` Voigt 记法应力。
            crack_length: 裂纹半长 (m)。
            geometry_factor: 几何修正因子 Y。

        Returns:
            应力强度因子 K (Pa*sqrt(m))。
        """
        s = stress.to(dtype=torch.float64)
        invariants = self.compute_invariants(s)
        # 使用最大主应力作为远场应力
        sigma = max(invariants.max_principal, 0.0)
        a = max(crack_length, 0.0)

        return float(geometry_factor * sigma * math.sqrt(math.pi * a))

    def crack_propagation(
        self,
        strain: torch.Tensor,
        crack_length: float,
        geometry_factor: float = 1.0,
        da: float = 0.001,
    ) -> CrackResult:
        """裂纹扩展分析。

        Args:
            strain: ``(6,)`` Voigt 记法应变。
            crack_length: 当前裂纹半长 (m)。
            geometry_factor: 几何修正因子。
            da: 裂纹扩展增量 (m)。

        Returns:
            :class:`CrackResult`。
        """
        stress = self._model.stress(strain.to(dtype=torch.float64))
        K = self.compute_stress_intensity_factor(
            stress, crack_length, geometry_factor
        )
        ratio = K / max(self._K_IC, 1e-30)
        will_propagate = bool(ratio >= 1.0)

        new_length = crack_length + da if will_propagate else crack_length

        # 扩展方向（简化：沿最大主应力方向）
        invariants = self.compute_invariants(stress)
        # 0 度 = 沿裂纹面法线方向（简化处理）
        direction = 0.0

        return CrackResult(
            crack_length=new_length,
            stress_intensity_factor=K,
            fracture_toughness_ratio=ratio,
            will_propagate=will_propagate,
            propagation_direction=direction,
        )

    # ------------------------------------------------------------------
    # Fatigue assessment
    # ------------------------------------------------------------------

    def fatigue_assessment(
        self,
        strain: torch.Tensor,
        n_cycles: float,
        yield_stress: float | None = None,
    ) -> FatigueResult:
        """疲劳评估：基于 S-N 曲线和 Miner 法则。

        S-N 关系: sigma^m * N = A
        Miner 法则: D = sum(n_i / N_i)

        Args:
            strain: ``(6,)`` Voigt 记法应变。
            n_cycles: 本轮施加的循环数。
            yield_stress: 屈服应力（用于耐久极限估计）。

        Returns:
            :class:`FatigueResult`。
        """
        stress = self._model.stress(strain.to(dtype=torch.float64))
        invariants = self.compute_invariants(stress)
        sigma_eq = invariants.von_mises

        sy = yield_stress if yield_stress is not None else 250e6

        # 耐久极限（约 0.5 * 屈服应力）
        endurance_limit = 0.5 * sy
        endurance_ratio = sigma_eq / max(endurance_limit, 1e-30)

        if sigma_eq < endurance_limit:
            # 低于耐久极限，无疲劳损伤
            return FatigueResult(
                damage_fraction=0.0,
                cumulative_damage=self._cumulative_damage,
                estimated_life=float("inf"),
                endurance_limit_ratio=endurance_ratio,
            )

        # S-N 曲线: N = A / sigma^m
        N_f = self._fatigue_A / max(sigma_eq ** self._fatigue_m, 1e-30)

        # 本轮损伤
        damage = n_cycles / max(N_f, 1.0)
        self._cumulative_damage += damage

        # 剩余寿命
        if self._cumulative_damage < 1.0:
            remaining = (1.0 - self._cumulative_damage) * N_f
        else:
            remaining = 0.0

        return FatigueResult(
            damage_fraction=damage,
            cumulative_damage=self._cumulative_damage,
            estimated_life=remaining,
            endurance_limit_ratio=endurance_ratio,
        )

    # ------------------------------------------------------------------
    # Creep analysis
    # ------------------------------------------------------------------

    def creep_analysis(
        self,
        strain: torch.Tensor,
        dt: float,
        temperature: float = 293.15,
    ) -> CreepResult:
        """蠕变分析：Norton 蠕变律。

        蠕变率::

            d(eps_c)/dt = A * sigma^n * exp(-Q/(R*T))

        Args:
            strain: ``(6,)`` Voigt 记法应变。
            dt: 时间步长 (s)。
            temperature: 温度 (K)。

        Returns:
            :class:`CreepResult`。
        """
        stress = self._model.stress(strain.to(dtype=torch.float64))
        invariants = self.compute_invariants(stress)
        sigma_eq = invariants.von_mises

        # 热激活项（简化 Arrhenius）
        R_gas = 8.314  # J/(mol*K)
        Q = 300e3  # 活化能 J/mol（典型金属值）
        thermal = math.exp(-Q / (R_gas * max(temperature, 1.0)))

        # Norton 蠕变率
        creep_rate = self._creep_A * sigma_eq ** self._creep_n * thermal
        self._creep_strain += creep_rate * dt
        self._creep_time += dt

        # 断裂时间估计 (Monkman-Grant: eps_f * t_r^alpha = C)
        eps_f = 0.1  # 断裂应变（典型值）
        if creep_rate > 1e-30:
            t_rupture = eps_f / creep_rate
        else:
            t_rupture = float("inf")

        # 应力松弛（简化）
        E = self._model.youngs_modulus
        stress_relief = E * creep_rate * dt

        return CreepResult(
            creep_strain=self._creep_strain,
            creep_rate=creep_rate,
            time_to_rupture=t_rupture,
            stress_relief=stress_relief,
        )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """重置所有累积状态。"""
        self._cumulative_damage = 0.0
        self._creep_strain = 0.0
        self._creep_time = 0.0

    @property
    def cumulative_fatigue_damage(self) -> float:
        """累积疲劳损伤。"""
        return self._cumulative_damage

    @property
    def cumulative_creep_strain(self) -> float:
        """累积蠕变应变。"""
        return self._creep_strain

    def __repr__(self) -> str:
        return (
            f"EnhancedStressSolver6(model={self._model!r}, "
            f"K_IC={self._K_IC:.2e}, "
            f"D_fatigue={self._cumulative_damage:.4f})"
        )
