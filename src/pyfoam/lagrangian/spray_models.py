"""
Spray models for Lagrangian particle tracking.

模拟喷雾的初级破碎和液滴形成过程。主要模型包括：

- blob 注入模型：将液体射流简化为大液滴（blob）
- KH-RT 破碎模型：基于 Kelvin-Helmholtz 和 Rayleigh-Taylor 不稳定性
- TAB 破碎模型：基于 Taylor Analogy Breakup

Provides:

- :class:`SprayModel`      — abstract base
- :class:`BlobAtomization` — blob atomization / primary breakup model
- :class:`TABBreakup`      — Taylor Analogy Breakup model

Usage::

    from pyfoam.lagrangian.spray_models import BlobAtomization

    model = BlobAtomization(blob_diameter=1e-3, we_crit=12.0)
    result = model.atomize(
        dt=1e-5,
        diameter=1e-3,
        relative_velocity=100.0,
        fluid_density=1.225,
        surface_tension=0.072,
    )
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


__all__ = [
    "SprayModel",
    "BlobAtomization",
    "TABBreakup",
]


# ======================================================================
# 抽象基类
# ======================================================================

class SprayModel(ABC):
    """Abstract base for spray atomization / primary breakup models.

    Subclasses implement :meth:`atomize`, which computes the post-atomization
    droplet diameter and a flag indicating whether atomization occurred.
    """

    @abstractmethod
    def atomize(
        self,
        dt: float,
        diameter: float,
        relative_velocity: float,
        fluid_density: float = 1.225,
        surface_tension: float = 0.072,
        particle_density: float = 800.0,
        fluid_viscosity: float = 1.8e-5,
    ) -> dict:
        """Compute post-atomization droplet state.

        Parameters
        ----------
        dt : float
            Time step (s).
        diameter : float
            Current droplet / blob diameter (m).
        relative_velocity : float
            Magnitude of the slip velocity |U_f - U_p| (m/s).
        fluid_density : float
            Carrier-phase density (kg/m3).
        surface_tension : float
            Liquid-gas surface tension coefficient (N/m).
        particle_density : float
            Droplet material density (kg/m3).
        fluid_viscosity : float
            Carrier-phase dynamic viscosity (Pa*s).

        Returns
        -------
        dict
            ``{"diameter": float, "atomized": bool}``
        """


# ======================================================================
# Blob Atomization 模型
# ======================================================================

# 模型默认参数
_DEFAULT_BLOB_DIAMETER = 1e-3   # 默认 blob 直径 (m)
_DEFAULT_WE_CRIT = 12.0         # 临界 Weber 数
_DEFAULT_KH_CONST = 0.61        # KH 不稳定性常数
_DEFAULT_B0 = 0.61              # KH 模型系数 B0
_DEFAULT_B1 = 10.0              # KH 模型系数 B1


class BlobAtomization(SprayModel):
    """Blob atomization model for primary spray breakup.

    Implements the blob injection approach where the intact liquid core
    is represented by large parcels (blobs) with diameter equal to the
    nozzle orifice diameter.  Atomization proceeds via a Kelvin-Helmholtz
    instability model:

    The child droplet diameter is:

    .. math::

        d_{child} = 2 B_0 \\Lambda_{KH}

    where :math:`\\Lambda_{KH}` is the fastest-growing KH wavelength:

    .. math::

        \\Lambda_{KH} = \\frac{9.02 r}{(1 + Oh)(1 + 1.46 Oh^{0.6})}
            \\frac{\\sqrt{We_s}}{1 + We_s / We_c}

    and :math:`Oh = \\mu_l / \\sqrt{\\rho_l \\sigma r}`.

    Parameters
    ----------
    blob_diameter : float
        Initial blob (nozzle) diameter (m). Default ``1e-3``.
    we_crit : float
        Critical Weber number for breakup onset. Default ``12.0``.
    b0 : float
        KH model constant B0. Default ``0.61``.
    b1 : float
        KH model constant B1. Default ``10.0``.
    """

    def __init__(
        self,
        blob_diameter: float = _DEFAULT_BLOB_DIAMETER,
        we_crit: float = _DEFAULT_WE_CRIT,
        b0: float = _DEFAULT_B0,
        b1: float = _DEFAULT_B1,
    ) -> None:
        if blob_diameter <= 0.0:
            raise ValueError(f"blob_diameter must be positive, got {blob_diameter}")
        if we_crit <= 0.0:
            raise ValueError(f"we_crit must be positive, got {we_crit}")
        if b0 <= 0.0:
            raise ValueError(f"b0 must be positive, got {b0}")
        if b1 <= 0.0:
            raise ValueError(f"b1 must be positive, got {b1}")

        self.blob_diameter = blob_diameter
        self.we_crit = we_crit
        self.b0 = b0
        self.b1 = b1

    def atomize(
        self,
        dt: float,
        diameter: float,
        relative_velocity: float,
        fluid_density: float = 1.225,
        surface_tension: float = 0.072,
        particle_density: float = 800.0,
        fluid_viscosity: float = 1.8e-5,
    ) -> dict:
        """Compute blob atomization using the KH instability model.

        Returns the original diameter unchanged when the Weber number is
        below the critical threshold or when the diameter is negligible.
        """
        if diameter < 1e-15 or relative_velocity < 1e-15:
            return {"diameter": diameter, "atomized": False}

        if surface_tension < 1e-15:
            return {"diameter": diameter, "atomized": False}

        r = diameter / 2.0

        # Weber 数
        We = fluid_density * relative_velocity ** 2 * r / surface_tension

        if We < self.we_crit:
            return {"diameter": diameter, "atomized": False}

        # Ohnesorge 数
        oh_arg = particle_density * surface_tension * r
        if oh_arg < 1e-30:
            return {"diameter": diameter, "atomized": False}
        Oh = fluid_viscosity / math.sqrt(oh_arg)

        # 最快增长 KH 波长
        We_s = math.sqrt(We)
        denom_oh = 1.0 + Oh
        denom_we = 1.0 + We / self.we_crit

        if denom_oh < 1e-30 or denom_we < 1e-30:
            return {"diameter": diameter, "atomized": False}

        lambda_kh = (
            9.02 * r
            / (denom_oh * (1.0 + 1.46 * Oh ** 0.6))
            * We_s / denom_we
        )

        # 子液滴直径: d_child = 2 * B0 * lambda_KH
        # 限制 KH 波长不超过液滴半径（物理约束）
        lambda_kh = min(lambda_kh, r)
        d_child = 2.0 * self.b0 * lambda_kh
        d_child = max(d_child, 1e-10)

        if d_child >= diameter:
            return {"diameter": diameter, "atomized": False}

        return {"diameter": d_child, "atomized": True}

    def __repr__(self) -> str:
        return (
            f"BlobAtomization("
            f"blob_diameter={self.blob_diameter}, "
            f"we_crit={self.we_crit}, "
            f"b0={self.b0}, b1={self.b1})"
        )


# ======================================================================
# TAB Breakup 模型
# ======================================================================

# TAB 模型默认参数
_DEFAULT_K_TAB = 10.0       # 弹簧常数
_DEFAULT_C_TAB = 0.5        # 阻尼系数
_DEFAULT_Y0 = 0.0           # 初始位移
_DEFAULT_Y_DOT0 = 0.0       # 初始速度
_DEFAULT_WE_TAB_CRIT = 6.0  # TAB 临界 Weber 数


class TABBreakup(SprayModel):
    """Taylor Analogy Breakup (TAB) model for secondary atomization.

    Models droplet deformation using a spring-mass-damper analogy
    (O'Rourke & Amsden, 1987).  The droplet is treated as a
    damped harmonic oscillator with the aerodynamic force driving
    deformation and surface tension providing the restoring force.

    The dimensionless displacement y satisfies:

    .. math::

        \\ddot{y} + \\frac{C_d}{K} \\dot{y} + \\frac{1}{K} y
            = \\frac{We}{3 K}

    Breakup occurs when y > 1 (displacement exceeds the droplet radius).

    The child droplet diameter is:

    .. math::

        d_{child} = d \\left(\\frac{6}{5} y\\right)^{-1/3} \\quad
            \\text{(when } y \\leq 1\\text{, no breakup)}

    Parameters
    ----------
    k_tab : float
        Spring constant (dimensionless). Default ``10.0``.
    c_tab : float
        Damping coefficient (dimensionless). Default ``0.5``.
    we_crit : float
        Critical Weber number for breakup onset. Default ``6.0``.
    """

    def __init__(
        self,
        k_tab: float = _DEFAULT_K_TAB,
        c_tab: float = _DEFAULT_C_TAB,
        we_crit: float = _DEFAULT_WE_TAB_CRIT,
    ) -> None:
        if k_tab <= 0.0:
            raise ValueError(f"k_tab must be positive, got {k_tab}")
        if c_tab < 0.0:
            raise ValueError(f"c_tab must be non-negative, got {c_tab}")
        if we_crit <= 0.0:
            raise ValueError(f"we_crit must be positive, got {we_crit}")

        self.k_tab = k_tab
        self.c_tab = c_tab
        self.we_crit = we_crit

    def atomize(
        self,
        dt: float,
        diameter: float,
        relative_velocity: float,
        fluid_density: float = 1.225,
        surface_tension: float = 0.072,
        particle_density: float = 800.0,
        fluid_viscosity: float = 1.8e-5,
    ) -> dict:
        """Compute TAB breakup using the spring-mass-damper model.

        Returns the original diameter unchanged when the Weber number is
        below the critical threshold, the diameter is negligible, or
        the relative velocity is negligible.
        """
        if diameter < 1e-15 or relative_velocity < 1e-15:
            return {"diameter": diameter, "atomized": False}

        if surface_tension < 1e-15:
            return {"diameter": diameter, "atomized": False}

        # Weber 数: We = rho_f * v^2 * d / (2 * sigma)
        We = (
            fluid_density * relative_velocity ** 2 * diameter
            / (2.0 * surface_tension)
        )

        if We < self.we_crit:
            return {"diameter": diameter, "atomized": False}

        # TAB ODE 解析解 (Euler 方法推进一步)
        # ddot{y} + C/K * dot{y} + y/K = We/(3K)
        # 稳态解 y_eq = We / 3
        y_eq = We / 3.0

        # 使用简化的 Euler 方法推进一阶系统
        # 假设初始 y=0, dy/dt=0
        # 一步后: y = 0.5 * a * dt^2 (a = (y_eq - C*dy - y) / K)
        # 简化: y ~ y_eq * (1 - exp(-dt/tau)) 其中 tau = K
        tau = self.k_tab
        if tau < 1e-30:
            return {"diameter": diameter, "atomized": False}

        y = y_eq * (1.0 - math.exp(-dt / tau))

        # 限制 y 的范围
        y = min(y, 10.0)

        if y < 1.0:
            # 未达到破碎阈值，液滴变形但不破碎
            return {"diameter": diameter, "atomized": False}

        # 破碎发生：子液滴直径
        # d_child = d / (6*y/5)^(1/3)
        factor = 6.0 * y / 5.0
        if factor <= 0.0:
            return {"diameter": diameter, "atomized": False}

        d_child = diameter * factor ** (-1.0 / 3.0)
        d_child = max(d_child, 1e-10)

        if d_child >= diameter:
            return {"diameter": diameter, "atomized": False}

        return {"diameter": d_child, "atomized": True}

    def compute_displacement(
        self,
        dt: float,
        diameter: float,
        relative_velocity: float,
        fluid_density: float = 1.225,
        surface_tension: float = 0.072,
    ) -> float:
        """Compute the dimensionless TAB displacement y.

        Useful for diagnostics and coupling with other sub-models.

        Returns 0 when the We is below the critical threshold.
        """
        if diameter < 1e-15 or relative_velocity < 1e-15:
            return 0.0
        if surface_tension < 1e-15:
            return 0.0

        We = (
            fluid_density * relative_velocity ** 2 * diameter
            / (2.0 * surface_tension)
        )

        if We < self.we_crit:
            return 0.0

        y_eq = We / 3.0
        tau = self.k_tab
        if tau < 1e-30:
            return 0.0

        y = y_eq * (1.0 - math.exp(-dt / tau))
        return min(y, 10.0)

    def __repr__(self) -> str:
        return (
            f"TABBreakup("
            f"k_tab={self.k_tab}, "
            f"c_tab={self.c_tab}, "
            f"we_crit={self.we_crit})"
        )
