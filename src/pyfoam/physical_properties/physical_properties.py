"""
physicalProperties — 物性参数统一框架。

对应 OpenFOAM-13 的 physicalProperties/physicalProperties/physicalProperties.H。
提供从算例目录读取物性参数的统一接口。
底层实际委托给 thermophysical 模块。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from pyfoam.core.dtype import CFD_DTYPE


class PhysicalProperties:
    """物性参数管理器。

    从 OpenFOAM 算例目录的 ``constant/physicalProperties`` 或
    ``constant/transportProperties`` 读取流体物性参数。

    Examples:
        >>> props = PhysicalProperties(nu=1e-5)
        >>> props.nu
        1e-05
    """

    def __init__(
        self,
        nu: float = 1e-5,
        rho: float = 1.0,
        Cp: float = 1005.0,
        kappa: float = 0.026,
        beta: float = 3.43e-3,
        T_ref: float = 300.0,
    ):
        """初始化物性参数。

        Args:
            nu: 运动粘度 (m²/s)。
            rho: 密度 (kg/m³)。
            Cp: 定压比热 (J/(kg·K))。
            kappa: 导热系数 (W/(m·K))。
            beta: 热膨胀系数 (1/K)。
            T_ref: 参考温度 (K)。
        """
        self._nu = nu
        self._rho = rho
        self._Cp = Cp
        self._kappa = kappa
        self._beta = beta
        self._T_ref = T_ref

    @property
    def nu(self) -> float:
        """运动粘度 (m²/s)。"""
        return self._nu

    @property
    def rho(self) -> float:
        """密度 (kg/m³)。"""
        return self._rho

    @property
    def mu(self) -> float:
        """动力粘度 (Pa·s) = rho * nu。"""
        return self._rho * self._nu

    @property
    def Cp(self) -> float:
        """定压比热 (J/(kg·K))。"""
        return self._Cp

    @property
    def kappa(self) -> float:
        """导热系数 (W/(m·K))。"""
        return self._kappa

    @property
    def beta(self) -> float:
        """热膨胀系数 (1/K)。"""
        return self._beta

    @property
    def T_ref(self) -> float:
        """参考温度 (K)。"""
        return self._T_ref

    @property
    def Pr(self) -> float:
        """普朗特数 = mu * Cp / kappa。"""
        if self._kappa == 0:
            return float("inf")
        return self.mu * self._Cp / self._kappa

    @property
    def alpha(self) -> float:
        """热扩散系数 (m²/s) = kappa / (rho * Cp)。"""
        return self._kappa / (self._rho * self._Cp)

    @classmethod
    def from_dict(cls, d: dict) -> "PhysicalProperties":
        """从字典创建（兼容 OpenFOAM physicalProperties 格式）。"""
        return cls(
            nu=float(d.get("nu", 1e-5)),
            rho=float(d.get("rho", 1.0)),
            Cp=float(d.get("Cp", 1005.0)),
            kappa=float(d.get("kappa", 0.026)),
            beta=float(d.get("beta", 3.43e-3)),
            T_ref=float(d.get("Tref", 300.0)),
        )

    def __repr__(self) -> str:
        return (
            f"PhysicalProperties(nu={self._nu}, rho={self._rho}, "
            f"Cp={self._Cp}, kappa={self._kappa})"
        )
