"""
Reaction kinetics and combustion models for reactive flow simulations.

Provides Arrhenius-type reaction rate expressions and eddy-dissipation
combustion closures commonly used in CFD reacting-flow solvers.

**Reaction Rate Models:**

- :class:`ReactionRateModel` — abstract base with RTS registry
- :class:`ArrheniusReaction` — k = A * T^b * exp(-Ea / RT)
- :class:`ThirdBodyReaction` — wraps a base rate with third-body efficiency
- :class:`FallOffReaction` — Lindemann fall-off pressure dependence

**Combustion Models:**

- :class:`CombustionModel` — abstract base with RTS registry
- :class:`PaSRModel` — Partially Stirred Reactor
- :class:`EDCModel` — Eddy Dissipation Concept
- :class:`InfinitelyFastChemistry` — mixing-limited (equilibrium) combustion
- :class:`FSDModel` — Flame Surface Density model for premixed flames

Usage::

    from pyfoam.thermophysical.reaction import ReactionRateModel, CombustionModel

    # Arrhenius rate
    rxn = ReactionRateModel.create("Arrhenius", A=1e10, b=0.0, Ea=8e4)
    k = rxn.rate(T=1000.0)

    # PaSR combustion
    comb = CombustionModel.create("PaSR", A=1e10, Ea=8e4, C_mix=0.1)
    Su, Sp = comb.source(Y_fuel=0.05, Y_ox=0.23, T=1000.0, rho=1.0)

    # FSD premixed combustion
    fsd = CombustionModel.create("FSD", S_L=0.4, Sigma_0=100.0)
    Su, Sp = fsd.source(Y_fuel=0.05, Y_ox=0.23, T=1000.0, rho=1.0)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "ReactionRateModel",
    "ArrheniusReaction",
    "ThirdBodyReaction",
    "FallOffReaction",
    "CombustionModel",
    "PaSRModel",
    "EDCModel",
    "InfinitelyFastChemistry",
    "FSDModel",
]

logger = logging.getLogger(__name__)

# 通用气体常数 (J/(mol·K))
R_UNIVERSAL = 8.314


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _to_tensor(
    value: torch.Tensor | float,
    ref: torch.Tensor | None = None,
) -> torch.Tensor:
    """将标量转为张量，若已是张量则直接返回。"""
    if isinstance(value, torch.Tensor):
        return value
    device = ref.device if ref is not None else get_device()
    dtype = ref.dtype if ref is not None else get_default_dtype()
    return torch.tensor(value, dtype=dtype, device=device)


# ===================================================================
# 反应速率模型
# ===================================================================


class ReactionRateModel(ABC):
    """反应速率模型抽象基类。

    子类必须实现 :meth:`rate` 方法，返回反应速率。

    RTS (Run-Time Selection) 注册表支持字符串查找::

        @ReactionRateModel.register("Arrhenius")
        class ArrheniusReaction(ReactionRateModel):
            ...

        rxn = ReactionRateModel.create("Arrhenius", A=1e10, b=0.0, Ea=8e4)
    """

    _registry: ClassVar[dict[str, Type[ReactionRateModel]]] = {}

    # ------------------------------------------------------------------
    # RTS 注册表
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str) -> callable:
        """装饰器：将反应速率模型类注册到 *name*。"""

        def decorator(model_cls: Type[ReactionRateModel]) -> Type[ReactionRateModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Reaction rate model '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> ReactionRateModel:
        """工厂：根据注册名创建反应速率模型实例。

        Parameters
        ----------
        name : str
            注册的模型类型名。
        **kwargs
            转发给模型构造函数的参数。

        Raises
        ------
        KeyError
            如果 *name* 不在注册表中。
        """
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown reaction rate model '{name}'. Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """返回已注册的模型类型名列表（排序）。"""
        return sorted(cls._registry.keys())

    # ------------------------------------------------------------------
    # 抽象接口
    # ------------------------------------------------------------------

    @abstractmethod
    def rate(
        self,
        T: torch.Tensor | float,
        concentrations: dict[str, torch.Tensor | float] | None = None,
    ) -> torch.Tensor:
        """计算反应速率。

        Parameters
        ----------
        T : torch.Tensor | float
            温度 (K) — 标量或 ``(n_cells,)`` 张量。
        concentrations : dict, optional
            组分浓度字典 ``{species_name: concentration}``，单位 mol/m³。

        Returns
        -------
        torch.Tensor
            反应速率 (mol/(m³·s)) — 与 T 同形状。
        """


# ===================================================================
# Arrhenius 反应
# ===================================================================


@ReactionRateModel.register("Arrhenius")
class ArrheniusReaction(ReactionRateModel):
    """Arrhenius 反应速率：k = A * T^b * exp(-Ea / (R * T))。

    Parameters
    ----------
    A : float
        指前因子 (pre-exponential factor)，单位取决于反应级数。
    b : float
        温度指数 (temperature exponent)，无量纲。默认 0.0。
    Ea : float
        活化能 (activation energy)，J/mol。默认 0.0。

    Examples::

        rxn = ArrheniusReaction(A=1e10, b=0.0, Ea=8e4)
        k = rxn.rate(T=1000.0)  # 阿伦尼乌斯速率
    """

    def __init__(
        self,
        A: float = 1.0,
        b: float = 0.0,
        Ea: float = 0.0,
    ) -> None:
        self.A = A
        self.b = b
        self.Ea = Ea

    def rate(
        self,
        T: torch.Tensor | float,
        concentrations: dict[str, torch.Tensor | float] | None = None,
    ) -> torch.Tensor:
        """计算 Arrhenius 速率: k = A * T^b * exp(-Ea / (RT))。

        Parameters
        ----------
        T : torch.Tensor | float
            温度 (K)。
        concentrations : dict, optional
            未使用，保持接口一致。

        Returns
        -------
        torch.Tensor
            反应速率常数 k。
        """
        T_t = _to_tensor(T)
        T_safe = T_t.clamp(min=1.0)

        k = self.A * T_safe.pow(self.b) * torch.exp(
            -self.Ea / (R_UNIVERSAL * T_safe)
        )
        return k

    def __repr__(self) -> str:
        return f"ArrheniusReaction(A={self.A}, b={self.b}, Ea={self.Ea})"


# ===================================================================
# 第三体反应
# ===================================================================


@ReactionRateModel.register("thirdBody")
class ThirdBodyReaction(ReactionRateModel):
    """第三体反应：k_eff = k_base * [M]。

    包装另一个反应速率模型，加入第三体增强效率。

    Parameters
    ----------
    base_rate : ReactionRateModel
        被包装的基础反应速率模型。
    efficiencies : dict[str, float], optional
        各组分的第三体效率，默认所有组分效率为 1.0。
        例如 ``{"H2O": 12.0, "N2": 0.5}``。

    Examples::

        base = ArrheniusReaction(A=1e10, Ea=8e4)
        rxn = ThirdBodyReaction(base_rate=base, efficiencies={"H2O": 12.0})
        k = rxn.rate(T=1000.0, concentrations={"H2O": 0.01, "N2": 0.78})
    """

    def __init__(
        self,
        base_rate: ReactionRateModel | None = None,
        efficiencies: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        # 支持 RTS 工厂传入 base_rate_type / base_rate_kwargs
        if base_rate is None and "base_rate_type" in kwargs:
            base_type = kwargs.pop("base_rate_type")
            base_kw = kwargs.pop("base_rate_kwargs", {})
            base_rate = ReactionRateModel.create(base_type, **base_kw)
        if base_rate is None:
            base_rate = ArrheniusReaction()

        self.base_rate = base_rate
        self.efficiencies = efficiencies or {}

    def rate(
        self,
        T: torch.Tensor | float,
        concentrations: dict[str, torch.Tensor | float] | None = None,
    ) -> torch.Tensor:
        """计算第三体增强速率: k_eff = k_base * [M]。

        [M] = sum(eff_i * C_i) for all species, plus default (1.0)
        for any unlisted species total concentration.

        Parameters
        ----------
        T : torch.Tensor | float
            温度 (K)。
        concentrations : dict, optional
            组分浓度 ``{species: concentration}`` (mol/m³)。

        Returns
        -------
        torch.Tensor
            有效反应速率。
        """
        k_base = self.base_rate.rate(T, concentrations)

        # 计算第三体有效浓度 [M]
        T_t = _to_tensor(T)
        M = torch.zeros_like(_to_tensor(0.0, ref=T_t))

        if concentrations is not None:
            for species, conc in concentrations.items():
                eff = self.efficiencies.get(species, 1.0)
                M = M + eff * _to_tensor(conc, ref=T_t)
        else:
            # 无浓度信息时假设 [M] = 1.0（归一化）
            M = torch.ones_like(T_t)

        return k_base * M

    def __repr__(self) -> str:
        return (
            f"ThirdBodyReaction(base_rate={self.base_rate!r}, "
            f"efficiencies={self.efficiencies})"
        )


# ===================================================================
# Lindemann 降压反应
# ===================================================================


@ReactionRateModel.register("fallOff")
class FallOffReaction(ReactionRateModel):
    """Lindemann 降压反应速率。

    在低压和高压极限之间插值::

        Pr = k0 * [M] / k_inf
        k = k_inf * Pr / (1 + Pr)

    Parameters
    ----------
    k0 : ReactionRateModel | None
        低压极限反应速率模型。默认 ArrheniusReaction(A=1e10)。
    k_inf : ReactionRateModel | None
        高压极限反应速率模型。默认 ArrheniusReaction(A=1e6)。

    Examples::

        k0 = ArrheniusReaction(A=1e14, b=-0.5, Ea=8e4)
        k_inf = ArrheniusReaction(A=1e11, b=0.0, Ea=8e4)
        rxn = FallOffReaction(k0=k0, k_inf=k_inf)
        k = rxn.rate(T=1000.0, concentrations={"M": 1.0})
    """

    def __init__(
        self,
        k0: ReactionRateModel | None = None,
        k_inf: ReactionRateModel | None = None,
        **kwargs: Any,
    ) -> None:
        # 支持 RTS 工厂参数
        if k0 is None and "k0_type" in kwargs:
            k0 = ReactionRateModel.create(kwargs.pop("k0_type"), **kwargs.pop("k0_kwargs", {}))
        if k_inf is None and "k_inf_type" in kwargs:
            k_inf = ReactionRateModel.create(kwargs.pop("k_inf_type"), **kwargs.pop("k_inf_kwargs", {}))

        self.k0 = k0 or ArrheniusReaction(A=1e10)
        self.k_inf = k_inf or ArrheniusReaction(A=1e6)

    def rate(
        self,
        T: torch.Tensor | float,
        concentrations: dict[str, torch.Tensor | float] | None = None,
    ) -> torch.Tensor:
        """计算 Lindemann 降压速率。

        Pr = k0 * [M] / k_inf
        k  = k_inf * Pr / (1 + Pr)

        Parameters
        ----------
        T : torch.Tensor | float
            温度 (K)。
        concentrations : dict, optional
            组分浓度，需包含总浓度或各组分浓度用于计算 [M]。

        Returns
        -------
        torch.Tensor
            降压反应速率。
        """
        k0_val = self.k0.rate(T, concentrations)
        k_inf_val = self.k_inf.rate(T, concentrations)

        # 计算 [M]
        T_t = _to_tensor(T)
        if concentrations is not None:
            M = torch.zeros_like(_to_tensor(0.0, ref=T_t))
            for conc in concentrations.values():
                M = M + _to_tensor(conc, ref=T_t)
        else:
            M = torch.ones_like(T_t)

        # Lindemann: Pr = k0 * [M] / k_inf
        k_inf_safe = k_inf_val.clamp(min=1e-30)
        Pr = k0_val * M / k_inf_safe

        # k = k_inf * Pr / (1 + Pr)
        return k_inf_val * Pr / (1.0 + Pr)

    def __repr__(self) -> str:
        return f"FallOffReaction(k0={self.k0!r}, k_inf={self.k_inf!r})"


# ===================================================================
# 燃烧模型
# ===================================================================


class CombustionModel(ABC):
    """燃烧模型抽象基类。

    子类必须实现 :meth:`source` 方法，返回燃烧源项 (Su, Sp)。

    RTS (Run-Time Selection) 注册表::

        @CombustionModel.register("PaSR")
        class PaSRModel(CombustionModel):
            ...

        comb = CombustionModel.create("PaSR", A=1e10, Ea=8e4, C_mix=0.1)
    """

    _registry: ClassVar[dict[str, Type[CombustionModel]]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """装饰器：将燃烧模型类注册到 *name*。"""

        def decorator(model_cls: Type[CombustionModel]) -> Type[CombustionModel]:
            if name in cls._registry:
                raise ValueError(
                    f"Combustion model '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> CombustionModel:
        """工厂：根据注册名创建燃烧模型实例。"""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown combustion model '{name}'. Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """返回已注册的燃烧模型类型名列表（排序）。"""
        return sorted(cls._registry.keys())

    @abstractmethod
    def source(
        self,
        Y_fuel: torch.Tensor | float,
        Y_ox: torch.Tensor | float,
        T: torch.Tensor | float,
        rho: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算燃烧源项。

        Parameters
        ----------
        Y_fuel : torch.Tensor | float
            燃料质量分数。
        Y_ox : torch.Tensor | float
            氧化剂质量分数。
        T : torch.Tensor | float
            温度 (K)。
        rho : torch.Tensor | float
            密度 (kg/m³)。

        Returns
        -------
        Su : torch.Tensor
            燃烧源项（正值 = 燃料消耗率）。
        Sp : torch.Tensor
            线性化源项对温度的偏导（用于隐式求解）。
        """


# ===================================================================
# PaSR 模型
# ===================================================================


@CombustionModel.register("PaSR")
class PaSRModel(CombustionModel):
    """Partially Stirred Reactor (PaSR) 燃烧模型。

    假设湍流混合与化学反应竞争::

        tau_mix = C_mix * (nu / epsilon)^0.5
        kappa   = tau_mix / (tau_mix + tau_react)
        source  = kappa * rho * Y_fuel * Y_ox * A * exp(-Ea / RT)

    Parameters
    ----------
    A : float
        Arrhenius 指前因子。默认 1e8。
    Ea : float
        活化能 (J/mol)。默认 5e4。
    C_mix : float
        混合时间系数。默认 0.1。
    stoich_ratio : float
        化学计量比 (氧化剂/燃料质量比)。默认 1.0。

    Examples::

        pasr = PaSRModel(A=1e10, Ea=8e4, C_mix=0.1)
        Su, Sp = pasr.source(Y_fuel=0.05, Y_ox=0.23, T=1000.0, rho=1.0)
    """

    def __init__(
        self,
        A: float = 1e8,
        Ea: float = 5e4,
        C_mix: float = 0.1,
        stoich_ratio: float = 1.0,
    ) -> None:
        self.A = A
        self.Ea = Ea
        self.C_mix = C_mix
        self.stoich_ratio = stoich_ratio

    def source(
        self,
        Y_fuel: torch.Tensor | float,
        Y_ox: torch.Tensor | float,
        T: torch.Tensor | float,
        rho: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算 PaSR 燃烧源项。

        Su = kappa * rho * min(Y_fuel, Y_ox/s) * A * exp(-Ea/RT)

        其中 kappa = tau_mix / (tau_mix + tau_react) ∈ [0, 1]。
        简化处理：tau_react = 1 / (A * exp(-Ea/RT))，
        tau_mix 由 C_mix 参数化。

        Parameters
        ----------
        Y_fuel : torch.Tensor | float
            燃料质量分数。
        Y_ox : torch.Tensor | float
            氧化剂质量分数。
        T : torch.Tensor | float
            温度 (K)。
        rho : torch.Tensor | float
            密度 (kg/m³)。

        Returns
        -------
        Su, Sp : tuple[torch.Tensor, torch.Tensor]
        """
        T_t = _to_tensor(T)
        Yf = _to_tensor(Y_fuel, ref=T_t)
        Yo = _to_tensor(Y_ox, ref=T_t)
        rho_t = _to_tensor(rho, ref=T_t)

        T_safe = T_t.clamp(min=1.0)

        # Arrhenius 速率
        k_arr = self.A * torch.exp(-self.Ea / (R_UNIVERSAL * T_safe))

        # 化学反应时间尺度
        tau_react = 1.0 / k_arr.clamp(min=1e-30)

        # 混合时间尺度 (用 C_mix 参数化，无显式 nu/epsilon 时简化)
        tau_mix = self.C_mix

        # 混合分数 kappa
        kappa = tau_mix / (tau_mix + tau_react)
        kappa = kappa.clamp(0.0, 1.0)

        # 化学计量约束
        Y_limit = torch.min(Yf, Yo / self.stoich_ratio)

        # 燃烧源项
        Su = kappa * rho_t * Y_limit * k_arr

        # 线性化 Sp（dSu/dT 的近似）
        Sp = Su * self.Ea / (R_UNIVERSAL * T_safe.pow(2))

        return Su, Sp

    def __repr__(self) -> str:
        return (
            f"PaSRModel(A={self.A}, Ea={self.Ea}, "
            f"C_mix={self.C_mix}, stoich_ratio={self.stoich_ratio})"
        )


# ===================================================================
# EDC 模型
# ===================================================================


@CombustionModel.register("EDC")
class EDCModel(CombustionModel):
    """Eddy Dissipation Concept (EDC) 燃烧模型。

    假设化学反应发生在湍流最小涡尺度的区域::

        tau_star = C_tau * (nu * epsilon)^0.25 / epsilon^0.5
        gamma    = min((tau_star / tau_chem)^0.5, 1)
        source   = rho * gamma^2 / (1 - gamma^2) * min(Y_fuel, Y_ox / s)

    Parameters
    ----------
    C_tau : float
        EDC 时间尺度常数。默认 0.4082 (2.1377^(-2))。
    tau_chem : float
        化学反应时间尺度 (s)。默认 1e-3。
    stoich_ratio : float
        化学计量比 (氧化剂/燃料质量比)。默认 1.0。

    Examples::

        edc = EDCModel(C_tau=0.4, tau_chem=1e-3)
        Su, Sp = edc.source(Y_fuel=0.05, Y_ox=0.23, T=1000.0, rho=1.0)
    """

    def __init__(
        self,
        C_tau: float = 0.4082,
        tau_chem: float = 1e-3,
        stoich_ratio: float = 1.0,
    ) -> None:
        self.C_tau = C_tau
        self.tau_chem = tau_chem
        self.stoich_ratio = stoich_ratio

    def source(
        self,
        Y_fuel: torch.Tensor | float,
        Y_ox: torch.Tensor | float,
        T: torch.Tensor | float,
        rho: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算 EDC 燃烧源项。

        gamma = min(sqrt(tau_star / tau_chem), 1)
        Su    = rho * gamma^2 / (1 - gamma^2) * min(Y_fuel, Y_ox/s)

        简化处理: tau_star 用 C_tau 参数化（无显式 nu/epsilon 时取 tau_star = C_tau）。

        Parameters
        ----------
        Y_fuel : torch.Tensor | float
            燃料质量分数。
        Y_ox : torch.Tensor | float
            氧化剂质量分数。
        T : torch.Tensor | float
            温度 (K)。
        rho : torch.Tensor | float
            密度 (kg/m³)。

        Returns
        -------
        Su, Sp : tuple[torch.Tensor, torch.Tensor]
        """
        T_t = _to_tensor(T)
        Yf = _to_tensor(Y_fuel, ref=T_t)
        Yo = _to_tensor(Y_ox, ref=T_t)
        rho_t = _to_tensor(rho, ref=T_t)

        # EDC 时间尺度参数 (简化: tau_star = C_tau)
        tau_star = self.C_tau

        # gamma = min(sqrt(tau_star / tau_chem), 1)
        gamma = torch.sqrt(
            torch.tensor(tau_star / self.tau_chem, dtype=T_t.dtype, device=T_t.device)
        )
        gamma = gamma.clamp(max=1.0)

        # 防止 gamma^2 = 1 时发散
        gamma_sq = gamma.pow(2)
        denom = (1.0 - gamma_sq).clamp(min=1e-10)

        # 化学计量约束
        Y_limit = torch.min(Yf, Yo / self.stoich_ratio)

        # 燃烧源项
        Su = rho_t * gamma_sq / denom * Y_limit

        # 线性化 Sp
        Sp = torch.zeros_like(Su)

        return Su, Sp

    def __repr__(self) -> str:
        return (
            f"EDCModel(C_tau={self.C_tau}, tau_chem={self.tau_chem}, "
            f"stoich_ratio={self.stoich_ratio})"
        )


# ===================================================================
# 无限快化学
# ===================================================================


@CombustionModel.register("infinitelyFast")
class InfinitelyFastChemistry(CombustionModel):
    """无限快化学（混合受限）燃烧模型。

    假设化学反应速率远快于混合速率::

        source = rho * min(Y_fuel, Y_ox / s) / dt

    Parameters
    ----------
    dt : float
        特征时间步长 (s)。默认 1e-3。
    stoich_ratio : float
        化学计量比 (氧化剂/燃料质量比)。默认 1.0。

    Examples::

        model = InfinitelyFastChemistry(dt=1e-3, stoich_ratio=2.0)
        Su, Sp = model.source(Y_fuel=0.05, Y_ox=0.23, T=1000.0, rho=1.0)
    """

    def __init__(
        self,
        dt: float = 1e-3,
        stoich_ratio: float = 1.0,
    ) -> None:
        self.dt = dt
        self.stoich_ratio = stoich_ratio

    def source(
        self,
        Y_fuel: torch.Tensor | float,
        Y_ox: torch.Tensor | float,
        T: torch.Tensor | float,
        rho: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算无限快化学燃烧源项。

        Su = rho * min(Y_fuel, Y_ox/s) / dt

        Parameters
        ----------
        Y_fuel : torch.Tensor | float
            燃料质量分数。
        Y_ox : torch.Tensor | float
            氧化剂质量分数。
        T : torch.Tensor | float
            温度 (K) — 未使用，保持接口一致。
        rho : torch.Tensor | float
            密度 (kg/m³)。

        Returns
        -------
        Su, Sp : tuple[torch.Tensor, torch.Tensor]
        """
        T_t = _to_tensor(T)
        Yf = _to_tensor(Y_fuel, ref=T_t)
        Yo = _to_tensor(Y_ox, ref=T_t)
        rho_t = _to_tensor(rho, ref=T_t)

        Y_limit = torch.min(Yf, Yo / self.stoich_ratio)

        Su = rho_t * Y_limit / self.dt

        # 无限快化学与温度无关
        Sp = torch.zeros_like(Su)

        return Su, Sp

    def __repr__(self) -> str:
        return f"InfinitelyFastChemistry(dt={self.dt}, stoich_ratio={self.stoich_ratio})"


# ===================================================================
# FSD 模型
# ===================================================================


@CombustionModel.register("FSD")
class FSDModel(CombustionModel):
    """Flame Surface Density (FSD) 燃烧模型。

    用于预混火焰的湍流燃烧封闭，基于火焰面密度概念::

        Sigma = Sigma_0 * (1 + c * u'/S_L)
        source = rho * Sigma * S_L * min(Y_fuel, Y_ox/s)

    其中 Sigma 是火焰面密度，S_L 是层流火焰速度，
    u' 是湍流脉动速度，c 是模型常数。

    Parameters
    ----------
    S_L : float
        层流火焰速度 (m/s)。默认 0.4。
    Sigma_0 : float
        基准火焰面密度 (1/m)。默认 100.0。
    C_sigma : float
        湍流增强系数。默认 0.5。
    u_prime : float
        湍流脉动速度 (m/s)。默认 1.0。
    stoich_ratio : float
        化学计量比 (氧化剂/燃料质量比)。默认 1.0。

    Examples::

        fsd = FSDModel(S_L=0.4, Sigma_0=100.0, C_sigma=0.5)
        Su, Sp = fsd.source(Y_fuel=0.05, Y_ox=0.23, T=1000.0, rho=1.0)
    """

    def __init__(
        self,
        S_L: float = 0.4,
        Sigma_0: float = 100.0,
        C_sigma: float = 0.5,
        u_prime: float = 1.0,
        stoich_ratio: float = 1.0,
    ) -> None:
        if S_L <= 0:
            raise ValueError(f"S_L must be positive, got {S_L}")
        if Sigma_0 <= 0:
            raise ValueError(f"Sigma_0 must be positive, got {Sigma_0}")

        self.S_L = S_L
        self.Sigma_0 = Sigma_0
        self.C_sigma = C_sigma
        self.u_prime = u_prime
        self.stoich_ratio = stoich_ratio

    def source(
        self,
        Y_fuel: torch.Tensor | float,
        Y_ox: torch.Tensor | float,
        T: torch.Tensor | float,
        rho: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算 FSD 燃烧源项。

        Sigma = Sigma_0 * (1 + C_sigma * u' / S_L)
        Su    = rho * Sigma * S_L * min(Y_fuel, Y_ox/s)

        Parameters
        ----------
        Y_fuel : torch.Tensor | float
            燃料质量分数。
        Y_ox : torch.Tensor | float
            氧化剂质量分数。
        T : torch.Tensor | float
            温度 (K) — 未使用，保持接口一致。
        rho : torch.Tensor | float
            密度 (kg/m³)。

        Returns
        -------
        Su, Sp : tuple[torch.Tensor, torch.Tensor]
        """
        T_t = _to_tensor(T)
        Yf = _to_tensor(Y_fuel, ref=T_t)
        Yo = _to_tensor(Y_ox, ref=T_t)
        rho_t = _to_tensor(rho, ref=T_t)

        # 火焰面密度（考虑湍流增强）
        Sigma = self.Sigma_0 * (1.0 + self.C_sigma * self.u_prime / self.S_L)

        # 化学计量约束
        Y_limit = torch.min(Yf, Yo / self.stoich_ratio)

        # 燃烧源项: Su = rho * Sigma * S_L * Y_limit
        Su = rho_t * Sigma * self.S_L * Y_limit

        # FSD 模型与温度无关（预混火焰面传播）
        Sp = torch.zeros_like(Su)

        return Su, Sp

    def __repr__(self) -> str:
        return (
            f"FSDModel(S_L={self.S_L}, Sigma_0={self.Sigma_0}, "
            f"C_sigma={self.C_sigma}, u_prime={self.u_prime}, "
            f"stoich_ratio={self.stoich_ratio})"
        )
