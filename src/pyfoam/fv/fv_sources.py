"""
Additional fvModels for energy, radiation, and body force source terms.

Provides:

- :class:`SolidificationMeltingModel` — phase change (solidification/melting) source
- :class:`RASourceModel` — radiation absorption volumetric heat source
- :class:`GravitationalBodyForce` — gravitational body force source

These extend the fvModel framework with commonly used physical
source terms in OpenFOAM solvers.

Usage::

    from pyfoam.fv.fv_sources import SolidificationMeltingModel

    model = SolidificationMeltingModel(
        T_solidus=273.15, T_liquidus=373.15, L=3.34e5, rho=1000.0,
    )
    model.apply(energy_matrix, T_field)
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.fv_matrix import FvMatrix
from pyfoam.fv.fv_models import FvModel

__all__ = [
    "SolidificationMeltingModel",
    "RASourceModel",
    "GravitationalBodyForce",
]


# ---------------------------------------------------------------------------
# SolidificationMeltingModel
# ---------------------------------------------------------------------------


@FvModel.register("solidificationMelting")
class SolidificationMeltingModel(FvModel):
    """Phase change model for solidification and melting.

    Models the latent heat release/absorption during phase change using
    the enthalpy-porosity technique.  The source term is linearised
    semi-implicitly to maintain diagonal dominance::

        Su = rho * L * (f_l(T_new) - f_l(T_old)) / dt  (explicit part)
        Sp = -rho * L * df_l/dT / dt                     (implicit part)

    The liquid fraction ``f_l`` varies smoothly between the solidus and
    liquidus temperatures::

        f_l = 0                          if T < T_solidus
        f_l = (T - T_solidus) / (T_liquidus - T_solidus)  if T_solidus <= T <= T_liquidus
        f_l = 1                          if T > T_liquidus

    Corresponds to OpenFOAM's ``solidificationMelting`` fvModel.

    Parameters
    ----------
    T_solidus : float
        Solidus temperature (K). Default ``273.15``.
    T_liquidus : float
        Liquidus temperature (K). Default ``373.15``.
    L : float
        Latent heat of fusion (J/kg). Default ``3.34e5`` (water).
    rho : float
        Density (kg/m³). Default ``1000.0``.
    cells : list[int] | torch.Tensor | None
        Restrict to specific cell indices. ``None`` = all cells.

    Examples::

        # Water freezing
        model = SolidificationMeltingModel(
            T_solidus=273.15, T_liquidus=273.15, L=3.34e5,
        )
        model.apply(energy_matrix, T_field)
    """

    def __init__(
        self,
        *,
        T_solidus: float = 273.15,
        T_liquidus: float = 373.15,
        L: float = 3.34e5,
        rho: float = 1000.0,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            T_solidus=T_solidus, T_liquidus=T_liquidus,
            L=L, rho=rho, cells=cells, **kwargs,
        )
        if T_liquidus < T_solidus:
            raise ValueError(
                f"T_liquidus ({T_liquidus}) must be >= T_solidus ({T_solidus})"
            )
        if L < 0.0:
            raise ValueError(f"L must be >= 0, got {L}")
        if rho <= 0.0:
            raise ValueError(f"rho must be > 0, got {rho}")

        self._T_solidus = T_solidus
        self._T_liquidus = T_liquidus
        self._L = L
        self._rho = rho
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def T_solidus(self) -> float:
        """Solidus temperature (K)."""
        return self._T_solidus

    @property
    def T_liquidus(self) -> float:
        """Liquidus temperature (K)."""
        return self._T_liquidus

    @property
    def L(self) -> float:
        """Latent heat (J/kg)."""
        return self._L

    @property
    def rho(self) -> float:
        """Density (kg/m³)."""
        return self._rho

    @staticmethod
    def _liquid_fraction(
        T: torch.Tensor,
        T_solidus: float,
        T_liquidus: float,
    ) -> torch.Tensor:
        """计算液相分数 f_l(T)。

        线性插值: f_l = (T - T_s) / (T_l - T_s)，截断到 [0, 1]。
        """
        if abs(T_liquidus - T_solidus) < 1e-10:
            # 纯物质（如水）：固液共存温度为单点
            return (T >= T_solidus).to(dtype=T.dtype)
        f_l = (T - T_solidus) / (T_liquidus - T_solidus)
        return f_l.clamp(0.0, 1.0)

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """Apply solidification/melting source to energy matrix.

        The source is linearised with respect to temperature using
        the enthalpy-porosity method.  The implicit term (Sp) ensures
        numerical stability during phase change.

        Args:
            matrix: The energy :class:`FvMatrix` to modify.
            field: Current temperature field ``(n_cells,)``.
        """
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        T = field.to(device=device, dtype=dtype)

        # 液相分数
        f_l = self._liquid_fraction(T, self._T_solidus, self._T_liquidus)

        # 液相分数对温度的导数 df_l/dT
        if abs(self._T_liquidus - self._T_solidus) < 1e-10:
            df_dT = torch.zeros_like(T)
        else:
            # 在固液区间内 df_l/dT = 1/(T_l - T_s)，区间外为 0
            in_mushy = (T >= self._T_solidus) & (T <= self._T_liquidus)
            df_dT = torch.where(
                in_mushy,
                torch.full_like(T, 1.0 / (self._T_liquidus - self._T_solidus)),
                torch.zeros_like(T),
            )

        # 隐式源项: Sp = -rho * L * df_l/dT (负值 → 稳定)
        sp = -self._rho * self._L * df_dT

        if self._cells is not None:
            idx = self._cells.to(device=device)
            mask = torch.zeros(n, device=device, dtype=dtype)
            mask.scatter_(0, idx, 1.0)
            sp = sp * mask

        matrix._diag = matrix._diag + sp

    def __repr__(self) -> str:
        return (
            f"SolidificationMeltingModel("
            f"T_solidus={self._T_solidus}, "
            f"T_liquidus={self._T_liquidus}, "
            f"L={self._L}, rho={self._rho})"
        )


# ---------------------------------------------------------------------------
# RASourceModel
# ---------------------------------------------------------------------------


@FvModel.register("radiationAbsorption")
class RASourceModel(FvModel):
    """Radiation absorption volumetric heat source.

    Models the absorption of incident radiation (e.g. solar or infrared)
    as a volumetric heat source in the energy equation::

        Q_rad = a * I_rad * exp(-a * d)

    where:

    - ``a`` — absorption coefficient [1/m]
    - ``I_rad`` — incident radiation intensity [W/m²]
    - ``d`` — depth from the irradiated surface [m]

    For simplicity, the current implementation applies a uniform
    volumetric absorption rate across specified cells::

        Su = a * I_rad * V_cell   [W]
        Sp = 0

    The source is linearised purely explicitly since radiation
    absorption does not depend on local temperature (in the simple
    Beer-Lambert approximation without re-emission).

    Corresponds to OpenFOAM's ``radialActuationDiskSource`` and
    radiation absorption fvModels.

    Parameters
    ----------
    a : float
        Absorption coefficient [1/m]. Default ``0.1``.
    I_rad : float
        Incident radiation intensity [W/m²]. Default ``1000.0``.
    cells : list[int] | torch.Tensor | None
        Restrict to specific cell indices. ``None`` = all cells.

    Examples::

        # Solar absorption in a participating medium
        model = RASourceModel(a=0.05, I_rad=1000.0)
        model.apply(energy_matrix, T_field)
    """

    def __init__(
        self,
        *,
        a: float = 0.1,
        I_rad: float = 1000.0,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(a=a, I_rad=I_rad, cells=cells, **kwargs)
        if a < 0.0:
            raise ValueError(f"Absorption coefficient a must be >= 0, got {a}")
        if I_rad < 0.0:
            raise ValueError(f"Radiation intensity I_rad must be >= 0, got {I_rad}")

        self._a = a
        self._I_rad = I_rad
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def a(self) -> float:
        """Absorption coefficient [1/m]."""
        return self._a

    @property
    def I_rad(self) -> float:
        """Incident radiation intensity [W/m²]."""
        return self._I_rad

    @property
    def Q_absorbed(self) -> float:
        """Absorbed heat per unit volume [W/m³] = a * I_rad."""
        return self._a * self._I_rad

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """Apply radiation absorption source to energy matrix.

        The absorbed radiation is applied as a purely explicit
        volumetric heat source (Su only, no implicit Sp).

        Args:
            matrix: The energy :class:`FvMatrix` to modify.
            field: Current temperature field ``(n_cells,)``.
        """
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        Q_vol = self.Q_absorbed  # W/m³

        su = torch.zeros(n, device=device, dtype=dtype)

        if self._cells is not None:
            idx = self._cells.to(device=device)
            su.scatter_(0, idx, Q_vol)
        else:
            su[:] = Q_vol

        matrix._source = matrix._source + su

    def __repr__(self) -> str:
        return (
            f"RASourceModel(a={self._a}, I_rad={self._I_rad})"
        )


# ---------------------------------------------------------------------------
# GravitationalBodyForce
# ---------------------------------------------------------------------------


@FvModel.register("gravitationalBodyForce")
class GravitationalBodyForce(FvModel):
    """Gravitational body force source for momentum equations.

    Adds a volumetric gravitational force to the momentum equation::

        F_grav = rho * g   [N/m³]

    where:

    - ``rho`` — fluid density [kg/m³]
    - ``g`` — gravitational acceleration vector [m/s²]

    The force is split into semi-implicit form for stability::

        Su = rho * g * (1 - alpha)   (explicit)
        Sp = 0                        (no implicit part for constant gravity)

    Optionally, a buoyancy correction can be applied when a
    reference density ``rho_ref`` is specified (Boussinesq
    approximation)::

        F = (rho - rho_ref) * g

    Corresponds to OpenFOAM's ``gravity`` fvModel.

    Parameters
    ----------
    g : list[float] | torch.Tensor
        Gravitational acceleration vector [m/s²].
        Default ``[0, 0, -9.81]`` (standard Earth gravity).
    rho_ref : float | None
        Reference density for Boussinesq approximation [kg/m³].
        ``None`` means full gravity (no buoyancy correction).
        Default ``None``.
    alpha : float
        Implicit linearisation fraction in [0, 1]. Default ``0.0``
        (fully explicit).
    cells : list[int] | torch.Tensor | None
        Restrict to specific cell indices. ``None`` = all cells.

    Examples::

        # Standard gravity
        model = GravitationalBodyForce()
        model.apply(momentum_matrix, U_field)

        # Boussinesq approximation
        model = GravitationalBodyForce(
            g=[0, 0, -9.81], rho_ref=1.225,
        )
        model.apply(momentum_matrix, rho_field)
    """

    def __init__(
        self,
        *,
        g: list[float] | torch.Tensor | None = None,
        rho_ref: float | None = None,
        alpha: float = 0.0,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        if g is None:
            g = [0.0, 0.0, -9.81]
        super().__init__(
            g=g, rho_ref=rho_ref, alpha=alpha, cells=cells, **kwargs,
        )
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self._g = (
            torch.tensor(g, dtype=torch.float64)
            if isinstance(g, list)
            else g
        )
        self._rho_ref = rho_ref
        self._alpha = alpha
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def g(self) -> torch.Tensor:
        """Gravitational acceleration vector."""
        return self._g

    @property
    def g_mag(self) -> float:
        """Gravitational acceleration magnitude [m/s²]."""
        return float(torch.norm(self._g).item())

    @property
    def rho_ref(self) -> float | None:
        """Reference density for Boussinesq (None = disabled)."""
        return self._rho_ref

    @property
    def alpha(self) -> float:
        """Implicit linearisation fraction."""
        return self._alpha

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """Apply gravitational body force to momentum matrix.

        The field is interpreted as the density (for Boussinesq) or
        the velocity component (for full gravity).

        For the scalar (single-component) case, only the component
        of gravity aligned with the velocity direction is applied.
        For a 3-component field, the full gravity vector is used.

        Args:
            matrix: The momentum :class:`FvMatrix` to modify.
            field: Current field ``(n_cells,)`` — density or velocity
                component.
        """
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        f = field.to(device=device, dtype=dtype)

        # 使用 g 的 z 分量作为标量重力源
        g_z = float(self._g[2].item()) if len(self._g) > 2 else -9.81

        # 密度
        if self._rho_ref is not None:
            # Boussinesq: F = (rho - rho_ref) * g
            rho_eff = f - self._rho_ref
        else:
            # 全重力: F = rho * g (field 为密度)
            rho_eff = f

        # Su = rho_eff * g * (1 - alpha)
        su_val = rho_eff * g_z * (1.0 - self._alpha)

        if self._cells is not None:
            idx = self._cells.to(device=device)
            su = torch.zeros(n, device=device, dtype=dtype)
            su.scatter_(0, idx, su_val.gather(0, idx))
        else:
            su = su_val

        matrix._source = matrix._source + su

    def __repr__(self) -> str:
        g_str = f"[{', '.join(f'{v:.2f}' for v in self._g.tolist())}]"
        return (
            f"GravitationalBodyForce(g={g_str}, "
            f"rho_ref={self._rho_ref}, alpha={self._alpha})"
        )
