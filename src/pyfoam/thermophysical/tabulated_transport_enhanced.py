"""
Enhanced tabulated transport model with temperature and pressure dependence.

Extends :class:`~pyfoam.thermophysical.tabulated_transport.TabulatedTransport`
with:

- Bilinear interpolation over (T, P) data tables
- Optional pressure correction factor for viscosity
- Improved clamping and extrapolation control

Usage::

    from pyfoam.thermophysical.tabulated_transport_enhanced import TabulatedTransportEnhanced

    transport = TabulatedTransportEnhanced(
        T_data=[200, 300, 400, 500],
        mu_data=[1.0e-5, 1.8e-5, 2.5e-5, 3.2e-5],
        P_data=[1e5, 5e5, 1e6],
        mu_P_data=[
            [0.95e-5, 1.75e-5, 2.45e-5, 3.15e-5],
            [1.0e-5,  1.8e-5,  2.5e-5,  3.2e-5],
            [1.05e-5, 1.85e-5, 2.55e-5, 3.25e-5],
        ],
    )
    mu = transport.mu(T=350.0, P=3e5)  # bilinear interpolated
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.tabulated_transport import TabulatedTransport

__all__ = ["TabulatedTransportEnhanced"]

logger = logging.getLogger(__name__)


class TabulatedTransportEnhanced(TabulatedTransport):
    """Enhanced tabulated transport with pressure dependence.

    Supports two modes:

    1. **Temperature-only** (default): behaves like the parent
       :class:`TabulatedTransport`, with an optional pressure correction
       factor ``mu_ref * (P / P_ref)^alpha``.

    2. **Bilinear (T, P)**: if ``P_data`` and ``mu_P_data`` are provided,
       uses bilinear interpolation over the (T, P) grid.

    Parameters
    ----------
    T_data : sequence of float
        Temperature data points (K), strictly increasing, at least 2.
    mu_data : sequence of float
        Dynamic viscosity values (Pa·s) at reference pressure.
    kappa_data : sequence of float or None
        Thermal conductivity values (W/(m·K)) at reference pressure.
    P_data : sequence of float or None
        Pressure data points (Pa) for bilinear mode. If None, uses
        T-only mode with optional pressure correction.
    mu_P_data : sequence of sequence of float or None
        2D viscosity table ``(len(P_data), len(T_data))`` for bilinear
        interpolation. Required if ``P_data`` is provided.
    kappa_P_data : sequence of sequence of float or None
        2D thermal conductivity table for bilinear mode.
    P_ref : float
        Reference pressure (Pa) for pressure correction mode.
        Default 101325.
    pressure_exponent : float
        Exponent for pressure correction: mu ~ (P/P_ref)^alpha.
        Default 0.0 (no correction).

    Examples::

        # T-only with pressure correction
        transport = TabulatedTransportEnhanced(
            T_data=[200, 300, 400, 500],
            mu_data=[1.0e-5, 1.8e-5, 2.5e-5, 3.2e-5],
            pressure_exponent=0.01,
        )
        mu = transport.mu(T=350.0, P=2e5)
    """

    def __init__(
        self,
        T_data: Sequence[float],
        mu_data: Sequence[float],
        kappa_data: Sequence[float] | None = None,
        P_data: Sequence[float] | None = None,
        mu_P_data: Sequence[Sequence[float]] | None = None,
        kappa_P_data: Sequence[Sequence[float]] | None = None,
        P_ref: float = 101325.0,
        pressure_exponent: float = 0.0,
    ) -> None:
        super().__init__(T_data=T_data, mu_data=mu_data, kappa_data=kappa_data)

        # Bilinear mode validation
        if P_data is not None:
            if len(P_data) < 2:
                raise ValueError("P_data must have at least 2 points")
            for i in range(1, len(P_data)):
                if P_data[i] <= P_data[i - 1]:
                    raise ValueError(
                        f"P_data must be strictly increasing, "
                        f"got P[{i-1}]={P_data[i-1]} >= P[{i}]={P_data[i]}"
                    )
            if mu_P_data is None:
                raise ValueError("mu_P_data required when P_data is provided")
            if len(mu_P_data) != len(P_data):
                raise ValueError(
                    f"mu_P_data rows ({len(mu_P_data)}) must match "
                    f"P_data length ({len(P_data)})"
                )
            for row_idx, row in enumerate(mu_P_data):
                if len(row) != len(T_data):
                    raise ValueError(
                        f"mu_P_data[{row_idx}] length ({len(row)}) must "
                        f"match T_data length ({len(T_data)})"
                    )

        self._P_data = list(P_data) if P_data is not None else None
        self._mu_P_data = (
            [list(row) for row in mu_P_data] if mu_P_data is not None else None
        )
        self._kappa_P_data = (
            [list(row) for row in kappa_P_data]
            if kappa_P_data is not None
            else None
        )
        self._P_ref = P_ref
        self._pressure_exponent = pressure_exponent

    @property
    def bilinear_mode(self) -> bool:
        """Whether bilinear (T, P) interpolation is active."""
        return self._P_data is not None

    @property
    def P_data(self) -> list[float] | None:
        """Pressure data points (Pa) or None."""
        return self._P_data.copy() if self._P_data is not None else None

    # ------------------------------------------------------------------
    # Bilinear interpolation
    # ------------------------------------------------------------------

    def _interp_bilinear(
        self,
        T: torch.Tensor,
        P: float,
        table: list[list[float]],
    ) -> torch.Tensor:
        """Bilinear interpolation over (T, P) grid.

        Args:
            T: Temperature tensor.
            P: Pressure (scalar, Pa).
            table: 2D data table ``(len(P_data), len(T_data))``.

        Returns:
            Interpolated values.
        """
        device = T.device
        dtype = T.dtype

        T_arr = torch.tensor(self._T_data, dtype=dtype, device=device)
        P_arr = torch.tensor(self._P_data, dtype=dtype, device=device)

        # Clamp T and P to data range
        T_c = T.clamp(min=float(T_arr[0]), max=float(T_arr[-1]))
        P_c = max(min(P, float(P_arr[-1])), float(P_arr[0]))

        # Find T interval
        idx_T = torch.searchsorted(T_arr, T_c).clamp(
            min=1, max=len(self._T_data) - 1
        )

        # Find P interval
        idx_P = int(torch.searchsorted(P_arr, torch.tensor(P_c, dtype=dtype)).item())
        idx_P = max(1, min(idx_P, len(self._P_data) - 1))

        # T interpolation weights
        T_lo = T_arr[idx_T - 1]
        T_hi = T_arr[idx_T]
        t = ((T_c - T_lo) / (T_hi - T_lo + 1e-30)).clamp(0, 1)

        # P interpolation weight
        P_lo = float(P_arr[idx_P - 1].item())
        P_hi = float(P_arr[idx_P].item())
        s = (P_c - P_lo) / (P_hi - P_lo + 1e-30)
        s = max(0.0, min(1.0, s))

        # Bilinear interpolation
        val_Tlo_Plo = torch.tensor(
            table[idx_P - 1], dtype=dtype, device=device
        )
        val_Thi_Plo = torch.tensor(
            table[idx_P - 1], dtype=dtype, device=device
        )
        val_Tlo_Phi = torch.tensor(
            table[idx_P], dtype=dtype, device=device
        )
        val_Thi_Phi = torch.tensor(
            table[idx_P], dtype=dtype, device=device
        )

        # Gather at the correct T indices
        f00 = val_Tlo_Plo[idx_T - 1]
        f10 = val_Tlo_Plo[idx_T]
        f01 = val_Tlo_Phi[idx_T - 1]
        f11 = val_Tlo_Phi[idx_T]

        # Bilinear: f(T,P) = (1-s)*[(1-t)*f00 + t*f10] + s*[(1-t)*f01 + t*f11]
        result = (1 - s) * ((1 - t) * f00 + t * f10) + s * ((1 - t) * f01 + t * f11)
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mu(
        self,
        T: torch.Tensor | float,
        P: float | None = None,
    ) -> torch.Tensor:
        """Compute dynamic viscosity with optional pressure dependence.

        In bilinear mode, performs (T, P) interpolation.
        In T-only mode, applies pressure correction factor if P is given:
            mu_corrected = mu(T) * (P / P_ref)^alpha

        Args:
            T: Temperature (K).
            P: Pressure (Pa). Optional. If None, returns T-only result.

        Returns:
            Dynamic viscosity (Pa·s).
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        if self.bilinear_mode and P is not None:
            return self._interp_bilinear(T, P, self._mu_P_data)

        # T-only mode
        mu_T = self._interp_linear(T, self._mu_data)

        if P is not None and self._pressure_exponent != 0.0:
            ratio = P / self._P_ref
            mu_T = mu_T * (ratio ** self._pressure_exponent)

        return mu_T

    def kappa(
        self,
        T: torch.Tensor | float,
        P: float | None = None,
        Cp: float = 1005.0,
        Pr: float = 0.7,
    ) -> torch.Tensor:
        """Compute thermal conductivity with optional pressure dependence.

        Args:
            T: Temperature (K).
            P: Pressure (Pa). Optional.
            Cp: Specific heat at constant pressure (J/(kg·K)).
            Pr: Prandtl number.

        Returns:
            Thermal conductivity (W/(m·K)).
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        if self.bilinear_mode and P is not None:
            if self._kappa_P_data is not None:
                return self._interp_bilinear(T, P, self._kappa_P_data)
            # Fall through to Pr-based

        if self._kappa_data is not None:
            return self._interp_linear(T, self._kappa_data)

        return self.mu(T, P=P) * Cp / Pr

    def __repr__(self) -> str:
        n = len(self._T_data)
        mode = "bilinear" if self.bilinear_mode else "T-only"
        return (
            f"TabulatedTransportEnhanced(T_range=[{self._T_data[0]:.0f}, "
            f"{self._T_data[-1]:.0f}], n_points={n}, mode={mode})"
        )
