"""
Tabulated transport model for viscosity and thermal conductivity.

Implements temperature-dependent dynamic viscosity and thermal conductivity
using linear interpolation from tabulated data, as used in OpenFOAM's
``tabulatedTransport`` class.

Usage::

    from pyfoam.thermophysical.tabulated_transport import TabulatedTransport

    transport = TabulatedTransport(
        T_data=[200, 300, 400, 500, 600],
        mu_data=[1.0e-5, 1.8e-5, 2.5e-5, 3.2e-5, 3.8e-5],
    )
    mu = transport.mu(T=350.0)  # interpolated
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.transport_model import TransportModel

__all__ = ["TabulatedTransport"]

logger = logging.getLogger(__name__)


class TabulatedTransport(TransportModel):
    """Tabulated viscosity and thermal conductivity model.

    Dynamic viscosity and (optionally) thermal conductivity are
    linearly interpolated from user-supplied data arrays as functions
    of temperature.

    Parameters
    ----------
    T_data : sequence of float
        Temperature values (K) — strictly increasing, at least 2 points.
    mu_data : sequence of float
        Dynamic viscosity values (Pa·s) corresponding to ``T_data``.
    kappa_data : sequence of float or None
        Thermal conductivity values (W/(m·K)) corresponding to ``T_data``.
        If None, thermal conductivity is computed from :math:`\\kappa = \\mu C_p / Pr`.

    Examples::

        transport = TabulatedTransport(
            T_data=[200, 300, 400, 500, 600],
            mu_data=[1.0e-5, 1.8e-5, 2.5e-5, 3.2e-5, 3.8e-5],
        )
        mu = transport.mu(T=350.0)
    """

    def __init__(
        self,
        T_data: Sequence[float],
        mu_data: Sequence[float],
        kappa_data: Sequence[float] | None = None,
    ) -> None:
        if len(T_data) < 2:
            raise ValueError("T_data must have at least 2 points")
        if len(mu_data) != len(T_data):
            raise ValueError(
                f"mu_data length ({len(mu_data)}) must match T_data length ({len(T_data)})"
            )
        if kappa_data is not None and len(kappa_data) != len(T_data):
            raise ValueError(
                f"kappa_data length ({len(kappa_data)}) must match T_data length ({len(T_data)})"
            )

        # 验证温度严格递增
        for i in range(1, len(T_data)):
            if T_data[i] <= T_data[i - 1]:
                raise ValueError(
                    f"T_data must be strictly increasing, got T[{i-1}]={T_data[i-1]} >= T[{i}]={T_data[i]}"
                )

        self._T_data = list(T_data)
        self._mu_data = list(mu_data)
        self._kappa_data = list(kappa_data) if kappa_data is not None else None

    def _interp_linear(
        self,
        T: torch.Tensor,
        y_data: list[float],
    ) -> torch.Tensor:
        """Linearly interpolate y(T) from tabulated data.

        Values outside the data range are clamped to the boundary values.

        Args:
            T: Temperature tensor.
            y_data: y values corresponding to ``self._T_data``.

        Returns:
            Interpolated y values.
        """
        device = T.device
        dtype = T.dtype

        T_arr = torch.tensor(self._T_data, dtype=dtype, device=device)
        y_arr = torch.tensor(y_data, dtype=dtype, device=device)

        # 裁剪到数据范围
        T_clamped = T.clamp(min=float(T_arr[0]), max=float(T_arr[-1]))

        # 找到插值区间
        idx = torch.searchsorted(T_arr, T_clamped).clamp(
            min=1, max=len(self._T_data) - 1
        )

        T_lo = T_arr[idx - 1]
        T_hi = T_arr[idx]
        y_lo = y_arr[idx - 1]
        y_hi = y_arr[idx]

        # 线性插值
        t = ((T_clamped - T_lo) / (T_hi - T_lo + 1e-30)).clamp(0, 1)
        return y_lo + t * (y_hi - y_lo)

    def mu(self, T: torch.Tensor | float) -> torch.Tensor:
        """Compute dynamic viscosity by linear interpolation.

        Args:
            T: Temperature (K) — scalar or ``(n_cells,)`` tensor.

        Returns:
            Dynamic viscosity (Pa·s).
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        return self._interp_linear(T, self._mu_data)

    def kappa(
        self,
        T: torch.Tensor | float,
        Cp: float = 1005.0,
        Pr: float = 0.7,
    ) -> torch.Tensor:
        """Compute thermal conductivity.

        If ``kappa_data`` was provided, interpolates from the table.
        Otherwise:

        .. math::

            \\kappa = \\frac{\\mu \\cdot C_p}{Pr}

        Args:
            T: Temperature (K).
            Cp: Specific heat at constant pressure (J/(kg·K)).
            Pr: Prandtl number.

        Returns:
            Thermal conductivity (W/(m·K)).
        """
        if self._kappa_data is not None:
            device = get_device()
            dtype = get_default_dtype()
            if not isinstance(T, torch.Tensor):
                T = torch.tensor(T, dtype=dtype, device=device)
            return self._interp_linear(T, self._kappa_data)

        return self.mu(T) * Cp / Pr

    @property
    def T_data(self) -> list[float]:
        """Temperature data points."""
        return self._T_data.copy()

    @property
    def mu_data(self) -> list[float]:
        """Viscosity data points."""
        return self._mu_data.copy()

    @property
    def kappa_data(self) -> list[float] | None:
        """Thermal conductivity data points (or None)."""
        return self._kappa_data.copy() if self._kappa_data is not None else None

    def __repr__(self) -> str:
        n = len(self._T_data)
        return (
            f"TabulatedTransport(T_range=[{self._T_data[0]:.0f}, {self._T_data[-1]:.0f}], "
            f"n_points={n}, kappa={'tabulated' if self._kappa_data else 'Pr-based'})"
        )
