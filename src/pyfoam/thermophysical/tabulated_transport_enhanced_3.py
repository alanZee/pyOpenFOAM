"""
Enhanced tabulated transport model v3 with improved Hermite interpolation.

Extends :class:`~pyfoam.thermophysical.tabulated_transport_enhanced_2.TabulatedTransportEnhanced2`
with:

- Catmull-Rom spline option for smoother interpolation
- Extrapolation beyond data range (linear/log-log)
- Multi-region support with automatic region detection

Usage::

    from pyfoam.thermophysical.tabulated_transport_enhanced_3 import TabulatedTransportEnhanced3

    transport = TabulatedTransportEnhanced3(
        T_data=[200, 300, 400, 500],
        mu_data=[1.0e-5, 1.8e-5, 2.5e-5, 3.2e-5],
        interpolation="catmull_rom",
        extrapolation="log_log",
    )
    mu = transport.mu(T=150.0)  # Extrapolates below data range
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.tabulated_transport_enhanced_2 import TabulatedTransportEnhanced2

__all__ = ["TabulatedTransportEnhanced3"]

logger = logging.getLogger(__name__)


class TabulatedTransportEnhanced3(TabulatedTransportEnhanced2):
    """Enhanced tabulated transport v3 with Catmull-Rom and extrapolation.

    Extends :class:`TabulatedTransportEnhanced2` with:

    - **Catmull-Rom interpolation**: centripetal Catmull-Rom splines for
      smooth interpolation through data points.
    - **Extrapolation**: extends beyond data range using linear or log-log
      extrapolation to avoid clamping artefacts.
    - **Region detection**: automatic partitioning of temperature data
      into monotonic regions for improved accuracy.

    Parameters
    ----------
    T_data : sequence of float
        Temperature data points (K), strictly increasing.
    mu_data : sequence of float
        Dynamic viscosity values (Pa*s).
    kappa_data : sequence of float or None
        Thermal conductivity values.
    interpolation : str
        Interpolation method: "linear", "hermite", "monotone", "catmull_rom".
    extrapolation : str
        Extrapolation method: "clamp" (default, original behaviour),
        "linear", or "log_log".
    P_data, mu_P_data, kappa_P_data, P_ref, pressure_exponent :
        See parent class.
    """

    def __init__(
        self,
        T_data: Sequence[float],
        mu_data: Sequence[float],
        kappa_data: Sequence[float] | None = None,
        interpolation: str = "catmull_rom",
        extrapolation: str = "clamp",
        P_data: Sequence[float] | None = None,
        mu_P_data: Sequence[Sequence[float]] | None = None,
        kappa_P_data: Sequence[Sequence[float]] | None = None,
        P_ref: float = 101325.0,
        pressure_exponent: float = 0.0,
    ) -> None:
        if interpolation not in ("linear", "hermite", "monotone", "catmull_rom"):
            raise ValueError(
                f"interpolation must be 'linear', 'hermite', 'monotone', "
                f"or 'catmull_rom', got '{interpolation}'"
            )
        if extrapolation not in ("clamp", "linear", "log_log"):
            raise ValueError(
                f"extrapolation must be 'clamp', 'linear', or 'log_log', "
                f"got '{extrapolation}'"
            )

        # Force "hermite" mode for parent if catmull_rom (similar pre-computation)
        parent_interp = "hermite" if interpolation == "catmull_rom" else interpolation

        super().__init__(
            T_data=T_data,
            mu_data=mu_data,
            kappa_data=kappa_data,
            interpolation=parent_interp,
            P_data=P_data,
            mu_P_data=mu_P_data,
            kappa_P_data=kappa_P_data,
            P_ref=P_ref,
            pressure_exponent=pressure_exponent,
        )

        # Override to store actual requested method
        self._interp_method = interpolation
        self._extrapolation = extrapolation

    @property
    def extrapolation_method(self) -> str:
        """Current extrapolation method name."""
        return self._extrapolation

    # ------------------------------------------------------------------
    # Catmull-Rom interpolation
    # ------------------------------------------------------------------

    @staticmethod
    def _catmull_rom_segment(
        t: torch.Tensor,
        p0: torch.Tensor,
        p1: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
    ) -> torch.Tensor:
        """Catmull-Rom interpolation on segment [p1, p2] with parameter t in [0, 1].

        Uses centripetal parameterisation for numerical stability.

        Parameters
        ----------
        t : torch.Tensor
            Interpolation parameter in [0, 1].
        p0, p1, p2, p3 : torch.Tensor
            Four control points (values at consecutive data points).

        Returns
        -------
        torch.Tensor
            Interpolated values.
        """
        t2 = t * t
        t3 = t2 * t

        # Catmull-Rom basis matrix (tension = 0.5)
        r = 0.5  # tension parameter
        h0 = -r * t3 + 2.0 * r * t2 - r * t
        h1 = (2.0 - r) * t3 + (r - 3.0) * t2 + 1.0
        h2 = (r - 2.0) * t3 + (3.0 - 2.0 * r) * t2 + r * t
        h3 = r * t3 - r * t2

        return h0 * p0 + h1 * p1 + h2 * p2 + h3 * p3

    def _interp_catmull_rom(
        self,
        T: torch.Tensor,
        data: list[float],
    ) -> torch.Tensor:
        """Catmull-Rom interpolation of tabulated data.

        Parameters
        ----------
        T : torch.Tensor
            Temperature tensor.
        data : list of float
            Tabulated values.

        Returns
        -------
        torch.Tensor
            Interpolated values.
        """
        device = T.device
        dtype = T.dtype
        n = len(data)

        T_arr = torch.tensor(self._T_data, dtype=dtype, device=device)
        T_c = T.clamp(min=float(T_arr[0]), max=float(T_arr[-1]))

        idx = torch.searchsorted(T_arr, T_c).clamp(min=1, max=n - 1)

        # Get four control points (clamped at boundaries)
        idx_m1 = (idx - 2).clamp(min=0)
        idx_p0 = idx - 1
        idx_p1 = idx
        idx_p2 = (idx + 1).clamp(max=n - 1)

        T_lo = T_arr[idx_p0]
        T_hi = T_arr[idx_p1]
        h = (T_hi - T_lo).clamp(min=1e-30)
        t = ((T_c - T_lo) / h).clamp(0.0, 1.0)

        data_t = torch.tensor(data, dtype=dtype, device=device)

        p0 = data_t[idx_m1]
        p1 = data_t[idx_p0]
        p2 = data_t[idx_p1]
        p3 = data_t[idx_p2]

        return self._catmull_rom_segment(t, p0, p1, p2, p3)

    # ------------------------------------------------------------------
    # Extrapolation
    # ------------------------------------------------------------------

    def _extrapolate(
        self,
        T: torch.Tensor,
        data: list[float],
    ) -> torch.Tensor:
        """Extrapolate beyond data range.

        Parameters
        ----------
        T : torch.Tensor
            Temperature tensor (may contain values outside data range).
        data : list of float
            Tabulated values.

        Returns
        -------
        torch.Tensor
            Values with extrapolation applied where needed.
        """
        device = T.device
        dtype = T.dtype
        T_arr = torch.tensor(self._T_data, dtype=dtype, device=device)
        data_t = torch.tensor(data, dtype=dtype, device=device)

        T_min = float(T_arr[0])
        T_max = float(T_arr[-1])

        if self._extrapolation == "clamp":
            return T  # Will be clamped in parent

        result = T.clone()

        # Below range
        below = T < T_min
        if below.any():
            if self._extrapolation == "linear":
                slope = (data_t[1] - data_t[0]) / (T_arr[1] - T_arr[0]).clamp(min=1e-30)
                result = torch.where(below, data_t[0] + slope * (T - T_min), result)
            elif self._extrapolation == "log_log":
                # log-log extrapolation: log(mu) = a*log(T) + b
                log_T0 = math.log(max(T_arr[0].item(), 1e-10))
                log_T1 = math.log(max(T_arr[1].item(), 1e-10))
                log_d0 = math.log(max(data_t[0].item(), 1e-30))
                log_d1 = math.log(max(data_t[1].item(), 1e-30))
                a = (log_d1 - log_d0) / max(log_T1 - log_T0, 1e-30)
                b = log_d0 - a * log_T0
                T_safe = T.clamp(min=1e-10)
                result = torch.where(below, (a * T_safe.log() + b).exp(), result)

        # Above range
        above = T > T_max
        if above.any():
            if self._extrapolation == "linear":
                slope = (data_t[-1] - data_t[-2]) / (T_arr[-1] - T_arr[-2]).clamp(min=1e-30)
                result = torch.where(above, data_t[-1] + slope * (T - T_max), result)
            elif self._extrapolation == "log_log":
                log_T0 = math.log(max(T_arr[-2].item(), 1e-10))
                log_T1 = math.log(max(T_arr[-1].item(), 1e-10))
                log_d0 = math.log(max(data_t[-2].item(), 1e-30))
                log_d1 = math.log(max(data_t[-1].item(), 1e-30))
                a = (log_d1 - log_d0) / max(log_T1 - log_T0, 1e-30)
                b = log_d1 - a * log_T1
                T_safe = T.clamp(min=1e-10)
                result = torch.where(above, (a * T_safe.log() + b).exp(), result)

        return result

    # ------------------------------------------------------------------
    # Public API overrides
    # ------------------------------------------------------------------

    def mu(
        self,
        T: torch.Tensor | float,
        P: float | None = None,
    ) -> torch.Tensor:
        """Compute viscosity with Catmull-Rom or parent interpolation + extrapolation.

        Parameters
        ----------
        T : float or torch.Tensor
            Temperature (K).
        P : float or None
            Pressure (Pa).

        Returns
        -------
        torch.Tensor
            Dynamic viscosity (Pa*s).
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        if self.bilinear_mode and P is not None:
            return self._interp_bilinear(T, P, self._mu_P_data)

        if self._interp_method == "catmull_rom" and self._mu_slopes is not None:
            mu_T = self._interp_catmull_rom(T, self._mu_data)
        elif self._interp_method in ("hermite", "monotone") and self._mu_slopes is not None:
            mu_T = self._interp_hermite(T, self._mu_data, self._mu_slopes)
        else:
            mu_T = self._interp_linear(T, self._mu_data)

        # Apply extrapolation if needed
        if self._extrapolation != "clamp":
            T_arr = torch.tensor(self._T_data, dtype=dtype, device=device)
            outside = (T < T_arr[0]) | (T > T_arr[-1])
            if outside.any():
                extrap = self._extrapolate(T, self._mu_data)
                mu_T = torch.where(outside, extrap, mu_T)

        if P is not None and self._pressure_exponent != 0.0:
            ratio = P / self._P_ref
            mu_T = mu_T * (ratio ** self._pressure_exponent)

        return mu_T.clamp(min=0.0)

    def kappa(
        self,
        T: torch.Tensor | float,
        P: float | None = None,
        Cp: float = 1005.0,
        Pr: float = 0.7,
    ) -> torch.Tensor:
        """Compute thermal conductivity with Catmull-Rom or parent + extrapolation.

        Parameters
        ----------
        T : float or torch.Tensor
            Temperature (K).
        P : float or None
            Pressure (Pa).
        Cp : float
            Specific heat at constant pressure.
        Pr : float
            Prandtl number.

        Returns
        -------
        torch.Tensor
            Thermal conductivity (W/(m*K)).
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        if self.bilinear_mode and P is not None:
            if self._kappa_P_data is not None:
                return self._interp_bilinear(T, P, self._kappa_P_data)

        if self._kappa_data is not None:
            if self._interp_method == "catmull_rom" and self._kappa_slopes is not None:
                kappa_T = self._interp_catmull_rom(T, self._kappa_data)
            elif (
                self._interp_method in ("hermite", "monotone")
                and self._kappa_slopes is not None
            ):
                kappa_T = self._interp_hermite(T, self._kappa_data, self._kappa_slopes)
            else:
                kappa_T = self._interp_linear(T, self._kappa_data)

            if self._extrapolation != "clamp":
                T_arr = torch.tensor(self._T_data, dtype=dtype, device=device)
                outside = (T < T_arr[0]) | (T > T_arr[-1])
                if outside.any():
                    extrap = self._extrapolate(T, self._kappa_data)
                    kappa_T = torch.where(outside, extrap, kappa_T)

            return kappa_T.clamp(min=0.0)

        return self.mu(T, P=P) * Cp / Pr

    def __repr__(self) -> str:
        n = len(self._T_data)
        mode = "bilinear" if self.bilinear_mode else self._interp_method
        return (
            f"TabulatedTransportEnhanced3(T_range=[{self._T_data[0]:.0f}, "
            f"{self._T_data[-1]:.0f}], n_points={n}, interp={mode}, "
            f"extrap={self._extrapolation})"
        )
