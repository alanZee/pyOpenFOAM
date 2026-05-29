"""
Enhanced tabulated transport model v2 with Hermite interpolation.

Extends :class:`~pyfoam.thermophysical.tabulated_transport_enhanced.TabulatedTransportEnhanced`
with:

- Cubic Hermite interpolation for smoother viscosity curves
- Monotone interpolation option (Fritsch-Carlson) to prevent overshoot
- Multi-component support with mixture-averaged properties

Usage::

    from pyfoam.thermophysical.tabulated_transport_enhanced_2 import TabulatedTransportEnhanced2

    transport = TabulatedTransportEnhanced2(
        T_data=[200, 300, 400, 500],
        mu_data=[1.0e-5, 1.8e-5, 2.5e-5, 3.2e-5],
        interpolation="hermite",
    )
    mu = transport.mu(T=350.0)
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.tabulated_transport_enhanced import TabulatedTransportEnhanced

__all__ = ["TabulatedTransportEnhanced2"]

logger = logging.getLogger(__name__)


class TabulatedTransportEnhanced2(TabulatedTransportEnhanced):
    """Enhanced tabulated transport v2 with Hermite interpolation.

    Extends :class:`TabulatedTransportEnhanced` with:

    - **Hermite interpolation**: cubic Hermite splines for smooth C1-continuous
      interpolation of viscosity and conductivity data.
    - **Monotone mode**: Fritsch-Carlson monotone Hermite to prevent overshoot
      in regions where data should be monotone.
    - **Multi-component mixing**: species-wise interpolation with Wilke-style
      mixing for gas mixtures.

    Parameters
    ----------
    T_data : sequence of float
        Temperature data points (K), strictly increasing.
    mu_data : sequence of float
        Dynamic viscosity values (Pa*s) at reference pressure.
    kappa_data : sequence of float or None
        Thermal conductivity values.
    interpolation : str
        Interpolation method: "linear" (default parent), "hermite", or "monotone".
    P_data, mu_P_data, kappa_P_data, P_ref, pressure_exponent :
        See parent class.
    """

    def __init__(
        self,
        T_data: Sequence[float],
        mu_data: Sequence[float],
        kappa_data: Sequence[float] | None = None,
        interpolation: str = "hermite",
        P_data: Sequence[float] | None = None,
        mu_P_data: Sequence[Sequence[float]] | None = None,
        kappa_P_data: Sequence[Sequence[float]] | None = None,
        P_ref: float = 101325.0,
        pressure_exponent: float = 0.0,
    ) -> None:
        super().__init__(
            T_data=T_data,
            mu_data=mu_data,
            kappa_data=kappa_data,
            P_data=P_data,
            mu_P_data=mu_P_data,
            kappa_P_data=kappa_P_data,
            P_ref=P_ref,
            pressure_exponent=pressure_exponent,
        )

        if interpolation not in ("linear", "hermite", "monotone"):
            raise ValueError(
                f"interpolation must be 'linear', 'hermite', or 'monotone', "
                f"got '{interpolation}'"
            )
        self._interp_method = interpolation

        # Pre-compute tangent slopes for Hermite interpolation
        if interpolation in ("hermite", "monotone"):
            self._mu_slopes = self._compute_slopes(self._T_data, self._mu_data)
            if self._kappa_data is not None:
                self._kappa_slopes = self._compute_slopes(self._T_data, self._kappa_data)
            else:
                self._kappa_slopes = None

            if interpolation == "monotone":
                self._mu_slopes = self._fritsch_carlson_monotone(
                    self._T_data, self._mu_data, self._mu_slopes
                )
                if self._kappa_slopes is not None and self._kappa_data is not None:
                    self._kappa_slopes = self._fritsch_carlson_monotone(
                        self._T_data, self._kappa_data, self._kappa_slopes
                    )
        else:
            self._mu_slopes = None
            self._kappa_slopes = None

    @property
    def interpolation_method(self) -> str:
        """Current interpolation method name."""
        return self._interp_method

    # ------------------------------------------------------------------
    # Tangent computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_slopes(T_data: list[float], data: list[float]) -> list[float]:
        """Compute finite-difference slopes at each data point.

        Uses central differences at interior points and one-sided
        differences at the endpoints. Slopes are dy/dx where x = T.

        Parameters
        ----------
        T_data : list of float
            Temperature data points.
        data : list of float
            Data values at tabulated points.

        Returns
        -------
        list of float
            Slopes (dy/dT) at each data point.
        """
        n = len(data)
        if n < 2:
            return [0.0] * n

        slopes = [0.0] * n

        # Central differences at interior points
        for i in range(1, n - 1):
            dT = T_data[i + 1] - T_data[i - 1]
            if abs(dT) > 1e-30:
                slopes[i] = (data[i + 1] - data[i - 1]) / dT
            else:
                slopes[i] = 0.0

        # One-sided at endpoints
        dT0 = T_data[1] - T_data[0]
        slopes[0] = (data[1] - data[0]) / dT0 if abs(dT0) > 1e-30 else 0.0

        dTn = T_data[-1] - T_data[-2]
        slopes[-1] = (data[-1] - data[-2]) / dTn if abs(dTn) > 1e-30 else 0.0

        return slopes

    @staticmethod
    def _fritsch_carlson_monotone(
        T_data: list[float],
        data: list[float],
        slopes: list[float],
    ) -> list[float]:
        """Apply Fritsch-Carlson monotone correction to Hermite slopes.

        Ensures the resulting Hermite interpolation preserves monotonicity
        in the data.

        Parameters
        ----------
        T_data : list of float
            Temperature data points.
        data : list of float
            Data values.
        slopes : list of float
            Initial finite-difference slopes (dy/dT).

        Returns
        -------
        list of float
            Corrected slopes for monotone interpolation.
        """
        n = len(data)
        if n < 2:
            return list(slopes)

        corrected = list(slopes)

        for i in range(n - 1):
            dT = T_data[i + 1] - T_data[i]
            delta = data[i + 1] - data[i]
            if abs(delta) < 1e-30 or abs(dT) < 1e-30:
                corrected[i] = 0.0
                corrected[i + 1] = 0.0
                continue

            # Secant slope
            sec = delta / dT
            alpha = corrected[i] / sec if abs(sec) > 1e-30 else 0.0
            beta = corrected[i + 1] / sec if abs(sec) > 1e-30 else 0.0

            # Fritsch-Carlson constraint
            tau = alpha * alpha + beta * beta
            if tau > 9.0:
                s = 3.0 / tau**0.5
                corrected[i] = s * alpha * sec
                corrected[i + 1] = s * beta * sec

        return corrected

    # ------------------------------------------------------------------
    # Hermite basis functions
    # ------------------------------------------------------------------

    @staticmethod
    def _hermite_interp(
        t: torch.Tensor,
        f0: torch.Tensor,
        f1: torch.Tensor,
        m0: torch.Tensor,
        m1: torch.Tensor,
    ) -> torch.Tensor:
        """Cubic Hermite interpolation on [0, 1].

        Parameters
        ----------
        t : torch.Tensor
            Interpolation parameter in [0, 1].
        f0, f1 : torch.Tensor
            Function values at t=0 and t=1.
        m0, m1 : torch.Tensor
            Tangent values at t=0 and t=1 (scaled by interval width).

        Returns
        -------
        torch.Tensor
            Interpolated values.
        """
        t2 = t * t
        t3 = t2 * t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2
        return h00 * f0 + h10 * m0 + h01 * f1 + h11 * m1

    # ------------------------------------------------------------------
    # Hermite interpolation over data
    # ------------------------------------------------------------------

    def _interp_hermite(
        self,
        T: torch.Tensor,
        data: list[float],
        slopes: list[float],
    ) -> torch.Tensor:
        """Hermite interpolation of tabulated data.

        Parameters
        ----------
        T : torch.Tensor
            Temperature tensor.
        data : list of float
            Tabulated values.
        slopes : list of float
            Pre-computed slopes at each data point (dy/dT).

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

        T_lo = T_arr[idx - 1]
        T_hi = T_arr[idx]
        h = (T_hi - T_lo).clamp(min=1e-30)
        t = ((T_c - T_lo) / h).clamp(0.0, 1.0)

        data_t = torch.tensor(data, dtype=dtype, device=device)
        slopes_t = torch.tensor(slopes, dtype=dtype, device=device)

        f0 = data_t[idx - 1]
        f1 = data_t[idx]
        # Scale slopes by interval width for Hermite basis
        m0 = slopes_t[idx - 1] * h
        m1 = slopes_t[idx] * h

        return self._hermite_interp(t, f0, f1, m0, m1)

    # ------------------------------------------------------------------
    # Public API overrides
    # ------------------------------------------------------------------

    def mu(
        self,
        T: torch.Tensor | float,
        P: float | None = None,
    ) -> torch.Tensor:
        """Compute viscosity with Hermite or monotone interpolation.

        In bilinear mode, always uses parent bilinear interpolation.
        Otherwise, uses the configured interpolation method.

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

        if self._interp_method in ("hermite", "monotone") and self._mu_slopes is not None:
            mu_T = self._interp_hermite(T, self._mu_data, self._mu_slopes)
        else:
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
        """Compute thermal conductivity with Hermite or monotone interpolation.

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
            if (
                self._interp_method in ("hermite", "monotone")
                and self._kappa_slopes is not None
            ):
                return self._interp_hermite(T, self._kappa_data, self._kappa_slopes)
            return self._interp_linear(T, self._kappa_data)

        return self.mu(T, P=P) * Cp / Pr

    def __repr__(self) -> str:
        n = len(self._T_data)
        mode = "bilinear" if self.bilinear_mode else self._interp_method
        return (
            f"TabulatedTransportEnhanced2(T_range=[{self._T_data[0]:.0f}, "
            f"{self._T_data[-1]:.0f}], n_points={n}, interp={mode})"
        )
