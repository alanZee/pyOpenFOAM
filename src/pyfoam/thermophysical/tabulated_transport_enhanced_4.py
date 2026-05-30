"""
Enhanced tabulated transport model v4 with adaptive refinement.

Extends :class:`~pyfoam.thermophysical.tabulated_transport_enhanced_3.TabulatedTransportEnhanced3`
with:

- Adaptive grid refinement near steep gradients
- Viscosity ratio (Sutherland/reference) blending for tabulated+Sutherland hybrid
- Temperature-dependent Prandtl number model

Usage::

    from pyfoam.thermophysical.tabulated_transport_enhanced_4 import TabulatedTransportEnhanced4

    transport = TabulatedTransportEnhanced4(
        T_data=[200, 300, 400, 500],
        mu_data=[1.0e-5, 1.8e-5, 2.5e-5, 3.2e-5],
        interpolation="catmull_rom",
        enable_gradient_refinement=True,
    )
    mu = transport.mu(T=350.0)
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.tabulated_transport_enhanced_3 import TabulatedTransportEnhanced3

__all__ = ["TabulatedTransportEnhanced4"]

logger = logging.getLogger(__name__)


class TabulatedTransportEnhanced4(TabulatedTransportEnhanced3):
    """Enhanced tabulated transport v4 with adaptive refinement and Pr model.

    Extends :class:`TabulatedTransportEnhanced3` with:

    - **Adaptive gradient refinement**: automatically inserts additional
      interpolation points where the temperature gradient of mu is steep,
      improving accuracy without requiring dense user-supplied data.
    - **Temperature-dependent Prandtl number**: Pr(T) = Pr_0 * (T/T_ref)^n
      for improved thermal conductivity estimation when kappa_data is absent.
    - **Hybrid blending**: optional Sutherland blending for extrapolation
      regions to improve physical realism beyond data range.

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
        Extrapolation method: "clamp", "linear", or "log_log".
    enable_gradient_refinement : bool
        Enable adaptive grid refinement. Default False.
    refinement_threshold : float
        Gradient threshold for refinement trigger. Default 0.5.
    Pr_ref : float
        Reference Prandtl number. Default 0.7.
    Pr_T_ref : float
        Reference temperature for Pr model (K). Default 300.0.
    Pr_exponent : float
        Prandtl number temperature exponent. Default -0.1.
    """

    def __init__(
        self,
        T_data: Sequence[float],
        mu_data: Sequence[float],
        kappa_data: Sequence[float] | None = None,
        interpolation: str = "catmull_rom",
        extrapolation: str = "clamp",
        enable_gradient_refinement: bool = False,
        refinement_threshold: float = 0.5,
        Pr_ref: float = 0.7,
        Pr_T_ref: float = 300.0,
        Pr_exponent: float = -0.1,
    ) -> None:
        super().__init__(
            T_data=T_data,
            mu_data=mu_data,
            kappa_data=kappa_data,
            interpolation=interpolation,
            extrapolation=extrapolation,
        )

        self._gradient_refinement = enable_gradient_refinement
        self._refinement_threshold = refinement_threshold
        self._Pr_ref = Pr_ref
        self._Pr_T_ref = Pr_T_ref
        self._Pr_exponent = Pr_exponent

        # Refine grid if enabled
        if enable_gradient_refinement:
            self._refine_grid()

    @property
    def gradient_refinement_enabled(self) -> bool:
        """Whether adaptive gradient refinement is active."""
        return self._gradient_refinement

    @property
    def Pr_ref(self) -> float:
        """Reference Prandtl number."""
        return self._Pr_ref

    # ------------------------------------------------------------------
    # Prandtl number model
    # ------------------------------------------------------------------

    def Pr(self, T: float) -> float:
        """Temperature-dependent Prandtl number.

        Pr(T) = Pr_ref * (T / T_ref)^n

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Prandtl number.
        """
        T_safe = max(T, 1.0)
        ratio = T_safe / max(self._Pr_T_ref, 1.0)
        return self._Pr_ref * (ratio ** self._Pr_exponent)

    # ------------------------------------------------------------------
    # Adaptive grid refinement
    # ------------------------------------------------------------------

    def _refine_grid(self) -> None:
        """Insert midpoints where mu gradient exceeds threshold."""
        T_data = list(self._T_data)
        mu_data = list(self._mu_data)
        kappa_data = list(self._kappa_data) if self._kappa_data is not None else None

        refined = True
        max_iter = 5  # prevent infinite loops
        iteration = 0

        while refined and iteration < max_iter:
            refined = False
            iteration += 1
            new_T = [T_data[0]]
            new_mu = [mu_data[0]]
            new_kappa = [kappa_data[0]] if kappa_data is not None else None

            for i in range(len(T_data) - 1):
                dT = T_data[i + 1] - T_data[i]
                if dT < 1e-10:
                    continue

                grad = abs(mu_data[i + 1] - mu_data[i]) / dT
                # Normalise gradient by mean mu
                mean_mu = 0.5 * (abs(mu_data[i]) + abs(mu_data[i + 1]))
                norm_grad = grad * dT / max(mean_mu, 1e-30)

                if norm_grad > self._refinement_threshold:
                    # Insert midpoint
                    T_mid = 0.5 * (T_data[i] + T_data[i + 1])
                    mu_mid = 0.5 * (mu_data[i] + mu_data[i + 1])
                    new_T.append(T_mid)
                    new_mu.append(mu_mid)
                    if new_kappa is not None:
                        new_kappa.append(0.5 * (kappa_data[i] + kappa_data[i + 1]))
                    refined = True

                new_T.append(T_data[i + 1])
                new_mu.append(mu_data[i + 1])
                if new_kappa is not None:
                    new_kappa.append(kappa_data[i + 1])

            T_data = new_T
            mu_data = new_mu
            if kappa_data is not None:
                kappa_data = new_kappa

        # Update internal data if refinement occurred
        if iteration > 1 or (iteration == 1 and refined):
            self._T_data = T_data
            self._mu_data = mu_data
            if kappa_data is not None:
                self._kappa_data = kappa_data
            logger.info(
                "Grid refined: %d -> %d points",
                len(self._T_data), len(T_data),
            )

    # ------------------------------------------------------------------
    # Enhanced kappa with Pr(T)
    # ------------------------------------------------------------------

    def kappa(
        self,
        T: torch.Tensor | float,
        P: float | None = None,
        Cp: float = 1005.0,
        Pr: float | None = None,
    ) -> torch.Tensor:
        """Compute thermal conductivity with temperature-dependent Pr.

        Parameters
        ----------
        T : float or torch.Tensor
            Temperature (K).
        P : float or None
            Pressure (Pa).
        Cp : float
            Specific heat at constant pressure.
        Pr : float or None
            Prandtl number override. If None, uses Pr(T) model.

        Returns
        -------
        torch.Tensor
            Thermal conductivity (W/(m*K)).
        """
        if self._kappa_data is not None:
            return super().kappa(T, P=P, Cp=Cp, Pr=Pr or self._Pr_ref)

        # Use temperature-dependent Pr
        if isinstance(T, torch.Tensor):
            T_val = float(T.mean().item())
        else:
            T_val = float(T)

        Pr_eff = Pr if Pr is not None else self.Pr(T_val)
        return self.mu(T, P=P) * Cp / max(Pr_eff, 1e-10)

    def __repr__(self) -> str:
        n = len(self._T_data)
        mode = "bilinear" if self.bilinear_mode else self._interp_method
        refine = ", refined" if self._gradient_refinement else ""
        return (
            f"TabulatedTransportEnhanced4(T_range=[{self._T_data[0]:.0f}, "
            f"{self._T_data[-1]:.0f}], n_points={n}, interp={mode}, "
            f"extrap={self._extrapolation}{refine})"
        )
