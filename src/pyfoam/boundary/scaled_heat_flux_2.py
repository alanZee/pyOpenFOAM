"""
Enhanced scaled heat flux boundary condition (v2).

Extends ``scaledHeatFlux`` with temperature-dependent scaling and a
time-varying modulation function::

    T_mean = (T_face + T_ref) / 2
    scale_eff = scale * (1 + alpha_T * (T_mean - T_scaleRef))
    q = scale_eff * q_ref * timeModulation(t)
    dT/dn = -q / k_eff
    k_eff = k * (1 + beta_k * (T_mean - T_kRef))

In OpenFOAM syntax::

    type        scaledHeatFlux2;
    scale       2.0;
    q_ref       500.0;
    k           0.025;
    alphaT      0.0;
    TscaleRef   300.0;
    betaK       0.0;
    TkRef       300.0;
    timeModulation 1.0;
    value       uniform 300;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["ScaledHeatFlux2BC"]


@BoundaryCondition.register("scaledHeatFlux2")
class ScaledHeatFlux2BC(BoundaryCondition):
    """Enhanced scaled heat flux BC v2 with temperature-dependent scaling.

    Coefficients:
        - ``scale``: Flux scaling factor (default 1.0).
        - ``q_ref``: Reference heat flux (W/m^2, default 0.0).
        - ``k``: Thermal conductivity (W/(m K), default 0.025).
        - ``alphaT``: Temperature sensitivity coefficient (1/K, default 0.0).
        - ``TscaleRef``: Reference temperature for scale correction (K, default 300.0).
        - ``betaK``: Temperature coefficient for conductivity (1/K, default 0.0).
        - ``TkRef``: Reference temperature for conductivity correction (K, default 300.0).
        - ``timeModulation``: Time modulation factor (default 1.0).
        - ``value``: Reference temperature (K, default 300.0).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._scale = float(self._coeffs.get("scale", 1.0))
        self._q_ref = float(self._coeffs.get("q_ref", 0.0))
        self._k = float(self._coeffs.get("k", 0.025))
        self._T_ref = float(self._coeffs.get("value", 300.0))
        self._alpha_T = float(self._coeffs.get("alphaT", 0.0))
        self._T_scale_ref = float(self._coeffs.get("TscaleRef", 300.0))
        self._beta_k = float(self._coeffs.get("betaK", 0.0))
        self._T_k_ref = float(self._coeffs.get("TkRef", 300.0))
        self._time_modulation = float(self._coeffs.get("timeModulation", 1.0))

    @property
    def scale(self) -> float:
        """Flux scaling factor."""
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value

    @property
    def q_ref(self) -> float:
        """Reference heat flux (W/m^2)."""
        return self._q_ref

    @q_ref.setter
    def q_ref(self, value: float) -> None:
        self._q_ref = value

    @property
    def k(self) -> float:
        """Thermal conductivity (W/(m K))."""
        return self._k

    @k.setter
    def k(self, value: float) -> None:
        self._k = value

    @property
    def T_ref(self) -> float:
        """Reference temperature (K)."""
        return self._T_ref

    @property
    def alpha_T(self) -> float:
        """Temperature sensitivity coefficient (1/K)."""
        return self._alpha_T

    @property
    def T_scale_ref(self) -> float:
        """Reference temperature for scale correction (K)."""
        return self._T_scale_ref

    @property
    def beta_k(self) -> float:
        """Temperature coefficient for conductivity (1/K)."""
        return self._beta_k

    @property
    def T_k_ref(self) -> float:
        """Reference temperature for conductivity correction (K)."""
        return self._T_k_ref

    @property
    def time_modulation(self) -> float:
        """Time modulation factor."""
        return self._time_modulation

    @time_modulation.setter
    def time_modulation(self, value: float) -> None:
        self._time_modulation = value

    @property
    def q(self) -> float:
        """Effective heat flux: scale * q_ref * timeModulation (W/m^2)."""
        return self._scale * self._q_ref * self._time_modulation

    def _effective_conductivity(self, T_mean: float) -> float:
        """Temperature-dependent thermal conductivity."""
        return self._k * (1.0 + self._beta_k * (T_mean - self._T_k_ref))

    def _effective_scale(self, T_mean: float) -> float:
        """Temperature-dependent scaling factor."""
        return self._scale * (1.0 + self._alpha_T * (T_mean - self._T_scale_ref))

    @property
    def gradient(self) -> float:
        """Wall-normal temperature gradient (K/m).

        dT/dn = -q / k
        """
        k_eff = self._effective_conductivity(self._T_ref)
        if abs(k_eff) < 1e-30:
            return 0.0
        return -self.q / k_eff

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        T_field: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply enhanced scaled heat flux BC to temperature field.

        Args:
            field: Temperature field.
            patch_idx: Optional start index.
            T_field: ``(n_faces,)`` current face temperatures for feedback.
        """
        device = field.device
        dtype = field.dtype
        n_faces = self._patch.n_faces

        delta = self._patch.delta_coeffs.to(device=device, dtype=dtype)
        delta_safe = delta.abs().clamp(min=1e-10)

        if T_field is not None:
            T_face_current = T_field.to(device=device, dtype=dtype)
            T_mean = (T_face_current + self._T_ref) / 2.0
            # Per-face temperature-dependent conductivity and scale
            k_eff = self._k * (1.0 + self._beta_k * (T_mean - self._T_k_ref))
            scale_eff = self._scale * (1.0 + self._alpha_T * (T_mean - self._T_scale_ref))
            q_local = scale_eff * self._q_ref * self._time_modulation
            grad_T = -q_local / (k_eff + 1e-30)
            T_face = self._T_ref + grad_T / delta_safe
        else:
            T_face = torch.full(
                (n_faces,), self._T_ref, dtype=dtype, device=device,
            )
            T_face = T_face + self.gradient / delta_safe

        if patch_idx is not None:
            field[patch_idx : patch_idx + n_faces] = T_face
        else:
            field[self._patch.face_indices] = T_face
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Matrix contributions for enhanced scaled heat flux BC v2.

        Uses fixedGradient treatment with temperature-dependent conductivity.
        """
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)

        if areas.dim() > 1:
            area_mag = areas.norm(dim=1)
        else:
            area_mag = areas.abs()

        # Heat flux contribution to source: q * A
        flux_contrib = self.q * area_mag
        source.scatter_add_(0, owners, flux_contrib)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
