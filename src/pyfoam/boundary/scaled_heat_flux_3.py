"""
Enhanced scaled heat flux boundary condition (v3).

Extends ``scaledHeatFlux2`` with a spatially-varying scaling map
and a radiative heat loss term::

    T_mean = (T_face + T_ref) / 2
    scale_eff = scale * (1 + alpha_T * (T_mean - T_scaleRef)) * spatialWeight
    q_conv = scale_eff * q_ref * timeModulation(t)
    q_rad = epsilon_sigma * sigma_sb * (T_mean^4 - T_amb^4)
    q_total = q_conv - q_rad
    dT/dn = -q_total / k_eff
    k_eff = k * (1 + beta_k * (T_mean - T_kRef))

In OpenFOAM syntax::

    type        scaledHeatFlux3;
    scale       2.0;
    q_ref       500.0;
    k           0.025;
    alphaT      0.0;
    TscaleRef   300.0;
    betaK       0.0;
    TkRef       300.0;
    timeModulation 1.0;
    spatialWeight 1.0;
    epsilonSigma 0.0;
    sigmaSB     5.670374419e-8;
    Tamb        300.0;
    value       uniform 300;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["ScaledHeatFlux3BC"]


@BoundaryCondition.register("scaledHeatFlux3")
class ScaledHeatFlux3BC(BoundaryCondition):
    """Enhanced scaled heat flux BC v3 with radiative loss and spatial weighting.

    Coefficients:
        - ``scale``: Flux scaling factor (default 1.0).
        - ``q_ref``: Reference heat flux (W/m^2, default 0.0).
        - ``k``: Thermal conductivity (W/(m K), default 0.025).
        - ``alphaT``: Temperature sensitivity coefficient (1/K, default 0.0).
        - ``TscaleRef``: Reference temperature for scale correction (K, default 300.0).
        - ``betaK``: Temperature coefficient for conductivity (1/K, default 0.0).
        - ``TkRef``: Reference temperature for conductivity correction (K, default 300.0).
        - ``timeModulation``: Time modulation factor (default 1.0).
        - ``spatialWeight``: Spatial weighting factor (default 1.0).
        - ``epsilonSigma``: Surface emissivity for radiative loss (default 0.0 = no radiation).
        - ``sigmaSB``: Stefan-Boltzmann constant (W/(m2 K4), default 5.670374419e-8).
        - ``Tamb``: Ambient temperature for radiation (K, default 300.0).
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
        self._spatial_weight = float(self._coeffs.get("spatialWeight", 1.0))
        self._epsilon_sigma = float(self._coeffs.get("epsilonSigma", 0.0))
        self._sigma_SB = float(self._coeffs.get("sigmaSB", 5.670374419e-8))
        self._T_amb = float(self._coeffs.get("Tamb", 300.0))

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

    @property
    def k(self) -> float:
        """Thermal conductivity (W/(m K))."""
        return self._k

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
    def spatial_weight(self) -> float:
        """Spatial weighting factor."""
        return self._spatial_weight

    @property
    def epsilon_sigma(self) -> float:
        """Surface emissivity for radiative loss."""
        return self._epsilon_sigma

    @property
    def sigma_SB(self) -> float:
        """Stefan-Boltzmann constant (W/(m2 K4))."""
        return self._sigma_SB

    @property
    def T_amb(self) -> float:
        """Ambient temperature for radiation (K)."""
        return self._T_amb

    @property
    def q(self) -> float:
        """Effective heat flux: scale * q_ref * timeModulation * spatialWeight (W/m^2)."""
        return self._scale * self._q_ref * self._time_modulation * self._spatial_weight

    def _effective_conductivity(self, T_mean: float) -> float:
        """Temperature-dependent thermal conductivity."""
        return self._k * (1.0 + self._beta_k * (T_mean - self._T_k_ref))

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
        """Apply enhanced scaled heat flux BC with radiative loss.

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

            # Temperature-dependent conductivity and scale
            k_eff = self._k * (1.0 + self._beta_k * (T_mean - self._T_k_ref))
            scale_eff = self._scale * (1.0 + self._alpha_T * (T_mean - self._T_scale_ref)) * self._spatial_weight
            q_conv = scale_eff * self._q_ref * self._time_modulation

            # Radiative heat loss
            if self._epsilon_sigma > 0:
                T_mean_K = torch.clamp(T_mean, min=1.0)
                T_amb_K = max(self._T_amb, 1.0)
                q_rad = self._epsilon_sigma * self._sigma_SB * (T_mean_K ** 4 - T_amb_K ** 4)
                q_total = q_conv - q_rad
            else:
                q_total = q_conv

            grad_T = -q_total / (k_eff + 1e-30)
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
        """Matrix contributions for enhanced scaled heat flux BC v3."""
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
