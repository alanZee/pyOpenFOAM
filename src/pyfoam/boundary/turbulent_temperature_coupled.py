"""
Turbulent temperature coupled boundary condition.

A coupled-temperature BC that accounts for enhanced thermal transport
due to turbulence.  The effective thermal diffusivity is::

    alpha_eff = alpha_lam + alpha_turb
              = k / (rho * cp) + nu_t / Pr_t

where:
    - k is the laminar thermal conductivity
    - nu_t is the turbulent viscosity (read from the ``nut`` field)
    - Pr_t is the turbulent Prandtl number (typically 0.85)

The boundary temperature is then computed by blending fixed-value and
gradient contributions using the effective diffusivity::

    T_face = (alpha_eff * T_interior / delta + h * T_coupled) /
             (alpha_eff / delta + h)

This is a Robin / mixed BC with the effective diffusivity modified by
turbulence.

In OpenFOAM syntax::

    type                    turbulentTemperatureCoupled;
    T_coupled               350;         // coupled / far-side temperature
    Pr_t                    0.85;        // turbulent Prandtl number
    alpha_lam               2.5e-5;     // laminar thermal diffusivity (m²/s)
    nut_field               nut;         // turbulent viscosity field name
    value                   uniform 300;

Usage::

    bc = BoundaryCondition.create("turbulentTemperatureCoupled", patch, coeffs={
        "T_coupled": 350.0,
        "Pr_t": 0.85,
    })
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentTemperatureCoupledBC"]


@BoundaryCondition.register("turbulentTemperatureCoupled")
class TurbulentTemperatureCoupledBC(BoundaryCondition):
    """Turbulent coupled temperature boundary condition.

    Extends a mixed (Robin) temperature BC by adding a turbulent
    contribution to the thermal diffusivity::

        alpha_eff = alpha_lam + nu_t / Pr_t

    Coefficients:
        - ``T_coupled``: Coupled / far-side temperature (K).  Default 300.
        - ``Pr_t``: Turbulent Prandtl number.  Default 0.85.
        - ``alpha_lam``: Laminar thermal diffusivity (m²/s).  Default 2.5e-5.
        - ``nut_field``: Turbulent viscosity field name (informational).
        - ``nut_values``: Optional tensor ``(n_faces,)`` of nut values
          injected at runtime.
        - ``value``: Initial temperature (default 300).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._t_coupled = float(self._coeffs.get("T_coupled", 300.0))
        self._pr_t = float(self._coeffs.get("Pr_t", 0.85))
        self._alpha_lam = float(self._coeffs.get("alpha_lam", 2.5e-5))
        self._nut_field_name = self._coeffs.get("nut_field", "nut")
        self._nut_values: torch.Tensor | None = self._coeffs.get("nut_values")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def t_coupled(self) -> float:
        """Coupled / far-side temperature (K)."""
        return self._t_coupled

    @property
    def pr_t(self) -> float:
        """Turbulent Prandtl number."""
        return self._pr_t

    @property
    def alpha_lam(self) -> float:
        """Laminar thermal diffusivity (m²/s)."""
        return self._alpha_lam

    @property
    def nut_values(self) -> torch.Tensor | None:
        """Per-face turbulent viscosity values, or ``None``."""
        return self._nut_values

    @nut_values.setter
    def nut_values(self, value: torch.Tensor | None) -> None:
        """Set per-face turbulent viscosity values."""
        self._nut_values = value

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _effective_alpha(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Compute effective thermal diffusivity per face.

        alpha_eff = alpha_lam + nut / Pr_t

        Returns:
            ``(n_faces,)`` effective diffusivity tensor.
        """
        n = self._patch.n_faces
        if self._nut_values is not None:
            nut = self._nut_values[:n].to(device=device, dtype=dtype)
        else:
            nut = torch.zeros(n, dtype=dtype, device=device)

        alpha_eff = self._alpha_lam + nut / (self._pr_t + 1e-30)
        return alpha_eff

    # ------------------------------------------------------------------
    # BC interface
    # ------------------------------------------------------------------

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Apply turbulent coupled temperature BC.

        Computes a Robin blend using the effective (laminar + turbulent)
        thermal diffusivity.

        Args:
            field: Temperature field.
            patch_idx: Optional start index into *field*.
        """
        device = field.device
        dtype = field.dtype

        owners = self._patch.owner_cells.to(device=device)
        owner_vals = field[owners]

        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)
        alpha_eff = self._effective_alpha(device, dtype)

        # Robin: T = (alpha_eff/delta * T_int + h * T_coupled) / (alpha_eff/delta + h)
        # Here we use alpha_eff as the effective diffusion "weight"
        grad_weight = alpha_eff * deltas
        value_weight = alpha_eff  # convective coupling proxy

        denom = grad_weight + value_weight + 1e-30
        t_face = (grad_weight * owner_vals + value_weight * self._t_coupled) / denom

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = t_face
        else:
            field[self._patch.face_indices] = t_face
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Robin-type matrix contribution with turbulent diffusivity.

        Adds implicit coupling using the effective diffusivity::

            diag   += alpha_eff * A * delta / (alpha_eff * delta + alpha_eff)
            source += alpha_eff * A * T_coupled / (alpha_eff * delta + alpha_eff)
        """
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)
        alpha_eff = self._effective_alpha(device, dtype)

        grad_weight = alpha_eff * deltas
        value_weight = alpha_eff
        denom = grad_weight + value_weight + 1e-30

        coeff = value_weight * areas / denom

        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * self._t_coupled)

        return diag, source


# Import at module level to trigger registration
from . import boundary_condition  # noqa: E402, F401
