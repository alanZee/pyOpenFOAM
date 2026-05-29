"""
Compressible turbulent temperature coupled boundary condition.

Enhanced variant of ``turbulentTemperatureCoupled`` for compressible
conjugate heat transfer.  Uses the **compressible** effective thermal
diffusivity::

    alpha_eff = k / (rho * cp) + nu_t / Pr_t

where the laminar part is density-weighted (``k / (rho * cp)`` instead
of a fixed scalar).  The boundary temperature is computed by a Robin
(mixed) blend::

    T_face = (alpha_eff * delta * T_owner + h * T_coupled) /
             (alpha_eff * delta + h)

In OpenFOAM syntax::

    type                    compressibleTurbulentTemperatureCoupled;
    T_coupled               350;
    Pr_t                    0.85;
    k                       0.025;       // laminar thermal conductivity (W/(m K))
    rho                     1.225;       // density (kg/m^3)
    cp                      1005;        // specific heat capacity (J/(kg K))
    nut_field               nut;
    value                   uniform 300;

Usage::

    bc = BoundaryCondition.create(
        "compressibleTurbulentTemperatureCoupled", patch,
        coeffs={"T_coupled": 350.0, "Pr_t": 0.85, "k": 0.025, "rho": 1.225, "cp": 1005.0},
    )
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["CompressibleTurbulentTemperatureCoupledBC"]


@BoundaryCondition.register("compressibleTurbulentTemperatureCoupled")
class CompressibleTurbulentTemperatureCoupledBC(BoundaryCondition):
    """Compressible coupled temperature boundary condition with turbulent diffusivity.

    The effective thermal diffusivity is computed from both laminar
    (density-weighted) and turbulent contributions::

        alpha_eff = k / (rho * cp) + nu_t / Pr_t

    The Robin BC blends interior and coupled temperatures:

        T_face = (alpha_eff * delta * T_owner + h * T_coupled) /
                 (alpha_eff * delta + h)

    where ``h`` is a proxy convective coefficient derived from
    ``alpha_eff * delta``.

    Coefficients:
        - ``T_coupled`` (float): Coupled / far-side temperature (K).  Default 300.
        - ``Pr_t`` (float): Turbulent Prandtl number.  Default 0.85.
        - ``k`` (float): Laminar thermal conductivity (W/(m K)).  Default 0.025.
        - ``rho`` (float): Density (kg/m^3).  Default 1.225.
        - ``cp`` (float): Specific heat capacity (J/(kg K)).  Default 1005.
        - ``nut_field`` (str): Turbulent viscosity field name (informational).
        - ``nut_values`` (Tensor | None): Per-face nut values injected at runtime.
        - ``value`` (float): Initial temperature (default 300).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._t_coupled = float(self._coeffs.get("T_coupled", 300.0))
        self._pr_t = float(self._coeffs.get("Pr_t", 0.85))
        self._k = float(self._coeffs.get("k", 0.025))
        self._rho = float(self._coeffs.get("rho", 1.225))
        self._cp = float(self._coeffs.get("cp", 1005.0))
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
    def k(self) -> float:
        """Laminar thermal conductivity (W/(m K))."""
        return self._k

    @property
    def rho(self) -> float:
        """Density (kg/m^3)."""
        return self._rho

    @property
    def cp(self) -> float:
        """Specific heat capacity (J/(kg K))."""
        return self._cp

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
        """Compute compressible effective thermal diffusivity per face.

        alpha_eff = k / (rho * cp) + nut / Pr_t
        """
        n = self._patch.n_faces
        alpha_lam = self._k / (self._rho * self._cp + 1e-30)

        if self._nut_values is not None:
            nut = self._nut_values[:n].to(device=device, dtype=dtype)
        else:
            nut = torch.zeros(n, dtype=dtype, device=device)

        return alpha_lam + nut / (self._pr_t + 1e-30)

    # ------------------------------------------------------------------
    # BC interface
    # ------------------------------------------------------------------

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Apply compressible turbulent coupled temperature BC.

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

        # Robin blend: T = (alpha_eff*delta*T_int + alpha_eff*T_coupled) /
        #                   (alpha_eff*delta + alpha_eff)
        grad_weight = alpha_eff * deltas
        value_weight = alpha_eff

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
        """Robin-type matrix contribution with compressible turbulent diffusivity.

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


# 触发 RTS 注册
from . import boundary_condition  # noqa: E402, F401
