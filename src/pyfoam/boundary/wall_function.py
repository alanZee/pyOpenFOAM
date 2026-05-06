"""
Wall-function boundary conditions.

Implements turbulence wall-function BCs following the OpenFOAM approach.
In OpenFOAM syntax::

    type   nutkWallFunction;
    value  uniform 0;

Wall functions bridge the viscous sublayer and log-law region,
providing effective turbulent viscosity ``ν_t`` at wall faces.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["NutkWallFunctionBC", "KqRWallFunctionBC"]

# Von Karman constant
_KAPPA: float = 0.41
# Empirical constant for log-law
_E: float = 9.8


@BoundaryCondition.register("nutkWallFunction")
class NutkWallFunctionBC(BoundaryCondition):
    """k-equation-based wall function for turbulent viscosity (ν_t).

    Computes ν_t at the wall from the log-law:

        u⁺ = (1/κ) ln(E y⁺)

    where y⁺ = u_τ y / ν and u_τ = √(C_μ^{1/2} k).

    Coefficients:
        - ``value``: initial/existing ν_t value (default 0)
        - ``Cmu``: k-ε model constant (default 0.09)
        - ``kappa``: von Karman constant (default 0.41)
        - ``E``: wall roughness parameter (default 9.8)
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._cmu: float = float(self._coeffs.get("Cmu", 0.09))
        self._kappa: float = float(self._coeffs.get("kappa", _KAPPA))
        self._E: float = float(self._coeffs.get("E", _E))

    def compute_nut(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
        nu: float,
    ) -> torch.Tensor:
        """Compute turbulent viscosity at wall faces.

        Args:
            k: Turbulent kinetic energy at wall-adjacent cells
                shape ``(n_faces,)``.
            y: Wall-normal distance from cell centre to face
                shape ``(n_faces,)``.
            nu: Molecular kinematic viscosity.

        Returns:
            ν_t at each wall face, shape ``(n_faces,)``.
        """
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        # Friction velocity: u_τ = C_μ^{1/4} * sqrt(k)
        u_tau = self._cmu**0.25 * torch.sqrt(k.clamp(min=1e-16))

        # y⁺ = u_τ * y / ν
        y_plus = u_tau * y / max(nu, 1e-30)
        y_plus = y_plus.clamp(min=1e-4)

        # Effective ν_t from log-law
        # ν_t = κ u_τ y / ln(E y⁺)
        nut = self._kappa * u_tau * y / torch.log(self._E * y_plus)

        # Ensure non-negative
        nut = nut.clamp(min=0.0)
        return nut

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """If a ``nut`` coefficient is provided, set face values.

        Otherwise, the field is left unchanged (nut must be computed
        externally via :meth:`compute_nut`).
        """
        if "value" in self._coeffs:
            val = self._coeffs["value"]
            if isinstance(val, torch.Tensor):
                val_tensor = val.to(device=field.device, dtype=field.dtype)
            else:
                val_tensor = torch.full(
                    (self._patch.n_faces,),
                    float(val),
                    device=field.device,
                    dtype=field.dtype,
                )
            if patch_idx is not None:
                n = self._patch.n_faces
                field[patch_idx : patch_idx + n] = val_tensor
            else:
                field[self._patch.face_indices] = val_tensor
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wall functions: zero matrix contribution (explicit treatment).

        Wall-function BCs modify the effective viscosity field rather
        than contributing to the matrix directly.
        """
        device = get_device()
        dtype = get_default_dtype()
        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)
        return diag, source


@BoundaryCondition.register("kqRWallFunction")
class KqRWallFunctionBC(BoundaryCondition):
    """Wall function for k, q (TKE), and R (Reynolds stress).

    Prescribes turbulence quantities at wall faces based on
    the local equilibrium assumption.

    Coefficients:
        - ``value``: initial/existing value (default 0)
        - ``Cmu``: model constant (default 0.09)
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._cmu: float = float(self._coeffs.get("Cmu", 0.09))

    def compute_k_wall(
        self,
        u_tau: torch.Tensor,
    ) -> torch.Tensor:
        """Compute k at wall faces from friction velocity.

        Under local equilibrium:
            k = u_τ² / √C_μ

        Args:
            u_tau: Friction velocity at wall faces.

        Returns:
            k at wall faces.
        """
        device = get_device()
        dtype = get_default_dtype()
        u_tau = u_tau.to(device=device, dtype=dtype)
        return u_tau**2 / math.sqrt(self._cmu)

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set wall-face values from coefficients if available."""
        if "value" in self._coeffs:
            val = self._coeffs["value"]
            if isinstance(val, torch.Tensor):
                val_tensor = val.to(device=field.device, dtype=field.dtype)
            else:
                val_tensor = torch.full(
                    (self._patch.n_faces,),
                    float(val),
                    device=field.device,
                    dtype=field.dtype,
                )
            if patch_idx is not None:
                n = self._patch.n_faces
                field[patch_idx : patch_idx + n] = val_tensor
            else:
                field[self._patch.face_indices] = val_tensor
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wall functions: zero matrix contribution (explicit treatment)."""
        device = get_device()
        dtype = get_default_dtype()
        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)
        return diag, source
