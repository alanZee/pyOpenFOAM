"""
Enhanced wall-function boundary condition with automatic y+ switching.

Implements a wall function that automatically detects the local y+ value
and switches between:

- **Viscous sublayer** (y+ < 11.225): linear (laminar) profile
    u+ = y+
- **Log-law region** (y+ >= 11.225): standard wall function
    u+ = (1/kappa) * ln(E * y+)

This "enhanced" approach removes the need for the user to choose between
``nutkWallFunction`` and ``nutLowReWallFunction`` — the same BC handles
both regimes seamlessly, following the OpenFOAM ``nutUSpaldingWallFunction``
approach.

In OpenFOAM syntax::

    type   enhancedWallFunction;
    kappa  0.41;
    E      9.8;
    Cmu    0.09;

Usage::

    from pyfoam.boundary.wall_function_2 import EnhancedWallFunctionBC

    bc = EnhancedWallFunctionBC(patch, coeffs={"kappa": 0.41, "E": 9.8})
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.boundary.boundary_condition import BoundaryCondition, Patch

__all__ = ["EnhancedWallFunctionBC"]

logger = logging.getLogger(__name__)

# Von Karman constant
_KAPPA: float = 0.41
# Empirical wall roughness constant
_E: float = 9.8
# Transition y+ (intersection of linear and log-law profiles)
_YPLUS_TRANSITION: float = 11.225


@BoundaryCondition.register("enhancedWallFunction")
class EnhancedWallFunctionBC(BoundaryCondition):
    """Enhanced wall function with automatic y+ detection and switching.

    Combines viscous sublayer (linear) and log-law treatments in a
    single boundary condition, eliminating the need to choose the
    appropriate wall function a priori.

    The effective turbulent viscosity is computed as:

        - y+ < 11.225:  nu_t = 0  (viscous sublayer resolved)
        - y+ >= 11.225: nu_t = kappa * u_tau * y / ln(E * y+)

    The wall shear stress is computed via Spalding's law which unifies
    both regions in a single implicit relation.

    Parameters
    ----------
    patch : Patch
        The wall boundary patch.
    coeffs : dict, optional
        Dictionary of coefficients:
        - ``kappa``: von Karman constant (default 0.41).
        - ``E``: wall roughness parameter (default 9.8).
        - ``Cmu``: k-epsilon model constant (default 0.09).
        - ``value``: initial nut value (default 0).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._kappa: float = float(self._coeffs.get("kappa", _KAPPA))
        self._E: float = float(self._coeffs.get("E", _E))
        self._cmu: float = float(self._coeffs.get("Cmu", 0.09))

    @property
    def kappa(self) -> float:
        """Von Karman constant."""
        return self._kappa

    @property
    def E(self) -> float:
        """Wall roughness parameter."""
        return self._E

    @property
    def yplus_transition(self) -> float:
        """Transition y+ between sublayer and log-law."""
        return _YPLUS_TRANSITION

    def compute_nut(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
        nu: float,
    ) -> torch.Tensor:
        """Compute turbulent viscosity at wall faces with automatic y+ switching.

        For each face, the local y+ is computed from the friction velocity
        u_tau = C_mu^{1/4} * sqrt(k), and the appropriate formula is applied:

        - y+ < 11.225 (viscous sublayer):  nut = 0
        - y+ >= 11.225 (log-law):           nut = kappa * u_tau * y / ln(E * y+)

        Parameters
        ----------
        k : torch.Tensor
            ``(n_faces,)`` turbulent kinetic energy at wall-adjacent cells.
        y : torch.Tensor
            ``(n_faces,)`` wall-normal distance.
        nu : float
            Molecular kinematic viscosity.

        Returns
        -------
        torch.Tensor
            ``(n_faces,)`` turbulent viscosity at each wall face.
        """
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        # Friction velocity
        u_tau = self._cmu**0.25 * torch.sqrt(k.clamp(min=1e-16))

        # y+
        y_plus = u_tau * y / max(nu, 1e-30)
        y_plus = y_plus.clamp(min=1e-4)

        # Log-law turbulent viscosity
        nut_loglaw = self._kappa * u_tau * y / torch.log(self._E * y_plus)

        # Switch: sublayer (y+ < 11.225) gets nut=0, log-law region gets nut_loglaw
        nut = torch.where(y_plus < _YPLUS_TRANSITION, torch.zeros_like(nut_loglaw), nut_loglaw)

        return nut.clamp(min=0.0)

    def compute_u_plus(self, y_plus: torch.Tensor) -> torch.Tensor:
        """Compute u+ from y+ using the appropriate wall law.

        - y+ < 11.225:  u+ = y+
        - y+ >= 11.225:  u+ = (1/kappa) * ln(E * y+)

        Parameters
        ----------
        y_plus : torch.Tensor
            Dimensionless wall distance.

        Returns
        -------
        torch.Tensor
            Dimensionless velocity u+.
        """
        device = y_plus.device
        dtype = y_plus.dtype

        u_plus_linear = y_plus.clone()
        u_plus_loglaw = torch.log(self._E * y_plus.clamp(min=1e-4)) / self._kappa

        u_plus = torch.where(
            y_plus < _YPLUS_TRANSITION,
            u_plus_linear,
            u_plus_loglaw,
        )

        return u_plus.clamp(min=0.0)

    def spalding_u_plus(self, y_plus: torch.Tensor) -> torch.Tensor:
        """Compute u+ via Spalding's law (unified, continuous).

        Spalding's law:

            y+ = u+ + exp(-kappa * E) * [exp(kappa * u+) - 1
                  - kappa * u+ - (kappa * u+)^2/2 - (kappa * u+)^3/6]

        This is solved iteratively using Newton's method.

        Parameters
        ----------
        y_plus : torch.Tensor
            Dimensionless wall distance.

        Returns
        -------
        torch.Tensor
            Dimensionless velocity u+.
        """
        device = y_plus.device
        dtype = y_plus.dtype

        # Initial guess from piecewise law
        u_plus = self.compute_u_plus(y_plus)

        # Newton iterations
        for _ in range(10):
            kappa = self._kappa
            E_param = self._E
            exp_term = math.exp(-kappa * E_param)
            ku = kappa * u_plus

            # f(u+) = u+ + exp(-kE) * [exp(ku) - 1 - ku - ku^2/2 - ku^3/6] - y+
            exp_ku = torch.exp(ku.clamp(max=50.0))  # prevent overflow
            f = (
                u_plus
                + exp_term * (exp_ku - 1.0 - ku - 0.5 * ku**2 - ku**3 / 6.0)
                - y_plus
            )

            # f'(u+) = 1 + exp(-kE) * [k * exp(ku) - k - k^2 * u - k^3 * u^2 / 2]
            df = (
                1.0
                + exp_term * (kappa * exp_ku - kappa - kappa**2 * u_plus - 0.5 * kappa**3 * u_plus**2)
            )

            df_safe = df.abs().clamp(min=1e-30)
            delta = f / df_safe
            u_plus = (u_plus - delta).clamp(min=0.0)

            if delta.abs().max() < 1e-8:
                break

        return u_plus

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Apply wall function value to the field.

        If a ``value`` coefficient is provided, sets face values.
        Otherwise the field is unchanged (nut computed externally).

        Parameters
        ----------
        field : torch.Tensor
            The field to modify.
        patch_idx : int, optional
            Start index into the field for contiguous patch data.
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

        Enhanced wall-function BCs modify the effective viscosity field
        rather than contributing to the matrix directly.
        """
        device = get_device()
        dtype = get_default_dtype()
        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)
        return diag, source

    def __repr__(self) -> str:
        return (
            f"EnhancedWallFunctionBC(patch={self._patch.name}, "
            f"kappa={self._kappa}, E={self._E})"
        )
