"""
Non-linear roughness wall boundary condition.

Implements a non-linear roughness wall function that modifies the
standard log-law wall function using a non-linear damping function
for sand-grain roughness.

The effective roughness height ``k_s`` shifts the log-law:

    u^+ = (1/kappa) * ln(y^+ + k_s^+)

where ``k_s^+ = k_s * u_tau / nu`` is the dimensionless roughness height.
The non-linear damping function modifies the near-wall behaviour for
transitionally rough regimes:

    f_damp = 1 - exp(-k_s^+ / A_r)

where ``A_r`` is a roughness model constant (default 2.0).

The effective wall viscosity is then:

    nu_t = kappa * u_tau * y * f_damp / ln(E * y^+ + k_s^+)

Usage::

    @BoundaryCondition.register("nonLinearRoughness")
    class NonLinearRoughnessBC(BoundaryCondition):
        ...
"""

from __future__ import annotations

import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["NonLinearRoughnessBC"]

# Von Karman constant
_KAPPA: float = 0.41
# Smooth-wall wall constant
_E: float = 9.8
# Roughness damping constant
_A_R: float = 2.0


@BoundaryCondition.register("nonLinearRoughness")
class NonLinearRoughnessBC(BoundaryCondition):
    """Non-linear roughness wall function boundary condition.

    Modifies the standard wall function with a sand-grain roughness
    model and non-linear damping for transitionally rough regimes.

    Coefficients:
        - ``ks``: sand-grain roughness height in m (default 0.0).
        - ``kappa``: von Karman constant (default 0.41).
        - ``E``: wall constant for smooth wall (default 9.8).
        - ``Ar``: roughness damping constant (default 2.0).
        - ``Cmu``: k-epsilon model constant (default 0.09).
        - ``value``: initial nut value (default 0).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._ks: float = float(self._coeffs.get("ks", 0.0))
        self._kappa: float = float(self._coeffs.get("kappa", _KAPPA))
        self._E: float = float(self._coeffs.get("E", _E))
        self._Ar: float = float(self._coeffs.get("Ar", _A_R))
        self._cmu: float = float(self._coeffs.get("Cmu", 0.09))

    def compute_nut(
        self,
        k: torch.Tensor,
        y: torch.Tensor,
        nu: float,
    ) -> torch.Tensor:
        """Compute turbulent viscosity at wall faces.

        Parameters
        ----------
        k : torch.Tensor
            Turbulent kinetic energy at wall-adjacent cells, shape ``(n_faces,)``.
        y : torch.Tensor
            Wall-normal distance from cell centre to face, shape ``(n_faces,)``.
        nu : float
            Molecular kinematic viscosity.

        Returns
        -------
        torch.Tensor
            nu_t at each wall face, shape ``(n_faces,)``.
        """
        device = get_device()
        dtype = get_default_dtype()
        k = k.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        # Friction velocity: u_tau = C_mu^{1/4} * sqrt(k)
        u_tau = self._cmu ** 0.25 * torch.sqrt(k.clamp(min=1e-16))

        # Dimensionless wall distance and roughness height
        y_plus = u_tau * y / max(nu, 1e-30)
        y_plus = y_plus.clamp(min=1e-4)
        ks_plus = self._ks * u_tau / max(nu, 1e-30)

        # Non-linear damping function for transitionally rough regime
        if self._ks > 0:
            f_damp = 1.0 - torch.exp(-ks_plus / self._Ar)
        else:
            f_damp = torch.ones_like(y_plus)

        # Effective E-shift for roughness
        E_eff = self._E
        if self._ks > 0:
            # For fully rough regime, E_eff decreases with roughness
            # Roughness function: Delta_B = (1/kappa) * ln(1 + 0.3 * ks_plus)
            delta_B = torch.log(1.0 + 0.3 * ks_plus) / self._kappa
            E_eff_raw = self._E * torch.exp(-delta_B * self._kappa)
            E_eff = float(E_eff_raw.mean().clamp(min=1e-10).item())

        # Effective nu_t from modified log-law
        denominator = torch.log(E_eff * y_plus + ks_plus + 1.0)
        denominator = denominator.clamp(min=1e-10)

        nut = self._kappa * u_tau * y * f_damp / denominator
        nut = nut.clamp(min=0.0)

        return nut

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Apply boundary values from coefficients if available."""
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
                field[patch_idx:patch_idx + n] = val_tensor
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
        """Non-linear roughness: zero matrix contribution (explicit treatment)."""
        device = get_device()
        dtype = get_default_dtype()
        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)
        return diag, source
