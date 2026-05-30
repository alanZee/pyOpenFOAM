"""
Enhanced turbulent intensity inlet boundary condition (v7).

Extends ``turbulentIntensityInlet6`` with wall-distance correction
and a turbulence cascade limiter::

    k_raw = 1.5 * (I * |U|)^2
    // Wall-distance correction
    y_plus = u_tau_est * y / nu
    I_wall = I * (1 + wallCoeff * exp(-y_plus / yPlusCrit))
    // Cascade limiter
    eps_est = C_mu^0.75 * k_raw^1.5 / l_mix
    tau_t = k_raw / (eps_est + 1e-30)
    tau_flow = lengthScale / (|U| + 1e-30)
    cascade_ratio = tau_t / (tau_flow + 1e-30)
    I_eff = I_wall * min(1, cascadeCoeff / (cascade_ratio + 1e-30))
    k = 1.5 * (I_eff * |U|)^2
    k = clamp(k, kMin, kMax)

In OpenFOAM syntax::

    type        turbulentIntensityInlet7;
    intensity   0.05;
    lengthScale 0.01;
    Cmu         0.09;
    kMin        1e-10;
    kMax        100.0;
    wallDist    0.01;
    wallCoeff   0.5;
    yPlusCrit   11.0;
    cascadeCoeff 10.0;
    value       uniform 0.01;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentIntensityInlet7BC"]


@BoundaryCondition.register("turbulentIntensityInlet7")
class TurbulentIntensityInlet7BC(BoundaryCondition):
    """v7 enhanced turbulent intensity inlet with wall-distance correction.

    Coefficients:
        - ``intensity``: Base turbulence intensity (default 0.05).
        - ``lengthScale``: Turbulent length scale for cascade estimate (m, default 0.01).
        - ``Cmu``: Model constant (default 0.09).
        - ``kMin``: Minimum turbulent kinetic energy (default 1e-10).
        - ``kMax``: Maximum turbulent kinetic energy (default 100.0).
        - ``wallDist``: Near-wall distance estimate (m, default 0.01).
        - ``wallCoeff``: Wall-correction amplitude (default 0.5).
        - ``yPlusCrit``: Critical y+ for wall correction (default 11.0).
        - ``cascadeCoeff``: Turbulent cascade limiting coefficient (default 10.0).
        - ``value``: Initial k value (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._intensity = float(self._coeffs.get("intensity", 0.05))
        self._length_scale = float(self._coeffs.get("lengthScale", 0.01))
        self._C_mu = float(self._coeffs.get("Cmu", 0.09))
        self._k_min = float(self._coeffs.get("kMin", 1e-10))
        self._k_max = float(self._coeffs.get("kMax", 100.0))
        self._wall_dist = float(self._coeffs.get("wallDist", 0.01))
        self._wall_coeff = float(self._coeffs.get("wallCoeff", 0.5))
        self._y_plus_crit = float(self._coeffs.get("yPlusCrit", 11.0))
        self._cascade_coeff = float(self._coeffs.get("cascadeCoeff", 10.0))

    @property
    def intensity(self) -> float:
        """Base turbulence intensity."""
        return self._intensity

    @property
    def length_scale(self) -> float:
        """Turbulent length scale (m)."""
        return self._length_scale

    @property
    def C_mu(self) -> float:
        """Model constant C_mu."""
        return self._C_mu

    @property
    def k_min(self) -> float:
        """Minimum turbulent kinetic energy."""
        return self._k_min

    @property
    def k_max(self) -> float:
        """Maximum turbulent kinetic energy."""
        return self._k_max

    @property
    def wall_dist(self) -> float:
        """Near-wall distance estimate (m)."""
        return self._wall_dist

    @property
    def wall_coeff(self) -> float:
        """Wall-correction amplitude."""
        return self._wall_coeff

    @property
    def y_plus_crit(self) -> float:
        """Critical y+ for wall correction."""
        return self._y_plus_crit

    @property
    def cascade_coeff(self) -> float:
        """Turbulent cascade limiting coefficient."""
        return self._cascade_coeff

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
        nu: float | None = None,
    ) -> torch.Tensor:
        """Set boundary-face k with wall-distance correction and cascade limiter.

        Args:
            field: Turbulent kinetic energy field.
            patch_idx: Optional start index.
            velocity: ``(n_faces, 3)`` velocity at boundary.
            nu: Kinematic viscosity (m2/s) for wall model.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if velocity is not None:
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
            k_raw = 1.5 * (self._intensity * u_mag) ** 2

            # Wall-distance correction
            if nu is not None and nu > 0:
                eps_est = (self._C_mu ** 0.75) * (k_raw ** 1.5) / (self._length_scale + 1e-30)
                u_tau_est = (self._C_mu ** 0.25) * torch.sqrt(torch.clamp(k_raw, min=1e-30))
                y_plus = u_tau_est * self._wall_dist / nu
                I_wall = self._intensity * (1.0 + self._wall_coeff * torch.exp(-y_plus / self._y_plus_crit))
            else:
                I_wall = torch.full((n,), self._intensity, dtype=dtype, device=device)

            # Cascade limiter
            eps_cascade = (self._C_mu ** 0.75) * (k_raw ** 1.5) / (self._length_scale + 1e-30)
            tau_t = k_raw / (eps_cascade + 1e-30)
            tau_flow = self._length_scale / (u_mag + 1e-30)
            cascade_ratio = tau_t / (tau_flow + 1e-30)
            I_eff = I_wall * torch.clamp(self._cascade_coeff / (cascade_ratio + 1e-30), max=1.0)

            k = 1.5 * (I_eff * u_mag) ** 2
            k = torch.clamp(k, self._k_min, self._k_max)
        else:
            k = torch.full((n,), max(self._k_min, 0.01), dtype=dtype, device=device)

        if patch_idx is not None:
            field[patch_idx : patch_idx + n] = k
        else:
            field[self._patch.face_indices] = k
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for v7 intensity inlet BC."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        k_default = max(self._k_min, 0.01)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * k_default)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
