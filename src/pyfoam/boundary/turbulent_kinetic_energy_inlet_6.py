"""
Enhanced turbulent kinetic energy inlet boundary condition (v6).

Extends ``turbulentKineticEnergyInlet5`` with a production-limited buoyancy
correction and a thermal-fluctuation energy term::

    k_intensity = 1.5 * (I * |U|)^2
    k_length = (epsilon * l_mix / C_mu^0.75)^(2/3)
    Ri = g * beta_thermal * dT * L / (U_ref^2)
    Ri_eff = clamp(Ri, RiMin, RiMax)
    // Production-limited buoyancy
    P_buoy = C_buoyancy * epsilon * max(Ri_eff, 0)
    P_limit = C_productionLimit * epsilon  (production limiter)
    k_buoyancy = min(P_buoy, P_limit)
    // Thermal fluctuation energy
    k_thermal = C_thermal * beta_thermal^2 * dT^2 * g^2 * l_mix^2 / (C_mu * epsilon + 1e-30)
    k = alpha * k_intensity + (1 - alpha) * k_length + k_buoyancy + k_thermal
    k = clamp(k, kMin, kMax)

In OpenFOAM syntax::

    type        turbulentKineticEnergyInlet6;
    intensity   0.05;
    lengthScale 0.01;
    Cmu         0.09;
    alpha       0.8;
    beta        0.05;
    ReTRef      100.0;
    kMin        1e-10;
    kMax        100.0;
    betaThermal 0.0034;
    Richardson  0.0;
    Cbuoyancy   0.1;
    CproductionLimit 2.0;
    Cthermal    0.1;
    gravityMag  9.81;
    deltaT      0.0;
    U           U;
    value       uniform 0.01;
"""

from __future__ import annotations

import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentKineticEnergyInlet6BC"]


@BoundaryCondition.register("turbulentKineticEnergyInlet6")
class TurbulentKineticEnergyInlet6BC(BoundaryCondition):
    """v6 enhanced turbulent kinetic energy inlet with buoyancy production.

    Coefficients:
        - ``intensity``: Turbulence intensity (default 0.05).
        - ``lengthScale``: Turbulent length scale (m, default 0.01).
        - ``Cmu``: Model constant (default 0.09).
        - ``alpha``: Base blending weight for intensity-based k (default 0.8).
        - ``beta``: Re_t sensitivity coefficient (default 0.05).
        - ``ReTRef``: Reference turbulent Reynolds number (default 100.0).
        - ``kMin``: Minimum k clamp (default 1e-10).
        - ``kMax``: Maximum k clamp (default 100.0).
        - ``betaThermal``: Thermal expansion coefficient (1/K, default 0.0034).
        - ``Richardson``: Gradient Richardson number (default 0.0).
        - ``Cbuoyancy``: Buoyancy production coefficient (default 0.1).
        - ``CproductionLimit``: Production-to-dissipation limit ratio (default 2.0).
        - ``Cthermal``: Thermal fluctuation energy coefficient (default 0.1).
        - ``gravityMag``: Gravitational acceleration magnitude (m/s2, default 9.81).
        - ``deltaT``: Temperature difference across inlet (K, default 0.0).
        - ``U``: Velocity field name (informational).
        - ``value``: Initial k value (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._intensity = float(self._coeffs.get("intensity", 0.05))
        self._length_scale = float(self._coeffs.get("lengthScale", 0.01))
        self._C_mu = float(self._coeffs.get("Cmu", 0.09))
        self._alpha = float(self._coeffs.get("alpha", 0.8))
        self._beta = float(self._coeffs.get("beta", 0.05))
        self._Re_t_ref = float(self._coeffs.get("ReTRef", 100.0))
        self._k_min = float(self._coeffs.get("kMin", 1e-10))
        self._k_max = float(self._coeffs.get("kMax", 100.0))
        self._beta_thermal = float(self._coeffs.get("betaThermal", 0.0034))
        self._Richardson = float(self._coeffs.get("Richardson", 0.0))
        self._C_buoyancy = float(self._coeffs.get("Cbuoyancy", 0.1))
        self._C_production_limit = float(self._coeffs.get("CproductionLimit", 2.0))
        self._C_thermal = float(self._coeffs.get("Cthermal", 0.1))
        self._gravity_mag = float(self._coeffs.get("gravityMag", 9.81))
        self._delta_T = float(self._coeffs.get("deltaT", 0.0))

    @property
    def intensity(self) -> float:
        """Turbulence intensity."""
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
    def alpha(self) -> float:
        """Base blending weight for intensity-based k."""
        return self._alpha

    @property
    def beta(self) -> float:
        """Re_t sensitivity coefficient."""
        return self._beta

    @property
    def Re_t_ref(self) -> float:
        """Reference turbulent Reynolds number."""
        return self._Re_t_ref

    @property
    def k_min(self) -> float:
        """Minimum k clamp."""
        return self._k_min

    @property
    def k_max(self) -> float:
        """Maximum k clamp."""
        return self._k_max

    @property
    def beta_thermal(self) -> float:
        """Thermal expansion coefficient (1/K)."""
        return self._beta_thermal

    @property
    def Richardson(self) -> float:
        """Gradient Richardson number."""
        return self._Richardson

    @property
    def C_buoyancy(self) -> float:
        """Buoyancy production coefficient."""
        return self._C_buoyancy

    @property
    def C_production_limit(self) -> float:
        """Production-to-dissipation limit ratio."""
        return self._C_production_limit

    @property
    def C_thermal(self) -> float:
        """Thermal fluctuation energy coefficient."""
        return self._C_thermal

    @property
    def gravity_mag(self) -> float:
        """Gravitational acceleration magnitude (m/s2)."""
        return self._gravity_mag

    @property
    def delta_T(self) -> float:
        """Temperature difference across inlet (K)."""
        return self._delta_T

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
        epsilon: torch.Tensor | None = None,
        nu: float | None = None,
    ) -> torch.Tensor:
        """Set boundary-face k with production-limited buoyancy and thermal terms.

        Args:
            field: Turbulent kinetic energy field.
            patch_idx: Optional start index.
            velocity: ``(n_faces, 3)`` velocity at boundary.
            epsilon: ``(n_faces,)`` dissipation rate (for length-scale k).
            nu: Kinematic viscosity (m2/s) for Re_t estimation.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if velocity is not None:
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
            k_intensity = 1.5 * (self._intensity * u_mag) ** 2

            if epsilon is not None:
                k_length = (epsilon * self._length_scale / (self._C_mu ** 0.75 + 1e-30)) ** (2.0 / 3.0)

                # Adaptive blending coefficient
                alpha_eff = self._alpha
                if nu is not None and nu > 0 and self._beta != 0:
                    eps_est = (self._C_mu ** 0.75) * (k_intensity ** 1.5) / (self._length_scale + 1e-30)
                    Re_t = k_intensity ** 2 / (nu * eps_est + 1e-30)
                    Re_t_mean = Re_t.mean().item()
                    alpha_eff = float(torch.clamp(
                        torch.tensor(self._alpha * (1.0 + self._beta * math.log10(
                            1.0 + Re_t_mean / self._Re_t_ref
                        ))),
                        0.0, 1.0,
                    ))

                k = alpha_eff * k_intensity + (1.0 - alpha_eff) * k_length

                # Production-limited buoyancy production
                if self._Richardson != 0:
                    P_buoy = self._C_buoyancy * epsilon * max(self._Richardson, 0.0)
                    P_limit = self._C_production_limit * epsilon
                    k_buoyancy = torch.min(
                        torch.clamp(P_buoy, min=0),
                        P_limit,
                    )
                    k = k + k_buoyancy

                # Thermal fluctuation energy
                if self._delta_T != 0:
                    k_thermal = (
                        self._C_thermal * self._beta_thermal ** 2 * self._delta_T ** 2 *
                        self._gravity_mag ** 2 * self._length_scale ** 2 /
                        (self._C_mu * epsilon + 1e-30)
                    )
                    k = k + torch.clamp(k_thermal, min=0.0)
            else:
                k = k_intensity

            # Clamp to physical range
            k = torch.clamp(k, self._k_min, self._k_max)
        else:
            k = torch.full((n,), 0.01, dtype=dtype, device=device)

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
        """Penalty method for v6 enhanced k inlet BC."""
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
