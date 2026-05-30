"""
Enhanced turbulent intensity inlet boundary condition (v9).

Extends ``turbulentIntensityInlet8`` with a spectral energy correction
and a pressure-gradient sensitivity model::

    k_raw = 1.5 * (I * |U|)^2
    // Wall-distance correction (from v8)
    // Cascade limiter (from v8)
    // Reynolds-stress anisotropy (from v8)
    // Strain-rate coupling (from v8)
    // Spectral energy correction
    f_flow = |U| / l_mix
    f_turb = eps / k
    spectral_ratio = f_flow / (f_turb + 1e-30)
    I_spec = I_eff * (1 + spectralCoeff * tanh(spectral_ratio / spectralRef))
    // Pressure-gradient sensitivity
    dp_dx_norm = |dp/dx| / (rho * |U|^2 + 1e-30)
    I_pg = I_spec * (1 + pgCoeff * dp_dx_norm / (pgNormRef + dp_dx_norm))
    k = 1.5 * (I_pg * |U|)^2
    k = clamp(k, kMin, kMax)

In OpenFOAM syntax::

    type        turbulentIntensityInlet9;
    intensity   0.05;
    lengthScale 0.01;
    Cmu         0.09;
    kMin        1e-10;
    kMax        100.0;
    wallDist    0.01;
    wallCoeff   0.5;
    yPlusCrit   11.0;
    cascadeCoeff 10.0;
    anisoCoeff  0.1;
    strainCoeff 0.05;
    tauRatioRef 1.0;
    spectralCoeff 0.05;
    spectralRef 5.0;
    pgCoeff     0.05;
    pgNormRef   1.0;
    rho         1.225;
    value       uniform 0.01;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentIntensityInlet9BC"]


@BoundaryCondition.register("turbulentIntensityInlet9")
class TurbulentIntensityInlet9BC(BoundaryCondition):
    """v9 enhanced turbulent intensity inlet with spectral correction and pressure-gradient model.

    Coefficients:
        - ``intensity``: Base turbulence intensity (default 0.05).
        - ``lengthScale``: Turbulent length scale (m, default 0.01).
        - ``Cmu``: Model constant (default 0.09).
        - ``kMin``: Minimum turbulent kinetic energy (default 1e-10).
        - ``kMax``: Maximum turbulent kinetic energy (default 100.0).
        - ``wallDist``: Near-wall distance estimate (m, default 0.01).
        - ``wallCoeff``: Wall-correction amplitude (default 0.5).
        - ``yPlusCrit``: Critical y+ for wall correction (default 11.0).
        - ``cascadeCoeff``: Turbulent cascade limiting coefficient (default 10.0).
        - ``anisoCoeff``: Reynolds-stress anisotropy correction coefficient (default 0.1).
        - ``strainCoeff``: Strain-rate coupling coefficient (default 0.05).
        - ``tauRatioRef``: Reference time-scale ratio for strain coupling (default 1.0).
        - ``spectralCoeff``: Spectral energy correction coefficient (default 0.05).
        - ``spectralRef``: Reference spectral ratio (default 5.0).
        - ``pgCoeff``: Pressure-gradient correction coefficient (default 0.05).
        - ``pgNormRef``: Normalized pressure gradient reference (default 1.0).
        - ``rho``: Fluid density for pressure-gradient normalization (kg/m3, default 1.225).
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
        self._aniso_coeff = float(self._coeffs.get("anisoCoeff", 0.1))
        self._strain_coeff = float(self._coeffs.get("strainCoeff", 0.05))
        self._tau_ratio_ref = float(self._coeffs.get("tauRatioRef", 1.0))
        self._spectral_coeff = float(self._coeffs.get("spectralCoeff", 0.05))
        self._spectral_ref = float(self._coeffs.get("spectralRef", 5.0))
        self._pg_coeff = float(self._coeffs.get("pgCoeff", 0.05))
        self._pg_norm_ref = float(self._coeffs.get("pgNormRef", 1.0))
        self._rho = float(self._coeffs.get("rho", 1.225))

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

    @property
    def aniso_coeff(self) -> float:
        """Reynolds-stress anisotropy correction coefficient."""
        return self._aniso_coeff

    @property
    def strain_coeff(self) -> float:
        """Strain-rate coupling coefficient."""
        return self._strain_coeff

    @property
    def tau_ratio_ref(self) -> float:
        """Reference time-scale ratio for strain coupling."""
        return self._tau_ratio_ref

    @property
    def spectral_coeff(self) -> float:
        """Spectral energy correction coefficient."""
        return self._spectral_coeff

    @property
    def spectral_ref(self) -> float:
        """Reference spectral ratio."""
        return self._spectral_ref

    @property
    def pg_coeff(self) -> float:
        """Pressure-gradient correction coefficient."""
        return self._pg_coeff

    @property
    def pg_norm_ref(self) -> float:
        """Normalized pressure gradient reference."""
        return self._pg_norm_ref

    @property
    def rho(self) -> float:
        """Fluid density for pressure-gradient normalization (kg/m3)."""
        return self._rho

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
        nu: float | None = None,
        strain_rate: torch.Tensor | None = None,
        pressure_gradient: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Set boundary-face k with spectral and pressure-gradient corrections.

        Args:
            field: Turbulent kinetic energy field.
            patch_idx: Optional start index.
            velocity: ``(n_faces, 3)`` velocity at boundary.
            nu: Kinematic viscosity (m2/s) for wall model.
            strain_rate: ``(n_faces,)`` mean strain rate magnitude |S|.
            pressure_gradient: ``(n_faces,)`` streamwise dp/dx.
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

            # Reynolds-stress anisotropy correction
            eps_k = (self._C_mu ** 0.75) * (k ** 1.5) / (self._length_scale + 1e-30)
            nut_est = self._C_mu * k ** 2 / (eps_k + 1e-30)
            if strain_rate is not None:
                P_k = 2.0 * nut_est * strain_rate ** 2
            else:
                P_k = eps_k
            a_ratio = P_k / (eps_k + 1e-30) - 1.0
            k = k * (1.0 + self._aniso_coeff * torch.clamp(a_ratio.abs(), max=2.0))

            # Strain-rate coupling
            if strain_rate is not None:
                tau_s = k / (eps_k + 1e-30) * strain_rate
                k = k * (1.0 + self._strain_coeff * tau_s / (self._tau_ratio_ref + tau_s + 1e-30))

            # Spectral energy correction
            f_flow = u_mag / (self._length_scale + 1e-30)
            f_turb = eps_k / (k + 1e-30)
            spectral_ratio = f_flow / (f_turb + 1e-30)
            k = k * (1.0 + self._spectral_coeff * torch.tanh(spectral_ratio / (self._spectral_ref + 1e-30)))

            # Pressure-gradient sensitivity
            if pressure_gradient is not None and self._pg_coeff > 0:
                dp_dx = pressure_gradient.to(device=device, dtype=dtype)
                dp_dx_norm = dp_dx.abs() / (self._rho * u_mag ** 2 + 1e-30)
                k = k * (1.0 + self._pg_coeff * dp_dx_norm / (self._pg_norm_ref + dp_dx_norm + 1e-30))

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
        """Penalty method for v9 intensity inlet BC."""
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
