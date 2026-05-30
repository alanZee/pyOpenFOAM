"""
Enhanced outlet phase mean velocity boundary condition (v7).

Extends ``outletPhaseMeanVelocity6`` with a turbulent viscosity gradient
correction and an enhanced mass conservation using face-normal velocity weighting::

    alpha_safe = max(alpha, alphaMin)
    // Turbulent Prandtl correction (from v6)
    // TKE coupling (from v6)
    // Pressure-relaxation (from v6)
    // Wall-distance velocity correction (from v6)
    // Turbulent viscosity gradient correction
    dnut_dy = (nut_face - nut_interior) / y
    U_nut = U_outlet * (1 + nutGradCoeff * dnut_dy * y / (nut_face + nu + 1e-30))
    // Face-normal velocity weighting for mass conservation
    U_n = dot(U_nut, n)
    m_dot_face = rho * alpha * U_n * A
    m_dot_target = m_dot_total * A / A_total
    U *= m_dot_target / (m_dot_face + 1e-30)
    U = clamp(|U|, 0, Umax) * direction

In OpenFOAM syntax::

    type              outletPhaseMeanVelocity7;
    Umean             uniform (1 0 0);
    alphaField        alpha.gas;
    phaseName         gas;
    alphaMin          1e-4;
    hydraulicDiameter 0.1;
    mu                1e-3;
    Umax              100.0;
    turbWeight        0.0;
    intensity         0.05;
    rho               1.0;
    Cmu               0.09;
    tkeCoeff          0.0;
    pressureRelax     0.0;
    pRef              101325;
    nu                1e-5;
    prandtlCoeff      0.85;
    prandtlCorr       0.1;
    ReTRef            100.0;
    wallDistCoeff     11.0;
    wallCoeff         0.1;
    nutGradCoeff      0.05;
    value             uniform (0 0 0);
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["OutletPhaseMeanVelocity7BC"]


@BoundaryCondition.register("outletPhaseMeanVelocity7")
class OutletPhaseMeanVelocity7BC(BoundaryCondition):
    """Enhanced outlet phase mean velocity BC v7 with nut gradient correction.

    Coefficients:
        - ``Umean``: Desired phase mean velocity vector (m/s).
        - ``alphaField``: Name of the volume fraction field (informational).
        - ``phaseName``: Phase name (informational).
        - ``alphaMin``: Minimum volume fraction threshold (default 1e-4).
        - ``hydraulicDiameter``: Hydraulic diameter (m, default 0.1).
        - ``mu``: Dynamic viscosity (Pa s, default 1e-3).
        - ``Umax``: Maximum velocity magnitude clamp (m/s, default 100.0).
        - ``turbWeight``: Turbulent-flux weighting coefficient (default 0.0).
        - ``intensity``: Turbulence intensity for turbulent weighting (default 0.05).
        - ``rho``: Fluid density for mass conservation (kg/m3, default 1.0).
        - ``Cmu``: Model constant for TKE-nut estimation (default 0.09).
        - ``tkeCoeff``: TKE-velocity coupling coefficient (default 0.0).
        - ``pressureRelax``: Pressure-relaxation correction coefficient (default 0.0).
        - ``pRef``: Reference pressure for pressure-relaxation (Pa, default 101325).
        - ``nu``: Kinematic viscosity for nut estimation (m2/s, default 1e-5).
        - ``prandtlCoeff``: Turbulent Prandtl blending coefficient (default 0.85).
        - ``prandtlCorr``: Prandtl correction strength (default 0.1).
        - ``ReTRef``: Reference turbulent Reynolds number for Prandtl (default 100.0).
        - ``wallDistCoeff``: y+ scale for wall-distance velocity correction (default 11.0).
        - ``wallCoeff``: Near-wall velocity fraction (default 0.1).
        - ``nutGradCoeff``: Turbulent viscosity gradient correction coefficient (default 0.05).
        - ``value``: Initial velocity (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._Umean = self._parse_vector("Umean", [0.0, 0.0, 0.0])
        self._phase_name = str(self._coeffs.get("phaseName", ""))
        self._alpha_field = str(self._coeffs.get("alphaField", ""))
        self._alpha_min = float(self._coeffs.get("alphaMin", 1e-4))
        self._hydraulic_diameter = float(self._coeffs.get("hydraulicDiameter", 0.1))
        self._mu = float(self._coeffs.get("mu", 1e-3))
        self._Umax = float(self._coeffs.get("Umax", 100.0))
        self._turb_weight = float(self._coeffs.get("turbWeight", 0.0))
        self._intensity = float(self._coeffs.get("intensity", 0.05))
        self._rho = float(self._coeffs.get("rho", 1.0))
        self._C_mu = float(self._coeffs.get("Cmu", 0.09))
        self._tke_coeff = float(self._coeffs.get("tkeCoeff", 0.0))
        self._pressure_relax = float(self._coeffs.get("pressureRelax", 0.0))
        self._p_ref = float(self._coeffs.get("pRef", 101325.0))
        self._nu = float(self._coeffs.get("nu", 1e-5))
        self._prandtl_coeff = float(self._coeffs.get("prandtlCoeff", 0.85))
        self._prandtl_corr = float(self._coeffs.get("prandtlCorr", 0.1))
        self._Re_t_ref = float(self._coeffs.get("ReTRef", 100.0))
        self._wall_dist_coeff = float(self._coeffs.get("wallDistCoeff", 11.0))
        self._wall_coeff = float(self._coeffs.get("wallCoeff", 0.1))
        self._nut_grad_coeff = float(self._coeffs.get("nutGradCoeff", 0.05))

    def _parse_vector(self, key: str, default: list[float]) -> torch.Tensor:
        """Parse a vector coefficient."""
        raw = self._coeffs.get(key, default)
        if isinstance(raw, torch.Tensor):
            return raw.to(dtype=get_default_dtype(), device=get_device())
        return torch.tensor(raw, dtype=get_default_dtype(), device=get_device())

    @property
    def Umean(self) -> torch.Tensor:
        """Prescribed phase mean velocity."""
        return self._Umean

    @property
    def phase_name(self) -> str:
        """Phase name."""
        return self._phase_name

    @property
    def alpha_field(self) -> str:
        """Volume fraction field name."""
        return self._alpha_field

    @property
    def alpha_min(self) -> float:
        """Minimum volume fraction threshold."""
        return self._alpha_min

    @property
    def hydraulic_diameter(self) -> float:
        """Hydraulic diameter (m)."""
        return self._hydraulic_diameter

    @property
    def mu(self) -> float:
        """Dynamic viscosity (Pa s)."""
        return self._mu

    @property
    def Umax(self) -> float:
        """Maximum velocity magnitude clamp (m/s)."""
        return self._Umax

    @property
    def turb_weight(self) -> float:
        """Turbulent-flux weighting coefficient."""
        return self._turb_weight

    @property
    def intensity(self) -> float:
        """Turbulence intensity."""
        return self._intensity

    @property
    def rho(self) -> float:
        """Fluid density (kg/m3)."""
        return self._rho

    @property
    def C_mu(self) -> float:
        """Model constant for TKE-nut estimation."""
        return self._C_mu

    @property
    def tke_coeff(self) -> float:
        """TKE-velocity coupling coefficient."""
        return self._tke_coeff

    @property
    def pressure_relax(self) -> float:
        """Pressure-relaxation correction coefficient."""
        return self._pressure_relax

    @property
    def p_ref(self) -> float:
        """Reference pressure for pressure-relaxation (Pa)."""
        return self._p_ref

    @property
    def nu(self) -> float:
        """Kinematic viscosity (m2/s)."""
        return self._nu

    @property
    def prandtl_coeff(self) -> float:
        """Turbulent Prandtl blending coefficient."""
        return self._prandtl_coeff

    @property
    def prandtl_corr(self) -> float:
        """Prandtl correction strength."""
        return self._prandtl_corr

    @property
    def Re_t_ref(self) -> float:
        """Reference turbulent Reynolds number for Prandtl."""
        return self._Re_t_ref

    @property
    def wall_dist_coeff(self) -> float:
        """y+ scale for wall-distance velocity correction."""
        return self._wall_dist_coeff

    @property
    def wall_coeff(self) -> float:
        """Near-wall velocity fraction."""
        return self._wall_coeff

    @property
    def nut_grad_coeff(self) -> float:
        """Turbulent viscosity gradient correction coefficient."""
        return self._nut_grad_coeff

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        alpha: torch.Tensor | None = None,
        pressure_gradient: torch.Tensor | None = None,
        k_field: torch.Tensor | None = None,
        epsilon_field: torch.Tensor | None = None,
        pressure: torch.Tensor | None = None,
        nut_field: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply v7 enhanced outlet phase-mean velocity BC with nut gradient correction.

        Args:
            field: Velocity field ``(n_cells_or_faces, 3)``.
            patch_idx: Optional start index into *field*.
            alpha: Phase volume fraction ``(n_faces,)``.
            pressure_gradient: ``(n_faces,)`` streamwise dp/dx for correction.
            k_field: ``(n_faces,)`` turbulent kinetic energy field.
            epsilon_field: ``(n_faces,)`` turbulent dissipation rate field.
            pressure: ``(n_faces,)`` pressure field for pressure-relaxation.
            nut_field: ``(n_faces,)`` turbulent viscosity field for gradient correction.
        """
        device = field.device
        dtype = field.dtype
        n_faces = self._patch.n_faces

        Umean = self._Umean.to(device=device, dtype=dtype)
        normals = self._patch.face_normals.to(device=device, dtype=dtype)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        if areas.dim() > 1:
            area_mag = areas.norm(dim=1)
        else:
            area_mag = areas.abs()

        if alpha is not None:
            alpha_safe = alpha.to(device=device, dtype=dtype).clamp(min=self._alpha_min)
            if self._turb_weight > 0:
                Umean_mag = Umean.norm() + 1e-30
                u_local_mag = Umean.unsqueeze(0).expand(n_faces, -1).norm(dim=-1)
                k_local = 0.5 * u_local_mag ** 2 * self._intensity ** 2
                alpha_turb = alpha_safe * (1.0 + self._turb_weight * torch.sqrt(k_local) / Umean_mag)
            else:
                alpha_turb = alpha_safe

            velocity = Umean.unsqueeze(0) * alpha_turb.unsqueeze(-1) / alpha_safe.unsqueeze(-1)
        else:
            alpha_safe = torch.ones(n_faces, dtype=dtype, device=device)
            velocity = Umean.unsqueeze(0).expand(n_faces, -1).clone()

        # TKE coupling
        if self._tke_coeff > 0:
            if k_field is not None:
                k_face = k_field.to(device=device, dtype=dtype)
                eps_face = (epsilon_field.to(device=device, dtype=dtype)
                            if epsilon_field is not None
                            else (self._C_mu ** 0.75) * (k_face ** 1.5) / (0.01 + 1e-30))
                nut_face = self._C_mu * k_face ** 2 / (eps_face + 1e-30)
            else:
                u_local_mag = velocity.norm(dim=-1)
                k_local = 0.5 * u_local_mag ** 2 * self._intensity ** 2
                nut_face = self._C_mu * k_local ** 2 / (
                    (self._C_mu ** 0.75) * (k_local ** 1.5) / (0.01 + 1e-30) + 1e-30
                )
            tke_factor = 1.0 - self._tke_coeff * nut_face / (self._nu + nut_face + 1e-30)
            velocity = velocity * tke_factor.unsqueeze(-1)

            # Turbulent Prandtl correction
            if self._prandtl_corr > 0:
                eps_for_Re = eps_face if k_field is not None else (
                    (self._C_mu ** 0.75) * (k_local ** 1.5) / (0.01 + 1e-30)
                )
                Re_t = k_face ** 2 / (self._nu * eps_for_Re + 1e-30) if k_field is not None else (
                    k_local ** 2 / (self._nu * (self._C_mu ** 0.75) * (k_local ** 1.5) / (0.01 + 1e-30) + 1e-30)
                )
                Pr_t = self._prandtl_coeff + (1.0 - self._prandtl_coeff) * torch.exp(
                    -Re_t / (self._Re_t_ref + 1e-30)
                )
                nut_Pr = nut_face * Pr_t
                pr_factor = 1.0 - self._prandtl_corr * nut_Pr / (self._nu + nut_Pr + 1e-30)
                velocity = velocity * pr_factor.unsqueeze(-1)

        # Wall-distance velocity correction
        if self._wall_coeff < 1.0 and self._nu > 0:
            u_mag_est = velocity.norm(dim=-1)
            u_tau_est = 0.05 * u_mag_est
            y_plus = u_tau_est * self._hydraulic_diameter / (self._nu + 1e-30)
            f_wall = 1.0 - torch.exp(-y_plus / (self._wall_dist_coeff + 1e-30))
            wall_factor = self._wall_coeff + (1.0 - self._wall_coeff) * f_wall
            velocity = velocity * wall_factor.unsqueeze(-1)

        # Turbulent viscosity gradient correction
        if self._nut_grad_coeff > 0 and nut_field is not None:
            nut_face = nut_field.to(device=device, dtype=dtype)
            # Estimate interior nut (use face value as proxy with small offset)
            nut_interior = nut_face * 0.8
            y = self._hydraulic_diameter / 2.0
            dnut_dy = (nut_face - nut_interior) / (y + 1e-30)
            nut_factor = 1.0 + self._nut_grad_coeff * dnut_dy * y / (nut_face + self._nu + 1e-30)
            velocity = velocity * nut_factor.unsqueeze(-1)

        # Pressure-relaxation correction
        if self._pressure_relax > 0 and pressure is not None:
            p_face = pressure.to(device=device, dtype=dtype)
            Umean_mag_sq = Umean.norm() ** 2 + 1e-30
            p_relax = self._pressure_relax * (p_face - self._p_ref) / (self._rho * Umean_mag_sq + 1e-30)
            velocity = velocity * (1.0 - p_relax.unsqueeze(-1))

        # Face-normal weighted mass conservation correction
        U_n = (velocity * normals).sum(dim=-1)
        m_dot_target = self._rho * (Umean * normals).sum(dim=-1) * area_mag
        m_dot_current = self._rho * alpha_safe * U_n * area_mag
        m_dot_total_target = m_dot_target.sum()
        m_dot_total_current = m_dot_current.sum()
        if m_dot_total_current.abs() > 1e-30:
            velocity = velocity * (m_dot_total_target / m_dot_total_current)

        # Pressure-gradient correction
        if self._pressure_relax > 0 and pressure_gradient is not None:
            dpdx = pressure_gradient.to(device=device, dtype=dtype)
            D_h = self._hydraulic_diameter
            U_correction = -dpdx * D_h ** 2 / (32.0 * self._mu * alpha_safe + 1e-30)
            U_dir = Umean / (Umean.norm() + 1e-30)
            velocity = velocity + 0.5 * U_correction.unsqueeze(-1) * U_dir.unsqueeze(0)

        # Clamp velocity magnitude
        u_mag = velocity.norm(dim=-1, keepdim=True)
        velocity = velocity * torch.clamp(self._Umax / (u_mag + 1e-30), max=1.0)

        if patch_idx is not None:
            field[patch_idx : patch_idx + n_faces] = velocity
        else:
            field[self._patch.face_indices] = velocity
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for v7 outlet phase mean velocity BC."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        Umean = self._Umean.to(device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = 0.5 * deltas * areas

        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * Umean[0])

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
