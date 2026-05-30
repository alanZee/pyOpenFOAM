"""
Enhanced mapped flow rate boundary condition (v10).

Extends ``mappedFlowRate9`` with anisotropic swirl damping and
coriolis-corrected wall-distance profile::

    rho_eff = rho * (1 - beta_thermal * (T - T_ref))
    n_eff = profileExponent * (1 + beta * log10(Re / ReRef))
    // Wall-distance-weighted profile (from v9)
    // Adaptive swirl damping (from v9)
    // Anisotropic swirl damping
    e_axial = normals / |normals|
    cos_theta = dot(U, e_axial) / (|U| + 1e-30)
    swirl_aniso = swirlRatio * (1 - anisoCoeff * (1 - cos_theta^2))
    // Coriolis correction
    Omega_local = swirl_local / r
    F_cor = 2 * rho_eff * Omega_local * U_radial
    U *= (1 + coriolisCoeff * F_cor / (rho_eff * |U|^2 + 1e-30))
    // Time-averaged correction (from v9)
    m_dot_avg = tauAvg * m_dot_prev + (1 - tauAvg) * m_dot_current
    U *= m_dot_target / (m_dot_avg + 1e-30)

In OpenFOAM syntax::

    type              mappedFlowRate10;
    neighbourPatch    outlet;
    rho               1.0;
    massFlowRate      1.0;
    profileExponent   7.0;
    hydraulicDiameter 0.1;
    beta              0.1;
    ReRef             1e4;
    nCorr             3;
    betaThermal       0.0;
    TRef              300.0;
    temperature       300.0;
    swirlRatio        0.0;
    swirlExponent     1.0;
    swirlCorrection   1.0;
    decayCoeff        0.5;
    tauAvg            0.5;
    blendCoeff        1.0;
    blendExponent     2.0;
    blendWeight       0.5;
    wallDistCoeff     11.0;
    wallWeight        0.1;
    swirlReRef        1e4;
    swirlDampExp      1.0;
    anisoCoeff        0.3;
    coriolisCoeff     0.05;
    value             uniform (0 0 0);
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["MappedFlowRate10BC"]


@BoundaryCondition.register("mappedFlowRate10")
class MappedFlowRate10BC(BoundaryCondition):
    """Enhanced mapped flow rate with anisotropic swirl and coriolis correction v10.

    Coefficients:
        - ``massFlowRate`` (float): Target mass flow rate (kg/s).  Default 1.0.
        - ``rho`` (float): Reference fluid density (kg/m3).  Default 1.0.
        - ``profileExponent`` (float): Base power-law exponent (default 7.0).
        - ``hydraulicDiameter`` (float): Hydraulic diameter (m).  Default 0.1.
        - ``beta`` (float): Reynolds-number sensitivity coefficient (default 0.1).
        - ``ReRef`` (float): Reference Reynolds number for adaptation (default 1e4).
        - ``nCorr`` (int): Number of iterative correction passes (default 3).
        - ``betaThermal`` (float): Volumetric thermal expansion coefficient (1/K).  Default 0.0.
        - ``TRef`` (float): Reference temperature for density correction (K).  Default 300.0.
        - ``temperature`` (float): Current bulk temperature (K).  Default 300.0.
        - ``swirlRatio`` (float): Ratio of swirl to axial velocity at outer radius (default 0.0).
        - ``swirlExponent`` (float): Radial exponent for swirl decay (default 1.0).
        - ``swirlCorrection`` (float): Swirl angular momentum correction factor (default 1.0).
        - ``decayCoeff`` (float): Radial exponential decay coefficient for swirl (default 0.5).
        - ``tauAvg`` (float): Time-averaging relaxation factor (default 0.5).
        - ``blendCoeff`` (float): Radial blending coefficient (default 1.0).
        - ``blendExponent`` (float): Exponent for blending weight function (default 2.0).
        - ``blendWeight`` (float): Weight for turbulent profile blending (default 0.5).
        - ``wallDistCoeff`` (float): y+ scale for wall-distance weighting (default 11.0).
        - ``wallWeight`` (float): Near-wall profile weight (default 0.1).
        - ``swirlReRef`` (float): Reference Re for swirl damping (default 1e4).
        - ``swirlDampExp`` (float): Exponent for swirl damping function (default 1.0).
        - ``anisoCoeff`` (float): Anisotropy coefficient for swirl damping (default 0.3).
        - ``coriolisCoeff`` (float): Coriolis correction coefficient (default 0.05).
        - ``neighbourPatch`` (str): Name of the mapped neighbour patch.
        - ``value``: Initial velocity (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._mass_flow_rate = float(self._coeffs.get("massFlowRate", 1.0))
        self._rho = float(self._coeffs.get("rho", 1.0))
        self._profile_exponent = float(self._coeffs.get("profileExponent", 7.0))
        self._hydraulic_diameter = float(self._coeffs.get("hydraulicDiameter", 0.1))
        self._beta = float(self._coeffs.get("beta", 0.1))
        self._Re_ref = float(self._coeffs.get("ReRef", 1e4))
        self._n_corr = int(self._coeffs.get("nCorr", 3))
        self._beta_thermal = float(self._coeffs.get("betaThermal", 0.0))
        self._T_ref = float(self._coeffs.get("TRef", 300.0))
        self._temperature = float(self._coeffs.get("temperature", 300.0))
        self._swirl_ratio = float(self._coeffs.get("swirlRatio", 0.0))
        self._swirl_exponent = float(self._coeffs.get("swirlExponent", 1.0))
        self._swirl_correction = float(self._coeffs.get("swirlCorrection", 1.0))
        self._decay_coeff = float(self._coeffs.get("decayCoeff", 0.5))
        self._tau_avg = float(self._coeffs.get("tauAvg", 0.5))
        self._blend_coeff = float(self._coeffs.get("blendCoeff", 1.0))
        self._blend_exponent = float(self._coeffs.get("blendExponent", 2.0))
        self._blend_weight = float(self._coeffs.get("blendWeight", 0.5))
        self._wall_dist_coeff = float(self._coeffs.get("wallDistCoeff", 11.0))
        self._wall_weight = float(self._coeffs.get("wallWeight", 0.1))
        self._swirl_Re_ref = float(self._coeffs.get("swirlReRef", 1e4))
        self._swirl_damp_exp = float(self._coeffs.get("swirlDampExp", 1.0))
        self._aniso_coeff = float(self._coeffs.get("anisoCoeff", 0.3))
        self._coriolis_coeff = float(self._coeffs.get("coriolisCoeff", 0.05))
        self._m_dot_prev: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mass_flow_rate(self) -> float:
        """Target mass flow rate (kg/s)."""
        return self._mass_flow_rate

    @property
    def rho(self) -> float:
        """Reference fluid density (kg/m3)."""
        return self._rho

    @property
    def profile_exponent(self) -> float:
        """Base power-law profile exponent."""
        return self._profile_exponent

    @property
    def hydraulic_diameter(self) -> float:
        """Hydraulic diameter (m)."""
        return self._hydraulic_diameter

    @property
    def beta(self) -> float:
        """Reynolds-number sensitivity coefficient."""
        return self._beta

    @property
    def Re_ref(self) -> float:
        """Reference Reynolds number."""
        return self._Re_ref

    @property
    def n_corr(self) -> int:
        """Number of iterative correction passes."""
        return self._n_corr

    @property
    def beta_thermal(self) -> float:
        """Volumetric thermal expansion coefficient (1/K)."""
        return self._beta_thermal

    @property
    def T_ref(self) -> float:
        """Reference temperature (K)."""
        return self._T_ref

    @property
    def temperature(self) -> float:
        """Current bulk temperature (K)."""
        return self._temperature

    @property
    def swirl_ratio(self) -> float:
        """Swirl-to-axial velocity ratio at outer radius."""
        return self._swirl_ratio

    @property
    def swirl_exponent(self) -> float:
        """Radial exponent for swirl decay."""
        return self._swirl_exponent

    @property
    def swirl_correction(self) -> float:
        """Swirl angular momentum correction factor."""
        return self._swirl_correction

    @property
    def decay_coeff(self) -> float:
        """Radial exponential decay coefficient for swirl."""
        return self._decay_coeff

    @property
    def tau_avg(self) -> float:
        """Time-averaging relaxation factor."""
        return self._tau_avg

    @property
    def blend_coeff(self) -> float:
        """Radial blending coefficient."""
        return self._blend_coeff

    @property
    def blend_exponent(self) -> float:
        """Exponent for blending weight function."""
        return self._blend_exponent

    @property
    def blend_weight(self) -> float:
        """Weight for turbulent profile blending."""
        return self._blend_weight

    @property
    def wall_dist_coeff(self) -> float:
        """y+ scale for wall-distance weighting."""
        return self._wall_dist_coeff

    @property
    def wall_weight(self) -> float:
        """Near-wall profile weight."""
        return self._wall_weight

    @property
    def swirl_Re_ref(self) -> float:
        """Reference Re for swirl damping."""
        return self._swirl_Re_ref

    @property
    def swirl_damp_exp(self) -> float:
        """Exponent for swirl damping function."""
        return self._swirl_damp_exp

    @property
    def aniso_coeff(self) -> float:
        """Anisotropy coefficient for swirl damping."""
        return self._aniso_coeff

    @property
    def coriolis_coeff(self) -> float:
        """Coriolis correction coefficient."""
        return self._coriolis_coeff

    @property
    def neighbour_patch_name(self) -> str | None:
        """Name of the mapped neighbour patch."""
        return self._coeffs.get("neighbourPatch", self._patch.neighbour_patch)

    # ------------------------------------------------------------------
    # BoundaryCondition interface
    # ------------------------------------------------------------------

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
        nu: float | None = None,
    ) -> torch.Tensor:
        """Set boundary-face velocity from v10 mapped mass flow rate with anisotropic swirl.

        Args:
            field: Velocity field ``(n_total, 3)``.
            patch_idx: Optional start index into field.
            velocity: ``(n_faces, 3)`` current velocity (for Re estimation).
            nu: Kinematic viscosity (m2/s) for Re calculation.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        normals = self._patch.face_normals.to(device=device, dtype=dtype)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)

        if areas.dim() > 1:
            area_mag = areas.norm(dim=1)
        else:
            area_mag = areas.abs()

        # Thermal-expansion-corrected density
        dT = self._temperature - self._T_ref
        rho_eff = self._rho * (1.0 - self._beta_thermal * dT)

        # Adaptive exponent based on Reynolds number
        n_eff = self._profile_exponent
        u_mean_for_re = torch.tensor(1.0, dtype=dtype, device=device)
        if velocity is not None and nu is not None and nu > 0:
            u_mean_for_re = torch.sqrt((velocity * velocity).sum(dim=-1)).mean()
            Re = u_mean_for_re * self._hydraulic_diameter / nu
            n_eff = self._profile_exponent * (
                1.0 + self._beta * torch.log10(torch.clamp(Re / self._Re_ref, min=1.0)).item()
            )

        # Compute profile weights with wall-distance weighting
        if n_eff > 0 and n > 1:
            r_frac = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
            w_turb = 1.0 / (1.0 + (r_frac * self._blend_coeff) ** self._blend_exponent)
            w_profile = (w_turb * self._blend_weight +
                         (1.0 - w_turb) * (1.0 - r_frac) ** n_eff)

            # Wall-distance weighting (y+ model)
            if nu is not None and nu > 0:
                u_tau_est = 0.05 * u_mean_for_re
                y_plus = u_tau_est * r_frac * self._hydraulic_diameter / (nu + 1e-30)
                w_wall = 1.0 - torch.exp(-y_plus / (self._wall_dist_coeff + 1e-30))
                weights = w_wall * w_profile + (1.0 - w_wall) * self._wall_weight
            else:
                weights = w_profile

            weighted_areas = area_mag * weights
        else:
            r_frac = torch.zeros(n, dtype=dtype, device=device)
            weights = torch.ones(n, dtype=dtype, device=device)
            weighted_areas = area_mag

        total_weighted = weighted_areas.sum()
        total_area = area_mag.sum()

        # Initial uniform velocity estimate
        if total_area > 0:
            u_base = self._mass_flow_rate / (rho_eff * total_area)
        else:
            u_base = 0.0

        # Time-averaged iterative correction
        m_dot_avg = self._m_dot_prev if self._m_dot_prev > 0 else self._mass_flow_rate
        for _ in range(max(1, self._n_corr)):
            velocity_mag = u_base * (area_mag * weights) / (total_weighted + 1e-30) * n
            m_dot_current = rho_eff * (velocity_mag * area_mag).sum()
            m_dot_avg = self._tau_avg * m_dot_avg + (1.0 - self._tau_avg) * m_dot_current.abs().item()
            if m_dot_avg > 1e-30:
                u_base *= self._mass_flow_rate / m_dot_avg

        self._m_dot_prev = m_dot_avg

        velocity_mag = u_base * (area_mag * weights) / (total_weighted + 1e-30) * n

        # Axial velocity component (along inward normal)
        vel_axial = -normals * velocity_mag.unsqueeze(-1) if normals.dim() == 2 else -normals * velocity_mag

        # Swirl velocity with anisotropic Reynolds-number damping and coriolis correction
        if self._swirl_ratio != 0 and n > 1:
            ref_axis = torch.zeros_like(normals)
            ref_axis[..., 2] = 1.0
            parallel_mask = (normals * ref_axis).sum(dim=-1).abs() > 0.99
            ref_axis[parallel_mask] = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)

            tangential = torch.cross(normals, ref_axis, dim=-1)
            tang_mag = tangential.norm(dim=-1, keepdim=True)
            tangential = tangential / (tang_mag + 1e-30)

            I_z = (r_frac ** 2 * area_mag).sum() + 1e-30
            L_target = self._swirl_ratio * self._mass_flow_rate * 0.5 * self._hydraulic_diameter
            omega_correction = self._swirl_correction * L_target / I_z

            # Adaptive swirl damping
            if nu is not None and nu > 0:
                Re_local = u_mean_for_re * self._hydraulic_diameter / (nu + 1e-30)
                swirl_damp = 1.0 / (1.0 + (Re_local / (self._swirl_Re_ref + 1e-30)) ** self._swirl_damp_exp)
            else:
                swirl_damp = 1.0

            # Anisotropic swirl: reduce swirl where flow is not aligned with axis
            e_axial = normals / (normals.norm(dim=-1, keepdim=True) + 1e-30)
            if velocity is not None and velocity.shape[0] == n:
                cos_theta = (velocity * e_axial).sum(dim=-1) / (velocity.norm(dim=-1) + 1e-30)
            else:
                cos_theta = torch.ones(n, dtype=dtype, device=device)
            swirl_aniso = self._swirl_ratio * (1.0 - self._aniso_coeff * (1.0 - cos_theta ** 2))

            swirl_local = swirl_aniso * swirl_damp * torch.exp(-self._decay_coeff * r_frac)
            u_theta = (swirl_local * omega_correction *
                       velocity_mag * (r_frac ** self._swirl_exponent))

            # Coriolis correction
            if self._coriolis_coeff > 0:
                Omega_local = swirl_local * omega_correction / (r_frac * self._hydraulic_diameter / 2.0 + 1e-30)
                F_cor = 2.0 * rho_eff * Omega_local * velocity_mag
                cor_factor = 1.0 + self._coriolis_coeff * F_cor / (rho_eff * velocity_mag ** 2 + 1e-30)
                vel_axial = vel_axial * cor_factor.unsqueeze(-1) if vel_axial.dim() == 2 else vel_axial * cor_factor

            vel_swirl = tangential * u_theta.unsqueeze(-1)
            vel = vel_axial + vel_swirl
        else:
            vel = vel_axial

        if patch_idx is not None:
            field[patch_idx : patch_idx + n] = vel
        else:
            field[self._patch.face_indices] = vel
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for v10 mapped flow rate BC."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas
        total_area = areas.abs().sum() if areas.dim() == 1 else areas.norm(dim=1).sum()
        dT = self._temperature - self._T_ref
        rho_eff = self._rho * (1.0 - self._beta_thermal * dT)
        if total_area > 0:
            u_n = self._mass_flow_rate / (rho_eff * total_area)
        else:
            u_n = 0.0

        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * u_n)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
