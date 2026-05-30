"""
Enhanced mapped flow rate boundary condition (v6).

Extends ``mappedFlowRate5`` with swirl correction and thermal expansion::

    rho_eff = rho * (1 - beta_thermal * (T - T_ref))
    Cp_eff = Cp * (1 + gamma_Cp * (T - T_ref))
    n_eff = profileExponent * (1 + beta * log10(Re / ReRef))
    U(r) = U_max * (1 - r/R)^n_eff
    U_theta(r) = swirlRatio * U_axial(r) * (R / r)^swirlExponent
    // Swirl correction: enforce angular momentum conservation
    L_target = swirlRatio * m_dot * R / (n_eff + 1)
    U *= m_dot_target / m_dot_actual  (iterative correction)

In OpenFOAM syntax::

    type              mappedFlowRate6;
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
    Cp                1005.0;
    gammaCp           0.0;
    swirlRatio        0.0;
    swirlExponent     1.0;
    swirlCorrection   1.0;
    value             uniform (0 0 0);
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["MappedFlowRate6BC"]


@BoundaryCondition.register("mappedFlowRate6")
class MappedFlowRate6BC(BoundaryCondition):
    """Enhanced mapped flow rate with swirl correction and thermal expansion v6.

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
        - ``Cp`` (float): Specific heat capacity (J/(kg K)).  Default 1005.0.
        - ``gammaCp`` (float): Temperature coefficient for Cp variation (1/K).  Default 0.0.
        - ``swirlRatio`` (float): Ratio of swirl to axial velocity at outer radius (default 0.0).
        - ``swirlExponent`` (float): Radial exponent for swirl decay (default 1.0).
        - ``swirlCorrection`` (float): Swirl angular momentum correction factor (default 1.0).
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
        self._Cp = float(self._coeffs.get("Cp", 1005.0))
        self._gamma_Cp = float(self._coeffs.get("gammaCp", 0.0))
        self._swirl_ratio = float(self._coeffs.get("swirlRatio", 0.0))
        self._swirl_exponent = float(self._coeffs.get("swirlExponent", 1.0))
        self._swirl_correction = float(self._coeffs.get("swirlCorrection", 1.0))

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
    def Cp(self) -> float:
        """Specific heat capacity (J/(kg K))."""
        return self._Cp

    @property
    def gamma_Cp(self) -> float:
        """Temperature coefficient for Cp variation (1/K)."""
        return self._gamma_Cp

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
        """Set boundary-face velocity from v6 mapped mass flow rate with swirl correction.

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

        # Thermal-expansion-corrected density with variable Cp
        dT = self._temperature - self._T_ref
        rho_eff = self._rho * (1.0 - self._beta_thermal * dT)
        Cp_eff = self._Cp * (1.0 + self._gamma_Cp * dT)

        # Adaptive exponent based on Reynolds number
        n_eff = self._profile_exponent
        if velocity is not None and nu is not None and nu > 0:
            u_mean = torch.sqrt((velocity * velocity).sum(dim=-1)).mean()
            Re = u_mean * self._hydraulic_diameter / nu
            n_eff = self._profile_exponent * (
                1.0 + self._beta * torch.log10(torch.clamp(Re / self._Re_ref, min=1.0)).item()
            )

        # Compute profile weights
        if n_eff > 0 and n > 1:
            r_frac = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
            weights = (1.0 - r_frac) ** n_eff
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

        # Iterative correction for exact mass conservation
        for _ in range(max(1, self._n_corr)):
            velocity_mag = u_base * (area_mag * weights) / (total_weighted + 1e-30) * n
            m_dot_current = rho_eff * (velocity_mag * area_mag).sum()
            if m_dot_current.abs() > 1e-30:
                u_base *= self._mass_flow_rate / m_dot_current

        velocity_mag = u_base * (area_mag * weights) / (total_weighted + 1e-30) * n

        # Axial velocity component (along inward normal)
        vel_axial = -normals * velocity_mag.unsqueeze(-1) if normals.dim() == 2 else -normals * velocity_mag

        # Swirl velocity with angular momentum correction
        if self._swirl_ratio != 0 and n > 1:
            ref_axis = torch.zeros_like(normals)
            ref_axis[..., 2] = 1.0
            parallel_mask = (normals * ref_axis).sum(dim=-1).abs() > 0.99
            ref_axis[parallel_mask] = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)

            tangential = torch.cross(normals, ref_axis, dim=-1)
            tang_mag = tangential.norm(dim=-1, keepdim=True)
            tangential = tangential / (tang_mag + 1e-30)

            # Swirl with angular momentum correction factor
            u_theta = (self._swirl_ratio * self._swirl_correction *
                       velocity_mag * (r_frac ** self._swirl_exponent))
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
        """Penalty method for v6 mapped flow rate BC."""
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
