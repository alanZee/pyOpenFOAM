"""
Enhanced pressure wave transmissive boundary condition (v10).

Extends ``pressureWaveTransmissive9`` with an acoustic impedance correction
and enhanced entropy-wave damping using local gradient information::

    T_eff = T_ref + gamma * (p - p_ref) / (rho * Cp)
    c_eff = sqrt(gamma * R_specific * T_eff)
    Ma = |U| / c_eff
    sigma = sigma_base * (1 + gamma * Ma^2)
    // Viscous damping (from v9)
    // Entropy wave (from v9)
    // Acoustic impedance correction
    Z_ac = rho * c_eff
    Z_local = rho * |U_n| + rho * c_eff * (1 + gamma * Ma)
    imp_corr = impCoeff * (Z_local - Z_ac) / (Z_ac + 1e-30)
    p_imp = owner_vals * (1 + imp_corr)
    // Gradient-based entropy damping
    grad_p = dp/dx along face-normal direction
    p_grad_damp = gradDampCoeff * rho * c_eff * grad_p * lInf / (1 + Ma)
    p_face = blend * p_nscbc + (1 - blend) * p_imp - p_damp - p_visc - p_grad_damp

In OpenFOAM syntax::

    type        pressureWaveTransmissive10;
    phi         phi;
    rho         rho;
    psi         psi;
    gamma       1.4;
    fieldInf    101325;
    lInf        1;
    blending    0.1;
    sigmaBase   0.25;
    damping     0.1;
    Rspecific   287.05;
    Cp          1005.0;
    fCutoff     1000.0;
    nscbcSigma  0.3;
    dampCoeff   0.5;
    entropyCoeff 0.05;
    dt          1e-3;
    viscCoeff   0.1;
    viscReRef   1000.0;
    mu          1.81e-5;
    impCoeff    0.1;
    gradDampCoeff 0.05;
    value       uniform 101325;
"""

from __future__ import annotations

import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["PressureWaveTransmissive10BC"]


@BoundaryCondition.register("pressureWaveTransmissive10")
class PressureWaveTransmissive10BC(BoundaryCondition):
    """Enhanced pressure wave transmissive BC (v10) with acoustic impedance correction.

    Coefficients:
        - ``fieldInf`` (float): Far-field reference pressure (Pa).  Default 101325.
        - ``lInf`` (float): Relaxation length scale (m).  Default 1.0.
        - ``gamma`` (float): Ratio of specific heats.  Default 1.4.
        - ``blending`` (float): Base NSCBC blending factor (default 0.1).
        - ``sigmaBase`` (float): Base relaxation coefficient (default 0.25).
        - ``damping`` (float): Turbulent fluctuation damping factor (default 0.1).
        - ``Rspecific`` (float): Specific gas constant (J/(kg K)).  Default 287.05.
        - ``Cp`` (float): Specific heat capacity (J/(kg K)).  Default 1005.0.
        - ``fCutoff`` (float): Frequency cutoff for damping roll-off (Hz).  Default 1000.0.
        - ``nscbcSigma`` (float): NSCBC relaxation coefficient (default 0.3).
        - ``dampCoeff`` (float): Multi-scale damping coefficient (default 0.5).
        - ``entropyCoeff`` (float): Entropy wave correction coefficient (default 0.05).
        - ``dt`` (float): Time step for Courant estimation (s, default 1e-3).
        - ``viscCoeff`` (float): Viscous damping coefficient (default 0.1).
        - ``viscReRef`` (float): Reference Reynolds number for viscous damping (default 1000.0).
        - ``mu`` (float): Dynamic viscosity (Pa s, default 1.81e-5).
        - ``impCoeff`` (float): Acoustic impedance correction coefficient (default 0.1).
        - ``gradDampCoeff`` (float): Gradient-based entropy damping coefficient (default 0.05).
        - ``rho`` (str): Density field name (informational).
        - ``phi`` (str): Flux field name (informational).
        - ``psi`` (str): Compressibility field name (informational).
        - ``value`` (float): Initial pressure (default 101325).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._field_inf = float(self._coeffs.get("fieldInf", 101325.0))
        self._l_inf = float(self._coeffs.get("lInf", 1.0))
        self._gamma = float(self._coeffs.get("gamma", 1.4))
        self._blending = float(self._coeffs.get("blending", 0.1))
        self._sigma_base = float(self._coeffs.get("sigmaBase", 0.25))
        self._damping = float(self._coeffs.get("damping", 0.1))
        self._R_specific = float(self._coeffs.get("Rspecific", 287.05))
        self._Cp = float(self._coeffs.get("Cp", 1005.0))
        self._f_cutoff = float(self._coeffs.get("fCutoff", 1000.0))
        self._nscbc_sigma = float(self._coeffs.get("nscbcSigma", 0.3))
        self._damp_coeff = float(self._coeffs.get("dampCoeff", 0.5))
        self._entropy_coeff = float(self._coeffs.get("entropyCoeff", 0.05))
        self._dt = float(self._coeffs.get("dt", 1e-3))
        self._visc_coeff = float(self._coeffs.get("viscCoeff", 0.1))
        self._visc_Re_ref = float(self._coeffs.get("viscReRef", 1000.0))
        self._mu = float(self._coeffs.get("mu", 1.81e-5))
        self._imp_coeff = float(self._coeffs.get("impCoeff", 0.1))
        self._grad_damp_coeff = float(self._coeffs.get("gradDampCoeff", 0.05))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def field_inf(self) -> float:
        """Far-field reference pressure (Pa)."""
        return self._field_inf

    @property
    def l_inf(self) -> float:
        """Relaxation length scale (m)."""
        return self._l_inf

    @property
    def gamma(self) -> float:
        """Ratio of specific heats."""
        return self._gamma

    @property
    def blending(self) -> float:
        """Base NSCBC blending factor."""
        return self._blending

    @property
    def sigma_base(self) -> float:
        """Base relaxation coefficient."""
        return self._sigma_base

    @property
    def damping(self) -> float:
        """Turbulent fluctuation damping factor."""
        return self._damping

    @property
    def R_specific(self) -> float:
        """Specific gas constant (J/(kg K))."""
        return self._R_specific

    @property
    def Cp(self) -> float:
        """Specific heat capacity (J/(kg K))."""
        return self._Cp

    @property
    def f_cutoff(self) -> float:
        """Frequency cutoff for damping roll-off (Hz)."""
        return self._f_cutoff

    @property
    def nscbc_sigma(self) -> float:
        """NSCBC relaxation coefficient."""
        return self._nscbc_sigma

    @property
    def damp_coeff(self) -> float:
        """Multi-scale damping coefficient."""
        return self._damp_coeff

    @property
    def entropy_coeff(self) -> float:
        """Entropy wave correction coefficient."""
        return self._entropy_coeff

    @property
    def dt(self) -> float:
        """Time step for Courant estimation (s)."""
        return self._dt

    @property
    def visc_coeff(self) -> float:
        """Viscous damping coefficient."""
        return self._visc_coeff

    @property
    def visc_Re_ref(self) -> float:
        """Reference Reynolds number for viscous damping."""
        return self._visc_Re_ref

    @property
    def mu(self) -> float:
        """Dynamic viscosity (Pa s)."""
        return self._mu

    @property
    def imp_coeff(self) -> float:
        """Acoustic impedance correction coefficient."""
        return self._imp_coeff

    @property
    def grad_damp_coeff(self) -> float:
        """Gradient-based entropy damping coefficient."""
        return self._grad_damp_coeff

    # ------------------------------------------------------------------
    # BoundaryCondition interface
    # ------------------------------------------------------------------

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
        rho: torch.Tensor | float | None = None,
        c: float | None = None,
        k: torch.Tensor | None = None,
        T_ref: float | None = None,
        nu: float | None = None,
        pressure_gradient: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply v10 enhanced wave transmissive pressure BC with impedance correction.

        Args:
            field: Pressure field.
            patch_idx: Optional start index into field.
            velocity: ``(n_faces, 3)`` velocity at boundary.
            rho: Density (scalar or per-face tensor).
            c: Speed of sound (m/s).  If None, computed from temperature.
            k: ``(n_faces,)`` turbulent kinetic energy for damping.
            T_ref: Reference temperature (K) for speed-of-sound correction.
            nu: Kinematic viscosity (m2/s) for Re_t estimation.
            pressure_gradient: ``(n_faces,)`` pressure gradient for entropy damping.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        owner_vals = field[owners]

        normals = self._patch.face_normals.to(device=device, dtype=dtype)

        if velocity is not None:
            u_n = (velocity * normals).sum(dim=-1)
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
        else:
            u_n = torch.zeros(n, dtype=dtype, device=device)
            u_mag = torch.zeros(n, dtype=dtype, device=device)

        if isinstance(rho, torch.Tensor):
            rho_val = rho.to(device=device, dtype=dtype)
        elif rho is not None:
            rho_val = torch.full((n,), float(rho), dtype=dtype, device=device)
        else:
            rho_val = torch.full((n,), 1.225, dtype=dtype, device=device)

        # Speed of sound
        if c is not None:
            c_val = c
        elif T_ref is not None:
            T_eff = T_ref + self._gamma * (owner_vals - self._field_inf) / (rho_val * self._Cp + 1e-30)
            c_val_tensor = torch.sqrt(self._gamma * self._R_specific * torch.clamp(T_eff, min=1.0))
            c_val = c_val_tensor.mean().item()
        else:
            c_val = 343.0

        # Local Mach number
        ma = u_mag / (c_val + 1e-30)

        # Adaptive sigma
        sigma = self._sigma_base * (1.0 + self._gamma * ma ** 2)

        dp = owner_vals - self._field_inf

        # Entropy wave correction
        if T_ref is not None:
            T_eff_local = T_ref + self._gamma * dp / (rho_val * self._Cp + 1e-30)
            T_eff_safe = torch.clamp(T_eff_local, min=1.0)
            p_safe = torch.clamp(owner_vals, min=1.0)
            s_local = self._Cp * torch.log(T_eff_safe / T_ref) - self._R_specific * torch.log(p_safe / self._field_inf)
            p_entropy = rho_val * c_val * self._entropy_coeff * s_local
        else:
            p_entropy = torch.zeros(n, dtype=dtype, device=device)

        # NSCBC wave decomposition
        L_plus = (c_val / (2.0 * self._l_inf + 1e-30)) * dp * (1.0 + ma)
        L_minus = -(c_val / (2.0 * self._l_inf + 1e-30)) * dp * (1.0 - ma)

        denom = rho_val * c_val * self._l_inf * (1.0 + ma) + 1e-30
        p_wave = owner_vals - rho_val * c_val * (u_n - c_val) * dp / denom
        p_nscbc = owner_vals + 0.5 * rho_val * self._l_inf * (L_plus + L_minus) + p_entropy

        # Acoustic impedance correction
        Z_ac = rho_val * c_val
        Z_local = rho_val * u_mag.abs() + rho_val * c_val * (1.0 + self._gamma * ma)
        imp_corr = self._imp_coeff * (Z_local - Z_ac) / (Z_ac + 1e-30)
        p_imp = owner_vals * (1.0 + imp_corr)

        # Adaptive blending based on local Courant-like number
        Cr_local = u_mag * self._dt / (self._l_inf + 1e-30)
        blend_eff = self._blending * (1.0 + self._nscbc_sigma * Cr_local)
        p_face = blend_eff * p_nscbc + (1.0 - blend_eff) * p_imp

        # NSCBC sigma relaxation
        p_face = p_face - self._nscbc_sigma * rho_val * c_val * u_n * ma / (1.0 + ma + 1e-30)

        # Viscous damping correction
        mu_eff = self._mu if nu is None else rho_val * nu
        Re_tau = rho_val * u_mag * self._l_inf / (mu_eff + 1e-30)
        p_visc = self._visc_coeff * mu_eff * u_mag / (self._l_inf * (1.0 + Re_tau / (self._visc_Re_ref + 1e-30)) + 1e-30)
        p_face = p_face - p_visc

        # Gradient-based entropy damping
        if pressure_gradient is not None and self._grad_damp_coeff > 0:
            grad_p = pressure_gradient.to(device=device, dtype=dtype)
            p_grad_damp = self._grad_damp_coeff * rho_val * c_val * grad_p * self._l_inf / (1.0 + ma + 1e-30)
            p_face = p_face - p_grad_damp

        # Multi-scale turbulent fluctuation damping
        if k is not None:
            u_turb = torch.sqrt(torch.clamp(k, min=1e-30))
            f_turb = u_turb / (self._l_inf + 1e-30)
            l_turb = u_turb * self._l_inf / (u_mag + c_val + 1e-30)
            f_damp = (1.0 / (1.0 + (f_turb / (self._f_cutoff + 1e-30)) ** 2) *
                      torch.exp(-self._damp_coeff * l_turb / (self._l_inf + 1e-30)))
            p_damp = self._damping * rho_val * k * f_damp
            p_face = p_face - p_damp

        if patch_idx is not None:
            field[patch_idx : patch_idx + n] = p_face
        else:
            field[self._patch.face_indices] = p_face
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Relaxation-based matrix contribution for v10 wave transmissive."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)

        if areas.dim() > 1:
            area_mag = areas.norm(dim=1)
        else:
            area_mag = areas.abs()

        rho_c = 1.225 * 343.0
        relax_coeff = rho_c * area_mag / (self._l_inf + 1e-30)

        total_coeff = relax_coeff * (1.0 + self._sigma_base + self._nscbc_sigma + self._visc_coeff + self._imp_coeff) + self._blending * area_mag

        diag.scatter_add_(0, owners, total_coeff)
        source.scatter_add_(0, owners, total_coeff * self._field_inf)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
