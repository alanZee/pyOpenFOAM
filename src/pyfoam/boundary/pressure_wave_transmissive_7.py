"""
Enhanced pressure wave transmissive boundary condition (v7).

Extends ``pressureWaveTransmissive6`` with adaptive NSCBC blending
and a multi-scale turbulent damping model::

    T_eff = T_ref + gamma * (p - p_ref) / (rho * Cp)
    c_eff = sqrt(gamma * R_specific * T_eff)
    Ma = |U| / c_eff
    sigma = sigma_base * (1 + gamma * Ma^2) * (1 + beta_sigma * log(1 + Re_t))
    // Adaptive NSCBC with frequency-dependent blending
    L_plus = (c_eff / (2 * lInf)) * (p - p_ref) * (1 + Ma)
    L_minus = -(c_eff / (2 * lInf)) * (p - p_ref) * (1 - Ma)
    p_nscbc = p_owner + 0.5 * rho * lInf * (L_plus + L_minus)
    // Multi-scale damping
    l_turb = sqrt(k) * lInf / (|U| + c_eff + 1e-30)
    f_damp = 1 / (1 + (f_cutoff / f_turb)^2) * exp(-dampCoeff * l_turb / lInf)
    p_damp = damping * rho * k * f_damp
    // Adaptive blending based on local Courant-like number
    Cr_local = |U| * dt / lInf
    blend_eff = blending * (1 + nscbcSigma * Cr_local)
    p_face = (1 - blend_eff) * p_wave + blend_eff * p_nscbc - p_damp

In OpenFOAM syntax::

    type        pressureWaveTransmissive7;
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
    betaSigma   0.01;
    ReTRef      100.0;
    nscbcSigma  0.3;
    dampCoeff   0.5;
    dt          1e-3;
    value       uniform 101325;
"""

from __future__ import annotations

import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["PressureWaveTransmissive7BC"]


@BoundaryCondition.register("pressureWaveTransmissive7")
class PressureWaveTransmissive7BC(BoundaryCondition):
    """Enhanced pressure wave transmissive BC (v7) with adaptive NSCBC.

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
        - ``betaSigma`` (float): Turbulent Re sensitivity for adaptive sigma (default 0.01).
        - ``ReTRef`` (float): Reference turbulent Reynolds number (default 100.0).
        - ``nscbcSigma`` (float): NSCBC relaxation coefficient (default 0.3).
        - ``dampCoeff`` (float): Multi-scale damping coefficient (default 0.5).
        - ``dt`` (float): Time step for Courant estimation (s, default 1e-3).
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
        self._beta_sigma = float(self._coeffs.get("betaSigma", 0.01))
        self._Re_t_ref = float(self._coeffs.get("ReTRef", 100.0))
        self._nscbc_sigma = float(self._coeffs.get("nscbcSigma", 0.3))
        self._damp_coeff = float(self._coeffs.get("dampCoeff", 0.5))
        self._dt = float(self._coeffs.get("dt", 1e-3))

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
    def beta_sigma(self) -> float:
        """Turbulent Re sensitivity for adaptive sigma."""
        return self._beta_sigma

    @property
    def Re_t_ref(self) -> float:
        """Reference turbulent Reynolds number."""
        return self._Re_t_ref

    @property
    def nscbc_sigma(self) -> float:
        """NSCBC relaxation coefficient."""
        return self._nscbc_sigma

    @property
    def damp_coeff(self) -> float:
        """Multi-scale damping coefficient."""
        return self._damp_coeff

    @property
    def dt(self) -> float:
        """Time step for Courant estimation (s)."""
        return self._dt

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
    ) -> torch.Tensor:
        """Apply v7 enhanced wave transmissive pressure BC with adaptive NSCBC.

        Args:
            field: Pressure field.
            patch_idx: Optional start index into field.
            velocity: ``(n_faces, 3)`` velocity at boundary.
            rho: Density (scalar or per-face tensor).
            c: Speed of sound (m/s).  If None, computed from temperature.
            k: ``(n_faces,)`` turbulent kinetic energy for damping.
            T_ref: Reference temperature (K) for speed-of-sound correction.
            nu: Kinematic viscosity (m2/s) for Re_t estimation.
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

        # Adaptive sigma with turbulent Reynolds number correction
        sigma = self._sigma_base * (1.0 + self._gamma * ma ** 2)
        if k is not None and nu is not None and nu > 0 and self._beta_sigma != 0:
            eps_est = k ** 1.5 / (0.01 + 1e-30)
            Re_t = k ** 2 / (nu * eps_est + 1e-30)
            Re_t_mean = Re_t.mean().item()
            sigma = sigma * (1.0 + self._beta_sigma * math.log(1.0 + Re_t_mean / self._Re_t_ref))

        dp = owner_vals - self._field_inf

        # Improved NSCBC: multi-wave decomposition
        L_plus = (c_val / (2.0 * self._l_inf + 1e-30)) * dp * (1.0 + ma)
        L_minus = -(c_val / (2.0 * self._l_inf + 1e-30)) * dp * (1.0 - ma)

        # NSCBC pressure from wave decomposition
        denom = rho_val * c_val * self._l_inf * (1.0 + ma) + 1e-30
        p_wave = owner_vals - rho_val * c_val * (u_n - c_val) * dp / denom
        p_nscbc = owner_vals + 0.5 * rho_val * self._l_inf * (L_plus + L_minus)

        # Adaptive blending based on local Courant-like number
        Cr_local = u_mag * self._dt / (self._l_inf + 1e-30)
        blend_eff = self._blending * (1.0 + self._nscbc_sigma * Cr_local)
        p_face = (1.0 - blend_eff) * p_wave + blend_eff * p_nscbc

        # Additional NSCBC sigma relaxation
        p_face = p_face - self._nscbc_sigma * rho_val * c_val * u_n * ma / (1.0 + ma + 1e-30)

        # Multi-scale turbulent fluctuation damping
        if k is not None:
            u_turb = torch.sqrt(torch.clamp(k, min=1e-30))
            f_turb = u_turb / (self._l_inf + 1e-30)
            # Multi-scale length ratio
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
        """Relaxation-based matrix contribution for v7 wave transmissive."""
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

        total_coeff = relax_coeff * (1.0 + self._sigma_base + self._nscbc_sigma) + self._blending * area_mag

        diag.scatter_add_(0, owners, total_coeff)
        source.scatter_add_(
            0, owners, total_coeff * self._field_inf
        )

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
