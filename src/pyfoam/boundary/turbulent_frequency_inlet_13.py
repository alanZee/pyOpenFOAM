"""
Enhanced turbulent frequency inlet boundary condition (v13).

Extends ``turbulentFrequencyInlet12`` with a wall-pressure-fluctuation
correction and a dynamic Kolmogorov-scale blending function::

    k = 1.5 * (I * |U|)^2
    // Two-layer model (from v12)
    // Kolmogorov-scale limiter (from v12)
    // Dynamic time-scale correction (from v12)
    // Cross-diffusion term (from v12)
    // Frequency-dependent blending (from v12)
    // Compressibility correction (from v12)
    // Pressure-gradient sensitivity (from v12)
    // Enhanced SST limiter (from v12)
    // Wall-pressure-fluctuation correction
    p_rms = rho * u_tau^2 * (1 + wallFluctCoeff * exp(-y_plus / yPlusFluct))
    omega_wp = p_rms / (mu * sqrt(k) + 1e-30)
    omega_corr = omega * (1 + wpCoeff * omega_wp / (omega + 1e-30))
    // Dynamic Kolmogorov blending
    eta = (nu^3 / eps)^0.25
    blend_kolm = clamp(eta / gridScale, 0, 1)
    omega_out = blend_kolm * omega_corr + (1 - blend_kolm) * omega_kolm
    omega_out = clamp(omega_out, omegaMin, omegaMax)

In OpenFOAM syntax::

    type        turbulentFrequencyInlet13;
    mixingLength 0.01;
    Cmu         0.09;
    kappa       0.41;
    beta1       0.075;
    betaStar    0.09;
    sigmaD      0.5;
    intensity   0.05;
    wallDist    0.01;
    yPlusLow    5.0;
    yPlusHigh   30.0;
    omegaMin    1e-4;
    omegaMax    1e6;
    dynCoeff    0.1;
    dynRatioRef 10.0;
    gridScale   0.001;
    freqBlendScale 5.0;
    yPlusBlend  30.0;
    compCoeff   0.1;
    MaLimit     0.5;
    pgCoeff     0.05;
    rho         1.225;
    wpCoeff     0.02;
    wallFluctCoeff 0.5;
    yPlusFluct  15.0;
    value       uniform 0.01;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentFrequencyInlet13BC"]


@BoundaryCondition.register("turbulentFrequencyInlet13")
class TurbulentFrequencyInlet13BC(BoundaryCondition):
    """v13 enhanced turbulent frequency inlet with wall-pressure-fluctuation correction.

    Coefficients:
        - ``mixingLength``: Mixing length (m, default 0.01).
        - ``Cmu``: Model constant (default 0.09).
        - ``kappa``: Von Karman constant (default 0.41).
        - ``beta1``: k-omega model constant (default 0.075).
        - ``betaStar``: k-omega model constant beta* (default 0.09).
        - ``sigmaD``: Cross-diffusion coefficient (default 0.5).
        - ``intensity``: Fallback turbulence intensity (default 0.05).
        - ``wallDist``: Near-wall distance estimate (m, default 0.01).
        - ``yPlusLow``: Lower y+ bound for buffer-layer blending (default 5.0).
        - ``yPlusHigh``: Upper y+ bound for log-law blending (default 30.0).
        - ``omegaMin``: Minimum omega clamp (default 1e-4).
        - ``omegaMax``: Maximum omega clamp (default 1e6).
        - ``dynCoeff``: Dynamic time-scale correction coefficient (default 0.1).
        - ``dynRatioRef``: Reference time-scale ratio for dynamic correction (default 10.0).
        - ``gridScale``: Local grid scale for Kolmogorov limiter (m, default 0.001).
        - ``freqBlendScale``: Scale for frequency-dependent blending (default 5.0).
        - ``yPlusBlend``: y+ scale for F2 blending function (default 30.0).
        - ``compCoeff``: Compressibility correction coefficient (default 0.1).
        - ``MaLimit``: Turbulent Mach number limit (default 0.5).
        - ``pgCoeff``: Pressure-gradient sensitivity coefficient (default 0.05).
        - ``rho``: Fluid density for pressure-gradient normalization (kg/m3, default 1.225).
        - ``wpCoeff``: Wall-pressure-fluctuation correction coefficient (default 0.02).
        - ``wallFluctCoeff``: Wall fluctuation amplitude coefficient (default 0.5).
        - ``yPlusFluct``: y+ scale for wall-pressure fluctuation (default 15.0).
        - ``value``: Initial omega value (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._mixing_length = float(self._coeffs.get("mixingLength", 0.01))
        self._C_mu = float(self._coeffs.get("Cmu", 0.09))
        self._kappa = float(self._coeffs.get("kappa", 0.41))
        self._beta1 = float(self._coeffs.get("beta1", 0.075))
        self._beta_star = float(self._coeffs.get("betaStar", 0.09))
        self._sigma_d = float(self._coeffs.get("sigmaD", 0.5))
        self._intensity = float(self._coeffs.get("intensity", 0.05))
        self._wall_dist = float(self._coeffs.get("wallDist", 0.01))
        self._y_plus_low = float(self._coeffs.get("yPlusLow", 5.0))
        self._y_plus_high = float(self._coeffs.get("yPlusHigh", 30.0))
        self._omega_min = float(self._coeffs.get("omegaMin", 1e-4))
        self._omega_max = float(self._coeffs.get("omegaMax", 1e6))
        self._dyn_coeff = float(self._coeffs.get("dynCoeff", 0.1))
        self._dyn_ratio_ref = float(self._coeffs.get("dynRatioRef", 10.0))
        self._grid_scale = float(self._coeffs.get("gridScale", 0.001))
        self._freq_blend_scale = float(self._coeffs.get("freqBlendScale", 5.0))
        self._y_plus_blend = float(self._coeffs.get("yPlusBlend", 30.0))
        self._comp_coeff = float(self._coeffs.get("compCoeff", 0.1))
        self._Ma_limit = float(self._coeffs.get("MaLimit", 0.5))
        self._pg_coeff = float(self._coeffs.get("pgCoeff", 0.05))
        self._rho = float(self._coeffs.get("rho", 1.225))
        self._wp_coeff = float(self._coeffs.get("wpCoeff", 0.02))
        self._wall_fluct_coeff = float(self._coeffs.get("wallFluctCoeff", 0.5))
        self._y_plus_fluct = float(self._coeffs.get("yPlusFluct", 15.0))

    @property
    def mixing_length(self) -> float:
        """Mixing length (m)."""
        return self._mixing_length

    @property
    def C_mu(self) -> float:
        """Model constant C_mu."""
        return self._C_mu

    @property
    def kappa(self) -> float:
        """Von Karman constant."""
        return self._kappa

    @property
    def beta1(self) -> float:
        """k-omega model constant beta_1."""
        return self._beta1

    @property
    def beta_star(self) -> float:
        """k-omega model constant beta*."""
        return self._beta_star

    @property
    def sigma_d(self) -> float:
        """Cross-diffusion coefficient."""
        return self._sigma_d

    @property
    def intensity(self) -> float:
        """Fallback turbulence intensity."""
        return self._intensity

    @property
    def wall_dist(self) -> float:
        """Near-wall distance estimate (m)."""
        return self._wall_dist

    @property
    def y_plus_low(self) -> float:
        """Lower y+ bound for buffer-layer blending."""
        return self._y_plus_low

    @property
    def y_plus_high(self) -> float:
        """Upper y+ bound for log-law blending."""
        return self._y_plus_high

    @property
    def omega_min(self) -> float:
        """Minimum omega clamp."""
        return self._omega_min

    @property
    def omega_max(self) -> float:
        """Maximum omega clamp."""
        return self._omega_max

    @property
    def dyn_coeff(self) -> float:
        """Dynamic time-scale correction coefficient."""
        return self._dyn_coeff

    @property
    def dyn_ratio_ref(self) -> float:
        """Reference time-scale ratio for dynamic correction."""
        return self._dyn_ratio_ref

    @property
    def grid_scale(self) -> float:
        """Local grid scale for Kolmogorov limiter (m)."""
        return self._grid_scale

    @property
    def freq_blend_scale(self) -> float:
        """Scale for frequency-dependent blending."""
        return self._freq_blend_scale

    @property
    def y_plus_blend(self) -> float:
        """y+ scale for F2 blending function."""
        return self._y_plus_blend

    @property
    def comp_coeff(self) -> float:
        """Compressibility correction coefficient."""
        return self._comp_coeff

    @property
    def Ma_limit(self) -> float:
        """Turbulent Mach number limit."""
        return self._Ma_limit

    @property
    def pg_coeff(self) -> float:
        """Pressure-gradient sensitivity coefficient."""
        return self._pg_coeff

    @property
    def rho(self) -> float:
        """Fluid density for pressure-gradient normalization (kg/m3)."""
        return self._rho

    @property
    def wp_coeff(self) -> float:
        """Wall-pressure-fluctuation correction coefficient."""
        return self._wp_coeff

    @property
    def wall_fluct_coeff(self) -> float:
        """Wall fluctuation amplitude coefficient."""
        return self._wall_fluct_coeff

    @property
    def y_plus_fluct(self) -> float:
        """y+ scale for wall-pressure fluctuation."""
        return self._y_plus_fluct

    def _two_layer_omega(
        self, k: torch.Tensor, y: float, nu: float, n: int,
        device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute two-layer omega with buffer/log-law blending."""
        omega_log = torch.sqrt(k) / (self._C_mu ** 0.25 * self._kappa * y + 1e-30)
        omega_buf = 6.0 * nu / (self._beta1 * y ** 2 + 1e-30)

        u_tau = (self._C_mu ** 0.25) * torch.sqrt(k)
        y_plus = u_tau * y / (nu + 1e-30)

        blend = torch.clamp(
            (y_plus - self._y_plus_low) / (self._y_plus_high - self._y_plus_low + 1e-30),
            0.0, 1.0,
        )

        return blend * omega_log + (1.0 - blend) * omega_buf

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        k: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
        nu: float | None = None,
        c: float | None = None,
        pressure_gradient: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Set boundary-face omega with wall-pressure-fluctuation and Kolmogorov blending.

        Args:
            field: Specific dissipation rate field.
            patch_idx: Optional start index.
            k: ``(n_faces,)`` turbulent kinetic energy.
            velocity: ``(n_faces, 3)`` velocity at boundary.
            nu: Kinematic viscosity (m2/s) for two-layer model.
            c: Speed of sound (m/s) for compressibility correction.
            pressure_gradient: ``(n_faces,)`` streamwise dp/dx for correction.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if k is not None and nu is not None and nu > 0:
            omega_base = self._two_layer_omega(k, self._wall_dist, nu, n, device, dtype)

            # Kolmogorov-scale limiter
            eps_kolm = nu ** 3 / (self._grid_scale ** 4 + 1e-30)
            omega_kolm = torch.sqrt(torch.tensor(eps_kolm, dtype=dtype, device=device) / (self._C_mu * k + 1e-30))
            omega = torch.max(omega_base, omega_kolm)

            # Dynamic time-scale correction
            tau_t = 1.0 / (self._beta_star * omega + 1e-30)
            tau_K = torch.sqrt(nu / (self._C_mu * k * omega + 1e-30))
            omega_dyn = omega * (1.0 + self._dyn_coeff * torch.tanh(
                tau_t / (tau_K + 1e-30) / self._dyn_ratio_ref
            ))

            # Cross-diffusion term (SST CD_kw)
            if n > 1:
                d_omega = torch.diff(omega_dyn, prepend=omega_dyn[:1])
                sigma_d_term = torch.clamp(self._sigma_d * d_omega.abs() / (omega_dyn + 1e-30), min=0.0)
                omega_cd = omega_dyn + sigma_d_term / (self._beta_star * omega_dyn + 1e-30)
            else:
                omega_cd = omega_dyn

            # Frequency-dependent blending
            omega_ratio = omega_cd / (self._omega_min + 1e-30)
            freq_blend = torch.clamp(torch.log(1.0 + omega_ratio) / (self._freq_blend_scale + 1e-30), 0.0, 1.0)
            omega_blend = freq_blend * omega_cd + (1.0 - freq_blend) * omega_base

            # Compressibility correction
            c_eff = c if c is not None else 343.0
            Ma_t = torch.sqrt(2.0 * torch.clamp(k, min=1e-30)) / (c_eff + 1e-30)
            Ma_eff = torch.clamp(Ma_t, max=self._Ma_limit)
            omega_comp = omega_blend * (1.0 + self._comp_coeff * Ma_eff ** 2)

            # Pressure-gradient sensitivity
            if pressure_gradient is not None and self._pg_coeff > 0:
                dp_dx = pressure_gradient.to(device=device, dtype=dtype)
                if velocity is not None:
                    u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
                else:
                    u_mag = torch.ones(n, dtype=dtype, device=device)
                dp_dx_norm = dp_dx.abs() / (self._rho * u_mag ** 2 + 1e-30)
                omega_comp = omega_comp * (1.0 + self._pg_coeff * dp_dx_norm)

            # Wall-pressure-fluctuation correction
            if self._wp_coeff > 0:
                u_tau = (self._C_mu ** 0.25) * torch.sqrt(torch.clamp(k, min=1e-30))
                y_plus = u_tau * self._wall_dist / nu
                p_rms = self._rho * u_tau ** 2 * (1.0 + self._wall_fluct_coeff * torch.exp(-y_plus / (self._y_plus_fluct + 1e-30)))
                omega_wp = p_rms / (nu * torch.sqrt(torch.clamp(k, min=1e-30)) + 1e-30)
                omega_comp = omega_comp * (1.0 + self._wp_coeff * omega_wp / (omega_comp + 1e-30))

            # Dynamic Kolmogorov blending
            eps_local = self._C_mu * k * omega_comp
            eta = (nu ** 3 / (eps_local + 1e-30)) ** 0.25
            blend_kolm = torch.clamp(eta / (self._grid_scale + 1e-30), 0.0, 1.0)
            omega = blend_kolm * omega_comp + (1.0 - blend_kolm) * omega_kolm

            # Enhanced SST limiter with F2 blending function
            F2_arg = self._kappa * self._wall_dist / (self._C_mu ** 0.25 * torch.sqrt(k) * self._y_plus_blend + 1e-30)
            F2 = 1.0 / (1.0 + F2_arg ** 2)
            omega_sst = F2 * torch.sqrt(k) / (self._C_mu ** 0.25 * self._kappa * self._wall_dist + 1e-30)
            omega = torch.max(omega, omega_sst)
        elif k is not None:
            l = torch.full((n,), self._mixing_length, dtype=dtype, device=device)
            omega = torch.sqrt(k) / (self._C_mu ** 0.25 * l + 1e-30)
        elif velocity is not None:
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
            k_est = 1.5 * (self._intensity * u_mag) ** 2
            l = torch.full((n,), self._mixing_length, dtype=dtype, device=device)
            omega = torch.sqrt(k_est) / (self._C_mu ** 0.25 * l + 1e-30)
        else:
            omega = torch.full((n,), 0.01, dtype=dtype, device=device)

        # Clamp to physical range
        omega = torch.clamp(omega, self._omega_min, self._omega_max)

        if patch_idx is not None:
            field[patch_idx : patch_idx + n] = omega
        else:
            field[self._patch.face_indices] = omega
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for v13 enhanced omega inlet BC."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        omega_default = 0.01

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * omega_default)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
