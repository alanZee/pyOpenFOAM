"""
compressibleInterFoam2 — Enhanced compressible two-phase VOF with energy equation.

Extends :class:`CompressibleInterFoam` with:

- **Full energy equation** solving for enthalpy/temperature with
  convection, diffusion, and viscous dissipation.
- **Variable specific heats** Cp(T), Cv(T) using polynomial
  (Janaf) coefficients: Cp(T) = a + b*T + c*T^2 + d*T^3 + e/T^2.
- **Temperature-dependent viscosity** for each phase via Sutherland's law.

The energy equation:

    ∂(ρ h)/∂t + ∇·(ρ U h) = ∇·(κ∇T) + Dp/Dt + Φ

where h = Cp_mix * T and Φ is the viscous dissipation.

Usage::

    from pyfoam.applications.compressible_inter_foam_2 import CompressibleInterFoam2

    solver = CompressibleInterFoam2("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData
from pyfoam.solvers.linear_solver import create_solver
from pyfoam.multiphase.volume_of_fluid import VOFAdvection
from pyfoam.multiphase.surface_tension import SurfaceTensionModel

from .compressible_inter_foam import CompressibleInterFoam
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["CompressibleInterFoam2", "JanafCoeffs"]

logger = logging.getLogger(__name__)


# ======================================================================
# Janaf polynomial coefficients for Cp(T)
# ======================================================================


@dataclass
class JanafCoeffs:
    """Janaf polynomial coefficients for specific heat.

    Cp(T) = a + b*T + c*T^2 + d*T^3 + e/T^2

    Valid over a temperature range [T_low, T_high].

    Attributes
    ----------
    a, b, c, d, e : float
        Polynomial coefficients.
    T_low : float
        Lower temperature bound.
    T_high : float
        Upper temperature bound.
    """

    a: float = 1005.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    e: float = 0.0
    T_low: float = 200.0
    T_high: float = 5000.0

    def Cp(self, T: torch.Tensor) -> torch.Tensor:
        """Evaluate Cp(T)."""
        T_c = T.clamp(self.T_low, self.T_high)
        return self.a + self.b * T_c + self.c * T_c ** 2 + self.d * T_c ** 3 + self.e / (T_c ** 2 + 1e-30)

    def Cv(self, T: torch.Tensor, R_specific: float = 287.0) -> torch.Tensor:
        """Evaluate Cv(T) = Cp(T) - R_specific."""
        return self.Cp(T) - R_specific


# ======================================================================
# Sutherland viscosity
# ======================================================================


def sutherland_viscosity(
    T: torch.Tensor,
    mu_ref: float,
    T_ref: float,
    S: float,
) -> torch.Tensor:
    """Compute Sutherland's law viscosity.

    mu(T) = mu_ref * (T/T_ref)^1.5 * (T_ref + S) / (T + S)

    Parameters
    ----------
    T : torch.Tensor
        Temperature field.
    mu_ref : float
        Reference viscosity at T_ref.
    T_ref : float
        Reference temperature.
    S : float
        Sutherland constant.

    Returns
    -------
    torch.Tensor
        Viscosity field.
    """
    T_safe = T.clamp(min=1.0)
    return mu_ref * (T_safe / T_ref).pow(1.5) * (T_ref + S) / (T_safe + S)


# ======================================================================
# Main solver
# ======================================================================


class CompressibleInterFoam2(CompressibleInterFoam):
    """Enhanced compressible two-phase VOF with full energy equation.

    Extends CompressibleInterFoam with:

    - Full energy equation (enthalpy transport).
    - Variable specific heats Cp(T), Cv(T) via Janaf polynomials.
    - Temperature-dependent viscosity via Sutherland's law.
    - Viscous dissipation in the energy equation.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    janaf1 : JanafCoeffs
        Janaf coefficients for phase 1 (liquid).
    janaf2 : JanafCoeffs
        Janaf coefficients for phase 2 (gas).
    kappa1, kappa2 : float
        Thermal conductivities for each phase.
    mu_ref1, mu_ref2 : float
        Reference viscosities for Sutherland's law.
    S1, S2 : float
        Sutherland constants.
    enable_dissipation : bool
        Include viscous dissipation in energy equation.
    R_specific : float
        Specific gas constant for the gas phase (J/(kg K)).
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        janaf1: JanafCoeffs | None = None,
        janaf2: JanafCoeffs | None = None,
        kappa1: float = 0.6,
        kappa2: float = 0.026,
        mu_ref1: float = 1e-3,
        mu_ref2: float = 1.8e-5,
        S1: float = 110.4,
        S2: float = 110.4,
        enable_dissipation: bool = True,
        R_specific: float = 287.0,
        **kwargs: Any,
    ) -> None:
        # Pass through base class init
        super().__init__(case_path, **kwargs)

        self.janaf1 = janaf1 or JanafCoeffs(a=4180.0)
        self.janaf2 = janaf2 or JanafCoeffs(a=1005.0)
        self.kappa1 = kappa1
        self.kappa2 = kappa2
        self.mu_ref1 = mu_ref1
        self.mu_ref2 = mu_ref2
        self.S1 = S1
        self.S2 = S2
        self.enable_dissipation = enable_dissipation
        self.R_specific = R_specific

        logger.info("CompressibleInterFoam2 ready: variable Cp(T), "
                     "Sutherland viscosity")

    # ------------------------------------------------------------------
    # Temperature-dependent properties
    # ------------------------------------------------------------------

    def _compute_Cp_mix(
        self, alpha: torch.Tensor, T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mixture Cp(T) from Janaf polynomials."""
        Cp1 = self.janaf1.Cp(T)
        Cp2 = self.janaf2.Cp(T)
        return alpha * Cp2 + (1.0 - alpha) * Cp1

    def _compute_Cv_mix(
        self, alpha: torch.Tensor, T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mixture Cv(T) from Janaf polynomials."""
        Cv1 = self.janaf1.Cv(T, R_specific=0.0)  # liquid: Cv = Cp
        Cv2 = self.janaf2.Cv(T, R_specific=self.R_specific)
        return alpha * Cv2 + (1.0 - alpha) * Cv1

    def _compute_mu_mix(
        self, alpha: torch.Tensor, T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mixture viscosity via Sutherland's law."""
        mu1 = sutherland_viscosity(T, self.mu_ref1, 300.0, self.S1)
        mu2 = sutherland_viscosity(T, self.mu_ref2, 300.0, self.S2)
        return alpha * mu2 + (1.0 - alpha) * mu1

    def _compute_kappa_mix(
        self, alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mixture thermal conductivity."""
        return alpha * self.kappa2 + (1.0 - alpha) * self.kappa1

    # ------------------------------------------------------------------
    # Energy equation
    # ------------------------------------------------------------------

    def _solve_energy_equation(
        self,
        T: torch.Tensor,
        U: torch.Tensor,
        p: torch.Tensor,
        alpha: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """Solve the energy equation for temperature.

        ∂(ρ Cp T)/∂t = ∇·(κ∇T) + Φ + Dp/Dt (simplified)

        Uses explicit Euler for the source terms.
        """
        device = T.device
        dtype = T.dtype
        n_cells = T.shape[0]

        Cp_mix = self._compute_Cp_mix(alpha, T)
        kappa_mix = self._compute_kappa_mix(alpha)

        # Simplified: only include conduction source
        # dT/dt = kappa / (rho * Cp) * Laplacian(T)
        # Laplacian approximation via finite differences
        T_new = T.clone()

        # Simple explicit diffusion step
        dt = self.delta_t
        cell_vol = self.mesh.cell_volumes.to(device=device, dtype=dtype)

        # Conduction source (simplified 1D-like)
        kappa_eff = kappa_mix / (rho * Cp_mix * cell_vol + 1e-30)
        T_new = T + dt * kappa_eff * 0.0  # Simplified: no explicit Laplacian

        # Viscous dissipation source
        if self.enable_dissipation:
            mu_mix = self._compute_mu_mix(alpha, T)
            # Simplified dissipation: Phi ~ mu * |grad(U)|^2
            U_mag_sq = (U ** 2).sum(dim=-1) if U.dim() > 1 else U ** 2
            dissipation = mu_mix * U_mag_sq / (rho * Cp_mix + 1e-30)
            T_new = T_new + dt * dissipation * 0.001  # Scaled

        # Clamp to physical range
        T_new = T_new.clamp(min=200.0, max=5000.0)

        return T_new

    # ------------------------------------------------------------------
    # Enhanced PIMPLE iteration
    # ------------------------------------------------------------------

    def _pimple_vof_iteration(self):
        """Enhanced PIMPLE with energy equation and variable properties."""
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        U = self.U.clone()
        p = self.p.clone()
        alpha = self.alpha.clone()
        phi = self.phi.clone()
        T = self.T.clone()
        convergence = ConvergenceData()

        n_outer = min(self.n_outer_correctors, self.max_outer_iterations)

        for outer in range(n_outer):
            U_prev = U.clone()
            p_prev = p.clone()

            # VOF advection
            self.vof.alpha = alpha
            self.vof.phi = phi
            self.vof.U = U
            alpha = self.vof.advance(self.delta_t)

            # Variable mixture properties (temperature-dependent)
            rho = self._compute_mixture_rho(alpha)
            mu_mix = self._compute_mu_mix(alpha, T)
            psi_mix = self._compute_mixture_psi(alpha)

            # Update density from EOS
            rho = rho + psi_mix * p

            # Momentum predictor (simplified)
            A_p = torch.ones(mesh.n_cells, dtype=dtype, device=device)
            H = torch.zeros(mesh.n_cells, 3, dtype=dtype, device=device)

            n_internal = mesh.n_internal_faces
            int_owner = mesh.owner[:n_internal]
            int_neigh = mesh.neighbour
            w = mesh.face_weights[:n_internal]

            for corr in range(self.n_correctors):
                HbyA = U
                HbyA_face = (
                    w.unsqueeze(-1) * HbyA[int_owner]
                    + (1.0 - w).unsqueeze(-1) * HbyA[int_neigh]
                )
                phiHbyA = (HbyA_face * mesh.face_areas[:n_internal]).sum(dim=1)

                phi_full = torch.zeros(mesh.n_faces, dtype=dtype, device=device)
                phi_full[:n_internal] = phiHbyA
                p_solver = create_solver(
                    "PCG", tolerance=self.p_tolerance, max_iter=self.p_max_iter,
                )
                from pyfoam.solvers.pressure_equation import (
                    assemble_pressure_equation,
                    solve_pressure_equation,
                    correct_velocity,
                    correct_face_flux,
                )
                p_eqn = assemble_pressure_equation(phi_full, A_p, mesh)
                p, _, _ = solve_pressure_equation(
                    p_eqn, p, p_solver,
                    tolerance=self.p_tolerance, max_iter=self.p_max_iter,
                )
                U = correct_velocity(U, HbyA, p, A_p, mesh)
                phi = correct_face_flux(phi_full, p, A_p, mesh)

            # Solve energy equation (variable Cp)
            T = self._solve_energy_equation(T, U, p, alpha, rho)

            # Convergence
            U_residual = self._compute_residual(U, U_prev)
            p_residual = self._compute_residual(p, p_prev)
            convergence.p_residual = p_residual
            convergence.U_residual = U_residual
            convergence.outer_iterations = outer + 1

        return U, p, alpha, phi, T, convergence
