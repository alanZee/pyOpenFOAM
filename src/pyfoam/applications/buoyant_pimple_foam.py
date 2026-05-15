"""
buoyantPimpleFoam — transient buoyant compressible solver.

Implements the PIMPLE algorithm for transient buoyant, turbulent
flow of compressible fluids including radiation, for ventilation and
heat-transfer applications.

Combines the buoyancy features of buoyantSimpleFoam with the
transient PIMPLE algorithm of rhoPimpleFoam:
- Gravity vector g (read from ``constant/g``)
- Hydrostatic pressure decomposition: p = p_rgh + ρg·h
- Buoyancy-driven flow via density differences
- Transient energy equation with gravity source: ρ(U·g)
- P1 radiation model for thermal radiation
- PIMPLE algorithm: outer loop + PISO inner corrections
- Time derivative terms: ∂(ρU)/∂t, ∂(ρe)/∂t

Algorithm (per time step):
1. Store old fields (U_old, p_old, T_old, rho_old)
2. Outer corrector loop:
   a. Momentum predictor (with buoyancy source ρg + time derivative)
   b. PISO inner pressure correction loop:
      - Compute HbyA and face flux
      - Buoyancy flux correction (phig)
      - Solve pressure equation (p_rgh form)
      - Correct velocity and flux
   c. Update pressure from p_rgh: p = p_rgh + ρgh
   d. Update density from EOS: ρ = ρ(p, T)
   e. Solve energy equation (with radiation, buoyancy work, time derivative)
   f. Update density again
   g. Check convergence

Usage::

    from pyfoam.applications.buoyant_pimple_foam import BuoyantPimpleFoam

    solver = BuoyantPimpleFoam("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData
from pyfoam.thermophysical.thermo import BasicThermo, create_air_thermo
from pyfoam.models.radiation import RadiationModel

from .buoyant_simple_foam import BuoyantSimpleFoam
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["BuoyantPimpleFoam"]

logger = logging.getLogger(__name__)


class BuoyantPimpleFoam(BuoyantSimpleFoam):
    """Transient buoyant compressible PIMPLE solver.

    Solves the transient compressible Navier-Stokes equations with
    buoyancy (gravity) effects, radiation, and energy equation coupling.

    Inherits buoyancy infrastructure from BuoyantSimpleFoam:
    - Gravity vector g
    - Hydrostatic pressure decomposition (p_rgh = p - ρgh)
    - Radiation model (P1)
    - Buoyancy source in momentum (ρg)
    - Buoyancy work in energy (ρ(U·g))

    Adds transient PIMPLE algorithm from rhoPimpleFoam:
    - Time derivative terms: ∂(ρU)/∂t, ∂(ρe)/∂t
    - Outer corrector loop with under-relaxation
    - PISO inner pressure correction loop
    - Old field storage for time derivatives

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    thermo : BasicThermo, optional
        Thermophysical model. If None, uses air defaults.
    gravity : tuple[float, float, float], optional
        Gravity vector (m/s²). If None, reads from ``constant/g``.
    radiation : RadiationModel, optional
        Radiation model. If None, P1 radiation is used.

    Attributes
    ----------
    g : torch.Tensor
        ``(3,)`` gravity vector (m/s²).
    gh : torch.Tensor
        ``(n_cells,)`` gravity·cell_centre dot product.
    ghf : torch.Tensor
        ``(n_faces,)`` gravity·face_centre dot product.
    p_rgh : torch.Tensor
        ``(n_cells,)`` pressure minus hydrostatic component (Pa).
    radiation : RadiationModel
        Radiation model instance.
    U_old, p_old, T_old, rho_old : torch.Tensor
        Fields from previous time step (for time derivatives).
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        thermo: BasicThermo | None = None,
        gravity: tuple[float, float, float] | None = None,
        radiation: RadiationModel | None = None,
    ) -> None:
        # Initialize parent (BuoyantSimpleFoam → RhoSimpleFoam → SolverBase)
        super().__init__(case_path, thermo=thermo, gravity=gravity, radiation=radiation)

        # Override PIMPLE settings (parent reads SIMPLE settings)
        self._read_pimple_settings()

        # Store old fields for time derivative
        self.U_old = self.U.clone()
        self.p_old = self.p.clone()
        self.T_old = self.T.clone()
        self.rho_old = self.rho.clone()

        logger.info("BuoyantPimpleFoam ready: %s", self.thermo)
        if self.turbulence_enabled:
            logger.info("  Turbulence: %s", self.ras)

    # ------------------------------------------------------------------
    # PIMPLE settings reading
    # ------------------------------------------------------------------

    def _read_pimple_settings(self) -> None:
        """Read PIMPLE settings from fvSolution, overriding SIMPLE settings."""
        fv = self.case.fvSolution

        # PIMPLE-specific settings
        self.n_outer_correctors = int(
            fv.get_path("PIMPLE/nOuterCorrectors", 3)
        )
        self.n_correctors = int(
            fv.get_path("PIMPLE/nCorrectors", 2)
        )
        self.n_non_orth_correctors = int(
            fv.get_path("PIMPLE/nNonOrthogonalCorrectors", 0)
        )

        # Relaxation factors (may differ from SIMPLE)
        self.alpha_p = float(fv.get_path("PIMPLE/relaxationFactors/p", 0.3))
        self.alpha_U = float(fv.get_path("PIMPLE/relaxationFactors/U", 0.7))
        self.alpha_T = float(fv.get_path("PIMPLE/relaxationFactors/T", 1.0))

        self.convergence_tolerance = float(
            fv.get_path("PIMPLE/convergenceTolerance", 1e-4)
        )
        self.max_outer_iterations = int(
            fv.get_path("PIMPLE/maxOuterIterations", 100)
        )

        # Solver tolerances (may be under PIMPLE or default solvers)
        self.p_tolerance = float(fv.get_path("solvers/p/tolerance", 1e-6))
        self.p_max_iter = int(fv.get_path("solvers/p/maxIter", 1000))
        self.U_tolerance = float(fv.get_path("solvers/U/tolerance", 1e-6))
        self.U_max_iter = int(fv.get_path("solvers/U/maxIter", 1000))
        self.T_tolerance = float(fv.get_path("solvers/T/tolerance", 1e-6))
        self.T_max_iter = int(fv.get_path("solvers/T/maxIter", 1000))

    # ------------------------------------------------------------------
    # Main run loop (transient PIMPLE)
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the buoyantPimpleFoam solver.

        Returns:
            Final :class:`ConvergenceData`.
        """
        time_loop = TimeLoop(
            start_time=self.start_time,
            end_time=self.end_time,
            delta_t=self.delta_t,
            write_interval=self.write_interval,
            write_control=self.write_control,
        )

        convergence = ConvergenceMonitor(
            tolerance=self.convergence_tolerance,
            min_steps=1,
        )

        logger.info("Starting buoyantPimpleFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  nOuterCorrectors=%d, nCorrectors=%d",
                     self.n_outer_correctors, self.n_correctors)
        logger.info("  relaxation: alpha_U=%.2f, alpha_p=%.2f, alpha_T=%.2f",
                     self.alpha_U, self.alpha_p, self.alpha_T)
        logger.info("  gravity=%s", self.g.tolist())

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            # Store old fields for time derivative
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()
            self.T_old = self.T.clone()
            self.rho_old = self.rho.clone()

            # Update turbulence model (if active)
            mu_eff = self._update_turbulence()

            self.U, self.p, self.p_rgh, self.T, self.phi, self.rho, conv = (
                self._buoyant_pimple_iteration(mu_eff=mu_eff)
            )
            last_convergence = conv

            residuals = {
                "U": conv.U_residual,
                "p": conv.p_residual,
                "cont": conv.continuity_error,
            }
            converged = convergence.update(step + 1, residuals)

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("Converged at time step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        if last_convergence is not None:
            if last_convergence.converged:
                logger.info("buoyantPimpleFoam completed successfully (converged)")
            else:
                logger.warning("buoyantPimpleFoam completed without full convergence")

        return last_convergence or ConvergenceData()

    # ------------------------------------------------------------------
    # PIMPLE iteration with buoyancy
    # ------------------------------------------------------------------

    def _buoyant_pimple_iteration(
        self,
        mu_eff: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ConvergenceData]:
        """Run one PIMPLE time step with buoyancy.

        Combines the buoyancy features of BuoyantSimpleFoam with the
        transient PIMPLE algorithm of RhoPimpleFoam.

        Parameters
        ----------
        mu_eff : torch.Tensor, optional
            Effective dynamic viscosity field (molecular + turbulent).

        Returns:
            Tuple of (U, p, p_rgh, T, phi, rho, convergence).
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        U = self.U.clone()
        p = self.p.clone()
        p_rgh = self.p_rgh.clone()
        T = self.T.clone()
        phi = self.phi.clone()
        rho = self.rho.clone()

        convergence = ConvergenceData()

        n_outer = min(self.n_outer_correctors, self.max_outer_iterations)

        for outer in range(n_outer):
            U_prev = U.clone()
            p_prev = p.clone()
            T_prev = T.clone()

            # ============================================
            # Step 1: Momentum predictor (with buoyancy + time derivative)
            # ============================================
            U, A_p, H = self._buoyant_momentum_predictor_transient(
                U, p_rgh, phi, rho, mu_eff=mu_eff
            )

            # ============================================
            # Step 2: PISO inner pressure correction loop
            # ============================================
            for corr in range(self.n_correctors):
                # HbyA
                HbyA = H / A_p.abs().clamp(min=1e-30).unsqueeze(-1)

                # Face flux
                n_internal = mesh.n_internal_faces
                int_owner = mesh.owner[:n_internal]
                int_neigh = mesh.neighbour
                w = mesh.face_weights[:n_internal]

                HbyA_face = (
                    w.unsqueeze(-1) * HbyA[int_owner]
                    + (1.0 - w).unsqueeze(-1) * HbyA[int_neigh]
                )
                phiHbyA = (HbyA_face * mesh.face_areas[:n_internal]).sum(dim=1)

                # Buoyancy flux correction (from BuoyantSimpleFoam)
                rho_face = 0.5 * (
                    gather(rho, int_owner) + gather(rho, int_neigh)
                )
                A_p_face = w * gather(A_p, int_owner) + (1.0 - w) * gather(A_p, int_neigh)
                rhorAUf = rho_face / A_p_face.abs().clamp(min=1e-30)

                ghf_int = self.ghf[:n_internal]
                snGrad_rho = (gather(rho, int_neigh) - gather(rho, int_owner)) * mesh.delta_coefficients[:n_internal]
                S_mag = mesh.face_areas[:n_internal].norm(dim=1)
                phig = -rhorAUf * ghf_int * snGrad_rho * S_mag

                phiHbyA = phiHbyA + phig

                # Solve pressure equation (p_rgh form)
                p_rgh = self._solve_pressure_equation(
                    p_rgh, phiHbyA, A_p, rho, mesh
                )

                # Correct velocity
                grad_p_rgh = self._compute_grad(p_rgh, mesh)
                U = HbyA - grad_p_rgh / A_p.abs().clamp(min=1e-30).unsqueeze(-1)

                # Correct flux (internal faces only)
                p_P = gather(p_rgh, int_owner)
                p_N = gather(p_rgh, int_neigh)
                A_p_inv = 1.0 / A_p.abs().clamp(min=1e-30)
                A_p_inv_face = (
                    w * gather(A_p_inv, int_owner)
                    + (1.0 - w) * gather(A_p_inv, int_neigh)
                )
                phi_internal = phiHbyA - (p_N - p_P) * A_p_inv_face
                phi = phi.clone()
                phi[:n_internal] = phi_internal

                # Recompute H for subsequent corrections
                if corr < self.n_correctors - 1:
                    H = self._recompute_H_buoyant(U, phi, rho, mu_eff=mu_eff)

            # ============================================
            # Step 3: Under-relaxation
            # ============================================
            if self.alpha_U < 1.0:
                U = self.alpha_U * U + (1.0 - self.alpha_U) * U_prev
            if self.alpha_p < 1.0:
                p_rgh = self.alpha_p * p_rgh + (1.0 - self.alpha_p) * self.p_rgh

            # ============================================
            # Step 4: Update pressure from p_rgh
            # p = p_rgh + rho * gh
            # ============================================
            p = p_rgh + rho * self.gh

            # ============================================
            # Step 5: Update density from EOS
            # ============================================
            rho = self.thermo.rho(p, T)

            # ============================================
            # Step 6: Solve energy equation (transient, with buoyancy + radiation)
            # ============================================
            T = self._buoyant_solve_energy_equation_transient(
                T, U, phi, rho, p, T_prev, mu_eff=mu_eff
            )

            # Update density again after T update
            rho = self.thermo.rho(p, T)

            # Update p_rgh for consistency
            p_rgh = p - rho * self.gh

            # ============================================
            # Step 7: Check convergence
            # ============================================
            U_residual = self._compute_residual(U, U_prev)
            p_residual = self._compute_residual(p, p_prev)
            T_residual = self._compute_residual(T, T_prev)
            continuity_error = self._compute_continuity_error(phi, rho)

            convergence.p_residual = p_residual
            convergence.U_residual = U_residual
            convergence.continuity_error = continuity_error
            convergence.outer_iterations = outer + 1

            if outer % 5 == 0 or outer < 3:
                logger.info(
                    "buoyantPimple outer %d: U_res=%.6e, p_res=%.6e, "
                    "T_res=%.6e, cont=%.6e",
                    outer, U_residual, p_residual, T_residual, continuity_error,
                )

            if continuity_error < self.convergence_tolerance and outer > 0:
                convergence.converged = True
                break

        return U, p, p_rgh, T, phi, rho, convergence

    # ------------------------------------------------------------------
    # Transient momentum predictor with buoyancy
    # ------------------------------------------------------------------

    def _buoyant_momentum_predictor_transient(
        self,
        U: torch.Tensor,
        p_rgh: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        mu_eff: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve transient momentum equation with buoyancy source.

        The transient momentum equation for buoyant flow is:
            ∂(ρU)/∂t + ∇·(ρUU) = -∇p_rgh + ∇·(μ∇U) + ρg

        Discretised as:
            (ρV/Δt + A_p)·U = H + ρV·U_old/Δt - ∇p_rgh + ρg

        Parameters
        ----------
        mu_eff : torch.Tensor, optional
            Effective dynamic viscosity (molecular + turbulent).
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour
        cell_volumes = mesh.cell_volumes
        face_areas = mesh.face_areas
        delta_coeffs = mesh.delta_coefficients

        int_owner = owner[:n_internal]
        int_neigh = neighbour

        # Viscosity: molecular or effective
        if mu_eff is not None:
            mu = mu_eff
        else:
            mu = self.thermo.mu(T=self.T)
        mu_face = 0.5 * (gather(mu, int_owner) + gather(mu, int_neigh))

        # Diffusion coefficient
        S_mag = face_areas[:n_internal].norm(dim=1)
        delta_f = delta_coeffs[:n_internal]
        diff_coeff = mu_face * S_mag * delta_f

        # Convection (upwind) with variable density
        flux = phi[:n_internal]
        rho_P = gather(rho, int_owner)
        rho_N = gather(rho, int_neigh)
        rho_face = torch.where(flux >= 0, rho_P, rho_N)

        flux_pos = torch.where(flux >= 0, flux, torch.zeros_like(flux))
        flux_neg = torch.where(flux < 0, flux, torch.zeros_like(flux))

        cell_volumes_safe = cell_volumes.clamp(min=1e-30)
        V_P = gather(cell_volumes_safe, int_owner)
        V_N = gather(cell_volumes_safe, int_neigh)

        # Time derivative contribution: ρ * V / Δt
        dt = self.delta_t
        rho_V_dt = rho * cell_volumes / dt

        # Matrix coefficients
        lower = (-diff_coeff + flux_neg * rho_face) / V_P
        upper = (-diff_coeff - flux_pos * rho_face) / V_N

        A_p = torch.zeros(n_cells, dtype=dtype, device=device)
        A_p = A_p + scatter_add(
            (diff_coeff - flux_neg * rho_face) / V_P, int_owner, n_cells
        )
        A_p = A_p + scatter_add(
            (diff_coeff + flux_pos * rho_face) / V_N, int_neigh, n_cells
        )

        # Add time derivative to diagonal
        A_p = A_p + rho_V_dt

        # H(U)
        H = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        U_neigh = U[int_neigh]
        U_own = U[int_owner]

        owner_contrib = lower.unsqueeze(-1) * U_neigh * V_P.unsqueeze(-1)
        H.index_add_(0, int_owner, owner_contrib)

        neigh_contrib = upper.unsqueeze(-1) * U_own * V_N.unsqueeze(-1)
        H.index_add_(0, int_neigh, neigh_contrib)

        # Add time derivative source: ρ * V * U_old / dt
        H = H + rho_V_dt.unsqueeze(-1) * self.U_old

        # Pressure gradient (p_rgh form)
        w = mesh.face_weights[:n_internal]
        p_P = gather(p_rgh, int_owner)
        p_N = gather(p_rgh, int_neigh)
        p_face = w * p_P + (1.0 - w) * p_N
        p_contrib = p_face.unsqueeze(-1) * face_areas[:n_internal]

        grad_p = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad_p.index_add_(0, int_owner, p_contrib)
        grad_p.index_add_(0, int_neigh, -p_contrib)

        # Buoyancy source: ρg (per cell volume)
        rho_g = rho.unsqueeze(-1) * self.g.unsqueeze(0)

        source = H - grad_p + rho_g

        # Solve: U = source / A_p
        A_p_safe = A_p.abs().clamp(min=1e-30)
        U_solved = source / A_p_safe.unsqueeze(-1)

        # Under-relaxation
        U_new = self.alpha_U * U_solved + (1.0 - self.alpha_U) * U

        return U_new, A_p, H

    # ------------------------------------------------------------------
    # Recompute H with buoyancy (for PISO inner corrections)
    # ------------------------------------------------------------------

    def _recompute_H_buoyant(
        self,
        U: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        mu_eff: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Recompute H(U) from corrected velocity, including time derivative."""
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        # Viscosity
        if mu_eff is not None:
            mu = mu_eff
        else:
            mu = self.thermo.mu(T=self.T)
        mu_face = 0.5 * (gather(mu, int_owner) + gather(mu, int_neigh))

        S_mag = mesh.face_areas[:n_internal].norm(dim=1)
        delta_f = mesh.delta_coefficients[:n_internal]
        diff_coeff = mu_face * S_mag * delta_f

        flux = phi[:n_internal]
        rho_P = gather(rho, int_owner)
        rho_N = gather(rho, int_neigh)
        rho_face = torch.where(flux >= 0, rho_P, rho_N)

        flux_pos = torch.where(flux >= 0, flux, torch.zeros_like(flux))
        flux_neg = torch.where(flux < 0, flux, torch.zeros_like(flux))

        V_P = gather(mesh.cell_volumes.clamp(min=1e-30), int_owner)
        V_N = gather(mesh.cell_volumes.clamp(min=1e-30), int_neigh)

        lower = (-diff_coeff + flux_neg * rho_face) / V_P
        upper = (-diff_coeff - flux_pos * rho_face) / V_N

        H = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        U_neigh = U[int_neigh]
        U_own = U[int_owner]

        owner_contrib = lower.unsqueeze(-1) * U_neigh * V_P.unsqueeze(-1)
        H.index_add_(0, int_owner, owner_contrib)

        neigh_contrib = upper.unsqueeze(-1) * U_own * V_N.unsqueeze(-1)
        H.index_add_(0, int_neigh, neigh_contrib)

        # Add time derivative source: ρ * V * U_old / dt
        dt = self.delta_t
        rho_V_dt = rho * mesh.cell_volumes / dt
        H = H + rho_V_dt.unsqueeze(-1) * self.U_old

        return H

    # ------------------------------------------------------------------
    # Transient energy equation with buoyancy and radiation
    # ------------------------------------------------------------------

    def _buoyant_solve_energy_equation_transient(
        self,
        T: torch.Tensor,
        U: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        p: torch.Tensor,
        T_old: torch.Tensor,
        mu_eff: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Solve the transient energy equation with buoyancy and radiation.

        For transient buoyant flow:
            ∂(ρCpT)/∂t + ∇·(ρUCpT) = ∇·(κ_eff∇T) + p∇·U + Φ + ρ(U·g) + S_rad

        where:
            κ_eff = μ/Pr + μ_t/Prt  (effective thermal conductivity)
            Φ = 2μ(S:S) - (2/3)μ(∇·U)²  (viscous dissipation)
            ρ(U·g)  (gravity source — buoyancy work)
            S_rad  (radiation source from P1 model)

        Parameters
        ----------
        T_old : torch.Tensor
            Temperature from previous outer iteration (for under-relaxation).
        mu_eff : torch.Tensor, optional
            Effective dynamic viscosity (molecular + turbulent).
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes

        # Molecular viscosity
        mu_mol = self.thermo.mu(T)

        # Effective thermal conductivity: κ_eff = μ/Pr + μ_t/Prt
        if mu_eff is not None:
            mu_t = (mu_eff - mu_mol).clamp(min=0.0)
            kappa_eff = mu_mol / self.thermo.Pr + mu_t / self.thermo.Prt
        else:
            kappa_eff = mu_mol / self.thermo.Pr

        kappa_face = 0.5 * (gather(kappa_eff, int_owner) + gather(kappa_eff, int_neigh))

        # Diffusion coefficients
        face_areas = mesh.face_areas[:n_internal]
        S_mag = face_areas.norm(dim=1)
        delta_f = mesh.delta_coefficients[:n_internal]
        diff_coeff = kappa_face * S_mag * delta_f

        # Convection (upwind) with variable density
        flux = phi[:n_internal]
        rho_P = gather(rho, int_owner)
        rho_N = gather(rho, int_neigh)
        rho_face = torch.where(flux >= 0, rho_P, rho_N)

        flux_pos = torch.where(flux >= 0, flux, torch.zeros_like(flux))
        flux_neg = torch.where(flux < 0, flux, torch.zeros_like(flux))

        cp = self.thermo.Cp()
        cell_volumes_safe = cell_volumes.clamp(min=1e-30)
        V_P = gather(cell_volumes_safe, int_owner)
        V_N = gather(cell_volumes_safe, int_neigh)

        # Time derivative: ρ * V * Cp / Δt
        dt = self.delta_t
        rho_V_Cp_dt = rho * cell_volumes * cp / dt

        # Matrix coefficients: convection + diffusion
        lower = (-diff_coeff + flux_neg * rho_face * cp) / V_P
        upper = (-diff_coeff - flux_pos * rho_face * cp) / V_N

        diag = torch.zeros(n_cells, dtype=dtype, device=device)
        diag = diag + scatter_add(
            (diff_coeff - flux_neg * rho_face * cp) / V_P, int_owner, n_cells
        )
        diag = diag + scatter_add(
            (diff_coeff + flux_pos * rho_face * cp) / V_N, int_neigh, n_cells
        )

        # Add time derivative to diagonal
        diag = diag + rho_V_Cp_dt

        # Source 1: Viscous dissipation Φ = 2μ(S:S) - (2/3)μ(∇·U)²
        mu = mu_eff if mu_eff is not None else mu_mol
        grad_U = self._compute_grad_vector(U, mesh)
        S = 0.5 * (grad_U + grad_U.transpose(-1, -2))
        S_double_dot = (S * S).sum(dim=(1, 2))
        div_U = self._compute_div(U, phi, mesh)
        phi_viscous = 2.0 * mu * S_double_dot - (2.0 / 3.0) * mu * div_U**2

        # Source 2: p * div(U)
        # Source 3: ρ(U·g) — buoyancy work
        U_dot_g = (U * self.g.unsqueeze(0)).sum(dim=1)
        buoyancy_work = rho * U_dot_g

        # Source 4: Radiation source
        radiation_source = self.radiation.Sh(T)

        # Source 5: Time derivative source: ρ * V * Cp * T_old / dt
        time_source = rho_V_Cp_dt * self.T_old

        source = phi_viscous + p * div_U + buoyancy_work + radiation_source + time_source

        # Solve: diag * T = source - off-diag  (Jacobi iteration)
        diag_safe = diag.abs().clamp(min=1e-30)
        for _ in range(self.T_max_iter):
            off_diag = torch.zeros(n_cells, dtype=dtype, device=device)
            T_P = gather(T, int_owner)
            T_N = gather(T, int_neigh)
            off_diag = off_diag + scatter_add(lower * T_N, int_owner, n_cells)
            off_diag = off_diag + scatter_add(upper * T_P, int_neigh, n_cells)

            T_new = (source - off_diag) / diag_safe

            if (T_new - T).abs().max() < self.T_tolerance:
                break
            T = T_new

        # Under-relax (using T_old from previous outer iteration)
        if self.alpha_T < 1.0:
            T = self.alpha_T * T + (1.0 - self.alpha_T) * T_old

        return T

    # ------------------------------------------------------------------
    # Field writing
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write U, p, p_rgh, T to a time directory."""
        # Use fixed-point format for small times to avoid 'e' notation
        if abs(time) < 0.001 and time != 0:
            time_str = f"{time:.10f}".rstrip("0").rstrip(".")
        else:
            time_str = f"{time:g}"
        self.write_field("U", self.U, time_str, self._U_data)
        self.write_field("p", self.p, time_str, self._p_data)
        self.write_field("T", self.T, time_str, self._T_data)
