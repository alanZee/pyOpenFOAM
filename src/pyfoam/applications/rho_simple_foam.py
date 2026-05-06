"""
rhoSimpleFoam — steady-state compressible solver.

Implements the SIMPLE algorithm for steady-state compressible
Navier-Stokes equations with energy equation coupling.

Extends the incompressible SIMPLE solver with:
- Density from equation of state: ρ = ρ(p, T)
- Energy equation: ∂(ρe)/∂t + ∇·(ρUe) = ∇·(α∇T) + p∇·U + Φ
- Viscous dissipation: Φ = τ : ∇U
- Variable viscosity from transport model

Algorithm (per outer iteration):
1. Solve momentum equation (with variable density)
2. Solve pressure equation (compressible form)
3. Update density from EOS: ρ = ρ(p, T)
4. Solve energy equation
5. Update viscosity from transport model
6. Check convergence

Usage::

    from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

    solver = RhoSimpleFoam("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.fv_matrix import FvMatrix
from pyfoam.solvers.coupled_solver import CoupledSolverConfig, ConvergenceData
from pyfoam.thermophysical.thermo import BasicThermo, create_air_thermo

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["RhoSimpleFoam"]

logger = logging.getLogger(__name__)


class RhoSimpleFoam(SolverBase):
    """Steady-state compressible SIMPLE solver.

    Solves the steady-state compressible Navier-Stokes equations
    with energy equation coupling.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    thermo : BasicThermo, optional
        Thermophysical model. If None, uses air defaults.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        thermo: BasicThermo | None = None,
    ) -> None:
        super().__init__(case_path)

        # Thermophysical model
        self.thermo = thermo or create_air_thermo()

        # Read settings
        self._read_fv_solution_settings()

        # Initialise fields
        self.U, self.p, self.T, self.phi, self.rho = self._init_fields()
        self._U_data, self._p_data, self._T_data = self._init_field_data()

        logger.info("RhoSimpleFoam ready: %s", self.thermo)

    def _read_fv_solution_settings(self) -> None:
        """Read SIMPLE settings from fvSolution."""
        fv = self.case.fvSolution

        self.p_solver = str(fv.get_path("solvers/p/solver", "PCG"))
        self.p_tolerance = float(fv.get_path("solvers/p/tolerance", 1e-6))
        self.p_rel_tol = float(fv.get_path("solvers/p/relTol", 0.01))
        self.p_max_iter = int(fv.get_path("solvers/p/maxIter", 1000))

        self.U_solver = str(fv.get_path("solvers/U/solver", "PBiCGStab"))
        self.U_tolerance = float(fv.get_path("solvers/U/tolerance", 1e-6))
        self.U_rel_tol = float(fv.get_path("solvers/U/relTol", 0.01))
        self.U_max_iter = int(fv.get_path("solvers/U/maxIter", 1000))

        self.T_solver = str(fv.get_path("solvers/T/solver", "PCG"))
        self.T_tolerance = float(fv.get_path("solvers/T/tolerance", 1e-6))
        self.T_max_iter = int(fv.get_path("solvers/T/maxIter", 1000))

        self.n_non_orth_correctors = int(
            fv.get_path("SIMPLE/nNonOrthogonalCorrectors", 0)
        )
        self.alpha_p = float(fv.get_path("SIMPLE/relaxationFactors/p", 0.3))
        self.alpha_U = float(fv.get_path("SIMPLE/relaxationFactors/U", 0.7))
        self.alpha_T = float(fv.get_path("SIMPLE/relaxationFactors/T", 1.0))

        self.convergence_tolerance = float(
            fv.get_path("SIMPLE/convergenceTolerance", 1e-4)
        )
        self.max_outer_iterations = int(
            fv.get_path("SIMPLE/maxOuterIterations", 100)
        )

    def _init_fields(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialise U, p, T, phi, rho from the 0/ directory."""
        device = get_device()
        dtype = get_default_dtype()

        U_tensor, _ = self.read_field_tensor("U", 0)
        U = U_tensor.to(device=device, dtype=dtype)

        p_tensor, _ = self.read_field_tensor("p", 0)
        p = p_tensor.to(device=device, dtype=dtype)

        T_tensor, _ = self.read_field_tensor("T", 0)
        T = T_tensor.to(device=device, dtype=dtype)

        phi = torch.zeros(self.mesh.n_faces, dtype=dtype, device=device)

        # Compute initial density from EOS
        rho = self.thermo.rho(p, T)

        return U, p, T, phi, rho

    def _init_field_data(self):
        """Store raw FieldData for writing."""
        U_data = self.case.read_field("U", 0)
        p_data = self.case.read_field("p", 0)
        T_data = self.case.read_field("T", 0)
        return U_data, p_data, T_data

    def run(self) -> ConvergenceData:
        """Run the rhoSimpleFoam solver.

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

        logger.info("Starting rhoSimpleFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            self.U, self.p, self.T, self.phi, self.rho, conv = (
                self._simple_iteration()
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

        return last_convergence or ConvergenceData()

    def _simple_iteration(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ConvergenceData]:
        """Run one SIMPLE outer iteration for compressible flow.

        Returns:
            Tuple of (U, p, T, phi, rho, convergence).
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        U = self.U.clone()
        p = self.p.clone()
        T = self.T.clone()
        phi = self.phi.clone()
        rho = self.rho.clone()

        convergence = ConvergenceData()

        for outer in range(self.max_outer_iterations):
            U_prev = U.clone()
            p_prev = p.clone()

            # ============================================
            # Step 1: Momentum predictor
            # ============================================
            U, A_p, H = self._momentum_predictor(U, p, phi, rho)

            # ============================================
            # Step 2: Compute HbyA and face flux
            # ============================================
            HbyA = H / A_p.abs().clamp(min=1e-30).unsqueeze(-1)

            # Face flux from HbyA
            n_internal = mesh.n_internal_faces
            int_owner = mesh.owner[:n_internal]
            int_neigh = mesh.neighbour
            w = mesh.face_weights[:n_internal]

            HbyA_face = (
                w.unsqueeze(-1) * HbyA[int_owner]
                + (1.0 - w).unsqueeze(-1) * HbyA[int_neigh]
            )
            phiHbyA = (HbyA_face * mesh.face_areas[:n_internal]).sum(dim=1)

            # ============================================
            # Step 3: Compressible pressure equation
            # ============================================
            p = self._solve_pressure_equation(
                p, phiHbyA, A_p, rho, mesh
            )

            # Under-relax pressure
            if self.alpha_p < 1.0:
                p = self.alpha_p * p + (1.0 - self.alpha_p) * p_prev

            # ============================================
            # Step 4: Correct velocity and flux
            # ============================================
            grad_p = self._compute_grad(p, mesh)
            U = HbyA - grad_p / A_p.abs().clamp(min=1e-30).unsqueeze(-1)

            # Correct flux
            phi = phiHbyA.clone()
            p_P = gather(p, int_owner)
            p_N = gather(p, int_neigh)
            p_face = w * p_P + (1.0 - w) * p_N
            A_p_face = w * gather(A_p, int_owner) + (1.0 - w) * gather(A_p, int_neigh)
            phi = phi - (p_N - p_P) / A_p_face.clamp(min=1e-30)

            # ============================================
            # Step 5: Update density from EOS
            # ============================================
            rho = self.thermo.rho(p, T)

            # ============================================
            # Step 6: Solve energy equation
            # ============================================
            T = self._solve_energy_equation(T, U, phi, rho, p)

            # Update density again after T update
            rho = self.thermo.rho(p, T)

            # ============================================
            # Step 7: Check convergence
            # ============================================
            U_residual = self._compute_residual(U, U_prev)
            p_residual = self._compute_residual(p, p_prev)
            continuity_error = self._compute_continuity_error(phi, rho)

            convergence.p_residual = p_residual
            convergence.U_residual = U_residual
            convergence.continuity_error = continuity_error
            convergence.outer_iterations = outer + 1

            if outer % 10 == 0 or outer < 5:
                logger.info(
                    "rhoSimple iteration %d: U_res=%.6e, p_res=%.6e, "
                    "cont=%.6e",
                    outer, U_residual, p_residual, continuity_error,
                )

            if continuity_error < self.convergence_tolerance and outer > 0:
                convergence.converged = True
                break

        return U, p, T, phi, rho, convergence

    def _momentum_predictor(
        self,
        U: torch.Tensor,
        p: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve momentum equation with variable density."""
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

        # Viscosity from transport model
        mu = self.thermo.mu(T=self.T)
        mu_face = 0.5 * (gather(mu, int_owner) + gather(mu, int_neigh))

        # Diffusion coefficient
        S_mag = face_areas[:n_internal].norm(dim=1)
        delta_f = delta_coeffs[:n_internal]
        diff_coeff = mu_face * S_mag * delta_f

        # Convection (upwind)
        flux = phi[:n_internal]
        rho_P = gather(rho, int_owner)
        rho_N = gather(rho, int_neigh)
        rho_face = torch.where(flux >= 0, rho_P, rho_N)

        flux_pos = torch.where(flux >= 0, flux, torch.zeros_like(flux))
        flux_neg = torch.where(flux < 0, flux, torch.zeros_like(flux))

        cell_volumes_safe = cell_volumes.clamp(min=1e-30)
        V_P = gather(cell_volumes_safe, int_owner)
        V_N = gather(cell_volumes_safe, int_neigh)

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

        # H(U)
        H = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        U_neigh = U[int_neigh]
        U_own = U[int_owner]

        owner_contrib = lower.unsqueeze(-1) * U_neigh * V_P.unsqueeze(-1)
        H.index_add_(0, int_owner, owner_contrib)

        neigh_contrib = upper.unsqueeze(-1) * U_own * V_N.unsqueeze(-1)
        H.index_add_(0, int_neigh, neigh_contrib)

        # Pressure gradient
        w = mesh.face_weights[:n_internal]
        p_P = gather(p, int_owner)
        p_N = gather(p, int_neigh)
        p_face = w * p_P + (1.0 - w) * p_N
        p_contrib = p_face.unsqueeze(-1) * face_areas[:n_internal]

        grad_p = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad_p.index_add_(0, int_owner, p_contrib)
        grad_p.index_add_(0, int_neigh, -p_contrib)

        source = H - grad_p

        # Solve: U = source / A_p
        A_p_safe = A_p.abs().clamp(min=1e-30)
        U_solved = source / A_p_safe.unsqueeze(-1)

        # Under-relaxation
        U_new = self.alpha_U * U_solved + (1.0 - self.alpha_U) * U

        return U_new, A_p, H

    def _solve_pressure_equation(
        self,
        p: torch.Tensor,
        phiHbyA: torch.Tensor,
        A_p: torch.Tensor,
        rho: torch.Tensor,
        mesh: Any,
    ) -> torch.Tensor:
        """Solve the compressible pressure equation.

        The compressible pressure equation includes the density
        correction term: ∇·(ρ HbyA) = 0
        """
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes
        face_areas = mesh.face_areas
        delta_coeffs = mesh.delta_coefficients

        w = mesh.face_weights[:n_internal]

        # Face density
        rho_P = gather(rho, int_owner)
        rho_N = gather(rho, int_neigh)
        rho_face = 0.5 * (rho_P + rho_N)

        # 1/A_p on faces
        A_p_inv = 1.0 / A_p.abs().clamp(min=1e-30)
        A_p_inv_face = w * gather(A_p_inv, int_owner) + (1.0 - w) * gather(A_p_inv, int_neigh)

        # Laplacian coefficient
        S_mag = face_areas[:n_internal].norm(dim=1)
        delta_f = delta_coeffs[:n_internal]
        face_coeff = rho_face * A_p_inv_face * S_mag * delta_f

        V_P = gather(cell_volumes.clamp(min=1e-30), int_owner)
        V_N = gather(cell_volumes.clamp(min=1e-30), int_neigh)

        lower = -face_coeff / V_P
        upper = -face_coeff / V_N

        diag = torch.zeros(n_cells, dtype=p.dtype, device=p.device)
        diag = diag + scatter_add(face_coeff / V_P, int_owner, n_cells)
        diag = diag + scatter_add(face_coeff / V_N, int_neigh, n_cells)

        # Source: continuity error
        source = torch.zeros(n_cells, dtype=p.dtype, device=p.device)
        source = source + scatter_add(phiHbyA, int_owner, n_cells)
        source = source + scatter_add(-phiHbyA, int_neigh, n_cells)

        # Solve using simple Jacobi iteration
        diag_safe = diag.abs().clamp(min=1e-30)
        for _ in range(self.p_max_iter):
            # Off-diagonal contributions
            off_diag = torch.zeros(n_cells, dtype=p.dtype, device=p.device)
            p_P = gather(p, int_owner)
            p_N = gather(p, int_neigh)
            off_diag = off_diag + scatter_add(lower * p_N, int_owner, n_cells)
            off_diag = off_diag + scatter_add(upper * p_P, int_neigh, n_cells)

            p_new = (source - off_diag) / diag_safe

            # Check convergence
            if (p_new - p).abs().max() < self.p_tolerance:
                break
            p = p_new

        return p

    def _solve_energy_equation(
        self,
        T: torch.Tensor,
        U: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """Solve the energy equation.

        ∂(ρe)/∂t + ∇·(ρUe) = ∇·(α∇T) + p∇·U + Φ

        For steady-state: ∇·(ρUe) = ∇·(α∇T) + p∇·U + Φ
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes

        # Thermal diffusivity
        kappa = self.thermo.kappa(T)
        kappa_face = 0.5 * (gather(kappa, int_owner) + gather(kappa, int_neigh))

        # Diffusion coefficients
        face_areas = mesh.face_areas[:n_internal]
        S_mag = face_areas.norm(dim=1)
        delta_f = mesh.delta_coefficients[:n_internal]
        diff_coeff = kappa_face * S_mag * delta_f

        # Convection (upwind)
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

        # Matrix coefficients
        lower = (-diff_coeff + flux_neg * rho_face * cp) / V_P
        upper = (-diff_coeff - flux_pos * rho_face * cp) / V_N

        diag = torch.zeros(n_cells, dtype=dtype, device=device)
        diag = diag + scatter_add(
            (diff_coeff - flux_neg * rho_face * cp) / V_P, int_owner, n_cells
        )
        diag = diag + scatter_add(
            (diff_coeff + flux_pos * rho_face * cp) / V_N, int_neigh, n_cells
        )

        # Source: viscous dissipation Φ
        # Simplified: Φ = μ * (∇U + ∇U^T) : ∇U ≈ μ * |∇U|²
        mu = self.thermo.mu(T)
        grad_U = self._compute_grad_vector(U, mesh)
        # Strain rate magnitude squared
        S_mag_sq = (grad_U * grad_U).sum(dim=(1, 2))
        phi_viscous = mu * S_mag_sq

        # Source: p * div(U)
        div_U = self._compute_div(U, phi, mesh)
        source = phi_viscous + p * div_U

        # Solve: diag * T = source - off-diag
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

        # Under-relax
        if self.alpha_T < 1.0:
            T = self.alpha_T * T + (1.0 - self.alpha_T) * T

        return T

    def _compute_grad(self, phi: torch.Tensor, mesh: Any) -> torch.Tensor:
        """Compute gradient of scalar field."""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        face_areas = mesh.face_areas[:n_internal]
        w = mesh.face_weights[:n_internal]

        phi_P = gather(phi, int_owner)
        phi_N = gather(phi, int_neigh)
        phi_face = w * phi_P + (1.0 - w) * phi_N

        face_contrib = phi_face.unsqueeze(-1) * face_areas

        grad = torch.zeros(n_cells, 3, dtype=phi.dtype, device=phi.device)
        grad.index_add_(0, int_owner, face_contrib)
        grad.index_add_(0, int_neigh, -face_contrib)

        V = mesh.cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        return grad / V

    def _compute_grad_vector(self, U: torch.Tensor, mesh: Any) -> torch.Tensor:
        """Compute gradient of vector field (returns tensor)."""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        face_areas = mesh.face_areas[:n_internal]
        w = mesh.face_weights[:n_internal]

        U_P = U[int_owner]
        U_N = U[int_neigh]
        U_face = w.unsqueeze(-1) * U_P + (1.0 - w).unsqueeze(-1) * U_N

        # grad_U[i,j] = dU_j/dx_i
        # face_contrib[i,j,k] = U_face[j] * face_areas[i,k]
        grad_U = torch.zeros(n_cells, 3, 3, dtype=U.dtype, device=U.device)
        for j in range(3):
            face_contrib = U_face[:, j].unsqueeze(-1) * face_areas
            grad_U[:, :, j].index_add_(0, int_owner, face_contrib)
            grad_U[:, :, j].index_add_(0, int_neigh, -face_contrib)

        V = mesh.cell_volumes.unsqueeze(-1).unsqueeze(-1).clamp(min=1e-30)
        return grad_U / V

    def _compute_div(
        self,
        U: torch.Tensor,
        phi: torch.Tensor,
        mesh: Any,
    ) -> torch.Tensor:
        """Compute divergence of vector field."""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        face_areas = mesh.face_areas[:n_internal]
        w = mesh.face_weights[:n_internal]

        U_P = U[int_owner]
        U_N = U[int_neigh]
        U_face = 0.5 * (U_P + U_N)

        flux = (U_face * face_areas).sum(dim=1)

        div = torch.zeros(n_cells, dtype=U.dtype, device=U.device)
        div = div + scatter_add(flux, int_owner, n_cells)
        div = div + scatter_add(-flux, int_neigh, n_cells)

        V = mesh.cell_volumes.clamp(min=1e-30)
        return div / V

    def _compute_continuity_error(
        self, phi: torch.Tensor, rho: torch.Tensor
    ) -> float:
        """Compute continuity error for compressible flow."""
        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour

        # Mass flux
        rho_face = 0.5 * (
            gather(rho, owner[:n_internal]) + gather(rho, neighbour)
        )
        mass_flux = phi[:n_internal] * rho_face

        div_rho_phi = torch.zeros(n_cells, dtype=phi.dtype, device=phi.device)
        div_rho_phi = div_rho_phi + scatter_add(
            mass_flux, owner[:n_internal], n_cells
        )
        div_rho_phi = div_rho_phi + scatter_add(
            -mass_flux, neighbour, n_cells
        )

        V = mesh.cell_volumes.clamp(min=1e-30)
        div_rho_phi = div_rho_phi / V

        return float(div_rho_phi.abs().mean().item())

    def _write_fields(self, time: float) -> None:
        """Write U, p, T to a time directory."""
        time_str = f"{time:g}"
        self.write_field("U", self.U, time_str, self._U_data)
        self.write_field("p", self.p, time_str, self._p_data)
        self.write_field("T", self.T, time_str, self._T_data)
