"""
rhoSimpleFoam — steady-state compressible solver.

Implements the SIMPLE algorithm for steady-state compressible
Navier-Stokes equations with energy equation coupling.

Extends the incompressible SIMPLE solver with:
- Density from equation of state: ρ = ρ(p, T)
- Energy equation: ∇·(ρUCpT) = ∇·(κ∇T) + p∇·U + Φ
- Viscous dissipation: Φ = 2μ(S:S) - (2/3)μ(∇·U)²
- Variable viscosity from transport model
- Compressible turbulence coupling (ρk-ε)

Algorithm (per outer iteration):
1. Solve momentum equation (with variable density)
2. Solve pressure equation (compressible form with ρ-weighted Laplacian)
3. Update density from EOS: ρ = ρ(p, T)
4. Solve energy equation
5. Update turbulence model (if active)
6. Update viscosity from transport model
7. Check convergence

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
    with energy equation coupling and optional RANS turbulence modelling.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    thermo : BasicThermo, optional
        Thermophysical model. If None, uses air defaults.

    Attributes
    ----------
    U : torch.Tensor
        ``(n_cells, 3)`` velocity field.
    p : torch.Tensor
        ``(n_cells,)`` pressure field (Pa).
    T : torch.Tensor
        ``(n_cells,)`` temperature field (K).
    phi : torch.Tensor
        ``(n_faces,)`` face volumetric flux field.
    rho : torch.Tensor
        ``(n_cells,)`` density field (kg/m³).
    thermo : BasicThermo
        Thermophysical model.
    turbulence_enabled : bool
        Whether RANS turbulence modelling is active.
    ras : RASModel or None
        The RAS turbulence model wrapper (None if disabled).
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
        self._read_fv_schemes_settings()

        # Initialise fields
        self.U, self.p, self.T, self.phi, self.rho = self._init_fields()
        self._U_data, self._p_data, self._T_data = self._init_field_data()

        # Turbulence model (optional)
        self.ras, self.turbulence_enabled = self._init_turbulence()

        logger.info("RhoSimpleFoam ready: %s", self.thermo)
        if self.turbulence_enabled:
            logger.info("  Turbulence: %s", self.ras)

    # ------------------------------------------------------------------
    # Settings reading
    # ------------------------------------------------------------------

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

    def _read_fv_schemes_settings(self) -> None:
        """Read fvSchemes settings (for logging)."""
        fs = self.case.fvSchemes
        self.grad_scheme = str(fs.get_path("gradSchemes/default", "Gauss linear"))
        self.div_scheme = str(fs.get_path("divSchemes/default", "Gauss linear"))
        self.lap_scheme = str(fs.get_path("laplacianSchemes/default", "Gauss linear corrected"))

    # ------------------------------------------------------------------
    # Field initialisation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Turbulence model
    # ------------------------------------------------------------------

    def _init_turbulence(self) -> tuple[Any, bool]:
        """Initialise RAS turbulence model from turbulenceProperties.

        Reads ``constant/turbulenceProperties`` to determine:
        - ``simulationType``: ``laminar`` (default) or ``RAS``
        - ``RAS/model``: model name (e.g. ``kEpsilon``, ``kOmegaSST``)

        Returns:
            Tuple of ``(RASModel | None, enabled)``.
        """
        tp_path = self.case_path / "constant" / "turbulenceProperties"
        if not tp_path.exists():
            return None, False

        try:
            from pyfoam.io.dictionary import parse_dict_file
            from pyfoam.turbulence.ras_model import RASModel, RASConfig

            tp = parse_dict_file(tp_path)
            sim_type = str(tp.get("simulationType", "laminar")).strip()

            if sim_type != "RAS":
                return None, False

            # Read RAS settings
            ras_dict = tp.get("RAS", {})
            if isinstance(ras_dict, dict):
                model_name = str(ras_dict.get("model", "kEpsilon")).strip()
                ras_enabled = str(ras_dict.get("enabled", "true")).strip().lower() == "true"
            else:
                model_name = "kEpsilon"
                ras_enabled = True

            if not ras_enabled:
                return None, False

            # Compute kinematic viscosity at reference conditions for turbulence model
            nu_ref = float(self.thermo.nu(self.T, self.rho).mean().item())

            config = RASConfig(
                model_name=model_name,
                enabled=True,
                nu=nu_ref,
            )
            ras = RASModel(self.mesh, self.U, self.phi, config)
            logger.info("Turbulence model: %s", model_name)
            return ras, True

        except Exception as e:
            logger.warning("Could not initialise turbulence model: %s", e)
            return None, False

    def _update_turbulence(self) -> torch.Tensor | None:
        """Update the turbulence model and return the effective viscosity.

        For compressible flow, the turbulence model operates on
        kinematic quantities. The effective dynamic viscosity is
        returned: μ_eff = μ + μ_t = ρ(ν + ν_t).

        Returns:
            ``(n_cells,)`` effective dynamic viscosity, or ``None``.
        """
        if not self.turbulence_enabled or self.ras is None:
            return None

        # Update velocity and flux references in the turbulence model
        self.ras._model._U = self.U
        self.ras._model._phi = self.phi

        # Update kinematic viscosity reference (mean over cells)
        nu_mean = float(self.thermo.nu(self.T, self.rho).mean().item())
        self.ras._model._nu = nu_mean

        # Correct turbulence (solve k and ε transport equations)
        self.ras.correct()

        # Return effective dynamic viscosity: μ_eff = ρ * (ν + ν_t)
        nu_eff = self.ras.mu_eff()  # kinematic: ν + ν_t
        mu_eff = self.rho * nu_eff  # dynamic: μ_eff = ρ * ν_eff
        return mu_eff

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

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
        logger.info("  relaxation: alpha_U=%.2f, alpha_p=%.2f, alpha_T=%.2f",
                     self.alpha_U, self.alpha_p, self.alpha_T)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            # Update turbulence model (if active) before the SIMPLE iteration
            mu_eff = self._update_turbulence()

            self.U, self.p, self.T, self.phi, self.rho, conv = (
                self._simple_iteration(mu_eff=mu_eff)
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
                logger.info("rhoSimpleFoam completed successfully (converged)")
            else:
                logger.warning("rhoSimpleFoam completed without full convergence")

        return last_convergence or ConvergenceData()

    # ------------------------------------------------------------------
    # SIMPLE iteration
    # ------------------------------------------------------------------

    def _simple_iteration(
        self,
        mu_eff: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ConvergenceData]:
        """Run one SIMPLE outer iteration for compressible flow.

        Parameters
        ----------
        mu_eff : torch.Tensor, optional
            Effective dynamic viscosity field (molecular + turbulent).
            If None, uses molecular viscosity only.

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
            T_prev = T.clone()

            # ============================================
            # Step 1: Momentum predictor
            # ============================================
            U, A_p, H = self._momentum_predictor(U, p, phi, rho, mu_eff=mu_eff)

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

            # Correct internal face flux
            phi_internal = phiHbyA.clone()
            p_P = gather(p, int_owner)
            p_N = gather(p, int_neigh)
            A_p_face = w * gather(A_p, int_owner) + (1.0 - w) * gather(A_p, int_neigh)
            phi_internal = phi_internal - (p_N - p_P) / A_p_face.clamp(min=1e-30)

            # Copy to full phi (boundary faces get zero flux for now)
            phi = torch.zeros(mesh.n_faces, dtype=dtype, device=device)
            phi[:n_internal] = phi_internal

            # ============================================
            # Step 5: Update density from EOS
            # ============================================
            rho = self.thermo.rho(p, T)

            # ============================================
            # Step 6: Solve energy equation
            # ============================================
            T = self._solve_energy_equation(T, U, phi, rho, p, T_prev, mu_eff=mu_eff)

            # Update density again after T update
            rho = self.thermo.rho(p, T)

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

            if outer % 10 == 0 or outer < 5:
                logger.info(
                    "rhoSimple iteration %d: U_res=%.6e, p_res=%.6e, "
                    "T_res=%.6e, cont=%.6e",
                    outer, U_residual, p_residual, T_residual, continuity_error,
                )

            if continuity_error < self.convergence_tolerance and outer > 0:
                convergence.converged = True
                break

        return U, p, T, phi, rho, convergence

    # ------------------------------------------------------------------
    # Momentum equation
    # ------------------------------------------------------------------

    def _momentum_predictor(
        self,
        U: torch.Tensor,
        p: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        mu_eff: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve momentum equation with variable density.

        Parameters
        ----------
        mu_eff : torch.Tensor, optional
            Effective dynamic viscosity (molecular + turbulent).
            If None, uses molecular viscosity from the thermo model.
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

        # Viscosity: molecular or effective (molecular + turbulent)
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

    # ------------------------------------------------------------------
    # Pressure equation
    # ------------------------------------------------------------------

    def _solve_pressure_equation(
        self,
        p: torch.Tensor,
        phiHbyA: torch.Tensor,
        A_p: torch.Tensor,
        rho: torch.Tensor,
        mesh: Any,
    ) -> torch.Tensor:
        """Solve the compressible pressure equation.

        The compressible pressure equation ensures mass conservation:
            ∇·(ρ HbyA) = ∇·(ρ/A_p ∇p)

        The face coefficient includes the density-weighted 1/A_p term:
            face_coeff = ρ_f / (A_p)_f * |S_f| * δ_f
        """
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes
        face_areas = mesh.face_areas
        delta_coeffs = mesh.delta_coefficients

        w = mesh.face_weights[:n_internal]

        # Face density (harmonic mean for better accuracy)
        rho_P = gather(rho, int_owner)
        rho_N = gather(rho, int_neigh)
        rho_face = 2.0 * rho_P * rho_N / (rho_P + rho_N).clamp(min=1e-30)

        # 1/A_p on faces
        A_p_inv = 1.0 / A_p.abs().clamp(min=1e-30)
        A_p_inv_face = w * gather(A_p_inv, int_owner) + (1.0 - w) * gather(A_p_inv, int_neigh)

        # Laplacian coefficient: ρ_f * (1/A_p)_f * |S_f| * δ_f
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

        # Source: mass flux imbalance (divergence of ρ*HbyA)
        # Note: phiHbyA is volumetric flux, multiply by face density for mass flux
        mass_flux_HbyA = phiHbyA * rho_face
        source = torch.zeros(n_cells, dtype=p.dtype, device=p.device)
        source = source + scatter_add(mass_flux_HbyA, int_owner, n_cells)
        source = source + scatter_add(-mass_flux_HbyA, int_neigh, n_cells)

        # Solve using Jacobi iteration
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

    # ------------------------------------------------------------------
    # Energy equation
    # ------------------------------------------------------------------

    def _solve_energy_equation(
        self,
        T: torch.Tensor,
        U: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        p: torch.Tensor,
        T_old: torch.Tensor,
        mu_eff: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Solve the energy equation.

        For steady-state compressible flow:
            ∇·(ρUCpT) = ∇·(κ_eff∇T) + p∇·U + Φ

        where:
            κ_eff = κ_mol + κ_turb = μ/Pr + μ_t/Prt  (effective thermal conductivity)
            Φ = 2μ(S:S) - (2/3)μ(∇·U)²              (viscous dissipation)

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

        # Source 1: Viscous dissipation Φ = 2μ(S:S) - (2/3)μ(∇·U)²
        mu = mu_eff if mu_eff is not None else mu_mol
        grad_U = self._compute_grad_vector(U, mesh)
        # Strain rate tensor: S = 0.5(∇U + ∇U^T)
        S = 0.5 * (grad_U + grad_U.transpose(-1, -2))
        S_double_dot = (S * S).sum(dim=(1, 2))
        div_U = self._compute_div(U, phi, mesh)
        phi_viscous = 2.0 * mu * S_double_dot - (2.0 / 3.0) * mu * div_U**2

        # Source 2: p * div(U)
        source = phi_viscous + p * div_U

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
    # Gradient and divergence helpers
    # ------------------------------------------------------------------

    def _compute_grad(self, phi: torch.Tensor, mesh: Any) -> torch.Tensor:
        """Compute gradient of scalar field using Gauss theorem."""
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
        """Compute gradient of vector field (returns 3x3 tensor per cell)."""
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
        """Compute divergence of vector field using face flux.

        Uses the face-normal component of velocity (phi) for consistency
        with the discretised continuity equation.
        """
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        # Use the face flux directly (already U·S_f)
        flux = phi[:n_internal]

        div = torch.zeros(n_cells, dtype=U.dtype, device=U.device)
        div = div + scatter_add(flux, int_owner, n_cells)
        div = div + scatter_add(-flux, int_neigh, n_cells)

        V = mesh.cell_volumes.clamp(min=1e-30)
        return div / V

    # ------------------------------------------------------------------
    # Residual computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_residual(
        field: torch.Tensor,
        field_old: torch.Tensor,
    ) -> float:
        """Compute the L2 norm of the field change, normalised by magnitude."""
        diff = field - field_old
        norm_diff = float(torch.norm(diff).item())
        norm_field = float(torch.norm(field).item())
        if norm_field > 1e-30:
            return norm_diff / norm_field
        return norm_diff

    # ------------------------------------------------------------------
    # Continuity error
    # ------------------------------------------------------------------

    def _compute_continuity_error(
        self, phi: torch.Tensor, rho: torch.Tensor
    ) -> float:
        """Compute continuity error for compressible flow.

        The continuity error is the L1 norm of ∇·(ρφ) normalised by
        cell volume: Σ|∇·(ρU)| / n_cells.
        """
        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour

        # Mass flux: ρ_f * φ_f
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

    # ------------------------------------------------------------------
    # Field writing
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write U, p, T to a time directory."""
        time_str = f"{time:g}"
        self.write_field("U", self.U, time_str, self._U_data)
        self.write_field("p", self.p, time_str, self._p_data)
        self.write_field("T", self.T, time_str, self._T_data)
