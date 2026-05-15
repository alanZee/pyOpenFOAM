"""
buoyantBoussinesqSimpleFoam — steady-state Boussinesq buoyant solver.

Implements the SIMPLE algorithm for steady-state buoyant, incompressible
flow using the **Boussinesq approximation** for natural convection with
small temperature differences.

The Boussinesq approximation linearises density:
    ρ = ρ₀ [1 − β(T − T₀)]

where:
    ρ₀  = reference density (kg/m³)
    β   = thermal expansion coefficient (1/K)
    T₀  = reference temperature (K)

This approximation is valid when β·ΔT ≪ 1 (small temperature differences).

Key differences from buoyantSimpleFoam:
- **No full EOS**: density is constant ρ₀ except in the buoyancy term
- **Simpler energy equation**: no p∇·U, no viscous dissipation
- **No radiation**: Boussinesq is for small ΔT where radiation is negligible
- **Incompressible continuity**: ∇·U = 0

Algorithm (per outer iteration):
1. Solve momentum equation (with Boussinesq buoyancy source)
2. Solve pressure equation (p_rgh form, constant density)
3. Correct velocity and flux
4. Solve energy equation (convection-diffusion)
5. Update density: ρ = ρ₀[1 − β(T − T₀)]
6. Check convergence

Reference:
    OpenFOAM ``buoyantBoussinesqSimpleFoam``
    de Vahl Davis, G. (1983). "Natural convection of air in a square
    cavity: A bench mark numerical solution." Int. J. Numer. Methods Fluids.

Usage::

    from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam

    solver = BuoyantBoussinesqSimpleFoam("path/to/case")
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

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["BuoyantBoussinesqSimpleFoam"]

logger = logging.getLogger(__name__)


class BuoyantBoussinesqSimpleFoam(SolverBase):
    """Steady-state Boussinesq buoyant SIMPLE solver.

    Solves the steady-state Navier-Stokes equations with the Boussinesq
    approximation for buoyancy-driven flow.  Suitable for natural
    convection problems with small temperature differences (β·ΔT ≪ 1).

    The solver decomposes pressure as:
        p = p_rgh + ρ₀ g·h

    and uses the Boussinesq approximation for the buoyancy source:
        F_buoyancy = ρ g = ρ₀ [1 − β(T − T₀)] g
                    = ρ₀ g − ρ₀ β (T − T₀) g

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    rho_ref : float, optional
        Reference density ρ₀ (kg/m³). Default 1.0.
    beta : float, optional
        Thermal expansion coefficient β (1/K). Default 3.34e-3 (air at 300K).
    T_ref : float, optional
        Reference temperature T₀ (K). Default 300.0.
    gravity : tuple[float, float, float], optional
        Gravity vector (m/s²). If None, reads from ``constant/g``.
    thermo : BasicThermo, optional
        Thermophysical model for viscosity. If None, uses air defaults.

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
    rho_ref : float
        Reference density (kg/m³).
    beta : float
        Thermal expansion coefficient (1/K).
    T_ref : float
        Reference temperature (K).
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        rho_ref: float = 1.0,
        beta: float = 3.34e-3,
        T_ref: float = 300.0,
        gravity: tuple[float, float, float] | None = None,
        thermo: BasicThermo | None = None,
    ) -> None:
        super().__init__(case_path)

        # Thermophysical model (for viscosity only)
        self.thermo = thermo or create_air_thermo()

        # Boussinesq parameters
        self.rho_ref = rho_ref
        self.beta = beta
        self.T_ref = T_ref
        logger.info(
            "Boussinesq: rho_ref=%.4f, beta=%.6e, T_ref=%.2f",
            self.rho_ref, self.beta, self.T_ref,
        )

        # Read settings
        self._read_fv_solution_settings()
        self._read_fv_schemes_settings()

        # Gravity vector
        self.g = self._read_gravity(gravity)
        logger.info("Gravity: %s", self.g.tolist())

        # Compute gh and ghf for pressure decomposition
        self.gh, self.ghf = self._compute_gh()

        # Initialise fields
        self.U, self.p, self.T, self.phi = self._init_fields()
        self._U_data, self._p_data, self._T_data = self._init_field_data()

        # Initialise density from Boussinesq approximation
        self.rho = self._boussinesq_rho(self.T)

        # Pressure decomposition: p_rgh = p - rho_ref * gh
        self.p_rgh = self.p - self.rho_ref * self.gh

        logger.info("BuoyantBoussinesqSimpleFoam ready: %s", self.thermo)

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

    def _init_fields(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialise U, p, T, phi from the 0/ directory."""
        device = get_device()
        dtype = get_default_dtype()

        U_tensor, _ = self.read_field_tensor("U", 0)
        U = U_tensor.to(device=device, dtype=dtype)

        p_tensor, _ = self.read_field_tensor("p", 0)
        p = p_tensor.to(device=device, dtype=dtype)

        T_tensor, _ = self.read_field_tensor("T", 0)
        T = T_tensor.to(device=device, dtype=dtype)

        phi = torch.zeros(self.mesh.n_faces, dtype=dtype, device=device)

        return U, p, T, phi

    def _init_field_data(self):
        """Store raw FieldData for writing."""
        U_data = self.case.read_field("U", 0)
        p_data = self.case.read_field("p", 0)
        T_data = self.case.read_field("T", 0)
        return U_data, p_data, T_data

    # ------------------------------------------------------------------
    # Boussinesq density
    # ------------------------------------------------------------------

    def _boussinesq_rho(self, T: torch.Tensor) -> torch.Tensor:
        """Compute density using the Boussinesq approximation.

            ρ = ρ₀ [1 − β(T − T₀)]

        Parameters
        ----------
        T : torch.Tensor
            ``(n_cells,)`` temperature field (K).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` density field (kg/m³).
        """
        return self.rho_ref * (1.0 - self.beta * (T - self.T_ref))

    # ------------------------------------------------------------------
    # Gravity reading and computation
    # ------------------------------------------------------------------

    def _read_gravity(
        self,
        gravity: tuple[float, float, float] | None = None,
    ) -> torch.Tensor:
        """Read gravity vector from case or use provided value."""
        device = get_device()
        dtype = get_default_dtype()

        if gravity is not None:
            return torch.tensor(gravity, dtype=dtype, device=device)

        g_path = self.case_path / "constant" / "g"
        if g_path.exists():
            try:
                from pyfoam.io.dictionary import parse_dict_file
                g_dict = parse_dict_file(g_path)
                value = g_dict.get("value", "(0 -9.81 0)")
                if isinstance(value, str):
                    value = value.strip("()").split()
                    g = [float(v) for v in value]
                elif isinstance(value, (list, tuple)):
                    g = [float(v) for v in value]
                else:
                    g = [0.0, -9.81, 0.0]
                return torch.tensor(g, dtype=dtype, device=device)
            except Exception as e:
                logger.warning(
                    "Could not parse constant/g: %s, using default", e
                )

        return torch.tensor([0.0, -9.81, 0.0], dtype=dtype, device=device)

    def _compute_gh(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute gh = g·cell_centre and ghf = g·face_centre."""
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        gh = torch.zeros(mesh.n_cells, dtype=dtype, device=device)
        for i in range(3):
            gh = gh + self.g[i] * mesh.cell_centres[:, i]

        ghf = torch.zeros(mesh.n_faces, dtype=dtype, device=device)
        for i in range(3):
            ghf = ghf + self.g[i] * mesh.face_centres[:, i]

        return gh, ghf

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the buoyantBoussinesqSimpleFoam solver.

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

        logger.info("Starting buoyantBoussinesqSimpleFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  relaxation: alpha_U=%.2f, alpha_p=%.2f, alpha_T=%.2f",
                     self.alpha_U, self.alpha_p, self.alpha_T)
        logger.info("  gravity=%s", self.g.tolist())

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            self.U, self.p, self.p_rgh, self.T, self.phi, self.rho, conv = (
                self._boussinesq_simple_iteration()
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
                logger.info("buoyantBoussinesqSimpleFoam completed (converged)")
            else:
                logger.warning("buoyantBoussinesqSimpleFoam completed without full convergence")

        return last_convergence or ConvergenceData()

    # ------------------------------------------------------------------
    # SIMPLE iteration with Boussinesq buoyancy
    # ------------------------------------------------------------------

    def _boussinesq_simple_iteration(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ConvergenceData]:
        """Run one SIMPLE outer iteration with Boussinesq buoyancy.

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

        for outer in range(self.max_outer_iterations):
            U_prev = U.clone()
            p_prev = p.clone()
            T_prev = T.clone()

            # ============================================
            # Step 1: Momentum predictor (Boussinesq)
            # ============================================
            U, A_p, H = self._boussinesq_momentum_predictor(
                U, p_rgh, phi, rho, T,
            )

            # ============================================
            # Step 2: Compute HbyA and face flux
            # ============================================
            HbyA = H / A_p.abs().clamp(min=1e-30).unsqueeze(-1)

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
            # Step 3: Buoyancy flux correction
            # phig = -rhorAUf * ghf * snGrad(rho) * magSf
            # For Boussinesq: rho ≈ rho_ref (constant)
            # so snGrad(rho) is small but not zero
            # ============================================
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

            # ============================================
            # Step 4: Pressure equation (p_rgh form)
            # ============================================
            p_rgh = self._solve_pressure_equation(
                p_rgh, phiHbyA, A_p, rho, mesh
            )

            # Under-relax pressure
            if self.alpha_p < 1.0:
                p_rgh = self.alpha_p * p_rgh + (1.0 - self.alpha_p) * (self.p_rgh)

            # ============================================
            # Step 5: Correct velocity and flux
            # ============================================
            grad_p_rgh = self._compute_grad(p_rgh, mesh)
            U = HbyA - grad_p_rgh / A_p.abs().clamp(min=1e-30).unsqueeze(-1)

            # Correct internal face flux
            phi_internal = phiHbyA.clone()
            p_P = gather(p_rgh, int_owner)
            p_N = gather(p_rgh, int_neigh)
            A_p_face2 = w * gather(A_p, int_owner) + (1.0 - w) * gather(A_p, int_neigh)
            phi_internal = phi_internal - (p_N - p_P) / A_p_face2.clamp(min=1e-30)

            phi = torch.zeros(mesh.n_faces, dtype=dtype, device=device)
            phi[:n_internal] = phi_internal

            # ============================================
            # Step 6: Update pressure from p_rgh
            # ============================================
            p = p_rgh + rho * self.gh

            # ============================================
            # Step 7: Solve energy equation
            # ============================================
            T = self._boussinesq_solve_energy_equation(
                T, U, phi, T_prev,
            )

            # ============================================
            # Step 8: Update density from Boussinesq
            # ============================================
            rho = self._boussinesq_rho(T)

            # Update p_rgh for consistency
            p_rgh = p - rho * self.gh

            # ============================================
            # Step 9: Check convergence
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
                    "Boussinesq iteration %d: U_res=%.6e, p_res=%.6e, "
                    "T_res=%.6e, cont=%.6e",
                    outer, U_residual, p_residual, T_residual, continuity_error,
                )

            if continuity_error < self.convergence_tolerance and outer > 0:
                convergence.converged = True
                break

        return U, p, p_rgh, T, phi, rho, convergence

    # ------------------------------------------------------------------
    # Momentum predictor with Boussinesq buoyancy
    # ------------------------------------------------------------------

    def _boussinesq_momentum_predictor(
        self,
        U: torch.Tensor,
        p_rgh: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        T: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve momentum equation with Boussinesq buoyancy source.

        The buoyancy force is:
            F = ρ g = ρ₀ [1 − β(T − T₀)] g
              = ρ₀ g − ρ₀ β (T − T₀) g

        The first term (ρ₀ g) is constant and absorbed into the
        hydrostatic pressure decomposition.  The remaining term
        −ρ₀ β (T − T₀) g is the thermal buoyancy driving force.

        Parameters
        ----------
        U : torch.Tensor
            ``(n_cells, 3)`` velocity field.
        p_rgh : torch.Tensor
            ``(n_cells,)`` pressure minus hydrostatic component.
        phi : torch.Tensor
            ``(n_faces,)`` face flux.
        rho : torch.Tensor
            ``(n_cells,)`` density field.
        T : torch.Tensor
            ``(n_cells,)`` temperature field.
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

        # Viscosity
        mu = self.thermo.mu(T=self.T)
        mu_face = 0.5 * (gather(mu, int_owner) + gather(mu, int_neigh))

        # Diffusion coefficient
        S_mag = face_areas[:n_internal].norm(dim=1)
        delta_f = delta_coeffs[:n_internal]
        diff_coeff = mu_face * S_mag * delta_f

        # Convection (upwind) with constant density rho_ref
        flux = phi[:n_internal]
        rho_face = torch.full_like(flux, self.rho_ref)

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

        # Pressure gradient (p_rgh form)
        w = mesh.face_weights[:n_internal]
        p_P = gather(p_rgh, int_owner)
        p_N = gather(p_rgh, int_neigh)
        p_face = w * p_P + (1.0 - w) * p_N
        p_contrib = p_face.unsqueeze(-1) * face_areas[:n_internal]

        grad_p = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad_p.index_add_(0, int_owner, p_contrib)
        grad_p.index_add_(0, int_neigh, -p_contrib)

        # Boussinesq buoyancy source: -ρ₀ β (T − T₀) g
        # This is the thermal buoyancy driving force
        rho_g = self.rho_ref * (1.0 - self.beta * (T - self.T_ref))
        rho_g = rho_g.unsqueeze(-1) * self.g.unsqueeze(0)

        source = H - grad_p + rho_g

        # Solve: U = source / A_p
        A_p_safe = A_p.abs().clamp(min=1e-30)
        U_solved = source / A_p_safe.unsqueeze(-1)

        # Under-relaxation
        U_new = self.alpha_U * U_solved + (1.0 - self.alpha_U) * U

        return U_new, A_p, H

    # ------------------------------------------------------------------
    # Energy equation (Boussinesq)
    # ------------------------------------------------------------------

    def _boussinesq_solve_energy_equation(
        self,
        T: torch.Tensor,
        U: torch.Tensor,
        phi: torch.Tensor,
        T_old: torch.Tensor,
    ) -> torch.Tensor:
        """Solve the energy equation for Boussinesq flow.

        For steady-state Boussinesq flow:
            ∇·(ρ₀ U Cp T) = ∇·(κ ∇T)

        where:
            ρ₀ = reference density (constant)
            Cp = specific heat (constant)
            κ  = thermal conductivity = μ / Pr

        No viscous dissipation or p∇·U terms (small ΔT assumption).

        Parameters
        ----------
        T : torch.Tensor
            ``(n_cells,)`` temperature field.
        U : torch.Tensor
            ``(n_cells, 3)`` velocity field.
        phi : torch.Tensor
            ``(n_faces,)`` face flux.
        T_old : torch.Tensor
            Temperature from previous outer iteration (for under-relaxation).
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes

        # Thermal conductivity: κ = μ / Pr
        mu = self.thermo.mu(T)
        kappa = mu / self.thermo.Pr
        kappa_face = 0.5 * (gather(kappa, int_owner) + gather(kappa, int_neigh))

        # Diffusion coefficients
        face_areas = mesh.face_areas[:n_internal]
        S_mag = face_areas.norm(dim=1)
        delta_f = mesh.delta_coefficients[:n_internal]
        diff_coeff = kappa_face * S_mag * delta_f

        # Convection (upwind) with constant density
        flux = phi[:n_internal]
        rho_face = torch.full_like(flux, self.rho_ref)

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

        # Source: no viscous dissipation or p·div(U) for Boussinesq
        source = torch.zeros(n_cells, dtype=dtype, device=device)

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

        # Under-relax
        if self.alpha_T < 1.0:
            T = self.alpha_T * T + (1.0 - self.alpha_T) * T_old

        return T

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
        """Solve the pressure equation (constant density form).

        For Boussinesq flow, density is approximately constant:
            ∇·(HbyA) = ∇·(1/A_p ∇p)
        """
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes
        face_areas = mesh.face_areas
        delta_coeffs = mesh.delta_coefficients

        w = mesh.face_weights[:n_internal]

        # Face density (use rho_ref for Boussinesq)
        rho_face = torch.full(
            (n_internal,), self.rho_ref, dtype=p.dtype, device=p.device,
        )

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

        # Source: mass flux imbalance
        mass_flux_HbyA = phiHbyA * rho_face
        source = torch.zeros(n_cells, dtype=p.dtype, device=p.device)
        source = source + scatter_add(mass_flux_HbyA, int_owner, n_cells)
        source = source + scatter_add(-mass_flux_HbyA, int_neigh, n_cells)

        # Jacobi iteration
        diag_safe = diag.abs().clamp(min=1e-30)
        for _ in range(self.p_max_iter):
            off_diag = torch.zeros(n_cells, dtype=p.dtype, device=p.device)
            p_P = gather(p, int_owner)
            p_N = gather(p, int_neigh)
            off_diag = off_diag + scatter_add(lower * p_N, int_owner, n_cells)
            off_diag = off_diag + scatter_add(upper * p_P, int_neigh, n_cells)

            p_new = (source - off_diag) / diag_safe

            if (p_new - p).abs().max() < self.p_tolerance:
                break
            p = p_new

        return p

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
        """Compute divergence of vector field using face flux."""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

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
        """Compute continuity error.

        For Boussinesq flow, the continuity error is the L1 norm of ∇·U
        normalised by cell volume.
        """
        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour

        # Use volumetric flux directly (incompressible)
        div_phi = torch.zeros(n_cells, dtype=phi.dtype, device=phi.device)
        div_phi = div_phi + scatter_add(
            phi[:n_internal], owner[:n_internal], n_cells
        )
        div_phi = div_phi + scatter_add(
            -phi[:n_internal], neighbour, n_cells
        )

        V = mesh.cell_volumes.clamp(min=1e-30)
        div_phi = div_phi / V

        return float(div_phi.abs().mean().item())

    # ------------------------------------------------------------------
    # Field writing
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write U, p, p_rgh, T to a time directory."""
        time_str = f"{time:g}"
        self.write_field("U", self.U, time_str, self._U_data)
        self.write_field("p", self.p, time_str, self._p_data)
        self.write_field("T", self.T, time_str, self._T_data)
