"""
buoyantSimpleFoam — steady-state buoyant compressible solver.

Implements the SIMPLE algorithm for steady-state buoyant, turbulent
flow of compressible fluids including radiation, for ventilation and
heat-transfer applications.

Extends rhoSimpleFoam with:
- Gravity vector g (read from ``constant/g``)
- Hydrostatic pressure decomposition: p = p_rgh + ρg·h
- Buoyancy-driven flow via density differences
- Energy equation with gravity source: ρ(U·g)
- P1 radiation model for thermal radiation

Algorithm (per outer iteration):
1. Solve momentum equation (with buoyancy source ρg)
2. Solve pressure equation (p_rgh form with buoyancy correction)
3. Update density from EOS: ρ = ρ(p, T)
4. Solve energy equation (with radiation and gravity source)
5. Update turbulence model (if active)
6. Check convergence

Usage::

    from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam

    solver = BuoyantSimpleFoam("path/to/case")
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
from pyfoam.models.radiation import P1Radiation, RadiationModel

from .rho_simple_foam import RhoSimpleFoam

__all__ = ["BuoyantSimpleFoam"]

logger = logging.getLogger(__name__)


class BuoyantSimpleFoam(RhoSimpleFoam):
    """Steady-state buoyant compressible SIMPLE solver.

    Solves the steady-state compressible Navier-Stokes equations with
    buoyancy (gravity) effects and optional radiation.  Suitable for
    natural convection, ventilation, and heat-transfer problems.

    The solver decomposes pressure as:
        p = p_rgh + ρg·h

    where p_rgh is the pressure minus the hydrostatic component.  This
    avoids the need to resolve large hydrostatic pressure gradients
    that can cause numerical issues.

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
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        thermo: BasicThermo | None = None,
        gravity: tuple[float, float, float] | None = None,
        radiation: RadiationModel | None = None,
    ) -> None:
        # Initialize parent (reads mesh, thermo, fields)
        super().__init__(case_path, thermo=thermo)

        # Gravity vector
        self.g = self._read_gravity(gravity)
        logger.info("Gravity: %s", self.g.tolist())

        # Compute gh and ghf
        self.gh, self.ghf = self._compute_gh()

        # Pressure decomposition: p_rgh = p - rho*gh
        self.p_rgh = self.p - self.rho * self.gh

        # Radiation model
        self.radiation = radiation or self._init_radiation()
        logger.info("Radiation: %s", self.radiation)

        logger.info("BuoyantSimpleFoam ready: %s", self.thermo)
        if self.turbulence_enabled:
            logger.info("  Turbulence: %s", self.ras)

    # ------------------------------------------------------------------
    # Settings reading
    # ------------------------------------------------------------------

    def _read_gravity(
        self,
        gravity: tuple[float, float, float] | None = None,
    ) -> torch.Tensor:
        """Read gravity vector from case or use provided value.

        Tries to read from ``constant/g`` in OpenFOAM format:
        ``dimensions  [0 1 -2 0 0 0 0]; value  (0 -9.81 0);``

        If the file doesn't exist, uses the provided value or
        defaults to (0, -9.81, 0).
        """
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
                    # Parse "(0 -9.81 0)" format
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

        # Default: (0, -9.81, 0)
        return torch.tensor([0.0, -9.81, 0.0], dtype=dtype, device=device)

    def _init_radiation(self) -> RadiationModel:
        """Initialize radiation model from radiationProperties.

        If the file doesn't exist, uses default P1 with a=0.1.
        """
        rp_path = self.case_path / "constant" / "radiationProperties"
        if rp_path.exists():
            try:
                from pyfoam.io.dictionary import parse_dict_file
                rp = parse_dict_file(rp_path)
                model_name = str(rp.get("radiationModel", "P1")).strip()

                if model_name == "P1":
                    p1_dict = rp.get("P1", {})
                    absorption = 0.1
                    if isinstance(p1_dict, dict):
                        absorption = float(
                            p1_dict.get("absorptionCoeff", 0.1)
                        )
                    return P1Radiation(absorption_coeff=absorption)
                else:
                    logger.warning(
                        "Unknown radiation model '%s', using P1", model_name
                    )
                    return P1Radiation()
            except Exception as e:
                logger.warning(
                    "Could not parse radiationProperties: %s, using P1", e
                )

        return P1Radiation()

    # ------------------------------------------------------------------
    # Gravity dot product
    # ------------------------------------------------------------------

    def _compute_gh(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute gh = g·cell_centre and ghf = g·face_centre.

        These are used for the hydrostatic pressure decomposition:
            p = p_rgh + ρ*g·h
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        # gh = g · cell_centre
        gh = torch.zeros(mesh.n_cells, dtype=dtype, device=device)
        for i in range(3):
            gh = gh + self.g[i] * mesh.cell_centres[:, i]

        # ghf = g · face_centre
        ghf = torch.zeros(mesh.n_faces, dtype=dtype, device=device)
        for i in range(3):
            ghf = ghf + self.g[i] * mesh.face_centres[:, i]

        return gh, ghf

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the buoyantSimpleFoam solver.

        Returns:
            Final :class:`ConvergenceData`.
        """
        from .time_loop import TimeLoop
        from .convergence import ConvergenceMonitor

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

        logger.info("Starting buoyantSimpleFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  relaxation: alpha_U=%.2f, alpha_p=%.2f, alpha_T=%.2f",
                     self.alpha_U, self.alpha_p, self.alpha_T)
        logger.info("  gravity=%s", self.g.tolist())

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            # Update turbulence model (if active)
            mu_eff = self._update_turbulence()

            self.U, self.p, self.p_rgh, self.T, self.phi, self.rho, conv = (
                self._buoyant_simple_iteration(mu_eff=mu_eff)
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
                logger.info("buoyantSimpleFoam completed successfully (converged)")
            else:
                logger.warning("buoyantSimpleFoam completed without full convergence")

        return last_convergence or ConvergenceData()

    # ------------------------------------------------------------------
    # SIMPLE iteration with buoyancy
    # ------------------------------------------------------------------

    def _buoyant_simple_iteration(
        self,
        mu_eff: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ConvergenceData]:
        """Run one SIMPLE outer iteration with buoyancy.

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

        for outer in range(self.max_outer_iterations):
            U_prev = U.clone()
            p_prev = p.clone()
            T_prev = T.clone()

            # ============================================
            # Step 1: Momentum predictor (with buoyancy)
            # ============================================
            U, A_p, H = self._buoyant_momentum_predictor(
                U, p_rgh, phi, rho, mu_eff=mu_eff
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
            # In OpenFOAM: phig = -rhorAUf * ghf * snGrad(rho) * magSf
            # This accounts for the buoyancy-driven pressure correction
            # ============================================
            rho_face = 0.5 * (
                gather(rho, int_owner) + gather(rho, int_neigh)
            )
            A_p_face = w * gather(A_p, int_owner) + (1.0 - w) * gather(A_p, int_neigh)
            rhorAUf = rho_face / A_p_face.abs().clamp(min=1e-30)

            ghf_int = self.ghf[:n_internal]
            # snGrad(rho) on internal faces
            snGrad_rho = (gather(rho, int_neigh) - gather(rho, int_owner)) * mesh.delta_coefficients[:n_internal]
            S_mag = mesh.face_areas[:n_internal].norm(dim=1)
            phig = -rhorAUf * ghf_int * snGrad_rho * S_mag

            # Add buoyancy flux to phiHbyA
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

            # Copy to full phi
            phi = torch.zeros(mesh.n_faces, dtype=dtype, device=device)
            phi[:n_internal] = phi_internal

            # ============================================
            # Step 6: Update pressure from p_rgh
            # p = p_rgh + rho * gh
            # ============================================
            p = p_rgh + rho * self.gh

            # ============================================
            # Step 7: Update density from EOS
            # ============================================
            rho = self.thermo.rho(p, T)

            # ============================================
            # Step 8: Solve energy equation (with radiation)
            # ============================================
            T = self._buoyant_solve_energy_equation(
                T, U, phi, rho, p, T_prev, mu_eff=mu_eff
            )

            # Update density again after T update
            rho = self.thermo.rho(p, T)

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
                    "buoyantSimple iteration %d: U_res=%.6e, p_res=%.6e, "
                    "T_res=%.6e, cont=%.6e",
                    outer, U_residual, p_residual, T_residual, continuity_error,
                )

            if continuity_error < self.convergence_tolerance and outer > 0:
                convergence.converged = True
                break

        return U, p, p_rgh, T, phi, rho, convergence

    # ------------------------------------------------------------------
    # Momentum predictor with buoyancy
    # ------------------------------------------------------------------

    def _buoyant_momentum_predictor(
        self,
        U: torch.Tensor,
        p_rgh: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        mu_eff: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve momentum equation with buoyancy source.

        The momentum equation for buoyant flow is:
            ∇·(ρUU) = -∇p_rgh - ρg·∇h·∇ρ + ∇·(μ∇U)

        where the buoyancy term appears through the p_rgh decomposition.

        In practice, we solve:
            A·U = H - ∇p_rgh + ρg

        The ρg term is the buoyancy driving force.

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
    # Energy equation with radiation and gravity source
    # ------------------------------------------------------------------

    def _buoyant_solve_energy_equation(
        self,
        T: torch.Tensor,
        U: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        p: torch.Tensor,
        T_old: torch.Tensor,
        mu_eff: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Solve the energy equation with buoyancy and radiation.

        For steady-state buoyant flow:
            ∇·(ρUCpT) = ∇·(κ_eff∇T) + p∇·U + Φ + ρ(U·g) + S_rad

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

        source = phi_viscous + p * div_U + buoyancy_work + radiation_source

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
        time_str = f"{time:g}"
        self.write_field("U", self.U, time_str, self._U_data)
        self.write_field("p", self.p, time_str, self._p_data)
        self.write_field("T", self.T, time_str, self._T_data)
