"""
sprayFoam — Lagrangian spray solver with two-way Euler-Lagrange coupling.

Combines an Eulerian gas-phase solver (compressible PIMPLE) with a
Lagrangian cloud of liquid droplets.  The two phases exchange momentum,
heat and mass via volumetric source terms:

- **Momentum**: drag force on droplets appears as a sink in the gas
  momentum equation; the equal-and-opposite reaction accelerates the gas.
- **Heat**: convective heat transfer between gas and droplets.
- **Mass**: evaporation adds vapour mass to the gas continuity equation
  and reduces droplet mass.

The gas phase is solved with the compressible PIMPLE algorithm, and the
droplet cloud is advanced using the existing Lagrangian infrastructure
(``KinematicCloud``, injection, breakup, collision, evaporation models).

Usage::

    from pyfoam.applications.spray_foam import SprayFoam

    solver = SprayFoam("path/to/case", injector=injector)
    solver.run()
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Optional, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData
from pyfoam.lagrangian.cloud import KinematicCloud
from pyfoam.lagrangian.particle import Particle

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["SprayFoam"]

logger = logging.getLogger(__name__)


# ======================================================================
# Lagrangian-Eulerian coupling helper
# ======================================================================


class LagrangianCoupling:
    """Manages two-way coupling between a Lagrangian cloud and the
    Eulerian gas phase.

    Computes volumetric source terms for momentum, energy and mass that
    arise from the Lagrangian particles within each cell.

    Parameters
    ----------
    mesh : Any
        The finite-volume mesh.
    cloud : KinematicCloud
        The Lagrangian droplet cloud.
    Cp_fuel : float
        Specific heat capacity of the liquid fuel (J/(kg K)).
    L_vap : float
        Latent heat of vaporisation (J/kg).
    """

    def __init__(
        self,
        mesh: Any,
        cloud: KinematicCloud,
        Cp_fuel: float = 2000.0,
        L_vap: float = 2.5e6,
    ) -> None:
        self.mesh = mesh
        self.cloud = cloud
        self.Cp_fuel = Cp_fuel
        self.L_vap = L_vap

    def momentum_source(self) -> torch.Tensor:
        """Compute volumetric momentum source from drag on particles.

        For each alive particle in a cell *i*, the drag contribution is:

            S_mom,i = - sum_p  (m_p * (v_p - v_g) / tau_p)  / V_cell

        where tau_p is the particle relaxation time.

        Returns
        -------
        torch.Tensor
            ``(n_cells, 3)`` momentum source term.
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells
        cell_vol = self.mesh.cell_volumes.to(device=device, dtype=dtype)
        source = torch.zeros(n_cells, 3, dtype=dtype, device=device)

        for p in self.cloud.particles:
            if not p.alive:
                continue
            ci = self._locate_cell(p.position)
            if ci < 0 or ci >= n_cells:
                continue

            # Particle relaxation time: tau_p = rho_p * d^2 / (18 * mu)
            d = max(p.diameter, 1e-10)
            mu = max(self.cloud.fluid_viscosity, 1e-30)
            tau_p = p.density * d * d / (18.0 * mu)

            # Drag source (reaction force on gas)
            m_p = p.mass
            v_p = torch.tensor(p.velocity, dtype=dtype, device=device)
            v_g = torch.tensor(self.cloud.fluid_velocity, dtype=dtype, device=device)
            drag = m_p * (v_p - v_g) / max(tau_p, 1e-30)

            vol = max(float(cell_vol[ci].item()), 1e-30)
            source[ci] += drag / vol

        return source

    def heat_source(self, T_gas: torch.Tensor) -> torch.Tensor:
        """Compute volumetric heat source from convective particle heating.

        Q_i = sum_p  h * A_p * (T_gas - T_p)  / V_cell

        Parameters
        ----------
        T_gas : torch.Tensor
            ``(n_cells,)`` gas temperature field.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` heat source term.
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells
        cell_vol = self.mesh.cell_volumes.to(device=device, dtype=dtype)
        source = torch.zeros(n_cells, dtype=dtype, device=device)

        for p in self.cloud.particles:
            if not p.alive:
                continue
            ci = self._locate_cell(p.position)
            if ci < 0 or ci >= n_cells:
                continue

            d = max(p.diameter, 1e-10)
            A_p = math.pi * d * d  # surface area of one droplet

            # Convective heat transfer coefficient (Ranz-Marshall)
            Re_p = self._particle_reynolds(p)
            Pr = 0.7  # Prandtl number for air
            k_air = 0.026  # thermal conductivity of air (W/m K)
            Nu = 2.0 + 0.6 * math.sqrt(max(Re_p, 0.0)) * (Pr ** (1.0 / 3.0))
            h = Nu * k_air / d

            T_p = getattr(p, "temperature", 300.0)
            dT = float(T_gas[ci].item()) - T_p

            vol = max(float(cell_vol[ci].item()), 1e-30)
            source[ci] += h * A_p * dT / vol

        return source

    def mass_source(self, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute volumetric mass source from droplet evaporation.

        Uses a simple d^2-law evaporation model:

            dm/dt = - pi * rho_fuel * D_vap * Sh * d * (Y_s - Y_inf)

        Parameters
        ----------
        dt : float
            Time step (s).

        Returns
        -------
        source : torch.Tensor
            ``(n_cells,)`` mass source (positive = mass added to gas).
        evaporation_rate : torch.Tensor
            ``(n_cells,)`` total mass evaporated per cell this step.
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells
        cell_vol = self.mesh.cell_volumes.to(device=device, dtype=dtype)
        source = torch.zeros(n_cells, dtype=dtype, device=device)

        D_vap = 2.0e-5  # mass diffusivity (m^2/s)
        Y_s = 0.03  # vapour mass fraction at droplet surface
        Y_inf = 0.0  # far-field vapour mass fraction

        for p in self.cloud.particles:
            if not p.alive:
                continue
            ci = self._locate_cell(p.position)
            if ci < 0 or ci >= n_cells:
                continue

            d = max(p.diameter, 1e-10)
            Re_p = self._particle_reynolds(p)
            Sc = 0.7  # Schmidt number
            Sh = 2.0 + 0.6 * math.sqrt(max(Re_p, 0.0)) * (Sc ** (1.0 / 3.0))

            dm_dt = -math.pi * p.density * D_vap * Sh * d * (Y_s - Y_inf)

            vol = max(float(cell_vol[ci].item()), 1e-30)
            source[ci] += abs(dm_dt) / vol  # mass added to gas phase

            # Reduce particle mass
            p.mass = max(p.mass + dm_dt * dt, 0.0)
            if p.mass <= 0.0:
                p.alive = False

        return source, source * dt  # rate and total evaporated mass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _locate_cell(self, position: list[float]) -> int:
        """Locate cell index for a particle position.

        Uses a simple bounding-box check on cell centres.
        Returns -1 if not found.
        """
        centres = self.mesh.cell_centres
        dists = torch.sum(
            (centres - torch.tensor(position, dtype=centres.dtype, device=centres.device)) ** 2,
            dim=1,
        )
        return int(torch.argmin(dists).item())

    def _particle_reynolds(self, p: Particle) -> float:
        """Compute particle Reynolds number."""
        d = max(p.diameter, 1e-10)
        nu = max(self.cloud.fluid_viscosity / max(self.cloud.fluid_density, 1e-30), 1e-30)
        v_rel = sum(
            (p.velocity[i] - self.cloud.fluid_velocity[i]) ** 2 for i in range(3)
        ) ** 0.5
        return d * v_rel / nu


# ======================================================================
# Main solver
# ======================================================================


class SprayFoam(SolverBase):
    """Lagrangian spray solver with two-way Euler-Lagrange coupling.

    Solves the compressible Navier-Stokes equations for the gas phase
    (PIMPLE algorithm) while tracking a Lagrangian cloud of liquid
    droplets.  Source terms from drag, heat transfer and evaporation
    provide two-way coupling between the phases.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    cloud : KinematicCloud, optional
        Pre-configured Lagrangian droplet cloud.  If *None*, an empty
        cloud is created.
    rho_gas : float
        Gas-phase density (kg/m3, default 1.225).
    mu_gas : float
        Gas-phase dynamic viscosity (Pa s, default 1.8e-5).
    Cp_gas : float
        Gas specific heat capacity (J/(kg K), default 1005.0).
    Cp_fuel : float
        Liquid fuel specific heat capacity (J/(kg K), default 2000.0).
    L_vap : float
        Latent heat of vaporisation (J/kg, default 2.5e6).
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        cloud: Optional[KinematicCloud] = None,
        rho_gas: float = 1.225,
        mu_gas: float = 1.8e-5,
        Cp_gas: float = 1005.0,
        Cp_fuel: float = 2000.0,
        L_vap: float = 2.5e6,
    ) -> None:
        super().__init__(case_path)

        self.rho_gas = rho_gas
        self.mu_gas = mu_gas
        self.Cp_gas = Cp_gas
        self.Cp_fuel = Cp_fuel
        self.L_vap = L_vap

        # Lagrangian cloud
        self.cloud = cloud or KinematicCloud(
            fluid_velocity=[0.0, 0.0, 0.0],
            fluid_density=rho_gas,
            fluid_viscosity=mu_gas,
            domain_min=[0.0, 0.0, 0.0],
            domain_max=[1.0, 1.0, 1.0],
        )

        # Coupling helper
        self.coupling = LagrangianCoupling(
            mesh=self.mesh,
            cloud=self.cloud,
            Cp_fuel=Cp_fuel,
            L_vap=L_vap,
        )

        # fvSolution settings
        self._read_fv_solution_settings()

        # Fields
        self.U, self.p, self.T, self.phi = self._init_fields()
        self._U_data, self._p_data, self._T_data = self._init_field_data()

        logger.info(
            "SprayFoam ready: rho_gas=%.3f, mu_gas=%.2e, "
            "n_particles=%d",
            rho_gas, mu_gas, self.cloud.n_particles,
        )

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _read_fv_solution_settings(self) -> None:
        """Read PIMPLE settings from fvSolution."""
        fv = self.case.fvSolution

        self.p_solver = str(fv.get_path("solvers/p/solver", "PCG"))
        self.p_tolerance = float(fv.get_path("solvers/p/tolerance", 1e-6))
        self.p_max_iter = int(fv.get_path("solvers/p/maxIter", 1000))

        self.U_solver = str(fv.get_path("solvers/U/solver", "PBiCGStab"))
        self.U_tolerance = float(fv.get_path("solvers/U/tolerance", 1e-6))
        self.U_max_iter = int(fv.get_path("solvers/U/maxIter", 1000))

        self.n_outer_correctors = int(
            fv.get_path("PIMPLE/nOuterCorrectors", 3)
        )
        self.n_correctors = int(
            fv.get_path("PIMPLE/nCorrectors", 2)
        )

        self.alpha_p = float(fv.get_path("PIMPLE/relaxationFactors/p", 0.3))
        self.alpha_U = float(fv.get_path("PIMPLE/relaxationFactors/U", 0.7))

        self.convergence_tolerance = float(
            fv.get_path("PIMPLE/convergenceTolerance", 1e-4)
        )
        self.max_outer_iterations = int(
            fv.get_path("PIMPLE/maxOuterIterations", 100)
        )

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

        # Temperature: try T, fall back to uniform 300 K
        try:
            T_tensor, _ = self.read_field_tensor("T", 0)
            T = T_tensor.to(device=device, dtype=dtype)
        except Exception:
            T = torch.full((self.mesh.n_cells,), 300.0, dtype=dtype, device=device)

        phi = torch.zeros(self.mesh.n_faces, dtype=dtype, device=device)

        return U, p, T, phi

    def _init_field_data(self):
        """Store raw FieldData for writing."""
        U_data = self.case.read_field("U", 0)
        p_data = self.case.read_field("p", 0)
        try:
            T_data = self.case.read_field("T", 0)
        except Exception:
            T_data = U_data  # fallback
        return U_data, p_data, T_data

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the sprayFoam solver.

        Each time step:

        1. Advance Lagrangian cloud (inject, move, evaporate).
        2. Compute coupling source terms.
        3. Solve gas-phase equations (PIMPLE) with sources.
        4. Update cloud fluid conditions for next step.

        Returns
        -------
        ConvergenceData
            Final convergence information.
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

        logger.info("Starting sprayFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            # 1. Advance Lagrangian cloud
            self.cloud.advance(self.delta_t)
            self.cloud.remove_dead()

            # 2. Compute coupling sources
            S_mom = self.coupling.momentum_source()
            S_heat = self.coupling.heat_source(self.T)
            S_mass, _ = self.coupling.mass_source(self.delta_t)

            # 3. Solve gas phase with sources
            self.U, self.p, self.T, self.phi, conv = (
                self._pimple_spray_iteration(S_mom, S_heat, S_mass)
            )
            last_convergence = conv

            # 4. Update cloud fluid conditions for next step
            self._update_cloud_fluid_conditions()

            # Check convergence
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

    # ------------------------------------------------------------------
    # Cloud update
    # ------------------------------------------------------------------

    def _update_cloud_fluid_conditions(self) -> None:
        """Update the cloud's fluid velocity and density from current
        gas-phase fields (cell-averaged).
        """
        if self.cloud.n_particles == 0:
            return

        U_mean = self.U.mean(dim=0)
        self.cloud.fluid_velocity = [
            float(U_mean[0].item()),
            float(U_mean[1].item()),
            float(U_mean[2].item()),
        ]
        self.cloud.fluid_density = self.rho_gas

    # ------------------------------------------------------------------
    # PIMPLE spray iteration
    # ------------------------------------------------------------------

    def _pimple_spray_iteration(
        self,
        S_mom: torch.Tensor,
        S_heat: torch.Tensor,
        S_mass: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ConvergenceData]:
        """Run one PIMPLE time step with Lagrangian source terms.

        Parameters
        ----------
        S_mom : torch.Tensor
            ``(n_cells, 3)`` momentum source from spray.
        S_heat : torch.Tensor
            ``(n_cells,)`` heat source from spray.
        S_mass : torch.Tensor
            ``(n_cells,)`` mass source from spray.

        Returns
        -------
        Tuple of ``(U, p, T, phi, ConvergenceData)``.
        """
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        U = self.U.clone()
        p = self.p.clone()
        T = self.T.clone()
        phi = self.phi.clone()

        convergence = ConvergenceData()

        rho = torch.full((mesh.n_cells,), self.rho_gas, dtype=dtype, device=device)
        mu = torch.full((mesh.n_cells,), self.mu_gas, dtype=dtype, device=device)

        n_outer = min(self.n_outer_correctors, self.max_outer_iterations)

        for outer in range(n_outer):
            U_prev = U.clone()
            p_prev = p.clone()

            # Momentum predictor (with spray source)
            U, A_p, H = self._momentum_predictor(
                U, p, phi, rho, mu, S_mom,
            )

            # PISO corrections
            for corr in range(self.n_correctors):
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

                p = self._solve_pressure_equation(
                    p, phiHbyA, A_p, rho, mesh,
                )

                grad_p = self._compute_grad(p, mesh)
                U = HbyA - grad_p / A_p.abs().clamp(min=1e-30).unsqueeze(-1)

                p_P = gather(p, int_owner)
                p_N = gather(p, int_neigh)
                A_p_inv = 1.0 / A_p.abs().clamp(min=1e-30)
                A_p_inv_face = (
                    w * gather(A_p_inv, int_owner)
                    + (1.0 - w) * gather(A_p_inv, int_neigh)
                )
                phi = phiHbyA - (p_N - p_P) * A_p_inv_face

            # Temperature equation (simplified)
            T = self._solve_temperature(T, S_heat, S_mass)

            # Under-relaxation
            if self.alpha_U < 1.0:
                U = self.alpha_U * U + (1.0 - self.alpha_U) * U_prev
            if self.alpha_p < 1.0:
                p = self.alpha_p * p + (1.0 - self.alpha_p) * p_prev

            # Check convergence
            U_residual = self._compute_residual(U, U_prev)
            p_residual = self._compute_residual(p, p_prev)
            continuity_error = self._compute_continuity_error(phi, rho)

            convergence.p_residual = p_residual
            convergence.U_residual = U_residual
            convergence.continuity_error = continuity_error
            convergence.outer_iterations = outer + 1

            if continuity_error < self.convergence_tolerance and outer > 0:
                convergence.converged = True
                break

        return U, p, T, phi, convergence

    # ------------------------------------------------------------------
    # Momentum
    # ------------------------------------------------------------------

    def _momentum_predictor(
        self,
        U: torch.Tensor,
        p: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        mu: torch.Tensor,
        S_mom: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve momentum equation with spray momentum source."""
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes
        face_areas = mesh.face_areas
        delta_coeffs = mesh.delta_coefficients

        mu_face = 0.5 * (gather(mu, int_owner) + gather(mu, int_neigh))
        S_mag = face_areas[:n_internal].norm(dim=1)
        delta_f = delta_coeffs[:n_internal]
        diff_coeff = mu_face * S_mag * delta_f

        flux = phi[:n_internal]
        rho_P = gather(rho, int_owner)
        rho_N = gather(rho, int_neigh)
        rho_face = torch.where(flux >= 0, rho_P, rho_N)

        flux_pos = torch.where(flux >= 0, flux, torch.zeros_like(flux))
        flux_neg = torch.where(flux < 0, flux, torch.zeros_like(flux))

        cell_volumes_safe = cell_volumes.clamp(min=1e-30)
        V_P = gather(cell_volumes_safe, int_owner)
        V_N = gather(cell_volumes_safe, int_neigh)

        dt = self.delta_t
        rho_V_dt = rho * cell_volumes / dt

        lower = (-diff_coeff + flux_neg * rho_face) / V_P
        upper = (-diff_coeff - flux_pos * rho_face) / V_N

        A_p = torch.zeros(n_cells, dtype=dtype, device=device)
        A_p = A_p + scatter_add(
            (diff_coeff - flux_neg * rho_face) / V_P, int_owner, n_cells
        )
        A_p = A_p + scatter_add(
            (diff_coeff + flux_pos * rho_face) / V_N, int_neigh, n_cells
        )
        A_p = A_p + rho_V_dt

        H = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        U_neigh = U[int_neigh]
        U_own = U[int_owner]

        owner_contrib = lower.unsqueeze(-1) * U_neigh * V_P.unsqueeze(-1)
        H.index_add_(0, int_owner, owner_contrib)

        neigh_contrib = upper.unsqueeze(-1) * U_own * V_N.unsqueeze(-1)
        H.index_add_(0, int_neigh, neigh_contrib)

        H = H + rho_V_dt.unsqueeze(-1) * self.U

        # Pressure gradient
        w = mesh.face_weights[:n_internal]
        p_P = gather(p, int_owner)
        p_N = gather(p, int_neigh)
        p_face = w * p_P + (1.0 - w) * p_N
        p_contrib = p_face.unsqueeze(-1) * face_areas[:n_internal]

        grad_p = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad_p.index_add_(0, int_owner, p_contrib)
        grad_p.index_add_(0, int_neigh, -p_contrib)

        # Spray momentum source (explicit)
        spray_source = S_mom * cell_volumes.unsqueeze(-1)

        source = H - grad_p + spray_source

        A_p_safe = A_p.abs().clamp(min=1e-30)
        U_solved = source / A_p_safe.unsqueeze(-1)

        U_new = self.alpha_U * U_solved + (1.0 - self.alpha_U) * U
        return U_new, A_p, H

    # ------------------------------------------------------------------
    # Temperature
    # ------------------------------------------------------------------

    def _solve_temperature(
        self,
        T: torch.Tensor,
        S_heat: torch.Tensor,
        S_mass: torch.Tensor,
    ) -> torch.Tensor:
        """Solve simplified temperature equation with spray sources.

        Includes convective heat transfer and evaporative cooling.

        Parameters
        ----------
        T : torch.Tensor
            ``(n_cells,)`` current temperature.
        S_heat : torch.Tensor
            ``(n_cells,)`` convective heat source.
        S_mass : torch.Tensor
            ``(n_cells,)`` mass source (evaporation rate).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` updated temperature.
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells
        cell_vol = self.mesh.cell_volumes.to(device=device, dtype=dtype)

        rho = self.rho_gas
        Cp = self.Cp_gas

        # Total heat source = convective + evaporative cooling
        Q_total = S_heat - S_mass * self.L_vap

        # Explicit Euler update
        dT = self.delta_t * Q_total / (rho * Cp)
        T_new = T + dT

        # Clamp to physically reasonable range
        T_new = T_new.clamp(min=200.0, max=2000.0)

        return T_new

    # ------------------------------------------------------------------
    # Pressure
    # ------------------------------------------------------------------

    def _solve_pressure_equation(
        self,
        p: torch.Tensor,
        phiHbyA: torch.Tensor,
        A_p: torch.Tensor,
        rho: torch.Tensor,
        mesh: Any,
    ) -> torch.Tensor:
        """Solve pressure Poisson equation."""
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        cell_volumes = mesh.cell_volumes
        face_areas = mesh.face_areas
        delta_coeffs = mesh.delta_coefficients

        w = mesh.face_weights[:n_internal]

        A_p_inv = 1.0 / A_p.abs().clamp(min=1e-30)
        A_p_inv_face = (
            w * gather(A_p_inv, int_owner)
            + (1.0 - w) * gather(A_p_inv, int_neigh)
        )

        S_mag = face_areas[:n_internal].norm(dim=1)
        delta_f = delta_coeffs[:n_internal]
        face_coeff = A_p_inv_face * S_mag * delta_f

        V_P = gather(cell_volumes.clamp(min=1e-30), int_owner)
        V_N = gather(cell_volumes.clamp(min=1e-30), int_neigh)

        lower = -face_coeff / V_P
        upper = -face_coeff / V_N

        diag = torch.zeros(n_cells, dtype=p.dtype, device=p.device)
        diag = diag + scatter_add(face_coeff / V_P, int_owner, n_cells)
        diag = diag + scatter_add(face_coeff / V_N, int_neigh, n_cells)

        source = torch.zeros(n_cells, dtype=p.dtype, device=p.device)
        source = source + scatter_add(phiHbyA, int_owner, n_cells)
        source = source + scatter_add(-phiHbyA, int_neigh, n_cells)

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
    # Gradient
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Continuity error
    # ------------------------------------------------------------------

    def _compute_continuity_error(
        self, phi: torch.Tensor, rho: torch.Tensor,
    ) -> float:
        """Compute continuity error."""
        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour

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

    def _compute_residual(
        self,
        field: torch.Tensor,
        field_old: torch.Tensor,
    ) -> float:
        """Compute the L2 norm of the field change."""
        diff = field - field_old
        norm_diff = float(torch.norm(diff).item())
        norm_field = float(torch.norm(field).item())
        if norm_field > 1e-30:
            return norm_diff / norm_field
        return norm_diff

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write U, p, T to a time directory."""
        time_str = f"{time:g}"
        self.write_field("U", self.U, time_str, self._U_data)
        self.write_field("p", self.p, time_str, self._p_data)
        self.write_field("T", self.T, time_str, self._T_data)
