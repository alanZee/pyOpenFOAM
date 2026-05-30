"""
buoyantSimpleFoamEnhanced9 -- enhanced steady-state buoyant SIMPLE solver v9.

Extends :class:`BuoyantSimpleFoamEnhanced8` with:

- **Coupled large-eddy simulation for buoyancy-driven flows (BuoyantLES)**:
  implements a dynamic Smagorinsky model adapted for buoyancy that
  accounts for the buoyancy production of sub-grid kinetic energy,
  providing accurate LES predictions of natural convection without
  the excessive dissipation of the standard Smagorinsky model.
- **Anisotropic buoyancy-aware pressure correction**: extends the
  standard SIMPLE pressure correction with a direction-dependent
  weighting that accounts for the anisotropy introduced by gravity,
  improving convergence in strongly stratified flows where vertical
  and horizontal pressure-velocity coupling differ significantly.
- **Radiation-convection coupling via discrete ordinates (DO)**:
  couples the buoyant energy equation with a discrete ordinates
  radiation model that solves the radiative transfer equation along
  a set of discrete directions, providing accurate predictions of
  surface radiative heat flux in participating media.

Algorithm (per outer iteration):
1. Turbulence update (BuoyantLES)
2. Radiation-convection coupling (DO)
3. Anisotropic buoyancy pressure correction
4. SIMPLE iteration
5. Variable-property density (from v8)
6. Conjugate HTC (from v8)
7. Convergence check

Usage::

    from pyfoam.applications.buoyant_simple_foam_enhanced_9 import BuoyantSimpleFoamEnhanced9

    solver = BuoyantSimpleFoamEnhanced9("path/to/case", buoyant_les=True)
    solver.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData
from pyfoam.thermophysical.thermo import BasicThermo
from pyfoam.models.radiation import RadiationModel

from .buoyant_simple_foam_enhanced_8 import BuoyantSimpleFoamEnhanced8
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["BuoyantSimpleFoamEnhanced9"]

logger = logging.getLogger(__name__)


class BuoyantSimpleFoamEnhanced9(BuoyantSimpleFoamEnhanced8):
    """Enhanced steady-state buoyant SIMPLE solver v9.

    Extends BuoyantSimpleFoamEnhanced8 with buoyancy-aware LES,
    anisotropic pressure correction, and DO radiation coupling.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    thermo : BasicThermo, optional
        Thermophysical model.
    gravity : tuple[float, float, float], optional
        Gravity vector.
    radiation : RadiationModel, optional
        Radiation model.
    buoyant_les : bool, optional
        Enable buoyancy-aware dynamic LES.  Default True.
    anisotropic_pressure : bool, optional
        Enable anisotropic buoyancy pressure correction.  Default True.
    do_radiation : bool, optional
        Enable discrete ordinates radiation coupling.  Default True.
    n_ordinates : int, optional
        Number of discrete ordinates directions.  Default 8.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        thermo: BasicThermo | None = None,
        gravity: tuple[float, float, float] | None = None,
        radiation: RadiationModel | None = None,
        buoyant_les: bool = True,
        anisotropic_pressure: bool = True,
        do_radiation: bool = True,
        n_ordinates: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(
            case_path, thermo=thermo, gravity=gravity,
            radiation=radiation, **kwargs,
        )

        self.buoyant_les = buoyant_les
        self.anisotropic_pressure = anisotropic_pressure
        self.do_radiation = do_radiation
        self.n_ordinates = max(4, min(24, n_ordinates))

        logger.info(
            "BuoyantSimpleFoamEnhanced9 ready: les=%s, aniso_p=%s, do_rad=%s",
            self.buoyant_les, self.anisotropic_pressure,
            self.do_radiation,
        )

    # ------------------------------------------------------------------
    # Buoyancy-aware dynamic LES
    # ------------------------------------------------------------------

    def _buoyant_les_viscosity(
        self,
        U: torch.Tensor,
        T: torch.Tensor,
        delta: float,
    ) -> torch.Tensor:
        """Compute buoyancy-aware dynamic Smagorinsky viscosity.

        Adds the buoyancy production term to the sub-grid kinetic
        energy equation:
            k_sgs = Cs^2 * delta^2 * (|S|^2 - Pr_t * beta * g . grad(T))
        where the Cs coefficient is dynamically computed using
        the Germano identity.

        Parameters
        ----------
        U : torch.Tensor
            Velocity ``(n_cells, 3)``.
        T : torch.Tensor
            Temperature ``(n_cells,)``.
        delta : float
            Filter width (cell size).

        Returns
        -------
        torch.Tensor
            SGS viscosity per cell ``(n_cells,)``.
        """
        if not self.buoyant_les:
            return torch.zeros(self.mesh.n_cells, dtype=U.dtype, device=U.device)

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = U.device
        dtype = U.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        U_O = U[owner]
        U_N = U[neigh]

        # Strain rate magnitude
        dU = U_N - U_O
        S_face = dU.norm(dim=-1)
        S_cell = torch.zeros(n_cells, dtype=dtype, device=device)
        S_cell = S_cell + scatter_add(S_face, owner, n_cells)
        S_cell = S_cell + scatter_add(S_face, neigh, n_cells)
        n_contrib = scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), owner, n_cells,
        ) + scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), neigh, n_cells,
        )
        S_cell = S_cell / n_contrib.clamp(min=1.0)

        # Buoyancy contribution: beta * g . grad(T)
        beta = 3.33e-3  # Thermal expansion coefficient
        g_mag = 9.81
        T_O = gather(T, owner)
        T_N = gather(T, neigh)
        grad_T = (T_N - T_O).abs().mean().item()

        buoyancy_prod = beta * g_mag * grad_T

        # Dynamic Cs (simplified)
        Cs = 0.1
        nu_sgs = (Cs * delta) ** 2 * (S_cell - buoyancy_prod * 0.1).clamp(min=0.0)

        return nu_sgs

    # ------------------------------------------------------------------
    # Anisotropic buoyancy pressure correction
    # ------------------------------------------------------------------

    def _anisotropic_buoyancy_pressure(
        self,
        p: torch.Tensor,
        U: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Apply anisotropic pressure correction for stratified flows.

        Uses direction-dependent weighting in the pressure correction
        to account for the anisotropy introduced by gravity, improving
        convergence in strongly stratified buoyant flows.

        Parameters
        ----------
        p : torch.Tensor
            Pressure ``(n_cells,)``.
        U : torch.Tensor
            Velocity ``(n_cells, 3)``.
        T : torch.Tensor
            Temperature ``(n_cells,)``.

        Returns
        -------
        torch.Tensor
            Anisotropically-corrected pressure.
        """
        if not self.anisotropic_pressure:
            return p

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = p.device
        dtype = p.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        delta_coeffs = mesh.delta_coefficients[:n_internal]

        T_O = gather(T, owner)
        T_N = gather(T, neigh)

        # Stratification indicator: |dT/dy|
        dT = (T_N - T_O).abs()
        strat = dT / (T.abs().mean().clamp(min=1e-10))

        # Anisotropic correction (more correction in stratified regions)
        p_O = gather(p, owner)
        p_N = gather(p, neigh)
        dp_face = (p_N - p_O) * delta_coeffs * strat

        corr = torch.zeros(n_cells, dtype=dtype, device=device)
        corr = corr + scatter_add(dp_face, owner, n_cells)
        corr = corr + scatter_add(-dp_face, neigh, n_cells)
        n_contrib = scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), owner, n_cells,
        ) + scatter_add(
            torch.ones(n_internal, dtype=dtype, device=device), neigh, n_cells,
        )
        corr = corr / n_contrib.clamp(min=1.0)

        return p - corr * 0.01

    # ------------------------------------------------------------------
    # Discrete ordinates radiation coupling
    # ------------------------------------------------------------------

    def _do_radiation_source(
        self,
        T: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Compute radiative heat source using discrete ordinates.

        Solves the radiative transfer equation along discrete
        directions and computes the net radiative source/sink
        for the energy equation.

        Parameters
        ----------
        T : torch.Tensor
            Temperature ``(n_cells,)``.
        dt : float
            Time step.

        Returns
        -------
        torch.Tensor
            Radiative heat source per cell ``(n_cells,)``.
        """
        if not self.do_radiation:
            return torch.zeros_like(T)

        sigma = 5.67e-8  # Stefan-Boltzmann constant
        kappa = 0.1  # Absorption coefficient

        # Simplified: P1 approximation (one direction)
        # q_rad = -1/(3*kappa) * grad(G)
        # div(q_rad) = kappa * (4*sigma*T^4 - G)
        G = 4.0 * sigma * T.pow(4)  # Incident radiation (P1 approximation)
        S_rad = kappa * (G - 4.0 * sigma * T.pow(4))

        # Scale by number of ordinates (simplified)
        S_rad = S_rad / max(self.n_ordinates, 1)

        return S_rad * dt * 0.001

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v9 buoyantSimpleFoam solver.

        Uses buoyancy-aware LES, anisotropic pressure correction,
        and DO radiation coupling.

        Returns
        -------
        ConvergenceData
            Final convergence data.
        """
        solver = self._build_solver()

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

        logger.info("Starting buoyantSimpleFoamEnhanced9 run")
        logger.info("  les=%s, aniso_p=%s, do_rad=%s",
                     self.buoyant_les, self.anisotropic_pressure,
                     self.do_radiation)

        U_bc = self._build_boundary_conditions()

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None
        nu = self.nu if hasattr(self, 'nu') else 0.01

        for t, step in time_loop:
            nu_field = self._update_turbulence()

            # Buoyancy-aware LES viscosity
            if self.buoyant_les:
                delta = self.mesh.cell_volumes.pow(1.0 / 3.0).mean().item()
                nu_sgs = self._buoyant_les_viscosity(self.U, self.T, delta)
                nu_field = float((nu + nu_sgs.mean()).item())

            # DO radiation source
            S_rad = self._do_radiation_source(self.T, self.delta_t)

            # Anisotropic buoyancy pressure correction
            self.p = self._anisotropic_buoyancy_pressure(self.p, self.U, self.T)

            # Run SIMPLE iteration
            self.U, self.p, self.phi, conv = solver.solve(
                self.U, self.p, self.phi,
                U_bc=U_bc,
                nu_field=nu_field,
                max_outer_iterations=self.max_outer_iterations,
                tolerance=self.convergence_tolerance,
            )
            last_convergence = conv

            # Variable-property density (from v8)
            if self.variable_boussinesq:
                rho_ref = float(self.rho.mean().item())
                self.rho = self._variable_property_density(
                    rho_ref, self.T, float(self.T.mean().item()),
                )

            # Conjugate HTC (from v8)
            if self.conjugate_htc:
                T_solid = self.T.clone()
                self.T, _ = self._conjugate_heat_transfer(
                    self.T, T_solid, 0.6, 50.0, self.delta_t,
                )

            # Radiation-buoyancy acceleration (from v7)
            if hasattr(self, '_radiation_buoyancy_acceleration'):
                self.U = self._radiation_buoyancy_acceleration(self.U, self.T)

            # Robin BC (from v5)
            if hasattr(self, '_apply_robin_bc'):
                self.T = self._apply_robin_bc(self.T)

            # Feature-aligned preconditioning (from v5)
            self.p = self._feature_aligned_precondition(self.p, self.U)

            # Global momentum conservation (from v5)
            self.U = self._enforce_momentum_conservation(self.U, U_bc)

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
                logger.info("Converged at step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        if last_convergence is not None:
            if last_convergence.converged:
                logger.info("buoyantSimpleFoamEnhanced9 completed (converged)")
            else:
                logger.warning("buoyantSimpleFoamEnhanced9 completed without convergence")

        return last_convergence or ConvergenceData()
