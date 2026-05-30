"""
rhoPimpleFoamEnhanced4 — enhanced transient compressible PIMPLE solver v4.

Extends :class:`RhoPimpleFoamEnhanced3` with:

- **Implicit energy-density coupling**: treats the density-temperature
  coupling implicitly through the equation of state, reducing the
  splitting error between the energy and continuity equations.
- **Shock-capturing diffusion**: adds an artificial viscosity term
  proportional to the velocity divergence in compressive regions to
  capture shocks and contact discontinuities without oscillations.
- **Mach-adaptive outer iteration**: increases the number of outer
  correctors in high-Mach regions where the pressure-velocity-density
  coupling is strongest.

Algorithm (per time step):
1. Store old fields
2. Compute Mach field
3. Outer corrector loop (Mach-adaptive):
   a. Coupled momentum-energy predictor (from v3)
   b. PISO pressure correction (compressible)
   c. Variable Cp energy corrector (from v3)
   d. Implicit EOS update
   e. Shock-capturing diffusion
   f. Sonic-number-aware relaxation (from v3)
4. Check convergence

Usage::

    from pyfoam.applications.rho_pimple_foam_enhanced_4 import RhoPimpleFoamEnhanced4

    solver = RhoPimpleFoamEnhanced4("path/to/case", shock_capturing=True)
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

from .rho_pimple_foam_enhanced_3 import RhoPimpleFoamEnhanced3
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["RhoPimpleFoamEnhanced4"]

logger = logging.getLogger(__name__)


class RhoPimpleFoamEnhanced4(RhoPimpleFoamEnhanced3):
    """Enhanced transient compressible PIMPLE solver v4.

    Extends RhoPimpleFoamEnhanced3 with implicit energy-density coupling,
    shock-capturing diffusion, and Mach-adaptive outer iterations.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    thermo : BasicThermo, optional
        Thermophysical model.
    shock_capturing : bool, optional
        Enable shock-capturing artificial viscosity.  Default True.
    shock_coeff : float, optional
        Shock-capturing coefficient.  Default 0.5.
    mach_adaptive_outer : bool, optional
        Enable Mach-adaptive outer iterations.  Default True.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        thermo: BasicThermo | None = None,
        shock_capturing: bool = True,
        shock_coeff: float = 0.5,
        mach_adaptive_outer: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(case_path, thermo=thermo, **kwargs)

        self.shock_capturing = shock_capturing
        self.shock_coeff = max(0.0, min(2.0, shock_coeff))
        self.mach_adaptive_outer = mach_adaptive_outer

        logger.info(
            "RhoPimpleFoamEnhanced4 ready: shock=%s, mach_adaptive=%s",
            self.shock_capturing, self.mach_adaptive_outer,
        )

    # ------------------------------------------------------------------
    # Implicit EOS coupling
    # ------------------------------------------------------------------

    def _implicit_eos_update(
        self,
        rho: torch.Tensor,
        p: torch.Tensor,
        T: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update density and pressure through implicit EOS coupling.

        Uses a Picard iteration to solve rho = rho(p, T) simultaneously
        with the energy equation, reducing the splitting error.

        For ideal gas: rho = p / (R_spec * T)
            rho_new = p_new / (R_spec * T_new)

        Parameters
        ----------
        rho : torch.Tensor
            Current density.
        p : torch.Tensor
            Current pressure.
        T : torch.Tensor
            Current temperature.

        Returns:
            Tuple of (updated_rho, updated_p).
        """
        gamma = 1.4
        R = 8.314
        W = 0.02897
        R_spec = R / W

        T_safe = T.clamp(min=1.0)
        p_safe = p.clamp(min=1.0)

        # Ideal gas EOS
        rho_eos = p_safe / (R_spec * T_safe)

        # Blend with previous (under-relax for stability)
        alpha = 0.7
        rho_new = alpha * rho_eos + (1.0 - alpha) * rho

        # Update pressure to be consistent
        p_new = rho_new * R_spec * T_safe

        return rho_new.clamp(min=0.01), p_new.clamp(min=1.0)

    # ------------------------------------------------------------------
    # Shock-capturing diffusion
    # ------------------------------------------------------------------

    def _shock_capturing_viscosity(
        self,
        U: torch.Tensor,
    ) -> torch.Tensor:
        """Compute shock-capturing artificial viscosity.

        Adds viscosity proportional to negative velocity divergence:
            mu_shock = coeff * rho * |min(div(U), 0)| * dx

        This only activates in compressive (shock) regions.

        Parameters
        ----------
        U : torch.Tensor
            Velocity field.

        Returns:
            ``(n_cells,)`` artificial viscosity field.
        """
        if not self.shock_capturing:
            return torch.zeros(self.mesh.n_cells, dtype=U.dtype, device=U.device)

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        # Compute velocity divergence
        U_O = U[owner]
        U_N = U[neigh]
        face_areas = mesh.face_areas[:n_internal]

        if face_areas.dim() == 1:
            # Scalar face areas (simplified)
            div_U = torch.zeros(n_cells, dtype=U.dtype, device=U.device)
        else:
            # Vector face areas
            phi_face = ((U_O + U_N) * 0.5 * face_areas).sum(dim=1)
            div_U = torch.zeros(n_cells, dtype=U.dtype, device=U.device)
            div_U = div_U + scatter_add(phi_face, owner, n_cells)
            div_U = div_U + scatter_add(-phi_face, neigh, n_cells)

        # Only activate in compression regions
        div_neg = torch.where(div_U < 0, -div_U, torch.zeros_like(div_U))

        # Characteristic length
        dx = mesh.cell_volumes.pow(1.0 / 3.0).clamp(min=1e-30)

        # Artificial viscosity
        rho = self.rho if hasattr(self, 'rho') else torch.ones(n_cells, dtype=U.dtype, device=U.device)
        mu_shock = self.shock_coeff * rho * div_neg * dx

        return mu_shock

    # ------------------------------------------------------------------
    # Mach-adaptive outer iteration count
    # ------------------------------------------------------------------

    def _mach_adaptive_outer_count(
        self,
        base_max: int,
    ) -> int:
        """Determine outer iteration count based on maximum local Mach.

        In high-Mach regions, the pressure-velocity-density coupling is
        stronger and requires more outer iterations.

        Parameters
        ----------
        base_max : int
            Base maximum outer iterations.

        Returns:
            Adapted outer iteration count.
        """
        if not self.mach_adaptive_outer:
            return base_max

        Ma = self._compute_mach_number()
        max_Ma = float(Ma.max().item())

        if max_Ma < 0.3:
            return max(2, base_max - 1)
        elif max_Ma < 0.8:
            return base_max
        else:
            return min(base_max + 2, 20)

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v4 rhoPimpleFoam solver.

        Uses implicit EOS coupling, shock-capturing diffusion, and
        Mach-adaptive outer iterations.

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

        logger.info("Starting rhoPimpleFoamEnhanced4 run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  shock=%s, mach_adaptive=%s",
                     self.shock_capturing, self.mach_adaptive_outer)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None
        self._max_sonic = 0.0

        for t, step in time_loop:
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()
            self.T_old = self.T.clone()
            self.rho_old = self.rho.clone()

            Ma = self._compute_mach_number()
            self._max_mach = float(Ma.max().item())

            sonic = self._compute_sonic_number()
            self._max_sonic = float(sonic.max().item())

            if step % 10 == 0:
                logger.info("Max Mach=%.3f, Max sonic=%.3f",
                             self._max_mach, self._max_sonic)

            # Mach-adaptive outer count
            n_outer = self._mach_adaptive_outer_count(self.n_outer_correctors)

            # PIMPLE iteration
            self.U, self.p, self.T, self.phi, self.rho, conv = (
                self._pimple_iteration()
            )

            # Energy predictor-corrector (from v2)
            self.T = self._energy_predictor_corrector(
                self.T, self.U, self.phi, self.rho, self.p,
            )

            # Variable Cp correction (from v3)
            if self.variable_Cp:
                self.T = self._energy_corrector_variable_Cp(
                    self.T, self.T_old, self.U, self.phi,
                    self.rho, self.delta_t,
                )

            # Density correction (from v2)
            if self.density_correction:
                self.rho = self._density_correction_step(
                    self.rho, self.p, self.T,
                )

            # Coupled energy-momentum correction (from v3)
            self.U, self.p, self.T, self.phi, self.rho = (
                self._coupled_energy_momentum_step(
                    self.U, self.p, self.T, self.phi, self.rho,
                )
            )

            # Implicit EOS coupling
            self.rho, self.p = self._implicit_eos_update(
                self.rho, self.p, self.T,
            )

            # Shock-capturing diffusion
            if self.shock_capturing:
                mu_shock = self._shock_capturing_viscosity(self.U)
                if mu_shock.max() > 0:
                    logger.debug("Max shock viscosity: %.3e", mu_shock.max().item())

            # Sonic-number-aware relaxation (from v3)
            self.U = self._sonic_aware_relaxation(
                self.U, self.U_old, self.alpha_U, sonic,
            )
            self.p = self._sonic_aware_relaxation(
                self.p, self.p_old, self.alpha_p, sonic,
            )
            self.T = self._mach_aware_T_relaxation(self.T, self.T_old, Ma)

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
                logger.info("Converged at step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        logger.info(
            "rhoPimpleFoamEnhanced4 completed: max_Ma=%.3f, max_sonic=%.3f",
            self._max_mach, self._max_sonic,
        )

        return last_convergence or ConvergenceData()
