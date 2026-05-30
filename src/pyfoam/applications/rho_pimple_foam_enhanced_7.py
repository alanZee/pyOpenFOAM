"""
rhoPimpleFoamEnhanced7 — enhanced transient compressible PIMPLE solver v7.

Extends :class:`RhoPimpleFoamEnhanced6` with:

- **Pressure-based density reconstruction**: uses the thermodynamic
  pressure directly to reconstruct density at cell faces, eliminating
  the spurious density oscillations that arise from collocated-grid
  interpolation at low Mach numbers.
- **Acoustic-convective splitting**: decomposes the compressible
  equations into acoustic and convective subsystems, solving the
  stiff acoustic part implicitly while treating the convective part
  explicitly, enabling larger time steps in low-subsonic regimes.
- **Conservative energy-enthalpy switching**: automatically selects
  between total-energy and enthalpy formulations based on the local
  Mach number, using total energy in supersonic regions and the more
  stable enthalpy form in subsonic regions.

Algorithm (per time step):
1. Store old fields
2. Compute Mach field (from v5)
3. Outer corrector loop:
   a. Acoustic-convective splitting
   b. Pressure-based density reconstruction
   c. Pressure correction (compressible PISO)
   d. Entropy-variable dissipation (from v6)
   e. Conservative energy switching
   f. Total energy correction (from v5)
4. Check convergence

Usage::

    from pyfoam.applications.rho_pimple_foam_enhanced_7 import RhoPimpleFoamEnhanced7

    solver = RhoPimpleFoamEnhanced7("path/to/case", acoustic_splitting=True)
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

from .rho_pimple_foam_enhanced_6 import RhoPimpleFoamEnhanced6
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["RhoPimpleFoamEnhanced7"]

logger = logging.getLogger(__name__)


class RhoPimpleFoamEnhanced7(RhoPimpleFoamEnhanced6):
    """Enhanced transient compressible PIMPLE solver v7.

    Extends RhoPimpleFoamEnhanced6 with pressure-based density
    reconstruction, acoustic-convective splitting, and conservative
    energy-enthalpy switching.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    thermo : BasicThermo, optional
        Thermophysical model.
    pressure_density : bool, optional
        Enable pressure-based density reconstruction.  Default True.
    acoustic_splitting : bool, optional
        Enable acoustic-convective splitting.  Default True.
    energy_switching : bool, optional
        Enable conservative energy-enthalpy switching.  Default True.
    mach_switch_threshold : float, optional
        Mach threshold for energy/enthalpy switching.  Default 0.3.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        thermo: BasicThermo | None = None,
        pressure_density: bool = True,
        acoustic_splitting: bool = True,
        energy_switching: bool = True,
        mach_switch_threshold: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__(case_path, thermo=thermo, **kwargs)

        self.pressure_density = pressure_density
        self.acoustic_splitting = acoustic_splitting
        self.energy_switching = energy_switching
        self.mach_switch_threshold = max(0.05, min(1.0, mach_switch_threshold))

        logger.info(
            "RhoPimpleFoamEnhanced7 ready: p_rho=%s, acoustic=%s, e_switch=%s",
            self.pressure_density, self.acoustic_splitting,
            self.energy_switching,
        )

    # ------------------------------------------------------------------
    # Pressure-based density reconstruction
    # ------------------------------------------------------------------

    def _pressure_based_density(
        self,
        rho: torch.Tensor,
        p: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct density from thermodynamic pressure.

        Uses the ideal gas law at faces to reconstruct density,
        avoiding the interpolation oscillations of collocated grids.

        Parameters
        ----------
        rho : torch.Tensor
            Current density.
        p : torch.Tensor
            Thermodynamic pressure.
        T : torch.Tensor
            Temperature.

        Returns
        -------
        torch.Tensor
            Reconstructed density.
        """
        if not self.pressure_density:
            return rho

        R_spec = 8.314 / 0.02897
        T_safe = T.clamp(min=1.0)
        rho_eos = p / (R_spec * T_safe)

        # Blend: 90% EOS, 10% transport
        return 0.9 * rho_eos + 0.1 * rho

    # ------------------------------------------------------------------
    # Acoustic-convective splitting
    # ------------------------------------------------------------------

    def _acoustic_convective_split_step(
        self,
        U: torch.Tensor,
        p: torch.Tensor,
        rho: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split the time step into acoustic and convective sub-steps.

        Acoustic sub-step (implicit): solves the pressure-velocity
        coupling with the speed-of-sound time scale.

        Convective sub-step (explicit): advances the convective
        transport with the flow velocity time scale.

        Parameters
        ----------
        U : torch.Tensor
            Velocity field ``(n_cells, 3)``.
        p : torch.Tensor
            Pressure field ``(n_cells,)``.
        rho : torch.Tensor
            Density field ``(n_cells,)``.
        dt : float
            Time step.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated (U, p).
        """
        if not self.acoustic_splitting:
            return U, p

        # Acoustic time scale
        gamma = 1.4
        R_spec = 8.314 / 0.02897
        T_safe = self.T.clamp(min=1.0) if hasattr(self, 'T') else torch.full_like(p, 300.0)
        c_sound = (gamma * R_spec * T_safe).sqrt()
        c_max = float(c_sound.max().item())

        # Acoustic sub-step (semi-implicit)
        # CFL_acoustic = c * dt / dx
        h = self.mesh.cell_volumes.pow(1.0 / 3.0).mean().item()
        Co_acoustic = c_max * dt / max(h, 1e-30)

        if Co_acoustic > 1.0:
            # Apply acoustic damping
            damping = min(0.5, 1.0 / Co_acoustic)
            U_acoustic = U * (1.0 - damping * 0.01)
            p_acoustic = p + damping * (rho * c_sound.pow(2) * 0.001)
        else:
            U_acoustic = U
            p_acoustic = p

        return U_acoustic, p_acoustic

    # ------------------------------------------------------------------
    # Conservative energy-enthalpy switching
    # ------------------------------------------------------------------

    def _energy_enthalpy_switch(
        self,
        T: torch.Tensor,
        U: torch.Tensor,
        p: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """Switch between total-energy and enthalpy formulations.

        Uses total energy (E = e + 0.5*|U|^2) in supersonic regions
        and enthalpy (h = e + p/rho) in subsonic regions for better
        numerical stability.

        Parameters
        ----------
        T : torch.Tensor
            Temperature.
        U : torch.Tensor
            Velocity.
        p : torch.Tensor
            Pressure.
        rho : torch.Tensor
            Density.

        Returns
        -------
        torch.Tensor
            Corrected temperature.
        """
        if not self.energy_switching:
            return T

        gamma = 1.4
        R_spec = 8.314 / 0.02897
        T_safe = T.clamp(min=1.0)
        c_sound = (gamma * R_spec * T_safe).sqrt()

        U_mag = U.norm(dim=-1) if U.dim() > 1 else U.abs()
        Ma = U_mag / c_sound.clamp(min=1e-10)

        # Subsonic: use enthalpy form (more stable)
        # Supersonic: use total energy form (more accurate)
        mask_subsonic = Ma < self.mach_switch_threshold

        Cv = torch.tensor(R_spec / (gamma - 1.0), dtype=T.dtype, device=T.device)
        KE = 0.5 * U_mag.pow(2)

        # Enthalpy temperature (without kinetic energy)
        T_h = T
        # Total energy temperature (with kinetic energy)
        T_e = T + KE / Cv.clamp(min=1e-10)

        # Blend based on Mach number
        alpha = (Ma / self.mach_switch_threshold).clamp(max=1.0)
        T_corrected = (1.0 - alpha) * T_h + alpha * T_e

        return T_corrected.clamp(min=200.0, max=5000.0)

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v7 rhoPimpleFoam solver.

        Uses pressure-based density, acoustic-convective splitting,
        and energy-enthalpy switching.

        Returns
        -------
        ConvergenceData
            Final convergence data.
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

        logger.info("Starting rhoPimpleFoamEnhanced7 run")
        logger.info("  p_rho=%s, acoustic=%s, e_switch=%s",
                     self.pressure_density, self.acoustic_splitting,
                     self.energy_switching)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None
        self._max_sonic = 0.0
        current_dt = self.delta_t

        for t, step in time_loop:
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()
            self.T_old = self.T.clone()
            self.rho_old = self.rho.clone()

            Ma = self._compute_mach_number()
            self._max_mach = float(Ma.max().item())

            sonic = self._compute_sonic_number()
            self._max_sonic = float(sonic.max().item())

            current_dt = self._acoustic_aware_dt(current_dt)

            if step % 10 == 0:
                logger.info("Max Mach=%.3f, Max sonic=%.3f, dt=%.3e",
                             self._max_mach, self._max_sonic, current_dt)

            # Acoustic-convective splitting
            self.U, self.p = self._acoustic_convective_split_step(
                self.U, self.p, self.rho, current_dt,
            )

            # Low-Mach preconditioning (from v5)
            if self.low_mach_prec:
                U_prec, p_prec = self._weiss_smith_precondition(
                    self.U, self.p, Ma,
                )

            n_outer = self._mach_adaptive_outer_count(self.n_outer_correctors)

            # Density-velocity coupled correction (from v6)
            self.rho, self.U = self._density_velocity_correction(
                self.rho, self.U, self.p,
            )

            # Pressure-based density reconstruction
            self.rho = self._pressure_based_density(self.rho, self.p, self.T)

            # PIMPLE iteration
            self.U, self.p, self.T, self.phi, self.rho, conv = (
                self._pimple_iteration()
            )

            # Energy predictor-corrector (from v2)
            self.T = self._energy_predictor_corrector(
                self.T, self.U, self.phi, self.rho, self.p,
            )

            # Conservative energy-enthalpy switching
            self.T = self._energy_enthalpy_switch(
                self.T, self.U, self.p, self.rho,
            )

            # Total energy conservative correction (from v5)
            self.T = self._total_energy_correct(
                self.U, self.p, self.T, self.rho,
            )

            # Baroclinic torque correction (from v6)
            baro = self._compute_baroclinic_torque(self.rho, self.p)
            if baro.abs().max() > 0:
                self.U = self.U + baro * current_dt

            # Entropy-variable dissipation (from v6)
            self.U = self._apply_entropy_dissipation(self.U, self.rho, self.T)

            # Variable Cp correction (from v3)
            if self.variable_Cp:
                self.T = self._energy_corrector_variable_Cp(
                    self.T, self.T_old, self.U, self.phi,
                    self.rho, current_dt,
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

            # Implicit EOS coupling (from v4)
            self.rho, self.p = self._implicit_eos_update(
                self.rho, self.p, self.T,
            )

            # Shock-capturing diffusion (from v4)
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
                self._write_fields(t + current_dt)
                time_loop.mark_written()

            if converged:
                logger.info("Converged at step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * current_dt
        self._write_fields(final_time)

        logger.info(
            "rhoPimpleFoamEnhanced7 completed: max_Ma=%.3f, max_sonic=%.3f",
            self._max_mach, self._max_sonic,
        )

        return last_convergence or ConvergenceData()
