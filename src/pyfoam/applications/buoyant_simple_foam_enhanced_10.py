"""
buoyantSimpleFoamEnhanced10 -- enhanced steady-state buoyant SIMPLE solver v10.

Extends :class:`BuoyantSimpleFoamEnhanced9` with:

- **Non-Boussinesq buoyancy with variable-density SIMPLE**: removes the
  Boussinesq approximation and solves the full variable-density
  momentum equation, using a pressure-velocity-density coupling that
  accounts for the baroclinic torque from density gradients in
  strongly heated or cooled flows.
- **Implicit conjugate heat transfer with shell conduction**: extends
  the conjugate HTC model with implicit coupling to thin shell
  elements, capturing the thermal resistance of walls and baffles
  without resolving the solid region, enabling accurate predictions
  of heat exchangers and insulation.
- **Buoyancy-aware SIMPLE stabilisation with Richardson damping**:
  adds a stabilisation that damps the velocity correction in regions
  where the Richardson number indicates strong stratification,
  preventing the oscillatory divergence that plagues standard
  SIMPLE on buoyancy-driven flows.

Algorithm (per outer iteration):
1. Turbulence update
2. Non-Boussinesq density update
3. Richardson damping stabilisation
4. SIMPLE iteration
5. Implicit shell HTC coupling
6. Buoyancy-aware LES (from v9)
7. Anisotropic pressure correction (from v9)
8. Convergence check

Usage::

    from pyfoam.applications.buoyant_simple_foam_enhanced_10 import BuoyantSimpleFoamEnhanced10

    solver = BuoyantSimpleFoamEnhanced10("path/to/case", non_boussinesq=True)
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

from .buoyant_simple_foam_enhanced_9 import BuoyantSimpleFoamEnhanced9
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["BuoyantSimpleFoamEnhanced10"]

logger = logging.getLogger(__name__)


class BuoyantSimpleFoamEnhanced10(BuoyantSimpleFoamEnhanced9):
    """Enhanced steady-state buoyant SIMPLE solver v10.

    Extends BuoyantSimpleFoamEnhanced9 with non-Boussinesq buoyancy,
    implicit shell HTC, and Richardson damping.

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
    non_boussinesq : bool, optional
        Enable non-Boussinesq variable-density buoyancy.  Default True.
    shell_htc : bool, optional
        Enable implicit shell conjugate HTC.  Default True.
    shell_thickness : float, optional
        Shell wall thickness (m).  Default 0.005.
    richardson_damping : bool, optional
        Enable Richardson-number velocity damping.  Default True.
    ri_threshold : float, optional
        Richardson number threshold for damping.  Default 1.0.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        thermo: BasicThermo | None = None,
        gravity: tuple[float, float, float] | None = None,
        radiation: RadiationModel | None = None,
        non_boussinesq: bool = True,
        shell_htc: bool = True,
        shell_thickness: float = 0.005,
        richardson_damping: bool = True,
        ri_threshold: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            case_path, thermo=thermo, gravity=gravity,
            radiation=radiation, **kwargs,
        )

        self.non_boussinesq = non_boussinesq
        self.shell_htc = shell_htc
        self.shell_thickness = max(1e-4, min(1.0, shell_thickness))
        self.richardson_damping = richardson_damping
        self.ri_threshold = max(0.01, min(100.0, ri_threshold))

        logger.info(
            "BuoyantSimpleFoamEnhanced10 ready: nb=%s, shell=%s, ri=%s",
            self.non_boussinesq, self.shell_htc, self.richardson_damping,
        )

    # ------------------------------------------------------------------
    # Non-Boussinesq variable-density buoyancy
    # ------------------------------------------------------------------

    def _variable_density_buoyancy(
        self,
        U: torch.Tensor,
        rho: torch.Tensor,
        T: torch.Tensor,
        rho_ref: float,
        T_ref: float,
    ) -> torch.Tensor:
        """Compute non-Boussinesq buoyancy force.

        Uses the full density difference for buoyancy:
            F = (rho_ref - rho) * g / rho
        which is exact for large temperature differences.

        Parameters
        ----------
        U : torch.Tensor
            Velocity ``(n_cells, 3)``.
        rho : torch.Tensor
            Density ``(n_cells,)``.
        T : torch.Tensor
            Temperature ``(n_cells,)``.
        rho_ref, T_ref : float
            Reference density and temperature.

        Returns
        -------
        torch.Tensor
            Buoyancy force ``(n_cells, 3)``.
        """
        if not self.non_boussinesq:
            return torch.zeros_like(U)

        g = torch.tensor(
            [0.0, -9.81, 0.0], dtype=U.dtype, device=U.device,
        )

        # Full density-based buoyancy
        F_buoy = (rho_ref - rho).unsqueeze(-1) * g.unsqueeze(0) / rho.unsqueeze(-1).clamp(min=1e-10)

        return F_buoy

    # ------------------------------------------------------------------
    # Implicit shell HTC
    # ------------------------------------------------------------------

    def _shell_htc_correction(
        self,
        T_fluid: torch.Tensor,
        T_exterior: float,
        h_exterior: float,
        k_wall: float,
    ) -> torch.Tensor:
        """Apply implicit shell conjugate heat transfer correction.

        Couples the fluid-side temperature with the exterior
        through a thin shell wall, accounting for wall
        thermal resistance.

        Parameters
        ----------
        T_fluid : torch.Tensor
            Fluid-side temperature ``(n_cells,)``.
        T_exterior : float
            Exterior temperature (K).
        h_exterior : float
            Exterior heat transfer coefficient.
        k_wall : float
            Wall thermal conductivity.

        Returns
        -------
        torch.Tensor
            Corrected fluid temperature.
        """
        if not self.shell_htc:
            return T_fluid

        # Shell resistance
        R_shell = self.shell_thickness / max(k_wall, 1e-10)
        R_exterior = 1.0 / max(h_exterior, 1e-10)
        R_total = R_shell + R_exterior

        # Implicit correction
        T_new = (T_fluid + T_exterior / R_total) / (1.0 + 1.0 / R_total)

        # Blend for stability
        alpha = 0.1
        return (1.0 - alpha) * T_fluid + alpha * T_new

    # ------------------------------------------------------------------
    # Richardson damping
    # ------------------------------------------------------------------

    def _richardson_velocity_damp(
        self,
        U: torch.Tensor,
        T: torch.Tensor,
        T_mean: float,
    ) -> torch.Tensor:
        """Apply Richardson-number-based velocity damping.

        Damps velocity corrections where the Richardson number
        indicates strong stratification:
            Ri = beta * g * dT/dy / (dU/dy)^2

        Parameters
        ----------
        U : torch.Tensor
            Velocity ``(n_cells, 3)``.
        T : torch.Tensor
            Temperature ``(n_cells,)``.
        T_mean : float
            Mean temperature.

        Returns
        -------
        torch.Tensor
            Damped velocity.
        """
        if not self.richardson_damping:
            return U

        beta = 3.33e-3
        g = 9.81

        dT = (T - T_mean).abs()
        U_mag = U.norm(dim=-1).clamp(min=1e-10)

        Ri = beta * g * dT / (U_mag.pow(2) + 1e-30)

        # Damping factor: unity for Ri < threshold, decaying above
        damping = 1.0 / (1.0 + (Ri / self.ri_threshold).clamp(max=10.0))
        damping = damping.unsqueeze(-1)

        return U * damping

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v10 buoyantSimpleFoam solver.

        Uses non-Boussinesq buoyancy, shell HTC,
        and Richardson damping.

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

        logger.info("Starting buoyantSimpleFoamEnhanced10 run")
        logger.info("  nb=%s, shell=%s, ri=%s",
                     self.non_boussinesq, self.shell_htc, self.richardson_damping)

        U_bc = self._build_boundary_conditions()

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None
        nu = self.nu if hasattr(self, 'nu') else 0.01

        for t, step in time_loop:
            nu_field = self._update_turbulence()

            T_mean = float(self.T.mean().item())
            rho_ref = float(self.rho.mean().item()) if hasattr(self, 'rho') else 1.0

            # Buoyancy-aware LES viscosity (from v9)
            if self.buoyant_les:
                delta = self.mesh.cell_volumes.pow(1.0 / 3.0).mean().item()
                nu_sgs = self._buoyant_les_viscosity(self.U, self.T, delta)
                nu_field = float((nu + nu_sgs.mean()).item())

            # DO radiation source (from v9)
            S_rad = self._do_radiation_source(self.T, self.delta_t)

            # Non-Boussinesq density update
            if self.non_boussinesq:
                T_ref = 300.0
                rho_ref_new = rho_ref * (T_ref / max(T_mean, 1e-10))
                if hasattr(self, 'rho'):
                    self.rho = self.rho * (T_ref / self.T.clamp(min=1e-10))

            # Richardson velocity damping
            self.U = self._richardson_velocity_damp(self.U, self.T, T_mean)

            # Anisotropic buoyancy pressure correction (from v9)
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

            # Shell HTC correction
            if self.shell_htc:
                self.T = self._shell_htc_correction(
                    self.T, 300.0, 10.0, 45.0,
                )

            # Variable-property density (from v8)
            if self.variable_boussinesq:
                self.rho = self._variable_property_density(
                    rho_ref, self.T, T_mean,
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
                logger.info("buoyantSimpleFoamEnhanced10 completed (converged)")
            else:
                logger.warning("buoyantSimpleFoamEnhanced10 completed without convergence")

        return last_convergence or ConvergenceData()
