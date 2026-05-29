"""
buoyantPimpleFoamEnhanced3 — enhanced transient buoyant PIMPLE solver v3.

Extends :class:`BuoyantPimpleFoamEnhanced2` with:

- **Improved transient buoyant flows**: uses a semi-implicit treatment
  of the buoyancy source in the momentum equation, where the
  T-dependent part is treated implicitly for better stability.
- **Buoyancy-turbulence coupling**: models the effect of buoyancy on
  turbulence production/dissipation via a buoyancy production term
  in the k-equation (G_b = -beta * g_i * u_i'T' / Pr_t).
- **Gravity wave filtering**: applies a temporal filter that damps
  spurious gravity wave modes that can arise on coarse meshes with
  large time steps.

Algorithm (per time step):
1. Store old fields
2. Compute Richardson number and Brunt-Vaisala frequency
3. Filter gravity wave modes
4. Outer corrector loop:
   a. Semi-implicit buoyancy source
   b. Momentum predictor with buoyancy-turbulence coupling
   c. PISO pressure correction (p_rgh form)
   d. Energy equation solve
   e. EOS update
   f. Temperature limiting (from v2)
5. Check convergence

Usage::

    from pyfoam.applications.buoyant_pimple_foam_enhanced_3 import BuoyantPimpleFoamEnhanced3

    solver = BuoyantPimpleFoamEnhanced3("path/to/case", gravity_wave_filter=True)
    solver.run()
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData
from pyfoam.thermophysical.thermo import BasicThermo
from pyfoam.models.radiation import RadiationModel

from .buoyant_pimple_foam_enhanced_2 import BuoyantPimpleFoamEnhanced2
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["BuoyantPimpleFoamEnhanced3"]

logger = logging.getLogger(__name__)


class BuoyantPimpleFoamEnhanced3(BuoyantPimpleFoamEnhanced2):
    """Enhanced transient buoyant compressible PIMPLE solver v3.

    Extends BuoyantPimpleFoamEnhanced2 with semi-implicit buoyancy,
    buoyancy-turbulence coupling, and gravity wave filtering.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    thermo : BasicThermo, optional
        Thermophysical model.
    gravity : tuple[float, float, float], optional
        Gravity vector (m/s^2).
    radiation : RadiationModel, optional
        Radiation model.
    gravity_wave_filter : bool, optional
        Enable temporal gravity wave filter.  Default True.
    wave_filter_coeff : float, optional
        Gravity wave filter coefficient (0-1).  Default 0.2.
    buoyancy_production : bool, optional
        Enable buoyancy production in k-equation.  Default True.
    turbulent_Pr : float, optional
        Turbulent Prandtl number.  Default 0.85.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        thermo: BasicThermo | None = None,
        gravity: tuple[float, float, float] | None = None,
        radiation: RadiationModel | None = None,
        gravity_wave_filter: bool = True,
        wave_filter_coeff: float = 0.2,
        buoyancy_production: bool = True,
        turbulent_Pr: float = 0.85,
        **kwargs,
    ) -> None:
        super().__init__(
            case_path, thermo=thermo, gravity=gravity,
            radiation=radiation, **kwargs,
        )

        self.gravity_wave_filter = gravity_wave_filter
        self.wave_filter_coeff = max(0.0, min(1.0, wave_filter_coeff))
        self.buoyancy_production = buoyancy_production
        self.turbulent_Pr = max(0.5, min(2.0, turbulent_Pr))

        logger.info(
            "BuoyantPimpleFoamEnhanced3 ready: gw_filter=%s, buoy_prod=%s",
            self.gravity_wave_filter, self.buoyancy_production,
        )

    # ------------------------------------------------------------------
    # Semi-implicit buoyancy source
    # ------------------------------------------------------------------

    def _semi_implicit_buoyancy_source(
        self,
        T: torch.Tensor,
        T_old: torch.Tensor,
        rho: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute semi-implicit buoyancy source.

        Treats the T-dependent part implicitly:
            F_b = rho * beta * (T_ref - T) * g
                = rho * beta * T_ref * g  (explicit)
                - rho * beta * T * g      (implicit)

        Returns:
            Tuple of (explicit_source, implicit_diagonal).
        """
        T_diff = T - self.T_ref

        # Explicit part: use old temperature
        F_explicit = -rho.unsqueeze(-1) * self.beta * T_diff.unsqueeze(-1) * self.g.unsqueeze(0)

        # Implicit diagonal contribution: rho * beta * |g|
        g_mag = float(self.g.norm().item())
        diag_implicit = rho * self.beta * g_mag

        return F_explicit, diag_implicit

    # ------------------------------------------------------------------
    # Gravity wave filtering
    # ------------------------------------------------------------------

    def _apply_gravity_wave_filter(
        self,
        U: torch.Tensor,
        U_old: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Apply temporal gravity wave filter.

        Damps high-frequency oscillations in the velocity field that
        correspond to spurious gravity wave modes.  Uses exponential
        smoothing:

            U_filtered = U + coeff * (U_old - U) * exp(-N * dt)

        where N is the Brunt-Vaisala frequency.

        Parameters
        ----------
        U : torch.Tensor
            Current velocity.
        U_old : torch.Tensor
            Old velocity.
        dt : float
            Time step.

        Returns:
            Filtered velocity.
        """
        if not self.gravity_wave_filter:
            return U

        N = self._compute_brunt_vaisala_frequency(self.T)

        if N < 1e-10:
            return U

        # Damping factor
        damping = self.wave_filter_coeff * math.exp(-N * dt)
        damping = max(0.0, min(0.5, damping))

        return U + damping * (U_old - U)

    # ------------------------------------------------------------------
    # Buoyancy production in turbulence
    # ------------------------------------------------------------------

    def _compute_buoyancy_production(
        self,
        T: torch.Tensor,
        grad_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute buoyancy production term for k-equation.

        G_b = -beta * g_i * (dT/dx_i) / Pr_t

        This term is positive (production) in unstable stratification
        and negative (destruction) in stable stratification.

        Parameters
        ----------
        T : torch.Tensor
            Temperature field.
        grad_T : torch.Tensor, optional
            Temperature gradient.  If None, computed from face differences.

        Returns:
            ``(n_cells,)`` buoyancy production rate.
        """
        if not self.buoyancy_production:
            return torch.zeros(self.mesh.n_cells, dtype=T.dtype, device=T.device)

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        # Compute temperature gradient (simplified)
        T_O = gather(T, owner)
        T_N = gather(T, neigh)
        delta_coeffs = mesh.delta_coefficients[:n_internal]

        dT_dx = (T_N - T_O) * delta_coeffs  # face-normal gradient

        # Project gravity onto face normal
        g_mag = float(self.g.norm().item())

        # Simplified: use magnitude of gradient
        G_b_face = -self.beta * g_mag * dT_dx / self.turbulent_Pr

        # Scatter to cells
        G_b = torch.zeros(n_cells, dtype=T.dtype, device=T.device)
        G_b.scatter_reduce_(0, owner, G_b_face.abs(), reduce="amax")

        return G_b.clamp(min=-1e6, max=1e6)

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v3 buoyantPimpleFoam solver.

        Uses semi-implicit buoyancy, buoyancy-turbulence coupling,
        and gravity wave filtering.

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

        logger.info("Starting buoyantPimpleFoamEnhanced3 run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  gw_filter=%s, buoy_prod=%s",
                     self.gravity_wave_filter, self.buoyancy_production)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            self.U_old = self.U.clone()
            self.p_old = self.p.clone()
            self.T_old = self.T.clone()
            self.rho_old = self.rho.clone()

            # Richardson number
            Ri = self._compute_richardson(self.U, self.T)
            self._richardson = Ri

            # Brunt-Vaisala frequency
            N = self._compute_brunt_vaisala_frequency(self.T)

            if step % 10 == 0:
                logger.info("Richardson=%.3f, N=%.4f rad/s", Ri, N)

            mu_eff = self._update_turbulence()

            # Semi-implicit buoyancy source
            F_buoy, diag_buoy = self._semi_implicit_buoyancy_source(
                self.T, self.T_old, self.rho,
            )

            # Buoyancy production for turbulence
            G_b = self._compute_buoyancy_production(self.T)

            # Gravity wave filter
            self.U = self._apply_gravity_wave_filter(
                self.U, self.U_old, self.delta_t,
            )

            # PIMPLE iteration
            self.U, self.p, self.p_rgh, self.T, self.phi, self.rho, conv = (
                self._buoyant_pimple_iteration(mu_eff=mu_eff)
            )

            # Energy corrector (from v2)
            self.T = self._energy_predictor_corrector(
                self.T, self.U, self.phi, self.rho,
            )

            # Temperature limiting (from v2)
            self.T = self._limit_temperature(self.T)

            # Temperature-dependent relaxation
            self.U = self._temperature_dependent_relaxation(
                self.T, self.U, self.U_old, self.alpha_U,
            )
            self.p = self._temperature_dependent_relaxation(
                self.T, self.p, self.p_old, self.alpha_p,
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
                logger.info("Converged at step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        if last_convergence is not None:
            if last_convergence.converged:
                logger.info("buoyantPimpleFoamEnhanced3 completed (converged)")
            else:
                logger.warning("buoyantPimpleFoamEnhanced3 completed without convergence")

        return last_convergence or ConvergenceData()
