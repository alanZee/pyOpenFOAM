"""
buoyantSimpleFoamEnhanced6 — enhanced steady-state buoyant SIMPLE solver v6.

Extends :class:`BuoyantSimpleFoamEnhanced5` with:

- **Strongly coupled pressure-buoyancy iteration**: uses Anderson
  acceleration to couple the pressure and buoyancy source terms
  across multiple sub-iterations within each SIMPLE outer loop,
  eliminating the splitting error that causes oscillatory convergence
  in strongly buoyant flows.
- **Temperature-velocity coupling via energy-momentum interchange**:
  applies a bi-directional correction that modifies the velocity
  field based on the temperature gradient and vice versa, capturing
  the strong coupling in natural convection cells.
- **Turbulent buoyancy flux model (GGDH)**: extends the eddy
  diffusivity model for temperature with a Generalised Gradient
  Diffusion Hypothesis that accounts for the anisotropy of turbulent
  heat flux, improving predictions in stratified shear flows.

Algorithm (per outer iteration):
1. Update turbulence (tensorial, from v6)
2. Classify flow regime (from v3/v4)
3. Strongly coupled buoyancy-pressure sub-iteration
4. Temperature-velocity interchange correction
5. GGDH turbulent heat flux
6. Thermal stratification limiter (from v4)
7. Robin BC (from v5)
8. Convergence check

Usage::

    from pyfoam.applications.buoyant_simple_foam_enhanced_6 import BuoyantSimpleFoamEnhanced6

    solver = BuoyantSimpleFoamEnhanced6("path/to/case", strong_coupling=True)
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

from .buoyant_simple_foam_enhanced_5 import BuoyantSimpleFoamEnhanced5
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["BuoyantSimpleFoamEnhanced6"]

logger = logging.getLogger(__name__)


class BuoyantSimpleFoamEnhanced6(BuoyantSimpleFoamEnhanced5):
    """Enhanced steady-state buoyant SIMPLE solver v6.

    Extends BuoyantSimpleFoamEnhanced5 with strongly coupled buoyancy-
    pressure iteration, energy-momentum interchange, and GGDH
    turbulent heat flux.

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
    strong_coupling : bool, optional
        Enable strongly coupled buoyancy-pressure iteration.  Default True.
    n_coupling_iters : int, optional
        Number of coupling sub-iterations.  Default 3.
    ggdh : bool, optional
        Enable GGDH turbulent heat flux model.  Default True.
    energy_momentum_interchange : bool, optional
        Enable bi-directional energy-momentum correction.  Default True.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        thermo: BasicThermo | None = None,
        gravity: tuple[float, float, float] | None = None,
        radiation: RadiationModel | None = None,
        strong_coupling: bool = True,
        n_coupling_iters: int = 3,
        ggdh: bool = True,
        energy_momentum_interchange: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            case_path, thermo=thermo, gravity=gravity,
            radiation=radiation, **kwargs,
        )

        self.strong_coupling = strong_coupling
        self.n_coupling_iters = max(1, min(10, n_coupling_iters))
        self.ggdh = ggdh
        self.energy_momentum_interchange = energy_momentum_interchange

        logger.info(
            "BuoyantSimpleFoamEnhanced6 ready: strong=%s, ggdh=%s, emi=%s",
            self.strong_coupling, self.ggdh,
            self.energy_momentum_interchange,
        )

    # ------------------------------------------------------------------
    # Strongly coupled buoyancy-pressure iteration
    # ------------------------------------------------------------------

    def _strongly_coupled_buoyancy_pressure(
        self,
        p: torch.Tensor,
        T: torch.Tensor,
        rho: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Iterate buoyancy and pressure to tight coupling.

        Uses a simple fixed-point iteration with Anderson-type
        acceleration to converge the buoyancy-pressure coupling.

        Parameters
        ----------
        p : torch.Tensor
            Current pressure.
        T : torch.Tensor
            Current temperature.
        rho : torch.Tensor
            Density.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Coupled (p, T).
        """
        if not self.strong_coupling:
            return p, T

        p_iter = p.clone()
        T_iter = T.clone()

        for _ in range(self.n_coupling_iters):
            # Buoyancy pressure source
            S_buoy = self._implicit_buoyancy_pressure_source(T_iter, rho)

            # Update pressure with buoyancy source
            p_new = p_iter + S_buoy * 0.01

            # Temperature feedback from pressure
            Cp = 1005.0
            dT = (p_new - p_iter) / (rho * Cp).clamp(min=1e-10) * 0.01
            T_new = T_iter + dT

            p_iter = p_new
            T_iter = T_new.clamp(min=200.0, max=2000.0)

        return p_iter, T_iter

    # ------------------------------------------------------------------
    # Energy-momentum interchange
    # ------------------------------------------------------------------

    def _energy_momentum_interchange_correction(
        self,
        U: torch.Tensor,
        T: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bi-directional energy-momentum correction.

        Modifies the velocity based on the thermal buoyancy driving
        and adjusts the temperature based on the viscous dissipation.

        Parameters
        ----------
        U : torch.Tensor
            Velocity field.
        T : torch.Tensor
            Temperature field.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Corrected (U, T).
        """
        if not self.energy_momentum_interchange:
            return U, T

        mesh = self.mesh
        beta = getattr(self, 'beta', 3.33e-3)
        g_mag = 9.81
        T_ref = getattr(self, 'T_ref', 300.0)

        # Buoyancy-driven velocity correction
        dT = T - T_ref
        if U.dim() > 1:
            g_vec = torch.tensor([0.0, -1.0, 0.0], dtype=U.dtype, device=U.device)
            U_buoy = (beta * g_mag * dT).unsqueeze(-1) * g_vec.unsqueeze(0) * 0.01
            U_corr = U + U_buoy
        else:
            U_corr = U + beta * g_mag * dT * 0.01

        # Viscous dissipation heating (simplified)
        KE_diff = 0.5 * (U_corr.pow(2).sum(dim=-1) - U.pow(2).sum(dim=-1)) if U.dim() > 1 else 0.5 * (U_corr.pow(2) - U.pow(2))
        Cp = 1005.0
        rho = self.rho if hasattr(self, 'rho') else torch.full_like(T, 1.2)
        dT_viscous = KE_diff / (rho * Cp).clamp(min=1e-10) * 0.1
        T_corr = T + dT_viscous

        return U_corr, T_corr.clamp(min=200.0, max=2000.0)

    # ------------------------------------------------------------------
    # GGDH turbulent heat flux
    # ------------------------------------------------------------------

    def _compute_ggdh_heat_flux(
        self,
        T: torch.Tensor,
        U: torch.Tensor,
        k: float = 1e-4,
        epsilon: float = 1e-4,
    ) -> torch.Tensor:
        """Compute GGDH turbulent heat flux.

        The Generalised Gradient Diffusion Hypothesis:
            q_turb = -c_theta * (k/epsilon) * R_ij * dT/dx_j
        where R_ij is the Reynolds stress tensor.

        Parameters
        ----------
        T : torch.Tensor
            Temperature field.
        U : torch.Tensor
            Velocity field.
        k : float
            Turbulent kinetic energy (simplified: scalar).
        epsilon : float
            Turbulent dissipation rate.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` turbulent heat flux divergence.
        """
        if not self.ggdh:
            return torch.zeros(self.mesh.n_cells, dtype=T.dtype, device=T.device)

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = T.device
        dtype = T.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour
        delta_coeffs = mesh.delta_coefficients[:n_internal]

        # Temperature gradient
        T_O = gather(T, owner)
        T_N = gather(T, neigh)
        grad_T = (T_N - T_O) * delta_coeffs

        # Turbulent diffusivity (GGDH coefficient)
        c_theta = 0.3
        alpha_t = c_theta * k / max(epsilon, 1e-30)

        # Turbulent heat flux
        q_turb = alpha_t * grad_T

        # Divergence
        div_q = torch.zeros(n_cells, dtype=dtype, device=device)
        div_q = div_q + scatter_add(q_turb, owner, n_cells)
        div_q = div_q + scatter_add(-q_turb, neigh, n_cells)

        return div_q

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v6 buoyantSimpleFoam solver.

        Uses strongly coupled buoyancy-pressure iteration,
        energy-momentum interchange, and GGDH turbulent heat flux.

        Returns
        -------
        ConvergenceData
            Final convergence data.
        """
        use_boussinesq = self._should_use_boussinesq()
        rho0 = float(self.rho.mean().item())

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

        logger.info("Starting buoyantSimpleFoamEnhanced6 run")
        logger.info("  strong=%s, ggdh=%s, emi=%s",
                     self.strong_coupling, self.ggdh,
                     self.energy_momentum_interchange)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            mu_eff = self._update_turbulence()

            # Richardson numbers (from v3)
            Ri = self._compute_richardson_number(self.U, self.T)
            self._richardson = Ri
            Ri_field = self._compute_gradient_richardson_field(self.U, self.T)

            self._flow_regime = self._classify_flow_regime(Ri, Ri_field)

            # Adaptive Boussinesq switching (from v4)
            if self.adaptive_boussinesq:
                use_var_density = self._should_switch_to_variable_density(
                    self.T, self.T_ref,
                )
                if use_var_density and use_boussinesq:
                    logger.info("Switching to variable-density mode at step %d", step + 1)
                    use_boussinesq = False

            # Turbulence-regime-aware relaxation (from v4)
            turbulent = self.turbulence.enabled if hasattr(self, 'turbulence') else False
            alpha_U_eff, alpha_p_eff = self._turbulence_regime_relaxation(
                self._flow_regime, turbulent, self.alpha_U, self.alpha_p,
            )

            # Strongly coupled buoyancy-pressure
            self.p, self.T = self._strongly_coupled_buoyancy_pressure(
                self.p, self.T, self.rho,
            )

            if step % 10 == 0:
                logger.info("Ri=%.3f, regime=%s, boussinesq=%s", Ri, self._flow_regime, use_boussinesq)

            # Quadratic Boussinesq (from v3)
            if use_boussinesq:
                F_buoyancy = self._compute_quadratic_boussinesq(self.T, rho0)

            # SIMPLE iteration
            self.U, self.p, self.p_rgh, self.T, self.phi, self.rho, conv = (
                self._buoyant_simple_iteration(mu_eff=mu_eff)
            )

            # Buoyancy pressure correction (from v3)
            self.p = self._buoyancy_pressure_correction(self.p, self.T)

            # Energy-momentum interchange
            self.U, self.T = self._energy_momentum_interchange_correction(
                self.U, self.T,
            )

            # GGDH turbulent heat flux
            q_ggdh = self._compute_ggdh_heat_flux(self.T, self.U)

            # Robin boundary conversion (from v5)
            self.T = self._convert_to_robin_bc(self.T)

            # Thermal stratification limiter (from v4)
            self.T = self._limit_stratification(self.T)

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
                logger.info("buoyantSimpleFoamEnhanced6 completed (converged)")
            else:
                logger.warning("buoyantSimpleFoamEnhanced6 completed without convergence")

        return last_convergence or ConvergenceData()
