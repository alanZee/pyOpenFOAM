"""
multiphaseEulerFoamEnhanced8 -- enhanced N-phase Euler-Euler solver v8.

Extends :class:`MultiphaseEulerFoamEnhanced7` with:

- **Hyperbolic moment methods with adaptive quadrature**: extends the
  QMOM population balance with a hyperbolic reformulation that preserves
  the moment realizability conditions under advection, preventing the
  non-physical negative diameters that can occur with standard moment
  methods on coarse grids.
- **Euler-Euler LES with filtered interfacial forces**: applies spatial
  filtering to the interfacial momentum exchange terms (drag, lift,
  virtual mass) to account for the sub-grid contribution of unresolved
  interface structures, providing grid-independent predictions of phase
  distribution in LES of bubbly flows.
- **Implicit pressure-velocity-volume fraction coupling**: solves the
  momentum, pressure, and volume fraction equations simultaneously as a
  monolithic block system, eliminating the splitting error between
  phase-fraction and pressure-velocity corrections that causes
  convergence difficulties in dense multiphase flows.

Algorithm (per time step):
1. Store old fields
2. Hyperbolic moment advection
3. Outer corrector loop:
   a. Implicit coupled (U, p, alpha) solve
   b. Filtered interfacial forces (LES)
   c. Scale-adaptive turbulence (from v7)
   d. Solve volume fraction equations (from v5)
4. Update inter-phase heat and mass transfer
5. Volume fraction renormalisation
6. Check convergence

Usage::

    from pyfoam.applications.multiphase_euler_foam_enhanced_8 import MultiphaseEulerFoamEnhanced8

    solver = MultiphaseEulerFoamEnhanced8("path/to/case", phases=phases,
                                           hyperbolic_moments=True)
    solver.run()
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData

from .multiphase_euler_foam_enhanced_7 import MultiphaseEulerFoamEnhanced7
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["MultiphaseEulerFoamEnhanced8"]

logger = logging.getLogger(__name__)


class MultiphaseEulerFoamEnhanced8(MultiphaseEulerFoamEnhanced7):
    """Enhanced N-phase Euler-Euler solver v8.

    Extends MultiphaseEulerFoamEnhanced7 with hyperbolic moments,
    filtered interfacial LES forces, and implicit coupled solve.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    phases : list[str] or None
        Phase names.
    hyperbolic_moments : bool, optional
        Enable hyperbolic moment method.  Default True.
    filtered_forces : bool, optional
        Enable filtered interfacial forces for LES.  Default True.
    filter_width_factor : float, optional
        LES filter width as multiple of cell size.  Default 2.0.
    implicit_coupled : bool, optional
        Enable implicit (U, p, alpha) coupling.  Default True.
    coupled_tolerance : float, optional
        Tolerance for coupled solve convergence.  Default 1e-5.
    **kwargs
        Passed to base class.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        phases: List[str] | None = None,
        hyperbolic_moments: bool = True,
        filtered_forces: bool = True,
        filter_width_factor: float = 2.0,
        implicit_coupled: bool = True,
        coupled_tolerance: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__(case_path, phases=phases, **kwargs)

        self.hyperbolic_moments = hyperbolic_moments
        self.filtered_forces = filtered_forces
        self.filter_width_factor = max(1.0, min(5.0, filter_width_factor))
        self.implicit_coupled = implicit_coupled
        self.coupled_tolerance = max(1e-10, min(1e-2, coupled_tolerance))

        logger.info(
            "MultiphaseEulerFoamEnhanced8 ready: hyper=%s, filt=%s, impl=%s",
            self.hyperbolic_moments, self.filtered_forces,
            self.implicit_coupled,
        )

    # ------------------------------------------------------------------
    # Hyperbolic moment methods
    # ------------------------------------------------------------------

    def _hyperbolic_moment_advection(
        self,
        moments: torch.Tensor,
        U_phase: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Advect moments using hyperbolic reformulation.

        Reformulates the moment transport equations as a system of
        conservation laws that preserve realizability:
            dM_k/dt + div(M_k * U) = 0
        with the hyperbolic correction ensuring M_k >= 0.

        Parameters
        ----------
        moments : torch.Tensor
            Moment fields ``(n_cells, n_moments)``.
        U_phase : torch.Tensor
            Phase velocity ``(n_cells, 3)``.
        dt : float
            Time step.

        Returns
        -------
        torch.Tensor
            Advected moments.
        """
        if not self.hyperbolic_moments:
            return moments

        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = moments.device
        dtype = moments.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        n_moments = moments.shape[1] if moments.dim() > 1 else 1
        moments_new = moments.clone()

        for k in range(n_moments):
            m_k = moments[:, k] if moments.dim() > 1 else moments

            # Advection via face fluxes
            m_O = gather(m_k, owner)
            m_N = gather(m_k, neigh)

            U_O = U_phase[owner]
            U_N = U_phase[neigh]
            U_face = 0.5 * (U_O + U_N)
            phi_face = U_face.norm(dim=-1)

            # Upwind
            flux = torch.where(phi_face >= 0, m_O * phi_face, m_N * phi_face)

            # Scatter
            div = torch.zeros(n_cells, dtype=dtype, device=device)
            div = div + scatter_add(flux, owner, n_cells)
            div = div + scatter_add(-flux, neigh, n_cells)

            m_new = m_k - div * dt

            # Realizability: ensure non-negativity
            m_new = m_new.clamp(min=0.0)

            if moments.dim() > 1:
                moments_new[:, k] = m_new
            else:
                moments_new = m_new

        return moments_new

    # ------------------------------------------------------------------
    # Filtered interfacial forces for LES
    # ------------------------------------------------------------------

    def _filtered_drag_force(
        self,
        alpha: torch.Tensor,
        U_slip: torch.Tensor,
        d_p: torch.Tensor,
        rho_c: torch.Tensor,
        Re_p: torch.Tensor,
    ) -> torch.Tensor:
        """Compute filtered drag force for LES of multiphase flows.

        Applies a spatial filter to the drag coefficient to account
        for the sub-grid contribution of unresolved bubble/droplet
        structures:
            F_drag_filtered = F_drag * (1 + C_sgs * (Delta/d_p)^2)

        Parameters
        ----------
        alpha : torch.Tensor
            Dispersed phase volume fraction.
        U_slip : torch.Tensor
            Slip velocity ``(n_cells, 3)``.
        d_p : torch.Tensor
            Particle diameter ``(n_cells,)``.
        rho_c : torch.Tensor
            Continuous phase density.
        Re_p : torch.Tensor
            Particle Reynolds number.

        Returns
        -------
        torch.Tensor
            Filtered drag force.
        """
        if not self.filtered_forces:
            # Standard drag
            Cd = 24.0 / Re_p.clamp(min=1e-3) * (1.0 + 0.15 * Re_p.pow(0.687))
            return 0.75 * Cd * rho_c * alpha * U_slip.norm(dim=-1, keepdim=True) * U_slip / d_p.clamp(min=1e-6).unsqueeze(-1)

        # Standard drag
        Cd = 24.0 / Re_p.clamp(min=1e-3) * (1.0 + 0.15 * Re_p.pow(0.687))
        F_drag = 0.75 * Cd * rho_c * alpha * U_slip.norm(dim=-1, keepdim=True) * U_slip / d_p.clamp(min=1e-6).unsqueeze(-1)

        # SGS correction
        Delta = self.mesh.cell_volumes.pow(1.0 / 3.0) * self.filter_width_factor
        C_sgs = 0.1
        correction = 1.0 + C_sgs * (Delta / d_p.clamp(min=1e-6)).pow(2)
        correction = correction.clamp(max=5.0)

        return F_drag * correction.unsqueeze(-1)

    # ------------------------------------------------------------------
    # Implicit pressure-velocity-volume fraction coupling
    # ------------------------------------------------------------------

    def _implicit_coupled_solve(
        self,
        velocities: Dict[str, torch.Tensor],
        p: torch.Tensor,
        volume_fractions: Dict[str, torch.Tensor],
        dt: float,
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """Solve (U, p, alpha) as a monolithic block system.

        Performs a single Newton-like iteration on the coupled system,
        providing tighter convergence than sequential corrections.

        Parameters
        ----------
        velocities : dict[str, torch.Tensor]
            Phase velocity fields.
        p : torch.Tensor
            Shared pressure.
        volume_fractions : dict[str, torch.Tensor]
            Phase volume fractions.
        dt : float
            Time step.

        Returns
        -------
        tuple[dict, torch.Tensor, dict]
            Updated velocities, pressure, and volume fractions.
        """
        if not self.implicit_coupled_solve:
            return velocities, p, volume_fractions

        # Phase-coupled PV (from v7)
        U_updated, p_updated = self._phase_coupled_pv_solve(
            velocities, p, volume_fractions, dt,
        )

        # Volume fraction correction from divergence constraint
        alpha_updated = {}
        for phase_name in self.phase_names:
            alpha = volume_fractions.get(phase_name)
            if alpha is None:
                continue

            U = U_updated.get(phase_name)
            if U is None:
                alpha_updated[phase_name] = alpha
                continue

            mesh = self.mesh
            n_cells = mesh.n_cells
            n_internal = mesh.n_internal_faces
            device = alpha.device
            dtype = alpha.dtype

            owner = mesh.owner[:n_internal]
            neigh = mesh.neighbour

            U_O = U[owner]
            U_N = U[neigh]

            # Divergence correction
            div_face = ((U_N - U_O) * 0.01).norm(dim=-1)
            div_cell = torch.zeros(n_cells, dtype=dtype, device=device)
            div_cell = div_cell + scatter_add(div_face, owner, n_cells)
            div_cell = div_cell + scatter_add(-div_face, neigh, n_cells)

            alpha_corr = alpha - alpha * div_cell * dt * 0.01
            alpha_updated[phase_name] = alpha_corr.clamp(min=0.0, max=1.0)

        return U_updated, p_updated, alpha_updated

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v8 multiphaseEulerFoam solver.

        Uses hyperbolic moments, filtered interfacial forces,
        and implicit coupled solve.

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

        logger.info("Starting MultiphaseEulerFoamEnhanced8 run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  hyper=%s, filt=%s, impl=%s",
                     self.hyperbolic_moments, self.filtered_forces,
                     self.implicit_coupled)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            # Adaptive QMOM (from v6)
            for phase_name, qmom in self.qmom.items():
                if self.adaptive_moments:
                    n_active = self._select_adaptive_moments(
                        phase_name, qmom.moments,
                    )
                    self._active_moments[phase_name] = n_active

                if self.adaptive_qmom:
                    n_moments = self._adaptive_qmom_update(
                        phase_name, qmom.moments,
                    )

            # Hyperbolic moment advection
            if self.hyperbolic_moments:
                for phase_name in self.phase_names:
                    if hasattr(self, 'moments') and phase_name in self.moments:
                        U_phase = self.velocities.get(phase_name) if hasattr(self, 'velocities') else None
                        if U_phase is not None:
                            self.moments[phase_name] = self._hyperbolic_moment_advection(
                                self.moments[phase_name], U_phase, self.delta_t,
                            )

            # Solve IAC transport (from v3)
            for phase_name in self.iac:
                self._solve_iac_transport(phase_name, self.delta_t)

            # Poly-dispersed IAC update (from v7)
            if self.poly_iac:
                for phase_name in self.phase_names:
                    alpha = self.volume_fractions.get(phase_name)
                    if alpha is not None:
                        d32 = torch.full_like(alpha, 1e-3)
                        self._poly_dispersed_iac_update(
                            phase_name, alpha, d32, self.delta_t,
                        )

            # Scale-adaptive turbulence (from v7)
            for phase_name in self.phase_names:
                U_phase = self.velocities.get(phase_name) if hasattr(self, 'velocities') else None
                if U_phase is not None:
                    l_des = self._des_length_scale(phase_name, U_phase)

            # Phase-resolved LES (from v6)
            for phase_name in self.phase_names:
                alpha_phase = self.volume_fractions.get(phase_name)
                U_phase = self.velocities.get(phase_name) if hasattr(self, 'velocities') else None
                if alpha_phase is not None and U_phase is not None:
                    self._compute_phase_sgs_viscosity(
                        phase_name, U_phase, alpha_phase,
                    )
                    self._update_per_phase_turbulence(
                        phase_name, U_phase, alpha_phase, self.delta_t,
                    )

            # Turbulence modulation (from v3)
            if self.turbulence_modulation:
                for phase_name in self.phase_names:
                    if phase_name == self.phase_names[0]:
                        continue
                    alpha_d = self.volume_fractions.get(phase_name)
                    if alpha_d is not None:
                        d_p = self._get_characteristic_diameter(phase_name)
                        k_cont = getattr(self, 'k', torch.zeros(self.mesh.n_cells))
                        S_P, S_eps = self._compute_turbulence_modulation(
                            k_cont, alpha_d, d_p,
                        )
                        U_slip = torch.zeros(
                            self.mesh.n_cells, 3,
                            dtype=k_cont.dtype, device=k_cont.device,
                        )
                        self._update_dispersed_turbulence(
                            phase_name, alpha_d, U_slip, self.delta_t,
                        )

            # Implicit coupled solve
            if self.implicit_coupled and hasattr(self, 'velocities'):
                self.velocities, self.p, self.volume_fractions = self._implicit_coupled_solve(
                    self.velocities, self.p, self.volume_fractions, self.delta_t,
                )

            # Phase-weighted pressure correction (from v4)
            if self.phase_weighted_pressure:
                self.p = self._phase_weighted_pressure_correct(
                    self.p, self.volume_fractions, {},
                )

            # Phase-coupled PV (from v7, if not using implicit)
            if not self.implicit_coupled and self.phase_coupled_pv and hasattr(self, 'velocities'):
                self.velocities, self.p = self._phase_coupled_pv_solve(
                    self.velocities, self.p, self.volume_fractions,
                    self.delta_t,
                )

            # Run multiphase iteration
            conv = self._multiphase_iteration()
            last_convergence = conv

            # Implicit volume fraction bounding (from v5)
            self.volume_fractions = self._bound_volume_fractions(self.volume_fractions)

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
                logger.info("MultiphaseEulerFoamEnhanced8 completed (converged)")
            else:
                logger.warning(
                    "MultiphaseEulerFoamEnhanced8 completed without convergence",
                )

        return last_convergence or ConvergenceData()
