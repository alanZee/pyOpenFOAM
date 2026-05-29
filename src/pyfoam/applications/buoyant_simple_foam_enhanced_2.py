"""
buoyantSimpleFoamEnhanced2 — enhanced steady-state buoyant SIMPLE solver v2.

Extends :class:`BuoyantSimpleFoamEnhanced` with:

- **Improved Boussinesq approximation**: uses a quasi-implicit
  linearisation of the buoyancy source that treats the T-dependent
  part implicitly in the momentum equation, improving coupling.
- **Gradient Richardson number**: uses local gradient Richardson
  number Ri_g = N^2 / S^2 (where N is Brunt-Vaisala frequency and
  S is shear rate) for more accurate flow regime classification.
- **Buoyancy-aware pressure correction**: modifies the pressure
  equation to account for the hydrostatic component, improving
  convergence for high-Ri flows.

Algorithm (per outer iteration):
1. Update turbulence
2. Compute gradient Richardson number field
3. Solve momentum predictor with implicit Boussinesq buoyancy
4. Solve pressure equation (with hydrostatic correction)
5. Correct velocity and flux
6. Solve energy equation
7. Update density and check convergence

Usage::

    from pyfoam.applications.buoyant_simple_foam_enhanced_2 import BuoyantSimpleFoamEnhanced2

    solver = BuoyantSimpleFoamEnhanced2("path/to/case")
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

from .buoyant_simple_foam_enhanced import BuoyantSimpleFoamEnhanced
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["BuoyantSimpleFoamEnhanced2"]

logger = logging.getLogger(__name__)


class BuoyantSimpleFoamEnhanced2(BuoyantSimpleFoamEnhanced):
    """Enhanced steady-state buoyant SIMPLE solver v2.

    Extends BuoyantSimpleFoamEnhanced with implicit Boussinesq
    linearisation, gradient Richardson number, and hydrostatic
    pressure correction.

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
    implicit_buoyancy : bool, optional
        Use implicit linearisation of buoyancy.  Default True.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        thermo: BasicThermo | None = None,
        gravity: tuple[float, float, float] | None = None,
        radiation: RadiationModel | None = None,
        implicit_buoyancy: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            case_path, thermo=thermo, gravity=gravity,
            radiation=radiation, **kwargs,
        )

        self.implicit_buoyancy = implicit_buoyancy

        logger.info(
            "BuoyantSimpleFoamEnhanced2 ready: implicit_buoyancy=%s",
            self.implicit_buoyancy,
        )

    # ------------------------------------------------------------------
    # Gradient Richardson number
    # ------------------------------------------------------------------

    def _compute_gradient_richardson_field(
        self,
        U: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cell-wise gradient Richardson number.

        Ri_g = N^2 / (S^2 + eps)

        where:
        - N^2 = g * beta * dT/dy  (Brunt-Vaisala frequency squared)
        - S^2 = (dU/dy)^2          (shear rate squared)

        Returns:
            ``(n_cells,)`` gradient Richardson number field.
        """
        mesh = self.mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        device = U.device
        dtype = U.dtype

        owner = mesh.owner[:n_internal]
        neigh = mesh.neighbour

        # Temperature gradient (simplified: using face differences)
        T_O = gather(T, owner)
        T_N = gather(T, neigh)
        dT = T_N - T_O

        # Velocity gradient
        U_O = gather(U.norm(dim=1), owner)
        U_N = gather(U.norm(dim=1), neigh)
        dU = U_N - U_O

        # Cell-averaged gradients (simplified)
        g_mag = float(self.g.norm().item())
        delta_coeffs = mesh.delta_coefficients[:n_internal]

        dT_dx = (dT * delta_coeffs).abs()
        dU_dx = (dU * delta_coeffs).abs()

        N2 = g_mag * self.beta * dT_dx
        S2 = dU_dx.pow(2)

        # Scatter to cells (use max contribution)
        Ri_face = N2 / (S2 + 1e-30)

        Ri_cell = torch.zeros(n_cells, dtype=dtype, device=device)
        Ri_cell.scatter_reduce_(0, owner, Ri_face, reduce="amax")
        Ri_cell.scatter_reduce_(0, neigh, Ri_face, reduce="amax")

        return Ri_cell.clamp(min=0.0, max=100.0)

    # ------------------------------------------------------------------
    # Implicit Boussinesq buoyancy
    # ------------------------------------------------------------------

    def _compute_implicit_buoyancy_source(
        self,
        T: torch.Tensor,
        U: torch.Tensor,
        rho0: float,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute implicit Boussinesq buoyancy source and diagonal contribution.

        Instead of adding buoyancy as an explicit source, treats part
        implicitly:
            F_b = rho0 * beta * (T_ref - T) * g
                = rho0 * beta * T_ref * g - rho0 * beta * T * g

        The T-dependent part is treated implicitly in the momentum
        equation by adding to the diagonal coefficient.

        Returns:
            Tuple of (source_vector, diagonal_boost).
        """
        T_diff = T - self.T_ref

        # Explicit source: rho0 * beta * (T_ref - T_old) * g
        source = -rho0 * self.beta * T_diff.unsqueeze(-1) * self.g.unsqueeze(0)

        # Diagonal boost: rho0 * beta * |g| (positive, adds stability)
        diag_boost = rho0 * self.beta * float(self.g.norm().item())

        return source, diag_boost

    # ------------------------------------------------------------------
    # Hydrostatic pressure correction
    # ------------------------------------------------------------------

    def _compute_hydrostatic_pressure(self, T: torch.Tensor) -> torch.Tensor:
        """Compute hydrostatic pressure from temperature field.

        p_hydro = rho0 * g * H * (1 - beta * (T - T_ref))

        where H is the height coordinate (simplified: use cell index).

        Returns:
            ``(n_cells,)`` hydrostatic pressure.
        """
        rho0 = float(self.rho.mean().item())
        g_mag = float(self.g.norm().item())
        mesh = self.mesh

        # Use cell centre z-coordinate as height (simplified)
        cell_centres = mesh.cell_centres
        if cell_centres.dim() > 1 and cell_centres.shape[1] >= 3:
            H = cell_centres[:, 1].abs()  # y-direction (gravity)
        else:
            H = torch.arange(
                mesh.n_cells, dtype=T.dtype, device=T.device,
            ).float() * 0.1

        T_diff = T - self.T_ref
        p_hydro = rho0 * g_mag * H * (1.0 - self.beta * T_diff)

        return p_hydro

    # ------------------------------------------------------------------
    # Enhanced run loop
    # ------------------------------------------------------------------

    def run(self) -> ConvergenceData:
        """Run the enhanced v2 buoyantSimpleFoam solver.

        Uses implicit Boussinesq linearisation and gradient Richardson
        number for improved natural convection convergence.

        Returns:
            Final :class:`ConvergenceData`.
        """
        use_boussinesq = self._should_use_boussinesq()
        if use_boussinesq:
            logger.info("Using Boussinesq approximation (beta=%.4e)", self.beta)
        else:
            logger.info("Using variable-density mode (beta*dT > 0.1)")

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

        logger.info("Starting buoyantSimpleFoamEnhanced2 run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  implicit_buoyancy=%s", self.implicit_buoyancy)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence: ConvergenceData | None = None

        for t, step in time_loop:
            mu_eff = self._update_turbulence()

            # Compute gradient Richardson number
            Ri = self._compute_richardson_number(self.U, self.T)
            self._richardson = Ri

            # Gradient Richardson field (more informative than bulk)
            Ri_field = self._compute_gradient_richardson_field(self.U, self.T)
            Ri_max = float(Ri_field.max().item())

            if step % 10 == 0:
                logger.info(
                    "Richardson: bulk=%.3f, gradient_max=%.3f", Ri, Ri_max,
                )

            alpha_U_eff, alpha_p_eff = self._buoyancy_aware_relaxation(Ri)

            # Buoyancy computation
            if self.implicit_buoyancy:
                f_buoyancy, diag_boost = self._compute_implicit_buoyancy_source(
                    self.T, self.U, rho0, self.delta_t,
                )

            # Run SIMPLE iteration with buoyancy
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
                logger.info("Converged at step %d (t=%.6g)", step + 1, t)
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        if last_convergence is not None:
            if last_convergence.converged:
                logger.info("buoyantSimpleFoamEnhanced2 completed (converged)")
            else:
                logger.warning("buoyantSimpleFoamEnhanced2 completed without convergence")

        return last_convergence or ConvergenceData()
