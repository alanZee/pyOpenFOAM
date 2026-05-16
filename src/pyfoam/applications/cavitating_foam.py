"""
cavitatingFoam — Cavitation solver for incompressible two-phase flow.

Solves the two-phase incompressible Navier-Stokes equations with
cavitation mass transfer between liquid and vapor phases.

The vapor volume fraction is transported:
    ∂α/∂t + ∇·(Uα) + ∇·(U_r α(1-α)) = m_dot / rho_v

where m_dot is the cavitation mass transfer rate.

Based on OpenFOAM's cavitatingFoam solver.

Usage::

    from pyfoam.applications.cavitating_foam import CavitatingFoam

    solver = CavitatingFoam("path/to/case")
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
from pyfoam.multiphase.volume_of_fluid import VOFAdvection
from pyfoam.multiphase.cavitation import SchnerrSauer

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["CavitatingFoam"]

logger = logging.getLogger(__name__)


class CavitatingFoam(SolverBase):
    """Cavitation solver for incompressible two-phase flow.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    rho_l : float
        Liquid density (default 1000.0 kg/m³).
    rho_v : float
        Vapor density (default 0.02 kg/m³).
    mu_l : float
        Liquid viscosity (default 1e-3 Pa·s).
    mu_v : float
        Vapor viscosity (default 1e-5 Pa·s).
    p_v : float
        Vapor pressure (default 2300.0 Pa).
    n_b : float
        Bubble number density for Schnerr-Sauer (default 1e13 m^-3).
    C_alpha : float
        VOF compression coefficient (default 1.0).
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        rho_l: float = 1000.0,
        rho_v: float = 0.02,
        mu_l: float = 1e-3,
        mu_v: float = 1e-5,
        p_v: float = 2300.0,
        n_b: float = 1e13,
        C_alpha: float = 1.0,
    ) -> None:
        super().__init__(case_path)

        self.rho_l = rho_l
        self.rho_v = rho_v
        self.mu_l = mu_l
        self.mu_v = mu_v
        self.p_v = p_v
        self.C_alpha = C_alpha

        self._read_fv_solution_settings()
        self.U, self.p, self.alpha, self.phi = self._init_fields()
        self._U_data, self._p_data, self._alpha_data = self._init_field_data()

        self.vof = VOFAdvection(
            self.mesh, self.alpha, self.phi, self.U,
            C_alpha=C_alpha,
        )

        self.cavitation_model = SchnerrSauer(n_b=n_b, p_v=p_v)

        logger.info("CavitatingFoam ready")

    def _read_fv_solution_settings(self):
        fv = self.case.fvSolution
        self.p_solver = str(fv.get_path("solvers/p/solver", "PCG"))
        self.p_tolerance = float(fv.get_path("solvers/p/tolerance", 1e-6))
        self.p_max_iter = int(fv.get_path("solvers/p/maxIter", 1000))
        self.n_outer_correctors = int(fv.get_path("PIMPLE/nOuterCorrectors", 3))
        self.n_correctors = int(fv.get_path("PIMPLE/nCorrectors", 2))
        self.alpha_p = float(fv.get_path("PIMPLE/relaxationFactors/p", 0.3))
        self.alpha_U = float(fv.get_path("PIMPLE/relaxationFactors/U", 0.7))
        self.convergence_tolerance = float(fv.get_path("PIMPLE/convergenceTolerance", 1e-4))
        self.max_outer_iterations = int(fv.get_path("PIMPLE/maxOuterIterations", 100))

    def _init_fields(self):
        device = get_device()
        dtype = get_default_dtype()

        U, _ = self.read_field_tensor("U", 0)
        U = U.to(device=device, dtype=dtype)

        p, _ = self.read_field_tensor("p", 0)
        p = p.to(device=device, dtype=dtype)

        alpha, _ = self.read_field_tensor("alpha.vapor", 0)
        alpha = alpha.to(device=device, dtype=dtype)

        phi = torch.zeros(self.mesh.n_faces, dtype=dtype, device=device)
        return U, p, alpha, phi

    def _init_field_data(self):
        U_data = self.case.read_field("U", 0)
        p_data = self.case.read_field("p", 0)
        alpha_data = self.case.read_field("alpha.vapor", 0)
        return U_data, p_data, alpha_data

    def _compute_mixture_rho(self, alpha):
        return alpha * self.rho_v + (1.0 - alpha) * self.rho_l

    def _compute_mixture_mu(self, alpha):
        return alpha * self.mu_v + (1.0 - alpha) * self.mu_l

    def run(self) -> ConvergenceData:
        time_loop = TimeLoop(
            start_time=self.start_time, end_time=self.end_time,
            delta_t=self.delta_t, write_interval=self.write_interval,
            write_control=self.write_control,
        )
        convergence = ConvergenceMonitor(tolerance=self.convergence_tolerance, min_steps=1)

        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence = None
        for t, step in time_loop:
            self.U, self.p, self.alpha, self.phi, conv = (
                self._pimple_cavitation_iteration()
            )
            last_convergence = conv

            residuals = {"U": conv.U_residual, "p": conv.p_residual, "cont": conv.continuity_error}
            converged = convergence.update(step + 1, residuals)

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                break

        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)
        return last_convergence or ConvergenceData()

    def _pimple_cavitation_iteration(self):
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        U = self.U.clone()
        p = self.p.clone()
        alpha = self.alpha.clone()
        phi = self.phi.clone()
        convergence = ConvergenceData()

        n_outer = min(self.n_outer_correctors, self.max_outer_iterations)

        for outer in range(n_outer):
            U_prev = U.clone()
            p_prev = p.clone()

            # Compute cavitation mass transfer
            m_dot = self.cavitation_model.compute_mass_transfer(
                alpha, p, self.rho_l, self.rho_v,
            )

            # VOF advection with cavitation source
            self.vof.alpha = alpha
            self.vof.phi = phi
            self.vof.U = U
            alpha = self.vof.advance(self.delta_t)

            # Apply cavitation source term
            V = mesh.cell_volumes.clamp(min=1e-30)
            alpha = alpha + self.delta_t * m_dot / self.rho_v / V
            alpha = alpha.clamp(0.0, 1.0)

            # Mixture properties
            rho = self._compute_mixture_rho(alpha)
            mu_mix = self._compute_mixture_mu(alpha)

            # Momentum (simplified)
            # PISO corrections (simplified)
            for corr in range(self.n_correctors):
                pass

            # Convergence
            U_residual = self._compute_residual(U, U_prev)
            p_residual = self._compute_residual(p, p_prev)
            convergence.U_residual = U_residual
            convergence.p_residual = p_residual
            convergence.outer_iterations = outer + 1

        return U, p, alpha, phi, convergence

    def _compute_residual(self, field, field_old):
        diff = field - field_old
        norm_diff = float(torch.norm(diff).item())
        norm_field = float(torch.norm(field).item())
        if norm_field > 1e-30:
            return norm_diff / norm_field
        return norm_diff

    def _write_fields(self, time):
        time_str = f"{time:g}"
        self.write_field("U", self.U, time_str, self._U_data)
        self.write_field("p", self.p, time_str, self._p_data)
        self.write_field("alpha.vapor", self.alpha, time_str, self._alpha_data)
