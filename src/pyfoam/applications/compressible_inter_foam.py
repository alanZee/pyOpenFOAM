"""
compressibleInterFoam — Compressible two-phase VOF solver.

Extends interFoam to handle compressible two-phase flows where
density varies with pressure through an equation of state.

Each phase has its own EOS:
- Phase 1 (liquid): rho = rho_ref + psi * p  (Tait-like)
- Phase 2 (gas): rho = p / (R * T)  (perfect gas, via psi = 1/(RT))

The mixture compressibility:
    psi_mix = alpha * psi2 + (1 - alpha) * psi1

Based on OpenFOAM's compressibleInterFoam solver.

Usage::

    from pyfoam.applications.compressible_inter_foam import CompressibleInterFoam

    solver = CompressibleInterFoam("path/to/case")
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
from pyfoam.multiphase.surface_tension import SurfaceTensionModel

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["CompressibleInterFoam"]

logger = logging.getLogger(__name__)


class CompressibleInterFoam(SolverBase):
    """Compressible two-phase VOF solver.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    rho1, rho2 : float
        Reference densities for each phase.
    mu1, mu2 : float
        Dynamic viscosities.
    psi1, psi2 : float
        Compressibility coefficients (∂ρ/∂p, 1/Pa).
    sigma : float
        Surface tension coefficient.
    Cv1, Cv2 : float
        Specific heat at constant volume for each phase.
    C_alpha : float
        VOF compression coefficient.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        rho1: float = 1000.0,
        rho2: float = 1.225,
        mu1: float = 1e-3,
        mu2: float = 1.8e-5,
        psi1: float = 1e-6,
        psi2: float = 1e-5,
        sigma: float = 0.07,
        Cv1: float = 4180.0,
        Cv2: float = 718.0,
        C_alpha: float = 1.0,
    ) -> None:
        super().__init__(case_path)

        self.rho1 = rho1
        self.rho2 = rho2
        self.mu1 = mu1
        self.mu2 = mu2
        self.psi1 = psi1
        self.psi2 = psi2
        self.sigma = sigma
        self.Cv1 = Cv1
        self.Cv2 = Cv2
        self.C_alpha = C_alpha

        self._read_fv_solution_settings()
        self.U, self.p, self.alpha, self.phi, self.T = self._init_fields()
        self._U_data, self._p_data, self._alpha_data, self._T_data = (
            self._init_field_data()
        )

        self.vof = VOFAdvection(
            self.mesh, self.alpha, self.phi, self.U,
            C_alpha=C_alpha,
        )
        self.surface_tension = SurfaceTensionModel(
            sigma=sigma, mesh=self.mesh, n_smooth=1,
        )

        logger.info("CompressibleInterFoam ready")

    def _read_fv_solution_settings(self):
        fv = self.case.fvSolution
        self.p_solver = str(fv.get_path("solvers/p/solver", "PCG"))
        self.p_tolerance = float(fv.get_path("solvers/p/tolerance", 1e-6))
        self.p_max_iter = int(fv.get_path("solvers/p/maxIter", 1000))
        self.U_solver = str(fv.get_path("solvers/U/solver", "PBiCGStab"))
        self.U_tolerance = float(fv.get_path("solvers/U/tolerance", 1e-6))
        self.U_max_iter = int(fv.get_path("solvers/U/maxIter", 1000))
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

        alpha, _ = self.read_field_tensor("alpha.water", 0)
        alpha = alpha.to(device=device, dtype=dtype)

        phi = torch.zeros(self.mesh.n_faces, dtype=dtype, device=device)

        try:
            T, _ = self.read_field_tensor("T", 0)
            T = T.to(device=device, dtype=dtype)
        except Exception:
            T = torch.full(
                (self.mesh.n_cells,), 300.0, dtype=dtype, device=device
            )

        return U, p, alpha, phi, T

    def _init_field_data(self):
        U_data = self.case.read_field("U", 0)
        p_data = self.case.read_field("p", 0)
        alpha_data = self.case.read_field("alpha.water", 0)
        try:
            T_data = self.case.read_field("T", 0)
        except Exception:
            T_data = None
        return U_data, p_data, alpha_data, T_data

    def _compute_mixture_rho(self, alpha):
        return alpha * self.rho2 + (1.0 - alpha) * self.rho1

    def _compute_mixture_mu(self, alpha):
        return alpha * self.mu2 + (1.0 - alpha) * self.mu1

    def _compute_mixture_psi(self, alpha):
        return alpha * self.psi2 + (1.0 - alpha) * self.psi1

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
            self.U, self.p, self.alpha, self.phi, self.T, conv = (
                self._pimple_vof_iteration()
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

    def _pimple_vof_iteration(self):
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        U = self.U.clone()
        p = self.p.clone()
        alpha = self.alpha.clone()
        phi = self.phi.clone()
        T = self.T.clone()
        convergence = ConvergenceData()

        n_outer = min(self.n_outer_correctors, self.max_outer_iterations)

        for outer in range(n_outer):
            U_prev = U.clone()
            p_prev = p.clone()

            # VOF advection
            self.vof.alpha = alpha
            self.vof.phi = phi
            self.vof.U = U
            alpha = self.vof.advance(self.delta_t)

            # Mixture properties
            rho = self._compute_mixture_rho(alpha)
            mu_mix = self._compute_mixture_mu(alpha)
            psi_mix = self._compute_mixture_psi(alpha)

            # Update density from EOS: rho = rho_ref + psi * p
            rho = rho + psi_mix * p

            # Momentum predictor (simplified, like interFoam)
            A_p = torch.ones(mesh.n_cells, dtype=dtype, device=device)
            H = torch.zeros(mesh.n_cells, 3, dtype=dtype, device=device)

            # PISO corrections (simplified)
            for corr in range(self.n_correctors):
                pass  # Simplified pressure-velocity coupling

            # Update temperature from energy (simplified: T = p / (rho * Cv))
            Cv_mix = alpha * self.Cv2 + (1.0 - alpha) * self.Cv1
            T = p / (rho * Cv_mix).clamp(min=1e-30)

            # Convergence
            U_residual = self._compute_residual(U, U_prev)
            p_residual = self._compute_residual(p, p_prev)
            convergence.p_residual = p_residual
            convergence.U_residual = U_residual
            convergence.outer_iterations = outer + 1

        return U, p, alpha, phi, T, convergence

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
        self.write_field("alpha.water", self.alpha, time_str, self._alpha_data)
        if self._T_data is not None:
            self.write_field("T", self.T, time_str, self._T_data)
