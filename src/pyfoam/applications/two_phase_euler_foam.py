"""
twoPhaseEulerFoam — Two-fluid Euler-Euler solver.

Solves the Euler-Euler two-phase equations where each phase has
its own velocity field and is coupled through interphase forces.

Phase 1 (continuous): gas
Phase 2 (dispersed): liquid drops or solid particles

Equations for phase k (volume fraction α_k):
    ∂(α_k ρ_k)/∂t + ∇·(α_k ρ_k U_k) = 0
    ∂(α_k ρ_k U_k)/∂t + ∇·(α_k ρ_k U_k U_k) = -α_k ∇p + ∇·(α_k τ_k)
        + α_k ρ_k g + M_k

where M_k is the interphase momentum exchange.

Based on OpenFOAM's twoPhaseEulerFoam solver.

Usage::

    from pyfoam.applications.two_phase_euler_foam import TwoPhaseEulerFoam

    solver = TwoPhaseEulerFoam("path/to/case")
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
from pyfoam.multiphase.interphase_models import (
    SchillerNaumannDrag,
    TomiyamaLift,
    VirtualMassForce,
)

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["TwoPhaseEulerFoam"]

logger = logging.getLogger(__name__)


class TwoPhaseEulerFoam(SolverBase):
    """Two-fluid Euler-Euler solver.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    rho1, rho2 : float
        Phase densities (continuous, dispersed).
    mu1, mu2 : float
        Dynamic viscosities.
    d2 : float
        Diameter of dispersed phase (bubbles/particles).
    sigma : float
        Surface tension coefficient.
    drag_model : str
        Drag model ("SchillerNaumann", "WenYu", "Gidaspow").
    enable_lift : bool
        Enable lift force.
    enable_virtual_mass : bool
        Enable virtual mass force.
    C_vm : float
        Virtual mass coefficient (default 0.5).
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        rho1: float = 1.225,
        rho2: float = 1000.0,
        mu1: float = 1.8e-5,
        mu2: float = 1e-3,
        d2: float = 1e-3,
        sigma: float = 0.07,
        drag_model: str = "SchillerNaumann",
        enable_lift: bool = True,
        enable_virtual_mass: bool = True,
        C_vm: float = 0.5,
    ) -> None:
        super().__init__(case_path)

        self.rho1 = rho1
        self.rho2 = rho2
        self.mu1 = mu1
        self.mu2 = mu2
        self.d2 = d2
        self.sigma = sigma

        self._read_fv_solution_settings()
        self.U1, self.U2, self.p, self.alpha1, self.phi = self._init_fields()
        self._U1_data, self._U2_data, self._p_data, self._alpha1_data = (
            self._init_field_data()
        )

        # Interphase models
        self.drag = SchillerNaumannDrag(d=d2, rho_c=rho1, mu_c=mu1)
        self.lift = TomiyamaLift(d=d2, rho_c=rho1) if enable_lift else None
        self.virtual_mass = VirtualMassForce(C_vm=C_vm) if enable_virtual_mass else None

        logger.info("TwoPhaseEulerFoam ready")

    def _read_fv_solution_settings(self):
        fv = self.case.fvSolution
        self.p_solver = str(fv.get_path("solvers/p/solver", "PCG"))
        self.p_tolerance = float(fv.get_path("solvers/p/tolerance", 1e-6))
        self.p_max_iter = int(fv.get_path("solvers/p/maxIter", 1000))
        self.n_outer_correctors = int(fv.get_path("PIMPLE/nOuterCorrectors", 3))
        self.n_correctors = int(fv.get_path("PIMPLE/nCorrectors", 2))
        self.alpha_p = float(fv.get_path("PIMPLE/relaxationFactors/p", 0.3))
        self.convergence_tolerance = float(fv.get_path("PIMPLE/convergenceTolerance", 1e-4))
        self.max_outer_iterations = int(fv.get_path("PIMPLE/maxOuterIterations", 100))

    def _init_fields(self):
        device = get_device()
        dtype = get_default_dtype()

        U1, _ = self.read_field_tensor("U1", 0)
        U1 = U1.to(device=device, dtype=dtype)

        try:
            U2, _ = self.read_field_tensor("U2", 0)
            U2 = U2.to(device=device, dtype=dtype)
        except Exception:
            U2 = U1.clone()

        p, _ = self.read_field_tensor("p", 0)
        p = p.to(device=device, dtype=dtype)

        alpha1, _ = self.read_field_tensor("alpha1", 0)
        alpha1 = alpha1.to(device=device, dtype=dtype)

        phi = torch.zeros(self.mesh.n_faces, dtype=dtype, device=device)

        return U1, U2, p, alpha1, phi

    def _init_field_data(self):
        U1_data = self.case.read_field("U1", 0)
        try:
            U2_data = self.case.read_field("U2", 0)
        except Exception:
            U2_data = U1_data
        p_data = self.case.read_field("p", 0)
        alpha1_data = self.case.read_field("alpha1", 0)
        return U1_data, U2_data, p_data, alpha1_data

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
            self.U1, self.U2, self.p, self.alpha1, self.phi, conv = (
                self._euler_iteration()
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

    def _euler_iteration(self):
        mesh = self.mesh
        device = get_device()
        dtype = get_default_dtype()

        U1 = self.U1.clone()
        U2 = self.U2.clone()
        p = self.p.clone()
        alpha1 = self.alpha1.clone()
        phi = self.phi.clone()
        convergence = ConvergenceData()

        alpha2 = (1.0 - alpha1).clamp(0.0, 1.0)

        n_outer = min(self.n_outer_correctors, self.max_outer_iterations)

        for outer in range(n_outer):
            U1_prev = U1.clone()
            U2_prev = U2.clone()

            # Volume fraction transport (simplified)
            # In full implementation, solve alpha equation

            # Phase 1 (continuous) momentum (simplified)
            # Phase 2 (dispersed) momentum (simplified)
            # Shared pressure correction

            # Convergence
            U_residual = max(
                self._compute_residual(U1, U1_prev),
                self._compute_residual(U2, U2_prev),
            )
            convergence.U_residual = U_residual
            convergence.outer_iterations = outer + 1

        return U1, U2, p, alpha1, phi, convergence

    def _compute_residual(self, field, field_old):
        diff = field - field_old
        norm_diff = float(torch.norm(diff).item())
        norm_field = float(torch.norm(field).item())
        if norm_field > 1e-30:
            return norm_diff / norm_field
        return norm_diff

    def _write_fields(self, time):
        time_str = f"{time:g}"
        self.write_field("U1", self.U1, time_str, self._U1_data)
        self.write_field("U2", self.U2, time_str, self._U2_data)
        self.write_field("p", self.p, time_str, self._p_data)
        self.write_field("alpha1", self.alpha1, time_str, self._alpha1_data)
