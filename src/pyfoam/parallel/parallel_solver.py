"""
Parallel solver wrapper — domain-decomposed linear solver.

Wraps existing linear solvers (PCG, PBiCGStab, GAMG) for parallel execution
using domain decomposition with additive Schwarz or block-Jacobi iteration.

Strategy
--------
1. Each processor solves its local subdomain using the serial solver.
2. Ghost cell values are updated via halo exchange after each iteration.
3. The process repeats until global convergence.

This is a **loosely-coupled** parallel solver — it does not assemble a
global matrix.  Instead, it iterates on local matrices with boundary
conditions updated from neighbours.

Usage::

    from pyfoam.parallel.parallel_solver import ParallelSolver

    psolver = ParallelSolver(local_solver, halo_exchange, subdomain)
    x, iters, residual = psolver.solve(local_matrix, source, x0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.ldu_matrix import LduMatrix
from pyfoam.core.fv_matrix import FvMatrix
from pyfoam.parallel.decomposition import SubDomain
from pyfoam.parallel.processor_patch import HaloExchange

# Try to import mpi4py
try:
    from mpi4py import MPI as _MPI
    _MPI_AVAILABLE = True
except ImportError:
    _MPI = None  # type: ignore[assignment]
    _MPI_AVAILABLE = False


__all__ = ["ParallelSolver", "ParallelSolverConfig"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ParallelSolverConfig:
    """Configuration for the parallel solver.

    Attributes
    ----------
    max_outer_iterations : int
        Maximum number of outer (halo exchange) iterations.
    outer_tolerance : float
        Convergence tolerance for the outer iteration.
    update_halos_every : int
        Update ghost cells every N inner iterations.
    """

    max_outer_iterations: int = 100
    outer_tolerance: float = 1e-6
    update_halos_every: int = 1


# ---------------------------------------------------------------------------
# ParallelSolver
# ---------------------------------------------------------------------------


class ParallelSolver:
    """Parallel solver using domain decomposition.

    Wraps a serial linear solver and adds inter-processor communication
    via halo exchange.

    Parameters
    ----------
    local_solver : LinearSolver
        The serial linear solver (PCG, PBiCGStab, etc.).
    halo : HaloExchange
        Halo exchange manager.
    subdomain : SubDomain
        The subdomain for this processor.
    config : ParallelSolverConfig, optional
        Solver configuration.
    comm : mpi4py.MPI.Intracomm, optional
        MPI communicator.
    """

    def __init__(
        self,
        local_solver: Any,
        halo: HaloExchange,
        subdomain: SubDomain,
        config: ParallelSolverConfig | None = None,
        comm: object | None = None,
    ) -> None:
        self._local_solver = local_solver
        self._halo = halo
        self._subdomain = subdomain
        self._config = config or ParallelSolverConfig()
        self._device = get_device()
        self._dtype = get_default_dtype()

        if _MPI_AVAILABLE and comm is not None:
            self._comm = comm
        elif _MPI_AVAILABLE:
            self._comm = _MPI.COMM_WORLD  # type: ignore[union-attr]
        else:
            self._comm = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def rank(self) -> int:
        if self._comm is not None:
            return self._comm.Get_rank()  # type: ignore[union-attr]
        return 0

    @property
    def config(self) -> ParallelSolverConfig:
        return self._config

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        matrix: LduMatrix | FvMatrix,
        source: torch.Tensor,
        x0: torch.Tensor,
        tolerance: float = 1e-6,
        max_iter: int = 1000,
    ) -> tuple[torch.Tensor, int, float]:
        """Solve the parallel system A x = b.

        Uses additive Schwarz iteration:
        1. Solve local subdomain.
        2. Exchange ghost cell values.
        3. Repeat until global convergence.

        Args:
            matrix: Local LDU matrix.
            source: Local source vector.
            x0: Initial guess.
            tolerance: Convergence tolerance.
            max_iter: Maximum iterations per local solve.

        Returns:
            Tuple of ``(solution, total_iterations, final_residual)``.
        """
        x = x0.clone()
        n_owned = self._subdomain.n_owned_cells
        total_iters = 0
        global_residual = float("inf")

        for outer in range(self._config.max_outer_iterations):
            # Update ghost cells before solving
            if outer > 0:
                self._halo.exchange(x)

            # Solve locally
            x, local_iters, local_residual = self._local_solver(
                matrix, source, x, tolerance, max_iter
            )
            total_iters += local_iters

            # Check global convergence
            global_residual = self._global_residual(local_residual)
            if global_residual < self._config.outer_tolerance:
                break

        return x, total_iters, global_residual

    def solve_fv_matrix(
        self,
        fv_matrix: FvMatrix,
        x0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int, float]:
        """Solve an FvMatrix in parallel.

        Convenience method that extracts source and calls :meth:`solve`.

        Args:
            fv_matrix: The FvMatrix to solve.
            x0: Initial guess.  Defaults to zeros.

        Returns:
            Tuple of ``(solution, iterations, residual)``.
        """
        if x0 is None:
            x0 = torch.zeros(
                fv_matrix.n_cells, device=self._device, dtype=self._dtype
            )

        # Extract the LDU matrix and source
        ldu = fv_matrix
        source = fv_matrix.source

        return self.solve(ldu, source, x0)

    # ------------------------------------------------------------------
    # Global assembly
    # ------------------------------------------------------------------

    def assemble_global_rhs(
        self,
        local_rhs: torch.Tensor,
    ) -> torch.Tensor | None:
        """Assemble global RHS vector from local contributions.

        Each processor contributes its owned cell values.  The result
        is gathered onto rank 0.

        Args:
            local_rhs: Local RHS vector ``(n_local_cells,)``.

        Returns:
            On rank 0: global RHS ``(n_global_cells,)``.
            On other ranks: ``None``.
        """
        n_owned = self._subdomain.n_owned_cells
        owned_rhs = local_rhs[:n_owned].detach().cpu().numpy()

        if self._comm is not None and _MPI_AVAILABLE:
            gathered = self._comm.gather(owned_rhs, root=0)  # type: ignore[union-attr]
            if self.rank == 0:
                import numpy as np
                global_np = np.concatenate(gathered)
                return torch.from_numpy(global_np).to(
                    device=self._device, dtype=self._dtype
                )
            return None
        else:
            return local_rhs[:n_owned].clone()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _global_residual(self, local_residual: float) -> float:
        """Compute global residual via MPI allreduce (max norm)."""
        if self._comm is not None and _MPI_AVAILABLE:
            local_np = torch.tensor(local_residual, dtype=self._dtype).numpy()
            global_np = self._comm.allreduce(local_np, op=_MPI.MAX)  # type: ignore[union-attr]
            return float(global_np)
        return local_residual

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ParallelSolver(rank={self.rank}, "
            f"local_solver={self._local_solver}, "
            f"subdomain={self._subdomain.processor_id})"
        )
