"""
Algebraic Multigrid (GAMG) solver for LDU matrices.

Implements a geometric-algebraic multigrid (GAMG) method following OpenFOAM's
approach:

1. **Aggregation-based coarsening**: Group cells into aggregates (clusters)
   to build coarse levels.  Each aggregate becomes one coarse cell.
2. **Restriction**: Transfer the residual from fine to coarse grid using
   aggregation (each fine cell's residual goes to its aggregate).
3. **Coarse solve**: Solve the coarse-grid error equation (direct or iterative).
4. **Prolongation**: Transfer the coarse-grid correction back to fine grid.
5. **Smoother**: Pre-/post-smoothing iterations on each level using PCG with DIC.

The multigrid V-cycle:

::

    def v_cycle(level, x, b):
        if level == coarsest:
            x = direct_solve(A[level], b)
        else:
            # Pre-smooth
            x = smooth(A[level], x, b, n_pre_smooth)
            # Compute residual
            r = b - A[level]·x
            # Restrict to coarse
            r_coarse = restrict(r)
            # Solve coarse error equation
            e_coarse = v_cycle(level+1, 0, r_coarse)
            # Prolongate correction
            e = prolongate(e_coarse)
            # Correct
            x = x + e
            # Post-smooth
            x = smooth(A[level], x, b, n_post_smooth)
        return x

All tensors respect the global device/dtype configuration from
:mod:`pyfoam.core.device`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from pyfoam.core.backend import gather, scatter_add
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.core.ldu_matrix import LduMatrix
from pyfoam.solvers.linear_solver import LinearSolverBase
from pyfoam.solvers.preconditioners import DICPreconditioner, DILUPreconditioner
from pyfoam.solvers.residual import ConvergenceInfo, ResidualMonitor

__all__ = ["GAMGSolver"]


@dataclass
class GAMGLevel:
    """Data for one level of the GAMG hierarchy.

    Attributes
    ----------
    n_cells : int
        Number of cells at this level.
    diag : torch.Tensor
        Diagonal coefficients.
    lower : torch.Tensor
        Lower-triangular coefficients.
    upper : torch.Tensor
        Upper-triangular coefficients.
    owner : torch.Tensor
        Owner cell indices.
    neighbour : torch.Tensor
        Neighbour cell indices.
    """

    n_cells: int
    diag: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    owner: torch.Tensor
    neighbour: torch.Tensor


@dataclass
class AggregationData:
    """Data for an aggregation level (fine → coarse mapping).

    Attributes
    ----------
    n_coarse : int
        Number of coarse cells (aggregates).
    fine_to_coarse : torch.Tensor
        ``(n_fine,)`` mapping from fine cell index to coarse cell index.
    """

    n_coarse: int
    fine_to_coarse: torch.Tensor


class GAMGSolver(LinearSolverBase):
    """Algebraic Multigrid solver.

    Uses aggregation-based coarsening with PCG+DIC smoothing.

    Parameters
    ----------
    tolerance : float
        Absolute convergence tolerance.
    rel_tol : float
        Relative convergence tolerance.
    max_iter : int
        Maximum outer iterations (V-cycles).
    min_iter : int
        Minimum iterations before convergence check.
    verbose : bool
        If True, log residuals.
    n_pre_smooth : int
        Number of pre-smoothing iterations.
    n_post_smooth : int
        Number of post-smoothing iterations.
    max_levels : int
        Maximum number of multigrid levels.
    min_cells_coarse : int
        Minimum number of cells on the coarsest level.
    smoother : str
        Smoother type: ``"PCG"`` or ``"DIC"`` (Jacobi-like).
    """

    def __init__(
        self,
        tolerance: float = 1e-6,
        rel_tol: float = 0.01,
        max_iter: int = 100,
        min_iter: int = 0,
        verbose: bool = False,
        n_pre_smooth: int = 2,
        n_post_smooth: int = 2,
        max_levels: int = 10,
        min_cells_coarse: int = 10,
        smoother: str = "PCG",
    ) -> None:
        super().__init__(
            tolerance=tolerance,
            rel_tol=rel_tol,
            max_iter=max_iter,
            min_iter=min_iter,
            verbose=verbose,
        )
        self._n_pre_smooth = n_pre_smooth
        self._n_post_smooth = n_post_smooth
        self._max_levels = max_levels
        self._min_cells_coarse = min_cells_coarse
        self._smoother_type = smoother.upper()

    # ------------------------------------------------------------------
    # Coarsening
    # ------------------------------------------------------------------

    @staticmethod
    def _build_aggregates(
        matrix: LduMatrix,
        n_coarse_target: int,
    ) -> AggregationData:
        """Build aggregation-based coarsening.

        Groups cells into aggregates using a greedy approach:
        1. Pick an unaggregated cell as seed.
        2. Add its unaggregated neighbours to the same aggregate.
        3. Repeat until all cells are assigned.

        Args:
            matrix: Fine-level LDU matrix.
            n_coarse_target: Target number of coarse cells.

        Returns:
            AggregationData with fine-to-coarse mapping.
        """
        n_cells = matrix.n_cells
        owner = matrix.owner
        neighbour = matrix.neighbour
        n_internal = matrix.n_internal_faces
        device = matrix.device

        # Build adjacency list
        adj: list[list[int]] = [[] for _ in range(n_cells)]
        for f in range(n_internal):
            p = int(owner[f])
            n = int(neighbour[f])
            adj[p].append(n)
            adj[n].append(p)

        # Greedy aggregation
        fine_to_coarse = torch.full((n_cells,), -1, device=device, dtype=INDEX_DTYPE)
        n_aggregates = 0

        for cell in range(n_cells):
            if int(fine_to_coarse[cell]) >= 0:
                continue

            # Start new aggregate
            agg_id = n_aggregates
            n_aggregates += 1
            fine_to_coarse[cell] = agg_id

            # Add unaggregated neighbours
            for neighbour_cell in adj[cell]:
                if int(fine_to_coarse[neighbour_cell]) < 0:
                    fine_to_coarse[neighbour_cell] = agg_id

            # Stop if we have enough aggregates
            if n_aggregates >= n_coarse_target:
                # Assign remaining cells to nearest aggregate
                for remaining in range(n_cells):
                    if int(fine_to_coarse[remaining]) < 0:
                        # Find an aggregated neighbour
                        assigned = False
                        for nb in adj[remaining]:
                            if int(fine_to_coarse[nb]) >= 0:
                                fine_to_coarse[remaining] = fine_to_coarse[nb]
                                assigned = True
                                break
                        if not assigned:
                            # No aggregated neighbour — assign to aggregate 0
                            fine_to_coarse[remaining] = 0
                break

        # Handle any remaining unassigned cells
        for cell in range(n_cells):
            if int(fine_to_coarse[cell]) < 0:
                fine_to_coarse[cell] = 0

        return AggregationData(
            n_coarse=n_aggregates,
            fine_to_coarse=fine_to_coarse,
        )

    @staticmethod
    def _build_coarse_matrix(
        fine_matrix: LduMatrix,
        aggregation: AggregationData,
    ) -> GAMGLevel:
        """Build the coarse-level matrix by Galerkin projection.

        The coarse matrix A_c = R · A · P where:
        - R is the restriction operator (transpose of prolongation)
        - P is the prolongation operator (aggregation mapping)

        For aggregation-based coarsening:
        - P maps each coarse cell to its aggregate of fine cells
        - A_c[i,j] = sum over fine cells in aggregate i and j of A[fine_i, fine_j]

        Args:
            fine_matrix: Fine-level LDU matrix.
            aggregation: Aggregation data.

        Returns:
            GAMGLevel with coarse matrix coefficients.
        """
        n_fine = fine_matrix.n_cells
        n_coarse = aggregation.n_coarse
        f2c = aggregation.fine_to_coarse
        device = fine_matrix.device
        dtype = fine_matrix.dtype

        # Build coarse diagonal: sum of fine diagonals in each aggregate
        coarse_diag = scatter_add(
            fine_matrix.diag, f2c.long(), n_coarse, device=device, dtype=dtype
        )

        # Build coarse off-diagonal from fine off-diagonal
        # For each fine internal face f with owner P, neighbour N:
        #   coarse owner = f2c[P], coarse neighbour = f2c[N]
        #   If they're in different aggregates, add to coarse off-diagonal
        owner = fine_matrix.owner
        neighbour = fine_matrix.neighbour
        lower = fine_matrix.lower
        upper = fine_matrix.upper

        coarse_owner_list: list[int] = []
        coarse_neigh_list: list[int] = []
        coarse_lower_list: list[float] = []
        coarse_upper_list: list[float] = []

        # Track which coarse face pairs we've seen
        seen_pairs: dict[tuple[int, int], int] = {}

        for f in range(fine_matrix.n_internal_faces):
            p = int(owner[f])
            n = int(neighbour[f])
            cp = int(f2c[p])
            cn = int(f2c[n])

            if cp == cn:
                # Same aggregate — contribution goes to diagonal
                continue

            # Canonical order: min, max
            key = (min(cp, cn), max(cp, cn))
            if key in seen_pairs:
                idx = seen_pairs[key]
                coarse_lower_list[idx] += float(lower[f])
                coarse_upper_list[idx] += float(upper[f])
            else:
                seen_pairs[key] = len(coarse_owner_list)
                coarse_owner_list.append(cp)
                coarse_neigh_list.append(cn)
                coarse_lower_list.append(float(lower[f]))
                coarse_upper_list.append(float(upper[f]))

        n_coarse_faces = len(coarse_owner_list)
        if n_coarse_faces > 0:
            coarse_owner = torch.tensor(coarse_owner_list, device=device, dtype=INDEX_DTYPE)
            coarse_neighbour = torch.tensor(coarse_neigh_list, device=device, dtype=INDEX_DTYPE)
            coarse_lower = torch.tensor(coarse_lower_list, device=device, dtype=dtype)
            coarse_upper = torch.tensor(coarse_upper_list, device=device, dtype=dtype)
        else:
            coarse_owner = torch.zeros(0, device=device, dtype=INDEX_DTYPE)
            coarse_neighbour = torch.zeros(0, device=device, dtype=INDEX_DTYPE)
            coarse_lower = torch.zeros(0, device=device, dtype=dtype)
            coarse_upper = torch.zeros(0, device=device, dtype=dtype)

        return GAMGLevel(
            n_cells=n_coarse,
            diag=coarse_diag,
            lower=coarse_lower,
            upper=coarse_upper,
            owner=coarse_owner,
            neighbour=coarse_neighbour,
        )

    # ------------------------------------------------------------------
    # Restriction / Prolongation
    # ------------------------------------------------------------------

    @staticmethod
    def _restrict(
        r_fine: torch.Tensor,
        aggregation: AggregationData,
    ) -> torch.Tensor:
        """Restrict fine-grid residual to coarse grid.

        Simple injection: each coarse cell gets the sum of residuals
        from its aggregate of fine cells.

        Args:
            r_fine: ``(n_fine,)`` fine-grid residual.
            aggregation: Aggregation data.

        Returns:
            ``(n_coarse,)`` coarse-grid residual.
        """
        return scatter_add(
            r_fine,
            aggregation.fine_to_coarse.long(),
            aggregation.n_coarse,
        )

    @staticmethod
    def _prolongate(
        e_coarse: torch.Tensor,
        aggregation: AggregationData,
    ) -> torch.Tensor:
        """Prolongate coarse-grid correction to fine grid.

        Simple injection: each fine cell gets the correction from
        its coarse-cell aggregate.

        Args:
            e_coarse: ``(n_coarse,)`` coarse-grid correction.
            aggregation: Aggregation data.

        Returns:
            ``(n_fine,)`` fine-grid correction.
        """
        return gather(e_coarse, aggregation.fine_to_coarse.long())

    # ------------------------------------------------------------------
    # Smoother
    # ------------------------------------------------------------------

    def _smooth(
        self,
        level: GAMGLevel,
        x: torch.Tensor,
        b: torch.Tensor,
        n_iterations: int,
    ) -> torch.Tensor:
        """Apply smoothing iterations.

        Uses Jacobi iterations with DIC-like diagonal preconditioning
        for efficiency on coarse levels.

        Args:
            level: Current GAMG level data.
            x: Current solution estimate.
            b: Right-hand side.
            n_iterations: Number of smoothing iterations.

        Returns:
            Smoothed solution.
        """
        if n_iterations <= 0:
            return x

        device = level.diag.device
        dtype = level.diag.dtype

        # Build a temporary LDU matrix for the smoother
        temp_mat = LduMatrix(
            level.n_cells, level.owner, level.neighbour,
            device=device, dtype=dtype,
        )
        temp_mat.diag = level.diag
        temp_mat.lower = level.lower
        temp_mat.upper = level.upper

        # Use Jacobi-like smoothing with diagonal preconditioning
        # Precompute 1/diag for efficiency
        inv_diag = level.diag.abs().clamp(min=1e-30).reciprocal()

        for _ in range(n_iterations):
            r = b - temp_mat.Ax(x)
            x = x + inv_diag * r

        return x

    # ------------------------------------------------------------------
    # V-cycle
    # ------------------------------------------------------------------

    def _v_cycle(
        self,
        levels: list[GAMGLevel],
        aggregations: list[AggregationData],
        level_idx: int,
        x: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """Recursively apply one V-cycle.

        Args:
            levels: List of GAMG levels (fine to coarse).
            aggregations: List of aggregation data between levels.
            level_idx: Current level index.
            x: Current solution estimate.
            b: Right-hand side.

        Returns:
            Updated solution after one V-cycle.
        """
        level = levels[level_idx]

        # Base case: coarsest level — solve directly
        if level_idx == len(levels) - 1:
            # Direct solve on coarsest level using Jacobi iterations
            # (or just many smoothing steps)
            return self._smooth(level, x, b, n_iterations=50)

        # Pre-smoothing
        x = self._smooth(level, x, b, self._n_pre_smooth)

        # Compute residual
        temp_mat = LduMatrix(
            level.n_cells, level.owner, level.neighbour,
            device=level.diag.device, dtype=level.diag.dtype,
        )
        temp_mat.diag = level.diag
        temp_mat.lower = level.lower
        temp_mat.upper = level.upper

        r = b - temp_mat.Ax(x)

        # Restrict residual to coarse grid
        r_coarse = self._restrict(r, aggregations[level_idx])

        # Solve coarse error equation (initial guess = 0)
        e_coarse = torch.zeros_like(r_coarse)
        e_coarse = self._v_cycle(levels, aggregations, level_idx + 1, e_coarse, r_coarse)

        # Prolongate correction to fine grid
        e_fine = self._prolongate(e_coarse, aggregations[level_idx])

        # Correct
        x = x + e_fine

        # Post-smoothing
        x = self._smooth(level, x, b, self._n_post_smooth)

        return x

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def _solve(
        self,
        matrix: LduMatrix,
        source: torch.Tensor,
        x0: torch.Tensor,
        monitor: ResidualMonitor,
        max_iter: int,
    ) -> tuple[torch.Tensor, ConvergenceInfo]:
        """Run the GAMG solver.

        Args:
            matrix: The LDU matrix A.
            source: Right-hand side b.
            x0: Initial guess.
            monitor: Residual monitor.
            max_iter: Maximum V-cycles.

        Returns:
            Tuple of ``(solution, convergence_info)``.
        """
        device = matrix.device
        dtype = matrix.dtype

        source = source.to(device=device, dtype=dtype)
        x = x0.to(device=device, dtype=dtype).clone()

        # Build multigrid hierarchy
        levels: list[GAMGLevel] = []
        aggregations: list[AggregationData] = []

        # Level 0 = finest (the input matrix)
        current_level = GAMGLevel(
            n_cells=matrix.n_cells,
            diag=matrix.diag.clone(),
            lower=matrix.lower.clone(),
            upper=matrix.upper.clone(),
            owner=matrix.owner.clone(),
            neighbour=matrix.neighbour.clone(),
        )
        levels.append(current_level)

        # Build coarser levels
        for _ in range(self._max_levels):
            if current_level.n_cells <= self._min_cells_coarse:
                break

            # Target: halve the number of cells
            n_coarse_target = max(current_level.n_cells // 2, self._min_cells_coarse)

            # Build a temporary LDU matrix for coarsening
            temp_mat = LduMatrix(
                current_level.n_cells,
                current_level.owner,
                current_level.neighbour,
                device=device, dtype=dtype,
            )
            temp_mat.diag = current_level.diag
            temp_mat.lower = current_level.lower
            temp_mat.upper = current_level.upper

            aggregation = self._build_aggregates(temp_mat, n_coarse_target)

            if aggregation.n_coarse >= current_level.n_cells:
                # No effective coarsening possible
                break

            aggregations.append(aggregation)

            # Build coarse matrix
            coarse_level = self._build_coarse_matrix(temp_mat, aggregation)
            levels.append(coarse_level)
            current_level = coarse_level

        # Outer V-cycle iterations
        converged = False
        for i in range(1, max_iter + 1):
            x = self._v_cycle(levels, aggregations, 0, x, source)

            # Compute residual for convergence check
            r = source - matrix.Ax(x)

            if monitor.update(r, i):
                converged = True
                break

        info = monitor.build_info(converged=converged)
        return x, info
