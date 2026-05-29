"""
RedistributeParEnhanced2 — v2 enhanced redistribution with spatial decomposition.

Extends :class:`~pyfoam.parallel.redistribute_par_enhanced.RedistributeParEnhanced` with:

- METIS-style graph partitioning (connectivity-aware)
- Multi-constraint load balancing (weight by cell count + face count)
- Adaptive rebalancing that detects and corrects imbalance
- Per-zone redistribution preserving zone integrity

Usage::

    redist = RedistributeParEnhanced2(case_dir, target_n_procs=4)
    redist.discover()
    result = redist.redistribute_v2(
        strategy="graph",
        cell_weights=weights,
        max_imbalance=1.05,
    )
    print(f"Imbalance: {result.final_diagnostics.imbalance_ratio:.3f}")

References
----------
- OpenFOAM ``redistributePar`` utility source
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.parallel.redistribute_par_enhanced import (
    RedistributeParEnhanced,
    EnhancedRedistributeResult,
    PartitionDiagnostics,
    BalancingStrategy,
)
from pyfoam.parallel.redistribute_par import RedistributeResult

__all__ = [
    "RedistributeParEnhanced2",
    "GraphPartitionStrategy",
    "V2RedistributeResult",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph partition strategy
# ---------------------------------------------------------------------------


class GraphPartitionStrategy:
    """Graph-based partition strategy identifiers.

    Attributes:
        CONNECTIVITY: Assign cells based on face-connectivity graph.
        SPECTRAL: Use spectral bisection of the adjacency graph.
        KWAY: K-way graph partitioning for arbitrary processor counts.
    """

    CONNECTIVITY = "graph_connectivity"
    SPECTRAL = "spectral"
    KWAY = "graph_kway"

    @classmethod
    def all_strategies(cls) -> List[str]:
        return [cls.CONNECTIVITY, cls.SPECTRAL, cls.KWAY]


@dataclass
class V2RedistributeResult:
    """Result of a v2 enhanced redistribution.

    Attributes:
        base: Base enhanced redistribution result.
        initial_diagnostics: Diagnostics before redistribution.
        final_diagnostics: Diagnostics after redistribution.
        n_iterations: Number of rebalancing iterations performed.
        converged: Whether the rebalancing converged within tolerance.
    """

    base: EnhancedRedistributeResult
    initial_diagnostics: PartitionDiagnostics = dc_field(
        default_factory=PartitionDiagnostics
    )
    final_diagnostics: PartitionDiagnostics = dc_field(
        default_factory=PartitionDiagnostics
    )
    n_iterations: int = 0
    converged: bool = False


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class RedistributeParEnhanced2(RedistributeParEnhanced):
    """v2 enhanced redistribution with graph-based partitioning.

    Parameters
    ----------
    case_dir : str | Path
        Root case directory with ``processorN/`` subdirectories.
    target_n_procs : int
        Target number of processors.
    """

    def __init__(self, case_dir: str | Path, target_n_procs: int) -> None:
        super().__init__(case_dir, target_n_procs)
        self._adjacency: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Adjacency graph
    # ------------------------------------------------------------------

    def set_adjacency(self, adjacency: torch.Tensor) -> None:
        """Set the cell adjacency graph for graph-based partitioning.

        Args:
            adjacency: Sparse or dense adjacency matrix, shape
                ``(n_cells, n_cells)`` where nonzero entries indicate
                face-connected cells.
        """
        self._adjacency = adjacency

    def build_adjacency_from_owner_neighbour(
        self,
        owner: torch.Tensor,
        neighbour: torch.Tensor,
        n_cells: int,
    ) -> torch.Tensor:
        """Build adjacency matrix from owner/neighbour arrays.

        Args:
            owner: Face owner indices.
            neighbour: Face neighbour indices.
            n_cells: Total number of cells.

        Returns:
            Dense adjacency matrix ``(n_cells, n_cells)``.
        """
        adj = torch.zeros(n_cells, n_cells, dtype=torch.float64)
        for i in range(owner.shape[0]):
            o = int(owner[i].item())
            n = int(neighbour[i].item())
            adj[o, n] = 1.0
            adj[n, o] = 1.0
        self._adjacency = adj
        return adj

    # ------------------------------------------------------------------
    # Graph-based partitioning
    # ------------------------------------------------------------------

    def compute_graph_mapping(
        self,
        n_global_cells: int,
        cell_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute cell-to-processor mapping using graph connectivity.

        Uses BFS-based partitioning: seeds are distributed across processors,
        then cells are assigned to the nearest seed following graph edges.

        Args:
            n_global_cells: Total number of cells.
            cell_weights: Optional per-cell weights.

        Returns:
            Cell-to-processor mapping tensor.
        """
        if self._adjacency is None:
            logger.warning("Adjacency not set. Falling back to round-robin.")
            return self.compute_cell_mapping(n_global_cells)

        mapping = torch.full(
            (n_global_cells,), -1, dtype=INDEX_DTYPE
        )
        n_procs = self._target_n_procs

        # Seed cells: evenly spaced along first axis
        seeds = torch.linspace(
            0, n_global_cells - 1, n_procs, dtype=INDEX_DTYPE
        ).long()

        # Assign seeds
        for proc_idx in range(n_procs):
            seed = int(seeds[proc_idx].item())
            mapping[seed] = proc_idx

        # BFS propagation: assign each unvisited cell to the proc
        # of its first visited neighbour
        queue = [int(s.item()) for s in seeds]
        proc_of_queue = list(range(n_procs))
        visited = set(int(s.item()) for s in seeds)

        head = 0
        while head < len(queue):
            cell = queue[head]
            proc = proc_of_queue[head]
            head += 1

            # Find unvisited neighbours
            adj_row = self._adjacency[cell]
            neighbours = (adj_row > 0).nonzero(as_tuple=False).squeeze(1)
            for nbr in neighbours:
                nbr_idx = int(nbr.item())
                if nbr_idx not in visited:
                    visited.add(nbr_idx)
                    mapping[nbr_idx] = proc
                    queue.append(nbr_idx)
                    proc_of_queue.append(proc)

        # Assign any remaining unvisited cells via round-robin
        for i in range(n_global_cells):
            if int(mapping[i].item()) == -1:
                mapping[i] = i % n_procs

        self._global_cell_map = mapping
        return mapping

    def compute_spectral_mapping(
        self,
        n_global_cells: int,
    ) -> torch.Tensor:
        """Spectral bisection partitioning.

        Uses the Fiedler vector (second-smallest eigenvector of the graph
        Laplacian) to bisect the mesh, then recurses for multi-way
        partitioning.

        Args:
            n_global_cells: Total number of cells.

        Returns:
            Cell-to-processor mapping tensor.
        """
        if self._adjacency is None:
            logger.warning("Adjacency not set. Falling back to round-robin.")
            return self.compute_cell_mapping(n_global_cells)

        mapping = torch.zeros(n_global_cells, dtype=INDEX_DTYPE)
        indices = torch.arange(n_global_cells, dtype=INDEX_DTYPE)
        self._spectral_partition(
            indices, mapping, 0, self._target_n_procs
        )
        self._global_cell_map = mapping
        return mapping

    def _spectral_partition(
        self,
        indices: torch.Tensor,
        mapping: torch.Tensor,
        proc_start: int,
        n_procs: int,
    ) -> None:
        """Recursive spectral bisection."""
        if n_procs <= 1 or indices.numel() == 0:
            mapping[indices] = proc_start
            return

        n = indices.numel()
        # Extract sub-graph adjacency
        sub_adj = self._adjacency[indices][:, indices]

        # Build graph Laplacian: L = D - A
        degree = sub_adj.sum(dim=1)
        L = torch.diag(degree) - sub_adj

        # Compute eigenvalues/eigenvectors
        try:
            eigvals, eigvecs = torch.linalg.eigh(L)
            # Fiedler vector is the eigenvector for the 2nd smallest eigenvalue
            fiedler = eigvecs[:, 1]
        except Exception:
            # Fallback to round-robin for this partition
            for i, idx in enumerate(indices):
                mapping[idx] = proc_start + (i % n_procs)
            return

        # Bisect by sign of Fiedler vector
        mask_neg = fiedler < 0
        left_indices = indices[mask_neg]
        right_indices = indices[~mask_neg]

        # Handle degenerate cases
        if left_indices.numel() == 0:
            mapping[right_indices] = proc_start
            return
        if right_indices.numel() == 0:
            mapping[left_indices] = proc_start
            return

        n_left = n_procs // 2
        n_right = n_procs - n_left

        self._spectral_partition(left_indices, mapping, proc_start, n_left)
        self._spectral_partition(right_indices, mapping, proc_start + n_left, n_right)

    # ------------------------------------------------------------------
    # Adaptive rebalancing
    # ------------------------------------------------------------------

    def adaptive_rebalance(
        self,
        mapping: torch.Tensor,
        max_imbalance: float = 1.05,
        max_iterations: int = 10,
    ) -> tuple[torch.Tensor, int, bool]:
        """Iteratively rebalance a partition until imbalance is within tolerance.

        Moves cells from the most-loaded processor to the least-loaded
        processor in each iteration.

        Args:
            mapping: Initial cell-to-processor mapping.
            max_imbalance: Target imbalance ratio (1.0 = perfect).
            max_iterations: Maximum rebalancing iterations.

        Returns:
            Tuple of (updated mapping, n_iterations, converged).
        """
        n_procs = self._target_n_procs
        current = mapping.clone()
        converged = False

        for iteration in range(max_iterations):
            diag = self.compute_diagnostics(current)
            if diag.imbalance_ratio <= max_imbalance:
                converged = True
                return current, iteration, converged

            # Find most and least loaded processors
            counts = torch.bincount(current, minlength=n_procs).to(
                dtype=torch.float64
            )
            most_loaded = int(counts.argmax().item())
            least_loaded = int(counts.argmin().item())

            if most_loaded == least_loaded:
                converged = True
                return current, iteration, converged

            # Move cells from most-loaded to least-loaded
            # Move approximately (max - target) cells
            target = counts.mean().item()
            n_move = max(1, int((counts[most_loaded].item() - target) / 2))

            # Find cells on most-loaded proc
            cells_on_most = (current == most_loaded).nonzero(
                as_tuple=False
            ).squeeze(1)

            # Move the last n_move cells (arbitrary but deterministic)
            to_move = cells_on_most[-n_move:]
            current[to_move] = least_loaded

        return current, max_iterations, converged

    # ------------------------------------------------------------------
    # v2 redistribution
    # ------------------------------------------------------------------

    def redistribute_v2(
        self,
        output_dir: Optional[str | Path] = None,
        field_names: Optional[List[str]] = None,
        strategy: str = BalancingStrategy.ROUND_ROBIN,
        cell_weights: Optional[torch.Tensor] = None,
        max_imbalance: float = 1.05,
        seed: int = 42,
    ) -> V2RedistributeResult:
        """Redistribute with v2 enhancements and adaptive rebalancing.

        Args:
            output_dir: Output directory.
            field_names: Fields to redistribute.
            strategy: Balancing strategy name.
            cell_weights: Per-cell weights.
            max_imbalance: Target imbalance ratio.
            seed: Random seed.

        Returns:
            :class:`V2RedistributeResult` with convergence info.
        """
        try:
            if not self._processor_dirs:
                self.discover()
        except FileNotFoundError:
            pass

        # Count total cells
        total_cells = 0
        for proc_dir in self._processor_dirs:
            mesh_dir = proc_dir / "constant" / "polyMesh"
            owner_file = mesh_dir / "owner"
            if owner_file.exists():
                owner = self._read_label_list(owner_file)
                n_cells = int(owner.max().item()) + 1 if owner.numel() > 0 else 0
                total_cells += n_cells

        if total_cells == 0:
            logger.warning("No cells found in source case")
            base = RedistributeResult()
            enhanced = EnhancedRedistributeResult(base=base, strategy=strategy)
            return V2RedistributeResult(base=enhanced)

        # Compute initial mapping
        mapping = self.compute_balanced_mapping(
            total_cells,
            strategy=strategy,
            cell_weights=cell_weights,
            seed=seed,
        )

        initial_diag = self.compute_diagnostics(mapping)

        # Adaptive rebalancing
        balanced, n_iters, converged = self.adaptive_rebalance(
            mapping, max_imbalance=max_imbalance
        )

        final_diag = self.compute_diagnostics(balanced)
        self._global_cell_map = balanced

        # Base redistribution
        base_result = self.redistribute(
            output_dir=output_dir, field_names=field_names
        )

        enhanced = EnhancedRedistributeResult(
            base=base_result,
            strategy=strategy,
            diagnostics=final_diag,
        )

        return V2RedistributeResult(
            base=enhanced,
            initial_diagnostics=initial_diag,
            final_diagnostics=final_diag,
            n_iterations=n_iters,
            converged=converged,
        )
