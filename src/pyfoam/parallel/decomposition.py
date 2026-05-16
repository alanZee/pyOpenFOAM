"""
Domain decomposition for parallel CFD.

Splits a mesh into subdomains for distribution across MPI ranks.

Decomposition methods
---------------------
- **Geometric** (``method="simple"``): splits along the longest axis, assigning
  cells to processors by spatial position.  Fast, works for any mesh.
- **Scotch** (``method="scotch"``): graph-based partitioning using the Scotch
  library (if installed).  Minimises communication surface.

All tensors respect the global device/dtype from :mod:`pyfoam.core`.

Usage::

    from pyfoam.parallel.decomposition import Decomposition

    decomp = Decomposition(mesh, n_processors=4, method="simple")
    subdomains = decomp.decompose()
    # subdomains[i] is a SubDomain for processor i
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.poly_mesh import PolyMesh
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.parallel.processor_patch import ProcessorPatch

__all__ = ["Decomposition", "SubDomain"]

DecompositionMethod = Literal["simple", "scotch"]


# ---------------------------------------------------------------------------
# SubDomain — a subregion of the global mesh
# ---------------------------------------------------------------------------


@dataclass
class SubDomain:
    """A subdomain extracted from a global mesh.

    Attributes
    ----------
    processor_id : int
        MPI rank this subdomain belongs to.
    mesh : PolyMesh
        Subdomain mesh with local numbering (0-indexed).
    global_cell_ids : torch.Tensor
        Maps local cell index → global cell index.
    ghost_cells : torch.Tensor
        Indices of ghost cells in the local mesh (cells owned by neighbours).
    n_owned_cells : int
        Number of cells owned by this processor (ghost cells excluded).
    processor_patches : list[ProcessorPatch]
        Processor boundary patches for halo exchange with each neighbour.
    """

    processor_id: int
    mesh: PolyMesh
    global_cell_ids: torch.Tensor
    ghost_cells: torch.Tensor
    n_owned_cells: int
    processor_patches: list[ProcessorPatch] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


class Decomposition:
    """Domain decomposition for parallel CFD.

    Parameters
    ----------
    mesh : PolyMesh | FvMesh
        The global mesh to decompose.
    n_processors : int
        Number of processors (subdomains) to create.
    method : str
        Decomposition method: ``"simple"`` (geometric) or ``"scotch"``.

    Attributes
    ----------
    cell_assignment : torch.Tensor
        ``(n_cells,)`` int tensor — processor ID for each cell.
    """

    def __init__(
        self,
        mesh: PolyMesh | FvMesh,
        n_processors: int,
        method: DecompositionMethod = "simple",
    ) -> None:
        if n_processors < 1:
            raise ValueError(f"n_processors must be >= 1, got {n_processors}")
        if n_processors > mesh.n_cells:
            raise ValueError(
                f"n_processors ({n_processors}) > n_cells ({mesh.n_cells})"
            )

        self._mesh = mesh
        self._n_processors = n_processors
        self._method = method
        self._device = get_device()
        self._dtype = get_default_dtype()

        # Will be populated by decompose()
        self._cell_assignment: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mesh(self) -> PolyMesh | FvMesh:
        """The global mesh."""
        return self._mesh

    @property
    def n_processors(self) -> int:
        """Number of processors."""
        return self._n_processors

    @property
    def cell_assignment(self) -> torch.Tensor:
        """Processor ID per cell ``(n_cells,)``."""
        if self._cell_assignment is None:
            raise RuntimeError("Call decompose() first")
        return self._cell_assignment

    # ------------------------------------------------------------------
    # Decomposition
    # ------------------------------------------------------------------

    def decompose(self) -> list[SubDomain]:
        """Decompose the mesh into subdomains.

        Returns:
            List of :class:`SubDomain` objects, one per processor.
        """
        if self._method == "simple":
            self._cell_assignment = self._decompose_simple()
        elif self._method == "scotch":
            self._cell_assignment = self._decompose_scotch()
        else:
            raise ValueError(f"Unknown decomposition method: {self._method!r}")

        return self._build_subdomains()

    # ------------------------------------------------------------------
    # Simple geometric decomposition
    # ------------------------------------------------------------------

    def _decompose_simple(self) -> torch.Tensor:
        """Split mesh along the longest axis.

        Cells are assigned to processors based on their position along
        the longest extent of the bounding box.

        Returns:
            ``(n_cells,)`` int tensor of processor IDs.
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        n_procs = self._n_processors

        # Compute cell centres if we have an FvMesh, otherwise use face-avg
        if isinstance(mesh, FvMesh):
            centres = mesh.cell_centres  # (n_cells, 3)
        else:
            centres = self._estimate_cell_centres(mesh)

        # Find the longest axis
        mins = centres.min(dim=0).values
        maxs = centres.max(dim=0).values
        extents = maxs - mins
        longest_axis = int(extents.argmax().item())

        # Sort cells by position along longest axis
        coords = centres[:, longest_axis]
        # Assign processor IDs by uniform binning
        # Use quantile-based splitting for balanced partitions
        sorted_coords, _ = torch.sort(coords)
        # Boundaries at quantiles
        boundaries = torch.zeros(n_procs + 1, device=self._device, dtype=self._dtype)
        boundaries[0] = sorted_coords[0] - 1e-10
        boundaries[-1] = sorted_coords[-1] + 1e-10
        for i in range(1, n_procs):
            idx = int(i * n_cells / n_procs)
            boundaries[i] = sorted_coords[idx]

        # Assign each cell
        assignment = torch.zeros(n_cells, dtype=INDEX_DTYPE, device=self._device)
        for i in range(n_procs):
            mask = (coords >= boundaries[i]) & (coords < boundaries[i + 1])
            assignment[mask] = i

        # Ensure last processor gets boundary cells
        assignment[coords >= boundaries[-2]] = n_procs - 1

        return assignment

    def _estimate_cell_centres(self, mesh: PolyMesh) -> torch.Tensor:
        """Estimate cell centres from face-vertex averages for PolyMesh.

        For each cell, averages the centres of its faces.
        """
        n_cells = mesh.n_cells
        face_centres = torch.zeros(mesh.n_faces, 3, device=self._device, dtype=self._dtype)
        for f_idx in range(mesh.n_faces):
            pts = mesh.points[mesh.faces[f_idx]]
            face_centres[f_idx] = pts.mean(dim=0)

        # Accumulate face centres per cell
        cell_sum = torch.zeros(n_cells, 3, device=self._device, dtype=self._dtype)
        cell_count = torch.zeros(n_cells, device=self._device, dtype=INDEX_DTYPE)

        for f_idx in range(mesh.n_faces):
            owner_cell = mesh.owner[f_idx].item()
            cell_sum[owner_cell] += face_centres[f_idx]
            cell_count[owner_cell] += 1

        for f_idx in range(mesh.n_internal_faces):
            nbr_cell = mesh.neighbour[f_idx].item()
            cell_sum[nbr_cell] += face_centres[f_idx]
            cell_count[nbr_cell] += 1

        # Avoid division by zero
        safe_count = cell_count.float().clamp(min=1)
        return cell_sum / safe_count.unsqueeze(1)

    # ------------------------------------------------------------------
    # Scotch decomposition (optional)
    # ------------------------------------------------------------------

    def _decompose_scotch(self) -> torch.Tensor:
        """Graph-based decomposition using the Scotch library.

        Falls back to simple decomposition if Scotch is not installed.

        Returns:
            ``(n_cells,)`` int tensor of processor IDs.
        """
        try:
            import scotch  # type: ignore[import-untyped]
        except ImportError:
            import warnings
            warnings.warn(
                "scotch library not installed; falling back to simple decomposition",
                stacklevel=2,
            )
            return self._decompose_simple()

        mesh = self._mesh
        n_cells = mesh.n_cells

        # Build adjacency graph from mesh topology
        # Each cell is a vertex; internal faces define edges
        adj_offsets = torch.zeros(n_cells + 1, dtype=INDEX_DTYPE, device="cpu")
        adj_indices: list[int] = []

        # Build adjacency lists
        neighbours_per_cell: list[list[int]] = [[] for _ in range(n_cells)]
        for f_idx in range(mesh.n_internal_faces):
            own = mesh.owner[f_idx].item()
            nbr = mesh.neighbour[f_idx].item()
            neighbours_per_cell[own].append(nbr)
            neighbours_per_cell[nbr].append(own)

        offset = 0
        for c in range(n_cells):
            adj_offsets[c] = offset
            unique_nbrs = sorted(set(neighbours_per_cell[c]))
            adj_indices.extend(unique_nbrs)
            offset += len(unique_nbrs)
        adj_offsets[n_cells] = offset

        adj_indices_t = torch.tensor(adj_indices, dtype=torch.int32, device="cpu")
        adj_offsets_np = adj_offsets.numpy().astype("int32")
        adj_indices_np = adj_indices_t.numpy().astype("int32")

        # Use Scotch to partition
        strat = scotch.Strat()
        strat.archCmpltWgt(0)  # No weights

        graph = scotch.Graph()
        graph.init()
        graph.build(0, adj_offsets_np, None, None, None, None, adj_indices_np)

        part = torch.zeros(n_cells, dtype=torch.int32, device="cpu")
        part_np = part.numpy()

        graph.part(self._n_processors, strat, part_np)

        return torch.from_numpy(part_np).to(dtype=INDEX_DTYPE, device=self._device)

    # ------------------------------------------------------------------
    # Build subdomain meshes
    # ------------------------------------------------------------------

    def _build_subdomains(self) -> list[SubDomain]:
        """Build SubDomain objects from the cell assignment.

        For each processor:
        1. Identify owned cells
        2. Identify ghost cells (neighbours of owned cells that belong to other procs)
        3. Build a subdomain mesh with local numbering
        4. Create ProcessorPatch objects for halo exchange
        """
        assignment = self._cell_assignment
        mesh = self._mesh
        n_procs = self._n_processors
        subdomains: list[SubDomain] = []

        for proc_id in range(n_procs):
            # --- Step 1: Identify owned and ghost cells ---
            owned_mask = assignment == proc_id
            owned_global = torch.where(owned_mask)[0]
            n_owned = owned_global.shape[0]

            # Find ghost cells: cells adjacent to owned cells but belonging to other procs
            # ghost_global_set: set of global cell IDs that are ghosts
            # ghost_to_neighbour: maps global ghost cell ID → neighbour proc ID
            ghost_global_set: set[int] = set()
            ghost_to_neighbour: dict[int, int] = {}
            for f_idx in range(mesh.n_internal_faces):
                own = mesh.owner[f_idx].item()
                nbr = mesh.neighbour[f_idx].item()
                own_proc = assignment[own].item()
                nbr_proc = assignment[nbr].item()
                if own_proc == proc_id and nbr_proc != proc_id:
                    ghost_global_set.add(nbr)
                    ghost_to_neighbour[nbr] = nbr_proc
                elif nbr_proc == proc_id and own_proc != proc_id:
                    ghost_global_set.add(own)
                    ghost_to_neighbour[own] = own_proc

            ghost_global = torch.tensor(sorted(ghost_global_set), dtype=INDEX_DTYPE, device=self._device)
            n_ghost = ghost_global.shape[0]

            # All cells in subdomain (owned first, then ghost)
            all_cells = torch.cat([owned_global, ghost_global])
            n_total = all_cells.shape[0]

            # Build global-to-local mapping
            global_to_local = torch.full(
                (mesh.n_cells,), -1, dtype=INDEX_DTYPE, device=self._device
            )
            global_to_local[all_cells] = torch.arange(n_total, dtype=INDEX_DTYPE, device=self._device)

            # --- Step 2: Per-neighbor tracking for ProcessorPatch ---
            # For each neighbour proc, track:
            #   owned_boundary_cells: our owned cells adjacent to that neighbour (for SENDING)
            #   ghost_pairs: (local_ghost_idx, global_ghost_id) pairs (for RECEIVING)
            neighbour_owned_cells: dict[int, set[int]] = {}
            neighbour_ghost_pairs: dict[int, list[tuple[int, int]]] = {}  # nbr_proc → [(local_idx, global_id), ...]

            for f_idx in range(mesh.n_internal_faces):
                own = mesh.owner[f_idx].item()
                nbr = mesh.neighbour[f_idx].item()
                own_proc = assignment[own].item()
                nbr_proc = assignment[nbr].item()

                if own_proc == proc_id and nbr_proc != proc_id:
                    local_own = global_to_local[own].item()
                    neighbour_owned_cells.setdefault(nbr_proc, set()).add(local_own)
                    local_nbr = global_to_local[nbr].item()
                    neighbour_ghost_pairs.setdefault(nbr_proc, []).append((local_nbr, nbr))
                elif nbr_proc == proc_id and own_proc != proc_id:
                    local_nbr = global_to_local[nbr].item()
                    neighbour_owned_cells.setdefault(own_proc, set()).add(local_nbr)
                    local_own = global_to_local[own].item()
                    neighbour_ghost_pairs.setdefault(own_proc, []).append((local_own, own))

            # --- Step 3: Build subdomain mesh faces ---
            sub_faces_list: list[torch.Tensor] = []
            sub_owner_list: list[int] = []
            sub_neighbour_list: list[int] = []

            # Internal faces (both cells in subdomain)
            for f_idx in range(mesh.n_internal_faces):
                own = mesh.owner[f_idx].item()
                nbr = mesh.neighbour[f_idx].item()
                own_in = int(assignment[own].item()) == proc_id
                nbr_in = int(assignment[nbr].item()) == proc_id

                if own_in and nbr_in:
                    local_own = global_to_local[own].item()
                    local_nbr = global_to_local[nbr].item()
                    face_pts = mesh.faces[f_idx]
                    sub_faces_list.append(face_pts)
                    sub_owner_list.append(local_own)
                    sub_neighbour_list.append(local_nbr)

            n_sub_internal = len(sub_neighbour_list)

            # Cross-processor faces (one cell in subdomain → becomes processor boundary face)
            for f_idx in range(mesh.n_internal_faces):
                own = mesh.owner[f_idx].item()
                nbr = mesh.neighbour[f_idx].item()
                own_proc = assignment[own].item()
                nbr_proc = assignment[nbr].item()

                if own_proc == proc_id and nbr_proc != proc_id:
                    local_own = global_to_local[own].item()
                    local_nbr = global_to_local[nbr].item()
                    face_pts = mesh.faces[f_idx]
                    sub_faces_list.append(face_pts)
                    sub_owner_list.append(local_own)
                    sub_neighbour_list.append(local_nbr)
                elif nbr_proc == proc_id and own_proc != proc_id:
                    local_own = global_to_local[own].item()
                    local_nbr = global_to_local[nbr].item()
                    face_pts = mesh.faces[f_idx]
                    sub_faces_list.append(face_pts)
                    sub_owner_list.append(local_nbr)
                    sub_neighbour_list.append(local_own)

            # Original boundary faces that touch owned cells
            proc_boundary_faces: list[dict] = []
            for patch_idx, patch in enumerate(mesh.boundary):
                start = patch["startFace"]
                n_faces_patch = patch["nFaces"]
                patch_face_count = 0
                for f_off in range(n_faces_patch):
                    f_idx = start + f_off
                    own = mesh.owner[f_idx].item()
                    if int(assignment[own].item()) == proc_id:
                        local_own = global_to_local[own].item()
                        face_pts = mesh.faces[f_idx]
                        sub_faces_list.append(face_pts)
                        sub_owner_list.append(local_own)
                        patch_face_count += 1

                if patch_face_count > 0:
                    current_start = n_sub_internal + sum(
                        pf.get("nFaces", 0) for pf in proc_boundary_faces
                    )
                    proc_boundary_faces.append({
                        "name": patch["name"],
                        "type": patch.get("type", "patch"),
                        "startFace": current_start,
                        "nFaces": patch_face_count,
                    })

            # --- Step 4: Build ProcessorPatch objects ---
            processor_patches: list[ProcessorPatch] = []
            all_neighbours = set(neighbour_owned_cells.keys()) | set(neighbour_ghost_pairs.keys())
            for nbr_proc in sorted(all_neighbours):
                # Owned boundary cells that this neighbour needs (for sending)
                owned_bc = sorted(neighbour_owned_cells.get(nbr_proc, set()))

                # Ghost cells from this neighbour (for receiving)
                # Deduplicate and sort by local index
                ghost_pairs = neighbour_ghost_pairs.get(nbr_proc, [])
                # Deduplicate: same local ghost cell might appear from multiple faces
                seen: set[int] = set()
                unique_pairs: list[tuple[int, int]] = []
                for local_idx, global_id in ghost_pairs:
                    if local_idx not in seen:
                        seen.add(local_idx)
                        unique_pairs.append((local_idx, global_id))
                unique_pairs.sort(key=lambda p: p[0])  # Sort by local index

                ghost_lc = [p[0] for p in unique_pairs]
                ghost_gi = [p[1] for p in unique_pairs]

                if owned_bc:
                    owned_tensor = torch.tensor(owned_bc, dtype=INDEX_DTYPE, device=self._device)
                else:
                    owned_tensor = torch.zeros(0, dtype=INDEX_DTYPE, device=self._device)

                if ghost_lc:
                    ghost_tensor = torch.tensor(ghost_lc, dtype=INDEX_DTYPE, device=self._device)
                    global_tensor = torch.tensor(ghost_gi, dtype=INDEX_DTYPE, device=self._device)
                else:
                    ghost_tensor = torch.zeros(0, dtype=INDEX_DTYPE, device=self._device)
                    global_tensor = torch.zeros(0, dtype=INDEX_DTYPE, device=self._device)

                processor_patches.append(ProcessorPatch(
                    name=f"procBoundary{proc_id}To{nbr_proc}Patch",
                    neighbour_rank=nbr_proc,
                    local_ghost_cells=ghost_tensor,
                    remote_cells=owned_tensor,
                    remote_global_ids=global_tensor,
                ))

            # --- Step 5: Build subdomain mesh ---
            if sub_faces_list:
                sub_points = mesh.points.clone()
                sub_owner = torch.tensor(sub_owner_list, dtype=INDEX_DTYPE, device=self._device)
                sub_neighbour = torch.tensor(sub_neighbour_list, dtype=INDEX_DTYPE, device=self._device)
            else:
                sub_points = mesh.points.clone()
                sub_owner = torch.zeros(0, dtype=INDEX_DTYPE, device=self._device)
                sub_neighbour = torch.zeros(0, dtype=INDEX_DTYPE, device=self._device)

            sub_mesh = PolyMesh(
                points=sub_points,
                faces=sub_faces_list,
                owner=sub_owner,
                neighbour=sub_neighbour,
                boundary=proc_boundary_faces,
                validate=False,
            )

            subdomains.append(SubDomain(
                processor_id=proc_id,
                mesh=sub_mesh,
                global_cell_ids=all_cells,
                ghost_cells=torch.arange(n_owned, n_total, dtype=INDEX_DTYPE, device=self._device),
                n_owned_cells=n_owned,
                processor_patches=processor_patches,
            ))

        return subdomains

    # ------------------------------------------------------------------
    # Load balancing metrics
    # ------------------------------------------------------------------

    def load_balance_metrics(self) -> dict[str, float]:
        """Compute load balancing statistics.

        Returns:
            Dict with keys: ``min_cells``, ``max_cells``, ``mean_cells``,
            ``imbalance_ratio``, ``min_faces``, ``max_faces``.
        """
        if self._cell_assignment is None:
            raise RuntimeError("Call decompose() first")

        assignment = self._cell_assignment
        n_procs = self._n_processors

        cells_per_proc = torch.zeros(n_procs, dtype=INDEX_DTYPE, device=self._device)
        for i in range(n_procs):
            cells_per_proc[i] = (assignment == i).sum()

        mean_cells = cells_per_proc.float().mean().item()
        imbalance = cells_per_proc.float().max().item() / max(mean_cells, 1)

        # Count faces per processor (approximate: count internal face contacts)
        mesh = self._mesh
        faces_per_proc = torch.zeros(n_procs, dtype=INDEX_DTYPE, device=self._device)
        for f_idx in range(mesh.n_internal_faces):
            own = assignment[mesh.owner[f_idx].item()].item()
            nbr = assignment[mesh.neighbour[f_idx].item()].item()
            faces_per_proc[own] += 1
            if own != nbr:
                faces_per_proc[nbr] += 1
            else:
                pass  # Internal face within same processor

        return {
            "min_cells": float(cells_per_proc.min().item()),
            "max_cells": float(cells_per_proc.max().item()),
            "mean_cells": mean_cells,
            "imbalance_ratio": imbalance,
            "min_faces": float(faces_per_proc.min().item()),
            "max_faces": float(faces_per_proc.max().item()),
        }

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Decomposition(mesh={self._mesh}, "
            f"n_processors={self._n_processors}, "
            f"method={self._method!r})"
        )
