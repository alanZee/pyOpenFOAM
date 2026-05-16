"""
Multi-GPU support for domain-decomposed CFD simulations.

Provides:

- :class:`MultiGPUManager` — extends :class:`~pyfoam.core.device.DeviceManager`
  with multi-GPU awareness (device enumeration, partition-to-device mapping)
- :func:`partition_mesh` — split a mesh into *N* partitions for domain
  decomposition (simple geometric bisection along the longest axis)
- :class:`GpuCommunicator` — inter-GPU communication wrappers using
  ``torch.distributed`` (all_gather, all_reduce, send/recv)
- :class:`MultiGPUMatrix` — wraps :class:`~pyfoam.core.ldu_matrix.LduMatrix`
  with partitioned storage across devices

Design goal: code that runs on a single GPU should work **unchanged** on *N*
GPUs when wrapped with this module.  On CPU-only or single-GPU systems the
classes degrade gracefully to single-device operation.

All tensors respect the global device/dtype configuration from
:mod:`pyfoam.core.device`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype, DeviceManager
from pyfoam.core.dtype import INDEX_DTYPE

__all__ = [
    "MultiGPUManager",
    "MeshPartition",
    "partition_mesh",
    "GpuCommunicator",
    "MultiGPUMatrix",
]


# ---------------------------------------------------------------------------
# Partition data structure
# ---------------------------------------------------------------------------


@dataclass
class MeshPartition:
    """Describes one partition of a domain-decomposed mesh.

    Attributes
    ----------
    partition_id : int
        Zero-based partition index.
    cell_indices : torch.Tensor
        ``(n_local_cells,)`` global cell indices belonging to this partition.
    owner : torch.Tensor
        ``(n_local_internal_faces,)`` local owner indices (remapped to local).
    neighbour : torch.Tensor
        ``(n_local_internal_faces,)`` local neighbour indices (remapped to local).
    boundary_cells : torch.Tensor
        ``(n_boundary_cells,)`` global indices of cells on the partition
        boundary (halo cells that need inter-GPU communication).
    device : torch.device
        Device this partition resides on.
    """

    partition_id: int
    cell_indices: torch.Tensor
    owner: torch.Tensor
    neighbour: torch.Tensor
    boundary_cells: torch.Tensor
    device: torch.device


# ---------------------------------------------------------------------------
# Multi-GPU Manager
# ---------------------------------------------------------------------------


class MultiGPUManager:
    """Manages multiple GPU devices for domain-decomposed simulations.

    Extends :class:`DeviceManager` with:

    - Device enumeration (how many GPUs, which ones)
    - Partition-to-device mapping
    - Device selection per partition

    On single-GPU or CPU-only systems, all partitions map to the same device.

    Usage::

        mgm = MultiGPUManager()
        print(mgm.device_count)        # e.g. 2
        dev = mgm.device_for_partition(0)  # cuda:0
        dev = mgm.device_for_partition(1)  # cuda:1
    """

    def __init__(self, devices: list[str | torch.device] | None = None) -> None:
        """Initialise the multi-GPU manager.

        Args:
            devices: Explicit list of devices to use.  If ``None``,
                auto-detects available GPUs (falls back to CPU).
        """
        self._dm = DeviceManager()

        if devices is not None:
            self._devices = [torch.device(d) for d in devices]
        else:
            self._devices = self._auto_detect()

    @staticmethod
    def _auto_detect() -> list[torch.device]:
        """Auto-detect available GPU devices."""
        devices: list[torch.device] = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(torch.device(f"cuda:{i}"))
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append(torch.device("mps"))
        if not devices:
            devices.append(torch.device("cpu"))
        return devices

    @property
    def device_count(self) -> int:
        """Number of available devices."""
        return len(self._devices)

    @property
    def devices(self) -> list[torch.device]:
        """List of available devices."""
        return list(self._devices)

    @property
    def is_multi_gpu(self) -> bool:
        """True if more than one GPU is available."""
        return self.device_count > 1 and self._devices[0].type != "cpu"

    def device_for_partition(self, partition_id: int) -> torch.device:
        """Return the device for a given partition.

        Partitions are round-robin assigned to devices.

        Args:
            partition_id: Zero-based partition index.

        Returns:
            torch.device for this partition.
        """
        return self._devices[partition_id % self.device_count]

    def __repr__(self) -> str:
        return (
            f"MultiGPUManager(devices={self._devices}, "
            f"count={self.device_count})"
        )


# ---------------------------------------------------------------------------
# Mesh partitioning
# ---------------------------------------------------------------------------


def partition_mesh(
    n_cells: int,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    n_partitions: int,
    cell_centres: torch.Tensor | None = None,
    *,
    device: torch.device | None = None,
) -> list[MeshPartition]:
    """Partition a mesh into *n_partitions* for domain decomposition.

    Uses a simple geometric bisection strategy along the longest axis
    (if ``cell_centres`` is provided) or a round-robin assignment based
    on cell index.

    Each partition gets:

    - Its subset of cells (global indices)
    - Remapped owner/neighbour arrays (local indices)
    - Boundary cell list (cells adjacent to another partition)

    Args:
        n_cells: Total number of cells.
        owner: ``(n_internal_faces,)`` global owner indices.
        neighbour: ``(n_internal_faces,)`` global neighbour indices.
        n_partitions: Number of partitions to create.
        cell_centres: ``(n_cells, 3)`` cell centre coordinates.  If
            ``None``, uses round-robin index assignment.
        device: Target device for output tensors.

    Returns:
        List of :class:`MeshPartition` objects, one per partition.
    """
    device = device or get_device()
    owner = owner.to(device="cpu", dtype=INDEX_DTYPE)
    neighbour = neighbour.to(device="cpu", dtype=INDEX_DTYPE)

    if n_partitions <= 0:
        raise ValueError(f"n_partitions must be >= 1, got {n_partitions}")

    # --- Assign cells to partitions ---
    if cell_centres is not None and n_partitions > 1:
        # Geometric bisection along longest axis
        centres = cell_centres.to(device="cpu")
        mins = centres.min(dim=0).values
        maxs = centres.max(dim=0).values
        extents = maxs - mins
        axis = int(extents.argmax().item())

        # Sort cells by coordinate along the longest axis
        sorted_indices = centres[:, axis].argsort()
        cells_per_part = n_cells // n_partitions
        remainder = n_cells % n_partitions

        partition_of_cell = torch.zeros(n_cells, dtype=INDEX_DTYPE)
        offset = 0
        for p in range(n_partitions):
            count = cells_per_part + (1 if p < remainder else 0)
            partition_of_cell[sorted_indices[offset:offset + count]] = p
            offset += count
    else:
        # Round-robin assignment
        partition_of_cell = torch.arange(n_cells, dtype=INDEX_DTYPE) % n_partitions

    # --- Build partitions ---
    partitions: list[MeshPartition] = []
    n_internal = int(owner.shape[0])

    for p in range(n_partitions):
        # Global cell indices for this partition
        cell_mask = partition_of_cell == p
        global_cell_indices = torch.where(cell_mask)[0]

        # Build global-to-local mapping
        global_to_local = torch.full((n_cells,), -1, dtype=INDEX_DTYPE)
        global_to_local[global_cell_indices] = torch.arange(
            len(global_cell_indices), dtype=INDEX_DTYPE
        )

        # Find internal faces within this partition
        owner_part = partition_of_cell[owner]
        neigh_part = partition_of_cell[neighbour]
        internal_mask = (owner_part == p) & (neigh_part == p)
        face_indices = torch.where(internal_mask)[0]

        local_owner = global_to_local[owner[face_indices]]
        local_neighbour = global_to_local[neighbour[face_indices]]

        # Find boundary cells (cells adjacent to another partition)
        boundary_mask = ((owner_part == p) & (neigh_part != p)) | (
            (neigh_part == p) & (owner_part != p)
        )
        boundary_faces = torch.where(boundary_mask)[0]
        boundary_cells_list: list[int] = []
        for fi in boundary_faces:
            fi = int(fi.item())
            if int(owner_part[fi].item()) == p:
                boundary_cells_list.append(int(owner[fi].item()))
            if int(neigh_part[fi].item()) == p:
                boundary_cells_list.append(int(neighbour[fi].item()))
        boundary_cells = torch.unique(
            torch.tensor(boundary_cells_list, dtype=INDEX_DTYPE)
        )

        partitions.append(MeshPartition(
            partition_id=p,
            cell_indices=global_cell_indices.to(device=device),
            owner=local_owner.to(device=device),
            neighbour=local_neighbour.to(device=device),
            boundary_cells=boundary_cells.to(device=device),
            device=device,
        ))

    return partitions


# ---------------------------------------------------------------------------
# Inter-GPU communication
# ---------------------------------------------------------------------------


class GpuCommunicator:
    """Inter-GPU communication wrapper using ``torch.distributed``.

    Provides a simplified API for the collective operations needed by
    domain-decomposed CFD solvers:

    - :meth:`all_gather_field` — gather field fragments from all partitions
    - :meth:`all_reduce_sum` — sum a tensor across all processes
    - :meth:`exchange_halo` — exchange halo (ghost) cell values

    If ``torch.distributed`` is not initialised, operations are no-ops
    (single-process mode).

    Usage::

        comm = GpuCommunicator()
        if comm.is_distributed:
            comm.all_gather_field(local_field)
    """

    def __init__(self) -> None:
        self._distributed = torch.distributed.is_available()
        self._initialized = (
            self._distributed and torch.distributed.is_initialized()
        )

    @property
    def is_distributed(self) -> bool:
        """True if torch.distributed is initialised."""
        return self._initialized

    @property
    def world_size(self) -> int:
        """Number of processes in the distributed group."""
        if not self._initialized:
            return 1
        return torch.distributed.get_world_size()

    @property
    def rank(self) -> int:
        """Current process rank."""
        if not self._initialized:
            return 0
        return torch.distributed.get_rank()

    def all_gather_field(
        self, local_tensor: torch.Tensor
    ) -> list[torch.Tensor]:
        """Gather tensors from all processes.

        Args:
            local_tensor: Tensor from this process.

        Returns:
            List of tensors, one per process.  In single-process mode,
            returns ``[local_tensor]``.
        """
        if not self._initialized:
            return [local_tensor]

        gathered = [
            torch.empty_like(local_tensor)
            for _ in range(self.world_size)
        ]
        torch.distributed.all_gather(gathered, local_tensor)
        return gathered

    def all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sum a tensor across all processes.

        Args:
            tensor: Tensor to reduce (modified in-place).

        Returns:
            The reduced tensor (same object, modified in-place).
            In single-process mode, returns *tensor* unchanged.
        """
        if not self._initialized:
            return tensor

        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        return tensor

    def exchange_halo(
        self,
        field: torch.Tensor,
        partition: MeshPartition,
        partitions: list[MeshPartition],
    ) -> torch.Tensor:
        """Exchange halo (ghost) cell values between partitions.

        For each boundary cell in *partition*, receives the current value
        from the owning partition and overwrites the local copy.

        Args:
            field: ``(n_local_cells,)`` field values on this partition.
            partition: This partition's metadata.
            partitions: All partitions (including this one).

        Returns:
            Updated field with halo values refreshed.
        """
        if not self._initialized or len(partitions) <= 1:
            return field

        # Simplified: in a real implementation this would use point-to-point
        # send/recv between adjacent partitions.  For now, use all_gather
        # as a fallback.
        all_fields = self.all_gather_field(field)

        # Update boundary cells from the owning partition's field
        for bc_global in partition.boundary_cells:
            bc_global = int(bc_global.item())
            # Find which partition owns this cell
            for other_part in partitions:
                if other_part.partition_id == partition.partition_id:
                    continue
                mask = other_part.cell_indices == bc_global
                if mask.any():
                    local_idx_in_other = torch.where(mask)[0][0]
                    # Find local index in this partition
                    local_mask = partition.cell_indices == bc_global
                    if local_mask.any():
                        local_idx = torch.where(local_mask)[0][0]
                        field[local_idx] = all_fields[
                            other_part.partition_id
                        ][local_idx_in_other]
                    break

        return field

    def __repr__(self) -> str:
        return (
            f"GpuCommunicator(distributed={self._initialized}, "
            f"world_size={self.world_size}, rank={self.rank})"
        )


# ---------------------------------------------------------------------------
# Multi-GPU Matrix
# ---------------------------------------------------------------------------


class MultiGPUMatrix:
    """LDU matrix partitioned across multiple GPUs.

    Wraps :class:`~pyfoam.core.ldu_matrix.LduMatrix` with domain
    decomposition: each partition's matrix lives on its assigned GPU.

    On single-GPU systems, this degrades to a single partition on one device.

    Usage::

        partitions = partition_mesh(n_cells, owner, neighbour, n_gpus)
        mgm = MultiGPUManager()
        multi_mat = MultiGPUMatrix(partitions, mgm)

        # Solve on each partition
        for p in multi_mat.partitions:
            local_mat = multi_mat.get_matrix(p.partition_id)
            # ... solve locally ...
    """

    def __init__(
        self,
        partitions: list[MeshPartition],
        manager: MultiGPUManager | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Create a multi-GPU matrix from mesh partitions.

        Args:
            partitions: List of :class:`MeshPartition` objects.
            manager: Multi-GPU manager.  If ``None``, creates a default one.
            dtype: Floating-point dtype.  Defaults to global config.
        """
        from pyfoam.core.ldu_matrix import LduMatrix

        self._partitions = partitions
        self._manager = manager or MultiGPUManager()
        self._dtype = dtype or get_default_dtype()
        self._communicator = GpuCommunicator()

        # Build one LduMatrix per partition on its assigned device
        self._matrices: dict[int, LduMatrix] = {}
        for part in partitions:
            device = self._manager.device_for_partition(part.partition_id)
            n_local = int(part.cell_indices.shape[0])
            mat = LduMatrix(
                n_local,
                part.owner.to(device=device),
                part.neighbour.to(device=device),
                device=device,
                dtype=self._dtype,
            )
            self._matrices[part.partition_id] = mat

    @property
    def partitions(self) -> list[MeshPartition]:
        """All mesh partitions."""
        return self._partitions

    @property
    def manager(self) -> MultiGPUManager:
        """The multi-GPU manager."""
        return self._manager

    @property
    def communicator(self) -> GpuCommunicator:
        """The inter-GPU communicator."""
        return self._communicator

    @property
    def n_partitions(self) -> int:
        """Number of partitions."""
        return len(self._partitions)

    def get_matrix(self, partition_id: int):
        """Return the LDU matrix for a specific partition.

        Args:
            partition_id: Zero-based partition index.

        Returns:
            :class:`~pyfoam.core.ldu_matrix.LduMatrix` for this partition.
        """
        return self._matrices[partition_id]

    def set_coefficients(
        self,
        partition_id: int,
        diag: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> None:
        """Set matrix coefficients for a partition.

        Args:
            partition_id: Partition index.
            diag: ``(n_local_cells,)`` diagonal coefficients.
            lower: ``(n_local_internal_faces,)`` lower coefficients.
            upper: ``(n_local_internal_faces,)`` upper coefficients.
        """
        mat = self._matrices[partition_id]
        mat.diag = diag
        mat.lower = lower
        mat.upper = upper

    def Ax(
        self, partition_id: int, x: torch.Tensor
    ) -> torch.Tensor:
        """Compute local matrix-vector product for a partition.

        Args:
            partition_id: Partition index.
            x: ``(n_local_cells,)`` input vector.

        Returns:
            ``(n_local_cells,)`` result.
        """
        return self._matrices[partition_id].Ax(x)

    def Ax_with_halo(
        self,
        partition_id: int,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Ax with halo exchange for cross-partition coupling.

        First exchanges halo values, then computes the local Ax.

        Args:
            partition_id: Partition index.
            x: ``(n_local_cells,)`` input vector (includes halo slots).

        Returns:
            ``(n_local_cells,)`` result with halo contributions.
        """
        part = self._partitions[partition_id]
        x = self._communicator.exchange_halo(
            x, part, self._partitions
        )
        return self._matrices[partition_id].Ax(x)

    def __repr__(self) -> str:
        devices = [
            str(self._manager.device_for_partition(p.partition_id))
            for p in self._partitions
        ]
        return (
            f"MultiGPUMatrix(n_partitions={self.n_partitions}, "
            f"devices={devices})"
        )
