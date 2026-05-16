"""
Parallel field operations — gather, scatter, and global reduction.

Provides parallel-aware wrappers around :class:`~pyfoam.fields.GeometricField`
that handle halo exchange and MPI reductions.

Key operations
--------------
- **Gather**: collect distributed field values into a global array on one rank.
- **Scatter**: distribute a global array to local subdomain fields.
- **Global sum / max / min**: reduce local contributions across all ranks.
- **Update halos**: refresh ghost cell values via halo exchange.

All tensors respect the global device/dtype from :mod:`pyfoam.core`.

Usage::

    from pyfoam.parallel.parallel_field import ParallelField

    pf = ParallelField(local_field, halo_exchange, subdomain)
    pf.update_halos()          # refresh ghost cells
    total = pf.global_sum()    # MPI allreduce sum
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.parallel.decomposition import SubDomain
from pyfoam.parallel.processor_patch import HaloExchange

# Try to import mpi4py
try:
    from mpi4py import MPI as _MPI
    _MPI_AVAILABLE = True
except ImportError:
    _MPI = None  # type: ignore[assignment]
    _MPI_AVAILABLE = False


__all__ = ["ParallelField"]


class ParallelField:
    """Parallel-aware field wrapper.

    Wraps a local field tensor with halo exchange and global reduction
    capabilities.

    Parameters
    ----------
    field : torch.Tensor
        Local field tensor ``(n_cells,)`` or ``(n_cells, dim)``.
    halo : HaloExchange
        Halo exchange manager for ghost cell communication.
    subdomain : SubDomain
        The subdomain this field belongs to.
    comm : mpi4py.MPI.Intracomm, optional
        MPI communicator.  Defaults to ``MPI.COMM_WORLD``.

    Attributes
    ----------
    local : torch.Tensor
        The local field tensor (owned + ghost cells).
    n_owned : int
        Number of owned cells.
    """

    def __init__(
        self,
        field: torch.Tensor,
        halo: HaloExchange,
        subdomain: SubDomain,
        comm: object | None = None,
    ) -> None:
        self._field = field
        self._halo = halo
        self._subdomain = subdomain
        self._device = field.device
        self._dtype = field.dtype

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
    def local(self) -> torch.Tensor:
        """The local field tensor."""
        return self._field

    @property
    def n_owned(self) -> int:
        """Number of owned cells."""
        return self._subdomain.n_owned_cells

    @property
    def owned_values(self) -> torch.Tensor:
        """Values at owned cells only (no ghost cells)."""
        return self._field[: self.n_owned]

    @property
    def ghost_values(self) -> torch.Tensor:
        """Values at ghost cells."""
        return self._field[self.n_owned :]

    # ------------------------------------------------------------------
    # Halo exchange
    # ------------------------------------------------------------------

    def update_halos(self, all_fields: dict[int, torch.Tensor] | None = None) -> None:
        """Refresh ghost cell values via halo exchange.

        After this call, ghost cells contain the latest values from the
        owning processor.

        Args:
            all_fields: Dict mapping processor rank → field tensor.
                Used in serial fallback to access other processors' fields.
        """
        self._halo.exchange(self._field, all_fields=all_fields)

    # ------------------------------------------------------------------
    # Global reductions
    # ------------------------------------------------------------------

    def global_sum(self) -> torch.Tensor:
        """Compute the global sum of owned cell values.

        Returns:
            Scalar tensor with the sum across all processors.
        """
        local_sum = self.owned_values.sum()
        return self._allreduce(local_sum, "sum")

    def global_max(self) -> torch.Tensor:
        """Compute the global maximum of owned cell values.

        Returns:
            Scalar tensor with the max across all processors.
        """
        local_max = self.owned_values.max()
        return self._allreduce(local_max, "max")

    def global_min(self) -> torch.Tensor:
        """Compute the global minimum of owned cell values.

        Returns:
            Scalar tensor with the min across all processors.
        """
        local_min = self.owned_values.min()
        return self._allreduce(local_min, "min")

    def global_mean(self) -> torch.Tensor:
        """Compute the global mean of owned cell values.

        Returns:
            Scalar tensor with the mean across all processors.
        """
        local_sum = self.owned_values.sum()
        local_count = torch.tensor(
            self.n_owned, device=self._device, dtype=self._dtype
        )
        global_sum = self._allreduce(local_sum, "sum")
        global_count = self._allreduce(local_count, "sum")
        return global_sum / global_count.clamp(min=1)

    # ------------------------------------------------------------------
    # Gather / Scatter
    # ------------------------------------------------------------------

    def gather_to_root(self, root: int = 0) -> torch.Tensor | None:
        """Gather all owned values to root processor.

        Args:
            root: Rank to gather onto.

        Returns:
            On root: concatenated tensor of all owned values.
            On other ranks: ``None``.
        """
        owned = self.owned_values.detach().cpu().numpy()

        if self._comm is not None and _MPI_AVAILABLE:
            gathered = self._comm.gather(owned, root=root)  # type: ignore[union-attr]
            if self._rank == root:
                import numpy as np
                return torch.from_numpy(np.concatenate(gathered)).to(
                    device=self._device, dtype=self._dtype
                )
            return None
        else:
            # Serial mode: just return a copy
            return self.owned_values.clone()

    def scatter_from_root(self, global_values: torch.Tensor, root: int = 0) -> None:
        """Distribute global values from root to local subdomains.

        Args:
            global_values: Full global field on root rank (ignored on others).
            root: Rank that holds the global values.
        """
        if self._comm is not None and _MPI_AVAILABLE:
            local_count = self.n_owned
            counts = self._comm.allgather(local_count)  # type: ignore[union-attr]
            offsets = [0]
            for c in counts[:-1]:
                offsets.append(offsets[-1] + c)

            if self._rank == root:
                chunks = []
                for i, (off, cnt) in enumerate(zip(offsets, counts)):
                    chunks.append(global_values[off:off + cnt].cpu().numpy())
            else:
                chunks = None

            local_np = self._comm.scatter(chunks, root=root)  # type: ignore[union-attr]
            local_t = torch.from_numpy(local_np).to(
                device=self._device, dtype=self._dtype
            )
            self._field[: self.n_owned] = local_t
        else:
            # Serial mode
            self._field[: self.n_owned] = global_values[: self.n_owned]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _rank(self) -> int:
        if self._comm is not None:
            return self._comm.Get_rank()  # type: ignore[union-attr]
        return 0

    def _allreduce(self, value: torch.Tensor, op: str) -> torch.Tensor:
        """MPI allreduce with serial fallback."""
        if self._comm is not None and _MPI_AVAILABLE:
            value_np = value.detach().cpu().numpy()
            if op == "sum":
                result_np = self._comm.allreduce(value_np, op=_MPI.SUM)  # type: ignore[union-attr]
            elif op == "max":
                result_np = self._comm.allreduce(value_np, op=_MPI.MAX)  # type: ignore[union-attr]
            elif op == "min":
                result_np = self._comm.allreduce(value_np, op=_MPI.MIN)  # type: ignore[union-attr]
            else:
                raise ValueError(f"Unknown reduction op: {op}")
            return torch.tensor(result_np, device=self._device, dtype=self._dtype)
        else:
            # Serial mode: value is already the global value
            return value.clone()

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ParallelField(shape={tuple(self._field.shape)}, "
            f"n_owned={self.n_owned}, "
            f"device={self._device}, dtype={self._dtype})"
        )
