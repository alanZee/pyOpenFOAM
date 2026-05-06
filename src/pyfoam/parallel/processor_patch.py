"""
Processor patches and halo exchange for parallel CFD.

Manages ghost cells and inter-processor communication via mpi4py.

Key concepts
------------
- **Processor patch**: a boundary patch whose faces connect to cells owned by
  a different MPI rank.
- **Ghost cells**: cells in the local subdomain that are owned by a neighbour
  rank.  Their values must be received via halo exchange before computing
  stencil operations (gradient, divergence, etc.).
- **Halo exchange**: two-step communication pattern — each processor packs
  boundary values into a send buffer, sends to the neighbour, receives into
  a receive buffer, and unpacks into ghost cell positions.

All tensors respect the global device/dtype from :mod:`pyfoam.core`.

Usage::

    from pyfoam.parallel.processor_patch import ProcessorPatch, HaloExchange

    patch = ProcessorPatch(local_cells, remote_cells, neighbour_rank)
    halo = HaloExchange([patch], comm=comm)
    halo.exchange(field_values)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE

__all__ = ["ProcessorPatch", "HaloExchange"]

# Try to import mpi4py; provide serial fallback
try:
    from mpi4py import MPI as _MPI
    _MPI_AVAILABLE = True
except ImportError:
    _MPI = None  # type: ignore[assignment]
    _MPI_AVAILABLE = False


# ---------------------------------------------------------------------------
# ProcessorPatch — describes one neighbour's ghost cells
# ---------------------------------------------------------------------------


@dataclass
class ProcessorPatch:
    """A processor boundary patch linking local ghost cells to remote cells.

    Attributes
    ----------
    name : str
        Patch name (e.g. ``"procBoundary0Patch"``).
    neighbour_rank : int
        MPI rank of the neighbour processor.
    local_ghost_cells : torch.Tensor
        Indices of ghost cells in the local subdomain that belong to this
        neighbour.  Shape ``(n_ghost_cells,)``.
    remote_cells : torch.Tensor
        Corresponding cell indices in the neighbour's local numbering.
        Shape ``(n_ghost_cells,)``.
    """

    name: str
    neighbour_rank: int
    local_ghost_cells: torch.Tensor
    remote_cells: torch.Tensor

    @property
    def n_ghost_cells(self) -> int:
        """Number of ghost cells in this patch."""
        return self.local_ghost_cells.shape[0]


# ---------------------------------------------------------------------------
# HaloExchange — manages inter-processor communication
# ---------------------------------------------------------------------------


class HaloExchange:
    """Halo exchange for ghost cell communication.

    Coordinates send/receive of boundary values between neighbouring
    processors using mpi4py.

    Parameters
    ----------
    patches : list[ProcessorPatch]
        Processor patches for this subdomain.
    comm : mpi4py.MPI.Intracomm, optional
        MPI communicator.  Defaults to ``MPI.COMM_WORLD``.
        If mpi4py is not available, uses serial fallback (copy-back).

    Attributes
    ----------
    send_buffers : dict[int, torch.Tensor]
        Per-neighbour send buffers.
    recv_buffers : dict[int, torch.Tensor]
        Per-neighbour receive buffers.
    """

    def __init__(
        self,
        patches: list[ProcessorPatch],
        comm: object | None = None,
    ) -> None:
        self._patches = patches
        self._device = get_device()
        self._dtype = get_default_dtype()

        # MPI communicator
        if _MPI_AVAILABLE and comm is not None:
            self._comm = comm
        elif _MPI_AVAILABLE:
            self._comm = _MPI.COMM_WORLD  # type: ignore[union-attr]
        else:
            self._comm = None

        self._rank = self._get_rank()
        self._size = self._get_size()

        # Pre-allocate buffers per neighbour
        self.send_buffers: dict[int, torch.Tensor] = {}
        self.recv_buffers: dict[int, torch.Tensor] = {}
        for patch in patches:
            n = patch.n_ghost_cells
            self.send_buffers[patch.neighbour_rank] = torch.zeros(
                n, device=self._device, dtype=self._dtype
            )
            self.recv_buffers[patch.neighbour_rank] = torch.zeros(
                n, device=self._device, dtype=self._dtype
            )

    # ------------------------------------------------------------------
    # MPI helpers
    # ------------------------------------------------------------------

    def _get_rank(self) -> int:
        if self._comm is not None:
            return self._comm.Get_rank()  # type: ignore[union-attr]
        return 0

    def _get_size(self) -> int:
        if self._comm is not None:
            return self._comm.Get_size()  # type: ignore[union-attr]
        return 1

    @property
    def rank(self) -> int:
        """MPI rank."""
        return self._rank

    @property
    def size(self) -> int:
        """MPI world size."""
        return self._size

    # ------------------------------------------------------------------
    # Halo exchange
    # ------------------------------------------------------------------

    def exchange(self, field_values: torch.Tensor) -> torch.Tensor:
        """Perform halo exchange on field values.

        Packs boundary values from *field_values* at owned cell positions
        into send buffers, communicates with neighbours, and writes received
        values into ghost cell positions.

        Args:
            field_values: ``(n_local_cells,)`` or ``(n_local_cells, ...)`` tensor
                containing both owned and ghost cell values.

        Returns:
            The same tensor with ghost cell values updated.
        """
        if not self._patches:
            return field_values

        # Pack send buffers: extract values at positions that the neighbour needs
        for patch in self._patches:
            # The neighbour needs values of cells that are ghost on their side
            # = owned cells on our side that correspond to their ghost cells
            # remote_cells gives the neighbour's local indices; we need our
            # local owned cells that back those ghost cells.
            # For the send direction: we send values of our owned cells
            # that the neighbour needs (our cells that are its ghosts).
            send_vals = field_values[patch.local_ghost_cells].clone()
            self.send_buffers[patch.neighbour_rank] = send_vals

        # Communicate
        if self._comm is not None and _MPI_AVAILABLE:
            self._exchange_mpi()
        else:
            self._exchange_serial(field_values)

        # Unpack receive buffers into ghost cell positions
        for patch in self._patches:
            recv_buf = self.recv_buffers[patch.neighbour_rank]
            field_values[patch.local_ghost_cells] = recv_buf

        return field_values

    def _exchange_mpi(self) -> None:
        """Non-blocking MPI send/receive."""
        requests: list = []

        for patch in self._patches:
            rank = patch.neighbour_rank
            send_buf = self.send_buffers[rank]
            recv_buf = self.recv_buffers[rank]

            # Convert to numpy for mpi4py
            send_np = send_buf.cpu().numpy()
            recv_np = recv_buf.cpu().numpy()

            # Non-blocking send
            req_send = self._comm.Isend(send_np, dest=rank, tag=rank)  # type: ignore[union-attr]
            requests.append(req_send)

            # Non-blocking receive
            req_recv = self._comm.Irecv(recv_np, source=rank, tag=self._rank)  # type: ignore[union-attr]
            requests.append(req_recv)

            # Store numpy reference for later copy back
            self.recv_buffers[rank] = torch.from_numpy(recv_np).to(
                device=self._device, dtype=self._dtype
            )

        # Wait for all communications to complete
        _MPI.Request.Waitall(requests)  # type: ignore[union-attr]

    def _exchange_serial(self, field_values: torch.Tensor) -> None:
        """Serial fallback: copy send buffer to receive buffer (self-loopback).

        In serial mode there is only one rank, so ghost cells are actually
        owned cells in the same mesh.  We simply copy the values back.
        """
        for patch in self._patches:
            # In serial mode, the ghost cell values are already in field_values
            self.recv_buffers[patch.neighbour_rank] = (
                self.send_buffers[patch.neighbour_rank].clone()
            )

    # ------------------------------------------------------------------
    # Vector field exchange
    # ------------------------------------------------------------------

    def exchange_vector(self, field_values: torch.Tensor) -> torch.Tensor:
        """Halo exchange for vector fields ``(n_cells, dim)``.

        Same as :meth:`exchange` but handles multi-dimensional fields.

        Args:
            field_values: ``(n_local_cells, dim)`` tensor.

        Returns:
            Updated tensor with ghost cell values.
        """
        return self.exchange(field_values)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        ranks = [p.neighbour_rank for p in self._patches]
        return f"HaloExchange(rank={self._rank}, neighbours={ranks})"
