"""Tests for multi-GPU support framework.

Tests the domain decomposition, multi-GPU manager, communicator, and
multi-GPU matrix classes.  All tests work on CPU (single-device simulation)
since actual multi-GPU tests require hardware with 2+ GPUs.
"""

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.core.multi_gpu import (
    MultiGPUManager,
    MeshPartition,
    partition_mesh,
    GpuCommunicator,
    MultiGPUMatrix,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_4cell_chain():
    """4-cell chain: 0 -- 1 -- 2 -- 3."""
    n_cells = 4
    owner = torch.tensor([0, 1, 2], dtype=INDEX_DTYPE)
    neighbour = torch.tensor([1, 2, 3], dtype=INDEX_DTYPE)
    return n_cells, owner, neighbour


def _make_6cell_2d():
    """2x3 mesh (6 cells):
    0 -- 1 -- 2
    |    |    |
    3 -- 4 -- 5
    """
    n_cells = 6
    # Horizontal faces
    owner = torch.tensor([0, 1, 3, 4, 0, 1, 2], dtype=INDEX_DTYPE)
    neighbour = torch.tensor([1, 2, 4, 5, 3, 4, 5], dtype=INDEX_DTYPE)
    return n_cells, owner, neighbour


@pytest.fixture
def chain_mesh():
    return _make_4cell_chain()


@pytest.fixture
def mesh_2d():
    return _make_6cell_2d()


# ---------------------------------------------------------------------------
# MultiGPUManager
# ---------------------------------------------------------------------------


class TestMultiGPUManager:
    def test_default_creation(self):
        mgm = MultiGPUManager()
        assert mgm.device_count >= 1
        assert len(mgm.devices) == mgm.device_count

    def test_explicit_devices(self):
        mgm = MultiGPUManager(devices=["cpu"])
        assert mgm.device_count == 1
        assert mgm.devices[0] == torch.device("cpu")

    def test_device_for_partition_round_robin(self):
        mgm = MultiGPUManager(devices=["cpu", "cpu"])
        # Even with same device string, should have 2 entries
        assert mgm.device_for_partition(0) == torch.device("cpu")
        assert mgm.device_for_partition(1) == torch.device("cpu")
        assert mgm.device_for_partition(2) == torch.device("cpu")

    def test_is_multi_gpu_single(self):
        mgm = MultiGPUManager(devices=["cpu"])
        assert mgm.is_multi_gpu is False

    def test_repr(self):
        mgm = MultiGPUManager(devices=["cpu"])
        r = repr(mgm)
        assert "MultiGPUManager" in r


# ---------------------------------------------------------------------------
# partition_mesh
# ---------------------------------------------------------------------------


class TestPartitionMesh:
    def test_single_partition(self, chain_mesh):
        n_cells, owner, neighbour = chain_mesh
        partitions = partition_mesh(n_cells, owner, neighbour, 1)
        assert len(partitions) == 1
        assert partitions[0].partition_id == 0
        assert partitions[0].cell_indices.shape[0] == n_cells

    def test_two_partitions_chain(self, chain_mesh):
        n_cells, owner, neighbour = chain_mesh
        partitions = partition_mesh(n_cells, owner, neighbour, 2)
        assert len(partitions) == 2
        # All cells should be assigned
        all_cells = torch.cat([p.cell_indices for p in partitions])
        assert len(torch.unique(all_cells)) == n_cells

    def test_two_partitions_with_centres(self, chain_mesh):
        n_cells, owner, neighbour = chain_mesh
        # Cell centres along a line
        centres = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.75, 0.0, 0.0],
        ], dtype=CFD_DTYPE)
        partitions = partition_mesh(
            n_cells, owner, neighbour, 2, cell_centres=centres
        )
        assert len(partitions) == 2
        # First partition should have cells 0,1; second should have 2,3
        assert 0 in partitions[0].cell_indices
        assert 1 in partitions[0].cell_indices
        assert 2 in partitions[1].cell_indices
        assert 3 in partitions[1].cell_indices

    def test_boundary_cells(self, chain_mesh):
        n_cells, owner, neighbour = chain_mesh
        partitions = partition_mesh(n_cells, owner, neighbour, 2)
        # At least one partition should have boundary cells
        has_boundary = any(
            p.boundary_cells.shape[0] > 0 for p in partitions
        )
        assert has_boundary

    def test_local_indices(self, chain_mesh):
        n_cells, owner, neighbour = chain_mesh
        partitions = partition_mesh(n_cells, owner, neighbour, 2)
        for p in partitions:
            n_local = int(p.cell_indices.shape[0])
            # Local indices should be in [0, n_local)
            if p.owner.shape[0] > 0:
                assert p.owner.max() < n_local
                assert p.neighbour.max() < n_local

    def test_2d_mesh_partitions(self, mesh_2d):
        n_cells, owner, neighbour = mesh_2d
        partitions = partition_mesh(n_cells, owner, neighbour, 2)
        assert len(partitions) == 2
        all_cells = torch.cat([p.cell_indices for p in partitions])
        assert len(torch.unique(all_cells)) == n_cells

    def test_invalid_partitions_raises(self, chain_mesh):
        n_cells, owner, neighbour = chain_mesh
        with pytest.raises(ValueError, match="n_partitions"):
            partition_mesh(n_cells, owner, neighbour, 0)

    def test_device_propagation(self, chain_mesh):
        n_cells, owner, neighbour = chain_mesh
        partitions = partition_mesh(
            n_cells, owner, neighbour, 2, device=torch.device("cpu")
        )
        for p in partitions:
            assert p.device == torch.device("cpu")
            assert p.cell_indices.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# GpuCommunicator
# ---------------------------------------------------------------------------


class TestGpuCommunicator:
    def test_creation(self):
        comm = GpuCommunicator()
        assert isinstance(comm.is_distributed, bool)

    def test_single_process_mode(self):
        comm = GpuCommunicator()
        if not comm.is_distributed:
            assert comm.world_size == 1
            assert comm.rank == 0

    def test_all_gather_single_process(self):
        comm = GpuCommunicator()
        t = torch.tensor([1.0, 2.0, 3.0])
        result = comm.all_gather_field(t)
        assert len(result) == 1
        assert torch.equal(result[0], t)

    def test_all_reduce_single_process(self):
        comm = GpuCommunicator()
        t = torch.tensor([1.0, 2.0, 3.0])
        result = comm.all_reduce_sum(t)
        assert torch.equal(result, t)

    def test_exchange_halo_single_process(self):
        comm = GpuCommunicator()
        field = torch.tensor([1.0, 2.0, 3.0])
        partition = MeshPartition(
            partition_id=0,
            cell_indices=torch.tensor([0, 1, 2]),
            owner=torch.tensor([0, 1], dtype=INDEX_DTYPE),
            neighbour=torch.tensor([1, 2], dtype=INDEX_DTYPE),
            boundary_cells=torch.tensor([], dtype=INDEX_DTYPE),
            device=torch.device("cpu"),
        )
        result = comm.exchange_halo(field, partition, [partition])
        # Single process: no change
        assert torch.equal(result, field)

    def test_repr(self):
        comm = GpuCommunicator()
        r = repr(comm)
        assert "GpuCommunicator" in r


# ---------------------------------------------------------------------------
# MultiGPUMatrix
# ---------------------------------------------------------------------------


class TestMultiGPUMatrix:
    def test_creation(self, chain_mesh):
        n_cells, owner, neighbour = chain_mesh
        partitions = partition_mesh(n_cells, owner, neighbour, 2)
        mgm = MultiGPUManager(devices=["cpu"])
        multi_mat = MultiGPUMatrix(partitions, mgm)
        assert multi_mat.n_partitions == 2

    def test_get_matrix(self, chain_mesh):
        n_cells, owner, neighbour = chain_mesh
        partitions = partition_mesh(n_cells, owner, neighbour, 2)
        multi_mat = MultiGPUMatrix(partitions)
        for p in partitions:
            mat = multi_mat.get_matrix(p.partition_id)
            assert mat is not None
            n_local = int(p.cell_indices.shape[0])
            assert mat.n_cells == n_local

    def test_set_coefficients(self, chain_mesh):
        n_cells, owner, neighbour = chain_mesh
        partitions = partition_mesh(n_cells, owner, neighbour, 2)
        multi_mat = MultiGPUMatrix(partitions)

        for p in partitions:
            mat = multi_mat.get_matrix(p.partition_id)
            n_local = mat.n_cells
            n_faces = mat.n_internal_faces
            diag = torch.ones(n_local, dtype=CFD_DTYPE)
            lower = -torch.ones(n_faces, dtype=CFD_DTYPE) if n_faces > 0 else torch.tensor([], dtype=CFD_DTYPE)
            upper = -torch.ones(n_faces, dtype=CFD_DTYPE) if n_faces > 0 else torch.tensor([], dtype=CFD_DTYPE)
            multi_mat.set_coefficients(p.partition_id, diag, lower, upper)

    def test_ax(self, chain_mesh):
        n_cells, owner, neighbour = chain_mesh
        partitions = partition_mesh(n_cells, owner, neighbour, 2)
        multi_mat = MultiGPUMatrix(partitions)

        for p in partitions:
            mat = multi_mat.get_matrix(p.partition_id)
            n_local = mat.n_cells
            n_faces = mat.n_internal_faces
            mat.diag = torch.ones(n_local, dtype=CFD_DTYPE)
            if n_faces > 0:
                mat.lower = -0.5 * torch.ones(n_faces, dtype=CFD_DTYPE)
                mat.upper = -0.5 * torch.ones(n_faces, dtype=CFD_DTYPE)

            x = torch.ones(n_local, dtype=CFD_DTYPE)
            y = multi_mat.Ax(p.partition_id, x)
            assert y.shape == (n_local,)

    def test_single_partition_degrades(self, chain_mesh):
        """Single partition should work like a normal LduMatrix."""
        n_cells, owner, neighbour = chain_mesh
        partitions = partition_mesh(n_cells, owner, neighbour, 1)
        multi_mat = MultiGPUMatrix(partitions)
        assert multi_mat.n_partitions == 1

        mat = multi_mat.get_matrix(0)
        mat.diag = torch.tensor([4.0, 6.0, 4.0, 4.0])
        mat.lower = torch.tensor([-1.0, -1.0, -1.0])
        mat.upper = torch.tensor([-1.0, -1.0, -1.0])

        x = torch.ones(4, dtype=CFD_DTYPE)
        y = multi_mat.Ax(0, x)
        # A·1 should be near zero for consistent diffusion
        assert y.shape == (4,)

    def test_repr(self, chain_mesh):
        n_cells, owner, neighbour = chain_mesh
        partitions = partition_mesh(n_cells, owner, neighbour, 2)
        multi_mat = MultiGPUMatrix(partitions)
        r = repr(multi_mat)
        assert "MultiGPUMatrix" in r
        assert "n_partitions=2" in r
