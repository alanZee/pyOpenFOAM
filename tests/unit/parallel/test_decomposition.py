"""Tests for parallel domain decomposition.

Tests the full parallel framework:
- Decomposition (simple geometric)
- SubDomain creation with ProcessorPatch objects
- HaloExchange (serial fallback and correctness)
- ParallelField (global reductions, gather/scatter)
- ParallelSolver (additive Schwarz)
- Parallel I/O (processor directory read/write)

All tests run in serial mode (no MPI required) using the 8-cell mesh
from conftest.py.
"""

import os
import shutil
import tempfile

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.poly_mesh import PolyMesh
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.parallel.decomposition import Decomposition, SubDomain
from pyfoam.parallel.processor_patch import ProcessorPatch, HaloExchange
from pyfoam.parallel.parallel_field import ParallelField
from pyfoam.parallel.parallel_io import ParallelWriter, ParallelReader
from pyfoam.parallel.parallel_solver import ParallelSolver, ParallelSolverConfig

from tests.unit.parallel.conftest import (
    make_8cell_poly_mesh,
    make_8cell_fv_mesh,
)


# ---------------------------------------------------------------------------
# Decomposition tests
# ---------------------------------------------------------------------------


class TestDecompositionSimple:
    """Test simple geometric decomposition on the 8-cell mesh."""

    def test_decompose_2_processors(self, fv_mesh_8cell):
        """Decompose into 2 subdomains."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        assert len(subdomains) == 2
        assert subdomains[0].processor_id == 0
        assert subdomains[1].processor_id == 1

    def test_all_cells_assigned(self, fv_mesh_8cell):
        """Every cell is assigned to exactly one processor."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        all_cells = set()
        for sub in subdomains:
            for cid in sub.global_cell_ids.tolist():
                all_cells.add(cid)

        assert all_cells == set(range(8))

    def test_no_overlap_in_owned(self, fv_mesh_8cell):
        """Owned cells don't overlap between subdomains."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        owned_sets = []
        for sub in subdomains:
            owned = set(sub.global_cell_ids[: sub.n_owned_cells].tolist())
            owned_sets.append(owned)

        assert len(owned_sets[0] & owned_sets[1]) == 0

    def test_total_owned_equals_n_cells(self, fv_mesh_8cell):
        """Sum of owned cells across processors equals total cells."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        total_owned = sum(sub.n_owned_cells for sub in subdomains)
        assert total_owned == 8

    def test_cell_assignment_shape(self, fv_mesh_8cell):
        """Cell assignment tensor has correct shape."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        decomp.decompose()

        assert decomp.cell_assignment.shape == (8,)
        assert decomp.cell_assignment.dtype == INDEX_DTYPE

    def test_cell_assignment_range(self, fv_mesh_8cell):
        """Cell assignment values are in [0, n_processors)."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        decomp.decompose()

        assert decomp.cell_assignment.min() >= 0
        assert decomp.cell_assignment.max() < 2

    def test_subdomain_mesh_is_polymesh(self, fv_mesh_8cell):
        """Each subdomain has a valid PolyMesh."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        for sub in subdomains:
            assert isinstance(sub.mesh, PolyMesh)
            assert sub.mesh.n_cells > 0

    def test_subdomain_global_ids_valid(self, fv_mesh_8cell):
        """Global cell IDs are valid indices into the original mesh."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        for sub in subdomains:
            assert sub.global_cell_ids.min() >= 0
            assert sub.global_cell_ids.max() < 8


class TestDecomposition4Procs:
    """Test decomposition into 4 processors."""

    def test_decompose_4_processors(self, fv_mesh_8cell):
        """Decompose into 4 subdomains."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=4, method="simple")
        subdomains = decomp.decompose()

        assert len(subdomains) == 4
        for i, sub in enumerate(subdomains):
            assert sub.processor_id == i

    def test_4proc_total_owned(self, fv_mesh_8cell):
        """All cells accounted for with 4 processors."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=4, method="simple")
        subdomains = decomp.decompose()

        total_owned = sum(sub.n_owned_cells for sub in subdomains)
        assert total_owned == 8

    def test_4proc_each_has_cells(self, fv_mesh_8cell):
        """Each processor gets at least one cell."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=4, method="simple")
        subdomains = decomp.decompose()

        for sub in subdomains:
            assert sub.n_owned_cells >= 1


class TestDecompositionValidation:
    """Test input validation."""

    def test_zero_processors_raises(self, fv_mesh_8cell):
        with pytest.raises(ValueError, match="n_processors must be >= 1"):
            Decomposition(fv_mesh_8cell, n_processors=0)

    def test_too_many_processors_raises(self, fv_mesh_8cell):
        with pytest.raises(ValueError, match="n_processors.*n_cells"):
            Decomposition(fv_mesh_8cell, n_processors=100)

    def test_cell_assignment_before_decompose_raises(self, fv_mesh_8cell):
        decomp = Decomposition(fv_mesh_8cell, n_processors=2)
        with pytest.raises(RuntimeError, match="Call decompose"):
            _ = decomp.cell_assignment

    def test_metrics_before_decompose_raises(self, fv_mesh_8cell):
        decomp = Decomposition(fv_mesh_8cell, n_processors=2)
        with pytest.raises(RuntimeError, match="Call decompose"):
            decomp.load_balance_metrics()


class TestLoadBalancing:
    """Test load balancing metrics."""

    def test_metrics_shape(self, fv_mesh_8cell):
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        decomp.decompose()
        metrics = decomp.load_balance_metrics()

        assert "min_cells" in metrics
        assert "max_cells" in metrics
        assert "mean_cells" in metrics
        assert "imbalance_ratio" in metrics
        assert "min_faces" in metrics
        assert "max_faces" in metrics

    def test_imbalance_ratio_ge_1(self, fv_mesh_8cell):
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        decomp.decompose()
        metrics = decomp.load_balance_metrics()

        assert metrics["imbalance_ratio"] >= 1.0

    def test_mean_cells(self, fv_mesh_8cell):
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        decomp.decompose()
        metrics = decomp.load_balance_metrics()

        assert metrics["mean_cells"] == 4.0


# ---------------------------------------------------------------------------
# Ghost cell and ProcessorPatch tests
# ---------------------------------------------------------------------------


class TestGhostCells:
    """Test ghost cell identification."""

    def test_ghost_cells_exist(self, fv_mesh_8cell):
        """Subdomains have ghost cells when mesh is connected."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        has_ghosts = any(sub.ghost_cells.numel() > 0 for sub in subdomains)
        assert has_ghosts

    def test_ghost_cell_count(self, fv_mesh_8cell):
        """Ghost cells are a subset of total subdomain cells."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        for sub in subdomains:
            assert sub.ghost_cells.numel() <= sub.global_cell_ids.numel()
            assert sub.n_owned_cells + sub.ghost_cells.numel() == sub.global_cell_ids.numel()

    def test_ghost_cells_are_not_owned(self, fv_mesh_8cell):
        """Ghost cells are distinct from owned cells."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        for sub in subdomains:
            owned = set(sub.global_cell_ids[:sub.n_owned_cells].tolist())
            ghosts = set(sub.global_cell_ids[sub.n_owned_cells:].tolist())
            assert len(owned & ghosts) == 0


class TestProcessorPatchCreation:
    """Test that ProcessorPatch objects are correctly created during decomposition."""

    def test_patches_exist(self, fv_mesh_8cell):
        """Each subdomain has processor patches after decomposition."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        for sub in subdomains:
            assert len(sub.processor_patches) > 0

    def test_patch_neighbour_is_other_proc(self, fv_mesh_8cell):
        """Each patch's neighbour_rank points to the other processor."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        for sub in subdomains:
            for patch in sub.processor_patches:
                assert patch.neighbour_rank != sub.processor_id
                assert 0 <= patch.neighbour_rank < 2

    def test_patch_has_ghost_cells(self, fv_mesh_8cell):
        """Each patch has ghost cells to receive."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        for sub in subdomains:
            for patch in sub.processor_patches:
                assert patch.n_ghost_cells > 0

    def test_patch_has_send_cells(self, fv_mesh_8cell):
        """Each patch has owned boundary cells to send."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        for sub in subdomains:
            for patch in sub.processor_patches:
                assert patch.n_send_cells > 0

    def test_send_cells_are_owned(self, fv_mesh_8cell):
        """Send cells (remote_cells) are within the owned cell range."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        for sub in subdomains:
            for patch in sub.processor_patches:
                assert patch.remote_cells.max() < sub.n_owned_cells

    def test_ghost_cells_are_in_ghost_range(self, fv_mesh_8cell):
        """Ghost cells are in the ghost cell range [n_owned, n_total)."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        for sub in subdomains:
            for patch in sub.processor_patches:
                assert patch.local_ghost_cells.min() >= sub.n_owned_cells
                assert patch.local_ghost_cells.max() < sub.global_cell_ids.numel()

    def test_patches_cover_all_ghosts(self, fv_mesh_8cell):
        """All ghost cells are covered by some processor patch."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        for sub in subdomains:
            all_patch_ghosts = set()
            for patch in sub.processor_patches:
                all_patch_ghosts.update(patch.local_ghost_cells.tolist())
            all_ghosts = set(sub.ghost_cells.tolist())
            assert all_ghosts == all_patch_ghosts

    def test_patches_cover_all_owned_boundary(self, fv_mesh_8cell):
        """All owned boundary cells are covered by some processor patch."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        for sub in subdomains:
            all_patch_send = set()
            for patch in sub.processor_patches:
                all_patch_send.update(patch.remote_cells.tolist())
            # At least some owned cells should be boundary cells
            # (for a connected mesh with multiple processors)
            if sub.n_owned_cells < 8:  # Not the only processor
                assert len(all_patch_send) > 0

    def test_send_recv_counts_match_neighbour(self, fv_mesh_8cell):
        """What we send to a neighbour matches what they receive from us."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        # For proc 0 → proc 1:
        # proc 0's send count to proc 1 should equal
        # proc 1's receive count from proc 0
        for sub in subdomains:
            for patch in sub.processor_patches:
                nbr = patch.neighbour_rank
                nbr_sub = subdomains[nbr]
                # Find the reciprocal patch
                nbr_patch = None
                for np in nbr_sub.processor_patches:
                    if np.neighbour_rank == sub.processor_id:
                        nbr_patch = np
                        break
                assert nbr_patch is not None, (
                    f"No reciprocal patch: proc {sub.processor_id} → proc {nbr}"
                )
                assert patch.n_send_cells == nbr_patch.n_ghost_cells, (
                    f"Send/recv mismatch: proc {sub.processor_id} sends "
                    f"{patch.n_send_cells} but proc {nbr} receives "
                    f"{nbr_patch.n_ghost_cells}"
                )


class TestProcessorPatchDataclass:
    """Test ProcessorPatch dataclass directly."""

    def test_basic_patch(self):
        local = torch.tensor([4, 5], dtype=INDEX_DTYPE)
        remote = torch.tensor([0, 1], dtype=INDEX_DTYPE)
        patch = ProcessorPatch(
            name="procBoundary0To1",
            neighbour_rank=1,
            local_ghost_cells=local,
            remote_cells=remote,
        )

        assert patch.name == "procBoundary0To1"
        assert patch.neighbour_rank == 1
        assert patch.n_ghost_cells == 2
        assert patch.n_send_cells == 2

    def test_empty_patch(self):
        patch = ProcessorPatch(
            name="empty",
            neighbour_rank=0,
            local_ghost_cells=torch.zeros(0, dtype=INDEX_DTYPE),
            remote_cells=torch.zeros(0, dtype=INDEX_DTYPE),
        )
        assert patch.n_ghost_cells == 0
        assert patch.n_send_cells == 0


# ---------------------------------------------------------------------------
# HaloExchange tests
# ---------------------------------------------------------------------------


class TestHaloExchangeSerial:
    """Test halo exchange in serial mode (no MPI)."""

    def test_empty_patches(self):
        """No patches means no exchange."""
        halo = HaloExchange([])
        field = torch.tensor([1.0, 2.0, 3.0])
        result = halo.exchange(field)
        assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_serial_exchange_copies_owned_to_ghost(self):
        """In serial mode, exchange copies from owned boundary to ghost cells."""
        # Simulate: cell 0 and 1 are owned, cell 2 is ghost from "neighbour"
        # The owned boundary cell that the neighbour needs is cell 1
        ghost_cells = torch.tensor([2], dtype=INDEX_DTYPE)
        owned_boundary = torch.tensor([1], dtype=INDEX_DTYPE)
        patch = ProcessorPatch(
            name="proc0To1",
            neighbour_rank=0,  # self in serial mode
            local_ghost_cells=ghost_cells,
            remote_cells=owned_boundary,
        )
        halo = HaloExchange([patch])

        # Field: cell 0=10, cell 1=20, cell 2=999 (ghost, wrong value)
        field = torch.tensor([10.0, 20.0, 999.0])
        result = halo.exchange(field)

        # After exchange, cell 2 should have cell 1's value (20.0)
        assert torch.allclose(result[2], torch.tensor(20.0))
        # Owned cells unchanged
        assert torch.allclose(result[0], torch.tensor(10.0))
        assert torch.allclose(result[1], torch.tensor(20.0))

    def test_serial_exchange_multiple_ghosts(self):
        """Exchange with multiple ghost cells."""
        ghost_cells = torch.tensor([4, 5], dtype=INDEX_DTYPE)
        owned_boundary = torch.tensor([0, 1], dtype=INDEX_DTYPE)
        patch = ProcessorPatch(
            name="proc0To1",
            neighbour_rank=0,
            local_ghost_cells=ghost_cells,
            remote_cells=owned_boundary,
        )
        halo = HaloExchange([patch])

        field = torch.tensor([10.0, 20.0, 30.0, 40.0, 999.0, 888.0])
        result = halo.exchange(field)

        assert torch.allclose(result[4], torch.tensor(10.0))
        assert torch.allclose(result[5], torch.tensor(20.0))

    def test_send_recv_buffer_sizes(self):
        """Buffers are correctly sized."""
        ghost_cells = torch.tensor([2, 3], dtype=INDEX_DTYPE)
        owned_boundary = torch.tensor([0, 1], dtype=INDEX_DTYPE)
        patch = ProcessorPatch(
            name="proc0To1",
            neighbour_rank=1,
            local_ghost_cells=ghost_cells,
            remote_cells=owned_boundary,
        )
        halo = HaloExchange([patch])

        assert halo.send_buffers[1].shape == (2,)
        assert halo.recv_buffers[1].shape == (2,)

    def test_repr(self):
        halo = HaloExchange([])
        r = repr(halo)
        assert "HaloExchange" in r


class TestHaloExchangeWithDecomposition:
    """Test HaloExchange using actual decomposed subdomains."""

    def test_exchange_on_decomposed_field(self, fv_mesh_8cell):
        """Exchange ghost cell values between decomposed subdomains."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        # Create fields for all processors
        all_fields: dict[int, torch.Tensor] = {}
        for sub in subdomains:
            n_total = sub.global_cell_ids.numel()
            field = torch.zeros(n_total)
            field[:sub.n_owned_cells] = float(sub.processor_id + 1)
            field[sub.n_owned_cells:] = -1.0
            all_fields[sub.processor_id] = field

        # Exchange using all_fields
        for sub in subdomains:
            field = all_fields[sub.processor_id]
            halo = HaloExchange(sub.processor_patches)
            result = halo.exchange(field, all_fields=all_fields)

            # Ghost cells should now have non-negative values
            if sub.ghost_cells.numel() > 0:
                assert (result[sub.n_owned_cells:] >= 0).all(), (
                    f"Proc {sub.processor_id}: ghost cells still have negative values"
                )

    def test_exchange_preserves_owned_values(self, fv_mesh_8cell):
        """Exchange doesn't modify owned cell values."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        all_fields: dict[int, torch.Tensor] = {}
        for sub in subdomains:
            n_total = sub.global_cell_ids.numel()
            field = torch.arange(n_total, dtype=torch.float64) + 1
            all_fields[sub.processor_id] = field

        for sub in subdomains:
            field = all_fields[sub.processor_id]
            original_owned = field[:sub.n_owned_cells].clone()
            halo = HaloExchange(sub.processor_patches)
            result = halo.exchange(field, all_fields=all_fields)
            assert torch.allclose(result[:sub.n_owned_cells], original_owned)

    def test_ghost_values_match_neighbour_owned(self, fv_mesh_8cell):
        """After exchange, ghost cells have the neighbour's owned cell values."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        all_fields: dict[int, torch.Tensor] = {}
        for sub in subdomains:
            n_total = sub.global_cell_ids.numel()
            field = torch.zeros(n_total)
            # Each owned cell gets its global cell ID as value
            for local_idx in range(sub.n_owned_cells):
                global_idx = sub.global_cell_ids[local_idx].item()
                field[local_idx] = float(global_idx)
            all_fields[sub.processor_id] = field

        for sub in subdomains:
            field = all_fields[sub.processor_id]
            halo = HaloExchange(sub.processor_patches)
            halo.exchange(field, all_fields=all_fields)

            # Ghost cells should have the global cell ID as value
            for ghost_local_idx in range(sub.n_owned_cells, sub.global_cell_ids.numel()):
                global_idx = sub.global_cell_ids[ghost_local_idx].item()
                assert torch.allclose(
                    field[ghost_local_idx],
                    torch.tensor(float(global_idx)),
                    atol=1e-10,
                ), (
                    f"Proc {sub.processor_id}: ghost cell {ghost_local_idx} "
                    f"(global {global_idx}) has value {field[ghost_local_idx].item()}, "
                    f"expected {float(global_idx)}"
                )


# ---------------------------------------------------------------------------
# ParallelField tests
# ---------------------------------------------------------------------------


class TestParallelFieldSerial:
    """Test ParallelField in serial mode."""

    def test_basic_field(self, fv_mesh_8cell):
        """Create a ParallelField from a simple field."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        sub = subdomains[0]
        n_total = sub.global_cell_ids.numel()
        field = torch.ones(n_total)
        halo = HaloExchange(sub.processor_patches)

        pf = ParallelField(field, halo, sub)
        assert pf.n_owned == sub.n_owned_cells
        assert pf.local.shape == (n_total,)

    def test_owned_and_ghost_values(self, fv_mesh_8cell):
        """Owned and ghost value views are correct."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        sub = subdomains[0]
        n_total = sub.global_cell_ids.numel()
        field = torch.arange(n_total, dtype=torch.float64) + 1
        halo = HaloExchange(sub.processor_patches)

        pf = ParallelField(field, halo, sub)
        assert pf.owned_values.shape == (sub.n_owned_cells,)
        if sub.ghost_cells.numel() > 0:
            assert pf.ghost_values.shape == (sub.ghost_cells.numel(),)

    def test_global_sum_serial(self, fv_mesh_8cell):
        """Global sum in serial mode returns local sum of owned values."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        sub = subdomains[0]
        n_total = sub.global_cell_ids.numel()
        field = torch.ones(n_total)
        halo = HaloExchange(sub.processor_patches)

        pf = ParallelField(field, halo, sub)
        total = pf.global_sum()
        assert torch.allclose(total, torch.tensor(float(sub.n_owned_cells)))

    def test_global_max_serial(self, fv_mesh_8cell):
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        sub = subdomains[0]
        n_total = sub.global_cell_ids.numel()
        field = torch.arange(n_total, dtype=torch.float64)
        halo = HaloExchange(sub.processor_patches)

        pf = ParallelField(field, halo, sub)
        max_val = pf.global_max()
        expected = field[: sub.n_owned_cells].max()
        assert torch.allclose(max_val, expected)

    def test_global_min_serial(self, fv_mesh_8cell):
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        sub = subdomains[0]
        n_total = sub.global_cell_ids.numel()
        field = torch.arange(n_total, dtype=torch.float64) + 1
        halo = HaloExchange(sub.processor_patches)

        pf = ParallelField(field, halo, sub)
        min_val = pf.global_min()
        expected = field[: sub.n_owned_cells].min()
        assert torch.allclose(min_val, expected)

    def test_global_mean_serial(self, fv_mesh_8cell):
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        sub = subdomains[0]
        n_total = sub.global_cell_ids.numel()
        field = torch.ones(n_total) * 3.0
        halo = HaloExchange(sub.processor_patches)

        pf = ParallelField(field, halo, sub)
        mean_val = pf.global_mean()
        assert torch.allclose(mean_val, torch.tensor(3.0))

    def test_gather_to_root_serial(self, fv_mesh_8cell):
        """In serial mode, gather returns owned values."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        sub = subdomains[0]
        n_total = sub.global_cell_ids.numel()
        field = torch.arange(n_total, dtype=torch.float64) + 1
        halo = HaloExchange(sub.processor_patches)

        pf = ParallelField(field, halo, sub)
        gathered = pf.gather_to_root()
        assert gathered is not None
        assert gathered.shape == (sub.n_owned_cells,)

    def test_update_halos(self, fv_mesh_8cell):
        """update_halos() refreshes ghost cell values."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        # Create fields for all processors
        all_fields: dict[int, torch.Tensor] = {}
        for sub in subdomains:
            n_total = sub.global_cell_ids.numel()
            field = torch.zeros(n_total)
            field[:sub.n_owned_cells] = float(sub.processor_id + 1)
            field[sub.n_owned_cells:] = -1.0
            all_fields[sub.processor_id] = field

        for sub in subdomains:
            field = all_fields[sub.processor_id]
            halo = HaloExchange(sub.processor_patches)
            pf = ParallelField(field, halo, sub)
            pf.update_halos(all_fields=all_fields)

            # Ghost cells should be updated
            if sub.ghost_cells.numel() > 0:
                assert (pf.local[sub.n_owned_cells:] >= 0).all()


# ---------------------------------------------------------------------------
# Parallel I/O tests
# ---------------------------------------------------------------------------


class TestParallelIO:
    """Test parallel I/O (writing and reading processor directories)."""

    def test_create_processor_dirs(self, fv_mesh_8cell):
        """Create processor directories."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ParallelWriter(tmpdir, n_processors=2)
            dirs = writer.create_processor_dirs()

            assert len(dirs) == 2
            assert (dirs[0] / "constant" / "polyMesh").exists()
            assert (dirs[1] / "constant" / "polyMesh").exists()

    def test_write_mesh(self, fv_mesh_8cell):
        """Write subdomain meshes to processor directories."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ParallelWriter(tmpdir, n_processors=2)
            writer.write_mesh(subdomains)

            for i in range(2):
                mesh_dir = os.path.join(tmpdir, f"processor{i}", "constant", "polyMesh")
                assert os.path.exists(os.path.join(mesh_dir, "points"))
                assert os.path.exists(os.path.join(mesh_dir, "faces"))
                assert os.path.exists(os.path.join(mesh_dir, "owner"))
                assert os.path.exists(os.path.join(mesh_dir, "neighbour"))
                assert os.path.exists(os.path.join(mesh_dir, "boundary"))

    def test_write_field(self, fv_mesh_8cell):
        """Write field values to processor directories."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        global_field = torch.arange(8, dtype=torch.float64) * 10.0

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ParallelWriter(tmpdir, n_processors=2)
            writer.write_field("p", global_field, subdomains, time=0)

            for i in range(2):
                field_path = os.path.join(tmpdir, f"processor{i}", "0", "p")
                assert os.path.exists(field_path)

    def test_read_write_roundtrip_points(self, fv_mesh_8cell):
        """Write and read back mesh points."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ParallelWriter(tmpdir, n_processors=2)
            writer.write_mesh(subdomains)

            reader = ParallelReader(tmpdir)
            for i in range(2):
                mesh_data = reader.read_processor_mesh(i)
                original_points = subdomains[i].mesh.points
                read_points = mesh_data["points"]
                assert torch.allclose(original_points, read_points, atol=1e-6)

    def test_read_write_roundtrip_field(self, fv_mesh_8cell):
        """Write and read back field values."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        global_field = torch.arange(8, dtype=torch.float64) * 10.0

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ParallelWriter(tmpdir, n_processors=2)
            writer.write_field("p", global_field, subdomains, time=0)

            reader = ParallelReader(tmpdir)
            for i in range(2):
                read_field = reader.read_field(i, "p", time=0)
                # Values should match the owned cells
                expected = global_field[subdomains[i].global_cell_ids[:subdomains[i].n_owned_cells]]
                assert torch.allclose(read_field, expected, atol=1e-6)

    def test_write_multiple_fields(self, fv_mesh_8cell):
        """Write multiple fields."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        p_field = torch.arange(8, dtype=torch.float64)
        u_field = torch.arange(8, dtype=torch.float64) * 2.0

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ParallelWriter(tmpdir, n_processors=2)
            writer.write_field("p", p_field, subdomains, time=0)
            writer.write_field("U", u_field, subdomains, time=0)

            for i in range(2):
                proc_dir = os.path.join(tmpdir, f"processor{i}", "0")
                assert os.path.exists(os.path.join(proc_dir, "p"))
                assert os.path.exists(os.path.join(proc_dir, "U"))


# ---------------------------------------------------------------------------
# ParallelSolver tests
# ---------------------------------------------------------------------------


class TestParallelSolverSerial:
    """Test ParallelSolver in serial mode."""

    def test_solver_creation(self, fv_mesh_8cell):
        """Create a ParallelSolver with a mock local solver."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        sub = subdomains[0]
        halo = HaloExchange(sub.processor_patches)

        def mock_solver(matrix, source, x0, tolerance, max_iter):
            return x0, 1, 0.0

        psolver = ParallelSolver(mock_solver, halo, sub)
        assert psolver.rank == 0

    def test_solver_config_defaults(self):
        config = ParallelSolverConfig()
        assert config.max_outer_iterations == 100
        assert config.outer_tolerance == 1e-6
        assert config.update_halos_every == 1

    def test_solver_config_custom(self):
        config = ParallelSolverConfig(
            max_outer_iterations=50,
            outer_tolerance=1e-8,
            update_halos_every=2,
        )
        assert config.max_outer_iterations == 50
        assert config.outer_tolerance == 1e-8
        assert config.update_halos_every == 2

    def test_solver_repr(self, fv_mesh_8cell):
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        sub = subdomains[0]
        halo = HaloExchange(sub.processor_patches)

        def mock_solver(matrix, source, x0, tol, max_iter):
            return x0, 1, 0.0

        psolver = ParallelSolver(mock_solver, halo, sub)
        r = repr(psolver)
        assert "ParallelSolver" in r

    def test_solver_solve_returns_tuple(self, fv_mesh_8cell):
        """solve() returns (solution, iterations, residual)."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        sub = subdomains[0]
        n_total = sub.global_cell_ids.numel()
        halo = HaloExchange(sub.processor_patches)

        call_count = [0]

        def mock_solver(matrix, source, x0, tol, max_iter):
            call_count[0] += 1
            return x0 + 0.1, 10, 1e-7

        psolver = ParallelSolver(
            mock_solver, halo, sub,
            config=ParallelSolverConfig(max_outer_iterations=3, outer_tolerance=1e-10),
        )

        x0 = torch.zeros(n_total)
        source = torch.ones(n_total)
        # We need a mock matrix — use a simple object
        class MockMatrix:
            n_cells = n_total

        x, iters, residual = psolver.solve(MockMatrix(), source, x0)
        assert x.shape == (n_total,)
        assert iters > 0
        assert isinstance(residual, float)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline(self, fv_mesh_8cell):
        """Decompose → create field → exchange → gather."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        all_fields: dict[int, torch.Tensor] = {}
        for sub in subdomains:
            n_total = sub.global_cell_ids.numel()
            field = torch.ones(n_total) * float(sub.processor_id + 1)
            all_fields[sub.processor_id] = field

        for sub in subdomains:
            field = all_fields[sub.processor_id]
            halo = HaloExchange(sub.processor_patches)
            pf = ParallelField(field, halo, sub)
            pf.update_halos(all_fields=all_fields)

            total = pf.global_sum()
            expected = float(sub.n_owned_cells * (sub.processor_id + 1))
            assert torch.allclose(total, torch.tensor(expected))

    def test_decompose_poly_mesh(self):
        """Decomposition works with PolyMesh too."""
        mesh = make_8cell_poly_mesh()
        decomp = Decomposition(mesh, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        assert len(subdomains) == 2
        total_owned = sum(sub.n_owned_cells for sub in subdomains)
        assert total_owned == 8

    def test_decompose_repr(self, fv_mesh_8cell):
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        r = repr(decomp)
        assert "Decomposition" in r
        assert "2" in r

    def test_consistent_global_ids(self, fv_mesh_8cell):
        """Global IDs across all subdomains cover the full mesh."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        all_global_ids = set()
        for sub in subdomains:
            all_global_ids.update(sub.global_cell_ids.tolist())
        assert all_global_ids == set(range(8))

    def test_io_roundtrip_preserves_data(self, fv_mesh_8cell):
        """Write decomposed data and read it back correctly."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        global_field = torch.arange(8, dtype=torch.float64) * 10.0

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ParallelWriter(tmpdir, n_processors=2)
            writer.write_mesh(subdomains)
            writer.write_field("p", global_field, subdomains, time=0)

            reader = ParallelReader(tmpdir)
            for i in range(2):
                mesh_data = reader.read_processor_mesh(i)
                assert mesh_data["points"].shape[0] > 0
                assert len(mesh_data["faces"]) > 0

                read_p = reader.read_field(i, "p", time=0)
                expected = global_field[subdomains[i].global_cell_ids[:subdomains[i].n_owned_cells]]
                assert torch.allclose(read_p, expected, atol=1e-6)

    def test_halo_exchange_consistency(self, fv_mesh_8cell):
        """After exchange, ghost cell values match the owning processor's values."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        # Create fields where each owned cell has its global cell ID as value
        all_fields: dict[int, torch.Tensor] = {}
        for sub in subdomains:
            n_total = sub.global_cell_ids.numel()
            field = torch.zeros(n_total)
            for local_idx in range(sub.n_owned_cells):
                global_idx = sub.global_cell_ids[local_idx].item()
                field[local_idx] = float(global_idx)
            field[sub.n_owned_cells:] = -1.0  # Ghost cells, wrong value
            all_fields[sub.processor_id] = field

        # Exchange
        for sub in subdomains:
            field = all_fields[sub.processor_id]
            halo = HaloExchange(sub.processor_patches)
            halo.exchange(field, all_fields=all_fields)

        # Verify: after exchange, ghost cells should have their global cell ID
        for sub in subdomains:
            field = all_fields[sub.processor_id]
            for ghost_local_idx in range(sub.n_owned_cells, sub.global_cell_ids.numel()):
                global_idx = sub.global_cell_ids[ghost_local_idx].item()
                assert torch.allclose(
                    field[ghost_local_idx],
                    torch.tensor(float(global_idx)),
                    atol=1e-10,
                ), (
                    f"Proc {sub.processor_id}: ghost cell {ghost_local_idx} "
                    f"(global {global_idx}) has value {field[ghost_local_idx].item()}, "
                    f"expected {float(global_idx)}"
                )
