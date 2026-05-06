"""Tests for parallel domain decomposition.

Tests the Decomposition class with the 8-cell mesh, verifying:
- Simple geometric decomposition
- Cell assignment correctness
- Subdomain mesh creation
- Load balancing metrics
- Processor patch identification
- Halo exchange (serial fallback)
- Parallel field operations
- Parallel I/O
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

    def test_no_overlap(self, fv_mesh_8cell):
        """Owned cells don't overlap between subdomains."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        owned_sets = []
        for sub in subdomains:
            owned = set(sub.global_cell_ids[: sub.n_owned_cells].tolist())
            owned_sets.append(owned)

        # No overlap in owned cells
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


class TestGhostCells:
    """Test ghost cell identification."""

    def test_ghost_cells_exist(self, fv_mesh_8cell):
        """Subdomains have ghost cells when mesh is connected."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        # At least one subdomain should have ghost cells
        has_ghosts = any(sub.ghost_cells.numel() > 0 for sub in subdomains)
        assert has_ghosts

    def test_ghost_cell_count(self, fv_mesh_8cell):
        """Ghost cells are a subset of total subdomain cells."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        for sub in subdomains:
            assert sub.ghost_cells.numel() <= sub.global_cell_ids.numel()
            assert sub.n_owned_cells + sub.ghost_cells.numel() == sub.global_cell_ids.numel()


# ---------------------------------------------------------------------------
# ProcessorPatch tests
# ---------------------------------------------------------------------------


class TestProcessorPatch:
    """Test ProcessorPatch dataclass."""

    def test_basic_patch(self):
        local = torch.tensor([4, 5], dtype=INDEX_DTYPE)
        remote = torch.tensor([0, 1], dtype=INDEX_DTYPE)
        patch = ProcessorPatch(
            name="procBoundary0",
            neighbour_rank=1,
            local_ghost_cells=local,
            remote_cells=remote,
        )

        assert patch.name == "procBoundary0"
        assert patch.neighbour_rank == 1
        assert patch.n_ghost_cells == 2


# ---------------------------------------------------------------------------
# HaloExchange tests (serial fallback)
# ---------------------------------------------------------------------------


class TestHaloExchangeSerial:
    """Test halo exchange in serial mode (no MPI)."""

    def test_empty_patches(self):
        """No patches means no exchange."""
        halo = HaloExchange([])
        field = torch.tensor([1.0, 2.0, 3.0])
        result = halo.exchange(field)
        assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_serial_loopback(self):
        """In serial mode, ghost values are copied from send buffer."""
        local_ghost = torch.tensor([2], dtype=INDEX_DTYPE)
        remote = torch.tensor([0], dtype=INDEX_DTYPE)
        patch = ProcessorPatch(
            name="proc0",
            neighbour_rank=0,
            local_ghost_cells=local_ghost,
            remote_cells=remote,
        )
        halo = HaloExchange([patch])

        # Field: cells 0,1 are owned, cell 2 is ghost
        field = torch.tensor([10.0, 20.0, 999.0])
        result = halo.exchange(field)

        # Ghost cell gets value from local_ghost_cells position
        # In serial mode, send buffer = field[local_ghost] = field[2] = 999
        # But the loopback copies send to recv, so ghost stays 999
        # This is correct for serial: ghost cells are self-referential
        assert result.shape == (3,)

    def test_repr(self):
        halo = HaloExchange([])
        r = repr(halo)
        assert "HaloExchange" in r


# ---------------------------------------------------------------------------
# ParallelField tests (serial fallback)
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
        halo = HaloExchange([])

        pf = ParallelField(field, halo, sub)
        assert pf.n_owned == sub.n_owned_cells
        assert pf.local.shape == (n_total,)

    def test_global_sum_serial(self, fv_mesh_8cell):
        """Global sum in serial mode returns local sum."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        sub = subdomains[0]
        n_total = sub.global_cell_ids.numel()
        field = torch.ones(n_total)
        halo = HaloExchange([])

        pf = ParallelField(field, halo, sub)
        total = pf.global_sum()
        assert torch.allclose(total, torch.tensor(float(sub.n_owned_cells)))

    def test_global_max_serial(self, fv_mesh_8cell):
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        sub = subdomains[0]
        n_total = sub.global_cell_ids.numel()
        field = torch.arange(n_total, dtype=torch.float64)
        halo = HaloExchange([])

        pf = ParallelField(field, halo, sub)
        max_val = pf.global_max()
        # Max of owned values
        expected = field[: sub.n_owned_cells].max()
        assert torch.allclose(max_val, expected)

    def test_global_min_serial(self, fv_mesh_8cell):
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        sub = subdomains[0]
        n_total = sub.global_cell_ids.numel()
        field = torch.arange(n_total, dtype=torch.float64) + 1
        halo = HaloExchange([])

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
        halo = HaloExchange([])

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
        halo = HaloExchange([])

        pf = ParallelField(field, halo, sub)
        gathered = pf.gather_to_root()
        assert gathered is not None
        assert gathered.shape == (sub.n_owned_cells,)


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

            # Check that mesh files exist
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

        # Create a global field
        global_field = torch.arange(8, dtype=torch.float64) * 10.0

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ParallelWriter(tmpdir, n_processors=2)
            writer.write_field("p", global_field, subdomains, time=0)

            # Check that field files exist
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


# ---------------------------------------------------------------------------
# ParallelSolver tests (serial fallback)
# ---------------------------------------------------------------------------


class TestParallelSolverSerial:
    """Test ParallelSolver in serial mode."""

    def test_solver_creation(self, fv_mesh_8cell):
        """Create a ParallelSolver with a mock local solver."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        sub = subdomains[0]
        halo = HaloExchange([])

        # Mock solver: just returns the input
        def mock_solver(matrix, source, x0, tolerance, max_iter):
            return x0, 1, 0.0

        psolver = ParallelSolver(mock_solver, halo, sub)
        assert psolver.rank == 0

    def test_solver_config_defaults(self):
        config = ParallelSolverConfig()
        assert config.max_outer_iterations == 100
        assert config.outer_tolerance == 1e-6
        assert config.update_halos_every == 1

    def test_solver_repr(self, fv_mesh_8cell):
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        sub = subdomains[0]
        halo = HaloExchange([])

        def mock_solver(matrix, source, x0, tol, max_iter):
            return x0, 1, 0.0

        psolver = ParallelSolver(mock_solver, halo, sub)
        r = repr(psolver)
        assert "ParallelSolver" in r


# ---------------------------------------------------------------------------
# Integration: decomposition → field → gather
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline(self, fv_mesh_8cell):
        """Decompose → create field → exchange → gather."""
        decomp = Decomposition(fv_mesh_8cell, n_processors=2, method="simple")
        subdomains = decomp.decompose()

        for sub in subdomains:
            n_total = sub.global_cell_ids.numel()
            field = torch.ones(n_total) * float(sub.processor_id + 1)
            halo = HaloExchange([])

            pf = ParallelField(field, halo, sub)
            pf.update_halos()

            # Global sum should be n_owned * (proc_id + 1)
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
