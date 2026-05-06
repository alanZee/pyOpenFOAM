"""
Linear solver scaling benchmarks.

Measures solve time and iteration count for PCG, PBiCGSTAB, and GAMG
across increasing mesh sizes (10³, 20³, 40³, 60³ cells).

Results are saved as CSV for post-processing.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import torch

from pyfoam.core.ldu_matrix import LduMatrix
from pyfoam.solvers import PCGSolver, PBiCGSTABSolver, GAMGSolver

from benchmarks.mesh_generation import (
    generate_diffusion_matrix,
    generate_asymmetric_matrix,
    generate_structured_hex_mesh,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MESH_SIZES = [10, 20, 40, 60]  # cells per dimension → 1K, 8K, 64K, 216K cells
WARMUP_RUNS = 1
BENCHMARK_RUNS = 3
TOLERANCE = 1e-6
MAX_ITER = 1000


def _sync_device(device: torch.device) -> None:
    """Synchronize device for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_solver(
    solver,
    matrix: LduMatrix,
    source: torch.Tensor,
    x0: torch.Tensor,
    n_runs: int,
    device: torch.device,
) -> tuple[float, int, float]:
    """Time multiple solver runs and return median time, iterations, residual.

    Parameters
    ----------
    solver : LinearSolverBase
        Configured solver instance.
    matrix : LduMatrix
        The system matrix.
    source : torch.Tensor
        Right-hand side vector.
    x0 : torch.Tensor
        Initial guess.
    n_runs : int
        Number of timing runs.
    device : torch.device
        Device for synchronization.

    Returns
    -------
    tuple[float, int, float]
        (median_time_seconds, iterations, final_residual)
    """
    times = []
    iterations = 0
    residual = 0.0

    for _ in range(n_runs):
        _sync_device(device)
        t0 = time.perf_counter()

        solution, iterations, residual = solver(matrix, source, x0.clone())

        _sync_device(device)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    # Return median time
    times.sort()
    median_time = times[len(times) // 2]

    return median_time, iterations, residual


# ---------------------------------------------------------------------------
# Single-solver benchmark
# ---------------------------------------------------------------------------


def benchmark_solver(
    solver_name: str,
    matrix_factory,
    mesh_sizes: list[int] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
    tolerance: float = TOLERANCE,
    max_iter: int = MAX_ITER,
    warmup_runs: int = WARMUP_RUNS,
    benchmark_runs: int = BENCHMARK_RUNS,
) -> list[dict[str, Any]]:
    """Benchmark a single solver across multiple mesh sizes.

    Parameters
    ----------
    solver_name : str
        One of ``"PCG"``, ``"PBiCGSTAB"``, ``"GAMG"``.
    matrix_factory : callable
        Function ``(n_cells_per_dim, device=, dtype=) -> LduMatrix``.
    mesh_sizes : list[int], optional
        Cells per dimension. Defaults to ``MESH_SIZES``.
    device : torch.device or str, optional
        Target device.
    dtype : torch.dtype
        Floating-point dtype.
    tolerance : float
        Solver convergence tolerance.
    max_iter : int
        Maximum solver iterations.
    warmup_runs : int
        Warmup iterations (not timed).
    benchmark_runs : int
        Timed iterations (median reported).

    Returns
    -------
    list[dict]
        One record per mesh size with keys:
        solver, n_cells_per_dim, n_cells, n_internal_faces,
        time_seconds, iterations, residual, converged
    """
    if mesh_sizes is None:
        mesh_sizes = MESH_SIZES

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Create solver
    solver_map = {
        "PCG": lambda: PCGSolver(tolerance=tolerance, max_iter=max_iter),
        "PBICGSTAB": lambda: PBiCGSTABSolver(tolerance=tolerance, max_iter=max_iter),
        "GAMG": lambda: GAMGSolver(tolerance=tolerance, max_iter=max_iter),
    }
    key = solver_name.upper().replace("-", "").replace("_", "")
    if key not in solver_map:
        raise ValueError(f"Unknown solver: {solver_name}")

    results = []

    for N in mesh_sizes:
        print(f"  [{solver_name}] N={N} ({N**3} cells)...", end="", flush=True)

        # Generate matrix
        matrix = matrix_factory(N, device=device, dtype=dtype)

        # Create RHS: b = A · ones (known solution = all ones)
        ones = torch.ones(matrix.n_cells, device=device, dtype=dtype)
        source = matrix.Ax(ones)
        x0 = torch.zeros(matrix.n_cells, device=device, dtype=dtype)

        solver = solver_map[key]()

        # Warmup
        for _ in range(warmup_runs):
            solver(matrix, source, x0.clone())

        # Benchmark
        median_time, iterations, residual = _time_solver(
            solver, matrix, source, x0, benchmark_runs, device
        )

        converged = residual < tolerance

        record = {
            "solver": solver_name,
            "n_cells_per_dim": N,
            "n_cells": matrix.n_cells,
            "n_internal_faces": matrix.n_internal_faces,
            "time_seconds": median_time,
            "iterations": iterations,
            "residual": residual,
            "converged": converged,
            "device": str(device),
            "dtype": str(dtype),
        }
        results.append(record)

        status = "[OK]" if converged else "[FAIL]"
        print(f" {median_time:.4f}s, {iterations} iter, res={residual:.2e} {status}")

    return results


# ---------------------------------------------------------------------------
# Full benchmark
# ---------------------------------------------------------------------------


def run_linear_solve_benchmarks(
    mesh_sizes: list[int] | None = None,
    device: torch.device | str | None = None,
    output_dir: str | Path = "benchmarks/results",
) -> list[dict[str, Any]]:
    """Run the complete linear solver scaling benchmark.

    Benchmarks PCG (symmetric), PBiCGSTAB (asymmetric), and GAMG
    across the specified mesh sizes.

    Parameters
    ----------
    mesh_sizes : list[int], optional
        Cells per dimension. Defaults to ``MESH_SIZES``.
    device : torch.device or str, optional
        Target device.
    output_dir : str or Path
        Directory for CSV output.

    Returns
    -------
    list[dict]
        All benchmark results.
    """
    if mesh_sizes is None:
        mesh_sizes = MESH_SIZES

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"\n{'='*60}")
    print(f"Linear Solver Scaling Benchmark")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Mesh sizes: {mesh_sizes}")
    print(f"{'='*60}\n")

    all_results: list[dict[str, Any]] = []

    # PCG — symmetric diffusion matrix
    print("[PCG] Symmetric diffusion (Laplacian)")
    pcg_results = benchmark_solver(
        "PCG",
        generate_diffusion_matrix,
        mesh_sizes=mesh_sizes,
        device=device,
    )
    all_results.extend(pcg_results)

    # PBiCGSTAB — asymmetric convection-diffusion
    print("\n[PBiCGSTAB] Asymmetric convection-diffusion (Pe=10)")
    pbicg_results = benchmark_solver(
        "PBiCGSTAB",
        lambda N, device, dtype: generate_asymmetric_matrix(
            N, device=device, dtype=dtype, peclet=10.0
        ),
        mesh_sizes=mesh_sizes,
        device=device,
    )
    all_results.extend(pbicg_results)

    # GAMG — symmetric diffusion matrix
    print("\n[GAMG] Symmetric diffusion (Laplacian)")
    gamg_results = benchmark_solver(
        "GAMG",
        generate_diffusion_matrix,
        mesh_sizes=mesh_sizes,
        device=device,
    )
    all_results.extend(gamg_results)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_file = output_path / "linear_solve_benchmark.csv"

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to {csv_file}")

    return all_results


if __name__ == "__main__":
    run_linear_solve_benchmarks()
