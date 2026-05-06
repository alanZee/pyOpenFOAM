"""
GPU vs CPU speedup benchmarks.

Compares solver performance on CPU and GPU across mesh sizes.
Measures speedup ratio and reports which operations benefit most
from GPU acceleration.

Gracefully skips GPU benchmarks if CUDA is not available.
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
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MESH_SIZES = [10, 20, 40, 60]
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
    """Time multiple solver runs and return median time."""
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

    times.sort()
    median_time = times[len(times) // 2]
    return median_time, iterations, residual


def _benchmark_on_device(
    solver_name: str,
    matrix_factory,
    device: torch.device,
    mesh_sizes: list[int],
    dtype: torch.dtype = torch.float64,
) -> list[dict[str, Any]]:
    """Run solver benchmark on a specific device.

    Parameters
    ----------
    solver_name : str
        Solver name.
    matrix_factory : callable
        Matrix generation function.
    device : torch.device
        Target device.
    mesh_sizes : list[int]
        Cells per dimension.
    dtype : torch.dtype
        Float dtype.

    Returns
    -------
    list[dict]
        Results for each mesh size.
    """
    solver_map = {
        "PCG": lambda: PCGSolver(tolerance=TOLERANCE, max_iter=MAX_ITER),
        "PBICGSTAB": lambda: PBiCGSTABSolver(tolerance=TOLERANCE, max_iter=MAX_ITER),
        "GAMG": lambda: GAMGSolver(tolerance=TOLERANCE, max_iter=MAX_ITER),
    }
    key = solver_name.upper().replace("-", "").replace("_", "")
    results = []

    for N in mesh_sizes:
        print(f"    N={N} ({N**3} cells)...", end="", flush=True)

        matrix = matrix_factory(N, device=device, dtype=dtype)
        ones = torch.ones(matrix.n_cells, device=device, dtype=dtype)
        source = matrix.Ax(ones)
        x0 = torch.zeros(matrix.n_cells, device=device, dtype=dtype)

        solver = solver_map[key]()

        # Warmup
        for _ in range(WARMUP_RUNS):
            solver(matrix, source, x0.clone())

        # Benchmark
        median_time, iterations, residual = _time_solver(
            solver, matrix, source, x0, BENCHMARK_RUNS, device
        )

        results.append({
            "solver": solver_name,
            "n_cells_per_dim": N,
            "n_cells": matrix.n_cells,
            "device": str(device),
            "time_seconds": median_time,
            "iterations": iterations,
            "residual": residual,
        })

        print(f" {median_time:.4f}s ({iterations} iter)")
        del matrix, source, x0
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# GPU vs CPU comparison
# ---------------------------------------------------------------------------


def run_gpu_cpu_comparison(
    mesh_sizes: list[int] | None = None,
    output_dir: str | Path = "benchmarks/results",
) -> list[dict[str, Any]]:
    """Run GPU vs CPU speedup comparison.

    For each solver and mesh size, runs on both CPU and GPU (if available)
    and computes the speedup ratio.

    Parameters
    ----------
    mesh_sizes : list[int], optional
        Cells per dimension. Defaults to ``MESH_SIZES``.
    output_dir : str or Path
        Directory for CSV output.

    Returns
    -------
    list[dict]
        Results with speedup ratios.
    """
    if mesh_sizes is None:
        mesh_sizes = MESH_SIZES

    has_gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_gpu else "N/A"

    print(f"\n{'='*60}")
    print(f"GPU vs CPU Speedup Benchmark")
    print(f"{'='*60}")
    print(f"GPU Available: {has_gpu}")
    if has_gpu:
        print(f"GPU Device: {gpu_name}")
    print(f"Mesh sizes: {mesh_sizes}")
    print(f"{'='*60}\n")

    cpu_device = torch.device("cpu")
    gpu_device = torch.device("cuda") if has_gpu else None

    all_results: list[dict[str, Any]] = []

    benchmarks = [
        ("PCG", generate_diffusion_matrix, "PCG (symmetric diffusion)"),
        ("PBiCGSTAB", lambda N, device, dtype: generate_asymmetric_matrix(
            N, device=device, dtype=dtype, peclet=10.0
        ), "PBiCGSTAB (asymmetric convection-diffusion)"),
        ("GAMG", generate_diffusion_matrix, "GAMG (multigrid)"),
    ]

    for solver_name, factory, description in benchmarks:
        print(f"\n[{solver_name}] {description}")

        # CPU benchmark
        print(f"  CPU:")
        cpu_results = _benchmark_on_device(
            solver_name, factory, cpu_device, mesh_sizes
        )

        if has_gpu and gpu_device is not None:
            # GPU benchmark
            print(f"  GPU ({gpu_name}):")
            gpu_results = _benchmark_on_device(
                solver_name, factory, gpu_device, mesh_sizes
            )

            # Compute speedup
            for cpu_r, gpu_r in zip(cpu_results, gpu_results):
                speedup = cpu_r["time_seconds"] / max(gpu_r["time_seconds"], 1e-10)
                combined = {
                    **cpu_r,
                    "cpu_time": cpu_r["time_seconds"],
                    "gpu_time": gpu_r["time_seconds"],
                    "speedup": speedup,
                    "gpu_name": gpu_name,
                }
                # Remove the generic time_seconds key to avoid confusion
                del combined["time_seconds"]
                all_results.append(combined)

                print(
                    f"    N={cpu_r['n_cells_per_dim']}: "
                    f"CPU={cpu_r['time_seconds']:.4f}s, "
                    f"GPU={gpu_r['time_seconds']:.4f}s, "
                    f"speedup={speedup:.2f}x"
                )
        else:
            # No GPU — record CPU-only results
            for cpu_r in cpu_results:
                combined = {
                    **cpu_r,
                    "cpu_time": cpu_r["time_seconds"],
                    "gpu_time": None,
                    "speedup": None,
                    "gpu_name": "N/A",
                }
                del combined["time_seconds"]
                all_results.append(combined)

    # Save results
    if all_results:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        csv_file = output_path / "gpu_cpu_comparison.csv"

        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

        print(f"\nResults saved to {csv_file}")

    return all_results


if __name__ == "__main__":
    run_gpu_cpu_comparison()
