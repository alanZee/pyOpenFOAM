"""
Memory usage scaling benchmarks.

Measures memory consumption for matrix storage and solver execution
across increasing mesh sizes.

For GPU: tracks CUDA memory via ``torch.cuda.max_memory_allocated()``.
For CPU: tracks Python process memory via ``tracemalloc``.
"""

from __future__ import annotations

import csv
import gc
import tracemalloc
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

MESH_SIZES = [10, 20, 40, 60]
TOLERANCE = 1e-6
MAX_ITER = 100


def _get_gpu_memory_mb(device: torch.device) -> float:
    """Get current GPU memory usage in MB."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return 0.0


def _reset_gpu_memory(device: torch.device) -> None:
    """Reset GPU memory tracking."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------


def measure_matrix_memory(
    n_cells_per_dim: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    """Measure memory for matrix storage only (no solver).

    Parameters
    ----------
    n_cells_per_dim : int
        Cells per dimension (N).
    device : torch.device or str, optional
        Target device.
    dtype : torch.dtype
        Float dtype.

    Returns
    -------
    dict
        Memory breakdown in MB.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    gc.collect()
    if device.type == "cuda":
        _reset_gpu_memory(device)

    is_gpu = device.type == "cuda"

    if is_gpu:
        before = _get_gpu_memory_mb(device)
        matrix = generate_diffusion_matrix(n_cells_per_dim, device=device, dtype=dtype)
        after = _get_gpu_memory_mb(device)
        matrix_memory = after - before
    else:
        tracemalloc.start()
        matrix = generate_diffusion_matrix(n_cells_per_dim, device=device, dtype=dtype)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        matrix_memory = peak / (1024 * 1024)

    # Compute theoretical memory
    n_cells = matrix.n_cells
    n_faces = matrix.n_internal_faces
    bytes_per_elem = 4 if dtype == torch.float32 else 8

    # LDU storage: diag (n_cells) + lower (n_faces) + upper (n_faces)
    # + owner (n_faces) + neighbour (n_faces)
    theoretical_bytes = (
        n_cells * bytes_per_elem  # diag
        + 2 * n_faces * bytes_per_elem  # lower + upper
        + 2 * n_faces * 8  # owner + neighbour (int64)
    )
    theoretical_mb = theoretical_bytes / (1024 * 1024)

    result = {
        "n_cells_per_dim": n_cells_per_dim,
        "n_cells": n_cells,
        "n_internal_faces": n_faces,
        "device": str(device),
        "matrix_memory_mb": matrix_memory,
        "theoretical_mb": theoretical_mb,
        "bytes_per_element": bytes_per_elem,
    }

    del matrix
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def measure_solver_memory(
    solver_name: str,
    n_cells_per_dim: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    """Measure peak memory during solver execution.

    Parameters
    ----------
    solver_name : str
        One of ``"PCG"``, ``"PBiCGSTAB"``, ``"GAMG"``.
    n_cells_per_dim : int
        Cells per dimension (N).
    device : torch.device or str, optional
        Target device.
    dtype : torch.dtype
        Float dtype.

    Returns
    -------
    dict
        Memory usage in MB.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    gc.collect()
    if device.type == "cuda":
        _reset_gpu_memory(device)

    # Generate matrix
    if solver_name.upper() == "PBICGSTAB":
        matrix = generate_asymmetric_matrix(
            n_cells_per_dim, device=device, dtype=dtype, peclet=10.0
        )
    else:
        matrix = generate_diffusion_matrix(n_cells_per_dim, device=device, dtype=dtype)

    n_cells = matrix.n_cells

    # Create solver
    solver_map = {
        "PCG": lambda: PCGSolver(tolerance=TOLERANCE, max_iter=MAX_ITER),
        "PBICGSTAB": lambda: PBiCGSTABSolver(tolerance=TOLERANCE, max_iter=MAX_ITER),
        "GAMG": lambda: GAMGSolver(tolerance=TOLERANCE, max_iter=MAX_ITER),
    }
    key = solver_name.upper().replace("-", "").replace("_", "")
    solver = solver_map[key]()

    ones = torch.ones(n_cells, device=device, dtype=dtype)
    source = matrix.Ax(ones)
    x0 = torch.zeros(n_cells, device=device, dtype=dtype)

    is_gpu = device.type == "cuda"

    if is_gpu:
        before = _get_gpu_memory_mb(device)
        solution, iterations, residual = solver(matrix, source, x0)
        peak = _get_gpu_memory_mb(device)
        solver_memory = peak - before
    else:
        tracemalloc.start()
        solution, iterations, residual = solver(matrix, source, x0)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        solver_memory = peak / (1024 * 1024)

    result = {
        "solver": solver_name,
        "n_cells_per_dim": n_cells_per_dim,
        "n_cells": n_cells,
        "n_internal_faces": matrix.n_internal_faces,
        "device": str(device),
        "peak_memory_mb": solver_memory,
        "iterations": iterations,
        "converged": residual < TOLERANCE,
    }

    del matrix, source, x0, solution
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Full memory benchmark
# ---------------------------------------------------------------------------


def run_memory_scaling_benchmarks(
    mesh_sizes: list[int] | None = None,
    device: torch.device | str | None = None,
    output_dir: str | Path = "benchmarks/results",
) -> dict[str, list[dict[str, Any]]]:
    """Run memory scaling benchmarks.

    Measures:
    1. Matrix storage memory vs mesh size
    2. Peak solver memory vs mesh size (PCG, PBiCGSTAB, GAMG)

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
    dict
        Keys: ``"matrix"``, ``"PCG"``, ``"PBiCGSTAB"``, ``"GAMG"``,
        each mapping to a list of result dicts.
    """
    if mesh_sizes is None:
        mesh_sizes = MESH_SIZES

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"\n{'='*60}")
    print(f"Memory Scaling Benchmark")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Mesh sizes: {mesh_sizes}")
    print(f"{'='*60}\n")

    results: dict[str, list[dict[str, Any]]] = {}

    # 1. Matrix storage memory
    print("[Matrix Storage]")
    matrix_results = []
    for N in mesh_sizes:
        print(f"  N={N} ({N**3} cells)...", end="", flush=True)
        r = measure_matrix_memory(N, device=device)
        matrix_results.append(r)
        print(f" {r['matrix_memory_mb']:.2f} MB (theoretical: {r['theoretical_mb']:.2f} MB)")
    results["matrix"] = matrix_results

    # 2. Solver memory
    for solver_name in ["PCG", "PBiCGSTAB", "GAMG"]:
        print(f"\n[{solver_name}] Peak Solver Memory")
        solver_results = []
        for N in mesh_sizes:
            print(f"  N={N} ({N**3} cells)...", end="", flush=True)
            r = measure_solver_memory(solver_name, N, device=device)
            solver_results.append(r)
            status = "[OK]" if r["converged"] else "[FAIL]"
            print(f" {r['peak_memory_mb']:.2f} MB ({r['iterations']} iter) {status}")
        results[solver_name] = solver_results

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Matrix storage CSV
    if results["matrix"]:
        csv_file = output_path / "memory_matrix.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results["matrix"][0].keys())
            writer.writeheader()
            writer.writerows(results["matrix"])
        print(f"\nMatrix memory results saved to {csv_file}")

    # Solver memory CSV
    all_solver_results = []
    for solver_name in ["PCG", "PBiCGSTAB", "GAMG"]:
        all_solver_results.extend(results[solver_name])

    if all_solver_results:
        csv_file = output_path / "memory_solver.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_solver_results[0].keys())
            writer.writeheader()
            writer.writerows(all_solver_results)
        print(f"Solver memory results saved to {csv_file}")

    return results


if __name__ == "__main__":
    run_memory_scaling_benchmarks()
