"""
Sparse operations benchmark for pyOpenFOAM.

Compares the performance of different sparse matrix representations
and operations used in the FVM solver pipeline:

- LDU format Ax (scatter_add) vs sparse CSR Ax (torch.sparse.mm)
- COO vs CSR conversion times
- Batched vs sequential matrix-vector products

Results are saved as CSV for post-processing.

Usage::

    python -m benchmarks.sparse_ops_benchmark
    python -m benchmarks.sparse_ops_benchmark --mesh-sizes 10 20 40
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import torch

from pyfoam.core.ldu_matrix import LduMatrix
from pyfoam.core.sparse_ops import ldu_to_coo_indices

from benchmarks.mesh_generation import generate_structured_hex_mesh


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MESH_SIZES = [10, 20, 40, 60]
WARMUP_RUNS = 2
BENCHMARK_RUNS = 5


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _median(times: list[float]) -> float:
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def benchmark_coo_conversion(
    mesh_sizes: list[int],
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> list[dict[str, Any]]:
    """Benchmark LDU → COO conversion time."""
    results = []
    for N in mesh_sizes:
        print(f"  [COO conversion] N={N} ({N**3} cells)...", end="", flush=True)

        mesh = generate_structured_hex_mesh(N, device=device)
        mat = LduMatrix(
            mesh["n_cells"], mesh["owner"], mesh["neighbour"],
            device=device, dtype=dtype,
        )
        mat.diag = torch.ones(mesh["n_cells"], device=device, dtype=dtype)
        mat.lower = -torch.ones(mesh["n_internal_faces"], device=device, dtype=dtype)
        mat.upper = -torch.ones(mesh["n_internal_faces"], device=device, dtype=dtype)

        # Warmup
        for _ in range(WARMUP_RUNS):
            mat.to_sparse_coo()

        # Benchmark
        times = []
        for _ in range(BENCHMARK_RUNS):
            _sync(device)
            t0 = time.perf_counter()
            mat.to_sparse_coo()
            _sync(device)
            times.append(time.perf_counter() - t0)

        median = _median(times)
        results.append({
            "operation": "ldu_to_coo",
            "n_cells_per_dim": N,
            "n_cells": mesh["n_cells"],
            "n_internal_faces": mesh["n_internal_faces"],
            "device": str(device),
            "time_seconds": median,
        })
        print(f" {median:.6f}s")

    return results


def benchmark_csr_conversion(
    mesh_sizes: list[int],
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> list[dict[str, Any]]:
    """Benchmark LDU → CSR (via COO) conversion time."""
    results = []
    for N in mesh_sizes:
        print(f"  [CSR conversion] N={N} ({N**3} cells)...", end="", flush=True)

        mesh = generate_structured_hex_mesh(N, device=device)
        mat = LduMatrix(
            mesh["n_cells"], mesh["owner"], mesh["neighbour"],
            device=device, dtype=dtype,
        )
        mat.diag = torch.ones(mesh["n_cells"], device=device, dtype=dtype)
        mat.lower = -torch.ones(mesh["n_internal_faces"], device=device, dtype=dtype)
        mat.upper = -torch.ones(mesh["n_internal_faces"], device=device, dtype=dtype)

        # Warmup
        for _ in range(WARMUP_RUNS):
            mat.to_sparse_csr()

        # Benchmark
        times = []
        for _ in range(BENCHMARK_RUNS):
            _sync(device)
            t0 = time.perf_counter()
            mat.to_sparse_csr()
            _sync(device)
            times.append(time.perf_counter() - t0)

        median = _median(times)
        results.append({
            "operation": "ldu_to_csr",
            "n_cells_per_dim": N,
            "n_cells": mesh["n_cells"],
            "n_internal_faces": mesh["n_internal_faces"],
            "device": str(device),
            "time_seconds": median,
        })
        print(f" {median:.6f}s")

    return results


def benchmark_matvec(
    mesh_sizes: list[int],
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> list[dict[str, Any]]:
    """Benchmark Ax: LDU scatter_add vs sparse CSR mm."""
    results = []
    for N in mesh_sizes:
        print(f"  [Matvec] N={N} ({N**3} cells)...", end="", flush=True)

        mesh = generate_structured_hex_mesh(N, device=device)
        mat = LduMatrix(
            mesh["n_cells"], mesh["owner"], mesh["neighbour"],
            device=device, dtype=dtype,
        )
        mat.diag = torch.ones(mesh["n_cells"], device=device, dtype=dtype)
        mat.lower = -torch.ones(mesh["n_internal_faces"], device=device, dtype=dtype)
        mat.upper = -torch.ones(mesh["n_internal_faces"], device=device, dtype=dtype)

        x = torch.ones(mesh["n_cells"], device=device, dtype=dtype)

        # --- LDU Ax (scatter_add) ---
        for _ in range(WARMUP_RUNS):
            mat.Ax(x)
        ldu_times = []
        for _ in range(BENCHMARK_RUNS):
            _sync(device)
            t0 = time.perf_counter()
            mat.Ax(x)
            _sync(device)
            ldu_times.append(time.perf_counter() - t0)

        # --- Sparse Ax (CSR mm) ---
        for _ in range(WARMUP_RUNS):
            mat.Ax_sparse(x)
        sparse_times = []
        for _ in range(BENCHMARK_RUNS):
            _sync(device)
            t0 = time.perf_counter()
            mat.Ax_sparse(x)
            _sync(device)
            sparse_times.append(time.perf_counter() - t0)

        ldu_median = _median(ldu_times)
        sparse_median = _median(sparse_times)

        results.append({
            "operation": "ax_ldu",
            "n_cells_per_dim": N,
            "n_cells": mesh["n_cells"],
            "n_internal_faces": mesh["n_internal_faces"],
            "device": str(device),
            "time_seconds": ldu_median,
        })
        results.append({
            "operation": "ax_sparse_csr",
            "n_cells_per_dim": N,
            "n_cells": mesh["n_cells"],
            "n_internal_faces": mesh["n_internal_faces"],
            "device": str(device),
            "time_seconds": sparse_median,
        })

        ratio = ldu_median / max(sparse_median, 1e-10)
        print(f" LDU={ldu_median:.6f}s, CSR={sparse_median:.6f}s, ratio={ratio:.2f}x")

    return results


def benchmark_batched_matvec(
    mesh_sizes: list[int],
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> list[dict[str, Any]]:
    """Benchmark batched Ax (multiple RHS)."""
    results = []
    n_rhs = 4

    for N in mesh_sizes:
        print(f"  [Batched matvec] N={N} ({N**3} cells, {n_rhs} RHS)...", end="", flush=True)

        mesh = generate_structured_hex_mesh(N, device=device)
        mat = LduMatrix(
            mesh["n_cells"], mesh["owner"], mesh["neighbour"],
            device=device, dtype=dtype,
        )
        mat.diag = torch.ones(mesh["n_cells"], device=device, dtype=dtype)
        mat.lower = -torch.ones(mesh["n_internal_faces"], device=device, dtype=dtype)
        mat.upper = -torch.ones(mesh["n_internal_faces"], device=device, dtype=dtype)

        x_multi = torch.ones(mesh["n_cells"], n_rhs, device=device, dtype=dtype)
        x_single = torch.ones(mesh["n_cells"], device=device, dtype=dtype)

        # --- Sequential ---
        for _ in range(WARMUP_RUNS):
            for k in range(n_rhs):
                mat.Ax(x_single)
        seq_times = []
        for _ in range(BENCHMARK_RUNS):
            _sync(device)
            t0 = time.perf_counter()
            for k in range(n_rhs):
                mat.Ax(x_single)
            _sync(device)
            seq_times.append(time.perf_counter() - t0)

        # --- Batched ---
        for _ in range(WARMUP_RUNS):
            mat.Ax_batched(x_multi)
        batch_times = []
        for _ in range(BENCHMARK_RUNS):
            _sync(device)
            t0 = time.perf_counter()
            mat.Ax_batched(x_multi)
            _sync(device)
            batch_times.append(time.perf_counter() - t0)

        seq_median = _median(seq_times)
        batch_median = _median(batch_times)

        results.append({
            "operation": f"ax_sequential_{n_rhs}rhs",
            "n_cells_per_dim": N,
            "n_cells": mesh["n_cells"],
            "n_internal_faces": mesh["n_internal_faces"],
            "device": str(device),
            "time_seconds": seq_median,
        })
        results.append({
            "operation": f"ax_batched_{n_rhs}rhs",
            "n_cells_per_dim": N,
            "n_cells": mesh["n_cells"],
            "n_internal_faces": mesh["n_internal_faces"],
            "device": str(device),
            "time_seconds": batch_median,
        })

        ratio = seq_median / max(batch_median, 1e-10)
        print(f" Seq={seq_median:.6f}s, Batch={batch_median:.6f}s, speedup={ratio:.2f}x")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_sparse_ops_benchmarks(
    mesh_sizes: list[int] | None = None,
    device: torch.device | str | None = None,
    output_dir: str | Path = "benchmarks/results",
) -> list[dict[str, Any]]:
    """Run the complete sparse operations benchmark suite.

    Parameters
    ----------
    mesh_sizes : list[int], optional
        Cells per dimension. Defaults to ``MESH_SIZES``.
    device : torch.device or str, optional
        Target device. Defaults to auto-detect.
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
    print(f"Sparse Operations Benchmark")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Mesh sizes: {mesh_sizes}")
    print(f"{'='*60}\n")

    all_results: list[dict[str, Any]] = []

    print("[1/4] COO Conversion")
    all_results.extend(benchmark_coo_conversion(mesh_sizes, device))

    print("\n[2/4] CSR Conversion")
    all_results.extend(benchmark_csr_conversion(mesh_sizes, device))

    print("\n[3/4] Matrix-Vector Product (LDU vs CSR)")
    all_results.extend(benchmark_matvec(mesh_sizes, device))

    print("\n[4/4] Batched Matrix-Vector Product")
    all_results.extend(benchmark_batched_matvec(mesh_sizes, device))

    # Save
    if all_results:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        csv_file = output_path / "sparse_ops_benchmark.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {csv_file}")

    return all_results


if __name__ == "__main__":
    run_sparse_ops_benchmarks()
