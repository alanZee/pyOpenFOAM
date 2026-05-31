"""
大规模网格基准测试。

测试 100K+ 单元网格的性能与内存行为：
1. 内存使用量（CPU: tracemalloc, GPU: CUDA 内存统计）
2. 迭代耗时
3. 扩展行为（时间/单元 随网格尺寸的变化）

结果保存至 benchmarks/results/large_mesh.json。

用法::

    python -m benchmarks.large_mesh
    CUDA_VISIBLE_DEVICES='' python -m benchmarks.large_mesh
"""

from __future__ import annotations

import gc
import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

import torch

from pyfoam.core.ldu_matrix import LduMatrix
from pyfoam.core.fv_matrix import FvMatrix
from pyfoam.core.device import device_context
from pyfoam.solvers import PCGSolver

from benchmarks.mesh_generation import generate_structured_hex_mesh


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

MESH_SIZES = [40, 50, 64, 80]  # 64K ~ 512K cells
WARMUP_RUNS = 1
BENCHMARK_RUNS = 3
TOLERANCE = 1e-6
MAX_ITER = 500


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _median(times: list[float]) -> float:
    s = sorted(times)
    return s[len(s) // 2]


# ---------------------------------------------------------------------------
# 内存测量
# ---------------------------------------------------------------------------


def _measure_memory_cpu(func, *args) -> tuple[Any, float]:
    """在 CPU 上使用 tracemalloc 测量峰值内存（MB）。"""
    gc.collect()
    tracemalloc.start()
    result = func(*args)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_bytes / (1024 * 1024)
    return result, peak_mb


def _measure_memory_gpu(
    device: torch.device, func, *args
) -> tuple[Any, float]:
    """在 GPU 上测量峰值显存（MB）。"""
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    result = func(*args)
    torch.cuda.synchronize(device)
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return result, peak_mb


# ---------------------------------------------------------------------------
# 核心测试函数
# ---------------------------------------------------------------------------


def _build_and_solve(
    N: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    """构建网格、组装矩阵并求解，返回性能指标。"""
    n_cells = N ** 3

    # 1. 生成网格
    _sync(device)
    t0 = time.perf_counter()
    mesh = generate_structured_hex_mesh(N, device=device)
    _sync(device)
    mesh_time = time.perf_counter() - t0

    # 2. 组装扩散矩阵
    h = 1.0 / N
    coeff = 1.0 / (h * h)

    _sync(device)
    t0 = time.perf_counter()

    matrix = LduMatrix(
        n_cells, mesh["owner"], mesh["neighbour"],
        device=device, dtype=dtype,
    )
    matrix.lower = torch.full(
        (mesh["n_internal_faces"],), -coeff, device=device, dtype=dtype
    )
    matrix.upper = torch.full(
        (mesh["n_internal_faces"],), -coeff, device=device, dtype=dtype
    )

    owner = mesh["owner"]
    neighbour = mesh["neighbour"]
    abs_off = torch.full(
        (mesh["n_internal_faces"],), coeff, device=device, dtype=dtype
    )
    row_sum = torch.zeros(n_cells, device=device, dtype=dtype)
    row_sum.scatter_add_(0, owner.long(), abs_off)
    row_sum.scatter_add_(0, neighbour.long(), abs_off)
    matrix.diag = row_sum

    _sync(device)
    assembly_time = time.perf_counter() - t0

    # 3. 构造 RHS 并求解
    ones = torch.ones(n_cells, device=device, dtype=dtype)
    source = matrix.Ax(ones)
    x0 = torch.zeros(n_cells, device=device, dtype=dtype)

    solver = PCGSolver(
        tolerance=TOLERANCE, max_iter=MAX_ITER, preconditioner="none"
    )

    # 预热
    solver(matrix, source, x0.clone())

    # 计时
    solve_times: list[float] = []
    iters = 0
    residual = 0.0
    for _ in range(BENCHMARK_RUNS):
        _sync(device)
        t0 = time.perf_counter()
        solution, iters, residual = solver(matrix, source, x0.clone())
        _sync(device)
        t1 = time.perf_counter()
        solve_times.append(t1 - t0)

    solve_time = _median(solve_times)

    return {
        "n_cells_per_dim": N,
        "n_cells": n_cells,
        "n_internal_faces": mesh["n_internal_faces"],
        "device": str(device),
        "mesh_generation_time_s": round(mesh_time, 6),
        "assembly_time_s": round(assembly_time, 6),
        "solve_time_s": round(solve_time, 6),
        "iterations": iters,
        "residual": residual,
        "time_per_cell_us": round(solve_time / n_cells * 1e6, 4),
    }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def run_large_mesh_benchmarks(
    mesh_sizes: list[int] | None = None,
    output_dir: str | Path = "benchmarks/results",
) -> dict[str, Any]:
    """运行大规模网格基准测试。

    Parameters
    ----------
    mesh_sizes : list[int], optional
        每维网格数。默认 [50, 64, 80, 100]。
    output_dir : str or Path
        结果输出目录。

    Returns
    -------
    dict
        完整测试结果。
    """
    if mesh_sizes is None:
        mesh_sizes = MESH_SIZES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    has_gpu = device.type == "cuda"

    print(f"\n{'='*60}")
    print(f"Large Mesh Benchmark")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if has_gpu:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    cells_range = f"{min(mesh_sizes)**3:,} -- {max(mesh_sizes)**3:,}"
    print(f"Cells range: {cells_range}")
    print(f"{'='*60}\n")

    all_results: dict[str, Any] = {
        "metadata": {
            "pytorch_version": torch.__version__,
            "device": str(device),
            "gpu_available": has_gpu,
            "gpu_name": torch.cuda.get_device_name(0) if has_gpu else "N/A",
            "mesh_sizes": mesh_sizes,
            "benchmark_runs": BENCHMARK_RUNS,
        },
        "mesh_results": [],
        "scaling_analysis": [],
    }

    # --- 逐网格测试 ---
    with device_context(device=device):
        for N in mesh_sizes:
            n_cells = N ** 3
            print(f"--- N={N} ({n_cells:,} cells) ---")

            # 内存测量
            def _build_solve_wrapper():
                return _build_and_solve(N, device)

            if has_gpu:
                result, peak_mem = _measure_memory_gpu(device, _build_solve_wrapper)
            else:
                result, peak_mem = _measure_memory_cpu(_build_solve_wrapper)

            result["peak_memory_mb"] = round(peak_mem, 2)
            all_results["mesh_results"].append(result)

            print(
                f"  mesh={result['mesh_generation_time_s']:.3f}s  "
                f"assembly={result['assembly_time_s']:.3f}s  "
                f"solve={result['solve_time_s']:.3f}s  "
                f"memory={peak_mem:.1f}MB  "
                f"per_cell={result['time_per_cell_us']:.4f}us"
            )

    # --- 扩展分析 ---
    print(f"\n{'='*60}")
    print(f"Scaling Analysis")
    print(f"{'='*60}")
    print(
        f"{'N':>5s}  {'cells':>10s}  {'faces':>12s}  "
        f"{'solve(s)':>10s}  {'us/cell':>10s}  {'mem(MB)':>10s}  {'scaling':>8s}"
    )

    ref_time_per_cell = None
    for r in all_results["mesh_results"]:
        time_per_cell = r["solve_time_s"] / max(r["n_cells"], 1)
        if ref_time_per_cell is None:
            ref_time_per_cell = time_per_cell
            scaling = "ref"
        else:
            ratio = time_per_cell / ref_time_per_cell
            scaling = f"{ratio:.2f}x"

        all_results["scaling_analysis"].append({
            "n_cells_per_dim": r["n_cells_per_dim"],
            "n_cells": r["n_cells"],
            "n_internal_faces": r["n_internal_faces"],
            "solve_time_s": r["solve_time_s"],
            "time_per_cell_us": r["time_per_cell_us"],
            "peak_memory_mb": r["peak_memory_mb"],
            "scaling_vs_smallest": scaling,
        })

        print(
            f"{r['n_cells_per_dim']:5d}  {r['n_cells']:10,}  "
            f"{r['n_internal_faces']:12,}  "
            f"{r['solve_time_s']:10.4f}  {r['time_per_cell_us']:10.4f}  "
            f"{r['peak_memory_mb']:10.1f}  {scaling:>8s}"
        )

    # --- 保存 ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_file = output_path / "large_mesh.json"

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {json_file}")

    return all_results


if __name__ == "__main__":
    run_large_mesh_benchmarks()
