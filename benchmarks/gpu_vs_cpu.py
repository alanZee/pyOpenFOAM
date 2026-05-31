"""
GPU vs CPU 综合性能基准测试。

对比 CPU 与 GPU 在 SIMPLE 算法各阶段的性能表现：
1. 矩阵组装（动量方程 + 压力方程）
2. 线性求解（压力方程 PCG + 速度方程 PCG）
3. 完整 SIMPLE 迭代

支持的网格尺寸：32×32、64×64、128×128。

结果保存至 benchmarks/results/gpu_vs_cpu.json。

用法::

    # 自动检测 GPU
    python -m benchmarks.gpu_vs_cpu

    # 强制 CPU
    CUDA_VISIBLE_DEVICES='' python -m benchmarks.gpu_vs_cpu

    # WSL 环境
    wsl -d Ubuntu-20.04 /home/alanz/miniconda3/envs/pyopenfoam/bin/python -m benchmarks.gpu_vs_cpu
"""

from __future__ import annotations

import json
import time
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

MESH_SIZES = [32, 64]  # cells per dimension
WARMUP_RUNS = 1
BENCHMARK_RUNS = 3
TOLERANCE = 1e-6
MAX_ITER = 500


def _sync(device: torch.device) -> None:
    """GPU 同步，确保计时准确。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _median(times: list[float]) -> float:
    """返回中位数。"""
    s = sorted(times)
    return s[len(s) // 2]


# ---------------------------------------------------------------------------
# 矩阵组装函数
# ---------------------------------------------------------------------------


def _assemble_momentum(
    mesh: dict,
    U: torch.Tensor,
    p: torch.Tensor,
    dt: float = 1.0,
    nu: float = 0.01,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> FvMatrix:
    """组装动量方程（简化 UEqn）。

    使用一阶迎风对流 + 中心差分扩散。
    """
    n_cells = mesh["n_cells"]
    owner = mesh["owner"]
    neighbour = mesh["neighbour"]
    n_internal = mesh["n_internal_faces"]

    mat = FvMatrix(n_cells, owner, neighbour, device=device, dtype=dtype)

    h = 1.0 / (n_cells ** (1.0 / 3.0))
    diff_coeff = nu / (h * h)
    conv_coeff = 1.0 / h

    lower_coeff = -(diff_coeff + max(conv_coeff, 0.0))
    upper_coeff = -(diff_coeff + max(-conv_coeff, 0.0))

    mat.lower = torch.full((n_internal,), lower_coeff, device=device, dtype=dtype)
    mat.upper = torch.full((n_internal,), upper_coeff, device=device, dtype=dtype)

    abs_lower = mat.lower.abs()
    abs_upper = mat.upper.abs()

    row_sum = torch.zeros(n_cells, device=device, dtype=dtype)
    row_sum.scatter_add_(0, owner.long(), abs_lower)
    row_sum.scatter_add_(0, neighbour.long(), abs_upper)
    mat.diag = row_sum + 1.0 / dt

    mat.source = torch.zeros(n_cells, device=device, dtype=dtype)

    return mat


def _assemble_pressure(
    mesh: dict,
    U: torch.Tensor,
    dt: float = 1.0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> FvMatrix:
    """组装压力方程（简化 pEqn）。

    压力 Poisson 方程。
    """
    n_cells = mesh["n_cells"]
    owner = mesh["owner"]
    neighbour = mesh["neighbour"]
    n_internal = mesh["n_internal_faces"]

    mat = FvMatrix(n_cells, owner, neighbour, device=device, dtype=dtype)

    h = 1.0 / (n_cells ** (1.0 / 3.0))
    coeff = 1.0 / (h * h)

    mat.lower = -torch.full((n_internal,), coeff, device=device, dtype=dtype)
    mat.upper = -torch.full((n_internal,), coeff, device=device, dtype=dtype)

    abs_lower = mat.lower.abs()
    abs_upper = mat.upper.abs()

    row_sum = torch.zeros(n_cells, device=device, dtype=dtype)
    row_sum.scatter_add_(0, owner.long(), abs_lower)
    row_sum.scatter_add_(0, neighbour.long(), abs_upper)
    mat.diag = row_sum

    mat.source = torch.randn(n_cells, device=device, dtype=dtype) * 0.01

    return mat


# ---------------------------------------------------------------------------
# PCG 求解器（内联，用于组装到求解的计时分离）
# ---------------------------------------------------------------------------


def _solve_pcg(
    matrix: LduMatrix,
    source: torch.Tensor,
    x0: torch.Tensor,
    tolerance: float = TOLERANCE,
    max_iter: int = MAX_ITER,
) -> tuple[torch.Tensor, int, float]:
    """简单 PCG 求解器。"""
    x = x0.clone()
    r = source - matrix.Ax(x)
    p = r.clone()
    rsold = torch.dot(r, r)

    for i in range(max_iter):
        Ap = matrix.Ax(p)
        alpha = rsold / (torch.dot(p, Ap) + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r, r)
        residual = torch.sqrt(rsnew).item()
        if residual < tolerance:
            return x, i + 1, residual
        beta = rsnew / (rsold + 1e-30)
        p = r + beta * p
        rsold = rsnew

    return x, max_iter, torch.sqrt(rsold).item()


# ---------------------------------------------------------------------------
# 基准测试：SIMPLE 迭代各阶段计时
# ---------------------------------------------------------------------------


def _benchmark_simple_iteration(
    N: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    """对单个网格尺寸执行 SIMPLE 迭代各阶段的基准测试。

    返回各阶段中位数耗时。
    """
    n_cells = N ** 3
    print(f"  N={N} ({n_cells:,} cells) on {device}...", end="", flush=True)

    mesh = generate_structured_hex_mesh(N, device=device)
    U = torch.zeros(n_cells, device=device, dtype=dtype)
    p = torch.zeros(n_cells, device=device, dtype=dtype)

    # --- 预热 ---
    for _ in range(WARMUP_RUNS):
        Ueqn = _assemble_momentum(mesh, U, p, device=device, dtype=dtype)
        _solve_pcg(Ueqn, Ueqn.source, U, tolerance=1e-3, max_iter=20)
        peqn = _assemble_pressure(mesh, U, device=device, dtype=dtype)
        _solve_pcg(peqn, peqn.source, p, tolerance=1e-3, max_iter=20)

    # --- 分阶段计时 ---
    assembly_times: list[float] = []
    solve_times: list[float] = []
    full_iter_times: list[float] = []

    for _ in range(BENCHMARK_RUNS):
        _sync(device)
        t_full_start = time.perf_counter()

        # 1. 组装阶段
        _sync(device)
        t0 = time.perf_counter()

        Ueqn = _assemble_momentum(mesh, U, p, device=device, dtype=dtype)
        peqn = _assemble_pressure(mesh, U, device=device, dtype=dtype)

        _sync(device)
        t1 = time.perf_counter()
        assembly_times.append(t1 - t0)

        # 2. 求解阶段
        _sync(device)
        t2 = time.perf_counter()

        U, u_iters, u_res = _solve_pcg(
            Ueqn, Ueqn.source, U, tolerance=TOLERANCE, max_iter=MAX_ITER
        )
        p, p_iters, p_res = _solve_pcg(
            peqn, peqn.source, p, tolerance=TOLERANCE, max_iter=MAX_ITER
        )

        _sync(device)
        t3 = time.perf_counter()
        solve_times.append(t3 - t2)

        # 3. 速度修正（简化）
        U = U - 0.01 * torch.randn_like(U)

        _sync(device)
        t_full_end = time.perf_counter()
        full_iter_times.append(t_full_end - t_full_start)

    result = {
        "n_cells_per_dim": N,
        "n_cells": n_cells,
        "device": str(device),
        "assembly_time_s": round(_median(assembly_times), 6),
        "solve_time_s": round(_median(solve_times), 6),
        "full_iteration_time_s": round(_median(full_iter_times), 6),
        "momentum_iters": u_iters,
        "pressure_iters": p_iters,
        "momentum_residual": u_res,
        "pressure_residual": p_res,
    }

    print(
        f" assembly={result['assembly_time_s']:.4f}s, "
        f"solve={result['solve_time_s']:.4f}s, "
        f"total={result['full_iteration_time_s']:.4f}s"
    )

    return result


# ---------------------------------------------------------------------------
# 速度方程单独求解基准（使用 PCGSolver 类）
# ---------------------------------------------------------------------------


def _benchmark_solver_class(
    N: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    """使用 PCGSolver 类对扩散矩阵求解，衡量封装开销。"""
    n_cells = N ** 3
    mesh = generate_structured_hex_mesh(N, device=device)

    h = 1.0 / N
    coeff = 1.0 / (h * h)

    matrix = LduMatrix(n_cells, mesh["owner"], mesh["neighbour"],
                       device=device, dtype=dtype)
    matrix.lower = torch.full((mesh["n_internal_faces"],), -coeff,
                              device=device, dtype=dtype)
    matrix.upper = torch.full((mesh["n_internal_faces"],), -coeff,
                              device=device, dtype=dtype)

    owner = mesh["owner"]
    neighbour = mesh["neighbour"]
    abs_off = torch.full((mesh["n_internal_faces"],), coeff,
                         device=device, dtype=dtype)
    row_sum = torch.zeros(n_cells, device=device, dtype=dtype)
    row_sum.scatter_add_(0, owner.long(), abs_off)
    row_sum.scatter_add_(0, neighbour.long(), abs_off)
    matrix.diag = row_sum

    ones = torch.ones(n_cells, device=device, dtype=dtype)
    source = matrix.Ax(ones)
    x0 = torch.zeros(n_cells, device=device, dtype=dtype)

    solver = PCGSolver(
        tolerance=TOLERANCE, max_iter=MAX_ITER, preconditioner="none"
    )

    # 预热
    solver(matrix, source, x0.clone())

    # 计时
    times: list[float] = []
    for _ in range(BENCHMARK_RUNS):
        _sync(device)
        t0 = time.perf_counter()
        solution, iters, residual = solver(matrix, source, x0.clone())
        _sync(device)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return {
        "n_cells_per_dim": N,
        "n_cells": n_cells,
        "device": str(device),
        "pcg_solver_time_s": round(_median(times), 6),
        "iterations": iters,
        "residual": residual,
    }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def run_gpu_vs_cpu(
    mesh_sizes: list[int] | None = None,
    output_dir: str | Path = "benchmarks/results",
) -> dict[str, Any]:
    """运行 GPU vs CPU 综合基准测试。

    Parameters
    ----------
    mesh_sizes : list[int], optional
        每维网格数。默认 [32, 64]。
    output_dir : str or Path
        结果输出目录。

    Returns
    -------
    dict
        完整基准测试结果。
    """
    if mesh_sizes is None:
        mesh_sizes = MESH_SIZES

    has_gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_gpu else "N/A"

    print(f"\n{'='*60}")
    print(f"GPU vs CPU Performance Benchmark")
    print(f"{'='*60}")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU Available: {has_gpu}")
    if has_gpu:
        print(f"GPU: {gpu_name}")
        print(f"CUDA: {torch.version.cuda}")
    print(f"Mesh sizes: {mesh_sizes}")
    cells_range = f"{min(mesh_sizes)**3:,} -- {max(mesh_sizes)**3:,}"
    print(f"Cells range: {cells_range}")
    print(f"{'='*60}\n")

    devices = [torch.device("cpu")]
    if has_gpu:
        devices.append(torch.device("cuda"))

    all_results: dict[str, Any] = {
        "metadata": {
            "pytorch_version": torch.__version__,
            "gpu_available": has_gpu,
            "gpu_name": gpu_name,
            "cuda_version": torch.version.cuda if has_gpu else None,
            "mesh_sizes": mesh_sizes,
            "warmup_runs": WARMUP_RUNS,
            "benchmark_runs": BENCHMARK_RUNS,
        },
        "simple_iteration_results": [],
        "pcg_solver_results": [],
        "speedup_comparison": [],
    }

    # --- 阶段 1：SIMPLE 迭代各阶段基准 ---
    print("[Phase 1] SIMPLE Iteration Breakdown")
    for device in devices:
        with device_context(device=device):
            for N in mesh_sizes:
                result = _benchmark_simple_iteration(N, device)
                all_results["simple_iteration_results"].append(result)

    # --- 阶段 2：PCGSolver 类基准 ---
    print(f"\n[Phase 2] PCGSolver Class Benchmark")
    for device in devices:
        with device_context(device=device):
            for N in mesh_sizes:
                print(f"  PCGSolver N={N} on {device}...", end="", flush=True)
                result = _benchmark_solver_class(N, device)
                all_results["pcg_solver_results"].append(result)
                print(f" {result['pcg_solver_time_s']:.4f}s ({result['iterations']} iter)")

    # --- 阶段 3：计算加速比 ---
    if has_gpu:
        print(f"\n[Phase 3] Speedup Analysis")
        for N in mesh_sizes:
            cpu_simple = next(
                r for r in all_results["simple_iteration_results"]
                if r["n_cells_per_dim"] == N and r["device"] == "cpu"
            )
            gpu_simple = next(
                r for r in all_results["simple_iteration_results"]
                if r["n_cells_per_dim"] == N and r["device"] == "cuda"
            )
            cpu_pcg = next(
                r for r in all_results["pcg_solver_results"]
                if r["n_cells_per_dim"] == N and r["device"] == "cpu"
            )
            gpu_pcg = next(
                r for r in all_results["pcg_solver_results"]
                if r["n_cells_per_dim"] == N and r["device"] == "cuda"
            )

            speedup_entry = {
                "n_cells_per_dim": N,
                "n_cells": N ** 3,
                "gpu_name": gpu_name,
                # 组装加速比
                "assembly_speedup": round(
                    cpu_simple["assembly_time_s"]
                    / max(gpu_simple["assembly_time_s"], 1e-10),
                    3,
                ),
                # 求解加速比
                "solve_speedup": round(
                    cpu_simple["solve_time_s"]
                    / max(gpu_simple["solve_time_s"], 1e-10),
                    3,
                ),
                # 完整迭代加速比
                "full_iteration_speedup": round(
                    cpu_simple["full_iteration_time_s"]
                    / max(gpu_simple["full_iteration_time_s"], 1e-10),
                    3,
                ),
                # PCGSolver 类加速比
                "pcg_solver_speedup": round(
                    cpu_pcg["pcg_solver_time_s"]
                    / max(gpu_pcg["pcg_solver_time_s"], 1e-10),
                    3,
                ),
                # 绝对时间
                "cpu_times": {
                    "assembly": cpu_simple["assembly_time_s"],
                    "solve": cpu_simple["solve_time_s"],
                    "full_iteration": cpu_simple["full_iteration_time_s"],
                    "pcg_solver": cpu_pcg["pcg_solver_time_s"],
                },
                "gpu_times": {
                    "assembly": gpu_simple["assembly_time_s"],
                    "solve": gpu_simple["solve_time_s"],
                    "full_iteration": gpu_simple["full_iteration_time_s"],
                    "pcg_solver": gpu_pcg["pcg_solver_time_s"],
                },
            }
            all_results["speedup_comparison"].append(speedup_entry)

            print(
                f"  N={N}: assembly={speedup_entry['assembly_speedup']:.2f}x, "
                f"solve={speedup_entry['solve_speedup']:.2f}x, "
                f"full_iter={speedup_entry['full_iteration_speedup']:.2f}x, "
                f"pcg={speedup_entry['pcg_solver_speedup']:.2f}x"
            )

    # --- 保存结果 ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_file = output_path / "gpu_vs_cpu.json"

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {json_file}")

    return all_results


if __name__ == "__main__":
    run_gpu_vs_cpu()
