"""
torch.compile 加速基准测试。

测试关键操作在 torch.compile 前后的性能差异：
1. 梯度计算（Ax + 残差计算）
2. 矩阵-向量乘法（LDU SpMV）
3. 场插值（gather/scatter 操作）

结果保存至 benchmarks/results/torch_compile.json。

用法::

    python -m benchmarks.torch_compile_benchmark
    CUDA_VISIBLE_DEVICES='' python -m benchmarks.torch_compile_benchmark
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from pyfoam.core.ldu_matrix import LduMatrix
from pyfoam.core.device import device_context
from benchmarks.mesh_generation import generate_structured_hex_mesh


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

MESH_SIZES = [32, 64, 128]
WARMUP_RUNS = 2
BENCHMARK_RUNS = 5


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _median(times: list[float]) -> float:
    s = sorted(times)
    return s[len(s) // 2]


# ---------------------------------------------------------------------------
# 核心运算函数（可被 compile 优化的函数）
# ---------------------------------------------------------------------------


def _ldu_ax(
    diag: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """LDU 矩阵-向量乘法 y = A·x。"""
    y = diag * x
    x_p = torch.gather(x, 0, owner.long())
    x_n = torch.gather(x, 0, neighbour.long())
    y.scatter_add_(0, owner.long(), lower * x_n)
    y.scatter_add_(0, neighbour.long(), upper * x_p)
    return y


def _compute_residual(
    diag: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    x: torch.Tensor,
    source: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算残差 r = b - A·x 及其范数。"""
    Ax = _ldu_ax(diag, lower, upper, owner, neighbour, x)
    r = source - Ax
    norm = torch.dot(r, r)
    return r, norm


def _face_interpolation(
    phi: torch.Tensor,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    n_internal: int,
) -> torch.Tensor:
    """线性插值：cell → face（内部面取平均值）。"""
    phi_o = torch.gather(phi, 0, owner[:n_internal].long())
    phi_n = torch.gather(phi, 0, neighbour[:n_internal].long())
    return 0.5 * (phi_o + phi_n)


# ---------------------------------------------------------------------------
# 基准测试函数
# ---------------------------------------------------------------------------


def _benchmark_operation(
    name: str,
    func,
    compiled_func,
    args: tuple,
    device: torch.device,
    n_runs: int = BENCHMARK_RUNS,
) -> dict[str, float]:
    """对单个运算进行 eager vs compile 对比。"""
    # eager 模式预热 + 计时
    for _ in range(WARMUP_RUNS):
        func(*args)
    _sync(device)

    eager_times: list[float] = []
    for _ in range(n_runs):
        _sync(device)
        t0 = time.perf_counter()
        result_eager = func(*args)
        _sync(device)
        t1 = time.perf_counter()
        eager_times.append(t1 - t0)

    # compile 模式预热 + 计时
    for _ in range(WARMUP_RUNS):
        compiled_func(*args)
    _sync(device)

    compile_times: list[float] = []
    for _ in range(n_runs):
        _sync(device)
        t0 = time.perf_counter()
        result_compiled = compiled_func(*args)
        _sync(device)
        t1 = time.perf_counter()
        compile_times.append(t1 - t0)

    eager_median = _median(eager_times)
    compile_median = _median(compile_times)
    speedup = eager_median / max(compile_median, 1e-10)

    print(
        f"  {name:30s} eager={eager_median:.6f}s  "
        f"compile={compile_median:.6f}s  "
        f"speedup={speedup:.2f}x"
    )

    return {
        "operation": name,
        "eager_time_s": round(eager_median, 8),
        "compile_time_s": round(compile_median, 8),
        "speedup": round(speedup, 3),
    }


# ---------------------------------------------------------------------------
# 主基准测试
# ---------------------------------------------------------------------------


def run_compile_benchmarks(
    mesh_sizes: list[int] | None = None,
    output_dir: str | Path = "benchmarks/results",
) -> dict[str, Any]:
    """运行 torch.compile 基准测试。

    Parameters
    ----------
    mesh_sizes : list[int], optional
        每维网格数。默认 [32, 64, 128]。
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
    dtype = torch.float64

    print(f"\n{'='*60}")
    print(f"torch.compile Benchmark")
    print(f"{'='*60}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Mesh sizes: {mesh_sizes}")
    print(f"{'='*60}")

    all_results: dict[str, Any] = {
        "metadata": {
            "pytorch_version": torch.__version__,
            "device": str(device),
            "mesh_sizes": mesh_sizes,
            "warmup_runs": WARMUP_RUNS,
            "benchmark_runs": BENCHMARK_RUNS,
        },
        "results_by_mesh": [],
    }

    # 编译函数
    compiled_ax = torch.compile(_ldu_ax)
    compiled_residual = torch.compile(_compute_residual)
    compiled_interp = torch.compile(_face_interpolation)

    with device_context(device=device):
        for N in mesh_sizes:
            n_cells = N ** 3
            print(f"\n--- N={N} ({n_cells:,} cells) ---")

            mesh = generate_structured_hex_mesh(N, device=device)
            owner = mesh["owner"]
            neighbour = mesh["neighbour"]
            n_internal = mesh["n_internal_faces"]

            # 构造 LDU 矩阵数据
            h = 1.0 / N
            coeff = 1.0 / (h * h)

            diag = torch.full((n_cells,), 6.0 * coeff, device=device, dtype=dtype)
            lower = torch.full((n_internal,), -coeff, device=device, dtype=dtype)
            upper = torch.full((n_internal,), -coeff, device=device, dtype=dtype)
            x = torch.randn(n_cells, device=device, dtype=dtype)
            source = torch.randn(n_cells, device=device, dtype=dtype)

            mesh_results: dict[str, Any] = {
                "n_cells_per_dim": N,
                "n_cells": n_cells,
                "operations": [],
            }

            # 1. 矩阵-向量乘法
            args_ax = (diag, lower, upper, owner, neighbour, x)
            r = _benchmark_operation(
                "ldu_ax (SpMV)", _ldu_ax, compiled_ax, args_ax, device
            )
            r["category"] = "spmv"
            mesh_results["operations"].append(r)

            # 2. 残差计算
            args_res = (diag, lower, upper, owner, neighbour, x, source)
            r = _benchmark_operation(
                "compute_residual", _compute_residual, compiled_residual,
                args_res, device
            )
            r["category"] = "gradient"
            mesh_results["operations"].append(r)

            # 3. 场插值
            phi = torch.randn(n_cells, device=device, dtype=dtype)
            args_interp = (phi, owner, neighbour, n_internal)
            r = _benchmark_operation(
                "face_interpolation", _face_interpolation, compiled_interp,
                args_interp, device
            )
            r["category"] = "interpolation"
            mesh_results["operations"].append(r)

            all_results["results_by_mesh"].append(mesh_results)

    # --- 汇总 ---
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    for mesh_r in all_results["results_by_mesh"]:
        N = mesh_r["n_cells_per_dim"]
        for op in mesh_r["operations"]:
            tag = "BENEFITS" if op["speedup"] > 1.1 else (
                "NEUTRAL" if op["speedup"] > 0.9 else "SLOWER"
            )
            print(
                f"  N={N:3d}  {op['operation']:30s}  "
                f"speedup={op['speedup']:.2f}x  [{tag}]"
            )

    # --- 保存 ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_file = output_path / "torch_compile.json"

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {json_file}")

    return all_results


if __name__ == "__main__":
    run_compile_benchmarks()
