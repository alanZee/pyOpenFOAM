"""
icoFoam 定量基准测试。

运行 Couette 和 Poiseuille 算例，收集 L2 误差数据。
结果保存到 validation/results/icofoam_benchmarks.json。

用法:
    python validation/run_icofoam_benchmarks.py
    python validation/run_icofoam_benchmarks.py --mesh-size 32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main() -> None:
    parser = argparse.ArgumentParser(description="icoFoam 定量基准测试")
    parser.add_argument("--mesh-size", type=int, default=16, help="每维网格数 (默认 16)")
    parser.add_argument("--output", type=str, default="validation/results/icofoam_benchmarks.json")
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from validation.cases.icofoam_benchmarks import run_icofoam_couette, run_icofoam_poiseuille

    n = args.mesh_size
    results = {}

    # Couette 流
    print(f"\n{'='*50}")
    print(f"Couette 流: {n}x{n} 网格, nu=0.01, U_top=1.0")
    print(f"{'='*50}")
    t0 = time.perf_counter()
    try:
        r = run_icofoam_couette(n_cells_x=n, n_cells_y=n)
        t_couette = time.perf_counter() - t0
        results["couette"] = {
            "l2_error": r["l2_error"],
            "max_error": r["max_error"],
            "converged": r["converged"],
            "n_cells": r["n_cells"],
            "time_s": t_couette,
            "pass": r["l2_error"] < 0.05,
        }
        print(f"  L2 误差: {r['l2_error']*100:.2f}%")
        print(f"  最大误差: {r['max_error']:.6f}")
        print(f"  收敛: {r['converged']}")
        print(f"  耗时: {t_couette:.1f}s")
        print(f"  达标 (L2<5%): {'是' if r['l2_error'] < 0.05 else '否'}")
    except Exception as e:
        print(f"  错误: {e}")
        results["couette"] = {"error": str(e)}

    # Poiseuille 流
    print(f"\n{'='*50}")
    print(f"Poiseuille 流: {n}x{n} 网格, nu=0.01, u_mean=1.0")
    print(f"{'='*50}")
    t0 = time.perf_counter()
    try:
        r = run_icofoam_poiseuille(n_cells_x=n, n_cells_y=n)
        t_poiseuille = time.perf_counter() - t0
        results["poiseuille"] = {
            "l2_error": r["l2_error"],
            "max_error": r["max_error"],
            "converged": r["converged"],
            "n_cells": r["n_cells"],
            "time_s": t_poiseuille,
            "pass": r["l2_error"] < 0.05,
        }
        print(f"  L2 误差: {r['l2_error']*100:.2f}%")
        print(f"  最大误差: {r['max_error']:.6f}")
        print(f"  收敛: {r['converged']}")
        print(f"  耗时: {t_poiseuille:.1f}s")
        print(f"  达标 (L2<5%): {'是' if r['l2_error'] < 0.05 else '否'}")
    except Exception as e:
        print(f"  错误: {e}")
        results["poiseuille"] = {"error": str(e)}

    # 保存结果
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果保存至: {output}")


if __name__ == "__main__":
    main()
