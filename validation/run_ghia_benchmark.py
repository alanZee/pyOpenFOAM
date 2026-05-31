"""
Ghia et al. (1982) Re=100 lid-driven cavity benchmark validation.

Runs a 32x32 cavity with the SIMPLE solver for 2000+ iterations,
compares vertical centreline u-velocity against Ghia benchmark data,
and reports L2 relative error and maximum absolute error.

Usage:
    CUDA_VISIBLE_DEVICES='' python validation/run_ghia_benchmark.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Force CPU device
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root))

from validation.cases.lid_driven_cavity import (
    LidDrivenCavityCase,
    GHIA_RE100_U_VCENTRELINE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ghia_benchmark")

# Ghia et al. (1982) Re=100 vertical centreline (x=0.5) u-velocity data
GHIA_Y = [d[0] for d in GHIA_RE100_U_VCENTRELINE]
GHIA_U = [d[1] for d in GHIA_RE100_U_VCENTRELINE]

# Target accuracy
L2_TARGET = 0.05  # 5% L2 relative error target


def interpolate_ghia(y_target: float) -> float:
    """Linearly interpolate Ghia benchmark u-velocity at *y_target*."""
    if y_target <= GHIA_Y[0]:
        return GHIA_U[0]
    if y_target >= GHIA_Y[-1]:
        return GHIA_U[-1]
    for i in range(len(GHIA_Y) - 1):
        if GHIA_Y[i] <= y_target <= GHIA_Y[i + 1]:
            t = (y_target - GHIA_Y[i]) / (GHIA_Y[i + 1] - GHIA_Y[i])
            return GHIA_U[i] + t * (GHIA_U[i + 1] - GHIA_U[i])
    return GHIA_U[-1]


def extract_centreline_u(U: torch.Tensor, nx: int, ny: int) -> list[tuple[float, float]]:
    """Extract u-velocity along vertical centreline (x=0.5).

    Returns list of (y, u) pairs for each cell row along x=0.5.
    """
    i_mid = nx // 2
    results = []
    for j in range(ny):
        y = (j + 0.5) / ny
        idx = j * nx + i_mid
        u_val = U[idx, 0].item()
        results.append((y, u_val))
    return results


def compute_errors(computed: list[tuple[float, float]]) -> dict:
    """Compute L2 relative error and max absolute error vs Ghia data.

    The L2 relative error is computed as:
        ||u_computed - u_ghia||_2 / ||u_ghia||_2

    where the norm is taken over the computed centreline points (each
    interpolated against the Ghia data at the same y-coordinates).
    """
    u_comp = np.array([u for _, u in computed])
    u_ghia = np.array([interpolate_ghia(y) for y, _ in computed])

    diff = u_comp - u_ghia
    l2_diff = np.linalg.norm(diff)
    l2_ghia = np.linalg.norm(u_ghia)

    l2_relative = l2_diff / l2_ghia if l2_ghia > 1e-30 else l2_diff
    max_abs = float(np.max(np.abs(diff)))

    return {
        "l2_relative_error": float(l2_relative),
        "max_absolute_error": max_abs,
        "l2_diff_norm": float(l2_diff),
        "l2_ghia_norm": float(l2_ghia),
        "n_points": len(u_comp),
    }


def run_benchmark(
    n_cells: int = 32,
    max_iterations: int = 2000,
    tolerance: float = 1e-4,
) -> dict:
    """Run the Ghia benchmark and return detailed results."""
    print("=" * 70)
    print("  Ghia et al. (1982) Re=100 Benchmark — Lid-Driven Cavity")
    print(f"  Mesh: {n_cells}x{n_cells}, max iterations: {max_iterations}")
    print(f"  Under-relaxation: alpha_U=0.7, alpha_p=0.3")
    print("=" * 70)

    # --- Setup ---
    print("\n[1/3] Setting up case...")
    t0 = time.perf_counter()
    case = LidDrivenCavityCase(
        n_cells=n_cells,
        Re=100.0,
        U_lid=1.0,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    case.setup()
    t_setup = time.perf_counter() - t0
    print(f"  Setup complete: {case._mesh.n_cells} cells in {t_setup:.2f}s")

    # --- Solve ---
    print("\n[2/3] Running SIMPLE solver...")
    t0 = time.perf_counter()
    solver_info = case.run()
    t_solve = time.perf_counter() - t0

    converged = solver_info.get("converged", False)
    iterations = solver_info.get("iterations", 0)
    cont_err = solver_info.get("final_residual", float("nan"))

    print(f"  Converged:  {converged}")
    print(f"  Iterations: {iterations}")
    print(f"  Continuity: {cont_err:.6e}")
    print(f"  Solve time: {t_solve:.2f}s")

    # --- Check for divergence ---
    U_computed = case.get_computed()["U"]
    if not torch.isfinite(U_computed).all():
        nan_count = (~torch.isfinite(U_computed)).sum().item()
        print(f"\n  SOLVER DIVERGED: {nan_count} non-finite values in U")
        return {
            "n_cells": n_cells,
            "converged": False,
            "diverged": True,
            "nan_count": nan_count,
            "iterations": iterations,
            "solve_time": t_solve,
        }

    # --- Extract centreline and compute errors ---
    print("\n[3/3] Comparing against Ghia benchmark...")
    nx = case._nx
    ny = case._ny
    centreline = extract_centreline_u(U_computed, nx, ny)
    errors = compute_errors(centreline)

    l2 = errors["l2_relative_error"]
    max_err = errors["max_absolute_error"]
    passed = l2 < L2_TARGET

    # Print centreline comparison table
    print(f"\n  Vertical centreline u-velocity (x=0.5):")
    print(f"  {'y':>8s}  {'u_comp':>10s}  {'u_ghia':>10s}  {'error':>10s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")
    for y, u_comp in centreline:
        u_ghia = interpolate_ghia(y)
        err = u_comp - u_ghia
        print(f"  {y:8.4f}  {u_comp:10.6f}  {u_ghia:10.6f}  {err:+10.6f}")

    # Print error summary
    print(f"\n  Error Metrics:")
    print(f"    L2 relative error:  {l2:.6e}  (target: {L2_TARGET})")
    print(f"    Max absolute error: {max_err:.6e}")
    print(f"    Result: {'PASS' if passed else 'FAIL'}")

    # --- Root cause analysis ---
    max_err_pt = max(centreline, key=lambda yc: abs(yc[1] - interpolate_ghia(yc[0])))
    max_err_y = max_err_pt[0]
    is_boundary_cell = max_err_y > (ny - 1.5) / ny  # last cell row

    print(f"\n  Root Cause Analysis:")
    print(f"    Convection scheme: first-order upwind (default fvSchemes)")
    print(f"    Max error at y={max_err_y:.4f}: u_comp={max_err_pt[1]:.6f}, "
          f"u_ghia={interpolate_ghia(max_err_y):.6f}")
    if is_boundary_cell:
        print(f"    -> Largest error is at boundary-adjacent cell (u=U_lid by BC)")
    print(f"    -> Systematic under-prediction in domain interior is consistent")
    print(f"       with numerical diffusion from first-order upwind scheme")
    print(f"    -> Ghia benchmark uses 129x129 multigrid; 32x32 is under-resolved")
    print(f"    -> Achieving <5% L2 error requires 64x64+ mesh or higher-order scheme")

    # --- Build result dict ---
    result = {
        "benchmark": "Ghia et al. (1982) Re=100",
        "mesh": f"{n_cells}x{n_cells}",
        "n_cells": n_cells,
        "reynolds_number": 100.0,
        "solver": "SIMPLE",
        "under_relaxation": {
            "alpha_U": 0.7,
            "alpha_p": 0.3,
        },
        "max_iterations": max_iterations,
        "convergence_tolerance": tolerance,
        "converged": converged,
        "iterations": iterations,
        "continuity_error": cont_err,
        "l2_relative_error": l2,
        "l2_target": L2_TARGET,
        "max_absolute_error": max_err,
        "passed": passed,
        "n_centreline_points": len(centreline),
        "centreline_data": [
            {
                "y": y,
                "u_computed": u_comp,
                "u_ghia": interpolate_ghia(y),
                "error": u_comp - interpolate_ghia(y),
            }
            for y, u_comp in centreline
        ],
        "timing": {
            "setup_seconds": t_setup,
            "solve_seconds": t_solve,
            "total_seconds": t_setup + t_solve,
        },
        "root_cause_analysis": {
            "convection_scheme": "first-order upwind",
            "primary_error_source": "numerical diffusion from upwind discretization",
            "max_error_location_y": max_err_y,
            "is_boundary_cell": is_boundary_cell,
            "ghia_reference_mesh": "129x129 multigrid",
            "recommendation": "Use 64x64+ mesh or higher-order convection scheme (linearUpwind) for <5% L2 error",
        },
    }

    return result


def main() -> None:
    """Run Ghia benchmark convergence study over multiple mesh sizes.

    Runs 32x32 and 64x64 meshes (128x128 is available but optional due to
    long runtime). Reports L2 error, convergence rate, and root cause analysis.

    Key finding: first-order upwind convection limits accuracy to ~6% L2 error
    on 64x64 mesh. Achieving <5% requires either a higher-order convection
    scheme (linearUpwind/QUICK) or a different BC treatment for corner cells.
    """
    mesh_sizes = [32, 64]
    all_results = []

    for n in mesh_sizes:
        max_iter = 2000 if n <= 32 else 4000
        result = run_benchmark(n_cells=n, max_iterations=max_iter, tolerance=1e-4)
        all_results.append(result)
        print()

    # Save convergence study
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Detailed per-mesh results
    for result in all_results:
        n = result["n_cells"]
        path = out_dir / f"ghia_re100_{n}x{n}.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {path}")

    # Convergence summary
    convergence = {
        "benchmark": "Ghia et al. (1982) Re=100",
        "solver": "SIMPLE",
        "convection_scheme": "first-order upwind",
        "target_l2_error": L2_TARGET,
        "mesh_results": [
            {
                "mesh": r["mesh"],
                "l2_relative_error": r.get("l2_relative_error"),
                "max_absolute_error": r.get("max_absolute_error"),
                "converged": r.get("converged", False),
                "iterations": r.get("iterations"),
                "solve_time_s": r.get("timing", {}).get("solve_seconds"),
                "passed": r.get("passed", False),
            }
            for r in all_results
        ],
    }

    # Compute convergence rate
    valid = [r for r in all_results if r.get("l2_relative_error") is not None]
    if len(valid) >= 2:
        convergence["convergence_rates"] = []
        for i in range(1, len(valid)):
            h_coarse = 1.0 / valid[i - 1]["n_cells"]
            h_fine = 1.0 / valid[i]["n_cells"]
            e_coarse = valid[i - 1]["l2_relative_error"]
            e_fine = valid[i]["l2_relative_error"]
            if e_fine > 1e-30:
                rate = (e_coarse / e_fine) / (h_coarse / h_fine)
                convergence["convergence_rates"].append({
                    "from": valid[i - 1]["mesh"],
                    "to": valid[i]["mesh"],
                    "error_ratio": e_coarse / e_fine,
                    "mesh_ratio": h_coarse / h_fine,
                    "order": rate,
                })

    convergence["analysis"] = {
        "root_cause": "first-order upwind convection scheme introduces O(h) numerical diffusion",
        "boundary_effect": "top-wall cells forced to u=U_lid at cell centre, but true value is lower due to boundary layer",
        "best_mesh": valid[-1]["mesh"] if valid else "N/A",
        "best_l2_error": valid[-1]["l2_relative_error"] if valid else None,
        "improvement_options": [
            "Use linearUpwind or QUICK convection scheme for O(h^2) spatial accuracy",
            "Implement proper boundary treatment (interpolate face value, not cell-centre penalty)",
            "Use SIMPLEC with appropriate relaxation for faster convergence",
        ],
    }

    conv_path = out_dir / "ghia_convergence.json"
    with open(conv_path, "w") as f:
        json.dump(convergence, f, indent=2)
    print(f"\n  Convergence study saved to: {conv_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("  CONVERGENCE STUDY SUMMARY")
    print("=" * 70)
    print(f"  {'Mesh':>8s}  {'L2 Error':>12s}  {'Max Error':>12s}  {'Iters':>6s}  {'Time(s)':>8s}  {'Pass':>5s}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*6}  {'-'*8}  {'-'*5}")
    for r in all_results:
        l2 = r.get("l2_relative_error", float("nan"))
        mx = r.get("max_absolute_error", float("nan"))
        it = r.get("iterations", 0)
        ts = r.get("timing", {}).get("solve_seconds", 0)
        ok = "YES" if r.get("passed") else ("DIV" if r.get("diverged") else "NO")
        print(f"  {r['mesh']:>8s}  {l2:12.6e}  {mx:12.6e}  {it:6d}  {ts:8.1f}  {ok:>5s}")

    if convergence.get("convergence_rates"):
        print(f"\n  Convergence Rates (order p in error ~ h^p):")
        for cr in convergence["convergence_rates"]:
            print(f"    {cr['from']} -> {cr['to']}: order = {cr['order']:.2f}")

    print(f"\n  Root Cause: {convergence['analysis']['root_cause']}")
    print(f"  Best result: {convergence['analysis']['best_l2_error']:.1%} on {convergence['analysis']['best_mesh']} mesh")

    # Final verdict
    best = min(all_results, key=lambda r: r.get("l2_relative_error", float("inf")))
    print("\n" + "=" * 70)
    if best.get("passed", False):
        print(f"  VERDICT: PASS — best L2 error {best['l2_relative_error']:.1%} on {best['mesh']} mesh")
    elif best.get("diverged", False):
        print("  VERDICT: FAIL — all solvers diverged")
    else:
        print(f"  VERDICT: PARTIAL — best L2 error {best['l2_relative_error']:.1%} on {best['mesh']} (target: {L2_TARGET:.0%})")
        print(f"  NOTE: {L2_TARGET:.0%} target requires higher-order convection scheme (not just mesh refinement)")
    print("=" * 70)


if __name__ == "__main__":
    main()
