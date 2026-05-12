"""
Full validation: SIMPLE solver on lid-driven cavity vs Ghia benchmark.

Runs 4×4, 8×8, and 16×16 cavities at Re=100, extracts u-velocity
along the vertical centreline (x=0.5), and compares against the
Ghia, Ghia & Shin (1982) benchmark data.

Reports convergence status, continuity error, and velocity error.
"""

from __future__ import annotations

import sys
import time
import logging
import numpy as np
import torch

# Add project to path
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from validation.cases.lid_driven_cavity import (
    LidDrivenCavityCase,
    GHIA_RE100_U_VCENTRELINE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validation")


def interpolate_ghia(y_target: float) -> float:
    """Interpolate Ghia benchmark u-velocity at y_target."""
    ghia_y = [d[0] for d in GHIA_RE100_U_VCENTRELINE]
    ghia_u = [d[1] for d in GHIA_RE100_U_VCENTRELINE]

    if y_target <= ghia_y[0]:
        return ghia_u[0]
    if y_target >= ghia_y[-1]:
        return ghia_u[-1]
    for i in range(len(ghia_y) - 1):
        if ghia_y[i] <= y_target <= ghia_y[i + 1]:
            t = (y_target - ghia_y[i]) / (ghia_y[i + 1] - ghia_y[i])
            return ghia_u[i] + t * (ghia_u[i + 1] - ghia_u[i])
    return ghia_u[-1]


def extract_centreline_u(U: torch.Tensor, nx: int, ny: int) -> list[tuple[float, float]]:
    """Extract u-velocity along x=0.5 centreline.

    Returns list of (y, u) pairs for each cell row.
    """
    i_mid = nx // 2
    results = []
    for j in range(ny):
        y = (j + 0.5) / ny
        idx = j * nx + i_mid
        u_val = U[idx, 0].item()
        results.append((y, u_val))
    return results


def compute_centreline_error(
    computed: list[tuple[float, float]],
    n_cells: int,
) -> dict:
    """Compute L2 and max errors against Ghia benchmark."""
    errors = []
    for y, u_comp in computed:
        u_ghia = interpolate_ghia(y)
        errors.append(abs(u_comp - u_ghia))

    errors = np.array(errors)
    ghia_vals = np.array([interpolate_ghia(y) for y, _ in computed])
    ghia_norm = np.linalg.norm(ghia_vals)

    l2_error = np.linalg.norm(errors) / ghia_norm if ghia_norm > 1e-30 else np.linalg.norm(errors)
    max_error = np.max(errors)

    return {
        "l2_relative": l2_error,
        "max_absolute": max_error,
        "n_points": len(errors),
    }


def run_single_case(n_cells: int, max_iter: int = 2000, tol: float = 1e-4) -> dict:
    """Run a single cavity case and return results."""
    print(f"\n{'='*60}")
    print(f"  Running {n_cells}×{n_cells} cavity (Re=100)")
    print(f"{'='*60}")

    case = LidDrivenCavityCase(
        n_cells=n_cells,
        Re=100.0,
        U_lid=1.0,
        max_iterations=max_iter,
        tolerance=tol,
    )

    # Setup
    t0 = time.perf_counter()
    case.setup()
    t_setup = time.perf_counter() - t0

    # Run
    t0 = time.perf_counter()
    try:
        info = case.run()
    except Exception as e:
        print(f"  SOLVER FAILED: {e}")
        return {
            "n_cells": n_cells,
            "converged": False,
            "diverged_at": "setup/run",
            "error": str(e),
        }
    t_run = time.perf_counter() - t0

    # Extract results
    U_computed = case.get_computed()["U"]
    nx = case._nx
    ny = case._ny

    # Check for NaN/Inf
    if not torch.isfinite(U_computed).all():
        nan_count = (~torch.isfinite(U_computed)).sum().item()
        print(f"  SOLVER DIVERGED: {nan_count} non-finite values in U")
        return {
            "n_cells": n_cells,
            "converged": False,
            "diverged_at": info.get("iterations", "?"),
            "nan_count": nan_count,
        }

    # Centreline extraction
    centreline = extract_centreline_u(U_computed, nx, ny)
    errors = compute_centreline_error(centreline, n_cells)

    # Continuity error
    continuity = info.get("final_residual", float("nan"))

    print(f"  Setup time:       {t_setup:.2f}s")
    print(f"  Solve time:       {t_run:.2f}s")
    print(f"  Iterations:       {info.get('iterations', '?')}")
    print(f"  Converged:        {info.get('converged', False)}")
    print(f"  Continuity error: {continuity:.6e}")
    print(f"  L2 rel error:     {errors['l2_relative']:.6e}")
    print(f"  Max abs error:    {errors['max_absolute']:.6e}")

    # Print centreline comparison
    print(f"\n  Centreline u-velocity (x=0.5):")
    print(f"  {'y':>8s}  {'u_comp':>10s}  {'u_ghia':>10s}  {'error':>10s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")
    for y, u_comp in centreline:
        u_ghia = interpolate_ghia(y)
        err = u_comp - u_ghia
        print(f"  {y:8.4f}  {u_comp:10.6f}  {u_ghia:10.6f}  {err:+10.6f}")

    return {
        "n_cells": n_cells,
        "converged": info.get("converged", False),
        "iterations": info.get("iterations", 0),
        "continuity_error": continuity,
        "l2_error": errors["l2_relative"],
        "max_error": errors["max_absolute"],
        "setup_time": t_setup,
        "run_time": t_run,
        "centreline": centreline,
    }


def main():
    """Run full validation suite."""
    print("\n" + "#"*60)
    print("  SIMPLE Solver Validation — Lid-Driven Cavity (Re=100)")
    print("  Ghia, Ghia & Shin (1982) benchmark comparison")
    print("#"*60)

    mesh_sizes = [4, 8, 16]
    results = []

    for n in mesh_sizes:
        try:
            result = run_single_case(n, max_iter=2000, tol=1e-4)
            results.append(result)
        except Exception as e:
            print(f"\n  EXCEPTION for n={n}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "n_cells": n,
                "converged": False,
                "error": str(e),
            })

    # Summary table
    print("\n" + "="*80)
    print("  SUMMARY TABLE")
    print("="*80)
    print(f"  {'Mesh':>8s}  {'Converged':>10s}  {'Iters':>6s}  {'Cont.Err':>12s}  {'L2 Error':>12s}  {'Max Error':>12s}  {'Time(s)':>8s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*8}")

    for r in results:
        n = r.get("n_cells", "?")
        conv = "YES" if r.get("converged", False) else "NO"
        iters = r.get("iterations", "N/A")
        cont = r.get("continuity_error", float("nan"))
        l2 = r.get("l2_error", float("nan"))
        mx = r.get("max_error", float("nan"))
        t = r.get("run_time", float("nan"))

        if isinstance(cont, float):
            cont_str = f"{cont:.6e}"
        else:
            cont_str = str(cont)

        print(f"  {n:>4d}×{n:<3d}  {conv:>10s}  {str(iters):>6s}  {cont_str:>12s}  {l2:12.6e}  {mx:12.6e}  {t:8.2f}")

    print("="*80)

    # Overall assessment
    any_converged = any(r.get("converged", False) for r in results)
    all_converged = all(r.get("converged", False) for r in results)

    if all_converged:
        print("\n  OVERALL: ALL CASES CONVERGED ✓")
    elif any_converged:
        print("\n  OVERALL: SOME CASES CONVERGED (partial)")
    else:
        print("\n  OVERALL: NO CASES CONVERGED ✗")

    return results


if __name__ == "__main__":
    results = main()
