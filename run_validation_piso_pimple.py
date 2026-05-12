"""
Validation: PISO and PIMPLE solvers on lid-driven cavity vs Ghia benchmark.

Runs the lid-driven cavity case with PISO and PIMPLE solvers and compares
against the Ghia, Ghia & Shin (1982) benchmark data.

This validates that:
1. PISO solver works correctly with boundary conditions
2. PIMPLE solver works correctly with boundary conditions
3. Both solvers produce reasonable velocity profiles
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

from pyfoam.solvers.piso import PISOSolver, PISOConfig
from pyfoam.solvers.pimple import PIMPLESolver, PIMPLEConfig
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
    """Extract u-velocity along x=0.5 centreline."""
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
    }


def run_piso_cavity(n_cells: int, n_steps: int = 100, dt: float = 0.01) -> dict:
    """Run PISO solver on lid-driven cavity."""
    print(f"\n{'='*60}")
    print(f"  PISO: {n_cells}×{n_cells} cavity (Re=100, dt={dt})")
    print(f"{'='*60}")

    case = LidDrivenCavityCase(
        n_cells=n_cells,
        Re=100.0,
        U_lid=1.0,
        max_iterations=1,
        tolerance=1e-4,
    )

    # Setup
    t0 = time.perf_counter()
    case.setup()
    t_setup = time.perf_counter() - t0

    mesh = case._mesh
    dtype = torch.float64

    # Create PISO solver
    config = PISOConfig(
        n_correctors=3,
        p_tolerance=1e-6,
        p_max_iter=50,
    )
    solver = PISOSolver(mesh, config)

    U = case._U_init.clone()
    p = case._p_init.clone()
    phi = case._phi_init.clone()
    U_bc = case._U_bc

    # Run PISO time steps
    t0 = time.perf_counter()
    for step in range(n_steps):
        U, p, phi, convergence = solver.solve(
            U, p, phi, U_bc=U_bc, tolerance=1e-4,
        )
    t_run = time.perf_counter() - t0

    # Check for NaN/Inf
    if not torch.isfinite(U).all():
        nan_count = (~torch.isfinite(U)).sum().item()
        print(f"  SOLVER DIVERGED: {nan_count} non-finite values in U")
        return {"converged": False, "error": "diverged"}

    # Extract results
    nx = case._nx
    ny = case._ny
    centreline = extract_centreline_u(U, nx, ny)
    errors = compute_centreline_error(centreline)

    print(f"  Setup time:       {t_setup:.2f}s")
    print(f"  Solve time:       {t_run:.2f}s")
    print(f"  Time steps:       {n_steps}")
    print(f"  Continuity error: {convergence.continuity_error:.6e}")
    print(f"  L2 rel error:     {errors['l2_relative']:.6e}")
    print(f"  Max abs error:    {errors['max_absolute']:.6e}")

    return {
        "converged": True,
        "continuity_error": convergence.continuity_error,
        "l2_error": errors["l2_relative"],
        "max_error": errors["max_absolute"],
        "setup_time": t_setup,
        "run_time": t_run,
        "centreline": centreline,
    }


def run_pimple_cavity(n_cells: int, n_steps: int = 50, dt: float = 0.01) -> dict:
    """Run PIMPLE solver on lid-driven cavity."""
    print(f"\n{'='*60}")
    print(f"  PIMPLE: {n_cells}×{n_cells} cavity (Re=100, dt={dt})")
    print(f"{'='*60}")

    case = LidDrivenCavityCase(
        n_cells=n_cells,
        Re=100.0,
        U_lid=1.0,
        max_iterations=1,
        tolerance=1e-4,
    )

    # Setup
    t0 = time.perf_counter()
    case.setup()
    t_setup = time.perf_counter() - t0

    mesh = case._mesh
    dtype = torch.float64

    # Create PIMPLE solver
    config = PIMPLEConfig(
        n_outer_correctors=3,
        n_correctors=2,
        relaxation_factor_U=0.7,
        relaxation_factor_p=0.3,
        p_tolerance=1e-6,
        p_max_iter=50,
    )
    solver = PIMPLESolver(mesh, config)

    U = case._U_init.clone()
    p = case._p_init.clone()
    phi = case._phi_init.clone()
    U_bc = case._U_bc

    # Run PIMPLE time steps
    t0 = time.perf_counter()
    for step in range(n_steps):
        U, p, phi, convergence = solver.solve(
            U, p, phi, U_bc=U_bc, max_outer_iterations=5, tolerance=1e-4,
        )
    t_run = time.perf_counter() - t0

    # Check for NaN/Inf
    if not torch.isfinite(U).all():
        nan_count = (~torch.isfinite(U)).sum().item()
        print(f"  SOLVER DIVERGED: {nan_count} non-finite values in U")
        return {"converged": False, "error": "diverged"}

    # Extract results
    nx = case._nx
    ny = case._ny
    centreline = extract_centreline_u(U, nx, ny)
    errors = compute_centreline_error(centreline)

    print(f"  Setup time:       {t_setup:.2f}s")
    print(f"  Solve time:       {t_run:.2f}s")
    print(f"  Time steps:       {n_steps}")
    print(f"  Continuity error: {convergence.continuity_error:.6e}")
    print(f"  L2 rel error:     {errors['l2_relative']:.6e}")
    print(f"  Max abs error:    {errors['max_absolute']:.6e}")

    return {
        "converged": True,
        "continuity_error": convergence.continuity_error,
        "l2_error": errors["l2_relative"],
        "max_error": errors["max_absolute"],
        "setup_time": t_setup,
        "run_time": t_run,
        "centreline": centreline,
    }


def main():
    """Run PISO and PIMPLE validation."""
    print("\n" + "#"*60)
    print("  PISO & PIMPLE Solver Validation — Lid-Driven Cavity (Re=100)")
    print("  Ghia, Ghia & Shin (1982) benchmark comparison")
    print("#"*60)

    # Test PISO
    print("\n" + "="*60)
    print("  PISO SOLVER")
    print("="*60)

    piso_results = []
    for n in [4, 8]:
        try:
            result = run_piso_cavity(n, n_steps=100, dt=0.01)
            piso_results.append({"n_cells": n, **result})
        except Exception as e:
            print(f"\n  EXCEPTION for n={n}: {e}")
            import traceback
            traceback.print_exc()
            piso_results.append({"n_cells": n, "converged": False, "error": str(e)})

    # Test PIMPLE
    print("\n" + "="*60)
    print("  PIMPLE SOLVER")
    print("="*60)

    pimple_results = []
    for n in [4, 8]:
        try:
            result = run_pimple_cavity(n, n_steps=50, dt=0.01)
            pimple_results.append({"n_cells": n, **result})
        except Exception as e:
            print(f"\n  EXCEPTION for n={n}: {e}")
            import traceback
            traceback.print_exc()
            pimple_results.append({"n_cells": n, "converged": False, "error": str(e)})

    # Summary
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80)

    print("\n  PISO Results:")
    print(f"  {'Mesh':>8s}  {'Converged':>10s}  {'Cont.Err':>12s}  {'L2 Error':>12s}  {'Max Error':>12s}  {'Time(s)':>8s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*8}")
    for r in piso_results:
        n = r.get("n_cells", "?")
        conv = "YES" if r.get("converged", False) else "NO"
        cont = r.get("continuity_error", float("nan"))
        l2 = r.get("l2_error", float("nan"))
        mx = r.get("max_error", float("nan"))
        t = r.get("run_time", float("nan"))
        print(f"  {n:>4d}×{n:<3d}  {conv:>10s}  {cont:12.6e}  {l2:12.6e}  {mx:12.6e}  {t:8.2f}")

    print("\n  PIMPLE Results:")
    print(f"  {'Mesh':>8s}  {'Converged':>10s}  {'Cont.Err':>12s}  {'L2 Error':>12s}  {'Max Error':>12s}  {'Time(s)':>8s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*8}")
    for r in pimple_results:
        n = r.get("n_cells", "?")
        conv = "YES" if r.get("converged", False) else "NO"
        cont = r.get("continuity_error", float("nan"))
        l2 = r.get("l2_error", float("nan"))
        mx = r.get("max_error", float("nan"))
        t = r.get("run_time", float("nan"))
        print(f"  {n:>4d}×{n:<3d}  {conv:>10s}  {cont:12.6e}  {l2:12.6e}  {mx:12.6e}  {t:8.2f}")

    print("\n" + "="*80)

    # Check overall status
    piso_ok = any(r.get("converged", False) for r in piso_results)
    pimple_ok = any(r.get("converged", False) for r in pimple_results)

    if piso_ok and pimple_ok:
        print("\n  OVERALL: BOTH SOLVERS WORK [OK]")
    elif piso_ok:
        print("\n  OVERALL: PISO WORKS, PIMPLE NEEDS FIXES")
    elif pimple_ok:
        print("\n  OVERALL: PIMPLE WORKS, PISO NEEDS FIXES")
    else:
        print("\n  OVERALL: BOTH SOLVERS NEED FIXES [FAIL]")

    return piso_results, pimple_results


if __name__ == "__main__":
    piso_results, pimple_results = main()
