"""
Benchmark results visualization.

Generates plots from benchmark CSV files using matplotlib.
Falls back to CSV-only output if matplotlib is not available.

Plots:
1. Solve time vs mesh size (log-log) for each solver
2. GPU vs CPU speedup bar chart
3. Memory usage vs mesh size
4. Iteration count vs mesh size
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

# Try importing matplotlib
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _load_csv(filepath: str | Path) -> list[dict[str, Any]]:
    """Load a CSV file into a list of dicts."""
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _to_float(val: str | None) -> float | None:
    """Convert string to float, handling None."""
    if val is None or val == "" or val == "None":
        return None
    return float(val)


def _to_int(val: str | None) -> int | None:
    """Convert string to int, handling None."""
    if val is None or val == "" or val == "None":
        return None
    return int(float(val))


# ---------------------------------------------------------------------------
# Plot: Solve time vs mesh size
# ---------------------------------------------------------------------------


def plot_solve_time_vs_mesh(
    results_dir: str | Path = "benchmarks/results",
    output_dir: str | Path | None = None,
) -> None:
    """Plot solve time vs number of cells for each solver.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing ``linear_solve_benchmark.csv``.
    output_dir : str or Path, optional
        Directory for output plot. Defaults to results_dir.
    """
    results_path = Path(results_dir)
    csv_file = results_path / "linear_solve_benchmark.csv"

    if not csv_file.exists():
        print(f"Warning: {csv_file} not found, skipping plot.")
        return

    data = _load_csv(csv_file)

    # Group by solver
    solvers: dict[str, dict[str, list]] = {}
    for row in data:
        solver = row["solver"]
        if solver not in solvers:
            solvers[solver] = {"n_cells": [], "time": [], "iterations": []}
        solvers[solver]["n_cells"].append(int(row["n_cells"]))
        solvers[solver]["time"].append(float(row["time_seconds"]))
        solvers[solver]["iterations"].append(int(row["iterations"]))

    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot generation.")
        return

    output_path = Path(output_dir) if output_dir else results_path
    output_path.mkdir(parents=True, exist_ok=True)

    # Plot 1: Solve time vs mesh size
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"PCG": "#2196F3", "PBiCGSTAB": "#FF9800", "GAMG": "#4CAF50"}

    for solver, d in solvers.items():
        color = colors.get(solver, "#9E9E9E")
        ax.loglog(d["n_cells"], d["time"], "o-", label=solver, color=color, linewidth=2, markersize=8)

    ax.set_xlabel("Number of Cells", fontsize=12)
    ax.set_ylabel("Solve Time (seconds)", fontsize=12)
    ax.set_title("Linear Solver Scaling: Time vs Mesh Size", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    plot_file = output_path / "solve_time_vs_mesh.png"
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_file}")

    # Plot 2: Iterations vs mesh size
    fig, ax = plt.subplots(figsize=(10, 6))

    for solver, d in solvers.items():
        color = colors.get(solver, "#9E9E9E")
        ax.semilogx(d["n_cells"], d["iterations"], "s-", label=solver, color=color, linewidth=2, markersize=8)

    ax.set_xlabel("Number of Cells", fontsize=12)
    ax.set_ylabel("Iterations to Converge", fontsize=12)
    ax.set_title("Solver Iterations vs Mesh Size", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    plot_file = output_path / "iterations_vs_mesh.png"
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_file}")


# ---------------------------------------------------------------------------
# Plot: GPU vs CPU speedup
# ---------------------------------------------------------------------------


def plot_gpu_cpu_speedup(
    results_dir: str | Path = "benchmarks/results",
    output_dir: str | Path | None = None,
) -> None:
    """Plot GPU vs CPU speedup bar chart.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing ``gpu_cpu_comparison.csv``.
    output_dir : str or Path, optional
        Directory for output plot.
    """
    results_path = Path(results_dir)
    csv_file = results_path / "gpu_cpu_comparison.csv"

    if not csv_file.exists():
        print(f"Warning: {csv_file} not found, skipping plot.")
        return

    data = _load_csv(csv_file)

    # Check if GPU data exists
    has_gpu = any(_to_float(row.get("speedup")) is not None for row in data)

    if not has_gpu:
        print("No GPU data available, skipping GPU speedup plot.")
        return

    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot generation.")
        return

    output_path = Path(output_dir) if output_dir else results_path
    output_path.mkdir(parents=True, exist_ok=True)

    # Group by solver
    solvers: dict[str, dict[str, list]] = {}
    for row in data:
        solver = row["solver"]
        speedup = _to_float(row.get("speedup"))
        if speedup is None:
            continue
        if solver not in solvers:
            solvers[solver] = {"n_cells": [], "speedup": []}
        solvers[solver]["n_cells"].append(int(row["n_cells"]))
        solvers[solver]["speedup"].append(speedup)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"PCG": "#2196F3", "PBiCGSTAB": "#FF9800", "GAMG": "#4CAF50"}

    for solver, d in solvers.items():
        color = colors.get(solver, "#9E9E9E")
        ax.semilogx(d["n_cells"], d["speedup"], "o-", label=solver, color=color, linewidth=2, markersize=8)

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="No speedup (1x)")
    ax.set_xlabel("Number of Cells", fontsize=12)
    ax.set_ylabel("Speedup (CPU time / GPU time)", fontsize=12)
    ax.set_title("GPU vs CPU Speedup", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    plot_file = output_path / "gpu_cpu_speedup.png"
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_file}")


# ---------------------------------------------------------------------------
# Plot: Memory scaling
# ---------------------------------------------------------------------------


def plot_memory_scaling(
    results_dir: str | Path = "benchmarks/results",
    output_dir: str | Path | None = None,
) -> None:
    """Plot memory usage vs mesh size.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing memory CSV files.
    output_dir : str or Path, optional
        Directory for output plot.
    """
    results_path = Path(results_dir)
    solver_csv = results_path / "memory_solver.csv"

    if not solver_csv.exists():
        print(f"Warning: {solver_csv} not found, skipping memory plot.")
        return

    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot generation.")
        return

    output_path = Path(output_dir) if output_dir else results_path
    output_path.mkdir(parents=True, exist_ok=True)

    data = _load_csv(solver_csv)

    # Group by solver
    solvers: dict[str, dict[str, list]] = {}
    for row in data:
        solver = row["solver"]
        if solver not in solvers:
            solvers[solver] = {"n_cells": [], "memory_mb": []}
        solvers[solver]["n_cells"].append(int(row["n_cells"]))
        solvers[solver]["memory_mb"].append(float(row["peak_memory_mb"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"PCG": "#2196F3", "PBiCGSTAB": "#FF9800", "GAMG": "#4CAF50"}

    for solver, d in solvers.items():
        color = colors.get(solver, "#9E9E9E")
        ax.loglog(d["n_cells"], d["memory_mb"], "o-", label=solver, color=color, linewidth=2, markersize=8)

    ax.set_xlabel("Number of Cells", fontsize=12)
    ax.set_ylabel("Peak Memory (MB)", fontsize=12)
    ax.set_title("Memory Usage vs Mesh Size", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    plot_file = output_path / "memory_scaling.png"
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_file}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary_table(
    results_dir: str | Path = "benchmarks/results",
) -> None:
    """Print a formatted summary table of all benchmark results.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing benchmark CSV files.
    """
    results_path = Path(results_dir)

    # Linear solve benchmark
    csv_file = results_path / "linear_solve_benchmark.csv"
    if csv_file.exists():
        data = _load_csv(csv_file)
        print(f"\n{'='*70}")
        print("LINEAR SOLVE BENCHMARK SUMMARY")
        print(f"{'='*70}")
        print(f"{'Solver':<12} {'Cells':>10} {'Time (s)':>10} {'Iters':>8} {'Residual':>12} {'Conv':>5}")
        print(f"{'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*12} {'-'*5}")
        for row in data:
            converged = "[OK]" if row.get("converged") == "True" else "[FAIL]"
            print(
                f"{row['solver']:<12} "
                f"{int(row['n_cells']):>10,} "
                f"{float(row['time_seconds']):>10.4f} "
                f"{int(row['iterations']):>8} "
                f"{float(row['residual']):>12.2e} "
                f"{converged:>5}"
            )

    # GPU comparison
    csv_file = results_path / "gpu_cpu_comparison.csv"
    if csv_file.exists():
        data = _load_csv(csv_file)
        has_gpu = any(_to_float(row.get("speedup")) is not None for row in data)
        if has_gpu:
            print(f"\n{'='*70}")
            print("GPU vs CPU COMPARISON")
            print(f"{'='*70}")
            print(f"{'Solver':<12} {'Cells':>10} {'CPU (s)':>10} {'GPU (s)':>10} {'Speedup':>8}")
            print(f"{'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
            for row in data:
                speedup = _to_float(row.get("speedup"))
                if speedup is not None:
                    print(
                        f"{row['solver']:<12} "
                        f"{int(row['n_cells']):>10,} "
                        f"{float(row['cpu_time']):>10.4f} "
                        f"{float(row['gpu_time']):>10.4f} "
                        f"{speedup:>8.2f}x"
                    )


def generate_all_plots(
    results_dir: str | Path = "benchmarks/results",
    output_dir: str | Path | None = None,
) -> None:
    """Generate all benchmark plots.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing benchmark CSV files.
    output_dir : str or Path, optional
        Directory for output plots. Defaults to results_dir.
    """
    if not HAS_MATPLOTLIB:
        print("\nmatplotlib not installed. Skipping plot generation.")
        print("Install with: pip install matplotlib")
        return

    print(f"\nGenerating plots from {results_dir}...")
    plot_solve_time_vs_mesh(results_dir, output_dir)
    plot_gpu_cpu_speedup(results_dir, output_dir)
    plot_memory_scaling(results_dir, output_dir)
    print_summary_table(results_dir)
    print("Done.")


if __name__ == "__main__":
    generate_all_plots()
