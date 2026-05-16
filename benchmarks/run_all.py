"""
Automated benchmark runner.

Runs all pyOpenFOAM benchmarks and generates summary plots.

Usage::

    # Run all benchmarks (default mesh sizes)
    python benchmarks/run_all.py

    # Run with custom mesh sizes
    python benchmarks/run_all.py --mesh-sizes 10 20 40

    # Run only specific benchmarks
    python benchmarks/run_all.py --skip-gpu
    python benchmarks/run_all.py --only linear

    # Force CPU-only
    python benchmarks/run_all.py --device cpu
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


def main() -> None:
    """Run the benchmark suite."""
    parser = argparse.ArgumentParser(
        description="pyOpenFOAM Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mesh-sizes",
        type=int,
        nargs="+",
        default=[10, 20, 40, 60],
        help="Cells per dimension (default: 10 20 40 60)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cpu' or 'cuda' (default: auto-detect)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
        help="Output directory for results (default: benchmarks/results)",
    )
    parser.add_argument(
        "--only",
        type=str,
        nargs="+",
        choices=["linear", "gpu", "memory", "sparse", "simple", "comparison", "plots"],
        default=None,
        help="Run only specified benchmarks",
    )
    parser.add_argument(
        "--skip",
        type=str,
        nargs="+",
        choices=["linear", "gpu", "memory", "sparse", "simple", "comparison", "plots"],
        default=None,
        help="Skip specified benchmarks",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs (default: 1)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of timed runs for median (default: 3)",
    )

    args = parser.parse_args()

    # Determine which benchmarks to run
    all_benchmarks = {"linear", "gpu", "memory", "sparse", "simple", "comparison", "plots"}

    if args.only:
        to_run = set(args.only)
    else:
        to_run = all_benchmarks

    if args.skip:
        to_run -= set(args.skip)

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print(f"\n{'='*60}")
    print(f"pyOpenFOAM Benchmark Suite")
    print(f"{'='*60}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    print(f"Mesh sizes: {args.mesh_sizes}")
    print(f"Cells range: {min(args.mesh_sizes)**3:,} — {max(args.mesh_sizes)**3:,}")
    print(f"Output directory: {output_dir}")
    print(f"Benchmarks to run: {', '.join(sorted(to_run))}")
    print(f"Warmup runs: {args.warmup}, Timed runs: {args.runs}")
    print(f"{'='*60}\n")

    t_start = time.perf_counter()

    # Run benchmarks
    if "linear" in to_run:
        from benchmarks.linear_solve_benchmark import run_linear_solve_benchmarks
        run_linear_solve_benchmarks(
            mesh_sizes=args.mesh_sizes,
            device=device,
            output_dir=output_dir,
        )

    if "gpu" in to_run:
        from benchmarks.gpu_cpu_comparison import run_gpu_cpu_comparison
        run_gpu_cpu_comparison(
            mesh_sizes=args.mesh_sizes,
            output_dir=output_dir,
        )

    if "memory" in to_run:
        from benchmarks.memory_scaling import run_memory_scaling_benchmarks
        run_memory_scaling_benchmarks(
            mesh_sizes=args.mesh_sizes,
            device=device,
            output_dir=output_dir,
        )

    if "sparse" in to_run:
        from benchmarks.sparse_ops_benchmark import run_sparse_ops_benchmarks
        run_sparse_ops_benchmarks(
            mesh_sizes=args.mesh_sizes,
            device=device,
            output_dir=output_dir,
        )

    if "simple" in to_run:
        from benchmarks.simple_iteration_benchmark import run_simple_iteration_benchmarks
        run_simple_iteration_benchmarks(
            mesh_sizes=args.mesh_sizes,
            device=device,
            output_dir=output_dir,
        )

    if "comparison" in to_run:
        from benchmarks.openfoam_comparison import generate_comparison_report
        generate_comparison_report(
            results_dir=output_dir,
            output_dir=output_dir,
        )

    if "plots" in to_run:
        from benchmarks.plot_results import generate_all_plots
        generate_all_plots(results_dir=output_dir, output_dir=output_dir)

    t_total = time.perf_counter() - t_start

    print(f"\n{'='*60}")
    print(f"Benchmark Suite Complete")
    print(f"{'='*60}")
    print(f"Total time: {t_total:.1f}s")
    print(f"Results in: {output_dir}")

    # List output files
    csv_files = list(output_dir.glob("*.csv"))
    png_files = list(output_dir.glob("*.png"))
    if csv_files:
        print(f"\nCSV files:")
        for f in sorted(csv_files):
            print(f"  {f}")
    if png_files:
        print(f"\nPlot files:")
        for f in sorted(png_files):
            print(f"  {f}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Ensure the project root is on the path
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    main()
