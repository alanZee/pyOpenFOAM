"""
Run all pyOpenFOAM validation cases.

Executes the complete validation suite:
1. Couette flow (analytical: linear velocity profile)
2. Poiseuille flow (analytical: parabolic velocity profile)
3. Lid-driven cavity (benchmark: Ghia et al. 1982)

Results are saved to ``validation/results/`` as JSON files.

Usage::

    # Run all cases with defaults
    python validation/run_all.py

    # Run with custom mesh size
    python validation/run_all.py --mesh-size 64

    # Run specific cases only
    python validation/run_all.py --only couette poiseuille

    # Verbose output
    python validation/run_all.py -v
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main() -> None:
    """Run the validation suite."""
    parser = argparse.ArgumentParser(
        description="pyOpenFOAM Validation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Validation cases:
  couette     Plane Couette flow (linear velocity profile)
  poiseuille  Plane Poiseuille flow (parabolic velocity profile)
  cavity      Lid-driven cavity (Ghia et al. benchmark)
        """,
    )
    parser.add_argument(
        "--mesh-size",
        type=int,
        default=32,
        help="Cells per dimension (default: 32)",
    )
    parser.add_argument(
        "--only",
        type=str,
        nargs="+",
        choices=["couette", "poiseuille", "cavity"],
        default=None,
        help="Run only specified cases",
    )
    parser.add_argument(
        "--skip",
        type=str,
        nargs="+",
        choices=["couette", "poiseuille", "cavity"],
        default=None,
        help="Skip specified cases",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="validation/results",
        help="Output directory for results (default: validation/results)",
    )
    parser.add_argument(
        "--reynolds",
        type=float,
        default=10.0,
        help="Reynolds number for Couette/Poiseuille (default: 10.0)",
    )
    parser.add_argument(
        "--cavity-re",
        type=float,
        default=100.0,
        help="Reynolds number for cavity (default: 100.0)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5000,
        help="Maximum solver iterations (default: 5000)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Configure logging
    import logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Determine which cases to run
    all_cases = {"couette", "poiseuille", "cavity"}
    if args.only:
        to_run = set(args.only)
    else:
        to_run = all_cases
    if args.skip:
        to_run -= set(args.skip)

    # Print configuration
    print(f"\n{'='*60}")
    print(f"pyOpenFOAM Validation Suite")
    print(f"{'='*60}")
    print(f"Mesh size:      {args.mesh_size}x{args.mesh_size}")
    print(f"Reynolds number: {args.reynolds} (Couette/Poiseuille), "
          f"{args.cavity_re} (Cavity)")
    print(f"Max iterations: {args.max_iter}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Cases to run:   {', '.join(sorted(to_run))}")
    print(f"{'='*60}\n")

    # Import validation modules
    from validation.runner import ValidationRunner
    from validation.cases.couette_flow import CouetteFlowCase
    from validation.cases.poiseuille_flow import PoiseuilleFlowCase
    from validation.cases.lid_driven_cavity import LidDrivenCavityCase

    # Create runner
    runner = ValidationRunner(output_dir=args.output_dir)

    # Build case list
    cases = []

    if "couette" in to_run:
        cases.append(CouetteFlowCase(
            n_cells=args.mesh_size,
            Re=args.reynolds,
            U_top=1.0,
            max_iterations=args.max_iter,
        ))

    if "poiseuille" in to_run:
        cases.append(PoiseuilleFlowCase(
            n_cells=args.mesh_size,
            Re=args.reynolds,
            max_iterations=args.max_iter,
        ))

    if "cavity" in to_run:
        cases.append(LidDrivenCavityCase(
            n_cells=args.mesh_size,
            Re=args.cavity_re,
            U_lid=1.0,
            max_iterations=args.max_iter,
        ))

    # Run all cases
    t_start = time.perf_counter()
    results = runner.run_all(cases)
    t_total = time.perf_counter() - t_start

    # Print summary
    runner.print_summary()

    # Save overall summary
    summary = {
        "total_time": t_total,
        "cases": [r.to_dict() for r in results],
        "all_passed": all(r.overall_passed for r in results),
    }

    summary_path = Path(args.output_dir) / "validation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to: {args.output_dir}")
    print(f"Summary: {summary_path}")
    print(f"Total time: {t_total:.1f}s")
    print(f"{'='*60}\n")

    # Exit with appropriate code
    all_passed = all(r.overall_passed for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
