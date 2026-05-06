"""
Validation case runner.

Provides :class:`ValidationRunner` which orchestrates the execution of
validation cases, collects results, and generates summary reports.

Each validation case is a class implementing :class:`ValidationCaseBase`
with methods for setup, execution, and reference solution generation.

Usage::

    from validation.runner import ValidationRunner
    from validation.cases.couette_flow import CouetteFlowCase

    runner = ValidationRunner(output_dir="validation/results")
    case = CouetteFlowCase(n_cells=32, Re=10.0)
    result = runner.run(case)
    result.print_summary()
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from validation.comparator import FieldComparator, ComparisonResult

__all__ = ["ValidationRunner", "ValidationResult", "ValidationCaseBase"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation case base class
# ---------------------------------------------------------------------------


class ValidationCaseBase(ABC):
    """Abstract base class for validation cases.

    Subclasses must implement:
    - :meth:`setup` — prepare mesh, fields, and solver
    - :meth:`run` — execute the solver
    - :meth:`get_reference` — compute the reference (analytical) solution
    - :meth:`get_computed` — extract the computed solution for comparison
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable case name."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description of the case."""

    @abstractmethod
    def setup(self) -> None:
        """Prepare the case: mesh, fields, solver configuration."""

    @abstractmethod
    def run(self) -> dict[str, Any]:
        """Execute the solver and return raw results."""

    @abstractmethod
    def get_reference(self) -> dict[str, torch.Tensor]:
        """Compute the reference (analytical) solution.

        Returns
        -------
        dict
            Field name -> reference tensor (e.g. ``{"U": ..., "p": ...}``).
        """

    @abstractmethod
    def get_computed(self) -> dict[str, torch.Tensor]:
        """Extract the computed solution for comparison.

        Returns
        -------
        dict
            Field name -> computed tensor.
        """

    def get_tolerances(self) -> dict[str, float]:
        """Return tolerance overrides for this case.

        Returns
        -------
        dict
            Keys ``l2_tol`` and ``max_tol`` (optional).
        """
        return {}


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of running a single validation case.

    Attributes
    ----------
    case_name : str
        Name of the validation case.
    case_description : str
        Description of the case.
    field_results : dict[str, ComparisonResult]
        Per-field comparison results.
    overall_passed : bool
        True if all fields passed their tolerance checks.
    run_time : float
        Wall-clock time for the solver run (seconds).
    solver_info : dict
        Additional solver information (iterations, residuals, etc.).
    parameters : dict
        Case parameters (mesh size, Reynolds number, etc.).
    """

    case_name: str = ""
    case_description: str = ""
    field_results: dict[str, ComparisonResult] = field(default_factory=dict)
    overall_passed: bool = False
    run_time: float = 0.0
    solver_info: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)

    def print_summary(self) -> None:
        """Print a human-readable summary of the validation result."""
        status = "PASS" if self.overall_passed else "FAIL"
        print(f"\n{'#'*60}")
        print(f"  Validation: {self.case_name} — {status}")
        print(f"{'#'*60}")
        print(f"  Description: {self.case_description}")
        print(f"  Run time:    {self.run_time:.2f}s")

        if self.parameters:
            print(f"  Parameters:")
            for k, v in self.parameters.items():
                print(f"    {k}: {v}")

        if self.solver_info:
            print(f"  Solver info:")
            for k, v in self.solver_info.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.6e}")
                else:
                    print(f"    {k}: {v}")

        for field_name, result in self.field_results.items():
            result.print_report()

        print(f"  Overall: {'PASS' if self.overall_passed else 'FAIL'}")
        print(f"{'#'*60}\n")

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "case_name": self.case_name,
            "case_description": self.case_description,
            "overall_passed": self.overall_passed,
            "run_time": self.run_time,
            "parameters": self.parameters,
            "solver_info": self.solver_info,
            "field_results": {
                name: {
                    "passed": r.passed,
                    "l2_tol": r.l2_tol,
                    "max_tol": r.max_tol,
                    "metrics": r.metrics,
                    "component_metrics": r.component_metrics,
                }
                for name, r in self.field_results.items()
            },
        }


# ---------------------------------------------------------------------------
# Validation runner
# ---------------------------------------------------------------------------


class ValidationRunner:
    """Orchestrates validation case execution.

    Parameters
    ----------
    output_dir : str | Path
        Directory for saving results (JSON, plots).
    default_l2_tol : float
        Default L2 relative error tolerance.
    default_max_tol : float
        Default maximum absolute error tolerance.
    """

    def __init__(
        self,
        output_dir: str | Path = "validation/results",
        default_l2_tol: float = 1e-3,
        default_max_tol: float = 1e-2,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.default_l2_tol = default_l2_tol
        self.default_max_tol = default_max_tol
        self.results: list[ValidationResult] = []

    def run(self, case: ValidationCaseBase) -> ValidationResult:
        """Run a single validation case.

        Parameters
        ----------
        case : ValidationCaseBase
            The validation case to run.

        Returns
        -------
        ValidationResult
            Comparison results with metrics and pass/fail status.
        """
        logger.info("Running validation case: %s", case.name)

        # Setup
        case.setup()

        # Run solver
        t_start = time.perf_counter()
        solver_info = case.run()
        run_time = time.perf_counter() - t_start

        # Get fields
        computed_fields = case.get_computed()
        reference_fields = case.get_reference()

        # Tolerances
        tols = case.get_tolerances()
        l2_tol = tols.get("l2_tol", self.default_l2_tol)
        max_tol = tols.get("max_tol", self.default_max_tol)

        # Compare each field
        comparator = FieldComparator(l2_tol=l2_tol, max_tol=max_tol)
        field_results: dict[str, ComparisonResult] = {}
        all_passed = True

        for field_name in computed_fields:
            if field_name not in reference_fields:
                logger.warning("No reference for field '%s', skipping", field_name)
                continue

            comp = computed_fields[field_name]
            ref = reference_fields[field_name]

            # Ensure same shape
            if comp.shape != ref.shape:
                logger.warning(
                    "Shape mismatch for '%s': computed %s vs reference %s",
                    field_name, comp.shape, ref.shape,
                )
                continue

            result = comparator.compare(comp, ref, name=field_name)
            field_results[field_name] = result

            if not result.passed:
                all_passed = False

        # Collect parameters
        parameters = {}
        if hasattr(case, "n_cells"):
            parameters["n_cells"] = getattr(case, "n_cells")
        if hasattr(case, "Re"):
            parameters["Re"] = getattr(case, "Re")
        if hasattr(case, "n_cells_per_dim"):
            parameters["n_cells_per_dim"] = getattr(case, "n_cells_per_dim")

        validation_result = ValidationResult(
            case_name=case.name,
            case_description=case.description,
            field_results=field_results,
            overall_passed=all_passed,
            run_time=run_time,
            solver_info=solver_info if isinstance(solver_info, dict) else {},
            parameters=parameters,
        )

        self.results.append(validation_result)

        # Save to JSON
        self._save_result(validation_result)

        return validation_result

    def run_all(self, cases: list[ValidationCaseBase]) -> list[ValidationResult]:
        """Run multiple validation cases.

        Parameters
        ----------
        cases : list[ValidationCaseBase]
            Cases to run.

        Returns
        -------
        list[ValidationResult]
            Results for each case.
        """
        results = []
        for case in cases:
            try:
                result = self.run(case)
                results.append(result)
            except Exception as e:
                logger.error("Validation case '%s' failed: %s", case.name, e)
                results.append(ValidationResult(
                    case_name=case.name,
                    case_description=case.description,
                    overall_passed=False,
                    solver_info={"error": str(e)},
                ))
        return results

    def print_summary(self) -> None:
        """Print a summary of all validation results."""
        print(f"\n{'='*60}")
        print(f"  Validation Suite Summary")
        print(f"{'='*60}")

        total = len(self.results)
        passed = sum(1 for r in self.results if r.overall_passed)
        failed = total - passed

        print(f"  Total cases: {total}")
        print(f"  Passed:      {passed}")
        print(f"  Failed:      {failed}")
        print()

        for result in self.results:
            status = "PASS" if result.overall_passed else "FAIL"
            print(f"  [{status}] {result.case_name} ({result.run_time:.2f}s)")

        print(f"\n  Overall: {'ALL PASSED' if failed == 0 else 'SOME FAILED'}")
        print(f"{'='*60}\n")

    def _save_result(self, result: ValidationResult) -> None:
        """Save a validation result to JSON."""
        filename = result.case_name.lower().replace(" ", "_") + ".json"
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info("Saved result to %s", filepath)
