"""
pyOpenFOAM Validation Framework.

Provides tools for validating pyOpenFOAM solver results against analytical
solutions and established benchmark data.  The framework consists of:

- :mod:`validation.metrics` — accuracy metrics (L2 norm, max error, etc.)
- :mod:`validation.comparator` — field comparison utilities
- :mod:`validation.runner` — validation case runner
- :mod:`validation.cases` — analytical and benchmark test cases

Usage::

    from validation.runner import ValidationRunner
    from validation.cases.couette_flow import CouetteFlowCase

    case = CouetteFlowCase(n_cells=32)
    runner = ValidationRunner()
    result = runner.run(case)
    result.print_summary()
"""

__version__ = "0.1.0"

from validation.metrics import (
    l2_norm,
    l2_relative_error,
    max_absolute_error,
    max_relative_error,
    rms_error,
    compute_all_metrics,
)

from validation.comparator import FieldComparator, ComparisonResult

from validation.runner import ValidationRunner, ValidationResult

__all__ = [
    "l2_norm",
    "l2_relative_error",
    "max_absolute_error",
    "max_relative_error",
    "rms_error",
    "compute_all_metrics",
    "FieldComparator",
    "ComparisonResult",
    "ValidationRunner",
    "ValidationResult",
]
