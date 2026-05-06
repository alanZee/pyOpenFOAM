"""
Field comparison utilities for validation.

Provides :class:`FieldComparator` for comparing computed fields against
reference solutions (analytical or benchmark).  Supports:

- Scalar and vector field comparison
- Cell-centred and face-centred fields
- Per-component error analysis for vector fields
- Pass/fail assessment against configurable tolerances

Usage::

    from validation.comparator import FieldComparator

    comparator = FieldComparator(l2_tol=1e-3, max_tol=1e-2)
    result = comparator.compare(computed_U, reference_U, name="velocity")
    result.print_report()
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from validation.metrics import compute_all_metrics

__all__ = ["FieldComparator", "ComparisonResult"]


@dataclass
class ComparisonResult:
    """Result of a field comparison.

    Attributes
    ----------
    name : str
        Name of the compared field (e.g. ``"velocity"``, ``"pressure"``).
    metrics : dict
        Metric values from :func:`validation.metrics.compute_all_metrics`.
    component_metrics : dict[str, dict]
        Per-component metrics for vector fields (e.g. ``{"x": {...}, "y": {...}}``).
    passed : bool
        Whether the comparison passed the tolerance thresholds.
    l2_tol : float
        L2 relative error tolerance used.
    max_tol : float
        Maximum absolute error tolerance used.
    details : str
        Human-readable summary of the comparison.
    """

    name: str = ""
    metrics: dict = field(default_factory=dict)
    component_metrics: dict = field(default_factory=dict)
    passed: bool = False
    l2_tol: float = 1e-3
    max_tol: float = 1e-2
    details: str = ""

    def print_report(self) -> None:
        """Print a human-readable comparison report."""
        status = "PASS" if self.passed else "FAIL"
        print(f"\n{'='*60}")
        print(f"  {self.name}: {status}")
        print(f"{'='*60}")
        print(f"  L2 relative error:  {self.metrics.get('l2_relative_error', 'N/A'):.6e}  "
              f"(tol: {self.l2_tol:.6e})")
        print(f"  Max absolute error: {self.metrics.get('max_absolute_error', 'N/A'):.6e}  "
              f"(tol: {self.max_tol:.6e})")
        print(f"  Max relative error: {self.metrics.get('max_relative_error', 'N/A'):.6e}")
        print(f"  RMS error:          {self.metrics.get('rms_error', 'N/A'):.6e}")
        print(f"  L2 norm of error:   {self.metrics.get('l2_norm', 'N/A'):.6e}")

        if self.component_metrics:
            print(f"\n  Per-component errors:")
            for comp, comp_metrics in self.component_metrics.items():
                print(f"    {comp}: L2_rel={comp_metrics['l2_relative_error']:.6e}, "
                      f"Max_abs={comp_metrics['max_absolute_error']:.6e}")

        if self.details:
            print(f"\n  {self.details}")
        print(f"{'='*60}\n")


class FieldComparator:
    """Compare computed fields against reference solutions.

    Parameters
    ----------
    l2_tol : float
        L2 relative error tolerance for pass/fail (default 1e-3).
    max_tol : float
        Maximum absolute error tolerance for pass/fail (default 1e-2).
    """

    def __init__(
        self,
        l2_tol: float = 1e-3,
        max_tol: float = 1e-2,
    ) -> None:
        self.l2_tol = l2_tol
        self.max_tol = max_tol

    def compare(
        self,
        computed: torch.Tensor,
        reference: torch.Tensor,
        *,
        name: str = "field",
    ) -> ComparisonResult:
        """Compare a computed field against a reference.

        Parameters
        ----------
        computed : torch.Tensor
            Computed field values.  Can be 1-D ``(n,)`` for scalars
            or 2-D ``(n, 3)`` for vectors.
        reference : torch.Tensor
            Reference field values (same shape as *computed*).
        name : str
            Name for reporting.

        Returns
        -------
        ComparisonResult
            Comparison result with metrics and pass/fail status.
        """
        # Flatten if needed
        comp_flat = computed.reshape(-1)
        ref_flat = reference.reshape(-1)

        # Overall metrics
        metrics = compute_all_metrics(comp_flat, ref_flat)

        # Per-component metrics for vector fields
        component_metrics: dict[str, dict] = {}
        if computed.dim() >= 2 and computed.shape[-1] == 3:
            labels = ["x", "y", "z"]
            for i, label in enumerate(labels):
                comp_i = computed[..., i].reshape(-1)
                ref_i = reference[..., i].reshape(-1)
                component_metrics[label] = compute_all_metrics(comp_i, ref_i)

        # Pass/fail assessment
        l2_pass = metrics["l2_relative_error"] < self.l2_tol
        max_pass = metrics["max_absolute_error"] < self.max_tol
        passed = l2_pass and max_pass

        details_parts = []
        if not l2_pass:
            details_parts.append(
                f"L2 relative error {metrics['l2_relative_error']:.6e} "
                f"exceeds tolerance {self.l2_tol:.6e}"
            )
        if not max_pass:
            details_parts.append(
                f"Max absolute error {metrics['max_absolute_error']:.6e} "
                f"exceeds tolerance {self.max_tol:.6e}"
            )
        if passed:
            details_parts.append("All metrics within tolerance")

        return ComparisonResult(
            name=name,
            metrics=metrics,
            component_metrics=component_metrics,
            passed=passed,
            l2_tol=self.l2_tol,
            max_tol=self.max_tol,
            details="; ".join(details_parts),
        )
