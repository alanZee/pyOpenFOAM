"""
Accuracy metrics for validation comparisons.

Provides standard error metrics used in CFD validation:
- L2 norm (Euclidean norm of error)
- L2 relative error (normalised by field magnitude)
- Maximum absolute error
- Maximum relative error
- RMS (root mean square) error

All functions accept PyTorch tensors and return Python floats.

Usage::

    import torch
    from validation.metrics import compute_all_metrics

    computed = torch.randn(100)
    reference = torch.randn(100)
    metrics = compute_all_metrics(computed, reference)
    print(metrics["l2_relative_error"])
"""

from __future__ import annotations

import torch

__all__ = [
    "l2_norm",
    "l2_relative_error",
    "max_absolute_error",
    "max_relative_error",
    "rms_error",
    "compute_all_metrics",
]


def l2_norm(error: torch.Tensor) -> float:
    """Compute the L2 (Euclidean) norm of the error vector.

    Parameters
    ----------
    error : torch.Tensor
        Error vector (computed − reference).

    Returns
    -------
    float
        ``||error||_2 = sqrt(sum(error_i^2))``
    """
    return float(torch.norm(error.float()).item())


def l2_relative_error(
    computed: torch.Tensor,
    reference: torch.Tensor,
    *,
    epsilon: float = 1e-30,
) -> float:
    """Compute the L2 relative error.

    ``||computed - reference||_2 / max(||reference||_2, epsilon)``

    Parameters
    ----------
    computed : torch.Tensor
        Computed field values.
    reference : torch.Tensor
        Reference (analytical or benchmark) values.
    epsilon : float
        Small number to avoid division by zero.

    Returns
    -------
    float
        L2 relative error.
    """
    error = computed.float() - reference.float()
    norm_error = torch.norm(error)
    norm_ref = torch.norm(reference.float())
    return float((norm_error / norm_ref.clamp(min=epsilon)).item())


def max_absolute_error(
    computed: torch.Tensor,
    reference: torch.Tensor,
) -> float:
    """Compute the maximum absolute error.

    ``max_i |computed_i - reference_i|``

    Parameters
    ----------
    computed : torch.Tensor
        Computed field values.
    reference : torch.Tensor
        Reference values.

    Returns
    -------
    float
        Maximum absolute error.
    """
    error = (computed.float() - reference.float()).abs()
    return float(error.max().item())


def max_relative_error(
    computed: torch.Tensor,
    reference: torch.Tensor,
    *,
    epsilon: float = 1e-30,
) -> float:
    """Compute the maximum relative error.

    ``max_i |computed_i - reference_i| / max(|reference_i|, epsilon)``

    Parameters
    ----------
    computed : torch.Tensor
        Computed field values.
    reference : torch.Tensor
        Reference values.
    epsilon : float
        Small number to avoid division by zero.

    Returns
    -------
    float
        Maximum relative error.
    """
    error = (computed.float() - reference.float()).abs()
    ref_safe = reference.float().abs().clamp(min=epsilon)
    relative = error / ref_safe
    return float(relative.max().item())


def rms_error(
    computed: torch.Tensor,
    reference: torch.Tensor,
) -> float:
    """Compute the root mean square (RMS) error.

    ``sqrt(mean((computed - reference)^2))``

    Parameters
    ----------
    computed : torch.Tensor
        Computed field values.
    reference : torch.Tensor
        Reference values.

    Returns
    -------
    float
        RMS error.
    """
    error = computed.float() - reference.float()
    return float(torch.sqrt(torch.mean(error ** 2)).item())


def compute_all_metrics(
    computed: torch.Tensor,
    reference: torch.Tensor,
    *,
    epsilon: float = 1e-30,
) -> dict[str, float]:
    """Compute all validation metrics at once.

    Parameters
    ----------
    computed : torch.Tensor
        Computed field values.
    reference : torch.Tensor
        Reference values.
    epsilon : float
        Small number to avoid division by zero.

    Returns
    -------
    dict
        Dictionary with keys:
        ``l2_norm``, ``l2_relative_error``, ``max_absolute_error``,
        ``max_relative_error``, ``rms_error``.
    """
    error = computed.float() - reference.float()
    return {
        "l2_norm": l2_norm(error),
        "l2_relative_error": l2_relative_error(computed, reference, epsilon=epsilon),
        "max_absolute_error": max_absolute_error(computed, reference),
        "max_relative_error": max_relative_error(computed, reference, epsilon=epsilon),
        "rms_error": rms_error(computed, reference),
    }
