"""
Dtype utilities and precision constants for CFD operations.

float64 is critical for CFD convergence — float32 causes divergence in
iterative solvers due to accumulated rounding errors in pressure-velocity
coupling algorithms (SIMPLE, PISO).
"""

from __future__ import annotations

import torch
import numpy as np

__all__ = [
    # Precision constants
    "CFD_DTYPE",
    "CFD_REAL_DTYPE",
    "CFD_COMPLEX_DTYPE",
    "INDEX_DTYPE",
    # Dtype utilities
    "is_floating",
    "is_complex_dtype",
    "promote_dtype",
    "to_cfd_dtype",
    "dtype_to_numpy",
    "numpy_to_torch",
    "real_dtype",
    "complex_dtype",
    "assert_floating",
]

# ---------------------------------------------------------------------------
# Precision constants
# ---------------------------------------------------------------------------

#: Default floating-point dtype for CFD field values.  float64 is mandatory
#: for convergence in pressure-velocity coupling (SIMPLE/PISO).
CFD_DTYPE: torch.dtype = torch.float64

#: Alias — the real-valued CFD dtype is the same as CFD_DTYPE.
CFD_REAL_DTYPE: torch.dtype = torch.float64

#: Complex dtype matching CFD_REAL_DTYPE (for spectral / Fourier work).
CFD_COMPLEX_DTYPE: torch.dtype = torch.complex128

#: Integer dtype used for mesh indices, connectivity, and offsets.
INDEX_DTYPE: torch.dtype = torch.int64

# ---------------------------------------------------------------------------
# Dtype predicates
# ---------------------------------------------------------------------------


def is_floating(dtype: torch.dtype) -> bool:
    """Return True if *dtype* is a floating-point type (real or complex)."""
    return dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )


def is_complex_dtype(dtype: torch.dtype) -> bool:
    """Return True if *dtype* is a complex floating-point type."""
    return dtype in (torch.complex64, torch.complex128)


# ---------------------------------------------------------------------------
# Dtype promotion / conversion
# ---------------------------------------------------------------------------


def promote_dtype(*dtypes: torch.dtype) -> torch.dtype:
    """Return the widest dtype that can represent all *dtypes* losslessly.

    Follows the same rules as ``torch.result_type`` but accepts an arbitrary
    number of inputs.

    Raises:
        ValueError: If no dtypes are provided.
    """
    if not dtypes:
        raise ValueError("At least one dtype is required")
    result = dtypes[0]
    for d in dtypes[1:]:
        result = torch.result_type(torch.tensor(0, dtype=result), torch.tensor(0, dtype=d))
    return result


def to_cfd_dtype(tensor: torch.Tensor) -> torch.Tensor:
    """Cast *tensor* to the CFD default dtype (float64) if it is floating."""
    if is_floating(tensor.dtype) and tensor.dtype != CFD_DTYPE:
        return tensor.to(dtype=CFD_DTYPE)
    return tensor


def dtype_to_numpy(dtype: torch.dtype) -> np.dtype:
    """Convert a torch dtype to the equivalent NumPy dtype."""
    _TORCH_TO_NP: dict[torch.dtype, np.dtype] = {
        torch.float16: np.dtype("float16"),
        torch.bfloat16: np.dtype("float32"),  # numpy has no bfloat16
        torch.float32: np.dtype("float32"),
        torch.float64: np.dtype("float64"),
        torch.complex64: np.dtype("complex64"),
        torch.complex128: np.dtype("complex128"),
        torch.int8: np.dtype("int8"),
        torch.int16: np.dtype("int16"),
        torch.int32: np.dtype("int32"),
        torch.int64: np.dtype("int64"),
        torch.uint8: np.dtype("uint8"),
        torch.bool: np.dtype("bool"),
    }
    if dtype not in _TORCH_TO_NP:
        raise ValueError(f"Unsupported torch dtype for numpy conversion: {dtype}")
    return _TORCH_TO_NP[dtype]


def numpy_to_torch(dtype: np.dtype) -> torch.dtype:
    """Convert a NumPy dtype to the equivalent torch dtype."""
    _NP_TO_TORCH: dict[np.dtype, torch.dtype] = {
        np.dtype("float16"): torch.float16,
        np.dtype("float32"): torch.float32,
        np.dtype("float64"): torch.float64,
        np.dtype("complex64"): torch.complex64,
        np.dtype("complex128"): torch.complex128,
        np.dtype("int8"): torch.int8,
        np.dtype("int16"): torch.int16,
        np.dtype("int32"): torch.int32,
        np.dtype("int64"): torch.int64,
        np.dtype("uint8"): torch.uint8,
        np.dtype("bool"): torch.bool,
    }
    if dtype not in _NP_TO_TORCH:
        raise ValueError(f"Unsupported numpy dtype for torch conversion: {dtype}")
    return _NP_TO_TORCH[dtype]


def real_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return the real-valued counterpart of *dtype*.

    If *dtype* is already real, returns it unchanged.
    """
    _COMPLEX_TO_REAL: dict[torch.dtype, torch.dtype] = {
        torch.complex64: torch.float32,
        torch.complex128: torch.float64,
    }
    return _COMPLEX_TO_REAL.get(dtype, dtype)


def complex_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return the complex counterpart of *dtype*.

    If *dtype* is already complex, returns it unchanged.
    """
    _REAL_TO_COMPLEX: dict[torch.dtype, torch.dtype] = {
        torch.float32: torch.complex64,
        torch.float64: torch.complex128,
        torch.complex64: torch.complex64,
        torch.complex128: torch.complex128,
    }
    if dtype not in _REAL_TO_COMPLEX:
        raise ValueError(f"No complex counterpart for dtype: {dtype}")
    return _REAL_TO_COMPLEX[dtype]


def assert_floating(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Raise if *tensor* does not have a floating-point dtype.

    This is a guard for CFD field operations that require floating-point
    arithmetic (e.g., gradient computation, flux assembly).
    """
    if not is_floating(tensor.dtype):
        raise TypeError(
            f"{name} must have a floating-point dtype, got {tensor.dtype}"
        )
