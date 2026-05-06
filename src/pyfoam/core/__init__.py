"""
pyfoam.core — Core tensor backend, device management, and dtype utilities.

This is the foundational layer every other module builds on.
"""

from pyfoam.core.device import (
    DeviceManager,
    TensorConfig,
    device_context,
    get_device,
    get_default_dtype,
)
from pyfoam.core.dtype import (
    CFD_COMPLEX_DTYPE,
    CFD_DTYPE,
    CFD_REAL_DTYPE,
    INDEX_DTYPE,
    assert_floating,
    complex_dtype,
    dtype_to_numpy,
    is_complex_dtype,
    is_floating,
    numpy_to_torch,
    promote_dtype,
    real_dtype,
    to_cfd_dtype,
)
from pyfoam.core.backend import (
    Backend,
    gather,
    scatter_add,
    sparse_coo_tensor,
    sparse_mm,
)
from pyfoam.core.ldu_matrix import LduMatrix
from pyfoam.core.fv_matrix import FvMatrix, LinearSolver
from pyfoam.core.sparse_ops import (
    extract_diagonal,
    ldu_to_coo_indices,
    csr_matvec,
)

__all__ = [
    # device
    "DeviceManager",
    "TensorConfig",
    "device_context",
    "get_device",
    "get_default_dtype",
    # dtype
    "CFD_DTYPE",
    "CFD_REAL_DTYPE",
    "CFD_COMPLEX_DTYPE",
    "INDEX_DTYPE",
    "is_floating",
    "is_complex_dtype",
    "promote_dtype",
    "to_cfd_dtype",
    "dtype_to_numpy",
    "numpy_to_torch",
    "real_dtype",
    "complex_dtype",
    "assert_floating",
    # backend
    "Backend",
    "scatter_add",
    "gather",
    "sparse_coo_tensor",
    "sparse_mm",
    # ldu_matrix
    "LduMatrix",
    # fv_matrix
    "FvMatrix",
    "LinearSolver",
    # sparse_ops
    "extract_diagonal",
    "ldu_to_coo_indices",
    "csr_matvec",
]
