# GPU Acceleration Guide

pyOpenFOAM uses PyTorch as its tensor backend, enabling transparent GPU acceleration for all CFD operations. This guide covers device management, performance optimization, and best practices for GPU-accelerated CFD simulations.

## Device Detection

pyOpenFOAM automatically detects available hardware at startup:

```python
from pyfoam.core import DeviceManager

dm = DeviceManager()
print(dm.capabilities)
# DeviceCapabilities(cpu=True, cuda=True, mps=False, cuda_devices=1)

print(dm.capabilities.available_devices)
# ['cpu', 'cuda']

print(dm.device)
# device('cuda') — auto-selected best available
```

### Supported Backends

| Backend | Device String | Notes |
|---------|--------------|-------|
| CPU | `"cpu"` | Always available, float64 default |
| CUDA | `"cuda"` | NVIDIA GPUs, requires CUDA toolkit |
| CUDA (specific) | `"cuda:0"`, `"cuda:1"` | Multi-GPU selection |
| MPS | `"mps"` | Apple Silicon (M1/M2/M3) |

### Priority Order

The auto-selection order is: **CUDA > MPS > CPU**.

```python
# Manual device selection
dm.device = "cuda"    # Use NVIDIA GPU
dm.device = "mps"     # Use Apple Silicon
dm.device = "cpu"     # Force CPU
```

## Tensor Configuration

### Default Precision

CFD requires **float64** (double precision) for convergence. float32 causes divergence in pressure-velocity coupling algorithms due to accumulated rounding errors.

```python
from pyfoam.core import TensorConfig, get_default_dtype
import torch

print(get_default_dtype())  # torch.float64

config = TensorConfig()
print(config.dtype)    # torch.float64
print(config.device)   # device('cuda') if available
```

### Temporary Overrides

Use context managers for temporary device/dtype changes:

```python
from pyfoam.core import device_context, TensorConfig
import torch

config = TensorConfig()

# Temporary float32 for performance testing
with config.override(dtype=torch.float32):
    t = config.zeros(1000)  # float32 tensor
    print(t.dtype)  # torch.float32

# Back to float64
print(config.dtype)  # torch.float64

# Module-level context manager
with device_context(device='cpu', dtype=torch.float32):
    # All operations use CPU + float32 here
    pass
```

## GPU Memory Management

### PyTorch Memory Behavior

PyTorch uses a caching allocator for GPU memory. Key behaviors:

1. **Memory is not released** — PyTorch caches freed memory for reuse.
2. **Peak memory matters** — The allocator tracks peak usage, not current usage.
3. **`torch.cuda.empty_cache()`** — Releases unused cached memory (rarely needed).

### Monitoring GPU Memory

```python
import torch

# Check allocated memory
print(torch.cuda.memory_allocated())  # bytes
print(torch.cuda.memory_reserved())   # bytes (cached)

# Check max allocated
print(torch.cuda.max_memory_allocated())

# Reset peak stats
torch.cuda.reset_peak_memory_stats()
```

### Memory-Efficient Patterns

```python
# BAD: Creates intermediate tensors
result = a + b + c + d

# BETTER: In-place operations reduce peak memory
result = a.clone()
result += b
result += c
result += d

# BAD: Keeps old tensors in scope
for i in range(1000):
    temp = expensive_computation(i)
    results.append(temp)

# BETTER: Free intermediates
for i in range(1000):
    temp = expensive_computation(i)
    results.append(temp.clone())
    del temp
```

## Performance Optimization

### 1. Pre-compute Geometry

Geometric quantities are computed lazily. Pre-compute them before the simulation loop:

```python
mesh.compute_geometry()  # Computes all geometric quantities at once
# Now mesh.cell_volumes, mesh.face_areas, etc. are cached
```

### 2. Batch Operations

PyTorch operations are fastest when batched. Avoid Python loops over cells/faces:

```python
# BAD: Python loop over cells
for i in range(n_cells):
    result[i] = a[i] + b[i]

# GOOD: Vectorized tensor operation
result = a + b  # Single GPU kernel launch

# BAD: Python loop over faces
for f in range(n_internal_faces):
    flux[f] = phi[f] * (u[owner[f]] - u[neighbour[f]])

# GOOD: Gather + vectorized
u_owner = u[owner[:n_internal]]
u_neigh = u[neighbour]
flux = phi[:n_internal] * (u_owner - u_neigh)
```

### 3. Use scatter_add and gather

The `scatter_add` and `gather` operations are the core primitives for FVM assembly:

```python
from pyfoam.core import scatter_add, gather

# Gather: collect values by index (boundary lookup, neighbour access)
x_owner = gather(x, owner)     # x[owner[f]]
x_neigh = gather(x, neighbour) # x[neighbour[f]]

# Scatter-add: accumulate flux contributions into cells
y = scatter_add(flux, owner, n_cells)  # y[owner[f]] += flux[f]
```

### 4. Avoid Unnecessary Device Transfers

Moving tensors between CPU and GPU is expensive:

```python
# BAD: Frequent device transfers
for i in range(n_cells):
    value = tensor[i].item()  # GPU → CPU (slow!)
    result = process(value)
    tensor[i] = result         # CPU → GPU (slow!)

# GOOD: Keep everything on GPU
result = process_gpu(tensor)   # Entire operation on GPU
```

### 5. Use CSR for Solving

Convert LDU matrices to CSR format before solving:

```python
# Assembly: COO is cheap for incremental insertion
coo = ldu_matrix.to_sparse_coo()

# Solving: CSR is faster for matrix-vector products
csr = ldu_matrix.to_sparse_csr()
```

## Multi-GPU (Future)

Multi-GPU support via MPI is planned. The architecture supports it through:

1. **Domain decomposition** — Split mesh across GPUs.
2. **Boundary exchange** — Ghost cell communication via MPI.
3. **Parallel assembly** — Each GPU assembles its local matrix.
4. **Global solve** — Distributed linear solver.

## Benchmarking

### Timing GPU Operations

```python
import torch
import time

# Synchronize before timing (GPU operations are asynchronous)
torch.cuda.synchronize()
start = time.perf_counter()

# Run computation
result = expensive_operation()

torch.cuda.synchronize()
elapsed = time.perf_counter() - start
print(f"Elapsed: {elapsed:.4f} seconds")
```

### Using PyTorch Profiler

```python
import torch

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    result = expensive_operation()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### Expected Speedups

| Operation | CPU (float64) | GPU (float64) | Speedup |
|-----------|--------------|---------------|---------|
| Matrix assembly (100k cells) | 50ms | 5ms | 10× |
| PCG solve (1000 iterations) | 2s | 200ms | 10× |
| GAMG solve (100 V-cycles) | 5s | 500ms | 10× |
| Gradient computation | 20ms | 2ms | 10× |

*Actual speedups depend on mesh size, GPU model, and problem structure.*

## Common Issues

### "CUDA out of memory"

```python
# Reduce mesh size or use CPU
dm = DeviceManager()
dm.device = 'cpu'

# Or free GPU memory
import torch
torch.cuda.empty_cache()
```

### "Device 'cuda' is not available"

```python
# Check CUDA installation
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.device_count())  # Number of GPUs

# Install CUDA-enabled PyTorch
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Slow GPU Performance

1. **Check tensor sizes** — Very small tensors (< 1000 elements) may be faster on CPU.
2. **Avoid synchronization** — `tensor.item()`, `tensor.cpu()`, `tensor.numpy()` force synchronization.
3. **Use float64** — float32 is faster but causes CFD divergence.
4. **Profile first** — Use `torch.profiler` to find bottlenecks.

### MPS (Apple Silicon) Limitations

MPS backend has some limitations:

- Not all PyTorch operations are supported on MPS.
- float64 support is limited on some operations.
- Performance may not match CUDA for large meshes.

```python
# Check MPS availability
import torch
print(torch.backends.mps.is_available())  # True on Apple Silicon

# Force CPU if MPS has issues
dm.device = 'cpu'
```
