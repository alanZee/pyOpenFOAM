"""
pyOpenFOAM Benchmark Suite.

Performance benchmarks comparing pyOpenFOAM linear solvers across
different mesh sizes and hardware (CPU/GPU).

Modules:
- mesh_generation: Programmatic structured hex mesh generation
- linear_solve_benchmark: Linear solver scaling vs mesh size
- gpu_cpu_comparison: GPU vs CPU speedup analysis
- memory_scaling: Memory usage profiling
- plot_results: Visualization of benchmark results
- run_all: Automated benchmark runner
"""

__version__ = "0.1.0"
