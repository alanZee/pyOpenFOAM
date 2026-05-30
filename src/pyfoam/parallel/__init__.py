"""
pyfoam.parallel — Domain decomposition and parallel solving via MPI.

Provides:

- :class:`Decomposition` — split a mesh into subdomains (geometric or Scotch)
- :class:`DecompositionStrategy` — RTS-enabled decomposition strategy base
- :class:`SimpleDecomposition` — simple geometric decomposition
- :class:`ScotchDecomposition` — Scotch graph-based decomposition
- :class:`SubDomain` — a subregion of the global mesh with ghost cell mapping
- :class:`ProcessorPatch` — describes ghost cells on a processor boundary
- :class:`HaloExchange` — inter-processor ghost cell communication
- :class:`ParallelField` — parallel-aware field with gather/scatter/reduce
- :class:`ParallelSolver` — domain-decomposed solver wrapper
- :class:`ParallelWriter` / :class:`ParallelReader` — processor directory I/O

All operations respect the global device/dtype from :mod:`pyfoam.core`.
MPI is optional — all operations have serial fallbacks for testing.
"""

from pyfoam.parallel.decomposition import Decomposition, SubDomain
from pyfoam.parallel.decomposition_2 import (
    DecompositionStrategy,
    SimpleDecomposition,
    ScotchDecomposition,
)
from pyfoam.parallel.processor_patch import ProcessorPatch, HaloExchange
from pyfoam.parallel.parallel_field import ParallelField
from pyfoam.parallel.parallel_io import ParallelReader, ParallelWriter
from pyfoam.parallel.parallel_solver import ParallelSolver, ParallelSolverConfig
from pyfoam.parallel.reconstruct_par import ReconstructPar, ReconstructResult
from pyfoam.parallel.redistribute_par import RedistributePar, RedistributeResult
from pyfoam.parallel.reconstruct_par_enhanced import (
    ReconstructParEnhanced,
    ZoneInfo,
    EnhancedReconstructResult,
)
from pyfoam.parallel.redistribute_par_enhanced import (
    RedistributeParEnhanced,
    BalancingStrategy,
    EnhancedRedistributeResult,
    PartitionDiagnostics,
)
from pyfoam.parallel.processor_patch_enhanced import (
    NonConformalPatch,
    EnhancedHaloExchange,
)
from pyfoam.parallel.reconstruct_par_enhanced_2 import (
    ReconstructParEnhanced2,
    V2ReconstructResult,
    FieldMergeStats,
    MergeStrategy,
)
from pyfoam.parallel.redistribute_par_enhanced_2 import (
    RedistributeParEnhanced2,
    GraphPartitionStrategy,
    V2RedistributeResult,
)
from pyfoam.parallel.processor_patch_enhanced_2 import (
    NonConformalPatch2,
    EnhancedHaloExchange2,
    FaceCentreInterpolator,
)
from pyfoam.parallel.reconstruct_par_enhanced_3 import (
    ReconstructParEnhanced3,
    V3ReconstructResult,
    ZoneMergeResult,
)
from pyfoam.parallel.redistribute_par_enhanced_3 import (
    RedistributeParEnhanced3,
    SpatialDecompositionStrategy,
    V3RedistributeResult,
)
from pyfoam.parallel.processor_patch_enhanced_3 import (
    NonConformalPatch3,
    EnhancedHaloExchange3,
    AdaptiveInterpolator,
)
from pyfoam.parallel.reconstruct_par_enhanced_4 import (
    ReconstructParEnhanced4,
    V4ReconstructResult,
    GradientMergeConfig,
)
from pyfoam.parallel.redistribute_par_enhanced_4 import (
    RedistributeParEnhanced4,
    V4RedistributeResult,
    PartitionQualityMetrics,
)
from pyfoam.parallel.processor_patch_enhanced_4 import (
    NonConformalPatch4,
    EnhancedHaloExchange4,
    HigherOrderInterpolator,
)
from pyfoam.parallel.reconstruct_par_enhanced_5 import (
    ReconstructParEnhanced5,
    V5ReconstructResult,
    SmoothingConfig,
)
from pyfoam.parallel.redistribute_par_enhanced_5 import (
    RedistributeParEnhanced5,
    V5RedistributeResult,
    MigrationPlan,
    CostEstimator,
)
from pyfoam.parallel.processor_patch_enhanced_5 import (
    CoarsenablePatch5,
    EnhancedHaloExchange5,
    CompressionStats,
)
from pyfoam.parallel.reconstruct_par_enhanced_6 import (
    ReconstructParEnhanced6,
    V6ReconstructResult,
    AnisotropicSmoothingConfig,
    CheckpointInfo,
)
from pyfoam.parallel.redistribute_par_enhanced_6 import (
    RedistributeParEnhanced6,
    V6RedistributeResult,
    HierarchicalPartitionConfig,
    PartitionMetrics,
)
from pyfoam.parallel.processor_patch_enhanced_6 import (
    OverlappedPatch6,
    EnhancedHaloExchange6,
    WeightedInterpolation,
    BandwidthStats,
)
from pyfoam.parallel.reconstruct_par_enhanced_7 import (
    ReconstructParEnhanced7,
    V7ReconstructResult,
    WaveletCompressionConfig,
    FieldQualityMetrics,
    AMRLevelInfo,
)
from pyfoam.parallel.redistribute_par_enhanced_7 import (
    RedistributeParEnhanced7,
    V7RedistributeResult,
    SpectralPartitionConfig,
    LoadPrediction,
    CommunicationMetrics,
)
from pyfoam.parallel.processor_patch_enhanced_7 import (
    PrefetchablePatch7,
    EnhancedHaloExchange7,
    AsyncScheduleConfig,
    PrefetchStats,
)
from pyfoam.parallel.reconstruct_par_enhanced_8 import (
    ReconstructParEnhanced8,
    V8ReconstructResult,
    StreamingConfig,
    EntropyConfig,
    FieldCorrelation,
)
from pyfoam.parallel.redistribute_par_enhanced_8 import (
    RedistributeParEnhanced8,
    V8RedistributeResult,
    MultiObjectiveConfig,
    PartitionFingerprint,
    IncrementalPlan,
)
from pyfoam.parallel.processor_patch_enhanced_8 import (
    SparseAwarePatch8,
    EnhancedHaloExchange8,
    BatchedExchangeConfig,
    LatencyProfile,
    FaultToleranceConfig,
)
from pyfoam.parallel.reconstruct_par_enhanced_9 import (
    ReconstructParEnhanced9,
    V9ReconstructResult,
    ProgressiveConfig,
    FieldDependency,
    DistributedHash,
)
from pyfoam.parallel.redistribute_par_enhanced_9 import (
    RedistributeParEnhanced9,
    V9RedistributeResult,
    BandwidthScheduleConfig,
    OnlineCostPrediction,
    PartitionQualityMetrics,
)
from pyfoam.parallel.processor_patch_enhanced_9 import (
    TopologyAwarePatch9,
    EnhancedHaloExchange9,
    TopologyRoutingConfig,
    CoalescingConfig,
    CheckpointConfig,
)
from pyfoam.parallel.reconstruct_par_enhanced_10 import (
    ReconstructParEnhanced10,
    V10ReconstructResult,
    SpectralAnalysisConfig,
    CompressionConfig,
    ProvenanceEntry,
)
from pyfoam.parallel.redistribute_par_enhanced_10 import (
    RedistributeParEnhanced10,
    V10RedistributeResult,
    ParetoConfig,
    MultiLevelConfig,
    StabilityMetrics,
)
from pyfoam.parallel.processor_patch_enhanced_10 import (
    HierarchicalPatch10,
    EnhancedHaloExchange10,
    HierarchyConfig,
    CacheLayoutConfig,
    PriorityConfig,
)

__all__ = [
    # Decomposition
    "Decomposition",
    "SubDomain",
    # Decomposition strategies
    "DecompositionStrategy",
    "SimpleDecomposition",
    "ScotchDecomposition",
    # Processor patches
    "ProcessorPatch",
    "HaloExchange",
    # Parallel field
    "ParallelField",
    # Parallel I/O
    "ParallelWriter",
    "ParallelReader",
    # Parallel solver
    "ParallelSolver",
    "ParallelSolverConfig",
    # Reconstruction
    "ReconstructPar",
    "ReconstructResult",
    # Enhanced reconstruction
    "ReconstructParEnhanced",
    "ZoneInfo",
    "EnhancedReconstructResult",
    # V2 enhanced reconstruction
    "ReconstructParEnhanced2",
    "V2ReconstructResult",
    "FieldMergeStats",
    "MergeStrategy",
    # Redistribution
    "RedistributePar",
    "RedistributeResult",
    # Enhanced redistribution
    "RedistributeParEnhanced",
    "BalancingStrategy",
    "EnhancedRedistributeResult",
    "PartitionDiagnostics",
    # V2 enhanced redistribution
    "RedistributeParEnhanced2",
    "GraphPartitionStrategy",
    "V2RedistributeResult",
    # Enhanced processor patches
    "NonConformalPatch",
    "EnhancedHaloExchange",
    # V2 enhanced processor patches
    "NonConformalPatch2",
    "EnhancedHaloExchange2",
    "FaceCentreInterpolator",
    # V3 enhanced reconstruction
    "ReconstructParEnhanced3",
    "V3ReconstructResult",
    "ZoneMergeResult",
    # V3 enhanced redistribution
    "RedistributeParEnhanced3",
    "SpatialDecompositionStrategy",
    "V3RedistributeResult",
    # V3 enhanced processor patches
    "NonConformalPatch3",
    "EnhancedHaloExchange3",
    "AdaptiveInterpolator",
    # V4 enhanced reconstruction
    "ReconstructParEnhanced4",
    "V4ReconstructResult",
    "GradientMergeConfig",
    # V4 enhanced redistribution
    "RedistributeParEnhanced4",
    "V4RedistributeResult",
    "PartitionQualityMetrics",
    # V4 enhanced processor patches
    "NonConformalPatch4",
    "EnhancedHaloExchange4",
    "HigherOrderInterpolator",
    # V5 enhanced reconstruction
    "ReconstructParEnhanced5",
    "V5ReconstructResult",
    "SmoothingConfig",
    # V5 enhanced redistribution
    "RedistributeParEnhanced5",
    "V5RedistributeResult",
    "MigrationPlan",
    "CostEstimator",
    # V5 enhanced processor patches
    "CoarsenablePatch5",
    "EnhancedHaloExchange5",
    "CompressionStats",
    # V6 enhanced reconstruction
    "ReconstructParEnhanced6",
    "V6ReconstructResult",
    "AnisotropicSmoothingConfig",
    "CheckpointInfo",
    # V6 enhanced redistribution
    "RedistributeParEnhanced6",
    "V6RedistributeResult",
    "HierarchicalPartitionConfig",
    "PartitionMetrics",
    # V6 enhanced processor patches
    "OverlappedPatch6",
    "EnhancedHaloExchange6",
    "WeightedInterpolation",
    "BandwidthStats",
    # V7 enhanced reconstruction
    "ReconstructParEnhanced7",
    "V7ReconstructResult",
    "WaveletCompressionConfig",
    "FieldQualityMetrics",
    "AMRLevelInfo",
    # V7 enhanced redistribution
    "RedistributeParEnhanced7",
    "V7RedistributeResult",
    "SpectralPartitionConfig",
    "LoadPrediction",
    "CommunicationMetrics",
    # V7 enhanced processor patches
    "PrefetchablePatch7",
    "EnhancedHaloExchange7",
    "AsyncScheduleConfig",
    "PrefetchStats",
    # V8 enhanced reconstruction
    "ReconstructParEnhanced8",
    "V8ReconstructResult",
    "StreamingConfig",
    "EntropyConfig",
    "FieldCorrelation",
    # V8 enhanced redistribution
    "RedistributeParEnhanced8",
    "V8RedistributeResult",
    "MultiObjectiveConfig",
    "PartitionFingerprint",
    "IncrementalPlan",
    # V8 enhanced processor patches
    "SparseAwarePatch8",
    "EnhancedHaloExchange8",
    "BatchedExchangeConfig",
    "LatencyProfile",
    "FaultToleranceConfig",
    # V9 enhanced reconstruction
    "ReconstructParEnhanced9",
    "V9ReconstructResult",
    "ProgressiveConfig",
    "FieldDependency",
    "DistributedHash",
    # V9 enhanced redistribution
    "RedistributeParEnhanced9",
    "V9RedistributeResult",
    "BandwidthScheduleConfig",
    "OnlineCostPrediction",
    "PartitionQualityMetrics",
    # V9 enhanced processor patches
    "TopologyAwarePatch9",
    "EnhancedHaloExchange9",
    "TopologyRoutingConfig",
    "CoalescingConfig",
    "CheckpointConfig",
    # V10 enhanced reconstruction
    "ReconstructParEnhanced10",
    "V10ReconstructResult",
    "SpectralAnalysisConfig",
    "CompressionConfig",
    "ProvenanceEntry",
    # V10 enhanced redistribution
    "RedistributeParEnhanced10",
    "V10RedistributeResult",
    "ParetoConfig",
    "MultiLevelConfig",
    "StabilityMetrics",
    # V10 enhanced processor patches
    "HierarchicalPatch10",
    "EnhancedHaloExchange10",
    "HierarchyConfig",
    "CacheLayoutConfig",
    "PriorityConfig",
]
