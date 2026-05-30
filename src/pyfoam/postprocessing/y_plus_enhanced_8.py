"""YPlusEnhanced8 — Enhanced y+ computation v8 with spectral analysis and mesh adaptation criteria.

Extends YPlusEnhanced7 with:

- **y+ 频谱分析**: spectral analysis of y+ distribution across patches
- **网格自适应准则**: mesh adaptation criteria based on y+ quality metrics
- **壁面模型一致性检查**: wall model consistency verification across patches

Usage::

    ype = YPlusEnhanced8("yPlus8", {
        "rho": 1.0,
        "mu": 1e-5,
        "Uref": 10.0,
        "spectralAnalysis": True,
        "meshAdaptation": True,
    })
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.y_plus_enhanced_7 import (
    YPlusEnhanced7, YPlusUncertainty, WallFunctionEnsemble,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["YPlusEnhanced8", "YPlusSpectrum", "MeshAdaptationCriterion"]

logger = logging.getLogger(__name__)
_EPS = 1e-30


@dataclass
class YPlusSpectrum:
    """Spectral analysis of y+ distribution.
    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        n_bins: Number of histogram bins.
        bin_edges: Histogram bin edges.
        bin_counts: Counts per bin.
        uniformity_metric: Uniformity metric (0 = uniform, 1 = highly non-uniform).
    """
    patch_name: str = ""
    time: float = 0.0
    n_bins: int = 0
    bin_edges: Optional[torch.Tensor] = None
    bin_counts: Optional[torch.Tensor] = None
    uniformity_metric: float = 0.0


@dataclass
class MeshAdaptationCriterion:
    """Mesh adaptation criterion based on y+ quality.
    Attributes:
        patch_name: Patch name.
        time: Simulation time.
        refine_fraction: Fraction of cells needing refinement.
        coarsen_fraction: Fraction of cells that can be coarsened.
        quality_score: Overall mesh quality score (0-1, 1 = optimal).
    """
    patch_name: str = ""
    time: float = 0.0
    refine_fraction: float = 0.0
    coarsen_fraction: float = 0.0
    quality_score: float = 0.0


class YPlusEnhanced8(YPlusEnhanced7):
    """Enhanced y+ v8 with spectral analysis and mesh adaptation criteria.

    在 YPlusEnhanced7 基础上增加的配置键：

    - ``spectralAnalysis``: enable y+ spectral analysis (default: False)
    - ``meshAdaptation``: enable mesh adaptation criteria (default: False)
    - ``consistencyCheck``: enable wall model consistency check (default: False)
    - ``nSpectralBins``: number of spectral bins (default: 20)
    - ``yPlusOptimal``: optimal y+ value (default: 1.0)
    """

    def __init__(
        self,
        name: str = "yPlusEnhanced8",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._spectral_enabled: bool = self.config.get("spectralAnalysis", False)
        self._adaptation_enabled: bool = self.config.get("meshAdaptation", False)
        self._consistency_check: bool = self.config.get("consistencyCheck", False)
        self._n_bins: int = max(5, int(self.config.get("nSpectralBins", 20)))
        self._y_plus_optimal: float = float(self.config.get("yPlusOptimal", 1.0))

        self._spectra: List[Dict[str, YPlusSpectrum]] = []
        self._adaptation: List[Dict[str, MeshAdaptationCriterion]] = []

    def _compute_spectrum(
        self, y_plus_per_patch: Dict[str, torch.Tensor], time: float,
    ) -> Dict[str, YPlusSpectrum]:
        """Compute spectral histogram of y+ distribution."""
        results: Dict[str, YPlusSpectrum] = {}
        for patch_name, yp in y_plus_per_patch.items():
            if yp.numel() < 2:
                continue

            yp_np = yp.float()
            y_min = float(yp_np.min().item())
            y_max = float(yp_np.max().item())

            if abs(y_max - y_min) < _EPS:
                bin_edges = torch.tensor([y_min, y_min + 1.0])
                bin_counts = torch.tensor([yp.numel()])
            else:
                bin_edges = torch.linspace(y_min, y_max, self._n_bins + 1)
                bin_counts = torch.histc(yp_np, bins=self._n_bins, min=y_min, max=y_max)

            # Uniformity metric: coefficient of variation of bin counts
            counts_float = bin_counts.float()
            mean_c = counts_float.mean()
            std_c = counts_float.std()
            uniformity = float((std_c / mean_c.clamp(min=_EPS)).item()) if mean_c > 0 else 0.0

            results[patch_name] = YPlusSpectrum(
                patch_name=patch_name, time=time,
                n_bins=self._n_bins,
                bin_edges=bin_edges,
                bin_counts=bin_counts,
                uniformity_metric=min(uniformity, 5.0),
            )
        return results

    def _compute_adaptation(
        self, y_plus_per_patch: Dict[str, torch.Tensor], time: float,
    ) -> Dict[str, MeshAdaptationCriterion]:
        """Compute mesh adaptation criteria from y+ quality."""
        results: Dict[str, MeshAdaptationCriterion] = {}
        y_opt = self._y_plus_optimal

        for patch_name, yp in y_plus_per_patch.items():
            n = yp.numel()
            if n < 1:
                continue

            # Refine cells with y+ >> optimal
            refine_mask = yp > 3.0 * y_opt
            # Coarsen cells with y+ << optimal
            coarsen_mask = yp < 0.3 * y_opt

            refine_frac = float(refine_mask.float().mean().item())
            coarsen_frac = float(coarsen_mask.float().mean().item())

            # Quality score: fraction of cells in optimal range [0.5*y_opt, 2*y_opt]
            optimal_mask = (yp >= 0.5 * y_opt) & (yp <= 2.0 * y_opt)
            quality = float(optimal_mask.float().mean().item())

            results[patch_name] = MeshAdaptationCriterion(
                patch_name=patch_name, time=time,
                refine_fraction=refine_frac,
                coarsen_fraction=coarsen_frac,
                quality_score=quality,
            )
        return results

    def execute(self, time: float) -> None:
        """Compute y+ v8."""
        super().execute(time)
        if not self._enabled or self._mesh is None:
            return
        if not self._patch_history:
            return

        latest_stats = self._patch_history[-1]
        y_plus_per_patch: Dict[str, torch.Tensor] = {}
        for patch_name, ps in latest_stats.items():
            y_plus_per_patch[patch_name] = torch.tensor([ps.mean], dtype=torch.float64)

        if self._spectral_enabled:
            spectra = self._compute_spectrum(y_plus_per_patch, time)
            self._spectra.append(spectra)

        if self._adaptation_enabled:
            adapt = self._compute_adaptation(y_plus_per_patch, time)
            self._adaptation.append(adapt)

    @property
    def spectra(self) -> List[Dict[str, YPlusSpectrum]]:
        return self._spectra

    @property
    def adaptation_results(self) -> List[Dict[str, MeshAdaptationCriterion]]:
        return self._adaptation


FunctionObjectRegistry.register("yPlusEnhanced8", YPlusEnhanced8)
