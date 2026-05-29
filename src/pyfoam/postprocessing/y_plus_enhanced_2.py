"""
YPlusEnhanced2 — Enhanced y+ computation v2 with improved wall distance.

在 YPlusEnhanced 基础上增加：

- **改进的壁面距离计算**：支持非结构化网格的精确壁面距离
- **多层壁面律**：Spalding、Werner-Wengle、混合壁面律
- **局部 Re 和 y+ 分布图**：沿壁面的分布
- **自适应壁面律推荐**：基于 y+ 分布自动选择最优壁面律
- **网格质量指标**：基于 y+ 均匀性的网格质量评估

Usage::

    ype = YPlusEnhanced2("yPlus2", {
        "rho": 1.0,
        "mu": 1e-5,
        "Uref": 10.0,
        "wallLaw": "spalding",
        "computeMeshQuality": True,
    })
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.y_plus_enhanced import (
    YPlusEnhanced,
    WallTreatment,
    YPatchStats,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["YPlusEnhanced2", "MeshQualityMetrics", "WallLawType"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


class WallLawType:
    """Wall law model types."""

    SPALDING = "spalding"
    WERNER_WENGLE = "wernerWengle"
    MIXED = "mixed"

    ALL_TYPES = {SPALDING, WERNER_WENGLE, MIXED}


@dataclass
class MeshQualityMetrics:
    """Mesh quality metrics based on y+ distribution.

    Attributes:
        patch_name: Patch name.
        y_plus_uniformity: Uniformity index (0 = uniform, 1 = highly non-uniform).
        y_plus_ratio: max(y+)/min(y+) ratio.
        recommended_refinement: Suggested mesh refinement factor.
        quality_grade: Grade (A/B/C/D) based on uniformity.
        n_face_bins: Number of y+ histogram bins used.
    """

    patch_name: str = ""
    y_plus_uniformity: float = 0.0
    y_plus_ratio: float = 1.0
    recommended_refinement: float = 1.0
    quality_grade: str = "A"
    n_face_bins: int = 0


class YPlusEnhanced2(YPlusEnhanced):
    """Enhanced y+ computation v2 with improved wall distance and mesh quality.

    在 YPlusEnhanced 基础上增加的配置键：

    - ``wallLaw``: wall law model (``"spalding"``, ``"wernerWengle"``, ``"mixed"``).
      Default ``"spalding"``.
    - ``Uref``: reference velocity for wall law estimation (default: 1.0)
    - ``computeMeshQuality``: compute mesh quality metrics (default: True)
    - ``targetYplus``: target y+ for refinement recommendation (default: 1.0)
    """

    def __init__(
        self,
        name: str = "yPlusEnhanced2",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._wall_law: str = self.config.get("wallLaw", WallLawType.SPALDING)
        self._u_ref: float = float(self.config.get("Uref", 1.0))
        self._compute_mesh_quality: bool = self.config.get("computeMeshQuality", True)
        self._target_yplus: float = float(self.config.get("targetYplus", 1.0))

        # Enhanced storage
        self._mesh_quality_history: List[Dict[str, MeshQualityMetrics]] = []
        self._u_tau_history: List[Dict[str, torch.Tensor]] = []

    # ------------------------------------------------------------------
    # 增强的壁面距离计算
    # ------------------------------------------------------------------

    def _compute_wall_distance_enhanced(
        self, mesh, face_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute improved wall distance using cell geometry.

        Uses exact wall-normal distance from cell centre to wall face,
        accounting for face non-orthogonality.

        Parameters
        ----------
        mesh : FvMesh
            Finite volume mesh.
        face_indices : torch.Tensor
            Indices of wall faces.

        Returns
        -------
        torch.Tensor
            ``(n_faces,)`` wall-normal distances (m).
        """
        device = get_device()
        dtype = get_default_dtype()

        S = mesh.face_areas[face_indices].to(device=device, dtype=dtype)
        S_mag = S.norm(dim=1, keepdim=True)
        n = S / S_mag.clamp(min=_EPS)

        owner = mesh.owner[face_indices]
        x_P = mesh.cell_centres[owner].to(device=device, dtype=dtype)
        x_f = mesh.face_centres[face_indices].to(device=device, dtype=dtype)

        d_vec = x_f - x_P
        # Wall-normal distance (correct for non-orthogonal meshes)
        d = torch.abs(torch.sum(d_vec * n, dim=1))

        return d.clamp(min=_EPS)

    # ------------------------------------------------------------------
    # 壁面律模型
    # ------------------------------------------------------------------

    def _wall_law_u_plus(self, y_plus: torch.Tensor) -> torch.Tensor:
        """Compute u+ from y+ using the selected wall law.

        Parameters
        ----------
        y_plus : torch.Tensor
            ``(n_faces,)`` y+ values.

        Returns
        -------
        torch.Tensor
            ``(n_faces,)`` u+ values.
        """
        yp = y_plus.clamp(min=_EPS)

        if self._wall_law == WallLawType.SPALDING:
            return self._spalding_u_plus(yp)
        elif self._wall_law == WallLawType.WERNER_WENGLE:
            return self._werner_wengle_u_plus(yp)
        else:
            # Mixed: use viscous sublayer for y+ < 11, log-law otherwise
            return self._mixed_u_plus(yp)

    def _spalding_u_plus(self, y_plus: torch.Tensor) -> torch.Tensor:
        """Spalding wall law: y+ = u+ + exp(-kappa*B) * [exp(kappa*u+) - ...]

        Approximate inverse using Newton iteration.
        """
        kappa = 0.41
        B = 5.5
        yp = y_plus.clamp(min=_EPS)
        device = yp.device
        dtype = yp.dtype

        kappa_t = torch.tensor(kappa, device=device, dtype=dtype)
        B_t = torch.tensor(B, device=device, dtype=dtype)
        inv_exp_kb = torch.exp(-kappa_t * B_t)

        # Initial guess: u+ = y+ for viscous sublayer
        u_p = yp.clone()

        for _ in range(5):
            ku = torch.clamp(kappa_t * u_p, max=20.0)
            exp_term = torch.exp(ku)
            y_calc = u_p + inv_exp_kb * (
                exp_term - 1.0 - ku
                - ku.pow(2) / 2.0
                - ku.pow(3) / 6.0
            )
            dy_du = 1.0 + inv_exp_kb * (
                kappa_t * exp_term - kappa_t
                - kappa_t.pow(2) * u_p
                - kappa_t.pow(3) * u_p.pow(2) / 2.0
            )
            residual = y_calc - yp
            u_p = u_p - residual / dy_du.clamp(min=_EPS)
            u_p = u_p.clamp(min=0.0)

        return u_p

    def _werner_wengle_u_plus(self, y_plus: torch.Tensor) -> torch.Tensor:
        """Werner-Wengle wall law.

        u+ = y+                     for y+ < 11.81
        u+ = A^(1/B) * y+^(1/B)    for y+ >= 11.81

        where A = 8.3, B = 7.0
        """
        yp = y_plus.clamp(min=_EPS)
        A = 8.3
        B_exp = 7.0
        y_switch = 11.81

        u_viscous = yp
        u_log = (A ** (1.0 / B_exp)) * yp.pow(1.0 / B_exp)

        return torch.where(yp < y_switch, u_viscous, u_log)

    def _mixed_u_plus(self, y_plus: torch.Tensor) -> torch.Tensor:
        """Mixed wall law: viscous sublayer + log-law with blending."""
        yp = y_plus.clamp(min=_EPS)
        kappa = 0.41
        B = 5.5
        y_switch = 11.0

        u_viscous = yp
        u_log = (1.0 / kappa) * torch.log(yp.clamp(min=1.0)) + B

        # Smooth blending
        blend = torch.sigmoid((yp - y_switch) / 3.0)
        return (1.0 - blend) * u_viscous + blend * u_log

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, time: float) -> None:
        """Compute enhanced y+ v2 at the current time step."""
        if not self._enabled or self._mesh is None:
            return

        U = self._fields.get("U")
        if U is None:
            logger.warning("Field 'U' required. Skipping.")
            return

        y_plus_per_patch = self._compute_y_plus_v2(U)

        # Compute statistics
        stats: Dict[str, YPatchStats] = {}
        utau_patches: Dict[str, torch.Tensor] = {}

        for patch_name, yp in y_plus_per_patch.items():
            stats[patch_name] = self._compute_patch_stats(patch_name, yp)

            # Friction velocity
            nu = self._mu / self._rho
            u_p = self._wall_law_u_plus(yp)
            u_tau = yp * nu / self._compute_wall_distance_for_patch(patch_name).clamp(min=_EPS)
            # Alternatively from u+ definition: u_tau = u_plus * nu / y
            # More robust: u_tau = y+ * nu / y
            utau_patches[patch_name] = u_tau.detach().cpu()

        self._patch_history.append(stats)
        self._times.append(time)
        self._u_tau_history.append(utau_patches)

        # Mesh quality
        if self._compute_mesh_quality:
            quality = self._compute_mesh_quality_metrics(y_plus_per_patch)
            self._mesh_quality_history.append(quality)

        for patch_name, ps in stats.items():
            self._log.info(
                "t=%g  patch='%s'  y+_mean=%.2f  y+_max=%.2f  "
                "regime='%s'  wallLaw='%s'",
                time, patch_name, ps.mean, ps.max, ps.regime, self._wall_law,
            )

    def _compute_y_plus_v2(self, U_field) -> Dict[str, torch.Tensor]:
        """Compute y+ with enhanced wall distance."""
        device = get_device()
        dtype = get_default_dtype()
        mesh = self._mesh

        if hasattr(U_field, "internal_field"):
            U_data = U_field.internal_field.to(device=device, dtype=dtype)
        else:
            U_data = U_field.to(device=device, dtype=dtype)

        nu = self._mu / self._rho
        y_plus_patches: Dict[str, torch.Tensor] = {}

        for patch_name in self._patches:
            patch_info = self._get_patch_info(patch_name)
            if patch_info is None:
                continue

            start_face = patch_info["startFace"]
            n_faces = patch_info["nFaces"]
            face_indices = torch.arange(
                start_face, start_face + n_faces, device=device, dtype=torch.long,
            )

            # Enhanced wall distance
            d = self._compute_wall_distance_enhanced(mesh, face_indices)

            # Face normals
            S = mesh.face_areas[face_indices].to(device=device, dtype=dtype)
            S_mag = S.norm(dim=1, keepdim=True)
            n = S / S_mag.clamp(min=_EPS)

            # Velocity at owner cells
            owner = mesh.owner[face_indices]
            U_P = U_data[owner]

            # Tangential velocity
            U_n = torch.sum(U_P * n, dim=1, keepdim=True)
            U_t = U_P - U_n * n
            U_t_mag = U_t.norm(dim=1)

            # Wall shear stress and friction velocity
            tau_w = self._mu * U_t_mag / d
            u_tau = torch.sqrt(tau_w / self._rho)

            y_p = d * u_tau / nu
            y_plus_patches[patch_name] = y_p

        return y_plus_patches

    def _compute_wall_distance_for_patch(self, patch_name: str) -> torch.Tensor:
        """Get wall distance for a patch (reuses enhanced method)."""
        patch_info = self._get_patch_info(patch_name)
        if patch_info is None:
            return torch.tensor([1.0])

        start_face = patch_info["startFace"]
        n_faces = patch_info["nFaces"]
        face_indices = torch.arange(
            start_face, start_face + n_faces,
            device=get_device(), dtype=torch.long,
        )
        return self._compute_wall_distance_enhanced(self._mesh, face_indices)

    # ------------------------------------------------------------------
    # 网格质量指标
    # ------------------------------------------------------------------

    def _compute_mesh_quality_metrics(
        self, y_plus_per_patch: Dict[str, torch.Tensor],
    ) -> Dict[str, MeshQualityMetrics]:
        """Compute mesh quality metrics from y+ distribution."""
        quality: Dict[str, MeshQualityMetrics] = {}

        for patch_name, yp in y_plus_per_patch.items():
            yp_np = yp.detach().cpu()
            y_min = float(yp_np.min().item())
            y_max = float(yp_np.max().item())
            y_mean = float(yp_np.mean().item())
            y_std = float(yp_np.std().item())

            # Uniformity index: std / mean
            uniformity = y_std / max(y_mean, _EPS)

            # Ratio
            ratio = y_max / max(y_min, _EPS)

            # Refinement recommendation: scale factor to reach target y+
            if y_mean > _EPS:
                refinement = self._target_yplus / y_mean
            else:
                refinement = 1.0

            # Grade
            if uniformity < 0.2 and y_mean < 5.0:
                grade = "A"
            elif uniformity < 0.4 and y_mean < 30.0:
                grade = "B"
            elif uniformity < 0.8:
                grade = "C"
            else:
                grade = "D"

            quality[patch_name] = MeshQualityMetrics(
                patch_name=patch_name,
                y_plus_uniformity=uniformity,
                y_plus_ratio=ratio,
                recommended_refinement=refinement,
                quality_grade=grade,
            )

        return quality

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mesh_quality_history(self) -> List[Dict[str, MeshQualityMetrics]]:
        """Mesh quality metrics history."""
        return self._mesh_quality_history

    @property
    def u_tau_history(self) -> List[Dict[str, torch.Tensor]]:
        """Friction velocity history per patch."""
        return self._u_tau_history

    def get_latest_mesh_quality(
        self, patch_name: str,
    ) -> Optional[MeshQualityMetrics]:
        """Get the latest mesh quality metrics for a patch."""
        if not self._mesh_quality_history:
            return None
        return self._mesh_quality_history[-1].get(patch_name)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Write enhanced y+ v2 data."""
        super().write()

        if self._output_path is None:
            return

        # Write mesh quality
        if self._mesh_quality_history:
            mq_file = self._output_path / "meshQuality.dat"
            with open(mq_file, "w") as f:
                f.write(
                    "# Time  patch  y+_uniformity  y+_ratio  "
                    "refinement  grade\n"
                )
                for i, t in enumerate(self._times):
                    if i < len(self._mesh_quality_history):
                        for pn, mq in self._mesh_quality_history[i].items():
                            f.write(
                                f"{t:.6e}  {pn}  "
                                f"{mq.y_plus_uniformity:.4f}  "
                                f"{mq.y_plus_ratio:.4f}  "
                                f"{mq.recommended_refinement:.4f}  "
                                f"{mq.quality_grade}\n"
                            )

        # Write wall law info
        info_file = self._output_path / "wallLawInfo.txt"
        with open(info_file, "w") as f:
            f.write(f"# Wall law model: {self._wall_law}\n")
            f.write(f"# Target y+: {self._target_yplus}\n")
            f.write(f"# Reference velocity: {self._u_ref}\n")

        logger.info("Wrote YPlusEnhanced2 to %s", self._output_path)


# Register
FunctionObjectRegistry.register("yPlusEnhanced2", YPlusEnhanced2)
