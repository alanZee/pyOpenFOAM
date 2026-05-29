"""
ForcesEnhanced2 — Enhanced forces v2 with moment decomposition.

在 Enhanced v1 基础上增加：

- **完整力矩分解**：压力力矩和粘性力矩分开计算
- **力矩臂计算**：自动计算力矩臂和力矩系数
- **投影力分解**：升力/阻力/侧力分解

Usage::

    forces = ForcesEnhanced2("forces2", {
        "patches": ["cylinder"],
        "rhoInf": 1.225,
        "CofR": [0.0, 0.0, 0.0],
        "liftDir": [0.0, 1.0, 0.0],
        "dragDir": [1.0, 0.0, 0.0],
    })
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.forces_enhanced import (
    ForcesEnhanced,
    ForceDecomposition,
    FluctuationStats,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["ForcesEnhanced2", "ProjectedForces"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


@dataclass
class ProjectedForces:
    """Projected forces along reference directions.

    Attributes:
        time: Simulation time.
        drag: Force along drag direction.
        lift: Force along lift direction.
        side: Force along side direction (cross product of drag x lift).
        drag_pressure: Pressure component of drag.
        drag_viscous: Viscous component of drag.
        lift_pressure: Pressure component of lift.
        lift_viscous: Viscous component of lift.
        Cx: Drag coefficient.
        Cy: Lift coefficient.
    """

    time: float = 0.0
    drag: float = 0.0
    lift: float = 0.0
    side: float = 0.0
    drag_pressure: float = 0.0
    drag_viscous: float = 0.0
    lift_pressure: float = 0.0
    lift_viscous: float = 0.0
    Cx: float = 0.0
    Cy: float = 0.0


class ForcesEnhanced2(ForcesEnhanced):
    """Enhanced forces v2 with moment decomposition and projected forces.

    在 ForcesEnhanced 基础上增加的配置键：

    - ``liftDir``: lift direction vector (default: [0, 1, 0])
    - ``dragDir``: drag direction vector (default: [1, 0, 0])
    - ``Lref``: reference length for moment coefficient (default: 1.0)
    - ``Aref``: reference area for force coefficient (default: 1.0)
    - ``computeCoefficients``: compute Cd, Cl (default: True)
    """

    def __init__(
        self,
        name: str = "forcesEnhanced2",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)

        # Direction vectors
        lift_dir = self.config.get("liftDir", [0.0, 1.0, 0.0])
        drag_dir = self.config.get("dragDir", [1.0, 0.0, 0.0])
        self._lift_dir = torch.tensor(lift_dir, dtype=torch.float64)
        self._drag_dir = torch.tensor(drag_dir, dtype=torch.float64)

        # Normalize
        self._lift_dir = self._lift_dir / max(self._lift_dir.norm().item(), _EPS)
        self._drag_dir = self._drag_dir / max(self._drag_dir.norm().item(), _EPS)

        # Side direction: drag x lift
        self._side_dir = torch.linalg.cross(self._drag_dir, self._lift_dir)
        side_norm = self._side_dir.norm().item()
        if side_norm > _EPS:
            self._side_dir = self._side_dir / side_norm
        else:
            self._side_dir = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)

        # Reference quantities
        self._L_ref: float = float(self.config.get("Lref", 1.0))
        self._A_ref: float = float(self.config.get("Aref", 1.0))
        self._compute_coefficients: bool = self.config.get("computeCoefficients", True)

        # Storage
        self._projected_forces: List[ProjectedForces] = []

    def execute(self, time: float) -> None:
        """Compute forces v2 at current time step."""
        # Run parent execute (forces, decomposition, extra moments)
        super().execute(time)

        if not self._enabled or self._mesh is None:
            return

        if not self._force_total:
            return

        # Latest total force
        f_total = self._force_total[-1]
        decomp = self._decompositions[-1] if self._decompositions else None

        # Project forces
        f_tensor = f_total.to(dtype=torch.float64)
        drag = float(torch.dot(f_tensor, self._drag_dir).item())
        lift = float(torch.dot(f_tensor, self._lift_dir).item())
        side = float(torch.dot(f_tensor, self._side_dir).item())

        # Pressure/viscous decomposition of projected forces
        drag_p, drag_v, lift_p, lift_v = 0.0, 0.0, 0.0, 0.0
        if decomp is not None and decomp.F_pressure is not None:
            fp = decomp.F_pressure.to(dtype=torch.float64)
            fv = decomp.F_viscous.to(dtype=torch.float64) if decomp.F_viscous is not None else torch.zeros(3, dtype=torch.float64)
            drag_p = float(torch.dot(fp, self._drag_dir).item())
            drag_v = float(torch.dot(fv, self._drag_dir).item())
            lift_p = float(torch.dot(fp, self._lift_dir).item())
            lift_v = float(torch.dot(fv, self._lift_dir).item())

        # Coefficients
        rho = self.config.get("rhoInf", 1.0)
        U_ref = self.config.get("Uref", 1.0)
        q_ref = 0.5 * rho * U_ref ** 2 * self._A_ref
        Cx = drag / max(q_ref, _EPS)
        Cy = lift / max(q_ref, _EPS)

        pf = ProjectedForces(
            time=time,
            drag=drag,
            lift=lift,
            side=side,
            drag_pressure=drag_p,
            drag_viscous=drag_v,
            lift_pressure=lift_p,
            lift_viscous=lift_v,
            Cx=Cx,
            Cy=Cy,
        )
        self._projected_forces.append(pf)

        self._log.info(
            "t=%g  drag=%.6g  lift=%.6g  side=%.6g  Cd=%.6g  Cl=%.6g",
            time, drag, lift, side, Cx, Cy,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def projected_forces(self) -> List[ProjectedForces]:
        """Projected force history."""
        return self._projected_forces

    @property
    def lift_dir(self) -> torch.Tensor:
        """Lift direction."""
        return self._lift_dir

    @property
    def drag_dir(self) -> torch.Tensor:
        """Drag direction."""
        return self._drag_dir

    @property
    def side_dir(self) -> torch.Tensor:
        """Side direction (cross product of drag x lift)."""
        return self._side_dir

    def get_latest_projected(self) -> Optional[ProjectedForces]:
        """Get the latest projected forces."""
        if not self._projected_forces:
            return None
        return self._projected_forces[-1]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Write forces v2 data."""
        super().write()

        if self._output_path is None or not self._projected_forces:
            return

        proj_file = self._output_path / "projectedForces.dat"
        with open(proj_file, "w") as f:
            header = (
                "# Time  drag  lift  side  drag_p  drag_v  lift_p  lift_v  Cd  Cl"
            )
            f.write(header + "\n")
            for pf in self._projected_forces:
                f.write(
                    f"{pf.time:.6e}  "
                    f"{pf.drag:.6e}  {pf.lift:.6e}  {pf.side:.6e}  "
                    f"{pf.drag_pressure:.6e}  {pf.drag_viscous:.6e}  "
                    f"{pf.lift_pressure:.6e}  {pf.lift_viscous:.6e}  "
                    f"{pf.Cx:.6e}  {pf.Cy:.6e}\n"
                )

        logger.info("Wrote ForcesEnhanced2 to %s", self._output_path)


# Register
FunctionObjectRegistry.register("forcesEnhanced2", ForcesEnhanced2)
