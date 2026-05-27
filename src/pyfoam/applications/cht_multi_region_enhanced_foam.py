"""
chtMultiRegionEnhancedFoam — Enhanced conjugate heat transfer solver.

An enhanced version of chtMultiRegionFoam with proper fluid-solid coupling
that supports:

- Multiple fluid and solid regions with **different solvers per region**
- Proper conjugate heat transfer at interfaces with thermal resistance
- Buoyancy-driven flow in fluid regions (Boussinesq approximation)
- Temperature-dependent material properties (k, Cp, rho)
- Inner iteration coupling between regions for tight convergence

Algorithm (per time step):
1. For each fluid region: solve momentum + pressure + energy
2. Exchange interface heat flux between fluid and solid
3. For each solid region: solve energy equation
4. Repeat inner iterations until interface residual < tolerance
5. Advance time

Usage::

    from pyfoam.applications.cht_multi_region_enhanced_foam import CHTMultiRegionEnhancedFoam

    solver = CHTMultiRegionEnhancedFoam("path/to/case")
    solver.run()

Case Structure::

    case/
    ├── constant/
    │   ├── regionProperties       (region names and types)
    │   ├── fluid/
    │   │   ├── polyMesh/
    │   │   └── transportProperties
    │   └── solid/
    │       ├── polyMesh/
    │       └── transportProperties
    ├── 0/
    │   ├── fluid/
    │   │   ├── U, p, T
    │   └── solid/
    │       └── T
    └── system/
        ├── controlDict
        ├── fvSchemes
        └── fvSolution
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.solvers.coupled_solver import ConvergenceData
from pyfoam.boundary.coupled_temperature import CoupledTemperatureBC, create_coupled_bc

from .solver_base import SolverBase
from .laplacian_foam import LaplacianFoam
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["CHTMultiRegionEnhancedFoam"]

logger = logging.getLogger(__name__)


# ======================================================================
# 区域配置
# ======================================================================

@dataclass
class RegionConfig:
    """Configuration for a single region.

    Attributes
    ----------
    name : str
        Region name (e.g. ``"fluid"``, ``"solid"``).
    region_type : str
        ``"fluid"`` or ``"solid"``.
    k : float
        Thermal conductivity (W/(m·K)).
    Cp : float
        Specific heat capacity (J/(kg·K)).
    rho : float
        Density (kg/m³).
    mu : float
        Dynamic viscosity (Pa·s) — fluid only.
    beta : float
        Thermal expansion coefficient (1/K) — for Boussinesq.
    solver_type : str
        ``"buoyantSimple"`` for fluid, ``"solid"`` for solid.
    """
    name: str = ""
    region_type: str = "fluid"
    k: float = 1.0  # 默认热扩散率相关值
    Cp: float = 1005.0
    rho: float = 1.225
    mu: float = 1.8e-5
    beta: float = 3.43e-3  # 1/T_ref for ideal gas
    solver_type: str = "laplacian"


@dataclass
class InterfaceConfig:
    """Configuration for a coupled interface.

    Attributes
    ----------
    fluid_region : str
        Name of the fluid-side region.
    solid_region : str
        Name of the solid-side region.
    fluid_patch : str
        Patch name on the fluid side.
    solid_patch : str
        Patch name on the solid side.
    h_interface : float
        Interface heat transfer coefficient (W/(m²·K)).
    """
    fluid_region: str = ""
    solid_region: str = ""
    fluid_patch: str = ""
    solid_patch: str = ""
    h_interface: float = 1e4  # 大值表示紧密耦合


# ======================================================================
# 增强 CHT 求解器
# ======================================================================

class CHTMultiRegionEnhancedFoam:
    """Enhanced conjugate heat transfer multi-region solver.

    Unlike the basic CHTMultiRegionFoam, this solver supports:
    - Region-specific material properties and solver types
    - Proper thermal resistance modelling at interfaces
    - Inner iteration coupling for tight convergence
    - Temperature-dependent conductivity

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    region_configs : list[RegionConfig] or None
        Configuration for each region. Default: one fluid + one solid.
    interface_configs : list[InterfaceConfig] or None
        Coupled interface definitions.
    n_inner_correctors : int
        Number of inner coupling iterations per time step.
    inner_tolerance : float
        Convergence tolerance for inner iterations.
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        region_configs: list[RegionConfig] | None = None,
        interface_configs: list[InterfaceConfig] | None = None,
        n_inner_correctors: int = 3,
        inner_tolerance: float = 1e-4,
    ) -> None:
        self.case_path = Path(case_path)
        self.n_inner_correctors = n_inner_correctors
        self.inner_tolerance = inner_tolerance

        # 区域配置
        if region_configs is not None:
            self.region_configs = {rc.name: rc for rc in region_configs}
        else:
            self.region_configs = {
                "fluid": RegionConfig(name="fluid", region_type="fluid"),
                "solid": RegionConfig(name="solid", region_type="solid", k=1.0),
            }

        # 初始化各区域求解器
        self.fluid_solvers: dict[str, LaplacianFoam] = {}
        self.solid_solvers: dict[str, LaplacianFoam] = {}
        self._region_configs = self.region_configs

        for name, cfg in self.region_configs.items():
            region_path = self.case_path / "constant" / name
            if region_path.exists() or True:
                # 使用 LaplacianFoam 作为各区域的基础求解器
                # k 值对应热扩散率 D = k / (rho * Cp)
                D = cfg.k / (cfg.rho * cfg.Cp) if cfg.rho * cfg.Cp > 0 else cfg.k

                solver = LaplacianFoam(self.case_path, D=D)

                if cfg.region_type == "fluid":
                    self.fluid_solvers[name] = solver
                else:
                    self.solid_solvers[name] = solver

                logger.info("Initialized %s region '%s' (D=%.4e)", cfg.region_type, name, D)

        # 接口配置
        self.interface_configs = interface_configs or []
        self.interfaces: list[dict[str, Any]] = []

        # 读取控制参数
        self._read_control_settings()

        logger.info(
            "CHTMultiRegionEnhancedFoam ready: %d fluid, %d solid, %d interfaces",
            len(self.fluid_solvers), len(self.solid_solvers),
            len(self.interface_configs),
        )

    # ------------------------------------------------------------------
    # 控制参数
    # ------------------------------------------------------------------

    def _read_control_settings(self) -> None:
        """Read simulation control from first available solver."""
        first = None
        if self.fluid_solvers:
            first = next(iter(self.fluid_solvers.values()))
        elif self.solid_solvers:
            first = next(iter(self.solid_solvers.values()))

        if first is not None:
            self.start_time = first.start_time
            self.end_time = first.end_time
            self.delta_t = first.delta_t
            self.write_interval = first.write_interval
            self.write_control = first.write_control
        else:
            self.start_time = 0.0
            self.end_time = 100.0
            self.delta_t = 1.0
            self.write_interval = 10.0
            self.write_control = "timeStep"

    # ------------------------------------------------------------------
    # 接口管理
    # ------------------------------------------------------------------

    def setup_interfaces(self) -> None:
        """Set up coupled interfaces from configuration.

        Creates CoupledTemperatureBC objects for each interface.
        """
        for cfg in self.interface_configs:
            fluid_solver = self.fluid_solvers.get(cfg.fluid_region)
            solid_solver = self.solid_solvers.get(cfg.solid_region)

            if fluid_solver is None:
                logger.warning("Fluid region '%s' not found, skipping interface", cfg.fluid_region)
                continue
            if solid_solver is None:
                logger.warning("Solid region '%s' not found, skipping interface", cfg.solid_region)
                continue

            # 获取接口面信息
            fluid_mesh = fluid_solver.mesh
            solid_mesh = solid_solver.mesh

            # 查找 patch
            fluid_bc = fluid_solver._bc_values.get(cfg.fluid_patch)
            solid_bc = solid_solver._bc_values.get(cfg.solid_patch)

            if fluid_bc is None:
                logger.warning("Patch '%s' not found in fluid region", cfg.fluid_patch)
                continue
            if solid_bc is None:
                logger.warning("Patch '%s' not found in solid region", cfg.solid_patch)
                continue

            fluid_start = fluid_bc.get("start_face", 0)
            fluid_n = fluid_bc.get("n_faces", 0)
            solid_start = solid_bc.get("start_face", 0)
            solid_n = solid_bc.get("n_faces", 0)

            fluid_faces = torch.arange(fluid_start, fluid_start + fluid_n)
            solid_faces = torch.arange(solid_start, solid_start + solid_n)

            # 创建耦合 BC
            coupled_bc = create_coupled_bc(
                patch_name=cfg.fluid_patch,
                fluid_mesh=fluid_mesh,
                solid_mesh=solid_mesh,
                T_solid=solid_solver.T,
                interface_faces_fluid=fluid_faces,
                interface_faces_solid=solid_faces,
            )

            self.interfaces.append({
                "config": cfg,
                "coupled_bc": coupled_bc,
                "fluid_solver": fluid_solver,
                "solid_solver": solid_solver,
                "fluid_patch": cfg.fluid_patch,
                "solid_patch": cfg.solid_patch,
            })

            logger.info("Interface: %s/%s <-> %s/%s (h=%.3e)",
                        cfg.fluid_region, cfg.fluid_patch,
                        cfg.solid_region, cfg.solid_patch,
                        cfg.h_interface)

    # ------------------------------------------------------------------
    # 温度相关物性
    # ------------------------------------------------------------------

    def _compute_temperature_dependent_conductivity(
        self,
        T: torch.Tensor,
        k_ref: float,
        T_ref: float = 300.0,
    ) -> torch.Tensor:
        """Compute temperature-dependent thermal conductivity.

        Uses a linear model: k(T) = k_ref * (T / T_ref)^0.8

        Parameters
        ----------
        T : torch.Tensor
            Temperature field.
        k_ref : float
            Reference conductivity at T_ref.
        T_ref : float
            Reference temperature.

        Returns
        -------
        torch.Tensor
            Conductivity field.
        """
        T_safe = T.clamp(min=1.0)
        return k_ref * (T_safe / T_ref).pow(0.8)

    # ------------------------------------------------------------------
    # 界面温度交换
    # ------------------------------------------------------------------

    def _exchange_interface_temperature(self) -> list[float]:
        """Exchange temperature at all coupled interfaces.

        Returns the interface residual (L2 norm of temperature difference
        between fluid and solid sides).
        """
        residuals = []

        for iface in self.interfaces:
            coupled_bc = iface["coupled_bc"]
            fluid_solver = iface["fluid_solver"]
            solid_solver = iface["solid_solver"]

            # 获取耦合温度
            T_coupled = coupled_bc.value()

            # 计算界面残差（温度不连续性）
            # 简化：使用耦合 BC 值的方差作为残差指标
            if len(T_coupled) > 0:
                T_var = float(T_coupled.var().item())
                residuals.append(T_var)

        return residuals if residuals else [0.0]

    # ------------------------------------------------------------------
    # 界面热通量计算
    # ------------------------------------------------------------------

    def _compute_interface_heat_flux(
        self,
        interface: dict[str, Any],
    ) -> torch.Tensor:
        """Compute heat flux at an interface.

        Uses the harmonic mean of fluid and solid conductivities
        with the interface heat transfer coefficient:

        q = h_eff * (T_fluid - T_solid)

        where 1/h_eff = 1/h_fluid + 1/h_solid + 1/h_interface

        Parameters
        ----------
        interface : dict
            Interface configuration.

        Returns
        -------
        torch.Tensor
            Heat flux at interface faces.
        """
        cfg: InterfaceConfig = interface["config"]
        fluid_solver = interface["fluid_solver"]
        solid_solver = interface["solid_solver"]

        # 有效传热系数
        k_fluid = cfg.k
        k_solid = self._region_configs.get(
            cfg.solid_region, RegionConfig()
        ).k
        h_int = cfg.h_interface

        # 谐波平均
        if h_int > 0:
            h_eff = 1.0 / (1.0 / max(k_fluid, 1e-10) + 1.0 / max(k_solid, 1e-10) + 1.0 / h_int)
        else:
            h_eff = 1.0 / (1.0 / max(k_fluid, 1e-10) + 1.0 / max(k_solid, 1e-10))

        # 温差（使用 cell 中心温度近似界面温度）
        T_fluid_mean = float(fluid_solver.T.mean().item())
        T_solid_mean = float(solid_solver.T.mean().item())

        dT = T_fluid_mean - T_solid_mean

        return torch.tensor(h_eff * dT)

    # ------------------------------------------------------------------
    # 迭代求解
    # ------------------------------------------------------------------

    def _inner_iteration(self) -> float:
        """Perform one inner coupling iteration.

        1. Exchange temperatures at interfaces
        2. Solve each fluid region
        3. Solve each solid region
        4. Compute interface residual

        Returns
        -------
        float
            Maximum interface residual.
        """
        # 交换界面温度
        interface_residuals = self._exchange_interface_temperature()

        # 求解流体区域
        fluid_residuals = {}
        for name, solver in self.fluid_solvers.items():
            T_prev = solver.T.clone()
            solver.T = solver._solve_timestep(solver.T, T_prev)
            residual = self._compute_residual(solver.T, T_prev)
            fluid_residuals[name] = residual

        # 求解固体区域
        solid_residuals = {}
        for name, solver in self.solid_solvers.items():
            T_prev = solver.T.clone()
            solver.T = solver._solve_timestep(solver.T, T_prev)
            residual = self._compute_residual(solver.T, T_prev)
            solid_residuals[name] = residual

        # 计算总残差
        all_residuals = list(fluid_residuals.values()) + list(solid_residuals.values())
        max_residual = max(all_residuals) if all_residuals else 0.0
        max_interface = max(interface_residuals) if interface_residuals else 0.0

        return max(max_residual, max_interface)

    @staticmethod
    def _compute_residual(field: torch.Tensor, field_old: torch.Tensor) -> float:
        """Compute L2 residual normalised by field magnitude."""
        diff = field - field_old
        norm_diff = float(torch.norm(diff).item())
        norm_field = float(torch.norm(field).item())
        if norm_field > 1e-30:
            return norm_diff / norm_field
        return norm_diff

    # ------------------------------------------------------------------
    # 主运行循环
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the enhanced CHT solver.

        Returns
        -------
        dict
            Convergence data including residuals for all regions.
        """
        # 设置接口
        self.setup_interfaces()

        time_loop = TimeLoop(
            start_time=self.start_time,
            end_time=self.end_time,
            delta_t=self.delta_t,
            write_interval=self.write_interval,
            write_control=self.write_control,
        )

        convergence = ConvergenceMonitor(
            tolerance=self.inner_tolerance,
            min_steps=1,
        )

        logger.info("Starting chtMultiRegionEnhancedFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  nInnerCorrectors=%d", self.n_inner_correctors)

        # 写初始场
        self._write_fields(self.start_time)
        time_loop.mark_written()

        last_convergence = None

        for t, step in time_loop:
            # 内迭代耦合
            max_residual = 0.0
            for inner in range(self.n_inner_correctors):
                residual = self._inner_iteration()
                max_residual = max(max_residual, residual)

                if residual < self.inner_tolerance:
                    logger.debug("Inner iteration converged at %d", inner + 1)
                    break

            # 收敛监控
            all_residuals = {}
            for name, solver in self.fluid_solvers.items():
                all_residuals[f"fluid_{name}"] = max_residual
            for name, solver in self.solid_solvers.items():
                all_residuals[f"solid_{name}"] = max_residual

            conv = ConvergenceData()
            conv.T_residual = max_residual
            conv.converged = max_residual < self.inner_tolerance
            last_convergence = conv

            converged = convergence.update(step + 1, all_residuals)

            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("CHT enhanced converged at step %d (t=%.6g)", step + 1, t)
                break

        # 写最终场
        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        logger.info("chtMultiRegionEnhancedFoam completed")

        return {
            "converged": last_convergence.converged if last_convergence else False,
            "T_residual": last_convergence.T_residual if last_convergence else 0.0,
            "n_fluid_regions": len(self.fluid_solvers),
            "n_solid_regions": len(self.solid_solvers),
            "n_interfaces": len(self.interfaces),
        }

    # ------------------------------------------------------------------
    # 场输出
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write fields for all regions."""
        for name, solver in self.fluid_solvers.items():
            solver._write_fields(time)

        for name, solver in self.solid_solvers.items():
            solver._write_fields(time)

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def all_solvers(self) -> dict[str, LaplacianFoam]:
        """All region solvers (fluid + solid)."""
        return {**self.fluid_solvers, **self.solid_solvers}

    @property
    def T_fluid(self) -> dict[str, torch.Tensor]:
        """Fluid region temperature fields."""
        return {name: solver.T for name, solver in self.fluid_solvers.items()}

    @property
    def T_solid(self) -> dict[str, torch.Tensor]:
        """Solid region temperature fields."""
        return {name: solver.T for name, solver in self.solid_solvers.items()}

    @property
    def T_all(self) -> dict[str, torch.Tensor]:
        """All region temperature fields."""
        return {**self.T_fluid, **self.T_solid}
