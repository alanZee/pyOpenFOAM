"""
chemFoam — 0D chemistry solver.

Solves species and temperature ODEs in a single-cell (0D) reactor
using explicit Euler forward integration.  No mesh flow equations,
pure chemistry:

    dY_i/dt = ω_i(Y, T)
    dT/dt   = -Σ(h_i · ω_i) / Cp

where ω_i is the Arrhenius reaction source term:

    ω_i = ν_i · A · T^β · exp(-Ea / (R·T)) · ∏[Y_j^ν_j]

Reads:
- ``0/YA``, ``0/YB``, ... — initial mass fractions
- ``0/T`` — initial temperature
- ``constant/thermophysicalProperties`` — R, Cp, species molecular weights
- ``constant/reactions`` — reaction mechanism
- ``system/controlDict`` — endTime, deltaT

Usage::

    from pyfoam.applications.chem_foam import ChemFoam

    solver = ChemFoam("path/to/case")
    solver.run()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["ChemFoam", "ChemReaction"]

logger = logging.getLogger(__name__)


@dataclass
class ChemReaction:
    """A single chemical reaction with Arrhenius kinetics.

    Attributes
    ----------
    name : str
        Reaction name.
    A : float
        Pre-exponential factor.
    beta : float
        Temperature exponent.
    Ea : float
        Activation energy (J/mol).
    reactants : dict[str, float]
        Stoichiometric coefficients for reactants.
    products : dict[str, float]
        Stoichiometric coefficients for products.
    """

    name: str = ""
    A: float = 1.0
    beta: float = 0.0
    Ea: float = 0.0
    reactants: dict[str, float] = field(default_factory=dict)
    products: dict[str, float] = field(default_factory=dict)


class ChemFoam(SolverBase):
    """0D chemistry solver with Arrhenius kinetics.

    Solves species mass-fraction and temperature ODEs in a single cell
    using forward Euler integration.  No spatial transport, no mesh
    flow equations — pure chemical kinetics.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.

    Attributes
    ----------
    Y : dict[str, torch.Tensor]
        Mass fractions for each species.
    T : torch.Tensor
        ``(n_cells,)`` temperature field.
    species : list[str]
        Species names.
    reactions : list[ChemReaction]
        Reaction mechanism.
    R : float
        Universal gas constant (J/(mol·K)).
    Cp : float
        Specific heat capacity (J/(kg·K)).
    """

    def __init__(self, case_path: Union[str, Path]) -> None:
        super().__init__(case_path)

        # Thermodynamic properties
        self._read_thermo_properties()

        # Reaction mechanism
        self.reactions = self._read_reactions()

        # fvSolution settings
        self._read_fv_solution_settings()

        # Initialise fields
        self.Y, self.T = self._init_fields()

        # Store raw field data for writing
        self._field_data = self._init_field_data()

        logger.info(
            "ChemFoam ready: %d species, %d reactions",
            len(self.species),
            len(self.reactions),
        )

    # ------------------------------------------------------------------
    # Property reading
    # ------------------------------------------------------------------

    def _read_thermo_properties(self) -> None:
        """Read thermodynamic properties from constant/thermophysicalProperties."""
        self.R = 8.314       # 通用气体常数 (J/(mol·K))
        self.Cp = 1005.0     # 比热容 (J/(kg·K))
        self.W: dict[str, float] = {}   # 各组分分子量

        tp_path = self.case_path / "constant" / "thermophysicalProperties"
        if tp_path.exists():
            try:
                from pyfoam.io.dictionary import parse_dict_file

                tp = parse_dict_file(tp_path)
                self.R = float(tp.get("R", 8.314))
                self.Cp = float(tp.get("Cp", 1005.0))

                species_dict = tp.get("species", {})
                if isinstance(species_dict, dict):
                    for name, value in species_dict.items():
                        self.W[name] = float(value)
            except Exception as e:
                logger.warning("Could not read thermo properties: %s", e)

    def _read_reactions(self) -> list[ChemReaction]:
        """Read reaction mechanism from constant/reactions."""
        reactions: list[ChemReaction] = []

        rxn_path = self.case_path / "constant" / "reactions"
        if not rxn_path.exists():
            logger.warning("No reactions file found, using default A→B reaction")
            reactions.append(ChemReaction(
                name="reaction1", A=1.0, beta=0.0, Ea=0.0,
                reactants={"A": 1.0}, products={"B": 1.0},
            ))
            return reactions

        try:
            from pyfoam.io.dictionary import parse_dict_file

            rxn_dict = parse_dict_file(rxn_path)

            for key, value in rxn_dict.items():
                if isinstance(value, dict):
                    rxn = ChemReaction(name=key)
                    rxn.A = float(value.get("A", 1.0))
                    rxn.beta = float(value.get("beta", 0.0))
                    rxn.Ea = float(value.get("Ea", 0.0))

                    reactants = value.get("reactants", {})
                    if isinstance(reactants, dict):
                        rxn.reactants = {k: float(v) for k, v in reactants.items()}

                    products = value.get("products", {})
                    if isinstance(products, dict):
                        rxn.products = {k: float(v) for k, v in products.items()}

                    reactions.append(rxn)
        except Exception as e:
            logger.warning("Could not read reactions: %s", e)
            reactions.append(ChemReaction(
                name="default", A=1.0, beta=0.0, Ea=0.0,
                reactants={"A": 1.0}, products={"B": 1.0},
            ))

        return reactions

    def _read_fv_solution_settings(self) -> None:
        """Read convergence tolerance from fvSolution."""
        fv = self.case.fvSolution
        self.convergence_tolerance = float(
            fv.get_path("chemFoam/convergenceTolerance", 1e-4)
        )

    # ------------------------------------------------------------------
    # Field initialisation
    # ------------------------------------------------------------------

    def _init_fields(self) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Initialise species mass fractions and temperature from ``0/``.

        Returns:
            Tuple of ``(Y, T)``.
        """
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        # 读取组分质量分数：扫描 0/ 目录中 Y 前缀的场
        Y: dict[str, torch.Tensor] = {}
        self.species: list[str] = []

        zero_dir = self.case_path / "0"
        if zero_dir.exists():
            for f in sorted(zero_dir.iterdir()):
                if f.name.startswith("Y"):
                    species_name = f.name[1:]
                    if species_name:
                        self.species.append(species_name)
                        try:
                            y_tensor, _ = self.read_field_tensor(f.name, 0)
                            Y[species_name] = (
                                y_tensor.to(device=device, dtype=dtype).reshape(-1)
                            )
                        except Exception:
                            Y[species_name] = torch.zeros(
                                n_cells, dtype=dtype, device=device
                            )

        # 未检测到组分时创建默认 A、B
        if not self.species:
            self.species = ["A", "B"]
            Y["A"] = torch.ones(n_cells, dtype=dtype, device=device)
            Y["B"] = torch.zeros(n_cells, dtype=dtype, device=device)

        # 读取温度
        try:
            T_tensor, _ = self.read_field_tensor("T", 0)
            T = T_tensor.to(device=device, dtype=dtype).reshape(-1)
        except Exception:
            T = torch.full((n_cells,), 300.0, dtype=dtype, device=device)

        return Y, T

    def _init_field_data(self) -> dict[str, Any]:
        """Store raw FieldData for writing."""
        field_data: dict[str, Any] = {}

        for species_name in self.species:
            fname = f"Y{species_name}"
            try:
                field_data[fname] = self.case.read_field(fname, 0)
            except Exception:
                field_data[fname] = None

        try:
            field_data["T"] = self.case.read_field("T", 0)
        except Exception:
            field_data["T"] = None

        return field_data

    # ------------------------------------------------------------------
    # Chemistry source terms
    # ------------------------------------------------------------------

    def _compute_arrhenius_rate(
        self,
        reaction: ChemReaction,
        T: torch.Tensor,
        Y: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute Arrhenius reaction rate.

        k = A · T^β · exp(-Ea / (R·T))
        Rate = k · ∏[Y_j^ν_j]
        """
        T_safe = T.clamp(min=1.0)

        k = reaction.A * T_safe.pow(reaction.beta) * torch.exp(
            -reaction.Ea / (self.R * T_safe)
        )

        conc_term = torch.ones_like(T)
        for species, nu in reaction.reactants.items():
            if species in Y:
                conc_term = conc_term * Y[species].clamp(min=0.0).pow(nu)

        return k * conc_term

    def _compute_species_source(
        self,
        T: torch.Tensor,
        Y: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute species source terms ω_i for all species.

        Returns:
            ``{species_name: (n_cells,)}`` source terms.
        """
        dtype = T.dtype
        device = T.device
        n_cells = T.shape[0]

        omega = {
            name: torch.zeros(n_cells, dtype=dtype, device=device)
            for name in self.species
        }

        for reaction in self.reactions:
            rate = self._compute_arrhenius_rate(reaction, T, Y)

            # 反应物：消耗（负源项）
            for species, nu in reaction.reactants.items():
                if species in omega:
                    W_i = self.W.get(species, 1.0)
                    omega[species] = omega[species] - nu * W_i * rate

            # 生成物：产生（正源项）
            for species, nu in reaction.products.items():
                if species in omega:
                    W_i = self.W.get(species, 1.0)
                    omega[species] = omega[species] + nu * W_i * rate

        return omega

    def _compute_temperature_source(
        self,
        T: torch.Tensor,
        Y: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute temperature source term.

        dT/dt = Q / Cp, where Q = -Σ(ω_i · h_i) is the heat release.

        For simplicity, uses a constant enthalpy of formation per species
        (defaulting to zero).  The net heat release from each reaction is
        ΔH_rxn = Σ(prod · h) - Σ(react · h).

        Returns:
            ``(n_cells,)`` temperature rate of change.
        """
        dtype = T.dtype
        device = T.device
        n_cells = T.shape[0]

        heat_release = torch.zeros(n_cells, dtype=dtype, device=device)

        for reaction in self.reactions:
            rate = self._compute_arrhenius_rate(reaction, T, Y)

            # ΔH_rxn = Σ_products(ν_j · h_j) - Σ_reactants(ν_j · h_j)
            # 默认 h = 0（若未提供生成焓数据），结果为 0。
            # 用户可通过 reactions 文件扩展 h_f 字段。
            delta_H = 0.0
            for species, nu in reaction.products.items():
                h_f = getattr(reaction, "h_f_products", {}).get(species, 0.0)
                delta_H += nu * h_f
            for species, nu in reaction.reactants.items():
                h_f = getattr(reaction, "h_f_reactants", {}).get(species, 0.0)
                delta_H -= nu * h_f

            heat_release = heat_release + delta_H * rate

        return heat_release / self.Cp

    # ------------------------------------------------------------------
    # Main run loop (forward Euler)
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the chemFoam solver.

        Uses forward Euler integration for species and temperature ODEs.

        Returns:
            Dictionary with convergence info: ``converged``, ``steps``,
            ``residual``.
        """
        device = get_device()
        dtype = get_default_dtype()

        time_loop = TimeLoop(
            start_time=self.start_time,
            end_time=self.end_time,
            delta_t=self.delta_t,
            write_interval=self.write_interval,
            write_control=self.write_control,
        )

        convergence = ConvergenceMonitor(
            tolerance=self.convergence_tolerance,
            min_steps=1,
        )

        logger.info("Starting chemFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  Species: %s", self.species)

        # 写入初始场
        self._write_fields(self.start_time)
        time_loop.mark_written()

        converged = False

        for t, step in time_loop:
            # 保存旧值
            Y_old = {name: y.clone() for name, y in self.Y.items()}
            T_old = self.T.clone()

            # 计算化学源项
            omega = self._compute_species_source(self.T, self.Y)
            T_source = self._compute_temperature_source(self.T, self.Y)

            # 前向 Euler 更新组分
            for name in self.species:
                dYdt = omega[name] / self.rho_cell
                self.Y[name] = Y_old[name] + self.delta_t * dYdt

            # 前向 Euler 更新温度
            self.T = T_old + self.delta_t * T_source

            # 计算残差（最大变化量）
            residual = 0.0
            for name in self.species:
                change = float((self.Y[name] - Y_old[name]).abs().max().item())
                residual = max(residual, change)
            T_change = float((self.T - T_old).abs().max().item())
            residual = max(residual, T_change / max(float(self.T.mean().item()), 1.0))

            converged = convergence.update(step + 1, {"chem": residual})

            # 按写入间隔输出场
            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("Converged at step %d (t=%.6g)", step + 1, t)
                break

        # 写入最终场
        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        logger.info("chemFoam completed")
        logger.info(
            "  T range: [%.1f, %.1f] K",
            self.T.min().item(),
            self.T.max().item(),
        )

        return {
            "converged": converged,
            "steps": time_loop.step + 1,
            "residual": residual,
        }

    # ------------------------------------------------------------------
    # 密度（单胞近似：ρ = p / (R_specific · T)）
    # ------------------------------------------------------------------

    @property
    def rho_cell(self) -> torch.Tensor:
        """单胞密度 (kg/m³)，假设标准大气压。

        ρ = p / (R_specific · T)，其中 R_specific = R / W_avg。
        """
        p_ref = 101325.0  # 标准大气压 (Pa)
        # 平均分子量（未指定时默认为 29，空气近似）
        W_avg = 29.0
        R_specific = self.R / W_avg
        T_safe = self.T.clamp(min=1.0)
        return (p_ref / (R_specific * T_safe)).to(dtype=self.T.dtype, device=self.T.device)

    # ------------------------------------------------------------------
    # 场输出
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write species and temperature fields to a time directory."""
        time_str = f"{time:g}"

        # 写入组分质量分数
        for species_name in self.species:
            fname = f"Y{species_name}"
            if fname in self._field_data and self._field_data[fname] is not None:
                self.write_field(
                    fname, self.Y[species_name], time_str, self._field_data[fname]
                )

        # 写入温度
        if "T" in self._field_data and self._field_data["T"] is not None:
            self.write_field("T", self.T, time_str, self._field_data["T"])
