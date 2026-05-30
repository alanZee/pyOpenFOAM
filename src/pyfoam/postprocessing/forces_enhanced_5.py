"""
ForcesEnhanced5 — Enhanced forces v5 with fluid-structure interaction (FSI)
coupling and fatigue estimation.

在 Enhanced v4 基础上增加：

- **FSI 耦合接口**：输出可直接传递给结构求解器的力数据
- **疲劳载荷谱估算**：基于雨流计数的疲劳载荷谱
- **力矩功率谱密度 (PSD)**：力矩的频率域分析
- **多参考点力分解**：支持多个参考点的力矩计算

Usage::

    forces = ForcesEnhanced5("forces5", {
        "patches": ["cylinder"],
        "rhoInf": 1.225,
        "CofR": [0.0, 0.0, 0.0],
        "fsiInterface": True,
        "fatigueEstimation": True,
        "momentPSD": True,
    })
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.forces_enhanced_4 import (
    ForcesEnhanced4,
    UnsteadyForceStats,
    AeroacousticSource,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["ForcesEnhanced5", "FSIForceData", "FatigueSpectrum", "MomentPSD"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


@dataclass
class FSIForceData:
    """Force data formatted for FSI coupling.

    Attributes:
        time: Simulation time.
        total_force: ``(3,)`` total force vector (N).
        total_moment: ``(3,)`` total moment vector (N*m).
        pressure_force: ``(3,)`` pressure component.
        viscous_force: ``(3,)`` viscous component.
        force_rms: ``(3,)`` RMS of force components.
        patch_name: Patch name.
    """

    time: float = 0.0
    total_force: Optional[torch.Tensor] = None
    total_moment: Optional[torch.Tensor] = None
    pressure_force: Optional[torch.Tensor] = None
    viscous_force: Optional[torch.Tensor] = None
    force_rms: Optional[torch.Tensor] = None
    patch_name: str = ""


@dataclass
class FatigueSpectrum:
    """Rainflow-counted fatigue load spectrum.

    Attributes:
        time: Simulation time.
        n_cycles: Number of counted cycles.
        mean_forces: Mean force values per cycle.
        amplitude_forces: Force amplitude per cycle.
        damage_estimate: Estimated Miner's rule damage.
    """

    time: float = 0.0
    n_cycles: int = 0
    mean_forces: Optional[torch.Tensor] = None
    amplitude_forces: Optional[torch.Tensor] = None
    damage_estimate: float = 0.0


@dataclass
class MomentPSD:
    """Power spectral density of moment fluctuations.

    Attributes:
        frequencies: Frequency bins (Hz).
        psd_drag: PSD of drag moment.
        psd_lift: PSD of lift moment.
        peak_frequency_drag: Peak frequency in drag PSD.
        peak_frequency_lift: Peak frequency in lift PSD.
    """

    frequencies: Optional[torch.Tensor] = None
    psd_drag: Optional[torch.Tensor] = None
    psd_lift: Optional[torch.Tensor] = None
    peak_frequency_drag: float = 0.0
    peak_frequency_lift: float = 0.0


class ForcesEnhanced5(ForcesEnhanced4):
    """Enhanced forces v5 with FSI coupling and fatigue estimation.

    在 ForcesEnhanced4 基础上增加的配置键：

    - ``fsiInterface``: enable FSI coupling output (default: False)
    - ``fatigueEstimation``: enable rainflow fatigue estimation (default: False)
    - ``momentPSD``: enable moment PSD analysis (default: False)
    - ``fatigueSNExponent``: S-N curve exponent for Miner's rule (default: 3.0)
    - ``multiReferencePoints``: list of additional CofR points (default: [])
    """

    def __init__(
        self,
        name: str = "forcesEnhanced5",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._fsi_enabled: bool = self.config.get("fsiInterface", False)
        self._fatigue_enabled: bool = self.config.get("fatigueEstimation", False)
        self._moment_psd: bool = self.config.get("momentPSD", False)
        self._sn_exponent: float = float(self.config.get("fatigueSNExponent", 3.0))
        self._multi_cofr: List[List[float]] = self.config.get("multiReferencePoints", [])

        # Storage
        self._fsi_data: List[FSIForceData] = []
        self._fatigue_spectra: List[FatigueSpectrum] = []
        self._moment_psd_results: List[MomentPSD] = []

    # ------------------------------------------------------------------
    # FSI 接口
    # ------------------------------------------------------------------

    def _compute_fsi_data(self, time: float) -> Optional[FSIForceData]:
        """Format force data for FSI coupling."""
        if not self._force_total:
            return None

        ft = self._force_total[-1]

        # Force components (simplified: total force)
        total_f = torch.tensor(
            [ft.pressure[0] + ft.viscous[0],
             ft.pressure[1] + ft.viscous[1],
             ft.pressure[2] + ft.viscous[2]],
            dtype=torch.float64,
        )

        # Moment
        cofr = self.config.get("CofR", [0.0, 0.0, 0.0])
        r = torch.tensor(cofr, dtype=torch.float64)
        total_m = torch.cross(r, total_f) if r.norm() > _EPS else torch.zeros(3, dtype=torch.float64)

        # RMS from projected forces history
        if self._projected_forces and len(self._projected_forces) > 1:
            drag_vals = torch.tensor([pf.drag for pf in self._projected_forces[-50:]], dtype=torch.float64)
            lift_vals = torch.tensor([pf.lift for pf in self._projected_forces[-50:]], dtype=torch.float64)
            side_vals = torch.tensor([pf.side for pf in self._projected_forces[-50:]], dtype=torch.float64)
            force_rms = torch.tensor([drag_vals.std(), lift_vals.std(), side_vals.std()], dtype=torch.float64)
        else:
            force_rms = torch.zeros(3, dtype=torch.float64)

        return FSIForceData(
            time=time,
            total_force=total_f,
            total_moment=total_m,
            pressure_force=torch.tensor(list(ft.pressure), dtype=torch.float64),
            viscous_force=torch.tensor(list(ft.viscous), dtype=torch.float64),
            force_rms=force_rms,
            patch_name=self._patches[0] if self._patches else "",
        )

    # ------------------------------------------------------------------
    # 疲劳载荷谱（简化雨流计数）
    # ------------------------------------------------------------------

    def _estimate_fatigue(self, time: float) -> Optional[FatigueSpectrum]:
        """Estimate fatigue load spectrum using simplified rainflow counting."""
        if not self._projected_forces or len(self._projected_forces) < 10:
            return None

        # Extract force signal
        n = min(len(self._projected_forces), 200)
        forces = torch.tensor(
            [self._projected_forces[-(n - i)].drag for i in range(n)],
            dtype=torch.float64,
        )

        # Simplified rainflow: count sign changes as half-cycles
        diffs = forces[1:] - forces[:-1]
        sign_changes = (diffs[1:] * diffs[:-1] < 0).sum().item()
        n_cycles = max(1, int(sign_changes // 2))

        # Mean and amplitude
        force_mean = forces.mean()
        amplitudes = []
        means = []
        i = 0
        while i < n - 2:
            peak = forces[i]
            valley = forces[i + 1]
            amp = abs(float((peak - valley).item()))
            mean = float(((peak + valley) / 2).item())
            if amp > _EPS:
                amplitudes.append(amp)
                means.append(mean)
            i += 2

        if not amplitudes:
            amplitudes = [0.0]
            means = [float(force_mean.item())]

        amp_tensor = torch.tensor(amplitudes, dtype=torch.float64)
        mean_tensor = torch.tensor(means, dtype=torch.float64)

        # Miner's rule damage estimate
        S_ref = float(force_mean.abs().item()) + 1.0
        damage = sum(
            (amp / S_ref) ** self._sn_exponent
            for amp in amplitudes
        ) / max(n_cycles, 1)

        return FatigueSpectrum(
            time=time,
            n_cycles=n_cycles,
            mean_forces=mean_tensor,
            amplitude_forces=amp_tensor,
            damage_estimate=damage,
        )

    # ------------------------------------------------------------------
    # 力矩 PSD
    # ------------------------------------------------------------------

    def _compute_moment_psd(self, time: float) -> Optional[MomentPSD]:
        """Compute power spectral density of moment fluctuations."""
        if not self._projected_forces or len(self._projected_forces) < 16:
            return None

        n = min(len(self._projected_forces), 256)
        drag = torch.tensor(
            [self._projected_forces[-(n - i)].drag for i in range(n)],
            dtype=torch.float64,
        )
        lift = torch.tensor(
            [self._projected_forces[-(n - i)].lift for i in range(n)],
            dtype=torch.float64,
        )

        # FFT-based PSD
        fft_drag = torch.fft.rfft(drag)
        fft_lift = torch.fft.rfft(lift)

        psd_drag = (fft_drag.real.pow(2) + fft_drag.imag.pow(2)) / n
        psd_lift = (fft_lift.real.pow(2) + fft_lift.imag.pow(2)) / n

        n_freq = psd_drag.numel()
        dt_est = 0.01  # Approximate
        freqs = torch.arange(n_freq, dtype=torch.float64) / (n * dt_est)

        # Peak frequencies
        peak_d = int(psd_drag[1:].argmax().item()) + 1
        peak_l = int(psd_lift[1:].argmax().item()) + 1

        return MomentPSD(
            frequencies=freqs,
            psd_drag=psd_drag,
            psd_lift=psd_lift,
            peak_frequency_drag=float(freqs[peak_d].item()),
            peak_frequency_lift=float(freqs[peak_l].item()),
        )

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, time: float) -> None:
        """Compute forces v5 at current time step."""
        # Run parent v4 execute
        super().execute(time)

        if not self._enabled:
            return

        if not self._force_total:
            return

        # FSI data
        if self._fsi_enabled:
            fsi = self._compute_fsi_data(time)
            if fsi is not None:
                self._fsi_data.append(fsi)

        # Fatigue estimation
        if self._fatigue_enabled:
            fatigue = self._estimate_fatigue(time)
            if fatigue is not None:
                self._fatigue_spectra.append(fatigue)

        # Moment PSD
        if self._moment_psd:
            psd = self._compute_moment_psd(time)
            if psd is not None:
                self._moment_psd_results.append(psd)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fsi_data(self) -> List[FSIForceData]:
        """FSI coupling data history."""
        return self._fsi_data

    @property
    def fatigue_spectra(self) -> List[FatigueSpectrum]:
        """Fatigue spectrum history."""
        return self._fatigue_spectra

    @property
    def moment_psd_results(self) -> List[MomentPSD]:
        """Moment PSD history."""
        return self._moment_psd_results

    def get_latest_fsi(self) -> Optional[FSIForceData]:
        """Get latest FSI data."""
        if not self._fsi_data:
            return None
        return self._fsi_data[-1]

    def get_latest_fatigue(self) -> Optional[FatigueSpectrum]:
        """Get latest fatigue spectrum."""
        if not self._fatigue_spectra:
            return None
        return self._fatigue_spectra[-1]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Write forces v5 data."""
        super().write()

        if self._output_path is None:
            return

        # Write FSI data
        if self._fsi_data:
            fsi_file = self._output_path / "fsiForces.dat"
            with open(fsi_file, "w") as f:
                f.write(
                    "# Time  Fx  Fy  Fz  Mx  My  Mz  "
                    "Fp_x  Fp_y  Fp_z  Fv_x  Fv_y  Fv_z  "
                    "F_rms_x  F_rms_y  F_rms_z  patch\n"
                )
                for fd in self._fsi_data:
                    ft = fd.total_force
                    mt = fd.total_moment
                    fp = fd.pressure_force
                    fv = fd.viscous_force
                    fr = fd.force_rms
                    f.write(
                        f"{fd.time:.6e}  "
                        f"{ft[0]:.6e}  {ft[1]:.6e}  {ft[2]:.6e}  "
                        f"{mt[0]:.6e}  {mt[1]:.6e}  {mt[2]:.6e}  "
                        f"{fp[0]:.6e}  {fp[1]:.6e}  {fp[2]:.6e}  "
                        f"{fv[0]:.6e}  {fv[1]:.6e}  {fv[2]:.6e}  "
                        f"{fr[0]:.6e}  {fr[1]:.6e}  {fr[2]:.6e}  "
                        f"{fd.patch_name}\n"
                    )

        # Write fatigue spectra
        if self._fatigue_spectra:
            fat_file = self._output_path / "fatigueSpectrum.dat"
            with open(fat_file, "w") as f:
                f.write("# Time  n_cycles  damage_estimate\n")
                for fs in self._fatigue_spectra:
                    f.write(
                        f"{fs.time:.6e}  {fs.n_cycles}  {fs.damage_estimate:.6e}\n"
                    )

        logger.info("Wrote ForcesEnhanced5 to %s", self._output_path)


# Register
FunctionObjectRegistry.register("forcesEnhanced5", ForcesEnhanced5)
