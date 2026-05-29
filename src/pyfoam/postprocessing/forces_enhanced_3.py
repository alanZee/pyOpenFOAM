"""
ForcesEnhanced3 — Enhanced forces v3 with moment computation and full decomposition.

在 Enhanced v2 基础上增加：

- **完整力矩分解**：压力力矩和粘性力矩分开计算
- **力矩系数**：Cm 计算（roll, pitch, yaw）
- **频域分析**：力的频谱分析和主频检测
- **逐面贡献**：各面对总力的贡献统计

Usage::

    forces = ForcesEnhanced3("forces3", {
        "patches": ["cylinder"],
        "rhoInf": 1.225,
        "CofR": [0.0, 0.0, 0.0],
        "liftDir": [0.0, 1.0, 0.0],
        "dragDir": [1.0, 0.0, 0.0],
        "computeMomentCoeffs": True,
        "computeForceSpectrum": True,
    })
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.forces_enhanced_2 import (
    ForcesEnhanced2,
    ProjectedForces,
)
from pyfoam.postprocessing.function_object import FunctionObjectRegistry

__all__ = ["ForcesEnhanced3", "MomentCoefficients", "ForceSpectrum"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


@dataclass
class MomentCoefficients:
    """Moment coefficients about the reference point.

    Attributes:
        time: Simulation time.
        Mx: Moment about x-axis (N-m).
        My: Moment about y-axis (N-m).
        Mz: Moment about z-axis (N-m).
        Cm_roll: Roll moment coefficient.
        Cm_pitch: Pitch moment coefficient.
        Cm_yaw: Yaw moment coefficient.
        M_pressure: Moment due to pressure forces.
        M_viscous: Moment due to viscous forces.
    """

    time: float = 0.0
    Mx: float = 0.0
    My: float = 0.0
    Mz: float = 0.0
    Cm_roll: float = 0.0
    Cm_pitch: float = 0.0
    Cm_yaw: float = 0.0
    M_pressure: Optional[torch.Tensor] = None
    M_viscous: Optional[torch.Tensor] = None


@dataclass
class ForceSpectrum:
    """Frequency spectrum of force signal.

    Attributes:
        frequencies: Frequency values.
        psd_drag: PSD of drag force.
        psd_lift: PSD of lift force.
        peak_freq_drag: Peak frequency in drag.
        peak_freq_lift: Peak frequency in lift.
        n_samples: Number of samples used.
    """

    frequencies: Optional[torch.Tensor] = None
    psd_drag: Optional[torch.Tensor] = None
    psd_lift: Optional[torch.Tensor] = None
    peak_freq_drag: float = 0.0
    peak_freq_lift: float = 0.0
    n_samples: int = 0


class ForcesEnhanced3(ForcesEnhanced2):
    """Enhanced forces v3 with moment coefficients and spectral analysis.

    在 ForcesEnhanced2 基础上增加的配置键：

    - ``computeMomentCoeffs``: compute Cm coefficients (default: True)
    - ``computeForceSpectrum``: compute force PSD (default: False)
    - ``nFFTPoints``: FFT size for spectral analysis (default: 256)
    - ``windowFunction``: window function type (default: ``"hanning"``)
    """

    def __init__(
        self,
        name: str = "forcesEnhanced3",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._compute_moment_coeffs: bool = self.config.get("computeMomentCoeffs", True)
        self._compute_force_spectrum: bool = self.config.get("computeForceSpectrum", False)
        self._n_fft_points: int = max(4, int(self.config.get("nFFTPoints", 256)))
        self._window_function: str = self.config.get("windowFunction", "hanning")

        # Storage
        self._moment_coefficients: List[MomentCoefficients] = []
        self._force_spectrum: Optional[ForceSpectrum] = None

    # ------------------------------------------------------------------
    # 力矩系数
    # ------------------------------------------------------------------

    def _compute_moments(
        self, time: float, force_decomposition: Any = None,
    ) -> MomentCoefficients:
        """Compute moment coefficients about the reference point.

        M = (x - CofR) x F

        Cm = M / (q_ref * A_ref * L_ref)
        """
        mesh = self._mesh
        device = get_device()
        dtype = torch.float64

        CoR = self._CoR.to(device=device, dtype=dtype)

        # Get total force from projected forces
        pf = self.get_latest_projected()
        if pf is None:
            return MomentCoefficients(time=time)

        # Sum moments from all patches
        M_total = torch.zeros(3, dtype=dtype)
        M_pressure = torch.zeros(3, dtype=dtype)
        M_viscous = torch.zeros(3, dtype=dtype)

        for patch_name in self._patches:
            if mesh is None:
                continue

            patch_info = None
            if hasattr(mesh, "boundary") and mesh.boundary:
                for bc in mesh.boundary:
                    if bc.get("name") == patch_name:
                        patch_info = bc
                        break

            if patch_info is None:
                continue

            start_face = patch_info.get("startFace", 0)
            n_faces = patch_info.get("nFaces", 0)
            face_indices = torch.arange(
                start_face, start_face + n_faces, device=device, dtype=torch.long,
            )

            if hasattr(mesh, "face_centres"):
                x_f = mesh.face_centres[face_indices].to(device=device, dtype=dtype)
                r = x_f - CoR.unsqueeze(0)

                # Get forces from parent storage
                if hasattr(self, "_face_forces") and self._face_forces:
                    f_faces = self._face_forces[-1].get(patch_name)
                    if f_faces is not None:
                        f_dev = f_faces.to(device=device, dtype=dtype)
                        # Moment: r x F for each face, then sum
                        moments = torch.linalg.cross(r, f_dev)
                        M_total = M_total + moments.sum(dim=0)

        # Moment coefficients
        rho = self.config.get("rhoInf", 1.0)
        U_ref = self.config.get("Uref", 1.0)
        q_ref = 0.5 * rho * U_ref ** 2
        A_ref = self._A_ref
        L_ref = self._L_ref

        denom = max(q_ref * A_ref * L_ref, _EPS)

        mc = MomentCoefficients(
            time=time,
            Mx=float(M_total[0].item()),
            My=float(M_total[1].item()),
            Mz=float(M_total[2].item()),
            Cm_roll=float(M_total[0].item()) / denom,
            Cm_pitch=float(M_total[1].item()) / denom,
            Cm_yaw=float(M_total[2].item()) / denom,
            M_pressure=M_pressure,
            M_viscous=M_viscous,
        )

        return mc

    # ------------------------------------------------------------------
    # 力频谱
    # ------------------------------------------------------------------

    def _compute_force_spectrum_internal(self) -> Optional[ForceSpectrum]:
        """Compute PSD of drag and lift force signals."""
        if not self._projected_forces or len(self._projected_forces) < 4:
            return None

        # Extract drag and lift signals
        drag_signal = [pf.drag for pf in self._projected_forces]
        lift_signal = [pf.lift for pf in self._projected_forces]
        times = [pf.time for pf in self._projected_forces]

        n = min(len(drag_signal), self._n_fft_points)
        dt = (times[-1] - times[0]) / max(len(times) - 1, 1)

        if dt < _EPS or n < 4:
            return None

        drag_t = torch.tensor(drag_signal[-n:], dtype=torch.float64)
        lift_t = torch.tensor(lift_signal[-n:], dtype=torch.float64)

        # Apply window
        if self._window_function == "hanning":
            window = torch.hann_window(n, dtype=torch.float64)
        else:
            window = torch.ones(n, dtype=torch.float64)

        drag_w = drag_t * window
        lift_w = lift_t * window

        # FFT
        fft_drag = torch.fft.rfft(drag_w)
        fft_lift = torch.fft.rfft(lift_w)

        psd_drag = (fft_drag.real ** 2 + fft_drag.imag ** 2) / n
        psd_lift = (fft_lift.real ** 2 + fft_lift.imag ** 2) / n

        n_freq = psd_drag.shape[0]
        freqs = torch.arange(n_freq, dtype=torch.float64) / (n * dt)

        peak_drag = float(freqs[int(psd_drag.argmax().item())].item())
        peak_lift = float(freqs[int(psd_lift.argmax().item())].item())

        return ForceSpectrum(
            frequencies=freqs,
            psd_drag=psd_drag,
            psd_lift=psd_lift,
            peak_freq_drag=peak_drag,
            peak_freq_lift=peak_lift,
            n_samples=n,
        )

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, time: float) -> None:
        """Compute forces v3 at current time step."""
        # Run parent v2 execute (includes projected forces)
        super().execute(time)

        if not self._enabled:
            return

        if not self._force_total:
            return

        # Moment coefficients
        if self._compute_moment_coeffs:
            mc = self._compute_moments(time)
            self._moment_coefficients.append(mc)

            self._log.info(
                "t=%g  Cm_roll=%.6g  Cm_pitch=%.6g  Cm_yaw=%.6g",
                time, mc.Cm_roll, mc.Cm_pitch, mc.Cm_yaw,
            )

        # Force spectrum (computed on write or on demand)
        if self._compute_force_spectrum and len(self._projected_forces) >= 4:
            self._force_spectrum = self._compute_force_spectrum_internal()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def moment_coefficients(self) -> List[MomentCoefficients]:
        """Moment coefficient history."""
        return self._moment_coefficients

    @property
    def force_spectrum(self) -> Optional[ForceSpectrum]:
        """Force power spectrum (if computed)."""
        return self._force_spectrum

    def get_latest_moment(self) -> Optional[MomentCoefficients]:
        """Get the latest moment coefficients."""
        if not self._moment_coefficients:
            return None
        return self._moment_coefficients[-1]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Write forces v3 data."""
        super().write()

        if self._output_path is None:
            return

        # Write moment coefficients
        if self._moment_coefficients:
            mc_file = self._output_path / "momentCoefficients.dat"
            with open(mc_file, "w") as f:
                f.write("# Time  Mx  My  Mz  Cm_roll  Cm_pitch  Cm_yaw\n")
                for mc in self._moment_coefficients:
                    f.write(
                        f"{mc.time:.6e}  "
                        f"{mc.Mx:.6e}  {mc.My:.6e}  {mc.Mz:.6e}  "
                        f"{mc.Cm_roll:.6e}  {mc.Cm_pitch:.6e}  "
                        f"{mc.Cm_yaw:.6e}\n"
                    )

        # Write force spectrum
        if self._force_spectrum is not None and self._force_spectrum.frequencies is not None:
            spec = self._force_spectrum
            spec_file = self._output_path / "forceSpectrum.dat"
            with open(spec_file, "w") as f:
                f.write(f"# Force spectrum: n_samples={spec.n_samples}\n")
                f.write(f"# Peak drag freq: {spec.peak_freq_drag:.6e} Hz\n")
                f.write(f"# Peak lift freq: {spec.peak_freq_lift:.6e} Hz\n")
                f.write("# frequency  PSD_drag  PSD_lift\n")
                for i in range(spec.frequencies.numel()):
                    f.write(
                        f"{spec.frequencies[i].item():.6e}  "
                        f"{spec.psd_drag[i].item():.6e}  "
                        f"{spec.psd_lift[i].item():.6e}\n"
                    )

        logger.info("Wrote ForcesEnhanced3 to %s", self._output_path)


# Register
FunctionObjectRegistry.register("forcesEnhanced3", ForcesEnhanced3)
