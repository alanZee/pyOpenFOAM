"""
Noise — FFT-based acoustic noise analysis function object.

Computes sound pressure level (SPL) from pressure time-history data
using Fast Fourier Transform (FFT), mirroring OpenFOAM's ``noise``
function object.

Physics
-------
The sound pressure level in decibels is defined as:

    SPL = 20 * log10(p_rms / p_ref)

where p_ref = 2e-5 Pa (threshold of hearing in air).

The power spectral density (PSD) is computed from the FFT of the
pressure signal:

    PSD(f) = |FFT(p)|² / (N * Δt)

The overall SPL is computed by integrating the PSD over all frequencies:

    SPL_total = 10 * log10(∫ PSD(f) df / p_ref²)

References
----------
- OpenFOAM ``noise`` function object source
- ISO 226:2003 — Equal-loudness-level contours
- Beranek & Mellow, "Acoustics: Sound Fields and Transducers", 2012
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.function_object import FunctionObject, FunctionObjectRegistry

__all__ = ["Noise"]

logger = logging.getLogger(__name__)

# Reference pressure for SPL in air (Pa)
P_REF = 2e-5


class Noise(FunctionObject):
    """FFT-based acoustic noise analysis function object.

    Collects pressure samples at probe locations over time, then
    computes frequency spectra and sound pressure levels.

    Configuration keys:

    - ``fields``: list of field names (must include ``"p"``)
    - ``probeLocations``: list of ``(x, y, z)`` probe coordinates
    - ``pRef``: reference pressure for SPL (default: ``2e-5`` Pa)
    - ``windowFunction``: FFT window function (``"rectangular"``,
      ``"hanning"``, ``"hamming"``; default: ``"hanning"``)
    - ``nSamples``: expected number of samples (for pre-allocation)
    - ``writeSpectrum``: if True, write spectrum data to file (default: True)

    Example controlDict entry::

        noise1
        {
            type            noise;
            libs            ("libfieldFunctionObjects.so");
            fields          (p);
            probeLocations  ((0.5 0.5 0.5));
            pRef            2e-5;
            windowFunction  hanning;
            writeSpectrum   true;
        }
    """

    WINDOW_FUNCTIONS = {"rectangular", "hanning", "hamming"}

    def __init__(self, name: str = "noise", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config)
        self._field_name: str = self.config.get("fields", ["p"])[0] if isinstance(
            self.config.get("fields"), list
        ) else self.config.get("fields", "p")
        self._p_ref: float = float(self.config.get("pRef", P_REF))
        self._window_func: str = self.config.get("windowFunction", "hanning")
        self._write_spectrum: bool = self.config.get("writeSpectrum", True)
        self._probe_locations: List[List[float]] = self.config.get("probeLocations", [])

        if self._window_func not in self.WINDOW_FUNCTIONS:
            raise ValueError(
                f"Unknown window function '{self._window_func}'. "
                f"Available: {self.WINDOW_FUNCTIONS}"
            )

        # Time history: list of (time, pressure_values_per_probe)
        self._times: List[float] = []
        self._pressure_history: List[torch.Tensor] = []

        # Computed results (after finalise)
        self._frequencies: Optional[torch.Tensor] = None
        self._psd: Optional[torch.Tensor] = None
        self._spl: Optional[torch.Tensor] = None
        self._spl_total: Optional[torch.Tensor] = None

        # Map probe index to nearest cell
        self._probe_cells: List[int] = []

    def initialise(self, mesh, fields: Dict[str, Any]) -> None:
        """Store mesh, find nearest cells for probes."""
        self._mesh = mesh
        self._fields = fields

        # Map probe locations to nearest cells
        self._probe_cells = []
        if self._probe_locations and hasattr(mesh, "cell_centres"):
            cc = mesh.cell_centres.to(device=mesh.device, dtype=mesh.dtype)
            for loc in self._probe_locations:
                pt = torch.tensor(loc, dtype=mesh.dtype, device=mesh.device)
                dist = (cc - pt.unsqueeze(0)).norm(dim=1)
                self._probe_cells.append(dist.argmin().item())
        else:
            # Default: probe at cell 0
            self._probe_cells = [0]

        logger.info(
            "Noise '%s' initialised: %d probes, window=%s",
            self.name, len(self._probe_cells), self._window_func,
        )

    def execute(self, time: float) -> None:
        """Sample pressure at probe locations."""
        if not self._enabled or self._mesh is None:
            return

        p_field = self._fields.get(self._field_name)
        if p_field is None:
            logger.warning("Field '%s' not found. Skipping.", self._field_name)
            return

        # Extract pressure data
        device = get_device()
        dtype = get_default_dtype()

        if hasattr(p_field, "internal_field"):
            p_data = p_field.internal_field.to(device=device, dtype=dtype)
        elif hasattr(p_field, "data"):
            p_data = p_field.data.to(device=device, dtype=dtype)
        else:
            p_data = p_field.to(device=device, dtype=dtype)

        # Sample at probe cells
        p_sample = torch.tensor(
            [p_data[cell].item() if cell < len(p_data) else 0.0
             for cell in self._probe_cells],
            dtype=dtype, device=device,
        )

        self._times.append(time)
        self._pressure_history.append(p_sample.detach().cpu())

        self._log.info("t=%g  p_sample=%s", time, p_sample.tolist())

    def finalise(self) -> None:
        """Compute FFT, PSD, and SPL from collected pressure history."""
        if len(self._times) < 2:
            logger.warning("Not enough samples for FFT analysis (need >= 2)")
            return

        self._compute_spectra()

    def _compute_spectra(self) -> None:
        """Compute frequency spectra from pressure time history.

        Performs FFT on each probe's pressure signal, computes PSD and SPL.
        """
        dtype = get_default_dtype()
        n_samples = len(self._times)
        n_probes = len(self._probe_cells)

        # Stack pressure history: (n_samples, n_probes)
        p_stack = torch.stack(self._pressure_history, dim=0).to(dtype=dtype)

        # Time step (assumed uniform)
        dt = (self._times[-1] - self._times[0]) / max(n_samples - 1, 1)

        # Remove mean (DC component)
        p_mean = p_stack.mean(dim=0, keepdim=True)
        p_fluct = p_stack - p_mean

        # Apply window function
        window = self._create_window(n_samples, dtype)
        p_windowed = p_fluct * window.unsqueeze(1)

        # FFT
        # Use real-valued FFT (rfft)
        n_fft = n_samples
        p_fft = torch.fft.rfft(p_windowed, n=n_fft, dim=0)

        # Frequency vector
        freqs = torch.fft.rfftfreq(n_fft, d=dt)

        # Power spectral density
        # PSD = 2 * |FFT|² / (N² * dt)  (factor 2 for one-sided spectrum)
        psd = 2.0 * (p_fft.real ** 2 + p_fft.imag ** 2) / (n_fft ** 2 * dt)
        # Don't double-count DC and Nyquist
        psd[0, :] /= 2.0
        if n_fft % 2 == 0:
            psd[-1, :] /= 2.0

        # SPL per frequency bin: SPL(f) = 10 * log10(PSD(f) / p_ref²)
        p_ref_sq = self._p_ref ** 2
        spl_per_freq = 10.0 * torch.log10(
            psd.clamp(min=1e-40) / p_ref_sq
        )

        # Overall SPL: integrate PSD over frequency
        # SPL_total = 10 * log10(∫ PSD df / p_ref²)
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        total_power = psd.sum(dim=0) * df
        spl_total = 10.0 * torch.log10(total_power.clamp(min=1e-40) / p_ref_sq)

        self._frequencies = freqs.detach().cpu()
        self._psd = psd.detach().cpu()
        self._spl = spl_per_freq.detach().cpu()
        self._spl_total = spl_total.detach().cpu()

        logger.info("Noise analysis complete: %d samples, %d probes", n_samples, n_probes)
        for i in range(n_probes):
            logger.info("  Probe %d: SPL_total = %.2f dB", i, spl_total[i].item())

    def _create_window(self, n: int, dtype: torch.dtype) -> torch.Tensor:
        """Create FFT window function.

        Parameters
        ----------
        n : int
            Number of samples.
        dtype : torch.dtype
            Tensor dtype.

        Returns
        -------
        torch.Tensor
            ``(n,)`` window coefficients.
        """
        if self._window_func == "rectangular":
            return torch.ones(n, dtype=dtype)
        elif self._window_func == "hanning":
            k = torch.arange(n, dtype=dtype)
            return 0.5 * (1.0 - torch.cos(2.0 * math.pi * k / max(n - 1, 1)))
        elif self._window_func == "hamming":
            k = torch.arange(n, dtype=dtype)
            return 0.54 - 0.46 * torch.cos(2.0 * math.pi * k / max(n - 1, 1))
        else:
            return torch.ones(n, dtype=dtype)

    def write(self) -> None:
        """Write spectrum and SPL data to output files."""
        if self._output_path is None or self._frequencies is None:
            return

        n_probes = len(self._probe_cells)

        if self._write_spectrum:
            for i in range(n_probes):
                spec_file = self._output_path / f"spectrum_probe{i}.dat"
                with open(spec_file, "w") as f:
                    f.write("# Frequency(Hz)  PSD(Pa²/Hz)  SPL(dB)\n")
                    for j in range(len(self._frequencies)):
                        f.write(
                            f"{self._frequencies[j]:.6e}  "
                            f"{self._psd[j, i]:.6e}  "
                            f"{self._spl[j, i]:.6e}\n"
                        )
                logger.info("Wrote spectrum to %s", spec_file)

        # Write overall SPL
        spl_file = self._output_path / "spl_overall.dat"
        with open(spl_file, "w") as f:
            f.write("# Probe  SPL_total(dB)\n")
            for i in range(n_probes):
                f.write(f"{i}  {self._spl_total[i]:.6e}\n")
        logger.info("Wrote overall SPL to %s", spl_file)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def frequencies(self) -> Optional[torch.Tensor]:
        """Frequency vector (Hz). Available after :meth:`finalise`."""
        return self._frequencies

    @property
    def psd(self) -> Optional[torch.Tensor]:
        """Power spectral density (Pa^2/Hz). Shape ``(n_freq, n_probes)``."""
        return self._psd

    @property
    def spl(self) -> Optional[torch.Tensor]:
        """SPL per frequency bin (dB). Shape ``(n_freq, n_probes)``."""
        return self._spl

    @property
    def spl_total(self) -> Optional[torch.Tensor]:
        """Overall SPL per probe (dB). Shape ``(n_probes,)``."""
        return self._spl_total

    @property
    def times(self) -> List[float]:
        """Sample times."""
        return self._times

    @property
    def pressure_history(self) -> List[torch.Tensor]:
        """Pressure history (list of per-probe tensors)."""
        return self._pressure_history

    @staticmethod
    def compute_spl_from_signal(
        p: torch.Tensor,
        p_ref: float = P_REF,
    ) -> float:
        """Compute overall SPL from a pressure signal.

        Convenience static method for one-shot SPL computation.

        Parameters
        ----------
        p : torch.Tensor
            ``(n,)`` pressure time series (Pa).
        p_ref : float
            Reference pressure (Pa).

        Returns
        -------
        float
            Sound pressure level in dB.
        """
        p_rms = torch.sqrt((p ** 2).mean())
        spl = 20.0 * torch.log10(p_rms / p_ref + 1e-40)
        return float(spl.item())


# Register
FunctionObjectRegistry.register("noise", Noise)
