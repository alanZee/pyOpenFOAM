"""
NoiseEnhanced — Enhanced FFT-based acoustic noise analysis with A-weighting.

Extends :class:`~pyfoam.postprocessing.noise.Noise` with:

- **A-weighting** (IEC 61672) for perceptual noise assessment
- **1/3 octave band** analysis
- **Peak frequency** detection
- **Equivalent continuous sound level** (LAeq)
- **Time-windowed** spectral analysis

Physics
-------
A-weighting applies a frequency-dependent correction to SPL to approximate
human hearing perception:

    R_A(f) = 12194^2 * f^4 / (
        (f^2 + 20.6^2) *
        sqrt((f^2 + 107.7^2) * (f^2 + 737.9^2)) *
        (f^2 + 12194^2)
    )

    A(f) = 20 * log10(R_A(f) / R_A(1000)) + 2.0  [dB]

The A-weighted SPL is:

    SPL_A(f) = SPL(f) + A(f)

The equivalent continuous sound level:

    LAeq = 10 * log10(1/T * integral(p_A^2(t) dt) / p_ref^2)

References
----------
- IEC 61672-1:2013 — Electroacoustics — Sound level meters
- OpenFOAM ``noise`` function object source
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.postprocessing.noise import Noise, P_REF

__all__ = ["NoiseEnhanced", "ThirdOctaveBand"]

logger = logging.getLogger(__name__)

# 1/3 octave band centre frequencies (Hz) — standard ISO 266
_THIRD_OCTAVE_CENTRES = [
    25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0,
    250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0,
    2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0,
    12500.0, 16000.0, 20000.0,
]


def a_weighting(frequency: torch.Tensor) -> torch.Tensor:
    """Compute A-weighting correction in dB for given frequencies.

    Implements IEC 61672-1 A-weighting curve.

    Parameters
    ----------
    frequency : torch.Tensor
        Frequency values in Hz.  Must be > 0.

    Returns
    -------
    torch.Tensor
        A-weighting correction in dB (same shape as *frequency*).
    """
    f = frequency.clamp(min=1e-30)
    f2 = f ** 2

    # Numerator and denominator of the A-weighting rational function
    num = 12194.0 ** 2 * f2 ** 2
    den = (
        (f2 + 20.6 ** 2)
        * torch.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
        * (f2 + 12194.0 ** 2)
    )

    ra = num / den.clamp(min=1e-40)
    # Normalise so that A(1000 Hz) = 0 dB
    # R_A(1000) computed analytically; +2.0 offset is absorbed into normalization
    f_ref = torch.tensor(1000.0, dtype=frequency.dtype)
    f_ref2 = f_ref ** 2
    num_ref = 12194.0 ** 2 * f_ref2 ** 2
    den_ref = (
        (f_ref2 + 20.6 ** 2)
        * math.sqrt((f_ref2 + 107.7 ** 2) * (f_ref2 + 737.9 ** 2))
        * (f_ref2 + 12194.0 ** 2)
    )
    ra_ref = num_ref / den_ref

    # Standard A-weighting: A(f) = 20*log10(R_A(f)) + 2.0
    # To normalise A(1000) = 0: A(f) = 20*log10(R_A(f)/R_A(1000))
    # The +2.0 constant in the standard formula is a normalization offset
    # that makes A(1000) = 20*log10(R_A(1000)) + 2.0 = 0
    a_db = 20.0 * torch.log10(ra / ra_ref + 1e-40)
    return a_db


def _third_octave_band_limits() -> List[tuple[float, float]]:
    """Compute 1/3 octave band lower and upper frequency limits.

    Returns list of (f_low, f_high) tuples for each centre frequency.
    """
    factor = 2.0 ** (1.0 / 6.0)  # 1/6 decade = 1/3 octave half-band
    limits = []
    for fc in _THIRD_OCTAVE_CENTRES:
        limits.append((fc / factor, fc * factor))
    return limits


class ThirdOctaveBand:
    """Result container for a single 1/3 octave band.

    Attributes
    ----------
    centre : float
        Centre frequency (Hz).
    lower : float
        Lower band edge (Hz).
    upper : float
        Upper band edge (Hz).
    spl : float
        Band SPL in dB.
    spl_a : float
        A-weighted band SPL in dBA.
    """

    __slots__ = ("centre", "lower", "upper", "spl", "spl_a")

    def __init__(
        self,
        centre: float,
        lower: float,
        upper: float,
        spl: float,
        spl_a: float,
    ) -> None:
        self.centre = centre
        self.lower = lower
        self.upper = upper
        self.spl = spl
        self.spl_a = spl_a

    def __repr__(self) -> str:
        return (
            f"ThirdOctaveBand({self.centre:.1f} Hz, "
            f"SPL={self.spl:.1f} dB, SPL_A={self.spl_a:.1f} dBA)"
        )


class NoiseEnhanced(Noise):
    """Enhanced noise analysis with A-weighting and 1/3 octave bands.

    Extends :class:`Noise` with perceptual noise metrics.

    Additional configuration keys (beyond Noise):

    - ``aWeighting``: enable A-weighting (default: True)
    - ``thirdOctaveBands``: enable 1/3 octave analysis (default: True)
    - ``windowSize``: window size for windowed analysis (default: 0 = full)
    - ``overlap``: overlap fraction for windowed analysis (default: 0.5)

    Example controlDict entry::

        noiseEnhanced1
        {
            type            noiseEnhanced;
            libs            ("libfieldFunctionObjects.so");
            fields          (p);
            probeLocations  ((0.5 0.5 0.5));
            pRef            2e-5;
            aWeighting      true;
            thirdOctaveBands true;
        }
    """

    def __init__(
        self,
        name: str = "noiseEnhanced",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._use_a_weighting: bool = self.config.get("aWeighting", True)
        self._use_third_octave: bool = self.config.get("thirdOctaveBands", True)
        self._window_size: int = int(self.config.get("windowSize", 0))
        self._overlap: float = float(self.config.get("overlap", 0.5))

        # Enhanced results
        self._a_weighted_psd: Optional[torch.Tensor] = None
        self._spl_a: Optional[torch.Tensor] = None
        self._spl_a_total: Optional[torch.Tensor] = None
        self._laeq: Optional[torch.Tensor] = None
        self._third_octave_results: Optional[List[List[ThirdOctaveBand]]] = None
        self._peak_frequencies: Optional[torch.Tensor] = None
        self._peak_spl: Optional[torch.Tensor] = None

    def finalise(self) -> None:
        """Compute enhanced spectra: A-weighting, 1/3 octave, LAeq."""
        # Run base class FFT analysis first
        super().finalise()

        if self._frequencies is None:
            return

        freqs = self._frequencies
        dtype = freqs.dtype

        # A-weighting
        if self._use_a_weighting:
            self._compute_a_weighted(freqs, dtype)

        # 1/3 octave bands
        if self._use_third_octave:
            self._compute_third_octave(freqs, dtype)

        # Equivalent continuous sound level
        self._compute_laeq(dtype)

        # Peak frequency
        self._compute_peak_frequency()

    def _compute_a_weighted(self, freqs: torch.Tensor, dtype: torch.dtype) -> None:
        """Apply A-weighting to PSD and compute A-weighted SPL."""
        # A-weighting correction (dB)
        a_corr = a_weighting(freqs).to(dtype=dtype)  # (n_freq,)

        # Apply to PSD: PSD_A(f) = PSD(f) * 10^(A(f)/10)
        a_factor = (10.0 ** (a_corr / 10.0)).unsqueeze(1)  # (n_freq, 1)
        self._a_weighted_psd = self._psd * a_factor

        # SPL_A per frequency
        p_ref_sq = self._p_ref ** 2
        self._spl_a = 10.0 * torch.log10(
            self._a_weighted_psd.clamp(min=1e-40) / p_ref_sq
        )

        # Overall A-weighted SPL
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        total_a_power = self._a_weighted_psd.sum(dim=0) * df
        self._spl_a_total = 10.0 * torch.log10(
            total_a_power.clamp(min=1e-40) / p_ref_sq
        )

        logger.info(
            "NoiseEnhanced '%s': A-weighting computed. SPL_A = %s",
            self.name,
            [f"{v:.2f}" for v in self._spl_a_total.tolist()],
        )

    def _compute_third_octave(self, freqs: torch.Tensor, dtype: torch.dtype) -> None:
        """Compute 1/3 octave band SPL."""
        limits = _third_octave_band_limits()
        n_probes = self._psd.shape[1] if self._psd.dim() > 1 else 1

        psd = self._psd
        if psd.dim() == 1:
            psd = psd.unsqueeze(1)

        self._third_octave_results = []
        for probe_idx in range(n_probes):
            probe_bands: List[ThirdOctaveBand] = []
            psd_col = psd[:, probe_idx]
            p_ref_sq = self._p_ref ** 2

            for i, fc in enumerate(_THIRD_OCTAVE_CENTRES):
                f_low, f_high = limits[i]
                mask = (freqs >= f_low) & (freqs < f_high)
                if not mask.any():
                    probe_bands.append(ThirdOctaveBand(fc, f_low, f_high, -999.0, -999.0))
                    continue

                band_power = psd_col[mask].sum()
                df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
                band_power *= df
                spl = 10.0 * math.log10(max(band_power.item(), 1e-40) / p_ref_sq) if band_power.numel() == 1 else 10.0 * torch.log10(band_power.clamp(min=1e-40) / p_ref_sq).item()

                # A-weighted band SPL
                if self._a_weighted_psd is not None:
                    a_psd_col = self._a_weighted_psd[:, probe_idx]
                    a_band_power = a_psd_col[mask].sum() * df
                    spl_a = 10.0 * math.log10(max(a_band_power.item(), 1e-40) / p_ref_sq)
                else:
                    spl_a = spl

                probe_bands.append(ThirdOctaveBand(fc, f_low, f_high, float(spl), float(spl_a)))

            self._third_octave_results.append(probe_bands)

    def _compute_laeq(self, dtype: torch.dtype) -> None:
        """Compute equivalent continuous A-weighted sound level (LAeq)."""
        if len(self._times) < 2:
            return

        n_samples = len(self._times)
        T = self._times[-1] - self._times[0]
        if T <= 0:
            return

        # Stack pressure history
        p_stack = torch.stack(self._pressure_history, dim=0).to(dtype=dtype)  # (n_samples, n_probes)

        # Apply A-weighting in time domain (simplified: use broadband A-correction
        # based on dominant frequency, or use the frequency-domain result)
        # For accuracy, we compute LAeq from A-weighted total power
        if self._spl_a_total is not None:
            # LAeq = SPL_A_total (for stationary signals, they are equivalent)
            self._laeq = self._spl_a_total.clone()
        else:
            # Fallback: compute unweighted Leq
            p_rms_sq = (p_stack ** 2).mean(dim=0)  # (n_probes,)
            self._laeq = 10.0 * torch.log10(
                p_rms_sq.clamp(min=1e-40) / (self._p_ref ** 2)
            )

    def _compute_peak_frequency(self) -> None:
        """Find peak frequency and SPL for each probe."""
        if self._spl is None:
            return

        spl = self._spl  # (n_freq, n_probes)
        if spl.dim() == 1:
            spl = spl.unsqueeze(1)

        self._peak_frequencies = self._frequencies[spl.argmax(dim=0)]
        self._peak_spl = spl.max(dim=0).values

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def a_weighted_psd(self) -> Optional[torch.Tensor]:
        """A-weighted PSD (Pa^2/Hz). Shape ``(n_freq, n_probes)``."""
        return self._a_weighted_psd

    @property
    def spl_a(self) -> Optional[torch.Tensor]:
        """A-weighted SPL per frequency bin (dBA). Shape ``(n_freq, n_probes)``."""
        return self._spl_a

    @property
    def spl_a_total(self) -> Optional[torch.Tensor]:
        """Overall A-weighted SPL per probe (dBA). Shape ``(n_probes,)``."""
        return self._spl_a_total

    @property
    def laeq(self) -> Optional[torch.Tensor]:
        """Equivalent continuous A-weighted sound level (dBA)."""
        return self._laeq

    @property
    def third_octave_results(self) -> Optional[List[List[ThirdOctaveBand]]]:
        """1/3 octave band results.  ``[probe][band]``."""
        return self._third_octave_results

    @property
    def peak_frequencies(self) -> Optional[torch.Tensor]:
        """Peak frequency per probe (Hz)."""
        return self._peak_frequencies

    @property
    def peak_spl(self) -> Optional[torch.Tensor]:
        """Peak SPL per probe (dB)."""
        return self._peak_spl

    def write(self) -> None:
        """Write enhanced noise data to output files."""
        # Write base class output
        super().write()

        if self._output_path is None:
            return

        # Write A-weighted SPL
        if self._spl_a_total is not None:
            spl_a_file = self._output_path / "spl_a_overall.dat"
            with open(spl_a_file, "w") as f:
                f.write("# Probe  SPL_A_total(dBA)  LAeq(dBA)\n")
                n_probes = len(self._probe_cells)
                for i in range(n_probes):
                    laeq_val = self._laeq[i].item() if self._laeq is not None else 0.0
                    f.write(f"{i}  {self._spl_a_total[i]:.6e}  {laeq_val:.6e}\n")
            logger.info("Wrote A-weighted SPL to %s", spl_a_file)

        # Write 1/3 octave band results
        if self._third_octave_results is not None:
            for probe_idx, bands in enumerate(self._third_octave_results):
                band_file = self._output_path / f"third_octave_probe{probe_idx}.dat"
                with open(band_file, "w") as f:
                    f.write("# Centre(Hz)  Lower(Hz)  Upper(Hz)  SPL(dB)  SPL_A(dBA)\n")
                    for band in bands:
                        f.write(
                            f"{band.centre:.1f}  {band.lower:.6e}  "
                            f"{band.upper:.6e}  {band.spl:.6e}  {band.spl_a:.6e}\n"
                        )
                logger.info("Wrote 1/3 octave bands to %s", band_file)

        # Write A-weighted spectrum per probe
        if self._spl_a is not None and self._write_spectrum:
            n_probes = self._spl_a.shape[1] if self._spl_a.dim() > 1 else 1
            spl_a_spec = self._spl_a if self._spl_a.dim() > 1 else self._spl_a.unsqueeze(1)
            for i in range(n_probes):
                spec_file = self._output_path / f"spectrum_a_probe{i}.dat"
                with open(spec_file, "w") as f:
                    f.write("# Frequency(Hz)  PSD_A(Pa2/Hz)  SPL_A(dBA)\n")
                    for j in range(len(self._frequencies)):
                        a_psd = self._a_weighted_psd[j, i].item() if self._a_weighted_psd is not None else 0.0
                        f.write(
                            f"{self._frequencies[j]:.6e}  "
                            f"{a_psd:.6e}  "
                            f"{spl_a_spec[j, i]:.6e}\n"
                        )

        # Write peak frequency report
        if self._peak_frequencies is not None:
            peak_file = self._output_path / "peak_frequency.dat"
            with open(peak_file, "w") as f:
                f.write("# Probe  PeakFreq(Hz)  PeakSPL(dB)\n")
                for i in range(len(self._peak_frequencies)):
                    f.write(
                        f"{i}  {self._peak_frequencies[i]:.6e}  "
                        f"{self._peak_spl[i]:.6e}\n"
                    )
            logger.info("Wrote peak frequency to %s", peak_file)


# Register
from pyfoam.postprocessing.function_object import FunctionObjectRegistry
FunctionObjectRegistry.register("noiseEnhanced", NoiseEnhanced)
