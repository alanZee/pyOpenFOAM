"""
noiseEnhanced — enhanced noise analysis tools for aeroacoustic post-processing.

Extends OpenFOAM's ``noise`` utility with additional spectral analysis
capabilities:

- **Power Spectral Density (PSD)**: Welch's method for estimating PSD
  from pressure time series.
- **Sound Pressure Level (SPL)**: Compute SPL in dB (reference 20 uPa)
  with A-weighting option.
- **1/3-octave band analysis**: Standard fractional octave bands.
- **Overall SPL**: Band-integrated sound pressure level.
- **OASPL**: Overall sound pressure level from time-domain signal.

Usage::

    from pyfoam.tools.noise_enhanced import noise_analysis

    result = noise_analysis(
        pressure_signal,
        sample_rate=44100,
    )
    print(result.overall_spl_db)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

__all__ = ["NoiseResult", "noise_analysis"]


# ---------------------------------------------------------------------------
# Reference constants
# ---------------------------------------------------------------------------

# Standard reference pressure for SPL in air (Pa)
_P_REF = 20e-6

# A-weighting coefficients (IEC 61672)
# Simplified table: (freq_Hz, A_weight_dB)
_A_WEIGHT_TABLE = [
    (10.0, -70.4), (12.5, -63.4), (16.0, -56.7), (20.0, -50.5),
    (25.0, -44.7), (31.5, -39.4), (40.0, -34.6), (50.0, -30.2),
    (63.0, -26.2), (80.0, -22.5), (100.0, -19.1), (125.0, -16.1),
    (160.0, -13.4), (200.0, -10.9), (250.0, -8.6), (315.0, -6.6),
    (400.0, -4.8), (500.0, -3.2), (630.0, -1.9), (800.0, -0.8),
    (1000.0, 0.0), (1250.0, 0.6), (1600.0, 1.0), (2000.0, 1.2),
    (2500.0, 1.3), (3150.0, 1.2), (4000.0, 1.0), (5000.0, 0.5),
    (6300.0, -0.1), (8000.0, -1.1), (10000.0, -2.5), (12500.0, -4.3),
    (16000.0, -6.6), (20000.0, -9.3),
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class NoiseResult:
    """Result from :func:`noise_analysis`.

    Attributes
    ----------
    frequencies : np.ndarray
        Frequency array (Hz) from PSD estimation.
    psd : np.ndarray
        Power spectral density (Pa^2/Hz) at each frequency.
    spl : np.ndarray
        Sound pressure level (dB re 20 uPa) at each frequency.
    spl_a_weighted : np.ndarray
        A-weighted SPL (dBA) at each frequency.
    octave_bands : dict[str, float]
        1/3-octave band SPL values keyed by centre frequency string.
    overall_spl_db : float
        Overall sound pressure level (dB).
    overall_oaspl_dba : float
        Overall A-weighted sound pressure level (dBA).
    peak_frequency : float
        Frequency of the maximum SPL.
    peak_spl_db : float
        Maximum SPL value.
    sample_rate : float
        Sample rate used for analysis.
    n_samples : int
        Number of input samples.
    """

    frequencies: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    psd: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    spl: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    spl_a_weighted: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    octave_bands: dict = field(default_factory=dict)
    overall_spl_db: float = 0.0
    overall_oaspl_dba: float = 0.0
    peak_frequency: float = 0.0
    peak_spl_db: float = 0.0
    sample_rate: float = 0.0
    n_samples: int = 0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def noise_analysis(
    pressure: np.ndarray,
    sample_rate: float = 1.0,
    window: str = "hanning",
    n_fft: Optional[int] = None,
    overlap: float = 0.5,
    a_weighting: bool = True,
) -> NoiseResult:
    """Perform enhanced noise analysis on a pressure time series.

    Parameters
    ----------
    pressure : np.ndarray
        1-D pressure signal (Pa).
    sample_rate : float
        Sampling frequency (Hz).  Default ``1.0``.
    window : str
        FFT window type: ``"hanning"``, ``"hamming"``, or ``"none"``.
    n_fft : int, optional
        FFT size.  Defaults to the signal length (capped at 8192).
    overlap : float
        Segment overlap fraction for Welch's method (0 to 0.75).
    a_weighting : bool
        Apply A-weighting to SPL results.

    Returns
    -------
    NoiseResult
        Spectral analysis results.

    Raises
    ------
    ValueError
        If the pressure array is not 1-D or is too short.
    """
    p = np.asarray(pressure, dtype=np.float64).ravel()
    if p.size < 4:
        raise ValueError("Pressure signal must have at least 4 samples.")

    if n_fft is None:
        n_fft = min(p.size, 8192)
    n_fft = max(4, int(n_fft))

    # Welch PSD estimation
    freqs, psd = _welch_psd(p, sample_rate, n_fft, window, overlap)

    # SPL
    psd_safe = np.maximum(psd, 1e-40)
    spl = 10.0 * np.log10(psd_safe / (_P_REF ** 2) + 1e-40)

    # A-weighting
    if a_weighting:
        a_weights = _interp_a_weights(freqs)
        spl_a = spl + a_weights
    else:
        spl_a = spl.copy()

    # Overall SPL from PSD integration
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    overall_spl = 10.0 * np.log10(np.sum(psd) * df / (_P_REF ** 2) + 1e-40)
    overall_oaspl = 10.0 * np.log10(
        np.sum(10.0 ** (spl_a / 10.0)) * df + 1e-40
    )

    # 1/3-octave bands
    octave_bands = _third_octave_bands(freqs, psd)

    # Peak
    peak_idx = int(np.argmax(spl))
    peak_freq = float(freqs[peak_idx]) if len(freqs) > 0 else 0.0
    peak_spl = float(spl[peak_idx]) if len(spl) > 0 else 0.0

    return NoiseResult(
        frequencies=freqs,
        psd=psd,
        spl=spl,
        spl_a_weighted=spl_a,
        octave_bands=octave_bands,
        overall_spl_db=float(overall_spl),
        overall_oaspl_dba=float(overall_oaspl),
        peak_frequency=peak_freq,
        peak_spl_db=peak_spl,
        sample_rate=sample_rate,
        n_samples=p.size,
    )


# ---------------------------------------------------------------------------
# Welch PSD
# ---------------------------------------------------------------------------


def _welch_psd(
    signal: np.ndarray,
    fs: float,
    n_fft: int,
    window_name: str,
    overlap: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate PSD using Welch's method (averaged periodogram)."""
    overlap = max(0.0, min(0.75, overlap))
    step = max(1, int(n_fft * (1.0 - overlap)))
    n = signal.size

    # Create window
    if window_name == "hanning":
        win = np.hanning(n_fft)
    elif window_name == "hamming":
        win = np.hamming(n_fft)
    else:
        win = np.ones(n_fft)

    # Collect segments
    segments = []
    start = 0
    while start + n_fft <= n:
        seg = signal[start: start + n_fft] * win
        segments.append(seg)
        start += step

    if not segments:
        # Fallback: single segment
        padded = np.zeros(n_fft)
        padded[: min(n, n_fft)] = signal[: min(n, n_fft)]
        segments = [padded * win]

    n_seg = len(segments)
    n_freqs = n_fft // 2 + 1

    # Accumulate periodograms
    psd_sum = np.zeros(n_freqs, dtype=np.float64)
    for seg in segments:
        fft_result = np.fft.rfft(seg, n=n_fft)
        psd_sum += np.abs(fft_result) ** 2

    # Normalise
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    # Factor of 2 for one-sided spectrum, divided by window power and FFT size
    win_power = np.mean(win ** 2)
    psd = psd_sum / (n_seg * n_fft * fs * win_power + 1e-30)
    # One-sided: double all except DC and Nyquist
    psd[1:-1] *= 2.0

    return freqs, psd


# ---------------------------------------------------------------------------
# A-weighting
# ---------------------------------------------------------------------------


def _interp_a_weights(freqs: np.ndarray) -> np.ndarray:
    """Interpolate A-weighting values at arbitrary frequencies."""
    table_freqs = np.array([f for f, _ in _A_WEIGHT_TABLE])
    table_weights = np.array([w for _, w in _A_WEIGHT_TABLE])

    # Log-frequency interpolation with bounds extrapolation
    log_freq = np.log10(np.maximum(freqs, 1.0))
    log_table = np.log10(table_freqs)

    weights = np.interp(log_freq, log_table, table_weights)
    return weights


# ---------------------------------------------------------------------------
# 1/3-octave bands
# ---------------------------------------------------------------------------


def _third_octave_bands(
    freqs: np.ndarray,
    psd: np.ndarray,
) -> dict[str, float]:
    """Compute 1/3-octave band SPL from PSD."""
    # Standard 1/3-octave centre frequencies (Hz)
    centres = [
        10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0,
        100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0,
        630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0,
        3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0, 12500.0,
        16000.0, 20000.0,
    ]
    factor = 2.0 ** (1.0 / 6.0)  # half-bandwidth multiplier

    result: dict[str, float] = {}
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    for fc in centres:
        f_lo = fc / factor
        f_hi = fc * factor
        mask = (freqs >= f_lo) & (freqs < f_hi)
        if not np.any(mask):
            continue

        power = np.sum(psd[mask]) * df
        if power > 1e-40:
            spl = 10.0 * np.log10(power / (_P_REF ** 2) + 1e-40)
        else:
            spl = -np.inf

        # Format key
        if fc >= 1000.0:
            key = f"{fc / 1000:.1f}kHz"
        else:
            key = f"{fc:.0f}Hz"

        result[key] = float(spl)

    return result
