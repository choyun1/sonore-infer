"""The threshold of hearing and spectral dominance of Terhardt, Stoll & Seewann (1982).

Both are options for the Shepard pair experiments (design
relative-pitch-dominance, P19 and P20), off by default:

- :func:`threshold_of_hearing_db` is their Eq. 8, in dB SPL. The cochleagram
  can floor each channel at it (``Cochleagram.threshold_floor``), which reads
  the project's dB re 1e-6 RMS as dB SPL.
- :func:`dominance_weight` is the frequency factor of their spectral-pitch
  weight (Eq. 12, second factor), peaking at 1 at 700 Hz.
  :func:`dominance_sigma` turns it into a likelihood SD per channel,
  ``sigma / sqrt(weight)``, so a channel counts in proportion to its weight.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import ArrayLike


def threshold_of_hearing_db(freq: ArrayLike) -> np.ndarray:
    """Threshold of hearing [dB SPL] at ``freq`` [Hz] (Terhardt et al., 1982, Eq. 8)."""
    khz = np.asarray(freq, float) / 1000
    return 3.64 * khz**-0.8 - 6.5 * np.exp(-0.6 * (khz - 3.3) ** 2) + 1e-3 * khz**4


def dominance_weight(freq: ArrayLike) -> np.ndarray:
    """Spectral dominance at ``freq`` [Hz], 1 at 700 Hz (Terhardt et al., 1982, Eq. 12, second factor)."""
    ratio = np.asarray(freq, float) / 700
    return (1 + 0.07 * (ratio - 1 / ratio) ** 2) ** -0.5


def dominance_sigma(cfs: ArrayLike, sigma: float = 10.0, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Likelihood SD [dB] per channel, ``sigma / sqrt(dominance_weight(cf))``, shape ``(n_channels, 1)``."""
    return torch.as_tensor(sigma / np.sqrt(dominance_weight(cfs)), dtype=dtype)[:, None]
