"""The ERB scale (Glasberg & Moore, 1990), for NumPy arrays and torch tensors.

These are sonore's ``freq_to_erb``, ``erb_to_freq`` and ``erb_bandwidth``
(``sonore.core.utils``), with the same constants and the same order of
operations. They are written again here because the core does not import
sonore (design D2) and because f0 and frequency are inferred in ERB number,
so the conversion has to pass gradients. ``tests/test_erb.py`` holds them to
sonore's.

A torch tensor gives a torch tensor, differentiable; anything else gives a
float64 NumPy array, as sonore's do. :func:`freq_to_erb_float` is the same
conversion for one Python float with ``math.log1p``, which is how the scene's
and the tools' starting points were always computed; NumPy's ``log1p``
differs from it by one unit in the last place for some inputs, so keeping it
keeps their output bit for bit.

``9.265`` rounds ``1000 / (4.37 * 24.7) = 9.2645``, so the ERB number is the
integral of ``1 / erb_bandwidth`` to within 6e-5 (relative), not exactly.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def erb_bandwidth(freq):
    """Equivalent rectangular bandwidth [Hz] at ``freq`` [Hz]."""
    if isinstance(freq, torch.Tensor):
        return 24.7 * (4.37 * freq / 1000 + 1)
    return 24.7 * (4.37 * np.asarray(freq, float) / 1000 + 1)


def freq_to_erb(freq):
    """Frequency [Hz] to ERB number [Cams]."""
    if isinstance(freq, torch.Tensor):
        return 9.265 * torch.log1p(freq / (24.7 * 9.265))
    return 9.265 * np.log1p(np.asarray(freq, float) / (24.7 * 9.265))


def erb_to_freq(erb):
    """ERB number [Cams] to frequency [Hz]; the inverse of :func:`freq_to_erb`."""
    if isinstance(erb, torch.Tensor):
        return 24.7 * 9.265 * torch.expm1(erb / 9.265)
    return 24.7 * 9.265 * np.expm1(np.asarray(erb, float) / 9.265)


def freq_to_erb_float(freq: float) -> float:
    """:func:`freq_to_erb` for one Python float, with ``math.log1p``."""
    return 9.265 * math.log1p(freq / (24.7 * 9.265))
