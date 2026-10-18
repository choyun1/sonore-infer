"""Shepard tones and pairs of them (docs/design/relative-pitch.md, decision P6).

From Shepard, R. N. (1964), Circularity in judgments of relative pitch,
J. Acoust. Soc. Am. 36(12), 2346-2353. Tone ``t`` (1 to ``steps``) is made
of components ``c`` (1 to ``c_max``) an octave apart [his Eq. 1]:

    f = f_min * 2 ** (((c - 1) * steps + t - 1) / steps)

each at a level that is a raised cosine in dB over log frequency [Eqs. 2-3]:

    L = L_min + (L_max - L_min) * (1 - cos(theta)) / 2,
    theta = 2 pi ((c - 1) * steps + t - 1) / (steps * c_max)

with his experiment's values [§III]: 10 steps per octave, 10 components,
f_min 4.863 Hz, L_min 22 dB, L_max 56 dB. The envelope peaks at
f_min * 2**5 = 155.6 Hz. Tones last 1/8 s, and the two tones of a pair are
1/8 s apart [§III].

``components_per_octave`` generalizes Eq. 1 to Shepard's control in §V:
components a minor third apart (4 per octave, 40 components) under the same
envelope over the same 10 octaves.

Assumed, or departures (decision P6):

- Levels are dB re 1e-6 RMS per component (the project's convention,
  ``levels.py``); Shepard's are dB SPL.
- Components above 5 kHz are left out, as his 10 kHz sampling did.
- Each component starts in sine phase (upward from a zero crossing, as
  Shepard says all components did).
- Ramps are 10 ms raised cosine; Shepard's attack and decay times are
  illegible in the scan.
- 50 ms of silence before and after, as for the project's other stimuli;
  Shepard's noise bursts before and after each pair are left out.
"""

import numpy as np
import sonore as so

from sonore_infer.stimuli.levels import FS, sinusoid

STEPS_PER_OCTAVE = 10
N_OCTAVES = 10
F_MIN = 4.863
L_MIN_DB = 22.0
L_MAX_DB = 56.0
F_MAX = 5000.0
DURATION = 0.125
GAP = 0.125
RAMP = 0.010
PADDING = 0.050


def components(
    step: int, *, steps: int = STEPS_PER_OCTAVE, components_per_octave: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies [Hz] and levels [dB] of tone ``step`` (1 to ``steps``), up to ``F_MAX``."""
    if not 1 <= step <= steps:
        raise ValueError(f"step must be in 1..{steps}, got {step}")
    c = np.arange(N_OCTAVES * components_per_octave)
    octaves = c / components_per_octave + (step - 1) / steps  # above f_min
    freqs = F_MIN * 2**octaves
    theta = 2 * np.pi * octaves / N_OCTAVES
    levels = L_MIN_DB + (L_MAX_DB - L_MIN_DB) * (1 - np.cos(theta)) / 2
    keep = freqs <= F_MAX
    return freqs[keep], levels[keep]


def tone(
    step: int,
    *,
    steps: int = STEPS_PER_OCTAVE,
    components_per_octave: int = 1,
    duration: float = DURATION,
    fs: float = FS,
) -> so.Sound:
    """One ramped Shepard tone, no padding."""
    freqs, levels = components(step, steps=steps, components_per_octave=components_per_octave)
    parts = [
        sinusoid(duration, fs, f, level, phase=-np.pi / 2) for f, level in zip(freqs, levels, strict=True)
    ]
    return so.mix(parts).ramp(RAMP, shape="cosine")


def single_tone(step: int, *, components_per_octave: int = 1, fs: float = FS) -> so.Sound:
    """One tone alone, with ``PADDING`` before and after (experiments P1 and P2)."""
    return tone(step, components_per_octave=components_per_octave, fs=fs).pad(PADDING, PADDING)


def pair(first: int, second: int, *, fs: float = FS) -> so.Sound:
    """Tone ``first`` then tone ``second``, ``GAP`` apart, with ``PADDING`` before and after."""
    a = tone(first, fs=fs).pad(PADDING, GAP + DURATION + PADDING)
    b = tone(second, fs=fs).pad(PADDING + DURATION + GAP, PADDING)
    return so.mix([a, b])


def steps_clockwise(first: int, second: int, steps: int = STEPS_PER_OCTAVE) -> int:
    """How many steps up the circle ``second`` is from ``first`` (0 to ``steps - 1``)."""
    return (second - first) % steps
