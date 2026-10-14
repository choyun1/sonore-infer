"""Buildup of streaming (App. C.11, after Thompson et al., 2011).

From the paper: ABA triplets as in C.10 (A B A, then a silence as long as
one onset-to-onset interval); A at 500 Hz and B 4 or 8 semitones higher;
tones at 55 dB, 50 ms long, with 10 ms raised-cosine ramps; 125 ms
onset-to-onset, so each triplet takes 500 ms; 1 to 6 triplets; 12 stimuli.

Assumed, because the text does not say:

- Each tone starts in cosine phase, as in :mod:`.bistability`.
- No padding beyond the last triplet's silence, so the stimulus with ``n``
  triplets is the first ``lead + 0.5 n`` s of the one with 6, and the batch
  runs are prefixes of one sound (design online O2).
- ``lead`` s of silence before the first tone (0 by default, as in the
  paper). The model's runs use 50 ms, as ``tools/compare_aba.py`` does for
  C.10: the onset prior starts at 0, and a fit cannot move an onset that
  starts exactly there.
"""

import sonore as so

from sonore_infer.stimuli.levels import FS, sinusoid

SEMITONES = (4, 8)
TRIPLETS = (1, 2, 3, 4, 5, 6)
A_FREQ = 500.0
ONSET_INTERVAL = 0.125
TONE_DURATION = 0.050
TONE_LEVEL_DB = 55.0
RAMP = 0.010


def duration(triplets: int, lead: float = 0.0) -> float:
    """The stimulus's duration [s]: the lead, then four intervals per triplet."""
    return lead + 4 * ONSET_INTERVAL * triplets


def tone_onsets(triplets: int, lead: float = 0.0) -> list[tuple[float, str]]:
    """(onset time [s], "A" or "B") for every tone, in order."""
    onsets = []
    for repeat in range(triplets):
        start = lead + 4 * repeat * ONSET_INTERVAL
        onsets += [(start, "A"), (start + ONSET_INTERVAL, "B"), (start + 2 * ONSET_INTERVAL, "A")]
    return onsets


def tone_freqs(semitones: float) -> dict[str, float]:
    """Frequency [Hz] of tone A and of tone B, ``semitones`` above it."""
    return {"A": A_FREQ, "B": A_FREQ * 2 ** (semitones / 12)}


def buildup_stimulus(triplets: int, semitones: float, *, lead: float = 0.0, fs: float = FS) -> so.Sound:
    """``triplets`` ABA triplets, B ``semitones`` above A, after ``lead`` s of silence."""
    freqs = tone_freqs(semitones)
    tones = []
    for onset, name in tone_onsets(triplets, lead):
        tone = sinusoid(TONE_DURATION, fs, freqs[name], TONE_LEVEL_DB).ramp(RAMP, shape="cosine")
        tones.append(tone.pad_to(tone.n_samples + round(onset * fs), align="end"))
    return so.mix(tones).pad_to(round(duration(triplets, lead) * fs))


def buildup_set(**kwargs) -> list[tuple[dict, so.Sound]]:
    """All 12 stimuli as (condition, sound) pairs; ``kwargs`` go to the stimulus function."""
    return [
        ({"semitones": semitones, "triplets": n}, buildup_stimulus(n, semitones, **kwargs))
        for semitones in SEMITONES
        for n in TRIPLETS
    ]
