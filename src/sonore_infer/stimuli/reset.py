"""Reset of streaming after silence: two studies' stimuli (docs/design/order-effects.md, E2).

Unlike the other modules, these are not from App. C of the paper but from
the two listening studies the order-effects design compares with, copied
from their methods sections.

**Beauvois & Meddis (1997), Percept. Psychophys. 59:81-86.** An inducer of
10 s of A tones alone, a silence of 0 to 8 s, then a test of 8 AB cycles.
A 1000 Hz, B 1420 Hz; 40 ms tones with 5 ms raised-cosine ramps; 90 ms
onset to onset ("TRT") in both the inducer and the test; 65 dB SPL.

**Cusack, Deeks, Aikman & Carlyon (2004), JEP:HPP 30:643-656, Experiment 3.**
10 s sequences of ABA- triplets separated by silences of 1, 2, 5 or 10 s.
A 400 Hz, B 4 or 6 semitones higher (504 or 566 Hz as printed); 125 ms
tones with 20 ms linear ramps; a 125 ms silence after each triplet, so a
triplet takes 500 ms; 55 dB SPL.

Assumed, because the texts do not say or the model needs it:

- The inducer of Beauvois & Meddis holds 111 tones: as many 90 ms onsets
  as fit in 10 s with the last tone ending inside it.
- The silence is measured from the end of the last sound of the first part
  (the last tone for Beauvois & Meddis, the last triplet's trailing silence
  for Cusack et al.) to the first onset of the second part.
- Cusack et al.'s B tones are ``400 * 2 ** (semitones / 12)`` Hz, which
  rounds to their printed values.
- Cusack et al.'s second sequence is cut to ``test_duration`` s (4 s by
  default, where their reset shows; design E2); 10 s gives theirs.
- dB SPL is read as the model's dB (re 1e-6, :mod:`.levels`), as for the
  paper's own stimuli; each tone starts in cosine phase; ``lead`` s of
  silence come first (0 by default), as in :mod:`.buildup`.
"""

from dataclasses import dataclass

import sonore as so

from sonore_infer.stimuli.levels import FS, sinusoid

BM_A_FREQ = 1000.0
BM_B_FREQ = 1420.0
BM_TONE_DURATION = 0.040
BM_RAMP = 0.005
BM_ONSET_INTERVAL = 0.090
BM_LEVEL_DB = 65.0
BM_INDUCER_TONES = 111
BM_TEST_CYCLES = 8
BM_SILENCES = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)

CU_A_FREQ = 400.0
CU_TONE_DURATION = 0.125
CU_RAMP = 0.020
CU_ONSET_INTERVAL = 0.125
CU_LEVEL_DB = 55.0
CU_SEQUENCE_DURATION = 10.0
CU_SEMITONES = (4, 6)
CU_SILENCES = (1.0, 2.0, 5.0, 10.0)


@dataclass(frozen=True)
class Placed:
    """One tone of a reset stimulus: when it starts, which tone it is, and in which part."""

    onset: float  # s
    kind: str  # "A" or "B"
    part: str  # "first" (inducer or first sequence) or "second" (test)


@dataclass(frozen=True)
class ResetStimulus:
    """A reset stimulus and what a model needs to build its hypotheses."""

    sound: so.Sound
    tones: tuple[Placed, ...]
    freqs: dict[str, float]
    tone_duration: float  # s
    ramp: float  # s
    level_db: float
    gap: tuple[float, float]  # s: end of the first part, first onset of the second
    duration: float  # s


def _render(tones, freqs, tone_duration, ramp, shape, level_db, duration, fs) -> so.Sound:
    rendered = []
    for tone in tones:
        sound = sinusoid(tone_duration, fs, freqs[tone.kind], level_db).ramp(ramp, shape=shape)
        rendered.append(sound.pad_to(sound.n_samples + round(tone.onset * fs), align="end"))
    return so.mix(rendered).pad_to(round(duration * fs))


def beauvois_meddis(
    silence: float | None, *, inducer_tones: int = BM_INDUCER_TONES, lead: float = 0.0, fs: float = FS
) -> ResetStimulus:
    """Inducer, ``silence`` s, then the test; ``silence=None`` gives the test alone.

    ``inducer_tones`` shortens the inducer (for timing runs; the study's is 111)."""
    tones = []
    if silence is not None:
        tones += [Placed(lead + k * BM_ONSET_INTERVAL, "A", "first") for k in range(inducer_tones)]
        first_end = tones[-1].onset + BM_TONE_DURATION
        test_start = first_end + silence
    else:
        first_end = test_start = lead
    for k in range(2 * BM_TEST_CYCLES):
        tones.append(Placed(test_start + k * BM_ONSET_INTERVAL, "AB"[k % 2], "second"))
    duration = test_start + 2 * BM_TEST_CYCLES * BM_ONSET_INTERVAL
    freqs = {"A": BM_A_FREQ, "B": BM_B_FREQ}
    sound = _render(tones, freqs, BM_TONE_DURATION, BM_RAMP, "cosine", BM_LEVEL_DB, duration, fs)
    return ResetStimulus(
        sound, tuple(tones), freqs, BM_TONE_DURATION, BM_RAMP, BM_LEVEL_DB, (first_end, test_start), duration
    )


def _triplets(start: float, count: int, part: str) -> list[Placed]:
    tones = []
    for repeat in range(count):
        t = start + 4 * repeat * CU_ONSET_INTERVAL
        tones += [
            Placed(t, "A", part),
            Placed(t + CU_ONSET_INTERVAL, "B", part),
            Placed(t + 2 * CU_ONSET_INTERVAL, "A", part),
        ]
    return tones


def cusack(
    silence: float,
    semitones: float,
    *,
    test_duration: float = 4.0,
    first_duration: float = CU_SEQUENCE_DURATION,
    lead: float = 0.0,
    fs: float = FS,
) -> ResetStimulus:
    """A 10 s ABA- sequence, ``silence`` s, then the first ``test_duration`` s of another.

    ``first_duration`` shortens the first sequence (for timing runs; the study's is 10 s)."""
    per_triplet = 4 * CU_ONSET_INTERVAL
    tones = _triplets(lead, round(first_duration / per_triplet), "first")
    first_end = lead + first_duration
    test_start = first_end + silence
    tones += _triplets(test_start, round(test_duration / per_triplet), "second")
    duration = test_start + test_duration
    freqs = {"A": CU_A_FREQ, "B": CU_A_FREQ * 2 ** (semitones / 12)}
    sound = _render(tones, freqs, CU_TONE_DURATION, CU_RAMP, "linear", CU_LEVEL_DB, duration, fs)
    return ResetStimulus(
        sound, tuple(tones), freqs, CU_TONE_DURATION, CU_RAMP, CU_LEVEL_DB, (first_end, test_start), duration
    )
