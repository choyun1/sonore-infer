# P5. One Shepard tone, sparse spectra and a 30 Hz floor

> **Under review.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** first pass, one seed, all 10 tones.
**Paper target:** none in Cusimano et al.; listeners in Shepard (1964) §I.
**Code:** `src/sonore_infer/stimuli/shepard.py`,
`tools/relative_pitch/one_tone.py`, `tools/relative_pitch/combine_starts.py`.
Design: [relative pitch prior](../docs/design/relative-pitch-prior.md),
decisions P10 to P14.

## Question

Shared background: [how the experiments work](README.md#how-the-experiments-work).

[P1](P1-shepard-one-tone.md) again, with the two options of the relative
pitch prior design: each harmonic of a harmonic source may be absent
(decisions P10 and P11), and the f0 and whistle-frequency floor is 30 Hz
instead of 87.5 Hz (decision P12). Does the model now hear one Shepard tone
as one tone, as Shepard's listeners do [Shepard §I], and what sets the
octave it reads?

## Setup

As [P1](P1-shepard-one-tone.md), except:

- **Model options.** `Harmonic(sparse_spectrum=True, frequency_floor_hz=30)`
  and `Whistle(frequency_floor_hz=30)`. Each harmonic's level is the
  spectrum GP plus an offset whose prior is a two-part mixture: present,
  N(0, 1 dB), or absent, N(−40, 10 dB), with the label summed out; the
  probability of present is inferred per source, with a Beta(1, 1) prior.
- **Hypotheses.** `harmonic@F` for every component from 30 Hz up to
  350 Hz; `harmonic@F+whistle@F/2`, the search's best structure on P1's
  tone 1, for the same readings; and one whistle per component at or above
  30 dB and 30 Hz (5 or 6 whistles).
- **Two starts per harmonic source.** One puts the spectrum GP through the
  stimulus's levels with the harmonics the stimulus lacks marked absent;
  the other has every harmonic present, as on P1. They reach different
  modes, so each mode gets its own evidence estimate, and a hypothesis's
  log posterior is the log-sum-exp of its distinct modes
  (`combine_starts.py`; modes whose log joints agree within 1 count once).
  Keeping only the mode with the higher log joint, as a first pass did,
  lost up to 60 (tone 1, `harmonic@38.9`).

Model as B2: FFT cochleagram with BASS's gain, fixed kernels, likelihood SD
10 dB; sonore 0.6.0. One seed. Cloud CPU (4-core Xeon, 2.1 GHz), 4 fits at
a time; 130 fits took 189 to 922 s each.

## Results

Log posteriors are shifted by 4329.7, as on P1. All per-tone values are in
`data/P5/combined.txt`, made by
`python tools/relative_pitch/combine_starts.py results/data/P5/*.txt`.

| Tone | Best (one harmonic source at) | Its log posterior | Next octave, relative | Best tone + whistle, relative | Whistles, relative |
| --- | --- | --- | --- | --- | --- |
| 1 | 77.8 Hz | −65.4 | −11.3 (38.9 Hz) | −2.1 | −50.6 |
| 2 | 83.4 Hz | −64.2 | −15.2 (41.7 Hz) | −8.1 | −54.6 |
| 3 | 89.4 Hz | −67.3 | −6.0 (44.7 Hz) | −8.3 | −48.4 (ELBO) |
| 4 | 95.8 Hz | −70.2 | −4.8 (47.9 Hz) | −6.0 | −48.6 |
| 5 | 102.7 Hz | −71.9 | −1.2 (51.3 Hz) | −4.4 | −31.8 |
| 6 | 55.0 Hz | −72.9 | −4.3 (110.0 Hz) | −6.2 | −30.0 |
| 7 | 59.0 Hz | −73.6 | −9.0 (117.9 Hz) | −6.3 | −26.4 |
| 8 | 63.2 Hz | −69.8 | −16.8 (126.4 Hz) | −5.8 | −50.1 (ELBO) |
| 9 | 67.7 Hz | −69.9 | −17.1 (33.9 Hz) | −7.1 | −42.9 (ELBO) |
| 10 | 72.6 Hz | −71.8 | −12.5 (36.3 Hz) | −7.8 | −49.8 |

"Next octave" is the best other single reading, an octave below or above.
At every tone the best mode of the best reading is the sparse one (the
start with missing harmonics marked absent); the mode with every harmonic
present is 26 to 42 lower.

The effective sample size of the importance estimates is 1 to 4 for most
fits, as on P1, so margins under about 10 are not resolved.

**The four combinations on tone 1** (`data/P5/tone1-combinations/`, run
before the second start was added: the sparse runs there have the sparse
start only, which is the better mode for every reading on this tone).

| Spectrum prior, floor | Best enumerated | Its log posterior | Best one harmonic source | Whistles |
| --- | --- | --- | --- | --- |
| GP, 87.5 Hz ([P1](P1-shepard-one-tone.md)) | 4 whistles | −116.7 | 155.6 Hz, −126.0 | −116.7 |
| GP, 30 Hz | tone at 311.2 Hz + whistle at 155.6 Hz | −91.4 | 77.8 Hz, −93.6 | −116.0 |
| Sparse, 87.5 Hz | tone at 311.2 Hz + whistle at 155.6 Hz | −94.7 | 155.6 Hz, −108.1 | −116.7 |
| Sparse, 30 Hz | one tone at 77.8 Hz | −65.4 | 77.8 Hz, −65.4 | −116.0 |

P1 had no tone-plus-whistle hypotheses; its search found one at −88.9.

## What it says

- **With both options the model hears one tone at every step**, as
  Shepard's listeners do [Shepard §I]. One harmonic source beats the
  whistles by 26 to 55 at all 10 tones (P1: whistles won at 7). A tone plus
  a whistle comes 2 to 8 behind, within the estimates' noise.
- **Both options are needed, at least on tone 1.** Either alone leaves a
  tone plus a whistle best, about 26 to 29 below the one tone that both
  together give.
- **The floor no longer sets the octave.** The reading is the one component
  between 52 and 104 Hz at every tone. Where a lower component lies above
  30 Hz (tones 1 to 5 and 8 to 10), reading it loses by 1.2 to 17. So the reading climbs from 78 Hz at tone 1 to
  103 Hz at tone 5 and then drops an octave to 55 Hz at tone 6. Within one
  octave the circle still has a jump, now between tones 5 and 6 instead of
  between tones 2 and 3. At tones 5 and 6 the two octaves are within 1.2
  and 4.3, not resolved. What sets the 52 to 104 Hz octave is not
  established (not tested).
- **The two starts matter.** For the octave complex the mode with missing
  harmonics marked absent wins by 26 to 42; for the minor-third complex
  ([P6](P6-shepard-minor-third-sparse.md)) the mode with every harmonic
  present wins. Scoring only the mode with the higher log joint would have
  picked the wrong one for some readings.
- **Not checked:** other seeds; the search with the options; whether a
  pair of tones (P3) is heard to move in Shepard's direction, given the
  jump between tones 5 and 6.
