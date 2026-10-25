# P3. Shepard pairs: up or down, from one starting tone

> **Under review.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** first pass, one seed, one starting tone (9 of the 90 pairs).
**Paper target:** none in Cusimano et al.; listeners in Shepard (1964)
Fig. 3.
**Code:** `src/sonore_infer/stimuli/shepard.py`,
`tools/relative_pitch/pair.py` (with `--sparse-spectrum --floor 30`),
`tools/relative_pitch/pair_summary.py`.
Design: [relative pitch](../docs/design/relative-pitch.md), experiment P3
and decisions P1 (B), P4, P5 and P8; [relative pitch
prior](../docs/design/relative-pitch-prior.md) for the options.

## Question

Shared background: [how the experiments work](README.md#how-the-experiments-work).

Shepard (1964) played pairs of his octave-complex tones and asked whether
the second was higher or lower. Listeners heard the shorter way round the
circle of 10 steps: up when the second tone was 1 to 4 steps clockwise,
down at 6 to 9, and about even at 5 (Fig. 3). Does the model hear the same
direction, and how sure is it?

## Setup

- **Stimuli.** `shepard.pair(1, k)` for k = 2 to 10: tone 1, a 0.125 s
  gap, tone k, 0.125 s each, with 0.05 s of silence before and after
  (0.475 s). Shepard's parameters (decision P6).
- **Hypotheses** (decision P4, with one reading per pair). Each is one
  scene whose sources have two events, one per tone:
  - `harmonic up` and `harmonic down`: one harmonic source starting at
    77.8 Hz, tone 1's component between 52 and 104 Hz (the reading P5
    found), whose second event starts at the nearest component of tone k
    above (up) or below (down) it.
  - `whistles up` and `whistles down`: one whistle per component of tone 1
    at or above 30 dB and 30 Hz (6), each moving to the nearest component
    of tone k above or below it (5 or 6 whistles, those whose target is
    above 30 Hz).
- **Options.** Sparse harmonic spectra and a 30 Hz floor for f0 and whistle
  frequency, as on [P5](P5-shepard-one-tone-sparse.md). The harmonic
  source starts with the harmonics tone 1 lacks marked absent, the start
  that won at every tone on P5; it gets no dense start.
- **Readout** (decisions P1 (B) and P5). Each hypothesis's direction is read
  at its fitted mode, from the mean f0 (for whistles, the mean whistle
  frequency) over the second tone minus that over the first, in ERB number.
  P(up) is the posterior mass of the hypotheses whose fitted shift is
  upward: variational importance estimate plus structure prior, the ELBO
  where importance sampling fails. In every pair each hypothesis's fitted
  shift kept its start's sign.

Model as P5: FFT cochleagram with BASS's gain, fixed kernels, likelihood SD
10 dB; sonore 0.8.0. One seed. Cloud CPU (4-core Xeon, 2.1 GHz), 4 pairs at
a time; 36 fits took 442 to 1017 s each. Pair 1 to 10 was cut off by the
time limit after three of its fits and rerun in full; the three repeated
fits matched to the printed digits, except run time.

## Results

From `data/P3/summary.txt`, made by
`python tools/relative_pitch/pair_summary.py results/data/P3/*.txt`. Up
minus down is the log posterior of `harmonic up` minus `harmonic down`;
whistles is the better whistle hypothesis relative to the best hypothesis.
Shepard's fractions are read by eye from his Fig. 3, pooled over all 90
pairs; Group I are his 26 most accurate listeners on pure tones, Group III
his 14 least accurate.

| Steps clockwise | Best | Up minus down | Whistles | Fitted shift up / down (ERB) | P(up) | Shepard I | Shepard III |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | up, 77.8→83.4 Hz | 35.7 | −125.2 | +0.16 / −1.08 | 1.000 | 1.00 | 0.74 |
| 2 | up, 77.8→89.4 Hz | 25.2 | −125.5 | +0.31 / −0.98 | 1.000 | 1.00 | 0.78 |
| 3 | up, 77.8→95.8 Hz | 17.5 | −114.2 | +0.49 / −0.88 | 1.000 | 0.89 | 0.83 |
| 4 | down, 77.8→51.3 Hz | −0.1 | −107.9 | +0.66 / −0.77 | 0.471 | 0.75 | 0.75 |
| 5 | down, 77.8→55.0 Hz | −12.7 | −115.8 | +0.85 / −0.66 | 0.000 | 0.53 | 0.52 |
| 6 | down, 77.8→59.0 Hz | −23.6 | −90.8 | +1.03 / −0.54 | 0.000 | 0.25 | 0.29 |
| 7 | down, 77.8→63.2 Hz | −36.0 | −146.9 | +1.25 / −0.42 | 0.000 | 0.09 | 0.24 |
| 8 | down, 77.8→67.7 Hz | −53.5 | −120.6 | +1.44 / −0.29 | 0.000 | 0.02 | 0.26 |
| 9 | down, 77.8→72.6 Hz | −68.6 | −119.4 | +1.69 / −0.15 | 0.000 | 0.02 | 0.40 |

The raw outputs (`data/P3/1-k.txt`) give every fit's log joint, Laplace,
ELBO and importance estimates; effective sample sizes are 1 to 4 of 128,
as on P5, so the importance estimates are rough.

## What it says

- **The model hears the shorter way round, as listeners do.** One harmonic
  source moving up wins at 1 to 3 steps and one moving down at 5 to 9;
  the better whistle hypothesis is 91 to 147 behind everywhere. So the f0
  trajectory prior acts as a proximity rule, as the feasibility note
  expected.
- **Its crossover is early.** Up and down tie at 4 steps (−0.1), where
  listeners still say up on 75% of presentations; at 5 steps, where
  listeners are at about 50%, the model says down by 12.7. The two fitted
  shifts are about equal in ERB number between 4 and 5 steps (0.66 against
  0.77 at 4, 0.85 against 0.66 at 5): the same step in semitones is
  smaller in ERB number going down from 77.8 Hz than going up, so down is
  cheaper. Whether the crossover moves with the starting tone is
  experiment P4's question; this page has only tone 1, whose reading
  (77.8 Hz) sits in the middle of the 52 to 104 Hz band.
- **It is far too sure.** Away from 4 steps the log odds are 13 to 69, so
  P(up) is 0 or 1, where even Group I is graded at 3 to 6 steps and
  Group III never goes beyond 0.83 or below 0.24. This is the same
  overconfidence as for one tone against two on
  [P7](P7-earlier-experiments-sparse.md). Decision P9 plans to model
  listener differences as fitted noise per group; nothing here does that
  yet.
- **Not like for like.** Shepard's curve pools all 10 starting tones; this
  page has one. Only one reading of tone 1 is tried (the component in 52 to
  104 Hz), though P5 found the reading an octave down (38.9 Hz) 11 behind;
  the interval latent of decision P1 (D) is not built yet.
- **Not checked:** other starting tones, other readings, other seeds, the
  options off.
