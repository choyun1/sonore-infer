# P3. Shepard pairs: up or down, all 90 pairs

> **Under review.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** all 90 ordered pairs, one seed.
**Paper target:** none in Cusimano et al.; listeners in Shepard (1964)
Fig. 3.
**Code:** `src/sonore_infer/stimuli/shepard.py`,
`tools/relative_pitch/pair.py` (with `--sparse-spectrum --floor 30`),
`tools/relative_pitch/pair_summary.py`, `tools/results_plots.py`.
Design: [relative pitch](../docs/design/relative-pitch.md), experiments P3
and P4 and decisions P1 (B), P4, P5 and P8; [relative pitch
prior](../docs/design/relative-pitch-prior.md) for the options.

## Question

Shared background: [how the experiments work](README.md#how-the-experiments-work).

Shepard (1964) played every ordered pair of his 10 octave-complex tones and
asked whether the second was higher or lower. Listeners heard the shorter
way round the circle of 10 steps: up when the second tone was 1 to 4 steps
clockwise, down at 6 to 9, and about even at 5 (Fig. 3). Does the model
hear the same direction, how sure is it, and does its answer depend on the
starting tone (experiment P4, the half octave by starting tone, which
Shepard does not report)?

## Setup

- **Stimuli.** `shepard.pair(i, k)` for every first tone i and second tone
  k ≠ i: tone i, a 0.125 s gap, tone k, 0.125 s each, with 0.05 s of
  silence before and after (0.475 s). Shepard's parameters (decision P6).
  Tones played twice in a row are not in Shepard's set and are not run.
- **Hypotheses** (decision P4, with one reading per pair). Each is one
  scene whose sources have two events, one per tone:
  - `harmonic up` and `harmonic down`: one harmonic source starting at the
    first tone's component between 52 and 104 Hz (the reading
    [P5](P5-shepard-one-tone-sparse.md) found for every tone), whose second
    event starts at the nearest component of the second tone above (up) or
    below (down) it.
  - `whistles up` and `whistles down`: one whistle per component of the
    first tone at or above 30 dB and 30 Hz, each moving to the nearest
    component of the second tone above or below it (those whose target is
    above 30 Hz).
- **Options.** Sparse harmonic spectra and a 30 Hz floor for f0 and whistle
  frequency, as on P5. The harmonic source starts with the harmonics the
  first tone lacks marked absent, the start that won at every tone on P5;
  it gets no dense start.
- **Readout** (decisions P1 (B) and P5). Each hypothesis's direction is read
  at its fitted mode, from the mean f0 (for whistles, the mean whistle
  frequency) over the second tone minus that over the first, in ERB number.
  P(up) is the posterior mass of the hypotheses whose fitted shift is
  upward: variational importance estimate plus structure prior, the ELBO
  where importance sampling fails (40 of 360 fits). No fit's shift changed
  sign from its start.

Model as P5: FFT cochleagram with BASS's gain, fixed kernels, likelihood SD
10 dB; sonore 0.8.0. One seed. Cloud CPU (4-core Xeon, 2.1 GHz), 4 fits at
a time; 360 fits took 398 to 1463 s each, about 17 hours in all, in
2-hour chunks. Tone 1's pairs were run with all four hypotheses per file
(`1-k.txt`), the rest one family per file (`i-k-family.txt`, from pair.py's
`--only`). Pair 1 to 10 was cut off once and rerun in full; its three
repeated fits matched to the printed digits.

## Results

![P(up) for each pair, and against Shepard's Fig. 3](plots/P3-shepard-pairs.svg)

All values are in `data/P3/summary.txt`, made by
`python tools/relative_pitch/pair_summary.py --by-steps results/data/P3/[0-9]*.txt`,
and recomputed from the rows' printed log posteriors (one decimal), so
they can differ from pair.py's own P(up) lines in the third decimal.

**By steps clockwise**, P(up) averaged over the 10 starting tones, beside
Shepard's fractions read by eye from his Fig. 3 (pooled over all 90 pairs;
Group I his 26 most accurate listeners on pure tones, Group III his 14
least accurate):

| Steps clockwise | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model, mean P(up) | 1.00 | 1.00 | 0.77 | 0.63 | 0.24 | 0.00 | 0.00 | 0.00 | 0.00 |
| Model, lowest to highest | 1–1 | 1–1 | 0.00–1 | 0.00–1 | 0.00–1.00 | 0.00–0.01 | 0–0 | 0–0 | 0–0 |
| Shepard Group I | 1.00 | 1.00 | 0.89 | 0.75 | 0.53 | 0.25 | 0.09 | 0.02 | 0.02 |
| Shepard Group III | 0.74 | 0.78 | 0.83 | 0.75 | 0.52 | 0.29 | 0.24 | 0.26 | 0.40 |

**By starting tone**: the largest number of steps clockwise at which the
model still says up (P(up) above 0.5), and P(up) at the half octave (5
steps, experiment P4's question):

| First tone | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Its f0 (Hz) | 77.8 | 83.4 | 89.4 | 95.8 | 102.7 | 55.0 | 59.0 | 63.2 | 67.7 | 72.6 |
| Up up to (steps) | 3 | 4 | 2 | 2 | 2 | 5 | 5 | 4 | 4 | 4 |
| P(up) at 5 steps | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.96 | 1.00 | 0.06 | 0.38 | 0.03 |

Tone 1 ties at 4 steps (0.48) and tone 5 at 3 steps (0.50).

Across all 90 pairs the better whistle hypothesis is 67 to 147 behind the
best. P(up) is between 0.01 and 0.99 in 9 pairs, all at 3 to 5 steps, where
up and down are within 3.5 of each other. Effective sample sizes are 1 to 4
of 128, as on P5, so the importance estimates are rough.

## What it says

- **The model hears the shorter way round at small and large steps.** At
  1 and 2 steps every starting tone goes up, and at 6 to 9 every one goes
  down, by one harmonic source. So the f0 trajectory prior acts as a
  proximity rule, as the feasibility note expected.
- **Its crossover is early on average.** Averaged over starting tones, P(up)
  is 0.63 at 4 steps and 0.24 at 5, where listeners are at 0.75 and 0.53.
  On tone 1 alone the reason looked like the ERB scale: the same step in
  semitones is smaller in ERB number going down, so down is cheaper.
- **The crossover moves with the starting tone, by up to 3 steps.** Tones
  3 to 5, whose f0 sits near the top of the 52 to 104 Hz band, turn down
  after 2 steps; tones 6 and 7, near the bottom, still go up at 5. In
  every pair exactly one of the two harmonic hypotheses keeps the second
  tone's f0 inside that band, the band P5 found each tone alone is heard
  in. The model picks that one in 71 of the 89 decided pairs, so leaving
  the band looks costly and pulls the answer toward the in-band reading;
  this is inferred from the pattern, not tested.
- **So the half octave depends on the starting tone** (experiment P4): up
  from tones 6 and 7, down from tones 1 to 5, 8 and 10, graded (0.38) from
  tone 9. The design records this as a prediction; Shepard reports no
  data by starting tone.
- **It is far too sure.** P(up) is 0 or 1 in 81 of 90 pairs, where even
  Group I is graded from 3 to 6 steps and Group III never goes beyond 0.83
  or below 0.24. This is the same overconfidence as for one tone against
  two on [P7](P7-earlier-experiments-sparse.md). Decision P9 plans to model
  listener differences as fitted noise per group; nothing here does that
  yet.
- **Not like for like.** Only one reading of the first tone is tried (the
  component in 52 to 104 Hz), though P5 found the reading an octave down
  only 11 and 15 behind at tones 1 and 2; a hypothesis starting there would compete
  at the band's edges. The interval latent of decision P1 (D) is not built
  yet.
- **Not checked:** other readings, other seeds, the options off, and
  listener data by starting tone (none read; Shepard pools over tones).
