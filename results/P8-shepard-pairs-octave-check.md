# P8. Shepard pairs: the first tone heard an octave down or up

> **Under review.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** first pass, one seed, the 30 pairs 3 to 5 steps apart.
**Paper target:** none in Cusimano et al.; listeners in Shepard (1964)
Fig. 3.
**Code:** `tools/relative_pitch/pair.py` (with `--octave -1` and
`--octave 1`, `--sparse-spectrum --floor 30`),
`tools/relative_pitch/pair_summary.py`.
Design: [relative pitch](../docs/design/relative-pitch.md), experiments P3
and P4, decision P4 (hypotheses: one harmonic source per octave reading).

## Question

On [P3](P3-shepard-pairs.md) every harmonic hypothesis started the first
tone at its component between 52 and 104 Hz, and the point where the model
switched from up to down moved with the starting tone in step with that
range. Is that a property of the model, or of starting every fit at one
reading? Here the first tone may also be heard an octave lower or higher.

## Setup

As P3, with two more readings for the 30 pairs 3, 4 or 5 steps clockwise,
where P3's answers were close or varied with the starting tone:

- `harmonic up` and `harmonic down` again, with the first tone's f0 starting
  at its component an octave below the P3 reading (`--octave -1`, 31.6 to
  51.3 Hz) or above it (`--octave 1`, 104 to 205 Hz). The second event
  starts at the second tone's nearest component above or below, as on P3.
- A start below the 30 Hz floor is left out, since the prior gives it no
  weight. So the octave-down reading has a `down` hypothesis in only 6 of
  the 30 pairs (from tones 3 to 5; elsewhere the step down lands below
  30 Hz), and tones 6 and 7, whose octave-down reading is 27.5 and 29.5 Hz,
  have none. Adding readings from below can therefore add more up than
  down; this is an asymmetry of the floor, which Pressnitzer et al. (2001)
  put at about 30 Hz.
- P(up) is the posterior mass over all of a pair's hypotheses: P3's four
  plus these.

90 fits, one seed, cloud CPU, 4 at a time, 404 to 1635 s each; no
importance estimate failed and no fit's shift changed sign.

## Results

From `data/P8/summary.txt`, made by
`python tools/relative_pitch/pair_summary.py --by-steps results/data/P3/[0-9]*.txt results/data/P8/*.txt`.
P3 is P(up) with the 52 to 104 Hz reading only (the `p_up_band` column),
P8 with all readings (`p_up`); "Best" is the best hypothesis overall.

| Steps | First tone | P3 | P8 | Best |
| --- | --- | --- | --- | --- |
| 3 | 1 | 1.00 | 1.00 | up 77.8→95.8 |
| 3 | 2 | 1.00 | 1.00 | up 83.4→102.7 |
| 3 | 3 | 0.20 | 0.98 | up 44.7→55.0 (octave down) |
| 3 | 4 | 0.00 | 1.00 | up 47.9→59.0 (octave down) |
| 3 | 5 | 0.50 | 1.00 | up 51.3→63.2 (octave down) |
| 3 | 6 to 10 | 1.00 | 1.00 | up, P3's reading |
| 4 | 1 | 0.48 | 0.75 | up 38.9→51.3 (octave down) |
| 4 | 2 | 0.85 | 1.00 | up 41.7→55.0 (octave down) |
| 4 | 3 | 0.00 | 1.00 | up 44.7→59.0 (octave down) |
| 4 | 4 | 0.00 | 1.00 | up 47.9→63.2 (octave down) |
| 4 | 5 | 0.00 | 1.00 | up 51.3→67.7 (octave down) |
| 4 | 6 to 10 | 0.95 to 1.00 | 0.95 to 1.00 | up, P3's reading |
| 5 | 1, 2, 4 | 0.00 | 0.00 to 0.03 | down, P3's reading |
| 5 | 3 | 0.00 | 0.23 | down 89.4→63.2 |
| 5 | 5 | 0.00 | 0.98 | up 51.3→72.6 (octave down) |
| 5 | 6, 7 | 0.96, 1.00 | 0.96, 1.00 | up, P3's reading |
| 5 | 8, 9, 10 | 0.06, 0.38, 0.03 | unchanged | down, P3's reading |

The octave-up reading never wins; its better hypothesis is 7 to 137 behind
the best. The octave-down reading wins for tones 3 to 5 at 3 and 4 steps
(44.7 to 51.3 Hz), tones 1 and 2 at 4 steps (38.9 and 41.7 Hz) and tone 5
at 5 steps; every one of those wins is an up step.

By steps clockwise, P(up) averaged over the 10 starting tones (1, 2 and 6 to
9 steps as on P3):

| Steps clockwise | 3 | 4 | 5 |
| --- | --- | --- | --- |
| P3 (one reading) | 0.77 | 0.63 | 0.24 |
| P8 (three readings) | 1.00 | 0.97 | 0.37 |
| Shepard Group I | 0.89 | 0.75 | 0.53 |
| Shepard Group III | 0.83 | 0.75 | 0.52 |

## What it says

- **P3's starting-tone effect at 3 and 4 steps came from the single
  reading.** With the first tone free to be heard an octave lower, tones 3
  to 5 go up at 3 and 4 steps like the rest, so every starting tone now says
  up at 1 to 4 steps (P(up) 0.75 or more). P3's statement that the switch
  moves by up to 3 steps with the starting tone does not hold.
- **The half octave still depends on the starting tone**, less than on P3:
  up from tones 5, 6 and 7, down from 1, 2, 4, 8 and 10, graded from 3
  (0.23) and 9 (0.38). Tone 5 changed (down to up). This is experiment P4's
  question; Shepard reports no data by starting tone, so it stays a
  prediction, and the octave-down reading adds more up than down
  (setup), so the balance at 5 steps may still lean on the floor.
- **Against listeners the curve is now too steep the other way:** up with
  near certainty at 3 and 4 steps, where listeners are at 0.89 and 0.75,
  then 0.37 at 5 against 0.53. Overconfidence (P(up) 0 or 1 almost
  everywhere) is unchanged; that is decision P9's question.
- **Not checked:** other readings at 1, 2 and 6 to 9 steps (P3's margins
  there are large but an octave-down up step was not fitted), other seeds,
  and a start an octave down for the second tone alone.
