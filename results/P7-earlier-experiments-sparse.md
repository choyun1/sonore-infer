# P7. Mistuned harmonic and two notes, sparse spectra and a 30 Hz floor

> **Under review.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** first pass, one seed, 2 mistuning conditions and 2 intervals.
**Paper target:** Fig. 7D (mistuned harmonic); none for two notes (closest
§3.1.2).
**Code:** `tools/bass/scene_mistuned_harmonic.py`,
`tools/bass/compare_two_notes.py` (both with `--sparse-spectrum --floor 30`),
`tools/bass/sparse_options.py`, `tools/bass/variational_thresholds.py`,
`tools/bass/two_notes_summary.py`.
Design: [relative pitch prior](../docs/design/relative-pitch-prior.md),
decision P14, second check.

## Question

Shared background: [how the experiments work](README.md#how-the-experiments-work).

With sparse spectra and a 30 Hz floor the model hears an octave Shepard tone
as one tone ([P5](P5-shepard-one-tone-sparse.md)). Before either option
could become a default (decision P13), decision P14 asked what they do to
earlier results:

- **The mistuned harmonic**, at the two conditions that differ from the
  paper: 100 Hz harmonic 1 and 200 Hz harmonic 2
  ([A7](A7-variational-evidence.md); "Against the paper" in the
  [index](README.md#against-the-paper)). A sparse spectrum makes "a tone with
  that harmonic missing, plus a whistle" cheaper, which should lower the
  thresholds (design, P14).
- **Two simultaneous notes**, the octave and the just fifth with no
  asynchrony ([B2](B2-two-notes-variational.md)).

## Setup

As [A7](A7-variational-evidence.md)'s converged runs (0 to 30% in steps of
5) and [B2](B2-two-notes-variational.md)'s converged runs at 0 ms, with
every source given the options of P5:

- Each start with a harmonic 30 dB down also gets a twin with that harmonic
  marked absent. For the mistuned harmonic, the one-source hypothesis also
  gets a start with the mistuned harmonic 30 dB down, so it too can drop it.
  For two notes, `one@100` starts with the harmonics of 100 Hz that the
  stimulus lacks 30 dB down, so it gets a sparse twin; `one@200`, `one@300`,
  `one@400` and `two` have no missing harmonics in their starts.
- Each start is fitted and scored on its own, and a hypothesis's evidence
  is the log-sum-exp over its distinct modes (`sparse_options.py`; the raw
  outputs list each mode). Where a mode's importance estimate fails, its
  ELBO stands in.

Model as B2: FFT cochleagram with BASS's gain, fixed kernels, likelihood SD
10 dB. sonore 0.8.0 (A7 and B2 used 0.6.0). One seed. Cloud CPU, 4 jobs at a
time: 28 to 37 min per mistuning level, 5 to 17 min per two-note hypothesis.

## Results

### Mistuned harmonic

Eqn 2 thresholds (%), 0 to 30% in steps of 5, by the four estimates
(`data/P7/thresholds.txt`; A7's from the same tool on its converged outputs,
`data/P7/thresholds_A7.txt`):

| Condition | A7: Laplace / variational | Options: Laplace / variational |
| --- | --- | --- |
| 100 Hz, harmonic 1 | 27.4 / 25.2 | 22.1 / 22.4 |
| 200 Hz, harmonic 2 | 17.5 / 12.5 | 12.5 / 12.5 |

Log odds of two sources over one, variational (importance sampling from the
variational fit):

| Mistuning (%) | 0 | 5 | 10 | 15 | 20 | 25 | 30 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100/1, A7 | −23.5 | −12.7 | −28.9 | −14.2 | −15.1 | −inf | 3.1 |
| 100/1, options | −20.5 | −12.5 | −17.2 | −9.8 | −4.3 | 5.8 | 17.5 |
| 200/2, A7 | −18.2 | −14.2 | −17.7 | 5.9 | 28.6 | 56.6 | 79.6 |
| 200/2, options | −17.0 | −21.7 | −5.5 | 16.6 | 43.1 | 77.3 | 105.5 |

### Two notes, 0 ms

Log odds against the best one-source reading, variational
(`data/P7/two_notes_summary.txt`; B2 from `data/B2/converged/`):

| Interval | B2: best one | B2: two | B2: whistles | Options: best one | Options: two | Options: whistles |
| --- | --- | --- | --- | --- | --- | --- |
| Octave | one@200 | −33.1 | | one@200 | −38.9 | |
| Just fifth | one@100 | **+54.5** | +7.8 | one@100 | **−12.7** | −57.0 |

At the fifth, `one@100` has two modes. With its missing harmonics present
but quiet (as in B2) it scores −10053.2, close to B2's −10063.7; with them
marked absent it scores −9999.8, 53 higher, and wins
(`data/P7/two_notes/just_fifth_0ms_one@100.txt`).

## What it says

- **The mistuned harmonic moves as expected, a little.** At 100 Hz
  harmonic 1 the threshold drops from about 25% to 22%, and the log odds
  rise by 4 to 14 from 10% up. At 200 Hz harmonic 2 the variational
  threshold stays at 12.5%; the log odds rise by 11 to 26 from 10% up.
  Listeners' thresholds are about 1 to 5% (index, "Against the paper", from
  Moore, Glasberg & Peters 1986),
  so the options do not close that gap. The paper's 100/1 value (about
  45%) moves further away.
- **The just fifth flips from two notes to one.** With a sparse spectrum,
  one harmonic source at 100 Hz can drop the harmonics of 100 Hz the fifth
  lacks (the fifth's components are the harmonics of 200 and 300 Hz, all
  harmonics of 100 Hz) at little cost, and it then beats two notes by
  12.7. In B2 two notes won by 54.5. This changes milestone (b)'s result
  for the fifth, if the options become defaults.
- **The octave is unchanged:** one source, by 39 (B2: 33).
- **Not checked:** the fifth at other asynchronies and the tritones, other
  seeds, and the other two mistuning conditions. Whether listeners hear a
  synchronous just fifth of these complexes as one note or two is a
  question for the literature, not checked here.
