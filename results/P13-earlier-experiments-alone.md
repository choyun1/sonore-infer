# P13. Mistuned harmonic, two notes and ABA with the 30 Hz floor or the threshold of hearing alone

> **Under review.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** first pass, one seed, 2 mistuning conditions, 2 intervals, 3
ABA stimuli (threshold only).
**Paper target:** Fig. 7D (mistuned harmonic); Fig. 8D (ABA); none for two
notes (closest §3.1.2).
**Code:** `tools/bass/scene_mistuned_harmonic.py`,
`tools/bass/compare_two_notes.py` and `tools/grammar/compare_aba.py`, with
`--floor 30` or `--threshold-floor`; `tools/bass/variational_thresholds.py`,
`tools/bass/two_notes_summary.py`.
Design: [relative-pitch-prior.md §4](../docs/design/relative-pitch-prior.md#4-chos-answers)
(decision P13, 2026-11-05).

## Question

Shared background: [how the experiments work](README.md#how-the-experiments-work).

Decision P13 keeps the relative-pitch options off by default. Sparse
spectra and dominance were turned down on [P07](P07-earlier-experiments-sparse.md)
and [P10](P10-earlier-experiments-hearing.md). The 30 Hz f0 floor (P12)
and the threshold of hearing (P19) were only ever run together with one of
those. Does either one change earlier results on its own?

## Setup

P07's baseline runs (dense spectra, 87.5 Hz floor, `data/P07/baseline-0.8.0/`)
repeated twice, once with each option added:

- **Mistuned harmonic:** 100 Hz harmonic 1 and 200 Hz harmonic 2, 0 to 30% in
  steps of 5.
- **Two notes:** the octave and the just fifth at 0 ms, every hypothesis.
- **ABA, threshold only:** as P10, C02's runs at 1, 3 and 6 semitones,
  100 ms, kernels inferred. One stream is run to the stopping rule and two
  streams for 300 steps. C02 is the comparison. The floor is not run here:
  `compare_aba.py` has no `--floor` flag, and every ABA tone is 400 Hz or
  higher.

Variational evidence with 2000 steps, seed 0, sonore 0.10.0. The baseline
used 0.8.0, whose stimuli are bit-identical (PR #89). Cloud CPU, 4 jobs at a
time with one thread each, 52 jobs in 2.9 h. Each job took 9 to 14 min per
mistuning level, 4.5 to 10 min per two-note hypothesis, and 31 to 45 min per
ABA hypothesis.

## Results

![P13: mistuned harmonic, each option alone](plots/P13-earlier-experiments-alone.svg)

*Variational importance estimate, as the P07 and P10 plots. Drawn by `tools/results_plots.py`.*

### Mistuned harmonic

Log odds of two sources over one, Laplace estimate / variational importance
estimate (`data/P13/*/mistuned/`, baseline `data/P07/baseline-0.8.0/`):

| Mistuning (%) | 0 | 5 | 10 | 15 | 20 | 25 | 30 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100/1, baseline | −13.4 / −20.3 | −14.8 / −12.1 | −26.4 / −27.6 | −14.1 / −14.1 | −9.1 / −14.2 | −3.9 / −inf | 7.3 / 2.5 |
| 100/1, 30 Hz floor | −13.4 / −17.6 | −14.9 / −14.3 | −26.4 / −26.5 | −14.2 / −17.0 | −9.1 / −11.9 | −4.0 / −7.3 | 7.3 / 6.9 |
| 100/1, threshold | −12.9 / −14.7 | −13.5 / −13.0 | −16.2 / −16.0 | −23.2 / −26.6 | −16.9 / −17.2 | −8.6 / −10.8 | −4.1 / −2.2 |
| 200/2, baseline | −20.6 / −18.9 | −13.9 / −14.3 | −31.2 / −19.2 | −12.1 / 5.1 | 21.7 / 29.6 | 59.9 / 56.1 | 79.4 / 79.5 |
| 200/2, 30 Hz floor | −20.7 / −19.1 | −14.0 / −14.8 | −31.3 / −18.1 | −12.2 / 5.3 | 21.6 / 32.2 | 59.9 / 55.3 | 79.3 / 80.2 |
| 200/2, threshold | −20.6 / −18.9 | −13.9 / −14.3 | −31.2 / −19.2 | −12.1 / 5.1 | 21.7 / 29.6 | 59.9 / 56.1 | 79.4 / 79.5 |

Thresholds (Eqn 2, % of f0; `data/P13/thresholds_*.txt`, baseline
`data/P07/thresholds_baseline.txt`):

| Condition | Laplace | Variational importance |
| --- | --- | --- |
| 100/1, baseline | 27.4 | 25.4 (25% left out) |
| 100/1, 30 Hz floor | 27.4 | 27.5 |
| 100/1, threshold | 30.0 | 29.8 |
| 200/2, all three | 17.5 | 12.5 |

### Two notes, 0 ms

Log odds of two notes (and whistles) against the best one-source reading,
variational importance estimate / ELBO (`data/P13/two_notes_summary_*.txt`,
baseline `data/P07/two_notes_summary_baseline.txt`):

| Interval | Baseline | 30 Hz floor | Threshold |
| --- | --- | --- | --- |
| Octave, two | −34.9 / −42.2 | −28.5 / −40.0 | −34.9 / −42.2 |
| Just fifth, two | +54.1 / +53.4 | +54.6 / +53.2 | +54.1 / +53.4 |
| Just fifth, whistles | +2.4 / +3.5 | +1.2 / +1.8 | +2.4 / +3.5 |

### ABA, 100 ms, threshold

Log odds of two streams over one, variational importance estimate plus the
structure prior, as P10 (`data/P13/threshold/aba/`):

| Semitones | C02 | P10 (both options) | Threshold alone |
| --- | --- | --- | --- |
| 1 | −18.7 | −18.5 | −19.3 |
| 3 | 13.2 | 13.5 | 12.1 |
| 6 | 97.2 | 96.0 | 96.7 |

## What it says

- **The 30 Hz floor changes nothing.** Every fit reaches the same mode as
  the baseline. Laplace estimates are within 0.1 everywhere, so the only
  change is the prior's density, 0.06 nats per source. The differences in
  the importance estimates (up to 6.4, on the octave) are estimator noise:
  effective sample sizes are 1 to 9, and the ELBO moves by at most 2.2.
- **The threshold of hearing changes only 100 Hz harmonic 1.** At 200/2, on
  both two-note intervals and on ABA (within 1.1 of C02), it gives the same
  numbers as the baseline. At 200/2 and two notes they match to 0.1. At
  100/1 the log odds fall by about 10 from 10% mistuning up (Laplace), and two
  sources are never preferred up to 30%. P10 found the same with dominance
  alone (−8 to −17), and said only dominance did it, because "the
  threshold floor alone was not run". So both options push 100/1 the same
  way, away from listeners (about 1–5%).
- **Why only 100/1:** the threshold floor rises above the flat 20 dB floor
  only below 67 Hz (P10, Setup). It is inferred, not checked, that only at
  100 Hz does the two-source fit of the lowest component reach those
  channels.
- **For decision P13:** the 30 Hz floor could become a default without
  changing any result checked here. Cho made it the default on 2026-11-06,
  so the threshold runs below now need `--paper-floor` to reproduce exactly. It lets f0 go down to the lower limit
  of pitch (P12). The threshold of hearing widens the 100/1 gap with
  listeners, as dominance does, so it should not become a default as it
  stands.
- **Not checked:** other seeds, the other mistuning conditions, two notes
  at other asynchronies, and the floor on ABA.

## Reproduce

```
V="--steps 15000 --tolerance 0.5 --variational-steps 2000"
for opt in "--floor 30" "--threshold-floor --paper-floor"; do
  for p in 0 5 10 15 20 25 30; do
    python tools/bass/scene_mistuned_harmonic.py --f0 100 --harmonic 1 --percents $p $V $opt
    python tools/bass/scene_mistuned_harmonic.py --f0 200 --harmonic 2 --percents $p $V $opt
  done
  python tools/bass/compare_two_notes.py --interval octave --asynchronies 0 --only two $V $opt
  # likewise one@100, one@200, one@400 for the octave; two, one@100, one@200, one@300, whistles for just_fifth
done
for st in 1 3 6; do
  # compare_aba.py has no --paper-floor; since 2026-11-06 its whistles get the 30 Hz floor
  python tools/grammar/compare_aba.py --semitones $st --intervals 0.1 --hypotheses one --steps 15000 --tolerance 0.5 --infer-kernel --variational-steps 2000 --threshold-floor
  python tools/grammar/compare_aba.py --semitones $st --intervals 0.1 --hypotheses two --steps 300 --infer-kernel --variational-steps 2000 --threshold-floor
done
python tools/bass/variational_thresholds.py results/data/P13/floor30/mistuned/*.txt
python tools/bass/two_notes_summary.py results/data/P13/floor30/two_notes/*.txt
```

Each command's output is one file under `results/data/P13/<option>/`, and
`data/P13/jobs.tsv` lists all 52 jobs. Each ran with `OMP_NUM_THREADS=1`.
