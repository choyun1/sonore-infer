# P10. Mistuned harmonic, two notes and ABA with a hearing threshold and spectral dominance

> **Under review.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** first pass, one seed, 2 mistuning conditions, 2 intervals, 3
ABA stimuli.
**Paper target:** Fig. 7D (mistuned harmonic); Fig. 8D (ABA); none for two
notes (closest §3.1.2).
**Code:** `tools/bass/scene_mistuned_harmonic.py`,
`tools/bass/compare_two_notes.py` and `tools/grammar/compare_aba.py`, each
with `--threshold-floor --dominance`; `tools/bass/variational_thresholds.py`,
`tools/bass/two_notes_summary.py`.
Design: [relative-pitch-dominance.md](../docs/design/relative-pitch-dominance.md),
decisions P22 and P23.

## Question

Shared background: [how the experiments work](README.md#how-the-experiments-work).

On Shepard pairs, a threshold of hearing (P19) and spectral dominance (P20)
bring the model's crossover close to listeners'
([P9](P9-shepard-listener-noise.md)). Before either could become a default
(decision P13), what do they do to results they were not designed for? As
[P7](P7-earlier-experiments-sparse.md) did for the sparse spectrum and the
30 Hz floor.

## Setup

P7's runs repeated with both options added, and C2's ABA runs at 1, 3 and
6 semitones, 100 ms, repeated with both options added:

- **Mistuned harmonic** (100 Hz harmonic 1, 200 Hz harmonic 2, 0 to 30% in
  steps of 5) and **two notes** (octave and just fifth at 0 ms): as P7,
  with sparse spectra and a 30 Hz floor. P7's own runs are the comparison.
- **ABA**: as [C2](C2-aba-streaming.md), kernels inferred, one stream run
  to the stopping rule, two streams 300 steps. The hypotheses are whistles
  only, so the sparse spectrum and the f0 floor do not apply, and C2's runs
  are the comparison.

The threshold floor clips each channel at the threshold of hearing
[Terhardt et al. 1982, Eq. 8]; in the cochleagram's units it is above the
flat 20 dB floor only below 67 Hz (73 dB at 20 Hz, 25 at 51 Hz)
[measured, `Cochleagram.channel_floor_db`]. Dominance sets each channel's
likelihood SD to 10 dB / √w(f), with w 0.48 at 100 Hz, 0.76 at 200 Hz, 1
at 700 Hz and 0.83 at 2 kHz [Terhardt Eq. 12, computed]. Variational
evidence with 2000 steps, seed 0. sonore 0.8.0 (the stimuli are
bit-identical in 0.10.0, measured in PR #89). Cloud CPU, 4 jobs at a
time, one thread each: 26 to 40 min per mistuning level and per ABA
hypothesis, 5 to 16 min per two-note hypothesis.

## Results

### Mistuned harmonic

Eqn 2 thresholds (%), variational importance estimate
(`data/P10/thresholds.txt`; dominance only, run after the rest at Cho's
request, `data/P10/thresholds_dominance_only.txt` from
`data/P10/dominance_only/`; P7's from `data/P7/thresholds.txt`):

| Condition | P7 (sparse, 30 Hz floor) | With both options | Dominance only |
| --- | --- | --- | --- |
| 100 Hz, harmonic 1 | 22.4 | 30.0 (two sources never preferred) | 30.0 (two sources never preferred) |
| 200 Hz, harmonic 2 | 12.5 | 12.5 | not run |

The other three estimates agree (100/1: 29.7 to 30.0; 200/2: 12.5).
Eqn 2's threshold is the area under p(one source) over the range, so 30.0
is the top of the range, not a crossing.

Log odds of two sources over one, variational importance estimate:

| Mistuning (%) | 0 | 5 | 10 | 15 | 20 | 25 | 30 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100/1, P7 | −20.5 | −12.5 | −17.2 | −9.8 | −4.3 | 5.8 | 17.5 |
| 100/1, both options | −12.3 | −13.1 | −16.2 | −26.2 | −18.2 | −11.6 | −11.3 |
| 100/1, dominance only | −14.0 | −15.1 | −17.2 | −12.4 | −14.2 | −10.0 | −8.2 |
| 200/2, P7 | −17.0 | −21.7 | −5.5 | 16.6 | 43.1 | 77.3 | 105.5 |
| 200/2, both options | −19.3 | −20.6 | −6.4 | 14.3 | 40.9 | 72.3 | 100.1 |

At 100/1, 30%, the two-source modes still fit better than the one-source
modes (log joint at the mode −10518.5 against −10633.4), but their evidence
is lower (−10679.6 against −10670.7): the better fit no longer pays for
the extra parameters (`data/P10/mistuned/f0-100_h1_p30.txt`).

### Two notes, 0 ms

Log odds against the best one-source reading, variational
(`data/P10/two_notes_summary.txt`). The best one-source reading is
`one@200` at the octave and `one@100` at the fifth, as in P7.

| Interval | P7: two / whistles | With both options: two / whistles |
| --- | --- | --- |
| Octave | −38.9 / | −35.2 / |
| Just fifth | −12.7 / −57.0 | −10.3 / −57.5 |

### ABA, 100 ms

Log odds of two streams over one, variational importance estimate
(`data/P10/aba/`; C2 from `data/C2/one_converged/` and
`data/C2/run_300steps/`, as C2's map):

| Semitones | C2 | With both options |
| --- | --- | --- |
| 1 | −18.7 | −18.5 |
| 3 | 13.2 | 13.5 |
| 6 | 97.2 | 96.0 |

## What it says

- **ABA and two notes are unchanged** (within 4). ABA's tones are at 1000
  Hz and above, where the threshold floor does nothing and dominance is
  0.83 to 1. The two-note log odds change by at most 3.7, and every winner is the same.
- **The 200 Hz harmonic 2 threshold is unchanged** at 12.5%; the log odds
  move by at most 5.4.
- **At 100 Hz harmonic 1 the model no longer hears the mistuned
  fundamental as separate**, up to 30%: the log odds stay between −11 and
  −26, where P7 crossed zero between 20 and 25%. Listeners' thresholds for
  mistuned harmonics are about 1 to 5% (Moore, Glasberg & Peters 1986, as in
  the index's "Against the paper"), so this moves the model further from
  listeners. It moves toward the paper's model (about 45%), which by the
  project's rule is not the reference.
- **Dominance alone does it.** With dominance and no threshold floor the
  log odds at 100/1 stay between −8 and −17 and two sources are never
  preferred, as with both options (P23 (B) for this condition; the
  threshold floor alone was not run). The floor changes no channel at 67 Hz
  and above, and the mistuned component is at 100 to 130 Hz, where dominance
  is about 0.5. Why the halved weight costs the whistle its win is
  inferred, not checked: its gain in fit is counted at about half, and no
  longer pays for its parameters.
- **For decision P13:** the two options help on Shepard pairs and change
  nothing measurable on two notes, streaming or 200/2, but they widen the
  existing gap with listeners at 100/1. Dominance as a fixed per-channel
  weight treats a resolved low component as weak evidence everywhere, which
  Terhardt's weight was not built for (it scores the pitch salience of a
  component, design P20), so the 100/1 result is a reason not to make it a
  default as it stands.
- **Not checked:** other seeds, the threshold floor alone, dominance alone
  outside 100/1, the fifth at other
  asynchronies, the other mistuning conditions, and ABA at other intervals
  and 9 and 12 semitones.

## Reproduce

```
O="--threshold-floor --dominance --variational-steps 2000"
python tools/bass/scene_mistuned_harmonic.py --f0 100 --harmonic 1 --percents 0 --steps 15000 --tolerance 0.5 --sparse-spectrum --floor 30 $O
python tools/bass/compare_two_notes.py --interval octave --asynchronies 0 --only two --steps 15000 --tolerance 0.5 --sparse-spectrum --floor 30 $O
python tools/grammar/compare_aba.py --semitones 3 --intervals 0.1 --hypotheses one --steps 15000 --tolerance 0.5 --infer-kernel $O
python tools/grammar/compare_aba.py --semitones 3 --intervals 0.1 --hypotheses two --steps 300 --infer-kernel $O
python tools/bass/variational_thresholds.py results/data/P10/mistuned/*.txt
python tools/bass/scene_mistuned_harmonic.py --f0 100 --harmonic 1 --percents 0 --steps 15000 --tolerance 0.5 --sparse-spectrum --floor 30 --dominance --variational-steps 2000
python tools/bass/two_notes_summary.py results/data/P10/two_notes/*.txt
```

One run per mistuning level, two-note hypothesis and ABA hypothesis, as
in `data/P10/`.
