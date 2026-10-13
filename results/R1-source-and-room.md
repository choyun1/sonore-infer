# R1. Source and room (RT60) from one reverberant sound

> **Under review by Adrian Cho.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** done (2026-10-11). The building blocks, the inference
(step 5) and experiment (a) are merged. Experiment (b), atypical rooms,
is [R2](R2-atypical-rooms.md).

**Paper target:** none in Cusimano et al. (2024), which has no room model;
its §4.3 names reverberation as one of the aspects of sound generation a
model of scenes should include, citing Traer & McDermott (2016). A
comparison with Traer & McDermott's listeners is optional and, if done, will
be labelled a comparison, not a validation. Design:
[source and room](../docs/design/source-reverb.md) (R1–R8, accepted by Cho
2026-10-10).

## Question

Shared background: [how the experiments work](README.md#how-the-experiments-work).


One source plays in one room. From the single sound that results, can the
model infer the source's spectrotemporal statistics and the room's RT60?
This page is
experiment (a), recovery, from the design (R7): sounds the model could have
made, 10 seeds × RT60 in {0.2, 0.4, 0.8} s × three source settings. Success
is the truth falling in the 90% posterior interval in about 90% of runs,
per parameter. Experiment (b) is [R2](R2-atypical-rooms.md).

## Steps

| Step | What | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Source prior (`SpectrogramPrior`), matching sonore's `gaussian_spectrogram` | Whitening sonore's drawn levels gives back its standard normals (to 1e-9, 3 seeds) | PR [#18](https://github.com/choyun1/sonore-infer/pull/18), tests |
| 2 | Room's expected band energy (`RoomGain`) | Mean of 64 rendered noise bursts within 0.1 dB of the model on average (after the fix found in step 3); a single draw is off by −0.7 to −1.6 dB on average, sd 2.8–4.2 dB | PR [#19](https://github.com/choyun1/sonore-infer/pull/19); `data/R1/room_gain_check.txt` (`tools/room_gain_check.py`) |
| 3 | Block-power likelihood (`BlockPower`) | One render against the model at RT60 0.4 s, DRR 10 dB, all cells: bias −1.3 dB, sd 3.6 dB (the likelihood uses these) | PR [#21](https://github.com/choyun1/sonore-infer/pull/21); `data/R1/block_power_check.txt` (`tools/block_power_check.py`) |
| 3 | Likelihood slices through the truth, one sound (RT60 0.4 s, DRR 10 dB) | At floors 60 and 40 dB every slice peaks at the truth; at a 20 dB floor (BASS's) RT60 is nearly flat (0.1 s costs 1.3 nats) | `data/R1/likelihood_slices.txt` (`tools/likelihood_slices.py`) |
| 4 | RT60 prior from Traer & McDermott's survey (270 IRs, run by Cho) | Gamma, shape 1.97, scale 0.220 s (median 0.362 s), better than a lognormal by 18 AIC; typical DRR 9.9 dB | PR [#22](https://github.com/choyun1/sonore-infer/pull/22); constants in `src/sonore_infer/room.py` (`tools/rt60_prior_fit.py`) |
| 4 | Band RT60s of the 270 survey IRs against sonore's regression (relative to synth_ir rooms) | Median within 0.94–1.06 from 200 Hz to 5 kHz; 1.13–1.21 at 82–158 Hz and 1.14–1.17 at 5.7–7.1 kHz; outdoor rooms deviate most. Matters only for real rooms, not for experiments (a) and (b), whose sounds use synth_ir | Cho's rerun; step record row 18 in the project files (`source-reverb/evaluation.md`) |
| 5 | Inference over RT60 and the source's statistics (`sourceroom.infer`) | Exact Hessian, 3–5 s per RT60 point. The posterior is far narrower than the design's RT60 grid (on one test sound RT60 sd about 1.5%, against 24% between grid points), so the grid is refined around the peak, and the source settings are integrated in closed form plus Laplace. This departs from design R5. Tests: Hessian and Laplace match autograd and `log_evidence` | PR [#26](https://github.com/choyun1/sonore-infer/pull/26) (merged); `tests/test_sourceroom.py` |
| 5 | Fit fix: the loud setting's 10 sounds returned the prior unchanged on the first run (the fit started so far below the sound's level that the likelihood was flat); fits now start at the sound's overall level, and those 10 were rerun | Two of 180 inferences left out (degenerate interval) | PR [#35](https://github.com/choyun1/sonore-infer/pull/35) (`e3c7fb7`) |

The survey files are not redistributed; the prior fit runs where they are
downloaded, so its raw output is not here.

## Experiment (a): recovery

Run by `tools/recovery.py`: 10 seeds × RT60 {0.2, 0.4, 0.8} s × three
source settings (sonore's defaults, and two corners inside the prior box:
fine and slow with a small sd, coarse and fast with a large one), each
400 ms, at a 60 dB floor, and inferred at the measured likelihood sigma
(3.6 dB) and twice it: 90 sounds, 180 inferences, median 305 s each. Two
inferences had a degenerate interval and are left out
(`data/R1/recovery_summary.txt`).

Share of runs whose 90% interval contains the truth, all settings pooled:

| Sigma | RT60 | Band correlation | Time correlation | Level sd |
| --- | --- | --- | --- | --- |
| 3.6 dB (measured) | 66% | 78% | 74% | 78% |
| 7.2 dB (twice) | 90% | 86% | 56% | 79% |

Per setting and RT60 (9 or 10 runs each), the median of posterior median
over truth for RT60 is 0.98 to 1.03, and the 90% interval's width (95% over
5% quantile) is 1.02 to 1.08 at 3.6 dB and 1.04 to 1.18 at 7.2 dB.

## What it says

- RT60 is recovered: the posterior median sits within about 3% of the
  truth for every room and source setting.
- At the measured sigma the intervals are too narrow: they contain the
  truth in 66% of runs, not 90%. At twice the sigma they contain it in
  90%. The design's evaluation predicted this overconfidence from residuals
  that are correlated across cells while the likelihood treats them as
  independent.
- The source's time correlation reads long at twice the sigma: 1.7 to 1.8
  times the truth (median per cell), and coverage 56%. The coarse-fast
  setting's level sd reads 0.8 to 0.95 times the truth.
- Laplace could not be checked by importance sampling at 1,599
  dimensions: importance sampling minus Laplace at the best RT60 has median
  −3.5 nats (range −96.6 to 5.6), with a median effective sample size of
  1.9 of 128.

## Reproduce

```
python tools/recovery.py --out recovery.jsonl --shard 0 --shards 4   # and shards 1-3
python tools/recovery.py --summary recovery.jsonl
```

Raw outputs: `data/R1/recovery.jsonl` (one line per inference) and
`recovery_summary.txt` (the `--summary` output; rerunning it on the
committed file gives the same text).

## Open

- Intervals are overconfident at the measured sigma; a likelihood that
  models correlated residuals, or a larger sigma, would be needed for
  calibrated intervals.
- At BASS's 20 dB floor one sound barely constrains RT60 (see
  [R2](R2-atypical-rooms.md)), which matters for combining this with
  milestone (b)'s model.
- Licence of the survey's original files is unchecked.
