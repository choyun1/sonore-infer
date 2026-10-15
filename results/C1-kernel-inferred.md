# C1. Gaussian-process sigma and lengthscale inferred

> **Under review.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** done (2026-10-11). **Paper target:** Fig. 7D (right) for
100 Hz h1; none for the octave. The paper infers each Gaussian process's
hyperparameters per sound (Table A.2); A1 to B4 fix them at the table's
medians. **Code:** `--infer-kernel`, milestone (c) step 3, PR
[#34](https://github.com/choyun1/sonore-infer/pull/34) (`53dcba2`).
Design: [milestone (c)](../docs/design/milestone-c.md), decision C3.

## Question

Shared background: [how the experiments work](README.md#how-the-experiments-work).

Two results stand out against expectations: 100 Hz h1's threshold is 25%
where the paper's model has about 45% ([A7](A7-variational-evidence.md),
[A8](A8-ten-seeds.md)), and the octave stays one source up to 240 ms
([B2](B2-two-notes-variational.md)). Does inferring each Gaussian
process's sigma and lengthscale, as the paper does, change either?

## Setup

A7's model for 100 Hz h1 at the paper's 7 mistunings, and B2's octave at
its 8 asynchronies (one@200 against two), with each Gaussian process's
sigma and lengthscale added as latents with the paper's priors. Variational
evidence with 2000 steps, one seed. The structure prior counts n!
labellings (milestone (c) step 1), which adds 0.69 to two sources' log
odds against A7 and B2.

## Results

Log odds of two sources over one, variational importance estimate (`data/C1/`):

| Mistuning (%) | 0 | 5 | 10 | 20 | 30 | 40 | 50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kernel fixed (A7) | −21.7 | −12.1 | −13.9 | −12.8 | 9.9 | 19.6 | 24.5 |
| Kernel inferred | −18.6 | −13.7 | −18.7 | −5.3 | 11.6 | 24.6 | 27.3 |

Threshold, kernel inferred: 25% by Eqn 2 with every estimate (Laplace,
ELBO, both importance estimates), and 30% by BASS's rule, the same as with
the kernel fixed (`data/C1/thresholds.txt`).

| Octave asynchrony (ms) | 0 | 10 | 20 | 40 | 80 | 120 | 160 | 240 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kernel fixed (B2) | −33.6 | −33.1 | −28.3 | −27.0 | −26.3 | −23.4 | −25.4 | −21.9 |
| Kernel inferred | −44.3 | −38.7 | −40.4 | −32.0 | −25.1 | −21.9 | −26.1 | −23.5 |

## Plots

![C1-kernel-inferred](plots/C1-kernel-inferred.svg)

*Log odds of two sources over one, variational importance estimate, from [`data/C1/`](data/C1/), [`data/A7/`](data/A7/) and [`data/B2/`](data/B2/).*

## What it says

- Inferring sigma and lengthscale does not close the 100 Hz h1 gap.
- The octave stays one source at every asynchrony. With the kernel
  inferred it leans further toward one source at 0 to 40 ms, by 5 to 12,
  and is about the same from 80 ms on. Seed SDs at the octave are 1.1 to
  4.6 ([B4](B4-two-notes-seeds.md)), so the shift at 0 to 40 ms is likely
  real but small next to the margins.

## Reproduce

```
python tools/scene_mistuned_harmonic.py --f0 100 --harmonic 1 --percents 30 \
    --infer-kernel --variational-steps 2000 > a_30.txt      # one mistuning per job
python tools/compare_two_notes.py --interval octave --asynchronies 0.24 \
    --only one@200 two --infer-kernel --variational-steps 2000 > octave_0.24.txt
python tools/two_notes_summary.py data/C1/octave_*.txt
```

`thresholds.txt` is made by a local script that also computes BASS's rule
(not committed, as in A6). Eqn 2 alone can be checked with
`sonore_infer.thresholds` on the log odds above.

## Open

- One seed; one@200 was the only one-source start fitted for the octave
  (it was the best start at every asynchrony with the kernel fixed).
- Two-source octave fits have floored Hessian eigenvalues at several
  asynchronies, so the Laplace-based importance estimates there are
  useless; the variational estimates agree with the ELBO within 11.
- What else could raise 100 Hz h1's threshold is open. One untested guess is the
  paper's further latents raising the cost of the second source.
