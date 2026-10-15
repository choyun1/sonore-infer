# B4. Two notes over seeds

> **Under review.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** done (2026-10-12). **Paper target:** none (see
[B1](B1-two-notes-laplace.md)); the paper averaged its model over 10 seeds.
**Code:** `--seed` in `tools/compare_two_notes.py`, PR
[#28](https://github.com/choyun1/sonore-infer/pull/28) (`5e840c2`);
`tools/run_seeds.py`, PRs [#30](https://github.com/choyun1/sonore-infer/pull/30)
and [#33](https://github.com/choyun1/sonore-infer/pull/33); the n!
structure prior, PR [#31](https://github.com/choyun1/sonore-infer/pull/31).
The same ten-seed run covers milestone (a) in [A8](A8-ten-seeds.md).

## Question

Shared background: [how the experiments work](README.md#how-the-experiments-work).

B2 used one seed. How much do its log odds move with the seed, and does
any conclusion change?

## Setup

The Adam fits start from fixed points and are deterministic, so the seed
changes only the sampling in the evidence estimates: the spread is the
estimates' own noise for the same mode.

1. **Close cases, 5 seeds (2026-10-11).** The octave at 0 and 240 ms (two
   vs one@200), and the just fifth's harmonic plus whistles at 10 and 80 ms,
   seeds 1 to 4 in the container; seed 0 is B2.
2. **All stimuli, 10 seeds (2026-10-12).** Every B2 stimulus and control,
   seeds 0 to 9, all on Cho's PC with `tools/run_seeds.py`, with the n!
   structure prior of milestone (c) (which adds 0.69 to two notes' log
   odds and 3.2 to 10.6 to the whistles' against B2).

## Results

Ten seeds, log odds over the best one source, variational estimate, mean ±
SD (`data/A8/summary.txt`; ELBO and min-max there too):

| Interval | 0 ms | 10 | 20 | 40 | 80 | 120 | 160 | 240 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tritone, different spectra | 41.6 ± 2.7 | 43.1 ± 3.1 | 49.9 ± 2.2 | 56.0 ± 2.6 | 53.2 ± 1.8 | | | |
| Tritone | 34.3 ± 2.7 | 40.0 ± 2.5 | 40.9 ± 2.3 | 48.0 ± 2.3 | 58.1 ± 1.9 | | | |
| Just fifth | 69.3 ± 2.3 | 59.3 ± 1.7 | 62.5 ± 2.5 | 67.0 ± 2.0 | 59.1 ± 1.6 | | | |
| Just fifth, whistles | 14.4 ± 1.6 | 6.1 ± 1.7 | 11.4 ± 4.3 | 16.5 ± 4.1 | 7.8 ± 3.0 | | | |
| Octave | −32.1 ± 4.6 | −30.8 ± 1.1 | −29.8 ± 1.4 | −27.4 ± 1.7 | −25.6 ± 2.1 | −24.5 ± 1.3 | −22.7 ± 2.6 | −20.9 ± 1.9 |

Controls: 200 Hz −20.4 ± 2.4, 282.8 Hz −17.9 ± 1.5, 300 Hz −19.8 ± 1.6,
400 Hz −29.4 ± 2.2. The tritones' whistles lose to one source (means −56 to −109).

Five seeds on the close cases (`data/B4/`, seed 0 from `data/B2/`): octave
0 ms −34.4 ± 2.6, octave 240 ms −22.0 ± 2.1; fifth whistles at 10 ms
4.1 ± 4.6 (ELBO −1.2 ± 1.6), at 80 ms 2.3 ± 3.6 (ELBO 0.2 ± 0.6).

## Plots

![B4-two-notes-seeds](plots/B4-two-notes-seeds.svg)

*Mean log odds over 10 seeds (Cho's PC, variational estimate) ± 1 SD, from [`data/A8/summary.txt`](data/A8/summary.txt): two notes against the best one source, and for the fifth also harmonic plus whistles.*

## What it says

- Every sign is the same as with one seed. The tritones and the fifth are
  two notes at every asynchrony; the octave is one source up to 240 ms,
  rising by about 11 over 0 to 240 ms without crossing.
- Two notes' log odds move by an SD of 1.1 to 4.6 between seeds, small
  next to margins of 20 or more.
- The fifth's whistles beat one source by 6 to 17 with ten seeds (a tie in
  the five-seed container run), but two notes beat the whistles by 50 to 55
  either way.
- The machine matters for a few stimuli. Same seed, same code, Cho's PC
  against the container: most cells agree to within about 1, but the fifth
  at 0 ms differs by 10 (56.0 against 66.1) and the octave at 0 ms by 5.
  The mode is the same, the Hessian comes out differently, and the
  variational fit starts from Laplace-whitened coordinates. So runs from
  different machines are not pooled here.

## Reproduce

```
python tools/run_seeds.py --out seeds_out --seeds 0 1 2 3 4 5 6 7 8 9
python tools/compare_two_notes.py --seed 1 --variational-steps 2000 \
    --interval octave --asynchronies 0 0.24 --only one@200 two > octave_seed1.txt
python tools/compare_two_notes.py --seed 1 --variational-steps 2000 \
    --interval just_fifth --asynchronies 0.01 0.08 --only one@100 one@200 whistles > fifth_seed1.txt
```

`data/A8/summary.txt` is made by a local averaging script (`average.py`,
in the project files with the 550 raw outputs, 2.2 MB), not committed
because it also computes BASS's threshold rule.

## Open

- Two-source fits start at the truth.
- Importance sampling's effective sample sizes stay low (mostly 1 to 10 of
  256).
