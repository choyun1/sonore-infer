# A8. Mistuned harmonic over ten seeds

> **Under review by Adrian Cho.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** done (2026-10-12). **Paper target:** Fig. 7D (right); the paper
averaged its model over 10 seeds. **Code:** `tools/run_seeds.py`, PRs
[#30](https://github.com/choyun1/sonore-infer/pull/30) and
[#33](https://github.com/choyun1/sonore-infer/pull/33); its comment
on the grid fixed in PR [#36](https://github.com/choyun1/sonore-infer/pull/36).
The same run covers milestone (b) in [B4](B4-two-notes-seeds.md).

## Question

Shared background: [how the experiments work](README.md#how-the-experiments-work).

A7's thresholds come from one seed. Do ten seeds change them, and is the
100 Hz h1 gap (A7: 25%, against about 45% in the paper and 40% from BASS's
own code, [A6](A6-bass-own-code.md)) seed noise?

## Setup

A7's model and settings (300 Adam steps, variational evidence with 2000
steps, likelihood SD 10 dB), on A7's four conditions, seeds 0 to 9, all on
Cho's PC. The fits are deterministic, so the seed changes only the
sampling in the evidence estimates.

The run used mistunings of 0 to 30% in steps of 5, not the paper's grid
(0, 5, 10, 20, 30, 40, 50); `run_seeds.py` wrongly called it the paper's
until PR #36. Every condition has crossed well before 30% (mean log odds +8 to
+106 there), so thresholds can also be read on the paper's points up to 30%.

## Results

Thresholds (% of f0) from the mean log odds over ten seeds, variational
importance estimate (`data/A8/summary.txt`; the ELBO agrees within 0.5).
BASS's rule is from the local averaging script, as in A6.

| Condition | 5% grid, Eqn 2 | Per seed, Eqn 2 | Paper's points to 30%, Eqn 2 / BASS's rule | A7 (one seed) | BASS's code (A6) | Paper (Fig. 7D, by eye) |
| --- | --- | --- | --- | --- | --- | --- |
| 100 Hz h1 | 27.4 | 24.5–27.5 | 25 / 30 | 25 / 30 | 39.6 / 40 | about 45 |
| 200 Hz h2 | 12.6 | 12.5–14.1 | 15 / 20 | 15 / 20 | 14.8 / 20 | about 20 |
| 200 Hz h3 | 22.5 | 20.7–22.5 | 24.9 / 30 | 25 / 30 | | about 21 |
| 400 Hz h3 | 12.5 | 12.5 | 15 / 20 | 15 / 20 | | about 13 |

Per mistuning, the log odds vary by an SD of 0.4 to 3.7 between seeds,
with one 7.8 (100 Hz h1 at 0%, far below zero).

## Plots

![A8-ten-seeds](plots/A8-ten-seeds.svg)

*Log odds of two sources over one against mistuning, from [`data/A8/summary.txt`](data/A8/summary.txt) and [`data/A7/`](data/A7/). Bars are ± 1 SD over seeds. Above zero favors two sources.*

## What it says

- Ten seeds change no threshold on the paper's points.
- The seed noise is small next to the grid's steps, so the grid, not the
  seed, limits the threshold.
- The 100 Hz h1 gap (25–30% against 40 in BASS's code and about 45 in the
  paper) is not seed noise.

## Reproduce

```
python tools/run_seeds.py --out seeds_out --seeds 0 1 2 3 4 5 6 7 8 9
```

`data/A8/summary.txt` is made by a local averaging script (`average.py`,
in the project files with the 550 raw outputs, 2.2 MB), not committed
because it also computes BASS's threshold rule.

## Open

- Same seed, same code on two machines differs by more than the seed noise
  in a few cells (in (a): 200 Hz h3 at 10%, 400 Hz h3 at 5% and 20%, where
  the variational log odds move by 1 to 7). See [B4](B4-two-notes-seeds.md).
- Mistunings above 30% were not rerun.
