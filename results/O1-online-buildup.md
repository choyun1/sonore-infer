# O1. Buildup of streaming, batch and online

> **Under review by Adrian Cho.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** first pass done (2026-10-14). **Paper target:** Fig. 8F (right), App. C.11.
**Code:** `src/sonore_infer/online.py`, `tools/online_buildup.py`,
`src/sonore_infer/stimuli/buildup.py`. Design:
[online inference](../docs/design/online.md), step 2.

## Question

Shared background: [how the experiments work](README.md#how-the-experiments-work).

Listeners hearing a repeated ABA triplet report two streams more often
over time. The paper reproduced this offline, one sequence of 1 to 6
triplets at a time. Can the model update its belief as the sound arrives,
window by window, and give the same curve in one pass (design online,
decisions O1–O9)?

## Setup

App. C.11's stimuli: A at 500 Hz, B 4 or 8 semitones higher, 50 ms tones
at 55 dB with 10 ms ramps, 125 ms onset to onset, 1 to 6 triplets, after
50 ms of silence (as C.10's runs). No padding after the last triplet, so the
sequence of n triplets is the first 0.05 + 0.5 n s of the one with 6.
Two hypotheses, as in the paper: one whistle with every tone, or one with
the A tones and one with the B tones, each started from the tones'
frequencies and timings. One seed.

- **Batch** (the reference): Adam on each prefix from the tones, then
  the Laplace estimate, plus the structure prior. Run with 300 and with
  3000 steps, and the 6-triplet one-stream fits to 10,000.
- **Online**: one pass in 50 ms windows; each window adds its grid points
  and any tone that has started, takes 30 or 150 Adam steps on the free
  latents, and adds its term of the evidence. Lag infinite (nothing
  frozen) or 1 s.
- **Leaky** (design decision O10, accepted 2026-10-14): the online window
  terms summed with weight e^(−age/τ), from the logged terms, no new
  inference.

## Results

Log odds of two streams over one at the end of each triplet (`data/O1/`):

| Triplets | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| 4 st, batch 300 steps | 37.3 | 87.1 | 137.7 | 186.8 | 235.3 | 285.1 |
| 4 st, batch 3000 steps | −3.6 | −2.1 | 1.0 | 3.9 | 7.6 | 11.6 |
| 4 st, online, 150 steps, lag 1 s | −2.9 | −1.4 | 6.2 | 13.0 | 18.8 | 24.7 |
| 4 st, online, 30 steps, no lag | 148.4 | 106.8 | 119.8 | 112.0 | 118.0 | 64.0 |
| 4 st, online, 30 steps, lag 1 s | 148.4 | 107.3 | 119.5 | 121.7 | 133.6 | 142.8 |
| 8 st, batch 300 steps | 680.4 | 1470.0 | 2237.4 | 2988.0 | 3721.8 | 4464.2 |
| 8 st, batch 3000 steps | 6.1 | 33.1 | 46.3 | 67.4 | 80.4 | 92.5 |
| 8 st, online, 150 steps, lag 1 s | 135.7 | 95.9 | 157.8 | 204.0 | 255.7 | 295.0 |
| 8 st, online, 30 steps, no lag | 1107.5 | 1534.0 | 1569.8 | 1521.5 | 1578.0 | 1471.9 |
| 8 st, online, 30 steps, lag 1 s | 1107.5 | 1544.5 | 1729.5 | 1832.0 | 1950.9 | 2064.5 |

At 6 triplets with the one-stream fit run to 10,000 steps, the batch odds
are 12.0 (4 st) and 118.5 (8 st). The one-stream log joint is flat from
step 2,000 at 4 st and from step 5,000 at 8 st, where it gains 5.1 after
step 3,000 but the Laplace estimate moves by 26: the peak's curvature, not
the fit (`data/O1/one_stream_10000_steps_*.txt`).

Leaky log odds from the 150-step online runs (lag 1 s), structure prior
included (`data/O1/leaky_s150_lag1.txt`):

| Triplets | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| 4 st, τ 1 s | −6.4 | −5.6 | −0.5 | 2.4 | 3.7 | 4.1 |
| 4 st, τ 2.5 s | −4.4 | −3.5 | 2.6 | 7.1 | 10.0 | 12.3 |
| 4 st, τ 5 s | −3.6 | −2.5 | 4.2 | 9.6 | 13.7 | 17.4 |
| 8 st, τ 1 s | 58.3 | −24.3 | −7.9 | −10.5 | −11.2 | −19.0 |
| 8 st, τ 2.5 s | 102.7 | 36.2 | 67.7 | 79.0 | 91.9 | 92.3 |
| 8 st, τ 5 s | 118.9 | 63.7 | 107.2 | 131.7 | 158.2 | 171.0 |

Checks, from the same outputs:

- **The window terms add up.** With no lag, the online odds equal the
  batch Laplace odds at the online run's own values, to 0.01 at every
  triplet (`batch@online` column), as they must by construction.
- **Freezing costs little here.** With a 1 s lag the online odds differ
  from the batch Laplace odds at the same values by at most 2.3 (4 st)
  and 22.5 (8 st).
- **The two-stream fits agree everywhere**: batch at 300 and 3000 steps
  and both online runs are within 3.5 of each other at every triplet.
- **The one-stream fits do not.** At 6 triplets the one-stream log joint
  is −62223 after 300 steps and −61889 after 3000 (4 st), −66464 and
  −61991 (8 st). One whistle has to follow the A–B jumps against a prior
  with a 2.5 s lengthscale, and the fit climbs slowly.
- **Online one-stream fits, 150 steps per window**, at 6 triplets: 11
  below the 3000-step batch at 4 st, 200 below at 8 st (log joint plus structure);
  the two-stream fits are within 3 of the batch.
- **Time** on the cloud CPU (2 threads each, two runs at once), for the
  3.05 s sequence: online no lag 2214 s (one) and 4450 s (two), lag 1 s
  1052 and 2040 s; with 150 steps per window and lag 1 s 1787 and 5368 s;
  batch at 6 triplets with 3000 steps 652 and 797 s.
  Most of the online time is the Hessian of each window.

## What it says

- With fits run to (near) convergence, the batch odds rise with the number
  of triplets at both separations, faster at 8 st, and start near 0 at
  4 st: the shape of Fig. 8F. With 300 steps the rise was mostly the
  one-stream fit falling further behind on longer sequences.
- Online with 150 steps per window and a 1 s lag tracks the batch curve
  at 4 st: near 0 for 1–2 triplets, then rising, though about twice as
  steep (24.7 against 12.0 at 6 triplets), because one stream is still 11
  short of its batch fit. At 8 st one stream is far short (200 at 6
  triplets, and 136 against 6 already at 1 triplet), so the odds are too
  high. With 30 steps per window both are worse. More effort on one
  stream is needed, at least at the first A–B jumps.
- Online is not yet cheaper than batch: 590 to 1760 s of computing per
  second of sound with 150 steps and a 1 s lag, most of it in each
  window's Hessian.
- With a leaky decision (τ 2.5 s) the 4 st odds rise and begin to level
  off near 12, and the 8 st odds dip after the first triplet, then rise
  to about 92 and level off. These rest on the online fits above, so
  their 8 st values carry the same one-stream shortfall.
- The bookkeeping of the evidence over windows is exact, and a 1 s lag
  adds little error on this stimulus, although a toy model shows a lag
  shorter than the kernel's lengthscale can fail badly when the past is
  weakly observed (`tests/test_online.py`).
- Milestone (c)'s ABA map (C2) uses 300 steps; its two-stream odds may be
  inflated the same way.

## Reproduce

```
python tools/online_buildup.py --semitones 4 --triplets 6 --lag inf 1 \
    --batch-steps 300 > st4_w50_s30.txt
python tools/online_buildup.py --semitones 4 --triplets 6 --lag \
    --batch-steps 3000 > st4_batch3000.txt
python tools/online_buildup.py --semitones 4 --triplets 6 --lag 1 \
    --steps 150 > st4_w50_s150_lag1.txt
python tools/leaky_odds.py st4_w50_s150_lag1.txt st8_w50_s150_lag1.txt
```

The 10,000-step one-stream fits came from a session script (Adam on the
6-triplet one-stream scene, log joint printed at checkpoints, then
Laplace), not committed; `--batch-steps 10000` is the tool's equivalent,
not run here.

and the same with `--semitones 8`.

## Open

- More online effort where one stream needs it: surprise-scaled steps
  (decision O9) or more steps per window; whether a stopping rule
  (`--patience`) reaches the converged batch fits at lower cost.
- Whether the Laplace estimate's sensitivity to curvature (26 at 8 st
  for a gain of 5) needs importance sampling or variational checks.
- One seed; whether 3000 steps is converged for every prefix (the
  6-triplet one-stream fits gained less than 4 between steps 2000 and 3000).
- Importance sampling and variational estimates are not run here; Laplace only.
