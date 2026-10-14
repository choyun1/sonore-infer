# O1. Buildup of streaming, batch and online

> **Under review by Adrian Cho.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** first pass (2026-10-14); online runs with more Adam steps per
window in progress. **Paper target:** Fig. 8F (right), App. C.11.
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
  3000 steps.
- **Online**: one pass in 50 ms windows; each window adds its grid points
  and any tone that has started, takes 30 Adam steps on the free latents,
  and adds its term of the evidence. Lag infinite (nothing frozen) or 1 s.

## Results

Log odds of two streams over one at the end of each triplet (`data/O1/`):

| Triplets | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| 4 st, batch 300 steps | 37.3 | 87.1 | 137.7 | 186.8 | 235.3 | 285.1 |
| 4 st, batch 3000 steps | −3.6 | −2.1 | 1.0 | 3.9 | 7.6 | 11.6 |
| 4 st, online, no lag | 148.4 | 106.8 | 119.8 | 112.0 | 118.0 | 64.0 |
| 4 st, online, lag 1 s | 148.4 | 107.3 | 119.5 | 121.7 | 133.6 | 142.8 |
| 8 st, batch 300 steps | 680.4 | 1470.0 | 2237.4 | 2988.0 | 3721.8 | 4464.2 |
| 8 st, batch 3000 steps | 6.1 | 33.1 | 46.3 | 67.4 | 80.4 | 92.5 |
| 8 st, online, no lag | 1107.5 | 1534.0 | 1569.8 | 1521.5 | 1578.0 | 1471.9 |
| 8 st, online, lag 1 s | 1107.5 | 1544.5 | 1729.5 | 1832.0 | 1950.9 | 2064.5 |

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
- **Time** on the cloud CPU (2 threads each, two runs at once), for the
  3.05 s sequence: online no lag 2214 s (one) and 4450 s (two), lag 1 s
  1052 and 2040 s; batch at 6 triplets with 3000 steps 652 and 797 s.
  Most of the online time is the Hessian of each window.

## What it says

- With fits run to (near) convergence, the batch odds rise with the number
  of triplets at both separations, faster at 8 st, and start near 0 at
  4 st: the shape of Fig. 8F. With 300 steps the rise was mostly the
  one-stream fit falling further behind on longer sequences.
- The online runs inherit the same problem: 30 Adam steps per window leave
  one stream far from its best fit, so their odds are too high. They are
  not yet a test of online inference; the next run gives each window more
  steps (design decision O6).
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
```

and the same with `--semitones 8`.

## Open

- Online with more steps per window; whether a stopping rule
  (`--patience`) reaches the 3000-step fits at lower cost.
- One seed; whether 3000 steps is converged for every prefix (the
  6-triplet one-stream fits gained less than 4 between steps 2000 and 3000).
- Importance sampling and variational estimates are not run here; Laplace only.
