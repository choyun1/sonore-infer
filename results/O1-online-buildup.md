# O1. Buildup of streaming, batch and online

> **Under review.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** first pass done; online fits per window to convergence added (2026-10-14); sweeps of lag and window added (2026-10-15). **Paper target:** Fig. 8F (right), App. C.11.
**Code:** `src/sonore_infer/online.py`, `tools/online/online_buildup.py`,
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
  latents (or, "to convergence", until the fit gains less than 1 over 50
  steps, at most 2000: design decision O9), and adds its term of the
  evidence. Lag infinite (nothing frozen) or 1 s; in the sweep below,
  0.5, 1 and 2 s, and windows of 10, 50 and 100 ms. A tone's event enters
  the scene once 20 ms of the tone has arrived (see the sweep).
- **Leaky** (design decision O10, accepted 2026-10-14): the online window
  terms summed with weight e^(−age/τ), from the logged terms, no new
  inference.

## Results

Log odds of two streams over one at the end of each triplet (`data/O1/`):

| Triplets | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| 4 st, batch 300 steps | 37.3 | 87.1 | 137.7 | 186.8 | 235.3 | 285.1 |
| 4 st, batch 3000 steps | −3.6 | −2.1 | 1.0 | 3.9 | 7.6 | 11.6 |
| 4 st, online, to convergence, lag 1 s | −3.3 | −1.6 | 2.6 | 7.1 | 11.4 | 16.3 |
| 4 st, online, 150 steps, lag 1 s | −2.9 | −1.4 | 6.2 | 13.0 | 18.8 | 24.7 |
| 4 st, online, 30 steps, no lag | 148.4 | 106.8 | 119.8 | 112.0 | 118.0 | 64.0 |
| 4 st, online, 30 steps, lag 1 s | 148.4 | 107.3 | 119.5 | 121.7 | 133.6 | 142.8 |
| 8 st, batch 300 steps | 680.4 | 1470.0 | 2237.4 | 2988.0 | 3721.8 | 4464.2 |
| 8 st, batch 3000 steps | 6.1 | 33.1 | 46.3 | 67.4 | 80.4 | 92.5 |
| 8 st, online, to convergence, lag 1 s | 10.0 | 37.1 | 60.5 | 84.9 | 108.1 | 131.9 |
| 8 st, online, 150 steps, lag 1 s | 135.7 | 95.9 | 157.8 | 204.0 | 255.7 | 295.0 |
| 8 st, online, 30 steps, no lag | 1107.5 | 1534.0 | 1569.8 | 1521.5 | 1578.0 | 1471.9 |
| 8 st, online, 30 steps, lag 1 s | 1107.5 | 1544.5 | 1729.5 | 1832.0 | 1950.9 | 2064.5 |

![O1-online-buildup](plots/O1-online-buildup.svg)

*Log odds of two streams over one after each triplet: batch with 3000
Adam steps (the diamond: the one-stream fit run to 10,000 steps), online
with each window fitted to convergence and a 1 s lag, and the leaky
decision (τ 2.5 s) from the same online run. 50 ms windows.*

At 6 triplets with the one-stream fit run to 10,000 steps, the batch odds
are 11.9 (4 st) and 118.5 (8 st). The one-stream log joint is flat from
step 2,000 at 4 st and from step 5,000 at 8 st, where it gains 5.1 after
step 3,000 but the Laplace estimate moves by 26: the peak's curvature, not
the fit (`data/O1/one_stream_10000_steps_*.txt`).

Leaky log odds (decision O10) from the online runs to convergence (lag
1 s), structure prior included (`data/O1/leaky_p50_lag1.txt`):

| Triplets | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| 4 st, τ 1 s | −3.3 | −1.3 | 1.8 | 4.0 | 5.2 | 6.6 |
| 4 st, τ 2.5 s | −3.3 | −1.5 | 2.3 | 5.7 | 8.2 | 11.1 |
| 4 st, τ 5 s | −3.3 | −1.5 | 2.4 | 6.4 | 9.7 | 13.4 |
| 8 st, τ 1 s | 5.5 | 23.0 | 31.0 | 36.9 | 39.7 | 41.7 |
| 8 st, τ 2.5 s | 8.0 | 30.4 | 45.6 | 59.0 | 68.9 | 77.6 |
| 8 st, τ 5 s | 9.0 | 33.5 | 52.4 | 70.4 | 85.5 | 99.8 |

The same from the 150-step online runs (`data/O1/leaky_s150_lag1.txt`),
whose one-stream fits fall short:

| Triplets | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| 4 st, τ 1 s | −6.4 | −5.6 | −0.5 | 2.4 | 3.7 | 4.1 |
| 4 st, τ 2.5 s | −4.4 | −3.5 | 2.6 | 7.1 | 10.0 | 12.3 |
| 4 st, τ 5 s | −3.6 | −2.5 | 4.2 | 9.6 | 13.7 | 17.4 |
| 8 st, τ 1 s | 58.3 | −24.3 | −7.9 | −10.5 | −11.2 | −19.0 |
| 8 st, τ 2.5 s | 102.7 | 36.2 | 67.7 | 79.0 | 91.9 | 92.3 |
| 8 st, τ 5 s | 118.9 | 63.7 | 107.2 | 131.7 | 158.2 | 171.0 |

### Lag and window

Online, each window fitted to convergence (decision O9), log odds at the
end of each triplet, and the leaky odds (τ 2.5 s) at 6 triplets
(`data/O1/st*_p50_*.txt`):

| Run | 1 | 2 | 3 | 4 | 5 | 6 | leaky at 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 st, lag 0.5 s | −3.2 | −0.4 | 9.3 | 17.9 | 27.3 | 35.8 | 23.5 |
| 4 st, lag 1 s | −3.3 | −1.6 | 2.6 | 7.1 | 11.4 | 16.3 | 11.1 |
| 4 st, lag 2 s | −3.3 | −1.5 | 1.4 | 5.0 | 9.2 | 5.5 | 1.1 |
| 8 st, lag 0.5 s | 17.6 | 45.3 | 87.5 | 122.5 | 158.7 | 193.8 | 116.7 |
| 8 st, lag 1 s | 10.0 | 37.1 | 60.5 | 84.9 | 108.1 | 131.9 | 77.6 |
| 8 st, lag 2 s | 10.0 | 37.1 | 57.9 | 71.8 | 101.9 | 125.7 | 73.7 |
| 8 st, lag 1 s, 10 ms windows | 29.2 | 40.3 | 64.3 | 88.2 | 111.5 | 135.9 | 76.1 |
| 8 st, lag 1 s, 100 ms windows | 17.0 | 36.6 | 59.1 | 82.4 | 105.5 | 129.2 | 78.3 |

Windows are 50 ms unless named.

![O1-lag-window](plots/O1-lag-window.svg)

*Online log odds after each triplet by lag (50 ms windows) and, at 8 st
with a 1 s lag, by window length.*

- **A shorter lag gives a steeper buildup.** With a 0.5 s lag the odds at
  6 triplets are twice those with 1 s (35.8 against 16.3 at 4 st, 193.8
  against 131.9 at 8 st). The one-stream past is frozen sooner, so each
  new A–B jump is paid for again instead of being absorbed by revising
  the whistle. A 2 s lag is close to 1 s, except where noted below.
- **Window length barely matters after the first triplet.** At 8 st the
  10, 50 and 100 ms runs are within 7 of each other from triplet 2 on,
  and the leaky odds at 6 triplets within 3 (76.1, 77.6, 78.3). They
  differ at triplet 1 (29.2, 10.0, 17.0), where the fits of the first
  tones depend on how much of each tone the first window held.
- **The two drops with a 2 s lag are the Hessian floor.** At 4 st the
  odds fall from 9.2 to 5.5 at 6 triplets: in the last one-stream window
  a direction of the posterior had no curvature and was floored, and
  that window's term came out 8.8 above the 1 s run's. A floored term is
  undone by the next window's, as at 2.60 s (+8.5, then −8.2 at 2.65 s),
  but after the last window there is none. At 8 st the dip at 4 triplets
  (71.8 against 84.9 with 1 s) comes mostly from three floored one-stream
  windows in a row (1.95 to 2.05 s), whose terms add 9.8 to one stream
  over the 1 s run from 1.6 to 2.05 s.
- **Events must not enter on their ramp alone.** First run with 10 ms
  windows, a tone's event entered the scene as soon as its onset had
  arrived, with only its 10 ms ramp heard. The fit then pushed the
  event's timing off the tone (the A whistle's rest became 0.45 s instead
  of 0.2 s) and never pulled it back, and the log odds fell to −413 at 6
  triplets (`data/O1/st8_w10_p50_lag1_ramp_entry.txt`). Lowering the
  stopping tolerance from 1 to 0.2 did not help. Events now enter once
  20 ms of their tone has arrived (`--heard`). The 50 and 100 ms runs are
  unaffected: on these stimuli every event entered with at least 25 ms of
  its tone heard.
- **Time per window** (cloud CPU, s, one stream / two streams): lag
  0.5 s 27 / 26 (4 st), 49 / 26 (8 st); lag 1 s 35 / 42, 50 / 43; lag 2 s
  52 / 74, 69 / 75; at 8 st, lag 1 s, 10 ms windows 28 / 43 (300 windows),
  100 ms 58 / 41 (36 windows, triplet ends included). Per second of sound
  and hypothesis, 10 ms windows cost about 3 to 5 times 50 ms windows, and
  100 ms windows about 30% less: most of the time is each window's
  Hessian, whatever the window's length. The one-stream fit took up to
  1556 Adam steps in a window (8 st, lag 0.5 s), and with 10 ms windows
  reached the cap of 2000 once; two streams took at most 79 with 50 ms
  windows and 210 with 10 ms.

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
- **Online one-stream fits to convergence**, at 6 triplets: 2 below the
  10,000-step batch at 4 st, 11 below at 8 st; at 1 triplet 0.4 below and
  4 below the 3000-step batch. Windows took 61 Adam steps on average for
  two streams (at most 68) and 179 (4 st) and 378 (8 st) for one stream
  (at most 466 and 1043).
- **Online one-stream fits, 150 steps per window**, at 6 triplets: 11
  below the 3000-step batch at 4 st, 200 below at 8 st (log joint plus structure);
  the two-stream fits are within 3 of the batch.
- **Time** on the cloud CPU (2 threads each, two runs at once), for the
  3.05 s sequence: online no lag 2214 s (one) and 4450 s (two), lag 1 s
  1052 and 2040 s; with 150 steps per window and lag 1 s 1787 and 5368 s;
  to convergence with lag 1 s 2074 (4 st) and 3026 s (8 st) for one,
  2537 and 2572 s for two;
  batch at 6 triplets with 3000 steps 652 and 797 s.
  Most of the online time is the Hessian of each window.

## What it says

- With fits run to (near) convergence, the batch odds rise with the number
  of triplets at both separations, faster at 8 st, and start near 0 at
  4 st: the shape of Fig. 8F. With 300 steps the rise was mostly the
  one-stream fit falling further behind on longer sequences.
- Online in one pass, with each window fitted until it stops improving
  and a 1 s lag, reproduces the batch curve: at 4 st −3.3 to 16.3 against
  −3.6 to 12.0, at 8 st 10.0 to 131.9 against 6.1 to 118.5, rising at
  every triplet, faster at 8 st. The differences are within the
  Laplace estimate's own sensitivity (26 at 8 st, 6 triplets) and mostly
  the one-stream fit being 2 to 11 short. With a fixed 30 or 150 steps
  per window, one stream fell far behind at the A–B jumps and the online
  odds were too high.
- The stopping rule puts the effort where the sound is surprising: one
  stream takes up to 1043 steps in a window and two streams never more
  than 68.
- Online is not yet cheaper than batch: 680 to 990 s of computing per
  second of sound per hypothesis with a 1 s lag, most of it in each
  window's Hessian.
- With the leaky decision (τ 2.5 s), the odds still rise over 6 triplets
  but more slowly each triplet, at 4 st from −3.3 to 11.1 and at 8 st
  from 8.0 to 77.6; with τ 1 s, 8 st levels off near 40.
- The lag, not the window, sets how fast the online odds build up: a
  0.5 s lag doubles the slope of a 1 s lag, while 10, 50 and 100 ms
  windows give the same curve after the first triplet. Online is
  cheapest with long windows, because each window's Hessian costs about
  the same whatever its length.
- The bookkeeping of the evidence over windows is exact, and a 1 s lag
  adds little error on this stimulus, although a toy model shows a lag
  shorter than the kernel's lengthscale can fail badly when the past is
  weakly observed (`tests/test_online.py`).
- Milestone (c)'s ABA map (C2) uses 300 steps; its two-stream odds may be
  inflated the same way.

## Reproduce

```
python tools/online/online_buildup.py --semitones 4 --triplets 6 --lag inf 1 \
    --batch-steps 300 > st4_w50_s30.txt
python tools/online/online_buildup.py --semitones 4 --triplets 6 --lag \
    --batch-steps 3000 > st4_batch3000.txt
python tools/online/online_buildup.py --semitones 4 --triplets 6 --lag 1 \
    --steps 150 > st4_w50_s150_lag1.txt
python tools/online/online_buildup.py --semitones 4 --triplets 6 --lag 1 \
    --steps 2000 --window-patience 50 > st4_w50_p50_lag1.txt
python tools/online/leaky_odds.py st4_w50_p50_lag1.txt st8_w50_p50_lag1.txt
python tools/online/online_buildup.py --semitones 4 --triplets 6 --lag 0.5 \
    --steps 2000 --window-patience 50 > st4_w50_p50_lag0.5.txt
python tools/online/online_buildup.py --semitones 8 --triplets 6 --lag 1 \
    --window 0.01 --steps 2000 --window-patience 50 > st8_w10_p50_lag1.txt
python tools/results_plots.py
```

and the same with `--semitones 8`, and likewise for the other lags and
windows (lag 2; `--window 0.1`). The run with events entering on their
ramp is `--heard 0`.

The 10,000-step one-stream fits came from a session script (Adam on the
6-triplet one-stream scene, log joint printed at checkpoints, then
Laplace), not committed; `--batch-steps 10000` is the tool's equivalent,
not run here.

## Open

- What makes each window's Hessian cheaper; this decides whether online
  can be cheaper than batch.
- Floored Hessian directions in single windows (the 2 s lag drops): a
  floor that is undone a window later suggests a better treatment of
  near-flat directions.
- With 10 ms windows and a stopping tolerance of 0.2, the one-stream run
  of the first triplet went wrong (window terms near −4000, four floored
  directions); not looked into.
- Whether 20 ms is the right time for an event to enter, and whether a
  real detector (milestone (c), decision C9) gives the same.
- Whether the Laplace estimate's sensitivity to curvature (26 at 8 st
  for a gain of 5) needs importance sampling or variational checks.
- One seed; whether 3000 steps is converged for every prefix (the
  6-triplet one-stream fits gained less than 4 between steps 2000 and 3000).
- Importance sampling and variational estimates are not run here; Laplace only.
