# O2. Faster online inference

> **Under review.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** S1 and S2 checked (2026-10-15); S3 failed its check.
**Paper target:** none; this is about cost, checked against
[O1](O1-online-buildup.md). **Code:** `src/sonore_infer/sources.py`
(`trajectory_event`), `src/sonore_infer/online.py` (`run_online`),
`tools/online/online_buildup.py`, `tools/online/online_cost.py`. Design:
[faster online inference](../docs/design/online-speed.md), decisions S1 to S6.

## Question

O1's online runs took about 1,000 s of computing per second of sound per
hypothesis. Do the speed-ups of the design make them faster without
changing the answer? Decision S6's rule: the online log odds stay within 3
of O1's at every triplet, at 8 and then 4 semitones.

## Setup

O1's reference case: App. C.11's ABA sequence of 6 triplets, 8 or 4
semitones, lag 1 s, 50 ms windows, each window fitted until it gains less
than 1 over 50 Adam steps (at most 2000), events entering after 20 ms of
their tone. One seed, the cloud CPU (4 threads), as O1.

- **S1**: each event rendered over its own span instead of the whole sound
  so far. The waveform and gradients equal the old ones to rounding, which
  a test checks (`tests/test_sources_and_fit.py`); it is on by default.
- **S2**: the fit still every 50 ms, the Laplace evidence only every 0.5 s
  and at each triplet's end (`--evidence-every 0.5`); opt-in.
- **S3**: each window fitted by L-BFGS instead of Adam, stopping once it
  gains less than 1 over 5 iterations, at most 200 (`--optimizer lbfgs`).

## Results

### S1 leaves every answer as it was

One online window of the 8-semitone case, before and after S1, median of 5,
in ms (Hessian in s) [`data/O2/online_cost_st8_before_s1.txt`,
`data/O2/online_cost_st8.txt`]:

| Sound so far | Hypothesis | Render | Adam step | Hessian |
|---|---|---|---|---|
| 1.05 s | one | 8.7 → 2.1 | 39 → 14 | 8.6 → 3.7 |
| 3.05 s | one | 52.9 → 6.8 | 184 → 36 | 49 → 12 |
| 3.05 s | two | 54.0 → 9.5 | 197 → 48 | 98 → 26 |

The O1 case rerun with S1 gives the same window terms as O1 to 0.01, and
log odds within 0.09 at every triplet (131.85 against 131.94 at 6
triplets) [`data/O2/st8_w50_p50_lag1.txt`]. Its windows took 2,592 s in all
against O1's 5,598 s; a test run shared the machine for part of it, so the
gain is if anything larger.

One cheap case of each earlier milestone, old and new code side by side on
the same machine [`data/O2/rerun/`]: the Laplace estimates are identical in
all three; the importance-sampling estimates differ by up to 4 log units,
as their sampling does from run to run.

| Case | Laplace log odds, old → new | Importance, old → new |
|---|---|---|
| (a) f0 200 Hz, harmonic 2, 8 % | −34.1 → −34.2 | −28.3 → −29.4 |
| (b) octave, 120 ms | −3.3 → −3.3 | −27.5 → −27.5 |
| (c) ABA, 6 semitones, 100 ms | 181.9 → 181.9 | unusable in both (one stream: ESS 1) |

The first run of (c) with S1 crashed: an importance sample drew an
infinite event duration, and the new span's end overflowed. Fixed in PR
#64 before any result was made with it.

### S2 passes S6 and takes a seventh of the time

![O2 speed agreement](plots/O2-speed-agreement.svg)

| Triplets | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| 8 st, O1 | 9.97 | 37.06 | 60.53 | 84.92 | 108.11 | 131.94 |
| 8 st, S1 and S2 | 9.97 | 37.03 | 60.91 | 85.63 | 108.97 | 132.98 |
| 4 st, O1 | −3.27 | −1.58 | 2.56 | 7.09 | 11.36 | 16.31 |
| 4 st, S1 and S2 | −3.27 | −1.60 | 2.91 | 7.84 | 12.45 | 17.52 |

[`data/O2/st8_w50_p50_lag1_e0.5.txt`, `data/O2/st4_w50_p50_lag1_e0.5.txt`]

The difference is at most 1.04 at 8 semitones and 1.21 at 4, inside S6's
3. It grows by about 0.2 to 0.4 per triplet, so longer sequences need the
check again. The windows took 835 s in all at 8 semitones (O1: 5,598) and
559 s at 4 (O1: 4,611), six to eight times less; the whole run took
1,108 and 833 s.

### S3 fails S6

With L-BFGS the online log odds at 6 triplets were −1.4 million against
O1's 132 [`data/O2/st8_w50_lbfgs_p5_lag1.txt`]. Tracing the first windows
of the one-stream run shows why. Each window's L-BFGS fit moves far along
directions that the sound heard so far hardly constrains: in the first
window, the event's log duration rose by 2.0 (its duration seven times) for
a gain of 1.5 in the log joint; later windows moved level deviations by 10
to 16 dB. When the next 50 ms arrive, the fit starts far from the tones
(log joint −16,969 at 0.2 s against −2,680 at the end of the window
before) and does not find its way back. Adam's small steps avoid this.

On one window from a fixed start, L-BFGS does fit better and in fewer
evaluations: on the test's toy window (`tests/test_online.py`), log joint
−5,824 after 10 iterations (31 evaluations) against Adam's −5,837 after 40
steps. So the failure comes from fitting a sound that keeps arriving, not
from the optimizer itself.

## What is open

- Whether to keep L-BFGS as an option (asked of Cho).
- S2 as the default, now that it passed at 8 and 4 semitones.
- S4 (a Gaussian summary of the past) and S5 (the PC's GPU), per the design.
