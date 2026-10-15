# What real time would take: a note after the online buildup runs

Status: **note for Cho, 2026-10-15** (step 4 of [online inference](online.md),
the input to the next design). No decisions are asked here; §5 lists the
ones the next design would ask.

Every number below is either measured, with the file it comes from, or
labeled an estimate with its basis. Measurements are on the cloud CPU
(Intel Xeon, 2.1 GHz, 4 cores).

## 1. The target

Real time means each window's update finishes within the window: with
50 ms windows, every hypothesis held updated in under 50 ms per window,
whatever the length of the sound so far. For the phone app (online.md §9)
it also means on long input, minutes rather than 3 s.

## 2. Where we are

On the buildup stimulus (8 st, 6 triplets, 3.05 s), online with each
window fitted to convergence and a 1 s lag took 3026 s (one stream) and
2572 s (two streams) for the whole sound
[measured: `results/data/O1/st8_w50_p50_lag1.txt`, `seconds` column]:
about 1,000 and 850 s of computing per second of sound, per hypothesis.
That is three orders of magnitude from real time, and it gets worse as the
sound goes on: the last windows took 69 to 179 s (one stream) and 86 to
121 s (two streams), each for 50 ms of sound, so 1,400 to 3,600 times
slower than real time at 3 s.

## 3. Where the time goes

`tools/online/online_cost.py` times the parts of one window at four points in the
sound [measured: `results/data/O1/online_cost_st8.txt`, 8 st, lag 1 s,
4 threads, median of 5]:

| Sound so far (s) | 0.55 | 1.05 | 2.05 | 3.05 |
| --- | --- | --- | --- | --- |
| Latents free, one / two streams | 120 / 234 | 216 / 420 | 216 / 420 | 216 / 420 |
| Render, ms (one stream) | 3.3 | 8.6 | 27.9 | 53.8 |
| Cochleagram, ms | 0.5 | 0.7 | 0.8 | 1.0 |
| Likelihood, ms | 0.04 | 0.03 | 0.06 | 0.05 |
| Prior, ms (one / two) | 1.5 / 2.8 | 1.7 / 3.4 | 3.3 / 7.3 | 4.2 / 11.5 |
| Log joint, ms (one / two) | 5.1 / 7.7 | 13.2 / 15.2 | 35.3 / 39.6 | 64.6 / 71.5 |
| One Adam step, ms (one / two) | 13 / 19 | 43 / 44 | 99 / 118 | 205 / 238 |
| Laplace Hessian, s (one / two) | 1.7 / 4.1 | 11.1 / 22.3 | 25.7 / 51.8 | 54.2 / 107.7 |

These account for the measured windows: at 3.05 s, two streams took 62
Adam steps, 62 × 0.24 s + 108 s ≈ 123 s against 121 s measured; one
stream 105 steps, 105 × 0.20 + 54 ≈ 76 s against 69 s [the runs used 2
threads each, two at once].

Three things set the cost:

- **The Hessian is most of it.** It is built one row at a time, one
  backward pass per free latent, so it costs about as many Adam steps as
  there are free latents (216 or 420 with a 1 s lag; 0.25 s per row at
  3.05 s for both hypotheses). torch's vectorized Hessian was twice as slow
  here, not faster [measured once at 1.05 s: 20.2 s against 10.4 s for one
  stream, 38.4 s against 17.7 s for two; the two agree to 1e-8].
- **Every evaluation covers the whole sound so far, not the window.** The
  free latents are only the last second, but the means of each trajectory
  (and the kernel's latents, when inferred) stay free over the whole sound
  (online.md O3), so the whole sound is rendered, analyzed and compared
  at every step. Rendering grows faster than the sound: 16 times the cost
  for 5.5 times the length, because each event is synthesized over the
  whole sound so far [code: `sources.py` `trajectory_event`, `n_samples`
  from `total_duration`], and an event is added with every tone.
- **The steps per window vary with surprise.** One stream took up to 1043
  steps in a window, two streams 50 to 79, where 50 is the least the
  stopping rule allows (patience 50) [measured: O1 page].

The squared-exponential prior is not yet the bottleneck at 3 s (4 to
12 ms per evaluation) but grows as the cube of the grid: on a minute of
sound, 20 times the grid, it would be about 8,000 times more, tens of
seconds per evaluation [estimate: Cholesky cost n³, from the 3.05 s value;
not measured].

## 4. What would close the gap

In order of the gain each would bring, with what each changes.

1. **Compute the evidence less often than the fit.** The decision needs the
   evidence at the scale of the leaky integrator (τ 1 to 5 s), not every
   50 ms. Fitting every 50 ms and adding an evidence term every 500 ms
   would cut the Hessians tenfold. The window sweep says the curve barely
   depends on how often the terms are taken: 10, 50 and 100 ms windows
   agree within 7 from the second triplet on, and the leaky odds at 6
   triplets within 3 [measured: O1 page, "Lag and window"]. Changes no
   model; a gain of about 10 on the Hessian part [estimate: from the
   count of Hessians; not run at 500 ms].
2. **Make each evaluation local.** Render, analyze and compare only the
   last lag plus window, and summarize the frozen past's dependence on the
   whole-sound latents (the trajectory means) by a Gaussian, the one the
   previous window's Laplace state already gives (assumed-density
   filtering). Each step then costs the same however long the sound has
   run. At 3 s that is about 1.5 s of sound instead of 3.05, with each
   event rendered over its own span only; on a minute it is the difference
   between a bounded and an unbounded cost [estimate: from the render
   times above]. Changes the approximation, not the model; needs its own
   check against the full sum, as O5 did for the window terms.
3. **Fewer, better steps.** The Hessian is in hand at every evidence
   window, so a few Newton steps (or L-BFGS between Hessians) should
   replace hundreds of Adam steps where one stream is surprised. A
   relative stopping rule would also let two streams stop before 50 steps
   [estimate; not run].
4. **A cheaper Hessian.** The likelihood is Gaussian, so the Gauss–Newton
   form (the Jacobian of the cochleagram times its transpose, plus the
   prior's Hessian) needs only first derivatives; and only the rows of
   latents that changed need recomputing. Gains uncertain until measured.
5. **A kernel with a state-space form** (online.md O4 (C)): needed for
   minutes of sound, where the cubic prior cost would dominate; not needed
   at 3 s. This changes the model's priors away from the paper's Table A.2.
6. **Hardware and arithmetic.** float32 instead of float64, a GPU, and the
   two hypotheses (or many particles) evaluated as one batch. Commonly a
   factor of 10 or more [estimate; not measured here].

Put together, a rough budget for one hypothesis at 3 s: five Newton steps
on 1.5 s of sound at about 0.1 s each, plus a Hessian every tenth window,
is about 1 to 3 s of computing per 50 ms window, still 20 to 60 times
slower than real time on this CPU before hardware [estimate: from the
measured per-step times scaled by the evaluated length; every factor in it
is unmeasured]. The survey's estimate of a 10³ to 10⁴ gap
(`/mnt/project-files/realtime/realtime-inference-survey.md`) matches what
was measured (1,400 to 3,600 at 3 s).

## 5. What the next design would decide

- Whether the evidence is taken on a coarser grid than the fit (§4.1), and
  how often.
- Whether the past is summarized by a Gaussian in the whole-sound latents
  (§4.2), and how that is checked against the full sum.
- Newton or L-BFGS steps instead of Adam, and the stopping rule (§4.3).
- When to move to a state-space kernel (§4.5), which changes the priors.
- Structure search online (online.md O1 (C)), which multiplies all of the
  above by the number of hypotheses held.

The first three keep the model as it is and can be checked against the O1
results; I would do them first, in that order, and measure after each.

## Reproduce

```
python tools/online/online_cost.py --semitones 8 --ends 0.55 1.05 2.05 3.05 > online_cost_st8.txt
```
