# Relative pitch: Group III's rise at 9 steps, and other seeds — design

Status: **proposed, waiting for Cho** (P24 to P26). Written 2026-11-01.
Follows [P9](../../results/P9-shepard-listener-noise.md), whose fit with
both hearing options (all 90 pairs, three readings) still misses Group
III's rise from 0.26 at 8 steps to 0.40 at 9, and which, like every
Shepard page so far, is one seed. Decisions continue the numbering of
[relative-pitch-dominance.md](relative-pitch-dominance.md) (P19 to P23).
AI-assisted (Claude); code as of `752f48b`.

**How numbers and sources are marked** (as in relative-pitch.md).

- **[Shepard §x, Fig. 3]**: Shepard (1964), *Circularity in judgments of
  relative pitch*, JASA 36, 2346–2353.
- **[measured]**: from a script, with its path.
- **[estimate]**: my estimate, with its basis next to it.

## Decisions at a glance

| Decision | Proposed |
|---|---|
| P24. What could produce the rise | (A) Noise on where the second tone is heard round the pitch circle, before the model's up-or-down answer |
| P25. How it is fitted and checked | (A) Blur the model's own curve by that noise, fit noise and lapse per group, with P16's guards and a pure circle as a second baseline |
| P26. Other seeds | (A) Seeds 0, 1 and 2 for P9's 459 fits on Cho's PC, the mode fitted once and the sampling once per seed |

---

## 1. Why

Shepard's Group III (his 14 least accurate listeners) judged the second
tone higher at 1 to 9 steps clockwise in fractions .74 .78 .83 .75 .52
.29 .24 .26 .40 [Shepard Fig. 3, read by eye]. Two things in that curve
are not graded guessing:

- **It is best at 3 steps, not at 1** (.83 against .74), and on the
  other side best at 7 and 8 (.24, .26), not at 9 (.40). Nine steps
  clockwise is one step down, so the rise at 9 is the same thing as the
  dip at 1: the smallest steps are the hardest.
- **It leans up**: the mean over the nine steps is .53, against .51 for
  Group I.

P9's response model, λ/2 + (1 − λ) · logistic(β · L), cannot produce the
first, since it is monotone in the log odds L and L falls steadily from
about +20 at 1 step to about −24 at 9 [measured,
`results/data/P9/options/fit_all.txt`]. Its Group III fit is a flat .75
at 1 to 3 steps and a flat .25 at 7 to 9, with a lapse of 0.50. The
octave readings did not produce the rise either: every pair 9 steps apart
is down by 17.7 or more.

## 2. A quick check of the shape

A curve that is best at 2 to 3 steps and worse at 1 is what noise on the
perceived interval gives when the interval lives on a circle: a one-step
interval heard with noise often lands on the other side of 0 and so reads
as down, while a three-step interval rarely does. Noise on L does not wrap
round; noise on where the tone sits on the circle does.

Two fits by maximum binomial likelihood to Fig. 3 (trial counts as in P9),
grid search, a few seconds each [measured, scripts in this thread's
scratch, to go into `tools/relative_pitch/` with P25]:

| | Group III log likelihood | Group III fitted, 1 to 9 steps | Group I log likelihood |
|---|---|---|---|
| P9 model (A), β and λ | −745.1 | .75 .75 .75 .65 .44 .30 .25 .25 .25 | −716.5 |
| P9 baseline in steps | −740.1 | .74 .74 .74 .73 .52 .27 .26 .26 .26 | −704.6 |
| Pure circle, wrapped Gaussian noise σ and λ | −736.0 (σ 0.98 steps, λ 0.42) | .70 .78 .78 .70 .50 .30 .22 .22 .30 | −803.6 |
| Model's Group I curve blurred by wrapped noise, σ and λ | −745.4 (σ 1.6 steps, λ 0.07) | .68 .77 .74 .62 .46 .31 .21 .20 .31 | (σ → 0) −715.1 |

- **The pure circle gets Group III's shape**, rise at 9 included (.30,
  against .40), and is the best of the four for Group III. It is far the
  worst for Group I (−803.6): Group I is near 1.00 at 1 step, which
  symmetric noise on the circle cannot keep while also grading 3 to 6
  steps. So the two groups are not the same curve with different noise:
  Group I's grading comes from the direction being ambiguous near the
  half octave (what the model's L does), Group III's from noise on the
  position as well.
- **The blurred model gets the rise but not the middle**: .31 at 9, but
  .62 at 4 and .46 at 5, because the model's own crossover is early (P9:
  .42 at 5 for Group I against .53). So with this mechanism Group III's
  fit is held back by the same early crossover as Group I's, not by
  anything new.
- **Neither has the up lean.** A guess rate other than ½ would carry it
  (one parameter).
- These used the model's curve as the mean of P(up) per step with Group
  I's β, interpolated between steps, which is coarser than P25 (A) below.

## 3. Decisions

### P24. What could produce the rise

- **(A) Noise on where the second tone is heard on the pitch circle.**
  The listener hears the second tone k′ steps from the first, with k′ the
  true k plus Gaussian noise of SD σ (in steps), wrapped round the octave,
  and then answers as the model does for k′. This is Shepard's own
  account, a larger difference limen for less accurate listeners
  [Shepard §V], put in the model's terms: the noise is on the percept of
  pitch class, not on the decision. It needs no refitting (P25).
- **(B) The same noise put into inference.** Add noise to the
  cochleagram per simulated trial and rerun inference, as P15 (C). Closest
  to a listener, but the noise then acts on the whole spectrum, not on
  the position on the circle, and the cost is the 510 fits times the
  number of trials [estimate: tens of trials, so weeks of PC time].
- **(C) Hearing the second tone an octave off.** P9's octave readings
  were meant to allow this and gave no rise; ruled out on one seed.
- **(D) Leave Group III as guessing**, and record the rise as missed.

**Recommendation: A.** It is the one mechanism of the four that §2 shows
can make the shape, it is what Shepard proposed, and it is cheap. (B)
only if (A) works and the question becomes where in the model the noise
sits.

### P25. How it is fitted and checked

- **(A) The model's curve per pair, blurred.** For each pair, the model
  gives L at the true k. For k′ between steps, interpolate L between the
  pairs from the same first tone (the second tone moved round the circle),
  with k′ near 0 counted as up and near 10 as down (the shorter way
  round). Then P(judged up) = λ·g + (1 − λ) · E over k′ of
  logistic(β · L(k′)), with β fixed at Group I's value, and σ, λ and g (the
  guess rate) fitted per group. Group I is refitted the same way, with σ
  free, so the page says whether Group I needs any position noise.
  Guards as P16: the baseline in steps, leave one step out, adjacent pairs;
  plus the pure circle of §2 as a second baseline, since it ignores the
  model too.
- **(B) The model's step means, blurred**, as in §2. Simpler, but
  averages away the starting tone, which P3 and P8 showed matters.

**Recommendation: A**, with g fitted and also fixed at ½ so the page says
how much the up lean matters. Cost: a few minutes of fitting, no
inference.

What to expect, from §2: the rise at 9 steps should appear, and Group
III's fit should be limited by the early crossover. If the pure circle
still beats the blurred model clearly, the page says so: then for Group
III the model's contribution is the circle itself (the shorter way round,
which P3 found the inference produces), not its graded log odds.

### P26. Other seeds

The seed is used only by the importance sampling and the variational fit
(`scene.evaluate`); the Adam fit to the mode starts from fixed values and
does not use it. So another seed changes the importance and ELBO
estimates, which P9 found noisy (effective sample sizes of 1 to 7), and
not the mode or the Laplace estimate. Different machines do change the
mode a little [measured on (b), see results/README seeds note].

Measured on one fit (pair 1 → 2, harmonic up, cloud CPU, 4 threads,
alone): mode fit 42 s, Laplace and importance sampling 15 s, variational
fit 254 s, importance sampling from it 3 s [measured, `pair.py`]. So the
part that uses the seed is about 80% of a fit, and a seed costs nearly a
whole rerun. P9's last section is 459 fits (510 files; the rest have no
hypothesis above the floor), 100 fit-hours on the cloud at 4 at a time
[measured, sum of the `seconds` column]. The same seed on this container
gives the same mode and Laplace value and an importance estimate 0.3
apart (−10238.1 against −10238.4).

- **(A) Seeds 0, 1 and 2, from one mode fit.** `pair.py --seeds` (added
  with this note): fit the mode once, then run the sampling once per seed,
  each seed's rows to its own file in the usual format. All 459 fits on
  Cho's PC, 6 at a time with one thread each, so seed 0 is also rerun on
  the same machine and the machine's effect is separate from the seeds'.
  Cost about 27 hours [estimate: 100 fit-hours, PC 1.6 times the cloud
  per fit, 6 at a time, 2.6 fits' worth of work per hypothesis; could be
  22 to 35]. Report per pair the spread of L across seeds, and refit P9
  and P25 per seed and on the seed average.
- **(B) Seeds 0 to 4**, the same way: about 45 hours.
- **(C) Rerun the whole fit per seed**, as `--seed` does now: the same
  numbers as (A) (checked on one pair with short fits: identical rows),
  about 20% more time.
- **(D) Also P3 and P8** (without the options). Their listener fit is
  superseded by P9's options run; they can follow if (A) shows seeds
  matter.

**Recommendation: A**, and (B) only if three seeds disagree on a
conclusion.
