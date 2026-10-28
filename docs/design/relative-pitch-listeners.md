# Relative pitch: listener noise against Shepard's groups — design

Status: **accepted by Cho on 2026-10-28** (P15 to P18 as recommended; see §5).
Written 2026-10-28. Carries out decision P9 (B) of
[relative-pitch.md](relative-pitch.md), "experience as noise, fitted per
Shepard group when comparing with Fig. 3", and its decision P5 (B), a
response model on top of P(up). Decisions continue that document's
numbering after [relative-pitch-prior.md](relative-pitch-prior.md) (P10 to
P14). AI-assisted (Claude); code as of `197d15c`.

**How numbers and sources are marked** (as in relative-pitch.md).

- **[Shepard §x]**: Shepard (1964), *Circularity in judgments of relative
  pitch*, JASA 36, 2346–2353.
- **[measured]**: from a script, with its path.
- **[estimate]**: my estimate, with its basis next to it.

## Decisions at a glance

| Decision | Accepted |
|---|---|
| P15. Where the noise goes | (A) A decision temperature and a lapse rate on the model's log odds of up against down; no refitting |
| P16. What is fitted | (A) Temperature and lapse per group, to Groups I and III of Fig. 3 by binomial likelihood |
| P17. Which log odds | (A) P8's: all three readings at 3 to 5 steps, P3's elsewhere |
| P18. Check of the reading | (A) Refit a few pairs with the likelihood SD the fitted temperature implies, and compare |

---

## 1. Why

On [P3](../../results/P3-shepard-pairs.md) and
[P8](../../results/P8-shepard-pairs-octave-check.md) the model gives P(up)
of 0 or 1 for almost every pair, with log odds of up against down between
about 10 and 100 for most pairs [measured, `results/data/P8/summary.txt`].
Shepard's listeners are graded: Group I, his 26 most accurate on pure
tones, falls from 1.00 at 2 steps to 0.25 at 6; Group III, his 14 least
accurate, stays between 0.24 and 0.83 [Shepard Fig. 3, read by eye]. He
suggests that the less accurate listeners have a larger difference limen
[Shepard §V]. To compare the model with these curves it needs a way to be
unsure, and the way it is made unsure should say something about what
differs between listeners.

What Fig. 3 gives: for each group, the fraction of presentations judged
higher at 1 to 9 steps clockwise, pooled over the 10 starting tones; each
of the 90 pairs was presented once to each listener in the last sequence
[Shepard §III], so each point is about 260 trials for Group I and 140 for
Group III [estimate, 26 or 14 listeners × 10 pairs]. Group II (10
listeners) is not plotted.

## 2. Decisions

### P15. Where the noise goes

- **(A) Decision temperature and lapse.** For each pair, with L the model's
  log odds of up against down (the posterior mass of the hypotheses fitted
  as up against those fitted as down),

  P(judged up) = λ/2 + (1 − λ) · logistic(β · L).

  β ≤ 1 is an inverse temperature and λ a lapse rate (guesses). No fit is
  rerun: L comes from the existing outputs. Because L is mostly a
  difference of log likelihoods, scaling it by β is close to raising the
  likelihood's SD from 10 dB to 10/√β dB, a noisier internal
  representation, which is Shepard's difference-limen idea in the model's
  own terms. The priors are not scaled, so this is an approximation; P18
  checks it.
- **(B) Likelihood SD per group, refitted.** The same idea done exactly:
  rerun every pair at several likelihood SDs and pick the one that fits
  each group. Each SD costs the full 17-hour run (P3) [measured, P3 page],
  so a search over SDs is days of cloud time.
- **(C) Internal noise per trial.** Add noise to the cochleagram, rerun
  inference per simulated trial, and count the answers. Closest to a
  listener, but the cost is (B)'s times the number of trials.

**Recommendation: A**, with P18 to check that it reads as a likelihood SD.
(B) only if the check fails; (C) not now.

### P16. What is fitted

Fitted by maximum binomial likelihood to Fig. 3's fractions times the trial
counts above, with the model's prediction at each step the mean of
P(judged up) over its 10 pairs.

- **(A) β and λ per group.** 2 parameters per group, 9 points each.
- **(B) λ per group, β shared.** Tests whether the groups differ only in
  guessing.
- **(C) β per group, no lapse.** Tests whether noise alone does it.

**Recommendation: A, and report B and C beside it** (each a few seconds to
fit), so the page says which parameter the groups differ in. With 2
parameters for 9 points this is a description, not a test; what it can
show is where the shape fails. One thing to state in advance: Group III's
rise from 0.26 at 8 steps to 0.40 at 9 [Shepard Fig. 3] cannot come from
(A), which is monotone in L, and the model's L is most negative at 9 steps
[measured, P3 summary]. If the fit misses there, that points to something
the model lacks (for example hearing the second tone an octave off), not to
noise.

**Guards against overfitting** (added 2026-10-28 at Cho's question). Two
parameters cannot bend the curve into any shape, but the noise can absorb
the model's errors: an early crossover, for instance, is smoothed over by a
larger temperature, and a good fit then hides it. So the page also reports:

1. **A baseline that ignores the model:** P(judged up) = λ/2 + (1 − λ) ·
   logistic(a − b · steps), 3 parameters per group, the same likelihood.
   If the model-based fit is not better than this, the model adds nothing
   beyond "fewer steps clockwise, more up".
2. **Leave one step out:** fit on 8 steps, predict the 9th, for every step
   and both fits; the page reports the held-out binomial log likelihood,
   not only the fit.
3. **Data not used in the fit:** Shepard's first sequence of complex tones,
   adjacent pairs only, where Group I judged over 99% as the circle
   predicts, Group II 87 to 100% per pair and Group III 63 to 82%
   [Shepard §IV]. Group I's and III's fitted parameters predict their
   adjacent-pair accuracy (a weak check, since Fig. 3 at 1 step already
   covers it); Group II has no fit and is not predicted.

### P17. Which log odds

- **(A) P8's**: at 3 to 5 steps the posterior over all three readings of
  the first tone, elsewhere P3's.
- **(B) P3's** throughout (one reading).

**Recommendation: A.** P8 showed the single reading gave a starting-tone
effect that was not the model's.

### P18. Check of the reading

- **(A) Refit a few pairs at the implied SD.** For Group I's fitted β, rerun
  6 pairs (one each at 3, 4, 5 and 6 steps, plus two at 5 steps from tones
  where P(up) differs) with the likelihood SD 10/√β, and compare the new L
  with β · L. About 2 to 3 hours of cloud time [estimate, 6 pairs × 4 to 12
  fits at P3's 7 to 24 min each, 4 at a time].
- **(B) Skip it** and call β a decision temperature only.

**Recommendation: A.** It decides whether the page may say "listeners
differ in internal noise" or only "in decision noise".

## 3. What comes out

A results page, P9: the fitted parameters per group and per variant
(P16), the fitted curves over Fig. 3, the implied likelihood SD with the
check of P18, and where the fit misses. Code: a fitting script in
`tools/relative_pitch/` reading the existing P3 and P8 outputs, with a
test on made-up log odds whose β and λ it must recover.

## 4. Not in scope

- Learned intervals for musicians (P9 (C)): no data split by training.
- Order effects across trials (Shepard's sequences were counterbalanced
  [Shepard §III]).
- Other seeds; P18's refits use seed 0 as P3 did.

## 5. Cho's answers

2026-10-28: all four as recommended ("Accept all" on the decision card).
Then, asked whether this overfits, Cho accepted the three guards added
under P16.
