# P9. Shepard pairs: listener noise against Shepard's groups

> **Under review.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** first pass, one seed.
**Paper target:** none in Cusimano et al.; listeners in Shepard (1964)
Fig. 3 and §IV.
**Code:** `tools/relative_pitch/listener_noise.py`,
`tools/relative_pitch/pair.py` (`--sigma 25.4` for the check).
Design: [relative pitch listeners](../docs/design/relative-pitch-listeners.md),
decisions P15 to P18 and the guards against overfitting under P16.

## Question

On [P3](P3-shepard-pairs.md) and [P8](P8-shepard-pairs-octave-check.md)
the model answers up or down with near certainty, while Shepard's listeners
are graded. If each group's answers are the model's log odds passed through
some noise, how much noise does each group need, does the model then
describe the groups better than a curve that ignores it, and does the noise
read as a noisier internal representation (Shepard's larger difference
limen, §V)?

## Setup

As the design. For each pair, L is the model's log odds of up against down
over all its hypotheses (P8's three readings at 3 to 5 steps, P3's four
hypotheses elsewhere). A group judges up with probability
λ/2 + (1 − λ) · logistic(β · L); the predicted fraction at each number of
steps is the mean over its 10 pairs. β and λ are fitted per group by
maximum binomial likelihood on a grid, to Fig. 3's fractions read by eye,
with 260 trials per step for Group I and 140 for Group III [estimate, 26
or 14 listeners × 10 pairs]. The likelihood values below share these
counts, so only differences between them mean anything.

The check of decision P18 refitted six pairs (one at 3, 4 and 6 steps,
three at 5) with all their hypotheses at the likelihood SD that Group I's β
implies, 10/√β = 25.4 dB: 38 fits, one seed, cloud CPU, 510 to 1435 s
each; 3 importance estimates failed and their ELBOs stand in.

## Results

All from `data/P9/fit.txt`, made by
`python tools/relative_pitch/listener_noise.py results/data/P3/[0-9]*.txt results/data/P8/[0-9]*.txt --refits results/data/P9/refits/*.txt`.

**Fits.** Fraction judged up at 1 to 9 steps clockwise:

| | β | λ | Log likelihood | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Group I, Fig. 3 | | | | 1.00 | 1.00 | 0.89 | 0.75 | 0.53 | 0.25 | 0.09 | 0.02 | 0.02 |
| (A) β and λ | 0.155 | 0.035 | −750.6 | 0.97 | 0.92 | 0.87 | 0.73 | 0.45 | 0.11 | 0.04 | 0.03 | 0.02 |
| (C) no lapse | 0.126 | 0 | −759.3 | 0.97 | 0.91 | 0.85 | 0.71 | 0.45 | 0.12 | 0.04 | 0.02 | 0.00 |
| Baseline | | 0 | **−704.6** | 0.99 | 0.97 | 0.92 | 0.77 | 0.51 | 0.24 | 0.09 | 0.03 | 0.01 |
| Group III, Fig. 3 | | | | 0.74 | 0.78 | 0.83 | 0.75 | 0.52 | 0.29 | 0.24 | 0.26 | 0.40 |
| (A) β and λ | 0.759 | 0.520 | −742.4 | 0.74 | 0.74 | 0.74 | 0.72 | 0.44 | 0.26 | 0.26 | 0.26 | 0.26 |
| (C) no lapse | 0.024 | 0 | −792.4 | 0.68 | 0.62 | 0.59 | 0.54 | 0.49 | 0.37 | 0.29 | 0.24 | 0.18 |
| Baseline | | 0.515 | **−740.1** | 0.74 | 0.74 | 0.74 | 0.73 | 0.52 | 0.27 | 0.26 | 0.26 | 0.26 |

The baseline ignores the model: λ/2 + (1 − λ) · logistic(a − b · steps),
three parameters. (B), one β for both groups (0.172) and a λ each (0.045
and 0.485), has a total log likelihood of −1501.2, against −1493.0 for (A).

**Leave one step out**, summed held-out log likelihood: Group I, model (A)
−773.1, baseline −708.6; Group III, model (A) −746.8, baseline −756.8.

**Adjacent pairs**, Shepard's first sequence of complex tones (not used in
the fit): (A) predicts 0.97 to 0.98 for Group I (Shepard: over 0.99) and
0.74 for Group III (Shepard: 0.63 to 0.82 per pair).

**P18 check.** If β · L read as a likelihood SD of 25.4 dB, the refits'
log odds would match β times the 10 dB log odds:

| Pair | Steps | L at 10 dB | β · L | L refitted at 25.4 dB |
| --- | --- | --- | --- | --- |
| 1 → 4 | 3 | +17.4 | +2.69 | +8.56 |
| 2 → 6 | 4 | +5.9 | +0.92 | +7.20 |
| 1 → 6 | 5 | −9.7 | −1.51 | −0.15 |
| 6 → 1 | 5 | +3.1 | +0.48 | −4.16 |
| 9 → 4 | 5 | −0.5 | −0.08 | −0.23 |
| 3 → 9 | 6 | −36.5 | −5.65 | −4.00 |

## What it says

- **For Group I the model adds nothing over the number of steps; it does
  worse.** A curve in steps alone beats the model-based fit by 46 in log
  likelihood, and by 64 when each step is predicted from the others. The
  model's answers change too sharply between 4 and 6 steps (0.45 at 5 and
  0.11 at 6, against 0.53 and 0.25), and noise on its log odds cannot
  soften that without also losing the near-perfect answers at 1 and 2
  steps. This is the guard the design added for this case: a noise
  parameter would otherwise have hidden it.
- **Group III is mostly guessing in both fits** (λ about 0.5), and model
  and baseline are about equal (−742.4 and −740.1; held out, the model is
  10 better). Neither reproduces Group III's rise to 0.40 at 9 steps; the
  design said in advance that noise cannot, so it points to something the
  model lacks, such as hearing the second tone an octave off.
- **The noise does not read as a likelihood SD.** Refitting at 25.4 dB
  changes the log odds by amounts unrelated to β · L, up to 8 times larger
  and in one pair (6 → 1) of the opposite sign. So β is a decision noise,
  and these results do not support "Group III has a noisier internal
  representation" in the model's own terms; at 25.4 dB the priors, which β
  does not scale, carry more of the answer.
- **The groups differ mostly in guessing.** One shared β costs 8.2 in log
  likelihood for one parameter fewer, and Group III's β is poorly
  determined once its λ is 0.5.
- **Not checked:** other seeds; the refits at other SDs; fitting the
  likelihood SD per group directly (decision P15 (B)), which the P18
  result makes the relevant next step if a perceptual-noise account is
  wanted.

## Why the crossover is too sharp

Measured with `tools/relative_pitch/decompose.py` (outputs in
`results/data/P9/decompose/`; one seed, the harmonic hypotheses of P3
from the 52 to 104 Hz reading, refitted to their modes with P3's settings,
which give the same log joint at the mode as P3, for example −9758.2 for
1 → 3 up). Each mode's log joint is split into the likelihood of each
tone and the four prior terms. "Everything else" is P3's ELBO difference
minus the likelihood difference at the mode: the priors and the Occam
factor together. Up minus down:

| Pair | Steps | P3 log odds (ELBO) | Likelihood, tone 1 | Likelihood, tone 2 | of which channels below 60 Hz | Everything else |
|---|---|---|---|---|---|---|
| 1 → 3 | 2 | +22.8 | +7.5 | +3.9 | −0.7 | +11.4 |
| 1 → 5 | 4 | +3.3 | +1.7 | −9.3 | −10.9 | +10.9 |
| 1 → 6 | 5 | −10.4 | +0.4 | −20.2 | −18.5 | +9.4 |
| 1 → 7 | 6 | −25.0 | −3.1 | −26.2 | −21.6 | +4.3 |
| 1 → 9 | 8 | −53.6 | −8.1 | −48.5 | −30.5 | +3.0 |
| 6 → 10 | 4 | +13.3 | +1.7 | +0.6 | −0.3 | +11.0 |
| 6 → 1 | 5 | +1.3 | −1.5 | −2.9 | −1.8 | +5.7 |
| 6 → 2 | 6 | −5.7 | −5.3 | −6.6 | −4.1 | +6.2 |

- **The fit to the sound carries the change with steps; the priors do
  not.** "Everything else" is a steady 3 to 11 in favour of up. From tone
  1 the likelihood goes from +11.4 at 2 steps to −56.6 at 8.
- **Most of it is the second tone's lowest channels (20 to 51 Hz).** From
  tone 1 the up hypothesis's second f0 is 103 to 135 Hz at 4 to 8 steps,
  so it predicts those channels at or near the cochleagram's 20 dB floor,
  while they read 25 to 28 dB (means over the second half of the sound).
  The down hypothesis, with its second f0 at 51 to 68 Hz, covers them. The
  shortfall grows as the up f0 rises, so the log odds fall steeply with
  steps.
- **From tone 6 the effect is much smaller** (−0.3 to −4.1 below 60 Hz),
  because the up f0 is only 73 to 83 Hz and still spills into those
  channels. This is the starting-tone dependence P3 and P8 found.

So, inferred from these 8 pairs: the model's up or down answer is
largely decided by which hypothesis accounts for faint energy in the
lowest cochleagram channels, not by comparing pitch distances one way and
the other way round. That energy rises steeply in cost as the up f0 moves
away from it, which makes the crossover sharp. Whether listeners hear that
energy at all is not checked: the cochleagram's levels are not calibrated
to dB SPL here and it has no absolute-threshold or middle-ear weighting.
Not checked either: other seeds, the octave readings of P8, and pairs
from other starting tones.

## With a hearing threshold and spectral dominance (design P19 to P21)

Design [relative-pitch-dominance.md](../docs/design/relative-pitch-dominance.md),
accepted 2026-10-28: `--threshold-floor` floors each cochleagram channel
at the threshold of hearing (Terhardt et al. 1982, Eq. 8; 73 dB at
20 Hz, 39 at 35 Hz and 25 at 51 Hz in the cochleagram's units, the flat
20 dB from 67 Hz up), and `--dominance` gives each channel the likelihood
SD 10 dB / √w with Terhardt's dominance weight w (1 at 700 Hz, 0.27 at
51 Hz). Same 8 pairs and settings as above, one seed.

Up minus down, likelihood at the mode [measured, `decompose.py`, outputs
in `results/data/P9/options/{threshold,dominance,both}/`], and the log
odds of the harmonic hypotheses by the ELBO with both options [measured,
`pair.py --only harmonic --threshold-floor --dominance`, outputs in
`results/data/P9/options/evidence/`]:

| Pair | Steps | Likelihood, neither | threshold | dominance | both | Log odds, neither (P3) | Log odds, both |
|---|---|---|---|---|---|---|---|
| 1 → 3 | 2 | +11.4 | +11.4 | +8.3 | +7.9 | +22.8 | +19.9 |
| 1 → 5 | 4 | −7.6 | +1.9 | +0.8 | +2.1 | +3.3 | +10.3 |
| 1 → 6 | 5 | −19.8 | −5.2 | −4.0 | −1.7 | −10.4 | +7.6 |
| 1 → 7 | 6 | −29.3 | −13.3 | −8.6 | −5.6 | −25.0 | −1.0 |
| 1 → 9 | 8 | −56.6 | −32.0 | −19.9 | −15.0 | −53.6 | −10.6 |
| 6 → 10 | 4 | +2.3 | +3.4 | +2.2 | +2.4 | +13.3 | +13.0 |
| 6 → 1 | 5 | −4.4 | −1.1 | −1.0 | −0.5 | +1.3 | +5.2 |
| 6 → 2 | 6 | −11.9 | −4.8 | −4.2 | −2.9 | −5.7 | −3.0 |

- **Each option alone removes about half of the fall with steps from
  tone 1, and both together about three quarters** (likelihood at 8 steps
  −56.6, −32.0, −19.9, −15.0). The lowest channels' share goes from −39.5
  to −5.1 with both. Part of the dominance option's effect is not specific
  to the low channels: it raises the SD everywhere except near 700 Hz, so
  the channels above 60 Hz also count less (their share at 8 steps goes
  from −17.1 to −10.1).
- **With both, the log odds cross zero between 5 and 6 steps from both
  starting tones** (+7.6 and −1.0 from tone 1, +5.2 and −3.0 from tone 6),
  where without them they crossed between 4 and 5 from tone 1 and between
  5 and 6 from tone 6. Shepard's listeners cross at about 5 steps (Group I
  .53 at 5 steps).
- **The fall is gentler and more even**: from tone 1, +19.9 at 2 steps to
  −10.6 at 8, against +22.8 to −53.6 before.
- At the half octave the model now leans up by 5 to 8 from both tones.
- **The importance estimates are less clear.** With an effective sample
  size of 1 to 7 they differ from the ELBO by up to 5.9 in one
  hypothesis, and their log odds from tone 1 are +24.0, +4.7, +3.4, +1.6
  and −13.2 at 2, 4, 5, 6 and 8 steps, so by them the crossover from tone
  1 is between 6 and 8 steps. P9's fit used the importance estimate where
  it was finite; the rerun should report both.
- **Not checked:** the listener fit itself, which needs all 90 pairs
  (P3 and P8) rerun with the options; other seeds.

## All 90 pairs with both options, and the listener fit again

Every pair rerun with `--threshold-floor --dominance`, harmonic hypotheses
only (the whistles were 91 to 147 behind on P3), from all three readings
of the first tone (the 52 to 104 Hz reading and an octave either side, as
P8 did for 3 to 5 steps): 510 fits, one seed [measured, `pair.py`,
outputs in `results/data/P9/options/all/`; fit by `listener_noise.py`,
output in `results/data/P9/options/fit_all.txt`]. A first fit with the
extra readings at 3 to 5 steps only, as decision P17 had it, made the
predicted fraction rise from 0.40 at 5 steps to 0.46 at 6, because at 5
steps the extra readings turned 5 of the 10 pairs from up to down and at
6 steps there were none (`fit_all_p17.txt`); so all steps now have them.
Whistles are left out here, unlike above.

| | Before (above) | Both options, all readings | Baseline in steps |
|---|---|---|---|
| Group I, β | 0.155 | 0.247 | |
| Group I, λ | 0.035 | 0.010 | |
| Group I, log likelihood | −750.6 | −716.5 | −704.6 |
| Group I, held out | −773.1 | −723.8 | −708.6 |
| Group III, λ | 0.520 | 0.500 | |
| Group III, log likelihood | −742.4 | −745.1 | −740.1 |
| Group III, held out | −746.8 | −749.7 | −756.8 |

Fitted fractions at 1 to 9 steps, Group I: 0.99 0.97 0.89 0.69 0.42 0.21
0.06 0.02 0.01, against Shepard's 1.00 1.00 .89 .75 .53 .25 .09 .02 .02.

- **The options close most of Group I's gap.** The model-based fit gains
  34 in log likelihood and 49 held out, and is now 12 (15 held out) behind
  the curve in steps alone, from 46 (64). The log odds now fall evenly
  with steps, from about +20 at 1 step to about −24 at 9.
- **What is left is a crossover slightly early**: 0.42 at 5 steps
  against 0.53, and 0.69 at 4 against 0.75. At 5 steps the log odds range
  from −8.0 to +5.1 and 7 of the 10 pairs lean down.
- **Group III is unchanged**: still mostly guessing (λ 0.50), and its rise
  to 0.40 at 9 steps is still missed. The octave readings did not produce
  it: every pair 9 steps apart is down by 17.7 or more. Group III's β is
  at the grid's upper bound of 1, so it is not determined.
- **Not checked:** other seeds; the importance estimates (the log odds
  above use them where finite, as before, with effective sample sizes of
  a few); whether the options should become defaults (decision P13); the
  options' effect on the other experiments.
