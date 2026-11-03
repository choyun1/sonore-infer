# P12. Shepard pairs on three seeds

> **Under review.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** done, seeds 0 to 2 on Cho's PC.
**Paper target:** none in Cusimano et al.; listeners in Shepard (1964)
Fig. 3 and §IV.
**Code:** `tools/relative_pitch/pair.py --seeds`,
`tools/relative_pitch/run_seeds.py`, `tools/relative_pitch/average_seeds.py`,
`tools/relative_pitch/listener_noise.py --interval-noise`.
Design: [relative pitch, seeds](../docs/design/relative-pitch-seeds.md),
decision P27.

## Question

Every Shepard page so far is one seed. Do [P09](P09-shepard-listener-noise.md)'s
listener fit with both hearing options and [P11](P11-group-iii-interval-noise.md)'s
interval noise hold on other seeds?

## What was run

P09's last section again: all 90 pairs with `--threshold-floor --dominance`,
harmonic hypotheses only, from all three readings of the first tone. That
is 459 fits in 510 files; the other 51 files have no hypothesis above the
floor. Seeds 0, 1 and 2 were run on Cho's PC, 6 at a time with one thread
each [measured, `run_seeds.py`, 2026-11-01 to 2026-11-03, about 28 h of
running]. The seed is used only by the importance sampling and the
variational fit, so each hypothesis's mode was fitted once and then
sampled once per seed. Seed 0 is rerun on the PC so that the machine's
effect can be told apart from the seeds'. P09's own run is seed 0 on the
cloud.

Outputs are in `results/data/P09/seeds/` (`s0`, `s1`, `s2`, logs in
`logs/`). `mean/` holds the seed average from `average_seeds.py`: per
hypothesis, the log of the mean over seeds of the exponentiated evidence,
which is what pooling the seeds' importance samples with equal weight
gives. `average.txt` lists each pair's log odds on each seed, and
`fit_s0.txt` to `fit_mean.txt` hold the listener fits. Five jobs were
killed when a console window closed overnight and were rerun from
scratch. One of them had been skipped by the runner and was rerun on its
own; the runner is now fixed.

## Results

**Seed noise in the log odds.** The log odds L of up against down on
one pair vary across the three seeds by a median of 2.7 to 3.7 at each
number of steps, and by at most 9.6 [measured, `average.txt`]. The modes
and the Laplace values are the same on every seed. Seed 0 on the PC
against the cloud: modes within 0.1, Laplace within 0.4.

**The listener fit**, model (A) with β and λ per group, and P11's interval
noise on the model's log odds:

| | Cloud, seed 0 (P09) | PC, seed 0 | PC, seed 1 | PC, seed 2 | Seed average | Baseline in steps |
|---|---|---|---|---|---|---|
| Group I, β | 0.247 | 0.251 | 0.256 | 0.219 | 0.234 | |
| Group I, log likelihood | −716.5 | −717.1 | −720.7 | −730.5 | −724.1 | −704.6 |
| Group I, held out | −723.8 | −725.9 | −731.0 | −742.3 | −734.3 | −708.6 |
| Group I, fitted at 4, 5, 6 steps | .69 .42 .21 | .68 .43 .19 | .73 .40 .19 | .69 .40 .17 | .70 .40 .17 | Shepard .75 .53 .25 |
| Group III, λ | 0.500 | 0.500 | 0.505 | 0.520 | 0.510 | |
| Group III, log likelihood | −745.1 | −745.2 | −744.3 | −747.2 | −746.1 | −740.1 |
| Group III, interval noise on L, log likelihood | −741.3 | −740.9 | −738.7 | −745.1 | −743.7 | nominal circle −736.8 |
| Group III, interval noise on L, held out | −752.2 | −752.3 | −748.5 | −756.0 | −754.3 | nominal circle −738.3 |

## What it says

- **P09's Group I gap was at the lucky end.** On the cloud seed the model
  was 12 behind the curve in steps alone (15 held out). Across the three PC
  seeds it is 13 to 26 behind (17 to 34 held out), and 20 (26) on the seed
  average. So the seed moves the fit by up to 14. That is less than the
  options' gain of 34 (P09 before the options: −750.6), so the gain holds,
  but the gap that remains is about 20, not 12.
- **The early crossover holds on every seed**: 0.40 to 0.43 at 5 steps,
  against Shepard's 0.53.
- **The machine matters less than the seed.** Seed 0 gives −716.5 on the
  cloud and −717.1 on the PC. Seeds 1 and 2 on the same PC give −720.7
  and −730.5.
- **Group III does not change.** It is still mostly guessing (λ 0.50 to
  0.52). P11's finding holds: interval noise with no model beats interval
  noise on the model's log odds on every seed, by 2 to 8 in fit and 10 to
  18 held out. In the PC fits of P11's version on the model's log odds,
  β reaches the top of its grid (10), so that fit is not determined.
- **Seed noise is about 3 nats per pair** with effective sample sizes of a
  few. At Group I's β of about 0.24, that is about 0.7 on the logit scale,
  which is enough to move pairs near the crossover. Comparisons between
  hypotheses on these importance estimates should allow for it.
- **Not checked:** more seeds (decision P27 (B), about 45 h for five);
  more samples per seed, which would shrink the noise without more seeds;
  P03 and P08 without the options.
