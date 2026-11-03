# Relative pitch: Shepard pairs on other seeds — design

Status: **started on Cho's go-ahead, 2026-11-01** ("let's start ... other
seeds"); P27 (A) as recommended, run done 2026-11-03, results
[P12](../../results/P12-shepard-seeds.md). Every Shepard
page so far ([P03](../../results/P03-shepard-pairs.md) to
[P09](../../results/P09-shepard-listener-noise.md)) is one seed. Group III's
rise at 9 steps is a separate design (relative-pitch-group-iii.md, P24 to
P26); this note takes P27. AI-assisted (Claude); code as of
`752f48b`.

**How numbers are marked** (as in relative-pitch.md): **[measured]** from a
script, with its path; **[estimate]** my estimate, with its basis.

## Decisions at a glance

| Decision | Recommended |
|---|---|
| P27. Other seeds | (A) Seeds 0, 1 and 2 for P09's 459 fits on Cho's PC, the mode fitted once and the sampling once per seed |

---

## 1. What the seed changes

The seed is used only by the importance sampling and the variational fit
(`scene.evaluate`); the Adam fit to the mode starts from fixed values and
does not use it. So another seed changes the importance and ELBO
estimates, which P09 found noisy (effective sample sizes of 1 to 7), and
not the mode or the Laplace estimate. A different machine does change the
mode a little (the 10-seed runs of milestones (a) and (b)).

Measured on one fit (pair 1 → 2, harmonic up, cloud CPU, 4 threads,
alone): mode fit 42 s, Laplace and importance sampling 15 s, variational
fit 254 s, importance sampling from it 3 s [measured, `pair.py`]. So the
part that uses the seed is about 80% of a fit, and a seed costs nearly a
whole rerun. P09's last section (both options, three readings) is 459
fits in 510 files (the rest have no hypothesis above the floor), 100
fit-hours on the cloud at 4 at a time [measured, sum of the `seconds`
column].

## 2. P27. Other seeds

- **(A) Seeds 0, 1 and 2, from one mode fit.** `pair.py --seeds`: fit the
  mode once, then run the sampling once per seed, each seed's rows to its
  own file in the usual format (checked on one pair with short fits: rows
  identical to separate `--seed` runs). All 459 fits on Cho's PC with
  `tools/relative_pitch/run_seeds.py`, 6 at a time with one thread each, so
  seed 0 is also rerun on the same machine and the machine's effect is
  separate from the seeds'. Cost about 27 hours [estimate: 100 fit-hours,
  PC 1.6 times the cloud per fit, 6 at a time; could be 22 to 35]. Report
  per pair the spread of L across seeds, and refit P09's listener model per
  seed and on the seed average.
- **(B) Seeds 0 to 4**, the same way: about 45 hours.
- **(C) Rerun the whole fit per seed**, as `--seed` does: the same numbers
  as (A), about 20% more time.
- **(D) Also P03 and P08** (without the options). Their listener fit is
  superseded by P09's options run; they can follow if (A) shows seeds
  matter.

**Recommendation: A**, and (B) only if three seeds disagree on a
conclusion.

## 3. First numbers from the PC (2026-11-01, first 10 rows)

Projected run time about 28 hours. Seed 0 on the PC against the cloud run:
`log_joint_at_mode` within 0.1, `laplace` within 0.4, `vi_importance` 0.2
to 1.8 apart; across seeds 0 to 2 one row's `vi_importance` spreads by 3.0
(effective sample sizes 1 to 8 of 256) [measured, first rows of
`results/data/P09/seeds/`, not yet committed].
