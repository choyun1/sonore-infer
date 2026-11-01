# Relative pitch: Group III's errors at small steps, and other seeds — design

Status: **draft, waiting on Cho's answers** (P24 to P27).
Follows [P9](../../results/P9-shepard-listener-noise.md) and
[P10](../../results/P10-earlier-experiments-hearing.md). Decisions continue
the numbering of [relative-pitch-dominance.md](relative-pitch-dominance.md)
(P19 to P23). AI-assisted (Claude); code as of `752f48b`.

**How numbers and sources are marked** (as in relative-pitch.md).

- **[Shepard Fig. 3]**, **[Shepard §IV]**: Shepard (1964), *Circularity in
  judgments of relative pitch*, JASA 36, 2346–2353. Fig. 3 read by eye
  (`.notes/inputs.md` in the project files).
- **[measured]**: from a script, with its path.
- **[estimate]**: my estimate, with its basis next to it.

## Decisions at a glance

| Decision | Recommended |
|---|---|
| P24. What to explain | (A) Group III's lower accuracy at the smallest steps in both directions, with the 9-step rise as one end of it |
| P25. Mechanism | (A) Noise on the interval between the two pitches, after the scene model, fitted per group; no new fits |
| P26. What to report | (A) As P9, plus a curve of interval noise on the nominal steps as a second baseline and a rough size of the effect |
| P27. Other seeds | (A) Seeds 1 and 2 on the 50 pairs at 3 to 7 steps, all three readings, both options, on Cho's PC, plus a machine check of 30 seed-0 fits |

---

## 1. Why

Shepard's Group III (his 14 least accurate listeners on pure tones) judged
the second tone higher in 0.74, 0.78, 0.83, 0.75, 0.52, 0.29, 0.24, 0.26
and 0.40 of presentations at 1 to 9 steps clockwise [Shepard Fig. 3]. The
model, with the threshold and dominance options, gets the shape of Group I
roughly right but misses this rise from 0.26 at 8 steps to 0.40 at 9. P9
said in advance that its noise fit could not produce it, because the
predicted fraction is monotone in the model's log odds L, and L is most
negative at 9 steps.

**Read as accuracy, the 9-step point is not on its own.** 9 steps
clockwise is 1 step counterclockwise. Folding the two directions together,
Group III answers as the circle predicts in 0.67 of presentations at 1
step (0.74 up, 0.60 down), 0.76 at 2, 0.80 at 3 and 0.73 at 4 [computed
from Shepard Fig. 3]. So the rise at 9 steps is the down side of a dip at
the smallest step that is also there going up (0.74 at 1 step against 0.83
at 3). Shepard reads it the same way: at a tenth of an octave Group III
"give the 'correct' response on only about 68% of the presentations", and
at about three tenths "their performance improves to about 80%" before the
ambiguity near 5 steps [Shepard §IV]. In the adjacent-pair sequence,
Group III's accuracy was 63 to 82% per pair, with reversals "roughly
uniformly distributed around the circle" [Shepard §IV].

**The model cannot give this as it stands.** Its |L| is largest at the
smallest steps: +15.7 to +24.3 at 1 step and −17.7 to −27.6 at 9, against
+3.7 to +18.3 at 3 [measured, `results/data/P9/options/fit_all.txt`]. For
the model, a tenth of an octave is the easiest pair; for Group III it is
the hardest outside the half octave. Any decision noise or lapse applied to
L keeps the smallest steps the most accurate.

**How big the effect is** [estimate]: if every presentation were an
independent trial (14 listeners × 10 pairs × 2 directions = 280 per step),
accuracy at 1 step (0.67) is 2.4 standard errors below 2 steps (0.76) and
3.4 below 3 steps (0.80); the 9-step point alone, 0.40 against 0.26 at 8
steps, is 2.5. Presentations from the same listener are not independent,
and the values are read by eye to about ±0.01 to 0.02, so the true
uncertainty is larger. It is a modest effect in one group, but Shepard
reports it in words, and it is what separates his groups.

### A first look on the published numbers

Done for this note on Fig. 3's fractions and P9's log odds, no new fits
[measured, `circ.py` and `combo.py` with outputs in the project files under
`relative-pitch/group-iii/`]. Same binomial likelihood and trial counts as
P9, so only differences between log likelihoods mean anything.

- **Interval noise on the nominal steps.** The listener hears the interval
  as the true number of steps plus Gaussian noise of SD s, wrapped on the
  circle, and answers up when the heard interval is under 5 steps
  clockwise. One parameter.
- **Interval noise on the model's log odds.** The same noise moves each
  pair along the model's own L as a function of steps (interpolated between
  pairs from the same starting tone), then P9's β and λ apply. Three
  parameters; s = 0 is P9's fit.

| Group III | Parameters | Log likelihood | Held out (leave one step out) | Fitted fractions at 1 to 9 steps |
|---|---|---|---|---|
| Fig. 3 | | | | 0.74 0.78 0.83 0.75 0.52 0.29 0.24 0.26 0.40 |
| P9, β and λ on L (both options) | 2 | −745.1 | −749.7 | 0.75 0.75 0.75 0.65 0.44 0.30 0.25 0.25 0.25 |
| P9 baseline in steps | 3 | −740.1 | −756.8 | 0.74 0.74 0.74 0.73 0.52 0.27 0.26 0.26 0.26 |
| Interval noise on nominal steps, s = 1.94 steps | 1 | **−736.7** | **−738.5** | 0.68 0.79 0.79 0.68 0.50 0.32 0.21 0.21 0.32 |
| Interval noise on L, s = 1.09, β = 2.05, λ = 0.35 | 3 | −741.2 | −753.2 | 0.68 0.78 0.76 0.64 0.46 0.30 0.21 0.21 0.32 |

- **For Group III, noise on the interval is the best description so far,
  with one parameter**, better than every fit in P9 including held out. It
  gives the dip at 1 step and a rise at 9 (0.32 against Shepard's 0.40).
  An SD of 1.94 steps is 0.19 octave, about 2.3 semitones, on the interval
  between two 1/8 s tones.
- **Put on the model's log odds, it gains 3.8 over P9 but loses held
  out**, because the model's crossover is slightly early (P9) and the noise
  then carries that error to the neighbouring steps.
- **Group I is the opposite case.** Interval noise on nominal steps fits it
  very badly (−817.5 against the baseline's −704.6; it predicts a dip to
  0.82 at 1 step where Group I is at 1.00), and on L it fits s = 0.18 and
  gains nothing (−716.4 against −716.5). Group I's errors sit at the half
  octave, where the model's are; Group III's errors are also at the
  smallest steps, where the model makes none.

So, inferred from these fits: Group III's 9-step rise is the signature of
coarse pitch resolution for the interval, a limit the model does not have
(its f0 posteriors are much narrower than a tenth of an octave), and not of
anything in how the scene model groups or orders the tones.

## 2. Decisions

### P24. What to explain

- **(A) The dip at the smallest steps in both directions**, with the
  9-step rise as one end of it. Fits and checks use all nine steps.
- **(B) The 9-step point alone**, for example by asking what makes a
  1-step fall sound like a rise.

**Recommendation: A.** The data and Shepard's own reading treat it as one
effect (§1), and the mirror point at 1 step clockwise is there too. A
mechanism that raises only the 9-step point would also need to explain why
Group III's errors at 1 step clockwise are about as common.

### P25. Mechanism

- **(A) Noise on the interval, after the scene model.** The model infers
  each pair as now. The listener then holds the interval with Gaussian
  noise of SD s per group (in steps, that is in log frequency), as in §1:
  on nominal steps and on the model's L, with β and λ as in P9. This reads
  as noise in remembering or comparing the first tone's pitch, which a
  scene model of the sound does not include. No new fits: a few seconds on
  P9's log odds.
- **(B) Per-trial noise in the sensory representation.** Add noise to each
  presentation's cochleagram and refit, so that P(judged up) is the
  fraction of noisy presentations where the model says up. This is the
  generative "noisy observer" and would keep everything inside the model.
  But the up-minus-down evidence grows with the distance between the two
  hypotheses' predicted cochleagrams, and its noise with the same distance,
  so its accuracy should follow |L|, which is largest at 1 step [inferred,
  not checked]. Cost: about 6 fits per pair per noise draw at about 12 min
  each [estimate, median 711 s in `results/data/P9/options/all/`], so 10
  draws on 4 pairs is about 50 CPU hours.
- **(C) Listening to single components** (Terhardt's spectral pitch rather
  than virtual pitch). In a Shepard tone every component moves by the same
  tenth of an octave, so a listener following any one component gets the
  same shorter-way answer as the model. A dip at small steps still needs a
  limit on resolving a tenth of an octave, which is (A) again.
- **(D) Record and stop.** Document the dip as a known difference from
  Group III and leave it.

**Recommendation: A**, as a description of Group III, with its parameter
stated as noise outside the scene model. If Cho wants a check that (B)
cannot do it, the cheapest is 2 pairs (1 and 3 steps from tone 1) with 10
draws at one noise SD, about 25 CPU hours [estimate]; I would not run it
unless the answer to (A) makes it matter.

### P26. What to report

- **(A) As P9**: the fit, the baseline in steps, leave one step out, and
  the adjacent-pair check (Group III 63 to 82% per pair [Shepard §IV]),
  plus interval noise on nominal steps as a second baseline, and the size
  of the effect as in §1 with its caveats. Both groups, so the page shows
  that the noise Group III needs is not there in Group I.
- **(B) Only the Group III fit.**

**Recommendation: A.** The second baseline matters here: on §1's numbers it
beats the model-based fit, which says that for Group III the scene model
adds nothing beyond the number of steps. The page should say so.

### P27. Other seeds

Every Shepard pair result so far is one seed (seed 0, as P3, P8 and P9).
Earlier seed checks (A8, B4) moved values by 0.5 to 4 per point and
changed no conclusion, but running the same seed on a different machine
also moved some values by up to 10 (A8).

- **(A) Seeds 1 and 2 where the answer can turn**: the 50 pairs at 3 to
  7 steps, where the smallest |L| is 3.7 to 5.1 and many are under 10
  (P9's table above), all three readings, harmonic only, both options: 300
  fits per seed, 600 in all, on Cho's PC. Plus 30 seed-0 fits on the PC
  to separate machine from seed (below). Then refit P9's Group I and III
  fits and P25 on the mean L over seeds, with the spread per pair. About
  125 CPU hours at the cloud's speed, or about 10 to 16 hours of wall time
  on the PC with 6 fits at a time [estimate: 711 s median per fit on P9's
  run, PC CPU 1.3 to 2 times faster per process (O2), scaling to 6
  processes not measured].
- **(B) All 90 pairs, seeds 1 and 2**: 1020 fits, about twice (A). At 1, 2,
  8 and 9 steps |L| is 9.9 to 27.6, so a seed spread of a few cannot turn
  them.
- **(C) One more seed on all 90 pairs.**

**Recommendation: A.** It covers every pair whose answer a seed could turn
and keeps the run within a day on the PC. The machine check is 15 pairs ×
2 hypotheses (up and down, 52 to 104 Hz reading), chosen so each starting
tone appears at least once.

## 3. Code (if accepted)

- `tools/relative_pitch/listener_noise.py`: add the two interval-noise
  fits of §1 (on nominal steps and on L), with leave one step out and the
  adjacent-pair check, and a `--seeds` way of averaging L over seed
  directories. The probes `circ.py` and `combo.py` are the starting point.
- No change to the scene model or to `pair.py`, which already takes
  `--seed`.
- Results: a new page P11 for the Group III fits and the seeds, with data
  under `results/data/P11/`.

## 4. Cho's answers

(waiting)

## 5. Not in scope

- Group II, which Fig. 3 does not plot.
- Shepard's per-listener data, which the paper does not give; with it, a
  mixture of guessers and circle-followers within Group III could be
  tested against interval noise.
- Whether the options become defaults (P13), which waits on P10's 100 Hz
  harmonic 1 result.
