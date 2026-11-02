# Order effects: why the model resets wrongly after silence — design

Status: **E8–E11 accepted by Cho, 2026-11-02**, as recommended. E8's run
then showed the rhythm prior is the model's only push towards a reset, so
Cho dropped E9 the same day and asked for E10 to be designed: §6, decisions
E12 and E13, **accepted by Cho as recommended, 2026-11-02**. Written 2026-11-01. Follows experiment O3
(`results/O3-order-effects.md`) and continues the numbering of
[order-effects.md](order-effects.md) (E8 onwards). AI-assisted (Claude);
code as of `0bd1aa3`.

**How numbers and sources are marked** (as in order-effects.md).

- **[paper §x]**: Cusimano, Hewitt & McDermott (2024), *Listening with
  generative models*, Cognition 253, 105874.
- **[paper]**: Beauvois & Meddis (1997) or Cusack et al. (2004), read in
  full.
- **[measured]**: from a run or script in the project, with its path.
- **[estimate]** or **[inference]**: mine, with its basis next to it.

## Decisions at a glance

E8–E11 accepted by Cho on 2026-11-02 (reasons in §3); E9 dropped and E10
replaced by E12 the same day. E12 and E13 accepted as recommended the
same day (§6).

| Decision | Status |
|---|---|
| E8. Check the cause first | (B) One diagnostic run with the long rest left out of the rhythm prior, before any model change |
| E9. Rhythm across a silence | ~~(C) A heavy-tailed rest prior~~ dropped after E8: it would remove the model's only push towards a reset |
| E10. The decision across a silence | Replaced by E12 |
| E11. The two-stream baseline on Beauvois & Meddis's test | (A) Set aside with the ABA margins against graded listener data |
| E12. Reset after silence | Accepted: (C) after a silence of 0.5 s or more, the decision is emptied and the sources end, so what follows is heard afresh; a readout and structure rule, reported as a departure, with (B) beside it |
| E13. Beauvois & Meddis's inducer effect | Accepted: (A) out of reach of E12; recorded as a gap, its likely cause (adaptation) left for later |

---

## 1. What O3 found

With one run per stimulus [measured: `results/data/O3/`]:

- **Baseline.** Beauvois & Meddis's test alone (1.44 s of ABAB at 90 ms,
  6 semitones) gives two streams by 122. Listeners always heard it as one
  [paper].
- **Inducer.** Straight after 10 s of A tones the test leans *less* to two
  streams (65). From 0.5 to 8 s of silence it gives 113 to 137, with no
  decay. Listeners' segregation falls from 8.2 to 2.9 of 10 over 0 to
  8 s [paper].
- **Gap.** After a Cusack et al. gap, the leak carries the lean over (67%
  after 1 s). The first triplet after the gap then pushes towards one
  stream, harder after longer gaps (4 st: −15, −30, −31 after 1, 2, 5 s,
  against +4 for a fresh sequence). Listeners start every second sequence
  from one stream and build up as from scratch, whatever the gap
  [paper].
- **Restart.** The model prefers new sources after the gap only from 5 s
  on (Cusack) and always for Beauvois & Meddis.

## 2. The cause (inferred, then checked by E8)

Each source's log rests between events share one mean and precision,
drawn from a normal-gamma and integrated out [paper App. A.2, Eqns
A.9–A.10; `priors.NormalGammaMarginal`]. After 10 s of regular tones the
pooled rests are tight, so one long rest across the gap is very unlikely,
and every source that crosses the gap pays for it. Two streams have two
sources crossing a Cusack gap, one stream has one.

Evaluated at the tones' nominal times [measured: the calculation in O3's
"Why" section], this prior alone moves the two-over-one log odds by −36,
−51 and −70 for 1, 2 and 5 s gaps against no gap, and favours restarting
by +19, +36 and +58. That matches the sign and roughly the size of O3's
post-gap drops and restart gains. It also gives −33 for Beauvois &
Meddis at 0 s, where O3 measured −34, but it does not explain their
other silences.

So a silence acts on the model through rhythm, as a cost that grows with
the gap's length. Listeners behave as if a silence of 1 s or more simply
ends what came before (Cusack et al.), or leaves a bias that fades over
seconds (Beauvois & Meddis), and Cusack et al. suggest the two effects
involve different mechanisms [paper].

## 3. Decisions

### E8. Check the cause first

- **(A) Go straight to a model change** (E9).
- **(B) One diagnostic run first.** Rerun a short Cusack et al. stimulus
  (first sequence cut to 2 s, 5 s gap, 4 st, all four hypotheses) with
  the rest across the gap left out of the rhythm prior, a switch for
  this run only. If §2 is right, the first post-gap window goes back to
  about a fresh sequence's term and the restart gain shrinks to the
  other new-source costs.

**Recommendation: (B).** §2 is checked only at nominal times. One run of
a few hours on the cloud [estimate: O3's 5 s runs per hypothesis, cut to
a 2 s first part] settles it before the model changes.

### E9. Rhythm across a silence

- **(A) Leave the rhythm prior as the paper has it.** The model keeps
  getting Cusack et al. wrong.
- **(B) Exclude rests longer than a fixed cutoff** (say 1 s) from the
  pool. Simple, but a hard threshold the paper does not have and a free
  parameter.
- **(C) A heavy-tailed rest prior.** Each rest is either a regular rest,
  pooled as now, or with a small fixed probability a break drawn from a
  broad prior that does not use the source's rhythm; the two are
  marginalized, as P10 marginalized harmonic presence. A long silence
  then costs about the same for one source or two, and a source's rhythm
  survives a break.
- **(D) Sources end at a silence.** Make the restart structure the only
  one after a gap longer than a cutoff. This builds Cusack et al.'s
  reset in by hand and cannot give Beauvois & Meddis's gradual decay.

**Recommendation: (C), off by default** (as P10–P13). It is a departure
from the paper's prior, so it would be reported as one. **Prediction**
[inference]: the post-gap drop disappears, so after 1 s the second
sequence starts from the leak's carry-over (still wrong, see E10), and
restart stops winning at 5 s unless other new-source costs favour it.

### E10. The decision across a silence

The leak (O10, τ 2.5 s) carries 67% of the log odds across 1 s. Cusack
et al.'s listeners carry none.

- **(A) Leave the leak as it is** until E9 is measured.
- **(B) A gate:** no sound for longer than some time empties the decision,
  as E7 (D) proposed for attention.
- **(C) Fit τ** to Cusack et al. Ruled out by E4: fitting τ to the data it
  predicts would make the test circular.

**Recommendation: (A) now.** (B) may be what listeners do, but it is a
second change on top of E9 and needs its own evidence; Beauvois &
Meddis's gradual decay argues against a full gate after every silence.

### E11. The two-stream baseline

The test alone gives two streams by 122, the same over-strong streaming
at fast rates as C2's ABA margins (up to +217), which
`results/README.md` defers until graded listener data are compared.

- **(A) Set aside with the ABA margins.**
- **(B) Investigate now.**

**Recommendation: (A).** It is not specific to order effects, and the
reset questions above can be answered on the differences between
conditions.

## 4. Order of work

1. E8 (B): the diagnostic run; a short note on the O3 page.
2. If §2 holds: E9 (C) as an option, with a test; O3's Cusack runs at
   1, 2 and 5 s rerun with it on Cho's PC (two at a time,
   `OMP_NUM_THREADS=2`); page O3 updated.
3. E10 revisited with those results.

## 5. Questions for Cho

Answered 2026-11-02: E8–E11 accepted as recommended; E9 dropped and E10
replaced after E8 (§6).

E12 (C) and E13 (A) accepted as recommended, 2026-11-02.

## 6. After E8: resetting after silence (E12, E13)

E8 [measured: `results/O3-order-effects.md`, "Checked: the long rest left
out"] showed that without the long rest in the rhythm prior the model
gives *more* two-stream evidence after a gap than for a fresh sequence. A
reset therefore has to come from somewhere else. E10 had one candidate,
a gate on the decision; E12 sets it out with a stronger form.

Every option below can be computed from O3's outputs with no new
inference, because the window terms are logged per hypothesis
[measured: `tools/online/reset_summary.py results/data/O3/cu*.txt`, rows
"gated" and "restart"]. Leaky decision (τ 2.5 s) at times after the
second sequence's first tone, against the first sequence at the same
times:

| | s | 0.5 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- | --- |
| 4 st | fresh sequence | 3.6 | 7.7 | 17.9 | 25.7 | 30.6 |
| | (A) leak, after 1 / 2 / 5 s | 7.8 / −13.7 / −25.6 | 16.2 / −11.8 / −25.0 | 32.6 / 20.8 / −7.7 | 53.5 / 47.4 / 3.0 | 72.1 / 39.5 / 13.3 |
| | (B) gate, after 1 / 2 / 5 s | −15.2 / −29.5 / −31.3 | −2.6 / −24.8 / −29.8 | 20.1 / 12.1 / −10.9 | 45.1 / 41.5 / 0.9 | 66.4 / 35.6 / 11.9 |
| | (C) gate and restart, after 1 / 2 / 5 s | −1.1 / −0.4 / −0.7 | 9.8 / 8.8 / 8.5 | 4.1 / 11.4 / 18.3 | 45.2 / 50.1 / 26.8 | 44.2 / 47.5 / 30.2 |
| 6 st | fresh sequence | 7.0 | 22.7 | 46.5 | 58.8 | 77.0 |
| | (A) leak, after 1 / 2 / 5 s | 49.4 / 32.5 / −15.9 | 68.1 / 43.5 / 6.5 | 79.4 / 54.8 / 15.8 | 79.2 / 65.8 / 42.2 | 83.8 / 74.1 / 59.0 |
| | (B) gate, after 1 / 2 / 5 s | 4.7 / 1.7 / −25.3 | 31.4 / 18.3 / −1.1 | 54.8 / 37.9 / 10.7 | 62.7 / 54.5 / 38.7 | 72.7 / 66.5 / 56.6 |
| | (C) gate and restart, after 1 / 2 / 5 s | 8.8 / 8.6 / 7.5 | 21.1 / 20.5 / 21.2 | 44.8 / 44.1 / 44.1 | 57.8 / 57.7 / 59.8 | 78.7 / 78.2 / 70.1 |

### E12. Reset after silence

- **(A) The leak alone** (O10 as is). The second sequence starts leaning
  to two streams after 1 s and to one after 5 s: a Gap × Time interaction
  listeners do not show.
- **(B) Gate the decision.** At the end of a silence of at least s_g
  seconds the decision is emptied. Every second sequence then starts at
  0, as listeners' do, but its first triplet still pushes towards one
  stream (rhythm prior) and the buildup still depends on the gap (4 st:
  66 after 1 s, 12 after 5 s, against 31 fresh).
- **(C) Gate the decision and end the sources.** After such a silence,
  only the restart hypotheses (E3) go on: new sources, nothing carried in
  the evidence or the decision. At 6 st all three gaps then track the
  fresh sequence within 7 at every time. At 4 st the 5 s gap tracks it
  within 5, but after 1 and 2 s the curve runs 7 to 14 below it at 2 s
  and 14 to 24 above it at 3 and 4 s, which is not explained [measured,
  one run each; whether this is fit noise is not checked].
- **(D) A short time constant during silence** (the decision decays with
  τ_s ≈ 0.5 s while nothing sounds). For gaps of 1 s and more it is (B);
  it adds a parameter these data cannot pin down.

s_g has to exceed the silences inside a sequence (at most 125 ms plus
ramps in Cusack et al.'s ABA–, none in Beauvois & Meddis's) and be at
most 1 s, the shortest gap listeners reset after. 0.5 s is proposed, fixed
here and never fitted; every value from about 0.3 to 1 s gives the same
predictions on these stimuli.

**Recommendation: (C), with (B) reported beside it.** (C) is what Cusack
et al. describe, buildup "as from scratch" after every gap, and it is the
only option without a Gap × Time interaction. It also needs no new
inference: the restart hypotheses already exist. It is a rule built in,
not something the model infers, so it would be reported as a departure
from the paper's model and from the model's own evidence, which prefers
to continue after gaps under 5 s (O3). Whether a model can come to
restart on its own, for example with a structure prior that ends sources
at long silences, is left for later.

### E13. Beauvois & Meddis's inducer effect

None of E12's options touches it: the A-only inducer adds nothing to the
decision (both hypotheses explain A tones alike), and after (C) the test
is heard exactly as the test alone. Listeners show a bias that decays
over seconds, and Cusack et al. read it as involving more than the
mechanisms of streaming [paper].

- **(A) Record it as a gap.** The likely candidate is adaptation of the
  channels the inducer drove [inference; neither study tests it], which
  the model has nothing for.
- **(B) Design an adaptation mechanism now.**

**Recommendation: (A).** It is a different mechanism, and E11 already
leaves the test's baseline wrong.

### Order of work for E12

1. Cho decides E12 and E13.
2. With (C) or (B): the gated readout (and for (C) the restart rule) in
   `tools/online/reset_summary.py`, already computing both; O3's page
   gets the table above and a plot; no new runs.
3. A line in `docs/methods.md` on the readout rule.
