# Order effects: why the model resets wrongly after silence — design

Status: **draft for Cho**, written 2026-11-01. Follows experiment O3
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

Proposed; none accepted yet. The options and reasons are in §3.

| Decision | Recommended |
|---|---|
| E8. Check the cause first | (B) One diagnostic run with the long rest left out of the rhythm prior, before any model change |
| E9. Rhythm across a silence | (C) A heavy-tailed rest prior: each rest is a regular one or, rarely, a break that the source's rhythm does not explain; off by default |
| E10. The decision across a silence | (A) Leave the leak as it is until E9 is measured |
| E11. The two-stream baseline on Beauvois & Meddis's test | (A) Set aside with the ABA margins against graded listener data |

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

## 2. The likely cause [inference, partly checked]

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

Accept E8–E11 as recommended, or say which to change.
