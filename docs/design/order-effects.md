# Order effects in streaming — design

Status: **draft for Cho, 2026-10-15; E7 added 2026-10-17, checked against
Carlyon et al. (2001) 2026-10-18.** Nothing is
accepted yet; §8 lists the questions in short form.

## Decisions at a glance

Recommendations; nothing is accepted yet. The options and reasons are in
§4, and the questions in §8.

| Decision | Recommended |
|---|---|
| E1. Which effects first | Reset after silence, then the context effect of the previous trial; hysteresis once a source is found; bistable switching later |
| E2. Stimuli | The O1 (App. C.11) tones, an inducer, a silent gap of 0 to 8 s, a test; parameters checked against the papers before running |
| E3. Hypotheses across a gap | The same two, the whistles continuing through the gap; a third, restarting after the gap, as a check |
| E4. Readout | The leaky decision (O10), τ 2.5 s fixed, never fitted to these data; the window terms reported beside it |
| E5. Cost | One window spanning each gap, so silence costs one update; runs on Cho's PC |
| E6. What the model is expected to get wrong | Stated before running: the context effect of the previous Δf, which listeners show as contrast |
| E7. Compute per window | A fixed budget, part of the model, chosen once before these runs and never fitted to listeners; attention modelled later as a gate on what is integrated, not as less compute (Carlyon et al. 2001) |

**How numbers and sources are marked** (as in online.md).

- **[paper §x]**: Cusimano, Hewitt & McDermott (2024), *Listening with
  generative models*, Cognition 253, 105874.
- **[measured]**: from a run already in the project, with its path.
- **[estimate]**: my estimate, with its basis next to it.
- **[abstract]** or **[cited by]**: what I could read of a paper. I could
  not open any of the papers below in full from here; every statement about
  them comes from an abstract or from later papers citing them, and has to
  be checked against the originals before anything is run (E2).

---

## 1. What this is

Online design §8 listed order effects as the place where an online model
can say something a batch model cannot: a batch model gives the same answer
whatever order the sound arrived in, an online one need not. With the online
code merged (O1 page, PRs #46, #47, #54), the model now has a time course of
belief, and a leaky decision (O10). This design asks whether that time
course behaves like listeners' when the sequence is interrupted or preceded
by another one.

It serves the project's main goal, a model of perception that matches
listeners, errors and order effects included. It needs no speed-up: the
stimuli are a few seconds of tones, as in O1.

## 2. What listeners do

- **Reset after silence.** Streaming that has built up is reset, at least
  partly, by a short silent gap or a brief switch of attention (Cusack,
  Deeks, Aikman & Carlyon, 2004, JEP:HPP 30:643–656 [abstract; "partly":
  cited by Kondo et al.'s review]). The pull towards two streams left by
  an inducer decays roughly exponentially with the length of the silence
  (Beauvois & Meddis, 1997, Percept. Psychophys. 59:81–86 [cited by
  Frontiers 2014]), though a later study found some of it after 8 s of
  silence [cited by, 2015, PubMed 26375631].
- **Context of the previous trial.** With ABA– trials of several Δf, a
  larger Δf on the previous trial gives *less* streaming on the current one
  (contrast); hearing two streams on the previous trial gives *more*
  (assimilation) (Snyder, Carter, Lee, Hannon & Alain, 2008, JEP:HPP
  [abstract; the second effect cited by Snyder et al., 2009, not read]).
  The contrast effect weakens as the time between trials grows [cited by].
- **Hysteresis**: staying with an interpretation as Δf is swept up and back
  down. I found no study of it in a short search; not to be used until one
  is found.
- **Bistable switching** in long sequences (Mill et al., 2013, PLoS Comput.
  Biol. 9:e1002925, listed in online.md §8). Needs noise or sampling in the
  model, which the online code does not have.

Attention switches are out of scope: the model has no attention (but see
E7).

## 3. What the model should do, before running anything

With the two fixed hypotheses of O1, a silent gap adds window terms that
are close to equal for both, since neither hypothesis has a tone to explain
there. So:

- **The decision resets only by its leak.** If d is the log odds at the end
  of the inducer, after a gap of g seconds the leaky decision is about
  d e^(−g/τ), then the test sequence adds its own terms. That is an
  exponential decay with the gap, the shape Beauvois & Meddis describe, with
  a time constant τ = 2.5 s set in O10 before any of these data were seen.
  The full sum (no leak) does not reset at all.
- **The evidence may reset as well, and that is the open question.** After
  the gap, the one-stream whistle has to jump from B back to A again, but its
  prior's memory (lengthscale 2.5 s for frequency, 6.9 s for level, as in
  online.md §3) fades over a long gap. If the first test triplets after an
  8 s gap give the same window terms as the first triplets of a fresh
  sequence, the model resets in its evidence, not just its decision. If
  they give the terms of a continuing sequence, only the leak resets it.
  This is what the runs measure.
- **The context effect of the previous Δf should come out wrong.** The leak
  carries the previous trial's log odds into the next. A larger previous Δf
  gives larger log odds for two streams, so the model predicts *more*
  streaming after a large Δf: assimilation, where listeners show contrast.
  It predicts the assimilation from the previous percept correctly, for the
  same reason. Nothing in the model adapts to Δf, which is what contrast
  seems to need.

## 4. Decisions

### E1. Which effects first

**Options.**

- **(A) Reset after silence only.**
- **(B) Reset after silence, then the context of the previous trial.**
- **(C) (B) plus hysteresis**, with a sweep of Δf within one sequence.
- **(D) Bistable switching** in long sequences.

**Recommendation: (B).** Reset after silence has the clearest prediction
(§3) and the most direct test of whether the evidence itself resets.
The context effect is the most informative failure (E6). Hysteresis waits
for a study to compare with. Bistable switching needs a model with noise
and long sequences; it belongs with the speed-ups of
[online-realtime.md](online-realtime.md).

### E2. Stimuli

**Options.**

- **(A) The O1 tones** (A 500 Hz, B 4 or 8 semitones up, 50 ms tones at
  55 dB, 125 ms onset to onset, App. C.11), as an inducer of n triplets,
  a silence of g seconds, then a test of m triplets.
- **(B) Each study's own stimuli**, copied from its methods.

**Recommendation: (A), with values checked against the studies.** (A)
connects to O1 directly: the same tones, the same reference runs, and the
same code. But the gap lengths and inducer lengths should be the studies'
own, so that the comparison is fair: I would start with n = 6 (3 s),
g = 0, 0.5, 1, 2, 4 and 8 s, m = 4 (2 s), at 4 and 8 semitones, and change
them to match Beauvois & Meddis and Cusack et al. once the papers are read.
That needs the PDFs (§8, question 7). For the context effect, two sequences
of 6 triplets at different Δf (4 and 8, both orders, and each after itself),
separated by the silence the study used between trials.

### E3. Hypotheses across a gap

**Options.**

- **(A) The same two hypotheses**, each whistle continuing through the gap
  as one source (a long rest between its events).
- **(B) (A) plus a third hypothesis in which the sources end at the gap
  and new ones start after it**, compared by evidence like the others.
- **(C) An online search over structures**, letting the gap decide.

**Recommendation: (B).** (A) is what §3 reasons about. The third hypothesis
of (B) costs one more run and answers whether the model itself prefers to
start afresh after a long gap: if it does, that is a reset the model
explains rather than one the leak imposes. (C) is online design O1 (C) and
needs milestone (c)'s search, which is not yet run online.

### E4. Readout

**Options.**

- **(A) The leaky decision of O10**, τ 2.5 s, as accepted; τ 1 and 5 s
  beside it, as on the O1 page.
- **(B) τ fitted to Beauvois & Meddis's decay.**

**Recommendation: (A), with the window terms reported beside it.** Fitting
τ to the data it is meant to predict would make the reset test circular.
The window terms after the gap answer §3's open question on their own,
whatever τ is.

### E5. Cost

The O1 runs took about 1,000 s of computing per second of sound per
hypothesis [measured: `results/data/O1/st8_w50_p50_lag1.txt`], and each
window re-renders the whole sound so far [online-realtime.md §3].

**Options.**

- **(A) 50 ms windows through the gap**, as in O1.
- **(B) One window spanning each gap**: the window ends are already a
  parameter of `run_online`, so the silence becomes one update.
- **(C) First the speed-ups of online-realtime.md §4.**

**Recommendation: (B), run on Cho's PC.** Nothing is heard in the gap, so
nothing is lost by updating once across it; the 10 to 100 ms window sweep
showed the curve barely depends on the window length [measured: O1 page,
"Lag and window"]. A run is then about 5 s of tones: about 3 h per
hypothesis on the cloud CPU, longer for long gaps since the whole sound is
rendered at every step [estimate: from the O1 rate and online-realtime.md
§3]. Six gaps, two Δf and three hypotheses come to about 36 runs, or
four to five days of one cloud CPU; the PC can run several at once. (C) would be
cheaper per run but delays the science behind engineering.

### E6. What the model is expected to get wrong

**Recommendation: write the expected failure down before running** (§3):
contrast from the previous trial's Δf. If the runs show assimilation, as
expected, that is a result about the model: something in listeners adapts
to Δf, and the model has nothing that does. If the runs show contrast, §3
is wrong and the reason has to be found before anything else.

### E7. Compute per window

Each window's fit stops by a rule (O9: once it gains less than 1 over 50
Adam steps). That rule decides how far each hypothesis gets towards its
best fit before the next 50 ms arrive, so it changes the log odds, not only
the cost [measured: O1 page, where 30 and 150 fixed steps per window gave
log odds of 143 and 25 at 4 semitones, against 16 with the rule; O2 page,
stopping-rule sweep]. A perception model does not have to fit perfectly; listeners have
limited time and effort too, which is the idea of resource-rational models
(Lieder & Griffiths, 2020, Behav. Brain Sci. 43:e1).

**What less compute does** [measured: O2 sweep; the reason inferred]. The
two-stream fit already stops at the floor in nearly every window; the
one-stream fit needs hundreds of steps each time its whistle has to jump
to a B tone. With less compute the one-stream fit falls further behind, so
the log odds for two streams rise faster: less compute, faster buildup.

**What listeners do without attention** (Carlyon, Cusack, Foxton &
Robertson, 2001, JEP:HPP 27:115–127 [paper, read 2026-10-18]).

- *Experiment 1.* The stimulus was 21 s of ABA– triplets in the left ear:
  A 400 Hz, B 4, 6, 8 or 10 semitones up, 125 ms tones at 55 dB, a 125 ms
  silence after each triplet. Six listeners took part.
- *Two tasks.* For the first 10 s the listeners judged noise bursts in the
  right ear, then switched to judging the tones. When they switched,
  streaming was much lower than in the baseline at the same time. It was
  close to the baseline's level at the start: the comparison with the
  baseline shifted back 10 s just missed significance, p = .055 (their
  Fig. 3).
- *Ignored bursts.* The same bursts, merely ignored, changed nothing.
- *Experiment 2.* A task on the tones themselves (telling 8 from 16 Hz
  amplitude modulation for the first 10 s) left buildup intact: streaming
  had built up by the switch, as in the one-task condition (their Fig. 5).
- *What follows.* Attention to the tones, not to the streaming question,
  is what buildup needs. The authors read this as attention being needed
  for streaming itself, and suggest it could gate an evidence-accumulation
  process.

**What this means for the model** [my inference]. Less attention cannot be
less compute per window: in the model that gives *faster* buildup, the
opposite of Experiment 1. The paper fits a gate better: sound heard
without attention adds no window terms, or adds them to nothing that
carries over, so buildup starts afresh when attention returns.

In the model that is a reset of the evidence, the same kind of reset as
E3 (B)'s restart after a silent gap. It also makes this paper a test case
alongside E1's: inducer unattended, then the test, with a prediction that
can be stated before running. Its tones (400 Hz A, 125 ms) are not O1's,
so E2's stimulus choice applies here too.

**Options.**

- **(A) A numerical setting only**, chosen for cost under S6 and not
  discussed further.
- **(B) A fixed model parameter**: one rule, chosen once from the O2 sweep
  before any order-effect run, reported with every result, never fitted to
  listeners; other budgets only as a check.
- **(C) Compute as attention**: varied across conditions to model attended
  and unattended listening. Ruled out by the above: it predicts the wrong
  direction.
- **(D) Fixed compute, attention as a gate**: (B), plus unattended
  stretches of sound not integrated into the hypotheses' evidence.

**Recommendation: (B) now, (D) as the way to model attention later.** (B)
keeps the rule visible as a model choice without adding a free parameter.
(D) would be its own design, after E1's reset runs show whether the model
already resets its evidence across a gap.

## 5. What counts as success

- **Reset:** the log odds after the gap fall with the gap's length, with a
  decay described by its time constant and by whether the window terms
  reset (§3), compared with the studies' figures read by eye.
- **Context:** the sign of the previous trial's effect, for Δf and for the
  previous percept, against Snyder et al.'s.
- No claim about perception beyond those comparisons.

## 6. Code

No change to the library is expected: `run_online` takes any tones and any
window ends. A new tool in `tools/online/` builds the inducer–gap–test
sequences and the third hypothesis, and a results page (experiment O3) with
a plot, as for O1.

## 7. Order of work

1. Read the three papers (needs the PDFs), and fix E2's values.
2. Reset after silence, at 8 semitones first (the larger effect in O1), then
   4: runs, page O3, PR (O2 is now the speed-up page).
3. The context of the previous trial: runs, page O4, PR.

## 8. Questions for Cho, short form

1. **E1** Reset after silence first, then the previous trial's context?
   (recommended: yes)
2. **E2** The O1 tones, with gap and inducer lengths taken from the
   studies? (recommended: yes)
3. **E3** A third hypothesis, sources restarting after the gap?
   (recommended: yes)
4. **E4** τ 2.5 s fixed, not fitted, with the window terms reported?
   (recommended: yes)
5. **E5** One window spanning each gap, runs on your PC? (recommended: yes)
6. **E6** Record the expected failure (assimilation where listeners show
   contrast) before running? (recommended: yes)
7. Can you add the PDFs of Beauvois & Meddis (1997), Cusack et al. (2004)
   and Snyder et al. (2008) to the project? I could not open them from here.
8. **E7** Compute per window a fixed model parameter, chosen once from the
   O2 sweep and reported, with attention modelled later as a gate rather
   than less compute? (recommended: yes)
