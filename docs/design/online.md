# Online inference — design

Status: **accepted by Cho on 2026-10-13** (O1 to O9 as recommended; see §11).
Written 2026-10-12; O9 and the later items in §8 added the same day at Cho's
request.

**How numbers are marked** (as in milestone (c)).

- **[paper §x]**: Cusimano, Hewitt & McDermott (2024), *Listening with
  generative models*, Cognition 253, 105874.
- **[Särkkä §x]**: Särkkä (2013), *Bayesian Filtering and Smoothing*,
  Cambridge University Press (the author's free first-edition PDF).
- **[measured]**: from a run already in the project, with its path.
- **[code]**: read in this repository, with file and line.
- **[estimate]**: my estimate, with its basis next to it.

---

## 1. What this is

Today every hypothesis is fitted to the whole sound at once: Adam to the
mode, then Laplace, importance sampling or variational evidence, and the
hypotheses compared by evidence. Online inference does the same job as the
sound arrives. After each short window (10–100 ms), the belief about the
scene is updated from the new frames and carried forward. Nothing has to
converge quickly. What matters is that each update uses only the new window
and a bounded summary of the past, so its cost does not grow with elapsed
time.

In the standard terms [Särkkä §1.3, §4.2]: a state-space model (how the
latents drift from step to step, and what each step's observation looks
like given them), and the filtering recursion (predict the new state from
the old belief, then update it with the new observation). The Kalman
filter, the hidden Markov model's forward algorithm and the particle filter
are special cases. A fixed-lag smoother lets the last few windows still be
revised and fixes everything older [Särkkä §10.5].

This design's goal is narrow: **measure how far our current model is from
running online, and whether an online approximation gives the same answers
as the batch fits**, on a stimulus where the time course is itself the
result. It does not try to reach real time.

## 2. Why buildup of streaming is the test

The paper already has an experiment about time: buildup of streaming
[paper §3.1.3.3, App. C.11]. Listeners hearing a repeated ABA triplet
increasingly report two streams over several seconds (Thompson et al.,
2011, cited there). The paper reproduced this offline by evaluating separate
sequences of 1 to 6 triplets (Δf 4 and 8 semitones, A at 500 Hz, 125 ms
onset-to-onset, 50 ms tones, 55 dB), comparing one stream against two by
evidence [paper App. C.11], and found the log odds rising with the number
of triplets [paper Fig. 8F].

Those separate evaluations are exactly what a perfect filter would report
at the end of each triplet. So:

- the batch curve (C.11 as the paper ran it) is the **reference**, and
  milestone (c) already plans to run it after the ABA map (design C8 (B));
- the online version should produce the same curve **in one pass** over
  the longest sequence, at a fraction of the cost.

If the two curves agree, we have an online model that reproduces a
published time-course effect. If they differ, the difference says what the
approximation loses.

## 3. Where the current code stands

| Part | Now | What online needs |
|---|---|---|
| Latent trajectories | f0/freq and level as a Gaussian process on the 10 ms grid over the **whole scene**, squared-exponential kernel plus a per-event offset (β²) and a diagonal term [code: `priors.py` `TrajectoryPrior`, lines 280–345] | the grid extended window by window; old points fixed or summarized |
| Kernel lengthscales | Table A.2 medians, or inferred (C3 (B)) | see O4: for whistles the memory is long |
| Likelihood | Gaussian on the whole cochleagram (25 ms frames every 10 ms) [code: `cochleagram.py` lines 101–102; `scene.py` `log_joint`] | the same, summed over the new frames only |
| Fit | Adam on all latents at once [code: `scene.py` `evaluate`] | a few steps per window on the free latents |
| Evidence | Laplace, importance sampling, variational, all on the whole sound [code: `evidence.py`] | an estimate that grows window by window (O5) |
| Hypotheses | written by hand from the stimulus's tones [code: `tools/compare_aba.py`, milestone (c) branch] | the same, with sources and events appearing as their tones arrive (O7) |

**Cost now.** One ABA hypothesis (3.1 s, 12 whistle tones) takes
5,704–5,774 s for one stream and 6,621–6,695 s for two on the cloud CPU,
for 300 Adam steps, Laplace, importance sampling and 2,000 variational
steps [measured: four cells at 1 semitone, 67–117 ms,
`results/aba/raw/st1_dt*.txt`, `seconds` column]. That is about 1,840 to
2,160 s of computing per second of sound for one hypothesis [measured,
divided by 3.1 s], so the batch method is three orders of magnitude from
real time before anything else is considered.

**A correction to what I said in the thread.** I suggested that freezing
old variables would cost little because the squared-exponential
correlation fades beyond a few lengthscales. For whistles that is wrong:
the median lengthscales are 2.5 s for frequency and 6.9 s for level
[code: `priors.py` lines 366–369], so the prior links every point of a
3 s stimulus to every other. Only the harmonic level's 0.18 s is short.
Freezing is still correct (O3, O4), but the past is not forgotten; it has
to be conditioned on.

## 4. Decisions

### O1. Scope

**Options.**

- **(A) Batch reference only.** Run C.11 as the paper did; no online code.
- **(B) Reference, then an online approximation for the same fixed
  hypotheses**, compared against the reference at every triplet.
- **(C) (B) plus an online search over structures**: particles over which
  source each new tone joins, or whether it starts one (the paper's
  source-construction moves [paper App. B.2.2] made online, like the data
  association in multiple-target tracking [Särkkä §1.1, Example 7.2,
  §7.5]).

**Recommendation: (B).** (A) is already planned in milestone (c), so it
costs nothing extra here. (B) is the smallest step that answers the
question. (C) depends on milestone (c)'s search (C9), which does not exist
yet, and should be designed after (B) shows what an online update costs.

### O2. The reference

**Options.**

- **(A) C.11 exactly**: 12 separate stimuli (2 Δf × 1–6 triplets), each
  evaluated in full.
- **(B) Prefixes of one stimulus**: the 6-triplet sequence cut at each
  triplet boundary (or more finely), each prefix evaluated in full.

**Recommendation: (A).** It is the paper's experiment, so our reference
can be compared with Fig. 8F, and it is the run milestone (c) already
plans. (B) differs from (A) only in the silence after the last tone, and
can be added at a few finer points later if the online curve needs
checking between triplets.

### O3. What is updated in each window

**Options.**

- **(A) Warm start, refit everything**: add the new window's grid points
  and run a few Adam steps on all latents. Simple, but each update costs
  more as the sound gets longer.
- **(B) Fixed lag**: only the latents in the last L seconds (grid
  deviations there, and events overlapping that span) are free; older ones
  are fixed at their current values, and their prior terms are kept as a
  condition on the free ones. Each update's cost is bounded apart from the
  conditioning in O4.
- **(C) A filter proper**: a Kalman or Rao–Blackwellized particle filter
  [Särkkä §4.3, §7.5], which needs Markov dynamics (O4 (C)).

**Recommendation: (B), with L swept over 0.5, 1 and 2 s, and (A) run once
as a check.** (B) is the fixed-lag smoother in the form closest to our
code: the same `log_joint`, the same Adam, with some latents held fixed.
The sweep shows how much revision of the recent past the answer needs. Two
latents are never frozen: the GP kernel parameters when inferred (C3 (B)),
because they are shared by the whole sound; in filtering terms they are
static parameters, estimated alongside the state [Särkkä §12.1].

### O4. The kernel

**Options.**

- **(A) Keep the squared-exponential kernel and condition exactly on the
  frozen past.** The conditional of the new grid points given the fixed old
  ones is Gaussian; with an incrementally extended Cholesky factor, each
  update costs about (points so far) × (new points).
- **(B) Keep the kernel but truncate its memory** to the last M seconds.
  An approximation, and a poor one for whistles given §3's correction.
- **(C) Switch to a kernel with a state-space form** (Matérn family; the
  Ornstein–Uhlenbeck kernel is AR(1) on the grid), so the past is summarized
  by a small state vector and each update costs the same forever
  (Hartikainen & Särkkä, 2010, cited in [Särkkä §1.2]). This changes the
  model, and its priors would no longer be the paper's Table A.2.

**Recommendation: (A) for this design.** It keeps the model we have
reproduced, so any difference from the reference comes from the
approximation, not from a new prior. At ABA lengths the cost is small
[estimate: the whole stimulus is 311 grid points per trajectory, and each
source has two trajectories]. (C) is what real time on minutes of sound
would need, and should be its own decision once (A) has numbers.

### O5. Evidence over time

**Options.**

- **(A) Recompute** the evidence (Laplace, then importance sampling) on the
  whole prefix at each checkpoint. Exact in the sense the batch method is,
  but its cost grows with time.
- **(B) Accumulate it**: the evidence of a prefix is the product of each
  window's predictive likelihood given everything before,
  log p(x₁:ₜ) = Σₖ log p(xₖ | x₁:ₖ₋₁) [Särkkä §12.1, Thm 12.1]. Each term
  is estimated in its window from the free latents only (a Laplace
  estimate over the lag's latents with the frozen ones as a condition), so
  the cost per window is bounded.

**Recommendation: (B) for the online run, (A) at the triplet boundaries of
one stimulus to measure (B)'s error.** (B) is what a filter gives for free;
whether our approximate version of it tracks (A) is one of the results.
The exact form of each window's term is an implementation detail to settle
with a test on a toy model where the answer is known, as was done for C6.

### O6. Window length and steps per window

The window is how often the belief is updated; frames are 25 ms every
10 ms [code: `cochleagram.py` lines 101–102], so a frame is usable once its
25 ms of sound has arrived, a fixed 25 ms latency.

**Options.** (A) 10 ms (every frame); (B) 50 ms; (C) 100 ms.

**Recommendation: 50 ms as the default, with 10 and 100 ms on one stimulus
to see whether the answer depends on it.** 50 ms sits inside the 10–100 ms
range you suggested and gives each 125 ms tone slot two to three updates.
The number of Adam steps per window is the other knob: fix it on one
stimulus as the smallest number for which the online curve stays within
the reference's seed spread (SD 2–5 in (b), results/two_notes/seeds/) at
the triplet boundaries, then report the time per window. That time divided
by the window length is the measured real-time factor.

### O7. Hypotheses as the sound arrives

The two hypotheses stay the paper's (one stream; A tones in one source and
B tones in another), started from the true tone frequencies and timings as
in C.10 and C.11. Online, each event enters the scene when its tone's onset
arrives. The two-stream hypothesis's second source is born with the first B
tone; before that, both hypotheses are the same scene and the log odds is
0 by construction. A birth changes the structure prior (one more source and
event), which enters the evidence at that window, as the duration-scaled
p(n) does (design D8).

This keeps detection out of scope: the model is told when tones start, as
in every enumerative run so far.

### O8. Code

**Recommendation:** a new module `src/sonore_infer/online.py` that
holds the fixed-lag loop over an existing `Scene` (which latents are free,
the conditioning of O4 (A), the evidence of O5 (B)), and a tool
`tools/online_buildup.py` that runs it on the C.11 stimuli. Existing
modules change only where a latent vector has to be split into fixed and
free parts (each trajectory's deviations are now one vector over the whole
grid [code: `scene.py` `_trajectory`, line 273]). Pyro's `SMCFilter` is not
used: it expects Markov `init`/`step` models, which our kernel is not.

### O9. Spending effort where the sound is surprising

Listeners and animals work online, and one thing that buys them is that
most of the time the next moment is predicted by the current scene. The
predictive term of O5 (B), log p(xₖ | x₁:ₖ₋₁), measures exactly that: how
surprising the new window is under the current belief. This is the
predictive-processing view of auditory scene analysis (Winkler, Denham and
colleagues; reviewed in Szabó, Denham & Winkler, 2016, Front. Neurosci.).

**Options.**

- **(A) A fixed number of Adam steps per window** (O6).
- **(B) Steps scaled by surprise**: a few steps when the window is well
  predicted (its per-frame predictive log likelihood within some margin of
  its running average), the full number when it is not, for example at a
  tone onset.

**Recommendation: (A) as the default, (B) run on one stimulus beside it.**
(B) costs little once O5 (B) exists, and if it matches (A)'s curve with
fewer steps, it is a measured saving that carries over to real time. The
margin is set on that one stimulus and reported with the result.

### O10. A decision with finite memory (added and accepted 2026-10-14)

O5 (B) keeps the full sum of the window terms, so the log odds remember
everything since the start, and with each triplet adding a similar amount
they keep rising: a lag (O3) limits what the scene can still revise, not
what the decision counts. Listeners' buildup levels off within seconds, and
Cho expected the decision to have a finite memory, like the integration
window of texture averaging (about 2.4 to 2.8 s; McWalter & McDermott,
2018, 2019).

**Options.**

- **(A) Full sum only** (O5 as accepted).
- **(B) A leaky decision variable beside the full sum**:
  d(t) = Σₖ e^(−(t − tₖ)/τ) [termₖ(two) − termₖ(one)], plus the structure
  prior's log odds at t, where termₖ is window k's term of O5 (B) and tₖ
  its end. It levels off at about (gain per second) × τ. It is no longer
  any hypothesis's evidence; it is a decision model on top of the
  evidence, a leaky accumulator.
- **(C) Forget inside the inference**: the old sound's likelihood
  down-weighted (a tempered or discounted likelihood), so the scene and the
  evidence both forget. A change to the model, and to every term.

**Recommendation: (B), with τ = 2.5 s by default and 1 and 5 s beside it.**
It needs no new inference (the terms are already logged), keeps the exact
evidence available, and makes the memory a parameter that can be compared
with listeners. (C) should wait until (B) shows whether a finite memory
helps.

## 5. What counts as success

- **Agreement:** at every triplet boundary, the online log odds lies within
  the reference's seed spread, for both Δf.
- **Cost:** time per window and the real-time factor, measured on the
  laptop and the cloud CPU, with the lag L and steps per window used.
- **Comparison with the paper and listeners:** the online curve against
  [paper Fig. 8F] read by eye and, with care, against Thompson et al.'s
  listeners as the paper plotted them.

No claim about perception beyond those comparisons. In particular, an
online curve that rises over seconds is what any accumulation of evidence
does; whether its shape matches listeners is a separate question.

## 6. Size (estimates)

- **Reference:** 12 stimuli totalling 2 × (0.5 + 1 + … + 3) = 21 s of
  sound, two hypotheses each, at about 2,000 s of computing per second of
  sound per hypothesis [measured rate from §3] ≈ 84,000 s ≈ 23 h on one
  cloud CPU [estimate: linear in duration; the 500 Hz, 55 dB tones should
  cost the same as the 1 kHz ones]. Less in parallel on Cho's PC.
- **Online:** unknown, which is the point. A first guess: with 50 ms
  windows and 20–50 Adam steps each, a 3 s sequence is 60 windows and
  1,200–3,000 steps, about the 300 Adam plus 2,000 variational steps of one
  batch fit [estimate: step counts only; per-step cost should be lower
  because fewer latents are free].

## 7. Order of work

1. The reference: milestone (c)'s C.11 run, after the ABA map now running
   on the PC (no new code).
2. `online.py` on one stimulus (8 semitones, 6 triplets), with the toy-model
   test of O5 (B); compare with the reference at every triplet (PR).
3. Sweeps of L, window and steps; both Δf; results page and step log (PR).
4. A short note: what it would take to reach real time (O4 (C), O1 (C)),
   with the numbers from steps 2–3, as the input to the next design.

Each step is one PR; nothing past step 1 starts before Cho accepts this.

## 8. Later, out of scope here

- **A texture source** whose latent state is a vector of McDermott &
  Simoncelli statistics, with a likelihood on those statistics. The paper
  names this gap [paper §4.3.2], McWalter & McDermott (2018, 2019) give its
  time course (averaging over about 2.4–2.8 s, selective, with completion
  behind maskers), and sonore already computes the statistics and their
  gradients. A natural second test of an online model.
- **Structure search online** (O1 (C)), with structure moves (a new
  source, a new event, a tone joining a source) tried **only at detected
  onsets**. Sources and events begin when something starts, and common
  onset is among the strongest grouping cues (Bregman, 1990, from memory),
  so the search over structures need not run in every window. O7 already
  does this with given onsets; the search would detect them.
- **Real time** on long input (O4 (C)).
- **Order effects as tests.** A batch model gives the same answer whatever
  order the sound arrived in; an online one need not, which is where it
  can say something the batch model cannot. Candidates, each to be checked
  against its literature before it is run:
  - buildup of streaming (this design's test);
  - buildup resetting after a pause or a shift of attention (Cusack et
    al., 2004, from memory);
  - hysteresis: staying with an interpretation once chosen as Δf or Δt is
    swept up and then down;
  - switching between interpretations in long bistable sequences (Mill et
    al., 2013, PLoS Comput. Biol. 9:e1002925).
- **Attention as a budget**: refine one source in detail and keep the
  others coarse, as the paper suggests for attention [paper §4.3.7].
- **Proposals learned without labels.** Predicting the next window is a
  training signal that needs no labeled data: a proposal network could
  learn from how sounds actually continue, rather than from 100,000
  sampled scenes [paper App. B.2.1]. That would remove the reason given in
  milestone (c)'s decision C1 for leaving a learned network out, and bears on the
  long-term goal of separating parts in real recordings without labels.

## 9. Roadmap: a phone app

The eventual goal (Cho, 2026-10-12) is a phone app that runs live
inference on the phone's microphone: the scene hypotheses update as the
sound arrives and are shown as they change. This design is the first step
towards it. The steps after it, each its own design:

1. **Real time on a computer**: the update for each window finishes within
   the window, on long input. Needs O4 (C) (a kernel with a state-space
   form) and O1 (C) (structure search online), and a measured real-time
   factor below 1 on the laptop's CPU.
2. **Live input**: the same loop fed from a microphone instead of a file,
   with the cochleagram computed frame by frame as samples arrive.
3. **On the phone**: the model and the update loop on a phone's processor,
   or the phone streaming audio to a computer that runs them. Which one is
   open; it depends on the cost measured in step 1.

What a phone adds that the lab does not: an uncalibrated microphone (so the
level in dB SPL that the priors assume is unknown), room reverberation and
background noise in every recording (the source-and-room work in
`docs/design/source-reverb.md` is related), and limited power and memory.
None of these is addressed here.

## 10. Questions for Cho, short form

1. **O1** Reference plus an online approximation for the same hypotheses;
   online structure search later? (recommended: yes)
2. **O2** The reference is C.11 as the paper ran it? (recommended: yes)
3. **O3** Fixed lag, L swept over 0.5–2 s, kernel parameters never
   frozen? (recommended: yes)
4. **O4** Keep the squared-exponential kernel and condition exactly on the
   past; Markov kernels as a later decision? (recommended: yes)
5. **O5** Accumulate the evidence window by window, checked against full
   recomputation? (recommended: yes)
6. **O6** 50 ms windows by default, 10 and 100 ms checked; steps per window
   set by agreement with the reference? (recommended: yes)
7. **O7** The paper's two hypotheses, events entering at their onsets, the
   B source born with the first B tone? (recommended: yes)
8. **O8** New `online.py` and `tools/online_buildup.py`, existing modules
   changed only to split latents? (recommended: yes)
9. **O9** Fixed steps per window by default, with surprise-scaled steps
   run beside it on one stimulus? (recommended: yes)
10. **O10** A leaky decision variable (τ 2.5 s, also 1 and 5 s) reported
    beside the full sum? (recommended: yes; added and accepted
    2026-10-14)

## 11. Decisions as accepted (2026-10-13)

Cho accepted O1 to O9 as recommended, and the phone-app roadmap (§9) as the
long-term goal. On 2026-10-14 Cho accepted O10 as recommended (leaky
decision variable, τ 2.5 s, with 1 and 5 s beside it).
