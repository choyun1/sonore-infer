# Relative pitch from Shepard tones — design

Status: **draft, not accepted.** Written 2026-10-15; P8's cost updated
2026-10-18 for the speed-ups (S1). AI-assisted (Claude),
from the sonore project's handoff (HANDOFF-relative-pitch.md, 2026-10-14),
Shepard (1964), a feasibility note (§2), and the code as of `151018e` (P8 as of `0798ab8`).
No library code is written until Cho accepts the decisions below.

**How numbers are marked.**

- **[Shepard §x]**: Shepard, R. N. (1964). Circularity in judgments of
  relative pitch. *J. Acoust. Soc. Am.* 36(12), 2346–2353, at the section,
  equation or figure given. Read from a scan; figure values are read by eye.
- **[measured]**: from a script, with its path.
- **[estimate]**: my estimate, with its basis next to it.

Decision numbers use the letter P (pitch). The letters D, B, C, R, O, S and E
belong to other documents.

## Decisions at a glance

Recommendations, for Cho to accept or change. The options and reasons are
in §4. P1 and P2 were revised and P9 added on 2026-10-18, after Cho's
questions about intervals, timing and listener experience.

| Decision | Recommended |
|---|---|
| P1. What is inferred | The interval between the tones, in log frequency, as the latent (D); the paper's f0 per tone (B) run beside it |
| P2. Observation | The cochleagram as it is for now; Shepard tones become a test case for the planned timing likelihood |
| P3. Spectrum prior | Unchanged for the first experiments; decide whether to change it only from what experiment P1 shows |
| P4. Hypotheses | Enumerated: one harmonic source per octave reading, and a set of whistles; the search (C3) on a few stimuli only, as a check |
| P5. Response | The model's output is P(up); no lapse or noise model until it is compared with listeners |
| P6. Stimuli | Shepard's Eqs. 1–3 and his parameters, without the noise bursts |
| P7. Experiments | P1 one tone, P2 octave against minor-third spacing, P3 all 90 pairs, P4 the half octave by pitch class, as a prediction |
| P8. Scale, seeds | One seed and one starting tone first, then all pairs, then 10 seeds |
| P9. Listener experience | One listener first; experience as fitted noise per Shepard group when comparing with Fig. 3; learned interval categories later |

---

## 1. The question

Given two successive tones, did the pitch go up or down? Shepard built tones
of octave-spaced components under a fixed spectral envelope, so each tone
has a pitch class but no clear octave. Moving all components up a step
around the circle is heard as going up; after 10 or 12 steps the stimulus is
back where it started [Shepard §II]. Pairs of such tones are judged by
proximity: the second tone is heard higher when it is less than half an
octave clockwise, lower when more, and pairs half an octave apart are
ambiguous [Shepard §IV, Fig. 3].

This is a listening experiment with listener errors and a clear stimulus
specification, so it suits the project's goal: a model of human auditory
perception. The sonore project makes the stimuli; the inference and
anything about listeners belongs here (handoff §1).

## 2. What the feasibility note found

From `/mnt/project-files/relative-pitch/feasibility.md` (2026-10-14), which
evaluated the existing priors in numpy and fitted nothing
(`relative-pitch/probe/shepard_prior_probe.py`):

- **Proximity is already in the prior.** Two events of one harmonic source
  0.25 s apart share the f0 Gaussian process (σ 5.9 ERB, ℓ 2.5 s, β 0.437),
  which lets f0 change by about 0.87 ERB (SD, one grid point per event)
  [measured]. An octave at 155.6 Hz spans 3.1 ERB. The prior's log odds of
  "up" over "down" for k tenths of an octave, from a first tone at 155.6 Hz,
  are +2.4 at k = 1, +1.2 at 3, −0.5 at 5, −2.5 at 7, −5.1 at 9 [measured].
  The crossover sits just below k = 5, so at the half octave the prior leans
  slightly "down" at every pitch class. That comes from the ERB scale, not
  from listeners.
- **The spectrum prior works against hearing one tone.** The harmonic
  spectrum is a Gaussian process over the ERB numbers of the harmonics
  (σ 11.8 dB, ℓ 4.7 ERB). Read as one harmonic tone, an octave complex has
  to leave out every non-octave harmonic. Pushing them all 10 dB below their
  conditional mean costs about 590 nats at f0 155.6 Hz and 100 to 1300 nats
  over f0 622 Hz to 19.5 Hz [measured; a rough proxy, prior only]. So the
  unchanged model probably hears a Shepard tone as several whistles, a
  chord, where Shepard's listeners heard one organ-like tone [Shepard §I].

The search (decision C9, `detect.py`) already offers a harmonic candidate's
octave above and below, because octave errors are local minima of a fit.

## 3. What Shepard (1964) specifies

- **Tones [Shepard Eqs. 1–3].** Component c (1..c_max) of tone t
  (1..t_max) is at `f = f_min · 2^(((c−1)·t_max + t − 1)/t_max)`, with level
  `L = L_min + (L_max − L_min)(1 − cos θ)/2` dB, θ = 2π((c−1)·t_max + t − 1)/(t_max·c_max).
  The envelope is a raised cosine in dB over log frequency, not the
  handoff's Gaussian bell.
- **Experiment parameters [Shepard §III].** t_max = 10 (steps of 1.2
  semitones), c_max = 10, f_min = 4.863 Hz, L_min 22 dB, L_max 56 dB,
  about 66 dB overall; envelope peak at 155.6 Hz; 10 kHz sampling. Tones
  1/8 s, 1/8 s apart; a soft noise burst (54 dB, 1/8 s) 1/8 s before and
  after each pair; 3 s to answer.
- **Listeners [Shepard §IV, Fig. 3].** 50, in three groups by accuracy on
  pure-tone pairs: Group I (26) near 100% "higher" at 1–2 steps clockwise,
  near 0% at 8–9, about 50% at 5; Group III (14) about 68% correct at 1
  step, about 80% at 3, and a non-monotonic curve. Values only in the
  figure.
- **Qualitative findings [Shepard §I, §V].** Octave-spaced components
  sound like one tone; the same envelope with components a minor third
  apart sounds like a chord. At 5 steps listeners disagree, but each hears
  the pair clearly one way. In the cyclic scale, without the 0.84 s
  silences the sequence is heard to drop at one point, which depends partly
  on the starting tone.

The pitch-class dependence of the half-octave ambiguity (the tritone
paradox) is Deutsch (1986), cited in the handoff from memory. It is not on
file and its data are not in hand.

## 4. Decisions

### P1. What is inferred (handoff D1)

- **(A) The direction only.** A latent sign per pair. Simple, but nothing
  in the generative model makes a sign; it would be a model of the task,
  not of the sound.
- **(B) f0 per tone, with its octave reading.** Each tone is an event of a
  harmonic source; the octave reading of each tone is a discrete choice;
  "up" is the posterior probability that the second event's f0 is higher,
  summed over readings. It uses the existing source and the proximity the
  f0 prior already gives.
- **(C) Pitch class and height as separate latents.** Closer to the helix
  [Shepard Introduction], but a new source type with no counterpart in the
  paper's grammar.
- **(D) The interval as the latent.** The first tone's f0 is a nuisance,
  integrated out under its wide prior; the second tone is the first plus an
  interval Δ in log frequency (semitones), with a zero-mean prior on Δ, and
  the octave reading becomes a choice of Δ. "Up" is P(Δ > 0), and the
  posterior over Δ is the model's heard interval.

What separates B and D is the scale of the prior on the change. In B the
change is penalized in ERB number, so the same musical interval costs more
at high f0 than at low f0: proximity is weak near 100 Hz and the half
octave leans "down" (§2). In D the prior is the same at every register and
symmetric at the half octave, so any lean there comes from the sound.
Shepard writes that, except for listeners with absolute pitch, tonality
"seems to be a purely relative property" [Shepard §V], which fits D; it is
his reading, not a measurement. D's prior width is a free parameter: the
f0 prior of B implies about 4 semitones (SD) at the envelope peak, 155.6 Hz
[estimate: 0.87 ERB at 2.6 ERB per octave there], which can be the
starting value.

D departs from the paper's grammar, where f0 follows one Gaussian process
within a source; it needs a pair model or an f0 prior in log frequency for
these runs. B costs nothing extra, since it is the existing source.

**Recommendation: D, with B run beside it.** D is the quantity the task
asks about and is register-free; B shows what the paper's model, unchanged,
predicts, and their differences (register, the half-octave lean) are
predictions that set them apart. If the model hears a chord (P3), the
interval is the shared shift of the whistles' frequencies, and P5 says how
that becomes "up".

### P2. Observation (handoff D2)

- **(A) The cochleagram as it is.** FFT, BASS gain, 64 channels from 20 Hz
  to 9.4 kHz, 20 dB floor. It keeps the spread over octaves that chroma
  drops. Components below 20 Hz (4.9–19.5 Hz at t = 1, 22–34 dB) fall
  outside it.
- **(B) Chroma or the semitone filter bank** (sonore PR #214). Chroma cannot
  tell up from down by construction (handoff §3), and a new observation
  would take every earlier result out of comparison.

- **(C) A representation that keeps timing.** The cochleagram keeps
  magnitude only, in 25 ms windows every 10 ms: no phase, no temporal fine
  structure, and envelope modulation above a few tens of Hz smoothed away
  (results/README.md, "Planned", the first entry). Pitch reaches it only by
  place, from resolved components. A per-channel autocorrelation
  (correlogram) or envelope modulation channels would add periodicity. That
  is a likelihood change of its own, planned but not yet designed.

For Shepard tones, place is enough to locate each component: they are an
octave apart, far wider than one channel, so every component is resolved.
What timing could change is the octave reading and the grouping. A
correlogram shows a common period across channels (here the lowest
component's), and a shared period is a standard cue for hearing components
as one tone. That bears on P3 (one tone or a chord) and on which octave the
model hears. These are expectations, not checked.

**Recommendation: A for these experiments, and Shepard tones as a test case
for C when the timing likelihood is designed.** A keeps these runs
comparable with every earlier result, and the first question (P3) is about
the spectrum prior, which a new likelihood would not remove.

### P3. Spectrum prior (handoff D3)

- **(A) Unchanged, as a prediction.** Report how the model hears one
  Shepard tone (experiment P1) and whether it tells the octave complex from
  the minor-third complex (experiment P2).
- **(B) Allow sparse harmonic spectra.** For example a presence latent per
  harmonic (spike and slab), or a mixture over harmonic subsets. A
  departure from the paper and BASS, and Shepard's tones are the only
  evidence for it in hand.
- **(C) Change the kernel.** A shorter lengthscale would let neighboring
  harmonics differ more; it changes every earlier result.

**Recommendation: A first.** Decide between B and C, if at all, as a new
decision after experiments P1 and P2, with their numbers in hand.

### P4. Hypotheses

Per pair, enumerated:

1. One harmonic source, two events, with the second tone's f0 started at
   the nearer reading above and, separately, the nearer reading below the
   first (two fits). The first tone's f0 starts at its lowest component
   above the cochleagram floor; other readings of it are offered when
   experiment P1 shows they compete.
2. One whistle source per component that experiment P1 finds, each with two
   events, so each component is matched to its neighbor in the next tone.

The search (C3) took 20–74 min per (b) stimulus and 6–9 h per ABA sequence
[measured, `results/C3-search.md`], so it runs on a few pairs only, to check
that the enumerated set is not missing a better structure.

**Recommendation: as above.**

### P5. Response (handoff D5's readout)

- **(A) P(up) is the output.** For the harmonic hypotheses, the posterior
  mass on readings where the second f0 is higher; for the whistle set, the
  sign of the shared shift of the components. Weighted by each hypothesis's
  variational evidence, as in B2.
- **(B) A response model on top.** Argmax plus lapses, or decision noise,
  to reproduce Group III's errors. It has free parameters that only listener
  data can set.

**Recommendation: A.** Add B only when comparing with Fig. 3 (experiment P3).

### P6. Stimuli

A generator in `src/sonore_infer/stimuli/shepard.py`, built with sonore
0.6.0's `harmonic_complex` (it takes `harmonics` and per-harmonic
`amplitudes`), at 20 kHz.

- Shepard's Eqs. 1–3 and §III parameters, components up to 5 kHz as his
  10 kHz sampling allowed.
- Levels: his dB SPL read as our dB re 1e-6 RMS, with L_max 56 dB. An
  assumption, as for every stimulus in the project (`stimuli/levels.py`).
- Tones 1/8 s with 10 ms raised-cosine ramps (his attack and decay are
  illegible in the scan; 10 ms is our usual value), 1/8 s apart, 50 ms of
  silence before and after.
- **No noise bursts.** They marked trials for listeners; modeling them
  needs a noise source in each hypothesis. A labeled departure.
- The minor-third version for experiment P2: 40 components, 4 per octave,
  the same envelope [Shepard §V].

**Recommendation: as above.**

### P7. Experiments

| ID | Asks | Stimuli | Compared with |
|---|---|---|---|
| P1 | How does the model hear one Shepard tone: one harmonic source (which octave), whistles, or a mix? | 10 tones, alone | Shepard §I (one tone, not a chord) |
| P2 | Does it hear the octave complex as one tone and the minor-third complex as a chord? | One tone of each | Shepard §V |
| P3 | P(up) against steps clockwise | All 90 ordered pairs | Fig. 3, read by eye (estimates) |
| P4 | P(up) at 5 steps, by starting tone | The 10 half-octave pairs | No data; a prediction |

The cyclic scale with and without silences is left for the online and
order-effects work.

**Recommendation: P1 and P2 first; P3 and P4 only after Cho has seen them,
since P3 (the spectrum prior) may change.**

### P8. Scale and seeds

One seed throughout at first. Experiment P3 starts with one starting tone
(9 pairs), then all 90, then 10 seeds as in A8 and B4. Cost before S1: about
4 min per hypothesis for a 0.5 s stimulus on one thread [estimate, from C3's
control: 5 hypotheses in 20 min], so about 7 min for a 0.875 s pair
[estimate, scaled by length]. S1 (each event rendered over its own span,
`docs/design/online-speed.md`) made an Adam step 2.8 times and the Hessian
2.3 times faster at 1.05 s of sound [measured: `results/O2-online-speed.md`,
one hypothesis, 8 semitones]. Applied to the pairs, that is about 2.5 to
3 min per hypothesis, and 90 pairs × about 4 hypotheses about 15 to 18
thread-hours [estimate, the O2 ratios applied to the C3 figure]. Experiment
P1 times its first fits, which replaces this estimate with a measurement.

**Recommendation: as above.**

### P9. Listener experience

Relative-pitch judgments depend on the listener. Shepard's 50 listeners
ranged from no errors to chance on pure-tone pairs, and he grouped them by
that accuracy, not by training [Shepard §IV]. He adds that listeners who
were interested in music or played an instrument generally made few, if
any, errors, without numbers [Shepard §IV; part of the line is obscured in
the scan]. His own suggestion for the less accurate listeners is a larger
difference limen [Shepard §V].

- **(A) One listener.** The model with its priors as set, no experience
  term. It predicts one curve, to be compared with Group I.
- **(B) Experience as noise.** A per-group parameter, such as the
  likelihood SD, the width of the interval prior, or decision noise (P5's
  option B), fitted to Group I and Group III. It follows Shepard's
  difference-limen idea.
- **(C) Experience as learned intervals.** For trained listeners, a prior
  on Δ peaked at whole semitones (a learned scale). Shepard's steps are 1.2
  semitones, off that grid, so C predicts something testable, but no data
  in hand split listeners by training.

**Recommendation: A first, B when comparing with Fig. 3, C only with data
that separate musicians.**

## 5. Not in scope

Absolute pitch, and the Risset glide. Any claim that the model matches
listeners waits for data; Shepard's Fig. 3 is the only listener data in
hand.

## 6. Cho's answers

Not yet.
