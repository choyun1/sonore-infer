# Relative pitch: sparse harmonic spectra and the f0 floor — design

Status: **accepted by Cho on 2026-10-18** (P10 to P14 as recommended; see §4).
Written 2026-10-18. Follows experiments P1 and P2
(`results/P1-shepard-one-tone.md`, `results/P2-shepard-minor-third.md`)
and reopens decision P3 of [relative-pitch.md](relative-pitch.md), as P3
planned. Decisions continue that document's numbering (P10 to P14).
AI-assisted (Claude); code as of `1264ad4`.

**How numbers and sources are marked** (as in relative-pitch.md).

- **[paper §x]**: Cusimano, Hewitt & McDermott (2024), *Listening with
  generative models*, Cognition 253, 105874.
- **[Shepard §x]**: Shepard (1964), as in relative-pitch.md.
- **[measured]**: from a script, with its path.
- **[estimate]**: my estimate, with its basis next to it.
- **[from memory]**: a literature statement I could not check here; to be
  checked against the original before it is relied on.

## Decisions at a glance

| Decision | Accepted |
|---|---|
| P10. Spectrum prior | (B) Each harmonic present or absent, the presence marginalized, so a missing harmonic costs a few nats instead of hundreds |
| P11. Presence rate | (B) Inferred per source, with a Beta(1, 1) prior, so the model pays for sparsity only as much as the sound needs it |
| P12. f0 floor | (B) Lowered from 87.5 Hz to 30 Hz, near the lower limit of pitch, for harmonic sources and whistles alike |
| P13. Scope | (B) Options on the model, off by default; earlier results untouched until checked (P14) |
| P14. Checks | P1 and P2 again first; then the mistuned harmonic and two simultaneous notes on a few conditions, before either option becomes a default |

---

## 1. Why

P1 and P2 found, with one seed:

- The unchanged model hears the octave-spaced Shepard tone as whistles,
  or as a harmonic tone plus a whistle (the search, tone 1), and the
  minor-third control as one low tone, by about 270. Shepard's listeners
  hear the reverse: one tone and a chord [Shepard §I, §V].
- The octave it reads is the lowest component above 87.5 Hz, the floor of
  the f0 prior. Components below the floor (2 or 3 per tone, up to 53 dB)
  are explained by nothing.

Two parts of the model look responsible:

- **The harmonic spectrum prior.** A Gaussian process over harmonics in ERB
  (σ 11.8 dB, ℓ 4.7 ERB) [paper Table A.2]. It makes neighboring harmonics
  similar in level, so a tone with most harmonics missing is very
  improbable. At tone 1, the spectrum one_tone.py starts from (the
  stimulus's levels where it has a component, 30 dB below the loudest
  elsewhere) costs 726 nats more than a flat spectrum for the 155.6 Hz
  reading, 1615 for the minor-third tone's 92.5 Hz reading [measured,
  `/mnt/project-files/relative-pitch/probe/spectrum_cost_probe.txt`]. The
  fit then lowers that cost by raising the missing harmonics, which spends
  likelihood instead. These are costs at the start, not at the fitted mode.
- **The f0 floor.** The f0 and whistle-frequency means are uniform on 3 to
  33 ERB number, 87.5 Hz to 8.0 kHz [paper Table A.2]. The paper set these
  bounds by hand "to cover a natural range of values (fundamental frequency
  could span the range of frequencies audible to humans ...)" [paper App.
  A.3]. 87.5 Hz is well above the lowest audible frequencies.

Missing harmonics are not peculiar to Shepard tones: listeners hear one
pitch from complexes with harmonics left out, including the fundamental
(the missing fundamental) [from memory; textbook result, e.g. Plack, *The
Sense of Hearing*]. So a prior that tolerates missing harmonics is a
reasonable change for a model of listeners, not only a Shepard-tone fix.
Whether it makes the model hear Shepard tones as listeners do is what P14
measures; nothing here predicts it.

## 2. Decisions

### P10. Spectrum prior

- **(A) Unchanged.** Run the pair experiments as a prediction of the
  paper's model. Given P1 and P2, they would test whistles, not tones.
- **(B) Present or absent per harmonic, marginalized.** Each harmonic's
  level is the smooth GP spectrum (as now) when present, or a fixed level
  far below it (for example −40 dB, below the cochleagram's 20 dB floor
  for these stimuli [estimate, from the levels in `shepard.py`]) when
  absent. The prior of each harmonic's level is the two-part mixture, with
  the present or absent label summed out, so every latent stays continuous
  and the fitting (Adam, Laplace, variational) is unchanged. A harmonic
  that is clearly there or clearly not costs about the log of its
  probability, a few nats, instead of the GP's hundreds.
- **(C) A shorter spectrum lengthscale.** To let octave-spaced harmonics
  differ from the ones between them, the lengthscale must be shorter than
  the ERB spacing of the upper harmonics, which makes the harmonics almost
  independent. That removes the spectral smoothness the paper relies on
  for harmonic sources [paper §2.2.2, Fig. 3C, App. A.2], for every sound.
- **(D) Extra source types with fixed harmonic subsets** (octaves only,
  odd harmonics). Exact for Shepard tones, ad hoc for anything else, and
  it changes the source-type prior (3 types to 5) for every hypothesis.

**Recommendation: B.** It is the smallest change that makes a missing
harmonic cheap, it leaves the GP in charge of the harmonics that are
present, and it keeps the fitting code as it is.

### P11. Presence rate

Under P10 (B), how probable is a missing harmonic?

- **(A) Fixed**, for example 0.1. No data to set it from; the answer
  depends on it.
- **(B) Inferred per source**, as a latent with a Beta(1, 1) prior. A
  source with all harmonics present pays almost nothing; a sparse one pays
  about the log of the number of ways to choose its missing harmonics. For
  tone 1 that cost is 16 nats for the 155.6 Hz reading and 21 for the
  minor-third tone's 92.5 Hz reading, against the GP's 726 and 1615 at the
  start [measured, Beta-Binomial, same probe]. It also favors the higher
  reading over the octave below (39 Hz: 30 nats), so the lowest f0 does
  not win for free.
- **(C) Fitted to natural sounds**, as the paper fitted its other priors
  [paper App. A.3]. The datasets (URMP and others) are not in the project.

**Recommendation: B.** It adds one latent per harmonic source and no
constant to tune.

### P12. f0 floor

- **(A) 87.5 Hz** (3 ERB number), as now [paper Table A.2].
- **(B) 30 Hz** (1.15 ERB number), near the lower limit of pitch for
  harmonic complexes, about 30 Hz [from memory: Krumbholz, Patterson &
  Pressnitzer 2000, JASA 108; Pressnitzer, Patterson & Krumbholz 2001,
  JASA 109]. Shepard's components from 20 Hz up reach the cochleagram
  (lowest channel 20 Hz), so at tone 1 the components at 38.9 and 77.8 Hz
  become explainable.
- **(C) 20 Hz**, the cochleagram's lowest channel. Lets the model posit
  pitches listeners do not hear.

The floor is shared by harmonic sources and whistles (one f0 prior), so
(B) lowers both. The prior's density changes with its width: each source
costs log((33 − 1.15) / (33 − 3)) = 0.06 nats more [estimate, arithmetic],
negligible against the margins on the P1 and P2 pages. With a lower floor
the reading may still be set by the floor (the lowest component above it);
P1 measures that.

**Recommendation: B**, pending a check of the two papers.

### P13. Scope

- **(A) Model-wide defaults.** Every earlier result (A, B, C, O, R pages)
  would need rerunning before it could be cited.
- **(B) Options on the model, off by default.** Each experiment states
  which it uses. P1 to P4 use them; nothing earlier changes.
- **(C) Only in the relative-pitch tools.** Simplest, but the options are
  claims about listeners and belong in the model.

**Recommendation: B**, then decide on defaults from P14.

### P14. Checks

1. **P1 and P2 again**, all four combinations of P10 and P12 on tone 1 of
   each spacing, then the chosen one on all 10 octave tones and minor-third
   tones 1 and 4. Add the search's best on tone 1 (a harmonic tone plus a
   whistle) to the enumerated hypotheses. Cost: the first pass took 2.3 h
   for P1's 10 tones, about 1 h for P2's two tones and 30 min per search
   [measured, `results/data/P1`, `P2`]; the sparse prior adds one latent
   per harmonic, so about the same [estimate].
2. **Earlier experiments, one seed, a few conditions:** the mistuned
   harmonic at 100 Hz harmonic 1 and 200 Hz harmonic 2 (the two that
   differ from the paper), and the octave and just fifth of two
   simultaneous notes. A sparse prior makes "a tone with one harmonic
   missing, plus a whistle" cheaper, which should lower mistuning
   thresholds [estimate, reasoning from P10, not run]; listeners' are
   about 1–5% where the model's are 25% at 100 Hz harmonic 1
   ([results README, "Against the paper"](../../results/README.md#against-the-paper)).
3. **Then** the pair experiments P3 and P4 as relative-pitch.md planned.

**Recommendation: as above.**

## 3. Not in scope

- The timing likelihood (relative-pitch.md P2 (C)); Shepard tones stay a
  test case for it.
- Fitting the presence prior to natural sounds (P11 (C)).

## 4. Cho's answers

2026-10-18: all five as recommended ("Accept all" on the decision card,
after PR #74 merged). P12's two papers are still to be checked.

2026-10-24: P12's papers checked. Pressnitzer, Patterson & Krumbholz (2001,
JASA 109, 2074–2084) put the lower limit of melodic pitch at about 30 Hz for
broadband harmonic tones and whenever there is energy below 800 Hz, rising
to about 270 Hz when the energy is only above 3.2 kHz; Krumbholz et al.
(2000) is its companion. A fixed 30 Hz floor fits Shepard tones and the
other stimuli here, all with energy below 800 Hz; it would be too low for
high-passed sounds. Results of P14: [P5](../../results/P5-shepard-one-tone-sparse.md),
[P6](../../results/P6-shepard-minor-third-sparse.md),
[P7](../../results/P7-earlier-experiments-sparse.md); the options stay off
by default.
