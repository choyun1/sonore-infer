# Relative pitch: hearing threshold and spectral dominance — design

Status: **accepted by Cho on 2026-10-28** (P19 to P21 as recommended; see §4).
Follows [P9](../../results/P9-shepard-listener-noise.md), which traced
the model's too-sharp crossover on Shepard pairs to the lowest
cochleagram channels. Decisions continue the numbering of
[relative-pitch-listeners.md](relative-pitch-listeners.md) (P15 to P18).
AI-assisted (Claude); code as of `13f3638`.

**How numbers and sources are marked** (as in relative-pitch.md).

- **[Terhardt Eq. n]**: Terhardt, Stoll & Seewann (1982), *Algorithm for
  extraction of pitch and pitch salience from complex tonal signals*, JASA
  71, 679–688.
- **[Shepard Fig. 3]**: Shepard (1964), *Circularity in judgments of relative
  pitch*, JASA 36, 2346–2353.
- **[Dai]**: Dai (2000), *On the relative influence of individual
  harmonics on pitch judgment*, JASA 107, 953–959.
- **[measured]**: from a script, with its path.
- **[computed]**: a formula from a paper evaluated by hand, values shown.
- **[estimate]**: my estimate, with its basis next to it.

## Decisions at a glance

| Decision | Accepted |
|---|---|
| P19. Hearing threshold | (A) Floor each cochleagram channel at the threshold of hearing, as an option off by default |
| P20. Spectral dominance | (A) Weight each channel's likelihood by Terhardt's dominance curve, as an option off by default |
| P22. Earlier results to recheck | Proposed: (A) P7's set (mistuned harmonic 100/1 and 200/2, octave and fifth at 0 ms) and 3 ABA stimuli |
| P23. What to compare | Proposed: (A) sparse spectrum and 30 Hz floor with and without both options |
| P21. How to test | (A) The 8 pairs of P9's decomposition first, each option alone and both; the 90 pairs only for an option that changes them |

---

## 1. Why

P9 found that, from tone 1, the log odds of up against down fall from
+22.8 at 2 steps to −53.6 at 8, and that most of the fall is the
likelihood of the second tone's channels at 20 to 51 Hz [measured,
`results/data/P9/decompose/`]. The up hypothesis's f0 rises away from the
component an octave below it and predicts those channels at the 20 dB
floor; the down hypothesis covers them. From tone 6 the effect is small.

Two things listeners have that the model lacks bear on this.

- **A threshold of hearing.** It rises steeply below 100 Hz: 83 dB at
  20 Hz, 53 at 35, 39 at 51, 32 at 67 and 23 at 100 Hz [Terhardt Eq. 8,
  computed]. The model's only threshold is the cochleagram's flat 20 dB
  floor. Taking Shepard's levels as dB SPL (our stimuli use his numbers in
  the project's dB re 1e-6 RMS), the component the up hypothesis leaves
  out is 2.5 dB above threshold at 2 steps from tone 1, rising to 20.0 dB
  at 8 steps, and 1.0 to 8.4 dB below threshold for the three pairs from
  tone 6 [computed, Eq. 8 against `stimuli/shepard.py`'s levels; masking
  by the other components not included]. So a threshold alone would
  remove the effect from tone 6, where it is already small, but not from
  tone 1.
- **Spectral dominance.** Pitch depends mostly on components in a middle
  region. Terhardt et al. weight each component's contribution to pitch by
  [1 + 0.07 (f/0.7 kHz − 0.7 kHz/f)²]^(−1/2), peaking at 700 Hz [Terhardt
  Eq. 12, second factor]: 0.11 at 20 Hz, 0.27 at 51 Hz, 0.48 at 100 Hz,
  0.76 at 200 Hz, 1 at 700 Hz, 0.68 at 3 kHz [computed]. Dai finds the
  dominant harmonics are the 3 or 4 nearest 600 Hz, for f0 from 200 to
  800 Hz; at 100 Hz his weights are irregular and he did not test lower
  [Dai, Figs. 1 and 2]. Shepard's f0 readings (52 to 135 Hz) are below
  both papers' well-measured range, so applying either to them is an
  extrapolation.

Why dominance could fix the crossover [inference, not checked]: in a
Shepard tone every component moves by the same number of steps. If the
decision rests on the components near 600 Hz rather than on the lowest
ones, k steps up competes with 10 − k steps down on equal terms in log
frequency, and the crossover sits at 5 steps, where Shepard's listeners
put it [Shepard Fig. 3].

## 2. Decisions

### P19. Hearing threshold

- **(A) Floor at the threshold.** Clip both the observed and the
  predicted cochleagram from below, channel by channel, at the threshold
  of hearing [Terhardt Eq. 8] at the channel's center frequency instead of
  the flat 20 dB, converted into the cochleagram's units with the measured
  channel gains (`detect.channel_gains`; the default cochleagram reads a
  sinusoid 12.8 dB low at 104 Hz). This needs the assumption that the
  project's dB re 1e-6 RMS is dB SPL, which holds for our Shepard stimuli
  by construction but not for BASS's stimuli, whose reference is arbitrary
  (App. C).
- **(B) No threshold.**

**Recommendation: A, off by default** (a cochleagram option), so the
results of the other experiments stay as they are.

### P20. Spectral dominance

- **(A) Weighted likelihood.** Give channel c the likelihood SD
  10 dB / √w(f_c), with w Terhardt's dominance factor above, so a channel
  at 51 Hz counts about a quarter as much as one at 700 Hz. This reads
  "dominance" as the low channels being a less reliable source of
  evidence, which is our interpretation: Terhardt's weight is a pitch
  salience, not a noise level.
- **(B) Dai's region.** Weight the 3 or 4 harmonics nearest 600 Hz. It
  depends on the hypothesis's f0, so it is a weight on the source, not on
  the channel, and it is unmeasured below 200 Hz.
- **(C) A separate pitch representation.** Add a virtual-pitch stage
  (Terhardt's subharmonic coincidence, Eqs. 13 to 18) and a likelihood on
  it. Closest to the pitch literature, but a new observation model, and
  P2 (C) already reserves a timing likelihood for this role.
- **(D) None.**

**Recommendation: A, off by default.** It changes one line of the
likelihood and acts on every channel the same way for every hypothesis.

### P21. How to test

- **(A) Small first.** Rerun `tools/relative_pitch/decompose.py` on the
  same 8 pairs with P19 (A) alone, P20 (A) alone and both: 48 fits, about
  3 hours of cloud CPU 4 at a time [estimate, P9's 16 fits at one thread
  each]. If an option moves the log odds toward a crossover at 5 steps,
  rerun P3 and P8 with it (all 90 pairs; about 17 hours of cloud CPU as
  for P3 [measured, P3 page], less on Cho's PC) and refit P9.
- **(B) All 90 pairs straight away**, with the chosen option.

**Recommendation: A.**

## 3. Code

`src/sonore_infer/hearing.py` (Terhardt's Eq. 8 and dominance factor),
`Cochleagram.threshold_floor`, and a likelihood SD per channel in
`gaussian_log_likelihood`; `tools/relative_pitch/pair.py` and
`decompose.py` take `--threshold-floor` and `--dominance`. Tests in
`tests/test_hearing.py`.

## 4. Cho's answers

2026-10-28: all three as recommended ("Accept all" on the decision card).

## 5. Not in scope

- Masking between components (Terhardt Eqs. 4 to 7).
- Changing the defaults for other experiments; that is a later decision,
  like P13 for the sparse spectrum and the f0 floor.
- Moore, Glasberg & Peters (1985), Ritsma (1967) and Plomp (1967) were not
  available; their results are cited in Dai and Terhardt but not used.

## 6. Check on earlier results (P22, P23; proposed 2026-10-30)

On all 90 Shepard pairs the two options bring Group I's fit to within 12
of the curve in steps alone, from 46 [measured, P9]. Before either could
become a default (decision P13), they need checking on results they were
not designed for, as P14 did for the sparse spectrum and the 30 Hz floor.
Both options change every likelihood: the threshold floor removes the
lowest channels below about 60 Hz (a 73 dB floor at 20 Hz, 39 at 35 Hz,
25 at 51 Hz in the cochleagram's units [measured,
`Cochleagram.channel_floor_db`]), and dominance weights every channel by
w(f), so channels far from 700 Hz count less.

### P22. Which earlier results

- **(A) P7's set plus 3 ABA stimuli.** The mistuned harmonic at 100 Hz
  harmonic 1 and 200 Hz harmonic 2, 0 to 30% in steps of 5, and two
  simultaneous notes (octave and just fifth at 0 ms), as
  [P7](../../results/P7-earlier-experiments-sparse.md); and ABA streaming
  ([C2](../../results/C2-aba-streaming.md)) at 1, 3 and 6 semitones, 100
  ms, where the model goes from one stream to two. About 2.5 hours for
  P7's set [estimate, P7's 28 to 37 min per mistuning level and 5 to 17
  min per two-note hypothesis, 4 at a time] and about 3 hours for ABA
  [estimate, C2's 1 to 1.5 h per hypothesis with kernels inferred, 6 fits
  4 at a time], cloud CPU.
- **(B) P7's set only.** Leaves out streaming, where the components are
  at 1000 Hz and above and the threshold should change nothing, but
  dominance (0.81 at 2 kHz) still reweights the channels [computed].
- **(C) Everything** (A1 to C3, about two days of cloud CPU [estimate]).

**Recommendation: A.** It covers the three kinds of result the paper
reports and the conditions most sensitive to low channels (100 Hz
harmonic 1, the octave at 100 and 200 Hz).

### P23. What to compare

- **(A) With and without both options**, all runs with the sparse
  spectrum and 30 Hz floor as for Shepard; P7's own runs are the "without"
  for its set, and C2's are for ABA (C2 was run without the sparse
  spectrum, so its 3 stimuli are rerun with sparse spectra too).
- **(B) Each option alone as well**, doubling the cost.

**Recommendation: A.** Each option alone can follow where the two
together change a result.
