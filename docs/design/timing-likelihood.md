# Timing in the likelihood — design

Status: **draft for Cho, 2026-11-05.** Nothing is built or run beyond the
probe in §3; no decision is taken until Cho answers. Follows the "Timing in
the likelihood" entry under [Planned](../../results/README.md#planned)
(Cho, 2026-10-18) and decision P2 (C) of [relative-pitch.md](relative-pitch.md),
which reserved a timing likelihood and named Shepard tones as a test case
for it. Decisions are numbered T1 to T7. AI-assisted (Claude); code as of
`9ba1a5c`.

**How numbers and sources are marked** (as in relative-pitch.md).

- **[paper §x]**: Cusimano, Hewitt & McDermott (2024), *Listening with
  generative models*, Cognition 253, 105874.
- **[measured]**: from a script, with its path.
- **[estimate]**: my estimate, with its basis next to it.
- **[from memory]**: a literature statement I could not check here; to be
  checked against the original before it is relied on.

## Decisions at a glance

| Decision | Recommended |
|---|---|
| T1. First target | (A) The mistuned harmonic (all 9 conditions) and P06's minor-third complex: both existing stimuli where the model hears one tone and listeners hear more |
| T2. Representation | (A) A correlogram: per channel, the short-time autocorrelation of the half-wave rectified gammatone output, lags 0 to 35 ms |
| T3. A control for fine frequency | (A) Yes: the same tests with the correlogram computed without rectification, which by Wiener–Khinchin holds fine frequency but no timing, so a gain can be put down to one or the other |
| T4. Base cochleagram | (A) The gammatone cochleagram (A01), not BASS's FFT one; both the baseline and the new runs on it |
| T5. Noise and weight of the new term | (A) A second, independent Gaussian term on a reduced correlogram, with its own SD fitted to one listener threshold (200 Hz, harmonic 1) and the other conditions held out |
| T6. Phase of rendered components | (A) Keep phases fixed, not inferred, with the renderer's start phase set to the stimulus's (cosine for the mistuned harmonic, sine for Shepard tones) |
| T7. Scope and fitting | (A) An option, off by default; fit with the cochleagram first, then add the correlogram term from that mode |

---

## 1. What the cochleagram drops

The likelihood compares cochleagrams: 64 half-ERB channels, magnitude
only, pooled over 25 ms Hann windows every 10 ms, in dB, floored at 20 dB,
under independent Gaussian noise of 10 dB per cell
([methods §1.6, §1.7](../methods.md)). The renderer makes a waveform, so
everything is in the sound the model predicts; it is the comparison that
discards it. Three things go:

1. **Fine frequency.** Within one 25 ms window, frequency is known only to
   the width of the window's spectrum (a Hann window's main lobe is
   4 / 25 ms = 160 Hz wide), sharpened by the filter shapes. BASS's FFT
   cochleagram is coarser still below 600 Hz, where its channels are 2 to
   3.6 times wider than half-ERB filters
   ([A02](../../results/A02-fft-cochleagram.md)).
2. **Fast envelope modulation.** Squared envelopes are averaged over 25 ms,
   so modulation above a few tens of Hz is smoothed away. In channels where
   harmonics are unresolved, the envelope repeats at f0 (100 Hz and up for
   our stimuli), so the periodicity of unresolved harmonics is invisible.
   The paper lowered a modulation cutoff from 50 to 10 Hz for this reason
   [paper App. C.4].
3. **Phase**, within and across channels, and so interaural time
   differences. The model is monaural, so the last is moot for now.

One point shapes the decisions below. **The autocorrelation of a linear
filter's output is the inverse Fourier transform of its power spectrum**
(Wiener–Khinchin). A correlogram computed straight on the gammatone
outputs would therefore hold the same information as a power spectrum of
the same integration length: it would add fine frequency (item 1) only
because its window and lags are longer, and nothing of items 2 or 3. What
makes a correlogram a timing representation is the nonlinearity before it:
half-wave rectification (a crude inner hair cell) turns the envelope of
unresolved harmonics into a component at f0, so the autocorrelation peaks
at 1/f0 there, and keeps the fine structure of resolved components below
the phase-locking limit [from memory: Meddis & Hewitt (1991) and Slaney &
Lyon (1990), the standard correlogram pitch models]. So any gain from a
correlogram has two possible sources, finer frequency from its long lags
and the rectification, and only the second is "timing". T3 separates them.

## 2. Which listener mismatches it could bear on

| Mismatch | Now | Could timing help? |
|---|---|---|
| Mistuned harmonic, all 9 conditions (Fig. 7D) | Model thresholds at or above listeners' (about 1–5%, by eye): 2.5–15% on the gammatone cochleagram ([A01](../../results/A01-gammatone-baseline.md)), up to 25–30% on the FFT one ([A08](../../results/A08-ten-seeds.md)); they rise with harmonic number faster than listeners' | **Yes, the clearest case.** Listeners' detection of a mistuned low harmonic is usually explained by its fine structure or by the beats it makes with its neighbors [from memory: Moore, Glasberg & Peters (1986); Hartmann, McAdams & Smith (1990)] |
| The 100 Hz harmonic 1 gap | 25% Eqn 2 / 30% BASS's rule on the FFT cochleagram, against about 45% in the paper and 1–5% for listeners | **Mostly not a timing question.** On the gammatone cochleagram the same condition is at 6.7% (A01). The gap to the paper comes from the FFT front end and BASS's code (A06), and is not chased; what is left to listeners is the same question as the row above |
| Minor-third complex heard as one tone ([P06](../../results/P06-shepard-minor-third-sparse.md)) | With sparse spectra and a 30 Hz floor, still one low tone, about 100 ahead of a chord; Shepard's listeners hear a chord | **Possibly.** Components a minor third apart share no period short enough to be a pitch, which a correlogram shows directly; but a finer spectrum may show it too (T3) |
| Shepard pairs: early crossover ([P12](../../results/P12-shepard-seeds.md)) | P(up) 0.40–0.43 at 5 steps on three seeds, against listeners' 0.53 | **Not expected.** The fault traced to faint energy in the lowest channels (P09). A correlogram of octave-spaced components peaks at the lowest one's period and its multiples, so an f0 an octave lower predicts much the same; it should not settle the octave better than place does. Kept as a late check, as P2 (C) asked |
| Shepard tone heard as one tone (P01) | Fixed by sparse spectra ([P05](../../results/P05-shepard-one-tone-sparse.md)) | Nothing left to fix; a check that the new term does not undo it |
| Pitch of unresolved harmonics | Not tested. With the cochleagram, a harmonic source whose harmonics are all unresolved has no f0 the likelihood can see | **Yes, by construction**, but needs new stimuli and listener data [from memory: Houtsma & Smurzynski (1990)]. A second test once T1 is done |
| Co-modulation masking release at 50 Hz [paper App. C.4] | Not run here: the scene model has no noise source (`scene.py` has harmonic and whistle sources only) | Later: needs noise sources first |
| Periodic everyday sounds heard as noise [paper Fig. G.7] | Not run here | Out of scope (everyday sounds) |

## 3. Probe: how much each representation can see a mistuned harmonic

Not an inference run. For the 100 Hz stimuli with harmonic 1, 2 or 3
mistuned, `representation_probe.py` computes the summed squared difference
between the representation of the mistuned tone and of the same tone in
tune. Under a Gaussian likelihood with SD σ, that difference divided by
2σ² is an **upper bound on what the one-source hypothesis loses** by not
mistuning, because its best fit is at least as good as the in-tune tone at
the true f0. The two-source hypothesis can fit the mistuned tone, at an
Occam cost. A threshold is roughly where the bound passes that cost.

[measured, `/mnt/project-files/timing-likelihood/probe/representation_probe.py`,
output `probe-output.txt` beside it; cloud CPU, float64]

Summed squared difference, mistuned against in tune, f0 100 Hz. Units are
dB² for the cochleagrams and (normalized autocorrelation)² for the
correlograms, so **compare shapes within a row group, not sizes across
groups**.

| Representation | Harmonic | 1% | 2% | 3% | 5% | 10% | Cells |
|---|---|---|---|---|---|---|---|
| FFT cochleagram (default) | 1 | 584 | 384 | 396 | 677 | 2850 | 3072 |
| | 2 | 1600 | 739 | 635 | 314 | 589 | |
| | 3 | 1420 | 693 | 572 | 205 | 368 | |
| Gammatone cochleagram, 25 ms | 1 | 360 | 491 | 993 | 2690 | 10400 | 3072 |
| | 2 | 574 | 323 | 553 | 1420 | 5550 | |
| | 3 | 441 | 194 | 300 | 680 | 2710 | |
| Gammatone cochleagram, 100 ms windows | 1 | 170 | 404 | 895 | 2470 | 9650 | 2624 |
| | 2 | 115 | 235 | 478 | 1340 | 5240 | |
| | 3 | 49 | 113 | 233 | 649 | 2590 | |
| Correlogram, rectified (T2 (A)) | 1 | 1430 | 1170 | 1530 | 2760 | 9060 | 406080 |
| | 2 | 325 | 452 | 690 | 1690 | 6340 | |
| | 3 | 147 | 355 | 826 | 2580 | 12600 | |
| Correlogram, linear (T3's control) | 1 | 751 | 1510 | 3160 | 7850 | 27700 | 406080 |
| | 2 | 616 | 1630 | 3290 | 9040 | 35600 | |
| | 3 | 850 | 3270 | 7360 | 20700 | 89800 | |
| Summary correlogram (summed over channels) | 1 | 0.12 | 0.40 | 0.84 | 2.07 | 5.43 | 6345 |
| | 2 | 0.06 | 0.24 | 0.51 | 1.36 | 5.34 | |
| | 3 | 0.11 | 0.42 | 0.91 | 2.33 | 6.79 | |

Correlograms: 64 gammatone channels, lags 0 to 35 ms every 0.25 ms (141
lags), 25 ms Hann windows every 10 ms, each window divided by its lag-0
value (floored at a 20 dB sinusoid's, so silent windows read near 0).

What it says, with the caveat that these are bounds on one hypothesis's
loss, not fitted evidence:

- **The cochleagram's budget.** On the gammatone cochleagram, 100 Hz
  harmonic 1 crosses at 6.7% (A01). There the bound is about 4500 dB²
  (between 2690 at 5% and 5200 at 7%), or about 22 nats at σ = 10 dB. So
  the one-source hypothesis must lose of the order of 20 nats before the
  extra whistle pays; that is the Occam cost a new term has to beat
  [estimate, from the bound at A01's threshold].
- **The gammatone cochleagram's sensitivity falls with harmonic number**
  (at 3%: 993, 553, 300 for harmonics 1, 2, 3), which is the model's
  thresholds rising with harmonic number faster than listeners'.
- **The rectified correlogram is nearly flat across harmonics** from 3%
  on (1530, 690, 826 at 3%; 9060, 6340, 12600 at 10%), as listeners are.
  This is the shape a timing term would need. At 1–2%, harmonic 1 already
  differs more than at 3%: its beats with harmonic 2 in shared channels
  show up there.
- **But the linear correlogram, with no timing at all, is flat or rising
  across harmonics too** (3160, 3290, 7360 at 3%). By Wiener–Khinchin it
  is a power spectrum with about 35 ms of lag resolution within each
  channel: finer frequency, not timing. So the flattening could come from
  fine frequency alone. That is why T3's control is the linear correlogram.
- **Longer pooling on the gammatone cochleagram adds nothing** (the 100 ms
  rows are at or below the 25 ms ones): its frequency resolution is the
  filters', not the window's. That rules out a longer-window gammatone
  cochleagram as the control, and as a cheap fix.
- **The FFT cochleagram's differences do not grow steadily with mistuning**
  (harmonic 2: 1600 at 1%, 314 at 5%): in its wide low channels, the
  mistuned component's leakage interferes with its neighbors', so the
  difference depends on their drifting relative phase as much as on the
  mistuning. That fits A08's 100 Hz thresholds being high.

Cost [measured, same script]: one forward and backward pass on a 0.5 s
sound takes 3.2 ms for the FFT cochleagram, 91 ms for the gammatone
cochleagram, and 815 ms for the rectified correlogram (748 ms linear, 328
ms summary), on the cloud CPU, 4 threads, float64. The probe's correlogram
is a plain loop over lags; a faster version (float32, fewer channels in
the term) is untried.

## 4. Decisions

### T1. First target

- **(A) The mistuned harmonic, all 9 conditions, and P06's minor-third
  complex.** Both use existing stimuli and code (`tools/bass/scene_mistuned_harmonic.py`,
  `tools/relative_pitch/`), both have a baseline, and in both the model
  hears one source where listeners hear more. The mistuned harmonic has
  listener thresholds to compare with (Fig. 7D left, by eye; the original
  is Moore, Glasberg & Peters, 1986).
- **(B) Pitch of unresolved harmonics.** The case a timing likelihood is
  most clearly for, but it needs new stimuli, a listener dataset chosen and
  read, and a judgment (which f0 is heard) the scene model reports only
  through its f0 posterior.
- **(C) Co-modulation masking release at 50 Hz.** Needs noise sources in the
  scene model first.

**Recommendation: A, then B** as the second test with its own short design.

### T2. Representation

- **(A) Correlogram.** Gammatone outputs (the 64 channels of the gammatone
  cochleagram), half-wave rectified, then per channel and per 25 ms Hann
  window every 10 ms, the autocorrelation at lags 0 to 35 ms, divided by
  its value at lag 0 (so it says nothing about level, which the cochleagram
  term keeps). 35 ms covers a period at the 30 Hz f0 floor of P12. The
  paper names the correlogram first among its candidates [paper §4.2.3].
  No compression or low-pass for phase locking at first; a low-pass near
  the phase-locking limit is a one-line option if high channels turn out
  to matter.
- **(B) Envelope modulation channels.** Per channel, the envelope's power
  in a bank of modulation filters up to a few hundred Hz [from memory:
  Dau, Kollmeier & Kohlrausch (1997)]. Smaller than A and close to what
  CMR needs, but it sees only item 2 of §1, not the fine structure of
  resolved components, which is the usual account of the mistuned
  harmonic.
- **(C) A finer spectrum only.** A multi-resolution cochleagram (also in
  the paper's list), or the linear correlogram. Not timing; it is T3's
  control, not an answer. Longer windows alone do not do it on the
  gammatone cochleagram (§3).
- **(D) The waveform itself, with each component's phase a latent.** The
  most information, but listeners are largely deaf to the phases of
  resolved components, and the likelihood would be a jagged function of
  frequency and phase, hard to fit by gradient steps.

**Recommendation: A.**

### T3. A control for fine frequency

- **(A) Yes: the linear correlogram.** Run the same tests with the same
  correlogram term (same lags, windows, normalization and σ_r procedure)
  computed on the gammatone outputs without rectification. By §1 it holds
  fine frequency within each channel and no timing. If the rectified term
  helps and the linear one helps as much, the gain is fine frequency, and
  the honest conclusion is "the cochleagram is too coarse", not "the model
  needs timing". The probe (§3) shows the linear term also flattens the
  harmonic-number trend, so this control is needed, not a formality. A
  longer-window gammatone cochleagram is not a control: its resolution is
  the filters', and 100 ms windows change nothing (§3).
- **(B) No control.**

**Recommendation: A.** It is one flag on the same code, and doubles the
runs of steps 3 and 4 in §5.

### T4. Base cochleagram

- **(A) Gammatone (A01's).** The correlogram needs real filter outputs, so
  the gammatone bank is built anyway. The FFT cochleagram and BASS's gain
  were adopted to reproduce the paper and are "not a claim about the ear"
  ([methods §1.7](../methods.md)); a timing likelihood is a step toward
  listeners, so it should be compared with the closest-to-ear baseline.
  A01 has not been rerun on today's main, so the baseline is rerun too.
- **(B) FFT (today's default).** Keeps comparison with A02 to A08 and every
  later page, but mixes the FFT front end's coarse low channels into the
  comparison.

**Recommendation: A.**

### T5. Noise and weight of the new term

The new term is far larger than the cochleagram: 64 channels × 701 lags
per window against 64 values (the 35 ms of lags at 20 kHz). Neighboring
lags are strongly correlated, so treating each as independent evidence
would let the correlogram swamp the cochleagram, and its SD would have no
meaning of its own.

- **(A) Reduced correlogram, its own SD, fitted to one listener
  threshold.** Subsample the lags (for example every 0.25 ms, 141 lags)
  and treat each cell as an independent Gaussian with SD σ_r, added to the
  cochleagram term. Choose σ_r so that the model's threshold at 200 Hz,
  harmonic 1 matches listeners' (one number, by eye from Fig. 7D), and
  report the other 8 conditions and P06 as predictions. This is the same
  move as the paper's hand-set 10 dB, made against a listener datum.
- **(B) σ_r from first principles**, for example from a model of auditory
  nerve noise. No free parameter, but a long derivation whose assumptions
  are as arbitrary as one fitted number.
- **(C) The correlogram alone, no cochleagram term.** Simpler, but the
  normalized correlogram carries no level, so loudness and masking would
  be lost; unnormalized, it would duplicate the cochleagram.

**Recommendation: A**, with σ_r from a 1-D sweep over a few values on the
one calibration condition.

### T6. Phase of rendered components

After rectification, the correlogram of unresolved harmonics depends on
their relative phases (cosine phase gives a peaky envelope, random phase a
flat one). The renderer starts every component in cosine phase at its
event's onset ([methods §1.5](../methods.md)); the mistuned-harmonic
stimuli are in cosine phase (`stimuli/mistuned_harmonic.py`, an
assumption the paper's text leaves open). Listeners are themselves somewhat
sensitive to these phases in unresolved channels [from memory], so a
phase-blind likelihood would not be more human either.

- **(A) Fixed phases, start phase matched to the stimulus.** The renderer
  gets one setting, the start phase of every component (cosine now). The
  mistuned-harmonic stimuli start in cosine phase; the Shepard stimuli in
  sine phase (`stimuli/shepard.py`). Any test stimulus must have a known,
  common start phase, and pages using the new term say which.
- **(B) Phases as latents**, one per harmonic. Correct in general, but adds
  12 parameters per source and makes the fit harder (T2 (D)'s problem in a
  milder form).

**Recommendation: A.** For resolved components a per-channel correlogram
hardly depends on phase, so the setting matters mostly for unresolved
harmonics, which T1 (A)'s stimuli barely have; it matters for T1 (B).

### T7. Scope and fitting

- **(A) An option, off by default; fit in two stages.** A new module
  `correlogram.py` and a likelihood option, as P19/P20 were done, so no
  earlier result changes. Each hypothesis is fitted first with the
  cochleagram alone (as now), then the correlogram term is added and the
  fit continued from that mode. The correlogram is a rippled function of
  f0 (peaks move with every period), so starting the fit on it alone
  invites wrong optima; the cochleagram fit places f0 first.
- **(B) Fit with both terms from the start.**

**Recommendation: A.** The evidence is then computed at the joint mode as
now (Laplace and variational importance).

## 5. Order of work, once accepted

1. `correlogram.py` with tests (rectified ACF on a sinusoid and on a
   harmonic complex against numpy; gradients finite).
2. Baseline: the 9 mistuned-harmonic conditions on the gammatone
   cochleagram on today's main (A01 rerun).
3. Calibrate σ_r on 200 Hz harmonic 1 (T5), for the rectified term and
   for the linear control (T3) separately.
4. The 9 conditions with each term, then P06's minor-third complex (and
   P05's single tone as a no-harm check).
5. One results page with all of the above against listeners, and the
   methods page updated.

Cost: every likelihood evaluation costs about 0.8 s on the cloud CPU with the
probe's code, about 9 times the gammatone cochleagram (§3). Step 1 includes
making it faster and timing one full fit before any run, and I will give
the run time for steps 2 to 4 then. All runs on Cho's PC.

## 6. Not in scope

Interaural time differences (the model is monaural); noise sources and CMR
(T1 (C)); everyday sounds; latent phases (T6 (B)); a learned periodicity
representation (the paper also lists Saddler et al.'s networks
[paper §4.2.3]), which would bring a trained network into the likelihood.

## 7. Questions for Cho

1. T1–T7: accept as recommended, or which to change?
2. T5 (A) fits one number, σ_r, to one listener threshold. Is 200 Hz
   harmonic 1 the right condition to spend it on, or would you rather fit
   to the average over all 9 and give up the held-out test?
