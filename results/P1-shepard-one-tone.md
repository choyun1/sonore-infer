# P1. One Shepard tone alone

> **Under review.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** first pass, one seed, all 10 tones; search on tone 1.
**Paper target:** none in Cusimano et al.; listeners in Shepard (1964) §I.
**Code:** `src/sonore_infer/stimuli/shepard.py`,
`tools/relative_pitch/one_tone.py`, `tools/relative_pitch/search_tone.py`.
Design: [relative pitch](../docs/design/relative-pitch.md), decisions P3,
P4 and P7.

## Question

Shared background: [how the experiments work](README.md#how-the-experiments-work).

How does the unchanged model hear one Shepard tone: as one harmonic tone
(and at which octave), as a set of whistles, or as a mix? Shepard's
listeners hear one tone, not a chord [Shepard §I]. This decides what the
pair experiments (P3, P4) compare, and whether the spectrum prior has to
change first (decision P3).

## Setup

- **Stimuli.** Shepard's Eqs. 1–3 with his experiment's values: 10
  components an octave apart under a raised-cosine envelope in dB over log
  frequency, 22 to 56 dB, peak at 155.6 Hz; tones 1 to 10 step through the
  octave in tenths. 125 ms, 10 ms ramps, 50 ms of silence each side
  (0.225 s). Components above 5 kHz left out. Assumptions and departures
  are listed in `shepard.py` (decision P6).
- **Hypotheses** (decision P4), each fitted from a start built from the
  stimulus:
  - `harmonic@F`: one harmonic source with f0 started at component F, one
    per octave reading from the lowest component above 87.5 Hz up to
    1300 Hz. Its harmonics start at the stimulus's component levels, and
    the harmonics the stimulus lacks 30 dB below the loudest.
  - `whistles×N`: one whistle per component at or above 30 dB and above
    87.5 Hz (4 or 5 whistles).
- **The prior floor.** The f0 and whistle-frequency prior is uniform on 3
  to 33.19 ERB number, 87.5 Hz to 8.0 kHz (Table A.2's range, as
  everywhere else). No hypothesis can explain a component below 87.5 Hz.
  The cochleagram reaches down to 20 Hz, so the tone's components between
  20 and 87.5 Hz (2 or 3 per tone, up to 53 dB) are left unexplained by
  every hypothesis. A first pass that started readings below the floor
  gave −inf for them.
- **Evidence.** Adam to convergence (up to 15000 steps, stop when the log
  joint rises by less than 0.5 over 1000 steps), then 2000 variational
  steps and 128 importance samples; ranked by the variational importance
  estimate plus the structure prior, as in
  [B2](B2-two-notes-variational.md). Where importance sampling failed
  (−inf), the ELBO is used and marked.
- **Search.** The C3 search (`tools/grammar/search_scene.py`, unchanged,
  wrapped to build Shepard tones) on tone 1, as a check that the
  enumerated set is not missing a better structure.

Model as B2: FFT cochleagram with BASS's gain, fixed kernels, likelihood SD
10 dB. One seed. Cloud CPU (4-core Xeon, 2.1 GHz); several runs shared it,
so the times are rough.

## Results

Log posteriors are shifted by the likelihood's normalising term, 4329.7
for a 0.225 s tone (64 channels × 21 windows × 3.222), as on
[C3](C3-search.md); raw outputs in `data/P1/` are not shifted.

| Tone | Best harmonic reading (start) | Its log posterior | Next octave up, relative | Whistles | Whistles, relative to the best harmonic |
| --- | --- | --- | --- | --- | --- |
| 1 | 155.6 Hz | −126.0 | −137.6 | 4 | **+9.3** |
| 2 | 166.8 Hz | −136.8 | −140.7 | 4 | **+20.1** |
| 3 | 89.4 Hz | −95.3 | −51.6 | 5 | −7.3 |
| 4 | 95.8 Hz | −99.0 | −50.4 | 5 | −4.5 |
| 5 | 102.7 Hz | −96.2 | −80.5 | 4 | −0.1 |
| 6 | 110.0 Hz | −107.3 | −76.2 | 4 | **+9.1** |
| 7 | 117.9 Hz | −109.7 | −89.9 | 4 | **+13.6** |
| 8 | 126.4 Hz | −112.2 | −99.7 | 4 | **+14.5** |
| 9 | 135.5 Hz | −118.7 | −114.0 | 4 | **+14.5** (ELBO) |
| 10 | 145.2 Hz | −124.3 | −122.5 | 4 | **+10.1** |

The effective sample size of the importance estimate is 1 to 4 for every
best reading and every whistle set, and the ELBO differs from it by up to
10, so margins under about 10 are not resolved. The Laplace estimates
agree in sign at 8 of 10 tones (`data/P1/step*.txt`).

**Search on tone 1** (`data/P1/search_step1.txt`, 17 hypotheses, 29 min).
The detector offered whistles at the four components and harmonic tones at
155, 309 and 77 Hz. Its best is a harmonic tone fitted to 310 Hz plus a
whistle at 111 Hz, log posterior −88.9: 37 above the best enumerated
reading (one tone at 155.6 Hz) and 28 above the four whistles.

**Time per hypothesis**, fitted to convergence plus the variational fit:
86 to 247 s; 11 to 15 min per tone for 4 or 5 hypotheses.

## What it says

- **The octave the model hears is set by the prior floor, not by
  Shepard's envelope.** The best reading is always the lowest component
  above 87.5 Hz, 50 to 140 ahead of the next octave up. It climbs from
  89 Hz at tone 3 to 167 Hz at tone 2 and then drops an octave. Its log
  posterior is best at tones 3 to 5 and worst at tones 1 and 2. Probably
  (inferred, not tested) because at tones 1 and 2 a 53 dB component sits
  just below the floor, unexplained, while at tones 3 to 5 the strongest
  unexplained one is an octave lower and 5 to 7 dB weaker.
- **One harmonic tone is not clearly preferred.** Whistles win by 9 to 20
  at 7 of 10 tones, and the two are within the estimates' noise at tones 3
  to 5. Shepard's listeners hear one tone at every step.
- **The search finds a better mix than either.** On tone 1, a tone at
  310 Hz plus a whistle at 111 Hz beats both enumerated hypotheses by about
  30. So decision P4's enumerated set misses the model's preferred reading,
  at least on tone 1.
- **Energy below 87.5 Hz is unexplained by construction.** It costs every
  hypothesis about equally on a tone, so it does not change the rankings
  within a tone, but it changes which octave is read and makes tones
  differ from each other. A pair experiment (P3) would inherit this.
- **Not checked:** other seeds, other tones in the search, and whether a
  lower prior floor or a sparse-spectrum prior (decision P3 (B)) changes
  the picture.
