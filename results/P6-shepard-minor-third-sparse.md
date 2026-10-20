# P6. Octave against minor-third spacing, sparse spectra and a 30 Hz floor

> **Under review.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** first pass, one seed, tones 1 and 4.
**Paper target:** none in Cusimano et al.; listeners in Shepard (1964) §V.
**Code:** as [P5](P5-shepard-one-tone-sparse.md), with `--minor-third`.
Design: [relative pitch prior](../docs/design/relative-pitch-prior.md),
decisions P10 to P14.

## Question

Shared background: [how the experiments work](README.md#how-the-experiments-work).

[P2](P2-shepard-minor-third.md) again, with the options of
[P5](P5-shepard-one-tone-sparse.md). With them the model hears the
octave-spaced tone as one tone. Does it now hear the minor-third control
as a chord, as Shepard's listeners do [Shepard §V]?

## Setup

As [P5](P5-shepard-one-tone-sparse.md), with 4 components per octave.
Hypotheses:

- `harmonic@F` for every component from 30 Hz up to 200 Hz (11 readings).
- `chord×4`: a chord of four harmonic sources, one at each of the four
  lowest readings (32.7 to 55.0 Hz at tone 1, 33.9 to 57.0 Hz at tone 4).
  With sparse spectra each can keep only its own octave-spaced components,
  so the four together are four Shepard-like tones a minor third apart.
- One whistle per component at or above 30 dB and 30 Hz (23 whistles).

Each harmonic source has the two starts of P5, and each mode is scored
separately (`combine_starts.py`). 50 fits took 158 to 2046 s each; a
chord took about 30 min per start.

## Results

Log posteriors shifted by 4329.7, as on P1. All values in
`data/P6/combined.txt`.

| Tone | Best (one harmonic source at) | Its log posterior | Next best reading, relative | Chord of 4, relative | 23 whistles, relative |
| --- | --- | --- | --- | --- | --- |
| 1 | 92.5 Hz | −105.5 | −0.3 (77.8 Hz) | −101.2 (ELBO) | −453.0 |
| 4 | 95.8 Hz | −98.9 | −4.5 (80.6 Hz) | −113.7 (ELBO) | −420.8 (ELBO) |

Importance sampling failed (−inf) for both chords, so they are compared by
the ELBO, a lower bound. Their better mode is the one with every harmonic
present at the start; the other is about 90 lower.

For every single reading from about 46 Hz up, the better mode is the one
with every harmonic present, the reverse of P5; for the two lowest readings
(about 33 and 40 Hz) it is the sparse one. At tone 1 the sparse start alone gives 65.4 Hz
at −140.1 (`data/P6/tone1-combinations/mt_sparse_30.txt`), 35 below the
dense mode's best.

**The four combinations on tone 1** (`data/P6/tone1-combinations/`; the
sparse runs there have the sparse start only):

| Spectrum prior, floor | Best reading | Its log posterior | Whistles |
| --- | --- | --- | --- |
| GP, 87.5 Hz ([P2](P2-shepard-minor-third.md)) | 92.5 Hz | −100.6 | −379.6 (ELBO, 17 whistles) |
| GP, 30 Hz | 92.5 Hz | −99.9 | −558.5 (23 whistles) |
| Sparse, 87.5 Hz | 110.0 Hz | −152.9 | −398.1 (ELBO, 17 whistles) |
| Sparse, 30 Hz | 65.4 Hz | −140.1 | −558.5 (23 whistles) |

The sparse-start-only runs fit the minor-third complex 40 to 52 worse than
the GP, which is what led to the second start.

## What it says

- **The model still gets Shepard's contrast half backwards.** It now hears
  the octave complex as one tone ([P5](P5-shepard-one-tone-sparse.md)), as
  listeners do, but it still hears the minor-third complex as one low tone,
  by about 100 over a chord of four tones and 420 over whistles. Listeners
  hear a chord [Shepard §V].
- **Why, in the model's terms (inferred, not tested).** One harmonic source
  at about 90 Hz has a harmonic near most of the minor-third components,
  and with every harmonic present its spectrum GP shapes them under the
  envelope cheaply. Four sources pay four source priors and four sets of
  latents for about the same fit. Nothing in the options makes a chord
  cheaper than one dense tone.
- **The sparse prior costs a dense tone little.** At tone 1 the best
  reading is 5.6 lower with the sparse options than with the GP alone
  (−105.5 against −99.9), the price of the presence latents.
- **Not checked:** other seeds and tones; whether a chord of Shepard-like
  tones would win against one tone with a different chord structure (for
  example, three sources); the search with the options.
