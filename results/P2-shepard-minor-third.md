# P2. Octave against minor-third spacing

> **Under review.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Rerun** with sparse spectra and a 30 Hz floor: [P6](P6-shepard-minor-third-sparse.md).
**Status:** first pass, one seed, tones 1 and 4; search on tone 1.
**Paper target:** none in Cusimano et al.; listeners in Shepard (1964) §V.
**Code:** as [P1](P1-shepard-one-tone.md), with `--minor-third`.
Design: [relative pitch](../docs/design/relative-pitch.md), decisions P3,
P4 and P7.

## Question

Shared background: [how the experiments work](README.md#how-the-experiments-work).

Shepard's control: the same envelope with components a minor third apart
instead of an octave sounds like a chord, while the octave-spaced tone
sounds like one tone [Shepard §V]. Does the model tell the two apart the
same way?

## Setup

As [P1](P1-shepard-one-tone.md), with 4 components per octave (40, of which
those up to 5 kHz are kept) under the same envelope. Hypotheses: one
harmonic source per component reading from the lowest above 87.5 Hz (up to
1300 Hz at tone 1, 200 Hz at tone 4), and one whistle per component at or
above 30 dB and above 87.5 Hz (17 whistles). Tone 1's harmonic readings
and its whistles ran separately, because a container restart stopped the
first run (`data/P2/step1_harmonics.txt`, `step1_whistles.txt`). The
search ran on tone 1.

Log posteriors are shifted by 4329.7, as on P1. Importance sampling failed
(−inf) for every 17-whistle fit, so the whistles are compared by the ELBO,
a lower bound.

## Results

| Tone | Best harmonic reading (start) | Its log posterior | Next reading, a minor third up, relative | 17 whistles, relative (ELBO) |
| --- | --- | --- | --- | --- |
| 1 | 92.5 Hz | −100.6 | −8.5 | −279.0 |
| 4 | 95.8 Hz | −96.0 | −10.3 | −270.0 |

At tone 1 the whistles fit 727 better at their modes (log joint) than
the best harmonic reading, but their 850 parameters lose about
960 in the Laplace volume term and 52.5 in the structure prior
(`data/P2/step1_*.txt`). The Laplace estimates give the same margins
within 15.

**Search on tone 1** (`data/P2/search_step1.txt`, 12 hypotheses, 30 min).
Its best is one harmonic tone fitted to 97 Hz alone, log posterior −95.4,
5.2 above the enumerated reading at 92.5 Hz. Two whistles (305 and
113 Hz) came out 99 below it.

## What it says

- **The model gets Shepard's contrast backwards.** It hears the
  minor-third complex as one low tone, by about 270, and the octave
  complex ([P1](P1-shepard-one-tone.md)) as whistles or a tone plus a
  whistle. Listeners hear the octave complex as one tone and the
  minor-third complex as a chord.
- **Why, in the model's terms (inferred).** A harmonic tone at about
  90 Hz puts harmonics in most of the cochleagram channels the dense
  minor-third components excite, and the harmonic spectrum prior
  (σ 11.8 dB, ℓ 4.7 ERB) lets it shape them under the envelope cheaply. An octave-only complex leaves most of a harmonic
  tone's harmonics absent, which that prior penalises, so whistles compete.
  Each whistle adds its own parameters and source prior, so 17 whistles
  lose.
- **This bears on decision P3.** Decision P3 (A) kept the spectrum prior
  unchanged as a prediction, to be revisited with these numbers. Options
  are (B) sparse harmonic spectra or (C) a shorter spectrum lengthscale;
  the 87.5 Hz floor of the f0 prior is a separate question.
- **Not checked:** other seeds and tones, and whether 17 whistles lose by
  as much when their importance estimate succeeds.
