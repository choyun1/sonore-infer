# C3. Search over scene structures

> **Under review.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** first pass on four (b) stimuli and three ABA stimuli. **Paper target:**
App. B.2 (sequential inference); no figure. **Code:**
`src/sonore_infer/detect.py`, `src/sonore_infer/search.py`,
`tools/grammar/search_scene.py`, milestone (c) step 5. Design:
[milestone (c)](../docs/design/milestone-c.md), decisions C9 to C11.

## Question

Shared background: [how the experiments work](README.md#how-the-experiments-work).

Every page so far was told which hypotheses to compare (one note or two,
one stream or two). Can the model find the best hypothesis on its own, from
events it detects in the cochleagram? Decision C11 counts the search a
success on a stimulus when its best hypothesis has the same structure as
the winner of the enumerative comparison.

## Setup

Implemented from the paper's App. B.2 description only; no BASS code was
used.

- **Detector** (`detect.py`). Onsets are frames where the level rises by at
  least 40 dB summed over channels and is a local maximum. At each onset,
  spectral peaks of the rise become whistle candidates. Subharmonic
  summation over the peaks gives a harmonic candidate, plus its octave above
  and below, and a second harmonic when the peaks it leaves out fit another
  f0. Each onset gives one group of alternative explanations.
- **Rounds** (`search.py`). One round per onset group, fitted on the sound
  up to the next group's onset. Each explanation's events join an existing
  source of their kind, start a new source, or the group is left out (not
  in the first round). Limits as the paper's: at most 4 sources, 2 new
  sources and 3 new events per round; 20 hypotheses per round, fewest
  sources first; the best 2 go on.
- **Residual pass.** Run the detector on what the best fit leaves
  unexplained (observed minus rendered, floored at zero) and add up to 2
  more groups.
- **Cleanup.** Merge two sources, change a source's kind, remove a source,
  merge two events, remove an event; up to 8 moves from the best two.
- **Evidence** (decision C10). Each hypothesis is fitted to its mode (Adam,
  up to 15000 steps, stopping when the log joint rises by less than 0.5 over
  1000 steps) and screened by its Laplace log posterior. A Laplace that
  fails (covariance not positive definite) leaves the hypothesis out. The
  5 best get 2000 variational steps and are ranked by the variational
  importance estimate, as in [B2](B2-two-notes-variational.md).

Model as B2: FFT cochleagram, fixed kernels, likelihood SD 10 dB. One
seed. The (b) stimuli ran with one thread each on the cloud CPU; the ABA
stimuli ran on Cho's PC (CPU), three at once with two threads each.

## Results

Log posteriors on this page are shifted by a constant so they read as
misfit plus prior. The likelihood scores each cochleagram bin with a Gaussian
of SD 10 dB, and its normalising term costs log(10·√(2π)) = 3.222 nats per
bin even for a perfect fit: 9896.5 for a 0.5 s (b) stimulus (64 channels ×
48 time windows) and 63502.7 for an ABA sequence (64 × 308). That term is added
back here. It is the same for every hypothesis of a stimulus, so it cancels in
every comparison; the raw outputs in `data/` are not shifted.

Log posterior of the best structure, variational importance estimate
(`data/C3/`), against B2's enumerative values (`data/B2/`: `vi_importance` plus
`structure`). B2 predates the n! labelling count (decision C6), which adds
0.69 to the search's two-source values:

| Stimulus | Search's best | Its log posterior | Enumerative winner | Its log posterior | Same structure? | Fitted | Time (min) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Control, 200 Hz | one note | −64.7 | one note | −65.0 | yes | 5 | 20 |
| Octave, 40 ms | one note | −86.6 | one note | −85.2 | yes | 5 | 21 |
| Tritone, 40 ms | two notes | −124.3 | two notes | −119.3 | yes | 23 | 74 |
| Just fifth, 0 ms | note + whistle | −117.5 | two notes | −115.4 | **no** | 12 | 37 |

"Fitted" counts hypotheses fitted to their mode, before the 5 finalists'
variational fits. Time is wall clock for the whole search.

### ABA sequences

1, 6 and 12 semitones at 100 ms onset to onset, 12 tones (`data/C3/aba_pc/`),
against [C2](C2-aba-streaming.md)'s enumerative values (`data/C2/`: the
one-stream fit from `one_converged/`, two streams from `run_300steps/`,
`vi_importance` plus `structure`). C2 inferred each Gaussian process's sigma
and lengthscale; this search kept them fixed, so the two are not exactly the
same model.

| Stimulus | Search's best | Its log posterior | Enumerative winner | Its log posterior | Same structure? | Fitted | Time (h) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 st | one stream, 12 tones | −145.2 | one stream | −145.8 | yes | 82 | 5.7 |
| 6 st | two streams, A×8 and B×4 | −164.8 | two streams | −163.6 | yes | 99 | 8.1 |
| 12 st | two streams, A×8 and B×4 | −165.2 | two streams | −165.0 | yes | 100 | 9.2 |

At 1 semitone the best two-source hypothesis the search kept (−156.0, the
last tone split off) is 10.8 below one stream; C2's two streams are 18.4
below. At 6 and 12 semitones no one-stream hypothesis reached the finalists.
The fitted streams sit at 1000 and 1414 Hz, and 1000 and 2000 Hz, with every
tone in the right stream (`fitted:` lines).

### The just fifth's two-note fit (2026-10-17)

Why did the search's two-note fit to the just fifth (0 ms) end about 10
below B2's? The same two-note hypothesis was fitted from five starts,
on current main, as the C3 search was (fits to convergence, 2000 variational
steps; `data/C3/fifth_starts/`). The detector's candidates sit at 197
and 304 Hz. Their spectra come from the peaks. The lower note's harmonics 4, 7 and 10 (800, 1400 and
2000 Hz) start at the −30 dB floor, because no peak was resolved there
next to the upper note's 900, 1500 and 2100 Hz:

| Start | Variational log posterior, seed 0 | Seeds 0–4, mean (SD) |
| --- | --- | --- |
| Search's (197 and 304 Hz, spectra from the peaks) | −124.7 | – |
| Search's, floored harmonics started at 0 dB | −119.5 | – |
| Search's, f0s set to 200 and 300 Hz | −116.0 | −116.0 (1.5) |
| B2's (200 and 300 Hz, flat spectra) | −115.5 | −114.7 (3.0) |
| Note + whistle, the search's best | −111.5 | −115.3 (2.3) |

The seed changes only the variational fit and its samples. Every seed starts the
Adam fit at the same place. The search's own run gave −125.8 and −117.5 for its
start and its best. It ran on code from before PR #61, so its numbers differ
slightly. The Laplace estimates (−127 to −87) disagree with the variational ones
by up to 33 and do not rank these fits consistently (`data/C3/fifth_starts/`).

## What it says

- **Six of seven match.** The control, the octave, the tritone and all
  three ABA sequences end with the enumerative winner's structure. On ABA
  the search's log posteriors are within 1.2 of C2's, despite the kernel
  difference.
- **The octave matches for the wrong reason.** The detector does not find
  the upper note's onset (its harmonics coincide with the lower note's even
  harmonics), so two notes were never offered. The enumerative comparison
  also prefers one note, so the structures agree, but the search did not
  choose between them.
- **The tritone needed cleanup.** The rounds' best was two notes plus a
  whistle at 1715 Hz; cleanup removed the whistle, and that hypothesis came
  first (−124.3, against −145.7 with the whistle).
- **The fifth at 0 ms does not match.** The search did fit two notes, but
  its fit reached −125.8, about 10 below B2's −115.4 for the same
  structure. A note at 197 Hz plus a whistle at 304 Hz then scored higher
  (−117.5). The detector's f0s are the cause. Started from 200 and 300 Hz
  with the detector's spectra, the same fit recovers 9 of the 10. Lifting
  the floored harmonics alone recovers 5. With the right start the two
  notes and the note plus whistle tie: −116.0 against −115.3, over five
  seeds with SDs of 1.5 and 2.3. So this mismatch is a starting value, and
  behind it is a near tie that one seed cannot settle.
- **Cost.** 5 to 23 hypotheses and 20 to 74 minutes per 0.5 s stimulus on
  one cloud CPU, within decision C9's estimate of 1 to 3 hours. The ABA
  sequences (3.1 s with the 12 tones in the first 1.5 s, 12 rounds) took 82 to 100 hypotheses and 5.7 to 9.2
  hours each on the PC. Every round refits its hypotheses from scratch on the
  sound so far; starting each round from the previous fit would cut this.

## Reproduce

```
python tools/grammar/search_scene.py control:200
python tools/grammar/search_scene.py two:octave:0.04
python tools/grammar/search_scene.py two:tritone:0.04
python tools/grammar/search_scene.py two:just_fifth:0.0
python tools/grammar/search_scene.py aba:1:0.1 --threads 2
python tools/grammar/search_scene.py aba:6:0.1 --threads 2
python tools/grammar/search_scene.py aba:12:0.1 --threads 2
python tools/grammar/search_starts.py true-f0 --seed 1
```

Defaults: `--steps 15000 --tolerance 0.5 --variational-steps 2000 --keep 5
--max-cleanup 8`, seed 0.

## Open

- ABA at 3 semitones, where C2's preference for two streams (+8 to +13)
  disagrees with the paper; and the search with the kernel inferred, as C2.
- Warm starts between rounds, to bring the ABA cost down.
- The octave's upper onset: the detector would need a cue other than a rise
  in level at new frequencies, or the residual pass would need to find it.
- The detector's f0s: refine each harmonic candidate's f0 from its peaks
  (197 Hz here, for a 200 Hz note) before the fits start.
- More (b) stimuli and asynchronies, and more seeds.
