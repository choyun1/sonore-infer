# C3. Search over scene structures

> **Under review by Adrian Cho.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** first pass on four (b) stimuli; ABA pending. **Paper target:**
App. B.2 (sequential inference); no figure. **Code:**
`src/sonore_infer/detect.py`, `src/sonore_infer/search.py`,
`tools/search_scene.py`, milestone (c) step 5. Design:
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
seed, one thread per stimulus on the cloud CPU.

## Results

Log posterior of the best structure, variational importance estimate
(`data/C3/`), against B2's enumerative values (`data/B2/`: `vi_importance` plus
`structure`). B2 predates the n! labelling count (decision C6), which adds
0.69 to the search's two-source values:

| Stimulus | Search's best | Its log posterior | Enumerative winner | Its log posterior | Same structure? | Fitted | Time (min) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Control, 200 Hz | one note | −9961.2 | one note | −9961.5 | yes | 5 | 20 |
| Octave, 40 ms | one note | −9983.1 | one note | −9981.7 | yes | 5 | 21 |
| Tritone, 40 ms | two notes | −10020.8 | two notes | −10015.8 | yes | 23 | 74 |
| Just fifth, 0 ms | note + whistle | −10014.0 | two notes | −10011.9 | **no** | 12 | 37 |

"Fitted" counts hypotheses fitted to their mode, before the 5 finalists'
variational fits. Time is wall clock for the whole search.

## What it says

- **Three of four match.** The control, the octave and the tritone end with
  the enumerative winner's structure.
- **The octave matches for the wrong reason.** The detector does not find
  the upper note's onset (its harmonics coincide with the lower note's even
  harmonics), so two notes were never offered. The enumerative comparison
  also prefers one note, so the structures agree, but the search did not
  choose between them.
- **The tritone needed cleanup.** The rounds' best was two notes plus a
  whistle at 1715 Hz; cleanup removed the whistle, and that hypothesis came
  first (−10020.8, against −10042.2 with the whistle).
- **The fifth at 0 ms does not match.** The search did fit two notes, but
  its fit reached −10022.3, about 10 below B2's −10011.9 for the same
  structure. A note at 197 Hz plus a whistle at 304 Hz then scored higher
  (−10014.0). The gap between the two-note fits is not explained yet; the
  search's starting values (f0 197 Hz from the detector, its spectra from
  the peaks) are the first suspect, not checked.
- **Cost.** 5 to 23 hypotheses and 20 to 74 minutes per 0.5 s stimulus on
  one cloud CPU, within decision C9's estimate of 1 to 3 hours.

## Reproduce

```
python tools/search_scene.py control:200
python tools/search_scene.py two:octave:0.04
python tools/search_scene.py two:tritone:0.04
python tools/search_scene.py two:just_fifth:0.0
```

Defaults: `--steps 15000 --tolerance 0.5 --variational-steps 2000 --keep 5
--max-cleanup 8`, seed 0.

## Open

- ABA stimuli (C11 asks for them too): 12 onsets, so 12 rounds on a growing
  sound; being timed.
- The octave's upper onset: the detector would need a cue other than a rise
  in level at new frequencies, or the residual pass would need to find it.
- The fifth's two-note fit: start it from the true f0s to see whether the
  10 lost are starting values or spectra.
- More (b) stimuli and asynchronies, and more seeds.
