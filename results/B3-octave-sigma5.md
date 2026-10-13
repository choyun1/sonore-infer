# B3. Octave with the likelihood SD at 5 dB

> **Under review by Adrian Cho.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** done (2026-10-10). **Paper target:** none (see
[B1](B1-two-notes-laplace.md)). **Code:** `tools/compare_two_notes.py --sigma`,
PR [#24](https://github.com/choyun1/sonore-infer/pull/24) (`2ee4996`).
Design: [milestone (b)](../docs/design/milestone-b.md).

## Question

Shared background: [how the experiments work](README.md#how-the-experiments-work).

In [B2](B2-two-notes-variational.md) the octave stays one source up to
240 ms. Is that because the likelihood's 10 dB SD (BASS's value) makes the
upper note's late +6 dB on the shared harmonics cheap to explain with one
source?

## Setup

As B2 (300 Adam steps, variational evidence with 2000 steps, one seed),
with the likelihood SD at 5 dB instead of 10 dB, for the octave at all 8
asynchronies and the 4 one-note controls.

## Results

Log odds of two sources over the best one source, variational estimate,
ELBO in brackets (`data/B3/`, by `tools/two_notes_summary.py`):

| Asynchrony (ms) | 0 | 10 | 20 | 40 | 80 | 120 | 160 | 240 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SD 10 dB (B2) | −33.6 | −33.1 | −28.3 | −27.0 | −26.3 | −23.4 | −25.4 | −21.9 |
| SD 5 dB | −65.0 (−80.0) | −57.5 (−73.5) | −54.7 (−70.7) | −45.2 (−72.4) | −20.2 (−44.2) | −12.6 (−36.7) | −12.3 (−26.5) | −8.2 (−17.2) |

Controls at 5 dB: 200 Hz −25.6, 282.8 Hz −19.0, 300 Hz −21.7, 400 Hz
−19.5; all one source, as at 10 dB.

## What it says

- At 5 dB the octave is still one source at every asynchrony up to 240 ms.
  The odds rise faster with asynchrony than at 10 dB (by 57 against 12
  over 0 to 240 ms) but do not cross zero.
- Halving the SD raises both sides of the trade. Two sources fit better at
  the mode by about 51 to 65 (13 to 20 at 10 dB), but the cost of the
  second source's parameters grows too: from mode to evidence two sources
  lose about 145 to 200 and one source about 85 to 89 (at 10 dB, 87 to 103
  and 53 to 55). So the 10 dB SD alone does not explain the octave.
- Looking inside single fits (`decompose.txt`, `channels.txt`): one source
  misfits by about 1 dB RMS over the cochleagram at 0 ms as at 240 ms. At
  240 ms it follows about half the 6 dB step with its level trajectory. At
  0 ms its misfit comes from the spectrum prior, which penalises the
  alternating +6 dB even harmonics. These are readings of one fit each.

## Reproduce

```
for a in 0 0.01 0.02 0.04 0.08 0.12 0.16 0.24; do
  python tools/compare_two_notes.py --variational-steps 2000 --sigma 5 \
      --interval octave --asynchronies $a > octave_$a.txt
done
python tools/compare_two_notes.py --variational-steps 2000 --sigma 5 --controls > controls.txt
python tools/two_notes_summary.py data/B3/octave_*.txt data/B3/controls.txt
```

The decomposition scripts (`decompose.py`, `channels.py`) are local, in the
project files next to the raw outputs; their outputs are in `data/B3/`.

## Open

- One seed, one fit per hypothesis; two-source fits start at the truth.
- Effective sample sizes are 1 to 7 of 256, so the ELBO is the safer
  guide; it agrees in sign everywhere.
