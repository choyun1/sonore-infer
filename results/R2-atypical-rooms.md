# R2. Atypical rooms heard by an exponential room model

> **Under review.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** done (2026-10-12). **Paper target:** none in Cusimano et al.
(2024); see [R1](R1-source-and-room.md). The design's experiment (b) (decision R7),
after Traer & McDermott (2016), who found listeners' judgments of
reverberation biased by atypical rooms. Listeners are not compared here.
**Code:** `tools/source_room/atypical_rooms.py`, PR
[#37](https://github.com/choyun1/sonore-infer/pull/37) (`d03e2ef`), using
R1's inference. Design: [source and room](../docs/design/source-reverb.md).

## Question

Shared background: [how the experiments work](README.md#how-the-experiments-work).

The model assumes rooms decay exponentially with an ecological RT60
profile across frequency. Given a room that breaks one of those
assumptions, what RT60 does it infer, and is that the same reading a
standard acoustic measure gives?

## Setup

The default source (sonore's `gaussian_spectrogram` defaults, 400 ms) in a
`synth_ir` room at the typical DRR (9.9 dB), with one atypical property:

- a decay shape: **time-reversed**, **linear matched at the start**, or
  **linear matched at the end** (of the exponential's decay);
- or an RT60 profile across frequency: **inverted**, **exaggerated**, or
  **reduced**;
- and the normal exponential room as the control.

5 seeds × RT60 {0.4, 0.8} s × 7 rooms × floors of 60, 40 and 20 dB below
the peak = 210 inferences, at the measured sigma (3.6 dB). The source and
room noise are shared across rooms within a seed, so each room hears the
same source. For comparison, each room's RT60 is also measured from its
impulse response by Schroeder T20 × 3 (median over bands).

## Results

Posterior median RT60 over the true RT60, median over 5 seeds
(`data/R2/atypical_rooms_summary.txt`); T20 × 3 of the same rooms in the
last column.

| Room | 0.4 s, floor 60 | 0.4 s, 40 | 0.8 s, 60 | 0.8 s, 40 | T20 × 3, 0.4 and 0.8 s |
| --- | --- | --- | --- | --- | --- |
| Normal (control) | 0.99 | 0.95 | 1.01 | 1.04 | 1.03–1.11 |
| Exaggerated profile | 1.03 | 1.02 | 1.05 | 1.07 | 1.05–1.12 |
| Reduced profile | 0.95 | 0.87 | 0.99 | 1.08 | 1.04–1.07 |
| Linear, matched at end | 1.72 | 2.58 | 2.13 | 2.50 | 1.78–1.84 |
| Linear, matched at start | 0.49 | 0.56 | 0.43 | 0.60 | 0.37–0.39 |
| Inverted profile | 0.40 | 0.36 | 0.35 | 0.41 | 0.35–0.47 |
| Time-reversed | 2.93 | 2.60 | 2.50 | 0.13 | 0.14 |

At the 20 dB floor the posterior median RT60 (median over seeds) is 0.17
to 0.90 s in the 0.4 s rooms and 0.27 to 0.46 s in the 0.8 s rooms, which
all read about half their RT60, close to the prior's median of 0.36 s.
Single seeds scatter widely (0.14 to 1.41 s). Four of the 70 fits at that
floor failed (two returned the prior exactly, two have no finite
posterior) and are left out.

Log marginal likelihood against the control (same seed, RT60 and floor),
at floors 60 and 40: time-reversed −2,141 to −34,458 nats; the other rooms
−1,750 to +368.

## Plots

![R2-atypical-rooms](plots/R2-atypical-rooms.svg)

*Posterior median RT60 over the true RT60, median over 5 seeds, against T20 × 3 measured from the same impulse responses. From [`data/R2/atypical_rooms_summary.txt`](data/R2/atypical_rooms_summary.txt).*

## What it says

- Where the room breaks an assumption, the model's RT60 is biased the same
  way as T20 × 3, by a similar amount: linear decays matched at the end
  read long (1.7–2.6 against 1.8), matched at the start short (0.43–0.60
  against 0.37–0.39), and the inverted profile short (0.35–0.41 against
  0.35–0.47). The exaggerated and reduced profiles read about right, as
  does the normal room.
- The time-reversed room is the exception: the model reads it long (2.5 to
  2.9) where T20 × 3 reads it very short, except at 0.8 s and floor 40
  (0.13). Its evidence is thousands of nats below the normal room's, so
  the exponential model simply fails to fit it.
- The other rooms' evidence differences are between different sounds, so
  they are not a model comparison.
- At a 20 dB floor (BASS's), one sound leaves about 70 cells above the
  floor and RT60 is poorly identified: the 0.8 s rooms all read about
  0.45 s, whatever their shape.

## Reproduce

```
python tools/source_room/atypical_rooms.py --out atypical_rooms.jsonl --shard 0 --shards 4   # and shards 1-3
python tools/source_room/atypical_rooms.py --summary atypical_rooms.jsonl
```

Raw outputs: `data/R2/atypical_rooms.jsonl` (one line per inference) and
`atypical_rooms_summary.txt` (rerunning `--summary` on the committed file
gives the same text).

## Open

- One sound per inference. Combining several sounds heard in one room, as
  Traer & McDermott's listeners could not, is not tested.
- The fit fix for the 20 dB floor (fits now start with the audible cells
  raised to their observed level) still left 4 of 70 failures.
