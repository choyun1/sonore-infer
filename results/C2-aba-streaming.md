# C2. ABA sequences: one stream or two

> **Under review by Adrian Cho.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** map done (2026-10-14); seeds 0–4 for the two cells under 10
done (2026-10-14). **Paper target:** Fig. 8D (right), App. C.10. **Code:**
`tools/compare_aba.py`, milestone (c) step 4. Design:
[milestone (c)](../docs/design/milestone-c.md), decision C8.

## Question

Shared background: [how the experiments work](README.md#how-the-experiments-work).

The first test of the scene grammar. An ABA sequence (A, B, A, silence,
repeated) is heard as one galloping stream when the tones are close in
frequency and as two isochronous streams when they are far apart and fast
(Van Noorden, 1975). Does the model's evidence for two streams over one
follow that pattern across the paper's 30 stimuli?

## Setup

The paper's stimuli (App. C.10): A at 1000 Hz, B 1, 3, 6, 9 or 12 semitones
higher, onset-to-onset intervals of 67 to 150 ms, 50 ms tones at 70 dB with
10 ms ramps, four triplets, 3.1 s in all. Ours start 50 ms into the sound,
because a fit cannot move an onset that starts exactly at 0 s, the edge of
the onset prior.

Two hypotheses, each started from the stimulus's own tone frequencies,
onsets and durations, as the paper did:

- **one stream:** one whistle source with all 12 tones as its events
  (652 parameters);
- **two streams:** one whistle with the 8 A tones, one with the 4 B tones
  (1280 parameters).

Full model as in [C1](C1-kernel-inferred.md): trajectories, all timing
inferred, each GP's sigma and lengthscale inferred. Variational evidence
with 2000 steps, seed 0. The one-stream fits run until the best log joint
rises by less than 0.5 over 1000 Adam steps (1162 to 9532 steps). The
two-stream fits ran 300 steps, which is enough for them: on all 30 stimuli
they gain 0.2 from 300 to 1000 steps (see "Fit steps" below). All runs on
Cho's PC.

## Results

Log odds of two streams over one, variational importance estimate
(`data/C2/map.txt`; positive favours two streams):

| Semitones \ interval (ms) | 67 | 83 | 100 | 117 | 134 | 150 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | −23.1 | −18.5 | −18.7 | −16.4 | −17.7 | −17.9 |
| 3 | 11.5 | 7.8 | 13.2 | 10.2 | 10.2 | 9.5 |
| 6 | 93.0 | 92.1 | 97.2 | 94.0 | 25.6 | 17.7 |
| 9 | 216.9 | 66.6 | 48.1 | 41.0 | 34.1 | 33.9 |
| 12 | 136.1 | 83.9 | 62.4 | 50.3 | 42.7 | 39.0 |

The ELBO and Laplace estimates give the same signs everywhere. The ELBO
is within 6.4 of this table in every cell; Laplace is within 7.1 at 1 and 3
semitones and up to 30 higher at 6 to 12 semitones (`data/C2/map.txt`).

The paper's model, read by eye from Fig. 8D (right), whose color scale
stops at ±20: about −15 to −20 at 1 semitone, slightly negative (about −5)
at 3, slightly positive (about +5) at 6, and +10 to +20 or more at 9 and 12,
with little change across intervals.

### Seeds 0–4 on the two cells under 10

Two cells had margins under 10, so they were rerun with seeds 1 to 4
(`data/C2/seeds/`), as the design asks (milestone-c.md §8). The seed
changes only the draws of the evidence estimates; the fits are the same.
In these reruns the two-stream fit also used the stopping rule (1222
steps); its log joint at the mode is within 0.3 of the 300-step fit and its
Laplace evidence within 0.3.

| Stimulus | Seed 0 | Seeds 1–4 | Mean (SD), seeds 0–4 |
| --- | --- | --- | --- |
| 3 st, 83 ms | 7.8 | 8.5, 8.9, 10.5, 9.7 | 9.1 (1.0) |
| 3 st, 150 ms | 9.5 | 9.0, 7.9, 10.1, 9.1 | 9.1 (0.8) |

Variational log odds. The ELBO gives 7.3 to 9.6 on seeds 1–4, and Laplace
gives 6.0 and 5.4. Every seed prefers two streams, by 8 to 10.5, so the
3-semitone preference for two streams is not seed noise.

## Plots

![C2-aba-streaming](plots/C2-aba-streaming.svg)

*Log odds of two streams over one, variational importance estimate, from [`data/C2/`](data/C2/).*

## What it says

- The sign follows the expected pattern at 1, 6, 9 and 12 semitones: one
  stream at 1 semitone, two from 6 up. At 3 semitones the model prefers two
  streams by 8 to 13, where the paper's model is slightly negative and
  listeners are below or near Van Noorden's two-stream boundary (about 3 to
  4 semitones).
- Faster sequences favour two streams more at 6 to 12 semitones (6
  semitones drops from about 95 at 67–117 ms to 18–26 at 134–150 ms).
  The paper's model map changes little across intervals.
- Our margins away from the boundary are much larger than the paper's
  color scale shows. The paper averages 10 runs; this is one seed.

## Fit steps

With 300 Adam steps, as milestones (a) and (b) used, the one-stream fit is
far from its mode: at 6 semitones and 100 ms its log joint is −63313,
against −62875 when converged, and the map then read +48 there instead of
+97 (`data/C2/run_300steps/`, `data/C2/fit_steps_check/one_fit_curve.txt`).
At 3 semitones the 300-step map read +5 to +21. The stopping rule
gives the same Laplace evidence as 15000 steps and variational evidence
within 0.6, checked at 3 and 6 semitones, 100 ms
(`data/C2/fit_steps_check/st*_one_*.txt`).

## Reproduce

```
python tools/run_seeds.py --out aba_out --only aba --seeds 0          # both hypotheses, 300 steps
python tools/run_seeds.py --out aba_one --only aba --seeds 0 --hypotheses one --tolerance 0.5
python tools/aba_map.py aba_one/aba aba_out/aba > map.txt             # one from the rerun, two from the first
python tools/run_seeds.py --out aba_seeds --only aba --seeds 1 2 3 4 --tolerance 0.5 --stimuli 3:0.083 3:0.15
python tools/aba_fit_steps.py --semitones 6 --interval 0.1 --hypothesis one --steps 15000
```

Each `compare_aba.py` hypothesis takes about 1 to 1.5 h on one thread of
Cho's PC [measured]. `data/C2/cloud_1st/` holds four 1-semitone stimuli
run in the cloud container at 300 steps: the same Laplace evidence as the
PC, variational evidence within 5.

## Open

- The 3-semitone row prefers two streams, under all five seeds where it
  was closest; the paper's model and listeners lean to one.
- 9 semitones at 67 ms (217) stands out from its row.
- Milestones (a) and (b) used 300 Adam steps; whether their one-source fits
  had converged has not been rechecked.
