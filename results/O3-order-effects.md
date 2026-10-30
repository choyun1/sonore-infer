# O3. Reset of streaming after silence, online

> **Under review.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** first pass, one run per stimulus; Cusack et al.'s 10 s gaps
still running. **Paper target:** none (the paper has no order effects); the
references are two listening studies.
**Code:** `src/sonore_infer/online.py`, `src/sonore_infer/stimuli/reset.py`,
`tools/online/online_reset.py`, `tools/online/reset_summary.py`. Design:
[order effects](../docs/design/order-effects.md), decisions E1–E7, step 3
of §7.

## Question

Shared background: [how the experiments work](README.md#how-the-experiments-work).

A batch model gives the same answer whatever order the sound arrives in;
the online model of [O1](O1-online-buildup.md) need not. Two studies
interrupt a sequence with silence and disagree on what follows:

- **Beauvois & Meddis (1997).** 10 s of A tones alone, a silence, then a
  1.44 s ABAB test. The inducer makes the test sound segregated, and the
  bias fades gradually with the silence (time constant 3.84 s). The test
  alone always sounded coherent.
- **Cusack et al. (2004), Experiment 3.** 10 s ABA– sequences separated by
  silences of 1 to 10 s. After every gap, even 1 s, streaming starts again
  from mostly one stream and builds up as from scratch, whatever the gap.

Does the model's time course behave like either? Design §3 wrote down
what to expect before any run.

## Setup

Each study's stimuli as printed (design E2; `stimuli/reset.py` lists the
assumptions), after 50 ms of silence:

- **Beauvois & Meddis:** A 1000 Hz, B 1420 Hz, 40 ms tones every 90 ms,
  65 dB; silences 0, 0.5, 1, 2, 4 and 8 s, and the test alone.
- **Cusack et al.:** A 400 Hz, B 4 or 6 semitones up, 125 ms tones in
  500 ms ABA– triplets, 55 dB; silences 1, 2 and 5 s (10 s still running);
  the second sequence cut to its first 4 s.

Four hypotheses (design E3): one whistle with every tone (**one**), one
with the A tones and one with the B tones (**two**), and the same two with
their sources ending at the silence and new ones starting after it
(**one restart**, **two restart**). Each runs online as in O1 and O2: 100 ms
windows, lag 1 s, each window fitted until it gains less than 1 over 50
Adam steps (at most 2000; design E7), the evidence every 0.5 s, the past
summarized (S4). One window spans each silence (E5). Readout (E4): the
leaky decision of O10, τ 2.5 s, and the window terms beside it. Log odds on
this page leave out the structure prior, which adds under 2. One run per
stimulus.

The runs found a bug first: a source born more than one lag after the
start had its earlier grid points frozen at birth without ever entering an
evidence estimate, which put the restart hypotheses 600 to 1200 too high.
Fixed in `online.py` (commit ad75100) before these runs; with it, online
and batch agree on a short Cusack stimulus (one restart −83,423 online
against −83,419 batch).

## Results

Raw outputs are in `data/O3/` (B&M as `bm_s<silence>.txt` and
`bm_alone.txt`, Cusack et al. as `cu<semitones>_s<silence>.txt`); the
tables come from `tools/online/reset_summary.py`.

### Beauvois & Meddis

![Beauvois & Meddis](plots/O3-beauvois-meddis.svg)

Log odds of two streams over one: the evidence the test adds, and the
leaky decision at its end. The inducer adds exactly 0: both hypotheses
explain A tones alone the same way. Listeners are their fit,
2.14 + 6.1 exp(−S/3.84), segregation responses out of 10.

| Silence (s) | alone | 0 | 0.5 | 1 | 2 | 4 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Evidence over the test | 110.1 | 76.1 | 142.6 | 160.0 | 167.7 | 172.5 | 162.4 |
| Leaky decision at the end | 122.1 | 64.8 | 113.4 | 126.4 | 132.6 | 136.7 | 129.8 |
| Listeners, segregated of 10 | 0 | 8.2 | 7.5 | 6.8 | 5.7 | 4.3 | 2.9 |

- **The baseline is already wrong.** The test alone gives two streams by
  122, where listeners always heard one. This is the model's ABA streaming
  at about 6 semitones and a 90 ms rate (C2 gives two streams from
  3 semitones up).
- **The inducer acts in the wrong direction at 0 s.** Straight after it,
  the test favours two streams *less* (65 against 122). Listeners hear the
  test *most* segregated there.
- **No decay.** From 0.5 s on, the test gives a little more than the test
  alone (113 to 137) and does not fall with the silence. Listeners' bias
  falls from 7.5 to 2.9 over the same range.

### Cusack et al.

![Cusack et al.](plots/O3-cusack.svg)

The leaky decision through the second sequence, against the first
sequence at the same times from its first tone:

| | s after first tone | 0 | 0.5 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 st | first sequence | 0 | 3.6 | 7.7 | 17.9 | 25.7 | 30.6 |
| | after 1 s | 28.0 | 7.8 | 16.2 | 32.6 | 53.5 | 72.1 |
| | after 2 s | 19.3 | −13.7 | −11.8 | 20.8 | 47.4 | 39.5 |
| | after 5 s | 7.0 | −25.6 | −25.0 | −7.7 | 3.0 | 13.3 |
| 6 st | first sequence | 0 | 7.0 | 22.7 | 46.5 | 58.8 | 77.0 |
| | after 1 s | 54.7 | 49.4 | 68.1 | 79.4 | 79.2 | 83.8 |
| | after 2 s | 37.6 | 32.5 | 43.5 | 54.8 | 65.8 | 74.1 |
| | after 5 s | 11.4 | −15.9 | 6.5 | 15.8 | 42.2 | 59.0 |

Window terms (two over one) of the first 0.5 s of each sequence:

| | first sequence | after 1 s | after 2 s | after 5 s |
| --- | --- | --- | --- | --- |
| 4 st | +3.6 | −15.2 | −29.5 | −31.3 |
| 6 st | +7.0 | +4.7 | +1.7 | −25.3 |

After that first 0.5 s, the mean term per 0.5 s window also depends on
the gap at 4 st: +15.6, +10.7 and +3.8 after 1, 2 and 5 s, against +6.9
in the first sequence. At 6 st it stays close to the first sequence's:
+17.9, +15.8 and +13.9 against +17.7. `reset_summary.py` prints every
term.

- **The leak behaves as §3 said.** The decision carries 67% of its value
  across a 1 s gap and 45% across 2 s (6 st: 80.7 at the end of the first
  sequence, 54.7 and 37.6 after the gaps), so after 1 s the second sequence starts leaning to two streams,
  where listeners start at one.
- **The evidence does not simply continue: the first triplet after a gap
  pushes towards one stream,** more after longer gaps. At 4 st the later
  windows then build faster than a fresh sequence after 1 s and slower
  after 5 s. After 2 s at 4 st
  and 5 s at both, the decision dips below zero before rebuilding.
- **So the second sequence depends on the gap,** above the first sequence
  after 1 s, below it after 5 s. Listeners showed no Gap × Time
  interaction.

### Restart hypotheses

Gain of each restart hypothesis over the one that continues through the
silence, at the end of the test (log posterior differences):

| | B&M 0 | 0.5 | 1 | 2 | 4 | 8 | Cu 4 st 1 | 2 | 5 | Cu 6 st 1 | 2 | 5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| one restart | −56.7 | −61.1 | −29.8 | −8.2 | +12.7 | +19.6 | +5.0 | −35.0 | −34.2 | −21.2 | −21.3 | −28.0 |
| two restart | +35.9 | +48.6 | +62.4 | +75.6 | +88.9 | +101.9 | −19.8 | −2.1 | +21.7 | −22.0 | −5.1 | +21.7 |

The best hypothesis overall is two restart for every Beauvois & Meddis
silence, two for Cusack et al.'s 1 and 2 s gaps, and two restart after
5 s. The model itself prefers to start its sources afresh after a 5 s gap,
but not after 1 s, where listeners already reset.

## Why: the rhythm prior [inference, partly checked]

Each source's log rests between events share one mean and precision,
integrated out (App. A.2; `priors.NormalGammaMarginal`). A source that has
been regular for 10 s makes a long rest very unlikely, and each source
that crosses the gap pays for it once. Two streams have two sources that
cross a Cusack gap, one stream has one. Evaluated at the tones' nominal
times (not the fitted ones), this prior alone moves the two-over-one log
odds by −36, −51 and −70 for gaps of 1, 2 and 5 s against no gap, and
makes two restart better than two by +19, +36 and +58. That has the sign
and roughly the size of the drops in the first window after the gap and of
the restart gains above.

For Beauvois & Meddis the same calculation gives −33 at 0 s against the
test alone, close to what was measured (76 against 110, so −34): the A
source of **two** changes its rhythm from a tone every 90 ms to one every
180 ms. It does not explain the other silences (−42 to −16 predicted,
+33 to +62 measured); what does is not checked. Nor is how much of the
Cusack effects the frequency and level priors' fading memory adds.

So, as far as checked, the model's reset comes from rhythm, and it grows
with the gap's length, where listeners reset fully after 1 s.

## Against design §3

| §3 expected | Measured |
| --- | --- |
| The decision resets only by its leak | The leak is as computed, but the evidence also changes after a gap (the rhythm prior) |
| Cusack et al. wrong under the leak alone: a lean to two streams after 1 s that depends on the gap | Yes after 1 s, and after 2 s at 6 st; after 2 s at 4 st and after 5 s the first window after the gap overturns the lean (the rhythm prior, inferred) |
| B&M's inducer: no effect, or one set by the f0 prior's memory, decaying with the lengthscale | An effect at 0 s in the wrong direction (rhythm); no decay from 0.5 to 8 s |
| A restart hypothesis winning after 1 s would match Cusack et al. | It wins only after 5 s |

## Against listeners

The model gets both studies wrong. It hears two streams in a test
listeners hear as one, its inducer effect has the wrong sign and does not
decay, and its reset after a gap grows with the gap instead of being
complete after 1 s. The leak (τ 2.5 s, fixed in O10 before these data)
and the rhythm prior are the only things in it that act across a silence.

## Cost

On Cho's PC, two to three runs at a time without a thread cap: 326 s for
the test alone; 6.4 to 7.8 h for B&M up to 2 s, 11.5 h at 4 s, 23.0 h at
8 s; Cusack et al. 10.9 to 13.7 h at 1 and 2 s and 14.7 to 16.6 h at 5 s
(those two ran two at a time). Runs sharing a machine must split its cores
(`OMP_NUM_THREADS`): on the cloud CPU two uncapped runs were 20 to 100
times slower per window than one alone, and with two threads each as fast
as one alone. The prior's cost grows with the cube of the sound's length
(a Gaussian process over every 10 ms grid point), so a 10 s gap makes each
fit step after it about 8 times slower (0.26 s before, 2 s after,
measured on the cloud).

## Reproduce

    python tools/online/online_reset.py bm --silence 2 --restart
    python tools/online/online_reset.py bm --silence none
    python tools/online/online_reset.py cusack --semitones 6 --silence 1 --restart
    python tools/online/reset_summary.py results/data/O3/*.txt
    python tools/results_plots.py
