# O03. Reset of streaming after silence, online

> **Under review.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** first pass, one run per stimulus; Cusack et al.'s 10 s gaps
not run (see Cost). **Paper target:** none (the paper has no order effects); the
references are two listening studies.
**Code:** `src/sonore_infer/online.py`, `src/sonore_infer/stimuli/reset.py`,
`tools/online/online_reset.py`, `tools/online/reset_summary.py`. Design:
[order effects](../docs/design/order-effects.md), decisions E1–E7, step 3
of §7.

## Question

Shared background: [how the experiments work](README.md#how-the-experiments-work).

A batch model gives the same answer whatever order the sound arrives in;
the online model of [O01](O01-online-buildup.md) need not. Two studies
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
  500 ms ABA– triplets, 55 dB; silences 1, 2 and 5 s (10 s not run);
  the second sequence cut to its first 4 s.

Four hypotheses (design E3): one whistle with every tone (**one**), one
with the A tones and one with the B tones (**two**), and the same two with
their sources ending at the silence and new ones starting after it
(**one restart**, **two restart**). Each runs online as in O01 and O02: 100 ms
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

Raw outputs are in `data/O03/` (B&M as `bm_s<silence>.txt` and
`bm_alone.txt`, Cusack et al. as `cu<semitones>_s<silence>.txt`); the
tables come from `tools/online/reset_summary.py`.

### Beauvois & Meddis

![Beauvois & Meddis](plots/O03-beauvois-meddis.svg)

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
  at about 6 semitones and a 90 ms rate (C02 gives two streams from
  3 semitones up).
- **The inducer acts in the wrong direction at 0 s.** Straight after it,
  the test favours two streams *less* (65 against 122). Listeners hear the
  test *most* segregated there.
- **No decay.** From 0.5 s on, the test gives a little more than the test
  alone (113 to 137) and does not fall with the silence. Listeners' bias
  falls from 7.5 to 2.9 over the same range.

### Cusack et al.

![Cusack et al.](plots/O03-cusack.svg)

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

## Why: the rhythm prior

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

### Checked: the long rest left out (design E8, 2026-11-02)

[order-effects-reset.md](../docs/design/order-effects-reset.md) E8: one
Cusack et al. stimulus (4 st, first sequence cut to 2 s, 5 s gap, all four
hypotheses) run twice on the cloud, unchanged and with rests longer than
0.75 s left out of the rhythm prior (`tools/online/reset_rest_check.py`),
nothing else changed [measured: `data/O03/e8/`]:

| | unchanged | long rest left out |
| --- | --- | --- |
| Window term, first 0.5 s after the gap | −3.9 | +21.6 |
| Mean term, the 7 windows after that | +5.7 | +6.3 |
| Evidence over the 4 s second sequence | 35.7 | 65.5 |
| Leaky decision at its end | 23.1 | 36.2 |
| Two restart's gain over two | +3.7 | −32.7 |
| Best hypothesis | two restart | two |

(With a 2 s first sequence the rhythm is less settled than after 10 s, so
the unchanged drop is smaller than O03's −31.) Leaving the long rest out
moves the first window after the gap by +25.5 towards two streams and the
restart gain by −36, and turns the best hypothesis from restart to
continue. So the rhythm prior is the cause of both, as inferred. Without
it, the second sequence gives *more* two-stream evidence than a fresh one
(65.5 against 52.0 over the first 4 s of O03's 4 st first sequence): the
model then shows no reset at all. The rhythm prior is the only thing in
the model that pushes towards a reset, and it is too weak and too slow.

## With the reset rule (design E12, accepted 2026-11-02)

[order-effects-reset.md](../docs/design/order-effects-reset.md) E12 (C):
after a silence of 0.5 s or more, the leaky decision is emptied and the
sources end, so only the restart hypotheses go on and what follows is
heard afresh. It is a rule built in, a departure from the paper's model
and from the model's own evidence, which prefers to continue after gaps
under 5 s. No new runs: the restart hypotheses' window terms are already
in the outputs (`reset_summary.py`, row "restart"; row "gated" is E12 (B),
the decision emptied but the sources kept).

![Cusack et al. with the reset rule](plots/O03-cusack-reset-rule.svg)

| | s after first tone | 0.5 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- | --- |
| 4 st | first sequence | 3.6 | 7.7 | 17.9 | 25.7 | 30.6 |
| | reset rule, after 1 s | −1.1 | 9.8 | 4.1 | 45.2 | 44.2 |
| | after 2 s | −0.4 | 8.8 | 11.4 | 50.1 | 47.5 |
| | after 5 s | −0.7 | 8.5 | 18.3 | 26.8 | 30.2 |
| 6 st | first sequence | 7.0 | 22.7 | 46.5 | 58.8 | 77.0 |
| | reset rule, after 1 s | 8.8 | 21.1 | 44.8 | 57.8 | 78.7 |
| | after 2 s | 8.6 | 20.5 | 44.1 | 57.7 | 78.2 |
| | after 5 s | 7.5 | 21.2 | 44.1 | 59.8 | 70.1 |

At 6 st every gap now tracks the first sequence within 7, with no
Gap × Time interaction, as Cusack et al.'s listeners. At 4 st the 5 s gap
tracks it within 5, but after 1 and 2 s the curve runs 7 to 14 below at
2 s and 14 to 24 above at 3 and 4 s. The check below traces this to the
optimiser, not the rule.
The rule does nothing for Beauvois & Meddis (design E13): the inducer adds
nothing to the decision, so the test is heard as the test alone, and
their inducer effect stays out of reach.

### Checked: the 4 st stray is the optimiser (2026-11-05)

After the 1 and 2 s gaps, the one-stream restart fit stops for two
evidence steps at a point that is not a maximum: one Hessian eigenvalue is
below the floor (the `floored` column is 1 at 12.55 and 13.05 s in
`cu4_s1.txt`), which raises its Laplace term by about 7 per step. In the
next two steps the fit moves off that point and its terms fall by about 28
and 19 below the usual. The 5 s gap and 6 st show no such pair.

Rerun of the 1 s case, restart hypotheses only, second sequence cut to
3 s, on Cho's PC (`results/data/O03/check/`): once with the usual budget
(2000 Adam steps, patience 50), once with 6000 steps and patience 200.
The usual budget reproduces `cu4_s1.txt` within 0.1 in every window, so
the fresh checkout gives the same numbers. With the larger budget no
eigenvalue is floored, the one-stream terms are smooth (−10345 to −10342
from 1 s on), and the two-stream terms move by under 0.7. The gated
restart decision:

| s after first tone | 0.5 | 1 | 1.5 | 2 | 2.5 | 3 |
| --- | --- | --- | --- | --- | --- | --- |
| first sequence | 3.6 | 7.7 | | 17.9 | | 25.7 |
| after 1 s, usual budget | −1.0 | 9.7 | 5.6 | 4.0 | 31.5 | 45.2 |
| after 1 s, 6000 steps | −0.7 | 2.5 | 7.8 | 11.5 | 14.7 | 17.2 |
| after 5 s, usual budget | −0.7 | 8.5 | 13.3 | 18.3 | 23.7 | 26.8 |

With the larger budget the curve rises smoothly with no jump at 2.5 s,
but it now runs 5 to 9 below the first sequence from 1 s on, where the
usual budget ran up to 20 above. So the stray at 4 st was the one-stream
fit stalling, not the reset rule; with a fit that converges, the 1 s gap
is slightly slower to build up than a fresh sequence. One run each; the
first sequence and the other gaps were not rerun with the larger budget,
so the 5 to 9 is a comparison across budgets. Wall time: 20,425 s for the
usual budget and 28,175 s for 6000 steps, both on two threads.

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
measured on the cloud). Cusack et al.'s 10 s gaps were dropped for this
reason: two cloud runs got only partway through the first of four
hypotheses before the cloud machine was shut down, and on the PC they
would take about a day or more [estimate].

## Reproduce

    python tools/online/online_reset.py bm --silence 2 --restart
    python tools/online/online_reset.py bm --silence none
    python tools/online/online_reset.py cusack --semitones 6 --silence 1 --restart
    python tools/online/reset_rest_check.py --cutoff 0.75 cusack --semitones 4 --silence 5 \
        --first-duration 2 --restart
    python tools/online/online_reset.py cusack --semitones 4 --silence 1 --test-duration 3 --restart \
        --only one_restart two_restart [--steps 6000 --window-patience 200]
    python tools/online/reset_summary.py results/data/O03/*.txt results/data/O03/e8/*.txt
    python tools/results_plots.py
