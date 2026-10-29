# Live input: what sonore 0.10.0's streams give this project

Status: **note for Cho, 2026-10-29**, written with the move from sonore
0.8.0 to 0.10.0. It proposes three decisions (L1-L3); nothing is accepted
yet.

Every number below is measured, with the script that printed it, or
labeled an estimate with its basis. Measurements are on the cloud CPU
(Intel Xeon, 2.8 GHz, 4 threads).

## Decisions at a glance

Recommendations; nothing is accepted yet. The reasons are in §5.

| Decision | Recommended |
|---|---|
| L1. A second check of the cochleagram | Yes: a test that our `Cochleagram` agrees with sonore's streamed gammatone envelopes, beside the existing FFT check |
| L2. Live input to `online.py` now | No: wait until an online run is within a few times real time; the front end is not what is slow |
| L3. sonore's F0 track as a proposal for new sources | Later, with online structure search (O1 (C)); noted here so it is not lost |

## 1. What sonore 0.10.0 adds

A *stream* computes a view one block of sound at a time, with three pure
functions (`start(fs)`, `step(state, block)`, `finish(state)`) and an
immutable state. Joining the parts gives exactly what the batch view
gives, for any way of cutting the sound into blocks (sonore's
`docs/design/streaming.md`, accepted 2026-10-29). The views that stream:
the STFT, the cepstrum and MFCCs (with a fixed floor), a bank of causal
gammatones run as recursive (IIR) filters with pooled envelopes (a
cochleagram), the causal modulation spectrogram, and the F0 track (with a
fixed decision delay). `so.live_blocks(fs, size)` reads blocks from a
sound card (the `live` extra, python-sounddevice). sonore's design names
this project's online roadmap step 2 (`online.md` §9) as one of its two
reasons.

Streaming is not a speed-up for whole files: it buys latency. Each time
window comes out as soon as the samples it needs have arrived.

## 2. What the upgrade changes now: nothing in our results

Every stimulus set this repository makes (two notes, bistability,
buildup, mistuned harmonic, asynchronous onsets, all 120 Shepard single
tones and pairs) is bit-identical under 0.8.0 and 0.10.0. Of 20 `synth_ir`
rooms, 15 are bit-identical and 5 differ by at most 1e-17 of the peak
(0.9.0 reads the onset levels straight from the regressions' intercepts,
which its changelog says changes IRs by rounding only); the noises heard in
them differ by at most 3.3e-16 of the peak, and `band_rt60s` is unchanged
[measured: `tools/measurements/sonore_stimuli.py`, dump under each version,
then `compare`]. The model's core does not import sonore (design D2), so
no result page changes. The tests (248), ruff and mypy pass on 0.10.0.

0.9.0's HRIR gain fix, which changes the most sounds in sonore, does not
reach us: nothing here spatializes.

## 3. What the streams enable

### 3.1 The observed cochleagram, block by block (roadmap step 2)

Step 2 of the phone-app roadmap (`online.md` §9) is "the same loop fed
from a microphone instead of a file, with the cochleagram computed one
time window at a time as samples arrive". sonore now has that front end.
Our cochleagram's time windows are laid out differently from sonore's
`EnvelopePooling` (whole windows from sample 0, a symmetric Hann; sonore
centers its windows on multiples of the hop), but `SlidingWindowStream`
takes any layout, so a few lines give our own windows on sonore's streamed
envelopes. Against our `Cochleagram` (FFT filtering, Hilbert envelope of
the whole sound), on cells above the 20 dB floor
[measured: `tools/measurements/streamed_cochleagram.py`]:

| Stimulus | rms, dB | max, dB | Channels over 0.1 dB |
|---|---|---|---|
| Buildup, 6 triplets, 8 st | 0.000 | 0.000 | none |
| Two notes, first of the set | 0.000 | 0.003 | none |
| Mistuned harmonic, first of the set | 0.004 | 0.091 | none |
| Shepard pair 1-4 | 0.049 | 0.79 | 20 Hz (0.79), 35 Hz (0.25) |

The largest differences are in the two lowest channels, where a Shepard
tone's lowest components sit near the floor; the likelihood's sigma is 10
dB. 10 ms blocks give the same numbers as the whole sound as one block, as
sonore promises. One 10 ms block (64 channels at 20 kHz) takes a median
1.5 ms, so the front end runs at about a sixth of real time on this CPU.

What is slow is the inference. With 100 ms windows and S4, the 8 st
buildup case took 527 s of window time for its two hypotheses
[`results/O2-online-speed.md`, S4 table], for about 3 s of sound: roughly
175 times slower than real time [estimate: 527 s / 3 s]. Live input would
not change that: the predicted cochleagram must stay in PyTorch, because
the fit differentiates through it, and sonore's streams are NumPy.

Today `online.py` recomputes the observed cochleagram of the whole prefix
at every window (`run_online`, `observed = cochleagram(waveform[:n_samples])`).
That costs about 1 ms per window at 3 s [`online-realtime.md` §3], and on
the buildup stimulus each prefix's cochleagram equals the whole sound's
in the windows they share (0.000 dB to three decimals, checked once
for prefixes every 100 ms; not in a committed script). So a streamed
observation would change neither answers nor time noticeably now.

### 3.2 A second, independent check of our cochleagram

`tests/test_cochleagram.py` checks our filters and envelopes against
sonore's frequency-domain bank, which computes them the same way (FFT,
Hilbert). sonore's time-domain bank reaches the same envelopes by another
route (recursive filters, the complex output's magnitude), so agreement
with it is a check that does not share our method. That is decision L1.

### 3.3 Bottom-up proposals

`F0TrackStream` decides each time window's F0 a fixed delay after it, a
50 ms block at a time. A new F0 appearing in the track is a cheap signal
that a harmonic source may have been born, which is what online structure
search (O1 (C), roadmap step 1) needs from a proposal. The track assumes
one periodic source, so it would only propose, never decide; the
generative check stays (design C1). That is decision L3.

The causal modulation spectrogram also streams. It is one candidate
observation for the planned timing experiment (results index, "planned
experiments": phase-locking in the likelihood), but as an observation only;
the prediction side would need its own differentiable version.

### 3.4 The phone app

sonore's live page (`docs/live/` in sonore) runs its streams on a
browser's microphone under Pyodide. So the front end can already run on a
phone. PyTorch inference does not run there today. Roadmap step 3's second
option, the phone streaming audio to a computer, gains a variant: the
phone computes the cochleagram and sends that instead. That sends 64 channels × 100 windows per
second, about 25.6 kB/s in float32, against 40 kB/s for 16-bit audio at
20 kHz [arithmetic]: little saving, but no audio leaves the phone. The
open problems of §9 (an uncalibrated microphone, rooms and noise, power)
are untouched.

## 4. sonore-sketch

sonore-sketch, the visual front end for sonore's synthesis, pins
`sonore>=0.9,<0.10`, so it sees none of this until its pin moves. Two of
its tab designs (Erase spectrogram, Erase modulation) leave microphone
input "for a later PR"; sonore's streams and live page are the route. In
the browser that means a Web Audio callback calling `step` (sonore's
streaming design, "Streams: where the blocks come from"), since `live_blocks` needs sounddevice, which
does not run under Pyodide. Its Scatter events tab draws a modulation
spectrogram that follows the sound as it plays, which
`ModulationSpectrogramStream` could compute as it goes. None of this
touches sonore-infer; it belongs in a sonore-sketch thread.

## 5. Decisions

**L1. A second check of the cochleagram.** (A) Add a test comparing
`Cochleagram` with sonore's streamed time-domain envelopes, pooled into
our windows, within a tolerance set from §3.1 (the lowest channels need a
looser one). (B) Leave the checks as they are.
*Recommended: (A).* It is cheap, and it is the first check of the
cochleagram that does not share its method.

**L2. Live input to `online.py` now.** (A) Add a path that feeds
`run_online` a streamed observed cochleagram, from a file read in blocks or
from `live_blocks`. (B) Wait until an online run is within a few times
real time.
*Recommended: (B).* The front end takes a sixth of real time and the
inference about 175 times; live input now would show nothing about
perception and little about the app. The perception model comes first.

**L3. sonore's F0 track as a proposal for new sources.** (A) Design it
with online structure search (O1 (C)). (B) Not at all.
*Recommended: (A), later.* No work now.
