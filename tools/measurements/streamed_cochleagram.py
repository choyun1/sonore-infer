"""Our cochleagram computed one 10 ms block at a time by sonore's streams, against :class:`Cochleagram`.

sonore 0.10.0 runs a bank of causal gammatones without edge filters as
recursive (IIR) filters, a block of samples at a time, with each band's
envelope the magnitude of its complex output (no Hilbert transform). This
script feeds those envelopes to a ``SlidingWindowStream`` laid out as our
:class:`Cochleagram` lays out its time windows (whole windows from sample
0, symmetric Hann, half the mean squared envelope, dB re ``REFERENCE_RMS``,
floored), and prints, for a few stimuli:

- the dB difference from :class:`Cochleagram` (FFT filtering and a Hilbert
  envelope over the whole sound), over the cells where either is above the
  floor, and the channels where it exceeds 0.1 dB;
- whether 10 ms blocks give the same numbers as the whole sound as one block
  (sonore's contract: the stream equals its batch form);
- the time per 10 ms block for 64 channels at 20 kHz, after the first five.

Needs sonore 0.10.0 or later (the ``stimuli`` extra).

    python tools/measurements/streamed_cochleagram.py
"""

import time

import numpy as np
import sonore as so
import torch

from sonore_infer.cochleagram import REFERENCE_RMS, Cochleagram, freq_to_erb
from sonore_infer.stimuli import buildup, mistuned_harmonic, shepard, two_notes

BLOCK = 200  # 10 ms at 20 kHz


def gammatone_bank(cochleagram):
    """sonore's bare gammatone bank with the cochleagram's centers and bandwidths
    (as in tests/test_cochleagram.py: the outer knots only fix the spacing)."""
    erbs = np.linspace(freq_to_erb(cochleagram.f_lo), freq_to_erb(cochleagram.f_hi), cochleagram.n_channels)
    step = erbs[1] - erbs[0]
    knots = np.concatenate([[erbs[0] - step], erbs, [erbs[-1] + step]])
    return so.gammatone_filterbank(
        centers=so.erb_to_freq(knots),
        order=cochleagram.order,
        bandwidth_factor=cochleagram.bandwidth_factor,
        edges=False,
    )


class CochleagramPooling(so.SlidingWindowStream):
    """Envelopes pooled into :class:`Cochleagram`'s time windows, a block at a time."""

    block_type = so.TemporalEnvelope

    def __init__(self, cochleagram):
        self.cochleagram = cochleagram
        hann = np.hanning(cochleagram.win_samples)  # symmetric, as torch.hann_window(periodic=False)
        self.weights = hann / hann.sum()

    def window_layout(self, fs):
        return self.cochleagram.win_samples, self.cochleagram.hop_samples, 0

    def window_count(self, n_samples, fs):
        return max(self.cochleagram.n_windows(n_samples), 0)

    def time_windows(self, samples, fs, first, count, n_samples):
        length, hop = self.cochleagram.win_samples, self.cochleagram.hop_samples
        squared = samples[..., 0] ** 2  # (n, channels): one audio channel
        power = np.zeros((count, squared.shape[1]))
        for k in range(count):
            power[k] = self.weights @ squared[k * hop : k * hop + length] / 2
        level_db = 10 * np.log10(np.maximum(power, np.finfo(float).tiny) / REFERENCE_RMS**2)
        return np.maximum(level_db, self.cochleagram.floor_db).T  # (channels, count)


def streamed(bank, pooling, sound, block):
    envelopes = (part.envelopes() for part in so.stream(bank, so.blocks(sound, block)))
    return np.concatenate(list(so.stream(pooling, envelopes)), axis=1)


def main():
    cochleagram = Cochleagram()
    bank, pooling = gammatone_bank(cochleagram), CochleagramPooling(cochleagram)
    cases = {
        "buildup, 6 triplets, 8 st": buildup.buildup_stimulus(6, 8.0),
        "two notes, first of set": two_notes.two_notes_set()[0][1],
        "mistuned harmonic, first": mistuned_harmonic.mistuned_harmonic_set()[0][1],
        "Shepard pair 1-4": shepard.pair(1, 4),
    }
    print(f"sonore {so.__version__}; {cochleagram.n_channels} channels at {cochleagram.fs:g} Hz")
    print(f"dB differences from Cochleagram over cells above the {cochleagram.floor_db:g} dB floor in either")
    header = f"{'stimulus':28s} {'cells':>6s} {'rms':>6s} {'p99':>6s} {'max':>6s}"
    print(f"{header}  10 ms = one block  channels > 0.1 dB")
    for name, sound in cases.items():
        with torch.no_grad():
            ours = cochleagram(torch.tensor(sound.data[:, 0])).numpy()
        parts = streamed(bank, pooling, sound, BLOCK)
        whole = streamed(bank, pooling, sound, len(sound))
        assert parts.shape == ours.shape
        above = (ours > cochleagram.floor_db) | (parts > cochleagram.floor_db)
        diff = np.abs(parts - ours)
        worst = [
            f"{cf:.0f} Hz: {d:.2f}" for cf, d in zip(cochleagram.cfs, diff.max(1), strict=True) if d > 0.1
        ]
        cells = diff[above]
        print(
            f"{name:28s} {cells.size:6d} {np.sqrt(np.mean(cells**2)):6.3f} {np.quantile(cells, 0.99):6.3f} "
            f"{cells.max():6.3f}  {str(np.array_equal(parts, whole)):17s} {', '.join(worst) or 'none'}"
        )

    sound = cases["buildup, 6 triplets, 8 st"]
    bank_state, pooling_state = bank.start(cochleagram.fs), pooling.start(cochleagram.fs)
    seconds = []
    for block in so.blocks(sound, BLOCK):
        start = time.perf_counter()
        bank_state, part = bank.step(bank_state, block)
        pooling_state, _ = pooling.step(pooling_state, part.envelopes())
        seconds.append(time.perf_counter() - start)
    ms = 1e3 * np.array(seconds[5:])
    print(
        f"per 10 ms block: median {np.median(ms):.2f} ms, "
        f"95th percentile {np.quantile(ms, 0.95):.2f} ms ({ms.size} blocks)"
    )


if __name__ == "__main__":
    main()
