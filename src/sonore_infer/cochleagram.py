"""A differentiable cochleagram: gammatone envelopes pooled into time windows, in dB (design D4, option B).

The filters are sonore's gammatones (``sonore.gammatone_filterbank`` with
``edges=False``), re-implemented here so the core does not import sonore;
``tests/test_cochleagram.py`` checks them, and the envelopes, against sonore
in float64.

The pipeline, for a waveform ``x``:

1. Zero-pad ``x`` at both ends, so the filters' ringing does not wrap
   around the circular FFT.
2. Filter by multiplication on the DFT grid and take each band's analytic
   signal in the same step; its magnitude is the Hilbert envelope.
3. Pool the squared envelope over Hann-weighted time windows, 25 ms long every
   10 ms, as in BASS [§2.2 of Cusimano et al., 2024].
4. Express each window's power as the level of the sinusoid that would give
   it, in dB re ``REFERENCE_RMS``, and clip it from below at ``floor_db``.

So a sinusoid at a channel's center frequency, at L dB, reads L dB in that
channel once its time windows are past the onset. The defaults follow BASS: 64
channels with half-ERB bandwidths, centers 20 to 9423 Hz equally spaced in
ERB number, and a 20 dB floor. These numbers are BASS's; the filters and the
pooling are sonore's and ours, so values are comparable with BASS's in kind
but not identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from numpy.typing import ArrayLike

# room.py, spectrotemporal.py and some tools import these from here too.
from sonore_infer.erb import erb_bandwidth, erb_to_freq, freq_to_erb

REFERENCE_RMS = 1e-6


def gammatone_response(
    freqs: ArrayLike, cfs: ArrayLike, order: int = 4, bandwidth_factor: float = 1.019
) -> np.ndarray:
    """Causal gammatone responses at ``freqs`` [Hz], shape ``(len(freqs), len(cfs))``.

    The exact Fourier transform of ``t**(order - 1) exp(-2 pi b t) cos(2 pi cf t)``
    with ``b = bandwidth_factor * ERB(cf)``, normalized to unit gain at
    ``cf``, as in sonore's ``Gammatone`` filter type.
    """
    freq_col = np.asarray(freqs, float)[:, None]
    cf_row = np.asarray(cfs, float)[None, :]
    b = bandwidth_factor * erb_bandwidth(cf_row)

    def gammatone(freq: np.ndarray) -> np.ndarray:
        return (b + 1j * (freq - cf_row)) ** -order + (b + 1j * (freq + cf_row)) ** -order

    return gammatone(freq_col) / np.abs(gammatone(cf_row))


def ringing_samples(
    cfs: np.ndarray, fs: float, order: int, bandwidth_factor: float, level_db: float = -60.0
) -> int:
    """Samples until the slowest filter's envelope ``t**(order-1) exp(-2 pi b t)`` stays below ``level_db``.

    The envelope is that of the gammatone's own impulse response, and it is
    measured from ``t = 0``, so the padding covers the whole response.
    """
    b = bandwidth_factor * erb_bandwidth(np.min(cfs))
    t = np.arange(1, int(4 * fs)) / fs  # up to 4 s
    log_envelope = (order - 1) * np.log(t) - 2 * np.pi * b * t
    above = np.flatnonzero(log_envelope > log_envelope.max() + level_db / 20 * np.log(10))
    return int(above.max()) + 2


@dataclass(frozen=True)
class Cochleagram:
    """Settings of the cochleagram; call it on a waveform tensor.

    ``n_channels`` gammatones have centers from ``f_lo`` to ``f_hi`` [Hz]
    inclusive, equally spaced in ERB number, with bandwidth parameter
    ``bandwidth_factor * ERB(cf)``; 1.019 is a full ERB for order 4 (Slaney,
    1993), so the default 0.5 * 1.019 is BASS's half-ERB. ``win_dur`` (each
    time window's length) and ``hop_dur`` (the step between them) are in
    seconds, named as in sonore's views. ``pad`` is the zero-padding at each end in
    samples; ``None`` pads by the slowest filter's ringing to -60 dB.
    """

    fs: float = 20_000
    n_channels: int = 64
    f_lo: float = 20.0
    f_hi: float = 9423.0
    order: int = 4
    bandwidth_factor: float = 0.5 * 1.019
    win_dur: float = 0.025
    hop_dur: float = 0.010
    floor_db: float = 20.0
    pad: int | None = None

    @property
    def cfs(self) -> np.ndarray:
        """Center frequencies [Hz]."""
        return erb_to_freq(np.linspace(freq_to_erb(self.f_lo), freq_to_erb(self.f_hi), self.n_channels))

    @property
    def pad_samples(self) -> int:
        """Zero-padding at each end [samples]: ``pad``, or the slowest filter's ringing to -60 dB."""
        if self.pad is not None:
            return int(self.pad)
        return ringing_samples(self.cfs, self.fs, self.order, self.bandwidth_factor)

    @property
    def win_samples(self) -> int:
        """Length of each time window [samples]."""
        return round(self.win_dur * self.fs)

    @property
    def hop_samples(self) -> int:
        """Step between time windows [samples]."""
        return round(self.hop_dur * self.fs)

    def n_windows(self, n_samples: int) -> int:
        """Time windows for a waveform of ``n_samples``: every whole window from sample 0 on."""
        return (n_samples - self.win_samples) // self.hop_samples + 1

    def window_times(self, n_samples: int) -> np.ndarray:
        """Center time [s] of each time window."""
        starts = np.arange(self.n_windows(n_samples)) * self.hop_samples
        return (starts + (self.win_samples - 1) / 2) / self.fs

    @lru_cache(maxsize=8)  # noqa: B019 (the dataclass is frozen and small; the cache saves a recompute per step)
    def _analytic_transfer(self, n_padded: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Responses on the full DFT grid, times the analytic-signal weights
        (1 at DC and Nyquist, 2 at positive and 0 at negative frequencies),
        shape ``(n_channels, n_padded)``."""
        n_positive = n_padded // 2 + 1
        transfer = gammatone_response(
            np.fft.rfftfreq(n_padded, 1 / self.fs), self.cfs, self.order, self.bandwidth_factor
        )
        if n_padded % 2 == 0:
            transfer[-1] = transfer[-1].real  # irfft keeps only the real part at Nyquist
        weights = np.full(n_positive, 2.0)
        weights[0] = 1.0
        if n_padded % 2 == 0:
            weights[-1] = 1.0
        full = np.zeros((n_padded, self.n_channels), complex)
        full[:n_positive] = transfer * weights[:, None]
        complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        return torch.as_tensor(full.T, dtype=complex_dtype, device=device)

    def envelopes(self, waveform: torch.Tensor) -> torch.Tensor:
        """Hilbert envelope of every channel, shape ``(..., n_channels, n_samples)``.

        ``waveform`` has shape ``(..., n_samples)``; the padding is removed
        from the output.
        """
        pad = self.pad_samples
        n_samples = waveform.shape[-1]
        padded = torch.nn.functional.pad(waveform, (pad, pad))
        n_padded = padded.shape[-1]
        transfer = self._analytic_transfer(n_padded, waveform.dtype, waveform.device)
        spectrum = torch.fft.fft(padded)  # (..., n_padded)
        analytic = torch.fft.ifft(spectrum.unsqueeze(-2) * transfer)  # (..., n_channels, n_padded)
        return analytic[..., pad : pad + n_samples].abs()

    def window_power(self, envelopes: torch.Tensor) -> torch.Tensor:
        """Hann-weighted mean of the squared envelope in every time window, halved,
        so a sinusoid of RMS r gives r**2; shape ``(..., n_channels, n_windows)``."""
        hann = torch.hann_window(
            self.win_samples, periodic=False, dtype=envelopes.dtype, device=envelopes.device
        )
        hann = hann / hann.sum()
        windows = envelopes.pow(2).unfold(
            -1, self.win_samples, self.hop_samples
        )  # (..., C, n_windows, window)
        return (windows * hann).sum(-1) / 2

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Cochleagram in dB re ``REFERENCE_RMS``, floored, shape ``(..., n_channels, n_windows)``."""
        power = self.window_power(self.envelopes(waveform))
        tiny = torch.finfo(power.dtype).tiny
        level_db = 10 * torch.log10(power.clamp_min(tiny) / REFERENCE_RMS**2)
        return level_db.clamp_min(self.floor_db)


def gaussian_log_likelihood(
    observed: torch.Tensor, predicted: torch.Tensor, sigma: float = 10.0
) -> torch.Tensor:
    """Log density of ``observed`` under independent Gaussians of mean ``predicted`` and SD ``sigma`` [dB].

    BASS's likelihood [§2.2 of Cusimano et al., 2024], with BASS's sigma as
    the default. Summed over the last two dimensions (channels and time windows).
    """
    residual = (observed - predicted) / sigma
    log_density = -0.5 * residual**2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)
    return log_density.sum(dim=(-2, -1))


@dataclass(frozen=True)
class FFTCochleagram(Cochleagram):
    """The cochleagram BASS used: a short-time spectrum pooled into gammatone-shaped channels.

    Cusimano et al. (2024, App. A.5) compute their cochleagram with the FFT
    approximation of Ellis (2009): a spectrogram with ``win_dur``-long windows
    every ``hop_dur``, whose magnitudes are summed with weights shaped like the
    gammatones' magnitude responses. Each channel's resolution is then limited
    by the window as well as by the gammatone, so channels whose gammatone is
    narrower than the window's main lobe (low center frequencies) are wider
    than in :class:`Cochleagram`.

    Details follow Ellis's ``gammatonegram``: the FFT is ``fft_size`` points
    (the next power of two above twice the window); magnitudes, not powers,
    are pooled. Here the window is the same Hann window as
    :class:`Cochleagram` and the weights are :func:`gammatone_response`'s
    magnitudes. No zero-padding is needed: every time window is whole.

    By default (``bass_gain=True``) the channels are not calibrated: as in
    BASS's ``gammatonegram``, the pooled magnitudes are divided by
    ``fft_size`` and read in dB re ``REFERENCE_RMS`` directly, so a sinusoid
    reads below its level, by more at low center frequencies (about 13 dB at
    100 Hz, 4 dB at 3 kHz for the defaults), and nearer the floor. With
    ``bass_gain=False`` each channel is calibrated instead, so a sinusoid at
    its center frequency reads its level, as in :class:`Cochleagram`.
    """

    bass_gain: bool = True

    @property
    def fft_size(self) -> int:
        """FFT length: the power of two at least twice the window length."""
        return 2 ** int(np.ceil(np.log2(2 * self.win_samples)))

    @lru_cache(maxsize=8)  # noqa: B019 (frozen dataclass)
    def _weights(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Channel weights over the FFT bins, shape ``(n_channels, fft_size // 2 + 1)``, calibrated."""
        freqs = np.fft.rfftfreq(self.fft_size, 1 / self.fs)
        weights = np.abs(gammatone_response(freqs, self.cfs, self.order, self.bandwidth_factor)).T
        if self.bass_gain:
            return torch.as_tensor(weights / self.fft_size, dtype=dtype, device=device)
        hann = np.hanning(self.win_samples)
        t = np.arange(self.win_samples) / self.fs
        gains = np.empty(self.n_channels)
        for channel, cf in enumerate(self.cfs):
            tone = np.sqrt(2) * np.cos(2 * np.pi * cf * t)  # RMS 1
            gains[channel] = weights[channel] @ np.abs(np.fft.rfft(hann * tone, self.fft_size))
        return torch.as_tensor(weights / gains[:, None], dtype=dtype, device=device)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Cochleagram in dB re ``REFERENCE_RMS``, floored, shape ``(..., n_channels, n_windows)``."""
        hann = torch.hann_window(
            self.win_samples, periodic=False, dtype=waveform.dtype, device=waveform.device
        )
        windows = waveform.unfold(-1, self.win_samples, self.hop_samples) * hann  # (..., n_windows, window)
        spectrum = torch.fft.rfft(windows, n=self.fft_size)
        tiny = torch.finfo(waveform.dtype).tiny
        magnitude = (spectrum.real**2 + spectrum.imag**2 + tiny).sqrt()  # smooth at 0, for gradients
        rms = magnitude @ self._weights(waveform.dtype, waveform.device).T  # (..., n_windows, C)
        level_db = 20 * torch.log10(rms.clamp_min(tiny) / REFERENCE_RMS)
        return level_db.clamp_min(self.floor_db).transpose(-1, -2)
