"""Candidate events read off an observed cochleagram, for the search (design C9 (A)).

The paper proposes candidate events with a segmentation network [App. B.2.1
of Cusimano et al., 2024]. Design C1 rules out a learned proposer, so this
module reads candidates off the cochleagram by hand:

- **Onsets** are time windows where the level rises across many channels at once:
  the sum over channels of each channel's rise from the previous window
  (half-wave rectified), at its local maxima above a threshold.
- **What starts at an onset** is the rise of each channel over the level
  just before the onset (:func:`rise_pattern`). Its peaks across channels
  are the candidate whistles; the f0 whose harmonics best explain those
  peaks (subharmonic summation) is a candidate harmonic source, offered with
  its octave above and below, since octave errors are local minima a fit
  cannot leave [paper §4.2.2]. Once the best f0's harmonics are taken out,
  the summation is repeated for a second f0 among what is left.
- **Continuations** are left out: a peak at a partial an earlier onset
  started and that is still sounding, no more than 3 dB louder, is beating
  or fluctuation (two notes a tritone apart beat enough to look like onsets).
  The 3 dB, and the 0.3 ERB within which two peaks count as one tone, are
  not measured: they were set when the detector was written (commit
  e9f5649), and no tool or decision records why.
- **Durations** run from the onset until the event's channels fall by
  ``drop_db`` from their peak, or another onset raises them again.

Levels are read as the level a sinusoid would need to reach the observed
peak in that channel, so they are calibrated whatever the cochleagram's
gain (see :func:`channel_gains`).

All of it is a starting point for fits, not an estimate: the search fits
every hypothesis it builds from these candidates and compares them by
evidence (design C10).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from sonore_infer.cochleagram import REFERENCE_RMS, Cochleagram
from sonore_infer.erb import erb_to_freq, freq_to_erb


@dataclass(frozen=True)
class Candidate:
    """One candidate event: a whistle at ``freq`` or a harmonic source with f0 ``freq``.

    ``spectrum_db`` gives a harmonic candidate's level per harmonic relative
    to ``level_db`` (``None`` for a whistle). ``onset_index`` numbers the
    onset the candidate starts at; candidates of the same onset are
    alternatives or companions (see :class:`OnsetGroup`). ``salience`` ranks
    candidates of one onset: the peak's rise [dB] for a whistle, the
    summed rise of the explained peaks for a harmonic source.
    """

    kind: str  # "whistle" or "harmonic"
    onset: float  # s
    duration: float  # s
    freq: float  # Hz
    level_db: float
    salience: float
    onset_index: int
    spectrum_db: tuple[float, ...] | None = None


@dataclass(frozen=True)
class OnsetGroup:
    """What starts at one onset, as alternative explanations.

    Each explanation is a tuple of candidates that together account for the
    onset's peaks: all its peaks as whistles, or one or two harmonic sources
    (with any peaks they leave unexplained as whistles). A hypothesis takes
    exactly one explanation per group.
    """

    onset: float  # s
    explanations: tuple[tuple[Candidate, ...], ...]


@lru_cache(maxsize=4)
def channel_gains(cochleagram: Cochleagram, dtype: torch.dtype = torch.float64) -> np.ndarray:
    """Per channel, the level [dB] of a sinusoid at the channel's center frequency
    minus what the channel reads for it; add it to a reading to get a level."""
    level_db = 70.0
    n = round(0.2 * cochleagram.fs)
    t = torch.arange(n, dtype=dtype) / cochleagram.fs
    amplitude = math.sqrt(2) * REFERENCE_RMS * 10 ** (level_db / 20)
    gains = np.empty(cochleagram.n_channels)
    for channel, cf in enumerate(cochleagram.cfs):
        reading = cochleagram(amplitude * torch.cos(2 * math.pi * float(cf) * t))
        gains[channel] = level_db - reading[channel, reading.shape[-1] // 2].item()
    return gains


def onset_windows(
    level: np.ndarray, threshold_db: float, min_gap: int, clear_db: float = 60.0, steep_db: float = 6.0
) -> list[int]:
    """Time windows where the summed rise across channels peaks above ``threshold_db``.

    ``level`` is the cochleagram above its floor, shape ``(channels, windows)``;
    a window's rise is measured from the window before. A peak below
    ``clear_db`` also needs two channels rising by ``steep_db`` or more:
    beats between the partials of two notes raise many channels a little at
    a time, a new note raises a few by a lot. Peaks closer than ``min_gap``
    windows keep only the larger.
    """
    rises = np.clip(np.diff(level, axis=1), 0.0, None)
    rises = np.concatenate([level[:, :1], rises], axis=1)
    rise = rises.sum(axis=0)
    steep = (rises >= steep_db).sum(axis=0)
    peaks = [
        window
        for window in range(level.shape[1])
        if rise[window] >= threshold_db
        and (rise[window] >= clear_db or steep[window] >= 2)
        and rise[window] >= rise[max(window - 1, 0)]
        and rise[window] > rise[min(window + 1, level.shape[1] - 1)]
    ]
    kept: list[int] = []
    for window in sorted(peaks, key=lambda f: -rise[f]):
        if all(abs(window - other) >= min_gap for other in kept):
            kept.append(window)
    return sorted(kept)


def rise_pattern(level: np.ndarray, window: int, span: int = 3) -> np.ndarray:
    """Each channel's rise [dB] at onset ``window``: its peak over the next ``span``
    windows above its level two windows before (zero if it falls)."""
    before = level[:, max(window - 2, 0)]
    after = level[:, window : window + span].max(axis=1)
    return np.clip(after - before, 0.0, None)


def spectral_peaks(pattern: np.ndarray, erbs: np.ndarray, min_db: float) -> list[tuple[float, float]]:
    """(ERB number, height) of each local maximum of ``pattern`` above ``min_db``,
    refined by a parabola through the maximum and its neighbors."""
    peaks = []
    for c in range(len(pattern)):
        left = pattern[c - 1] if c > 0 else -np.inf
        right = pattern[c + 1] if c + 1 < len(pattern) else -np.inf
        if pattern[c] < min_db or pattern[c] < left or pattern[c] <= right:
            continue
        offset = 0.0
        if 0 < c < len(pattern) - 1:
            curvature = left - 2 * pattern[c] + right
            if curvature < 0:
                offset = 0.5 * (left - right) / curvature
        step = erbs[1] - erbs[0]
        peaks.append((float(erbs[c] + offset * step), float(pattern[c])))
    return peaks


def subharmonic_summation(
    peaks: list[tuple[float, float]],
    f0_range: tuple[float, float],
    tolerance_erb: float,
    weight: float = 0.84,
    n_f0: int = 400,
) -> list[tuple[float, float, list[int]]]:
    """Every f0 in ``f0_range`` [Hz] (``n_f0`` steps, even in log f0), scored by the
    peaks within ``tolerance_erb`` of one of its harmonics: the sum of
    their heights times ``weight ** (n - 1)`` for harmonic number ``n``.

    Returns (f0, score, indices of the explained peaks), best first.
    """
    peak_freqs = erb_to_freq(np.array([erb for erb, _ in peaks]))
    peak_erbs = np.array([erb for erb, _ in peaks])
    heights = np.array([height for _, height in peaks])
    results = []
    for f0 in np.geomspace(*f0_range, n_f0):
        numbers = np.maximum(np.round(peak_freqs / f0), 1)
        distance = np.abs(freq_to_erb(numbers * f0) - peak_erbs)
        explained = np.flatnonzero(distance <= tolerance_erb)
        # a peak below the f0 is not one of its harmonics
        explained = explained[peak_freqs[explained] >= f0 * 0.97]
        if len(explained) == 0:
            continue
        score = float((heights[explained] * weight ** (numbers[explained] - 1)).sum())
        results.append((float(f0), score, explained.tolist()))
    return sorted(results, key=lambda result: -result[1])


def _duration(
    level: np.ndarray, channels: list[int], start: int, next_onsets: list[int], drop_db: float
) -> int:
    """Time windows from ``start`` until ``channels`` fall ``drop_db`` below their peak, or
    the next onset that raises them."""
    trace = level[channels].max(axis=0)
    end = level.shape[1]
    for onset in next_onsets:
        if onset > start and rise_pattern(level, onset)[channels].max() >= drop_db:
            end = onset
            break
    peak = start + int(np.argmax(trace[start:end]))
    for window in range(peak, end):
        if trace[window] < trace[peak] - drop_db:
            return window - start
    return end - start


def _channels_near(erbs: np.ndarray, erb: float) -> list[int]:
    return [int(np.argmin(np.abs(erbs - erb)))]


def _refine(level: np.ndarray, erbs: np.ndarray, window: int, erb: float, span: int = 3) -> float:
    """The ERB number of the level's maximum after ``window`` within one channel of ``erb``.

    A rise pattern's peak sits on the flank of a tone when a neighboring
    tone was still sounding before the onset; the level itself peaks at the
    tone.
    """
    pattern = level[:, window : window + span].max(axis=1)
    near = int(np.argmin(np.abs(erbs - erb)))
    channels = range(max(near - 1, 0), min(near + 2, len(erbs)))
    c = max(channels, key=lambda channel: pattern[channel])
    if 0 < c < len(erbs) - 1:
        left, middle, right = pattern[c - 1], pattern[c], pattern[c + 1]
        curvature = left - 2 * middle + right
        if curvature < 0:
            return float(erbs[c] + 0.5 * (left - right) / curvature * (erbs[1] - erbs[0]))
    return float(erbs[c])


def _best_f0(whistles: tuple[Candidate, ...], f0: float, explained: list[int]) -> float:
    """The f0 that best fits the explained peaks: their frequencies over their
    harmonic numbers, averaged in log."""
    members = [whistles[i] for i in explained]
    numbers = [max(round(member.freq / f0), 1) for member in members]
    return math.exp(sum(math.log(m.freq / n) for m, n in zip(members, numbers, strict=True)) / len(members))


def _harmonic(whistles: tuple[Candidate, ...], f0: float, explained: list[int], f_hi: float) -> Candidate:
    """A harmonic candidate at ``f0`` taking the explained peaks as its harmonics."""
    members = [whistles[i] for i in explained]
    level_db = max(member.level_db for member in members)
    # harmonics up to one above the highest explained peak
    highest = max(max(round(member.freq / f0), 1) for member in members)
    n_harmonics = max(min(highest + 1, int(f_hi // f0)), 1)
    spectrum = [-30.0] * n_harmonics
    for member in members:
        number = max(round(member.freq / f0), 1)
        if number <= n_harmonics:
            spectrum[number - 1] = max(spectrum[number - 1], member.level_db - level_db)
    return Candidate(
        kind="harmonic",
        onset=members[0].onset,
        duration=max(member.duration for member in members),
        freq=f0,
        level_db=level_db,
        salience=sum(member.salience for member in members),
        onset_index=members[0].onset_index,
        spectrum_db=tuple(spectrum),
    )


def _explanations(
    whistles: tuple[Candidate, ...], f0_range: tuple[float, float], tolerance_erb: float, f_hi: float
) -> list[tuple[Candidate, ...]]:
    """The whistles of one onset, and the harmonic explanations of them (see :func:`detect`)."""
    explanations: list[tuple[Candidate, ...]] = [whistles]
    peaks = [(float(freq_to_erb(w.freq)), w.salience) for w in whistles]
    scored = subharmonic_summation(peaks, f0_range, tolerance_erb)
    if not scored or len(scored[0][2]) < 2:
        return explanations
    f0, _, explained = scored[0]
    f0 = _best_f0(whistles, f0, explained)
    rest = [i for i in range(len(peaks)) if i not in explained]
    leftover = tuple(whistles[i] for i in rest)
    for factor in (1.0, 2.0, 0.5):
        if f0_range[0] / 2 <= f0 * factor <= f0_range[1] * 2:
            explanations.append((_harmonic(whistles, f0 * factor, explained, f_hi), *leftover))
    # a second f0: the best one, not near f0 or its octaves, that explains a
    # peak f0 leaves out (its own harmonics may coincide with f0's)
    for f0_b, _, explained_b in scored[1:]:
        related = any(abs(math.log2(f0_b / (f0 * factor))) < 0.05 for factor in (0.5, 1.0, 2.0))
        if related or len(explained_b) < 2 or not set(explained_b) - set(explained):
            continue
        unexplained = tuple(whistles[i] for i in rest if i not in explained_b)
        f0_b = _best_f0(whistles, f0_b, explained_b)
        first, second = _harmonic(whistles, f0, explained, f_hi), _harmonic(whistles, f0_b, explained_b, f_hi)
        explanations.append((first, second, *unexplained))
        break
    return explanations


def detect(
    observed: torch.Tensor,
    cochleagram: Cochleagram,
    n_samples: int,
    *,
    onset_threshold_db: float = 40.0,
    peak_min_db: float = 6.0,
    drop_db: float = 10.0,
    f0_range: tuple[float, float] = (60.0, 1000.0),
    tolerance_erb: float = 0.35,
    min_duration: float = 0.03,
) -> list[OnsetGroup]:
    """The onset groups of an observed cochleagram (shape ``(channels, windows)``, dB)
    of a sound of ``n_samples`` samples.

    ``onset_threshold_db`` is the summed rise across channels that counts as
    an onset; ``peak_min_db`` the rise a channel peak needs to be a
    candidate. A harmonic explanation is offered only when its f0 explains
    at least two peaks; its octave above and below are offered with it.
    """
    level = observed.detach().cpu().numpy() - cochleagram.floor_db
    times = cochleagram.window_times(n_samples)
    erbs = freq_to_erb(cochleagram.cfs)
    gains = channel_gains(cochleagram, observed.dtype)
    hop_dur = cochleagram.hop_dur
    windows = onset_windows(level, onset_threshold_db, min_gap=2)
    groups: list[OnsetGroup] = []
    sounding: list[Candidate] = []
    for index, window in enumerate(windows):
        onset = max(float(times[window]) - 0.5 * hop_dur, 0.0)
        whistles: list[Candidate] = []
        for erb, height in spectral_peaks(rise_pattern(level, window), erbs, peak_min_db):
            erb = _refine(level, erbs, window, erb)
            channels = _channels_near(erbs, erb)
            n_windows = _duration(level, channels, window, windows[index + 1 :], drop_db)
            reading = observed[channels[0], window : window + max(n_windows, 1)].max().item()
            candidate = Candidate(
                kind="whistle",
                onset=onset,
                duration=max(n_windows * hop_dur, min_duration),
                freq=float(erb_to_freq(erb)),
                level_db=reading + float(gains[channels[0]]),
                salience=height,
                onset_index=index,
            )
            # two rise peaks can refine to the same tone
            if all(abs(freq_to_erb(candidate.freq) - freq_to_erb(w.freq)) > 0.3 for w in whistles):
                whistles.append(candidate)
        # a peak at a partial that is still sounding, no louder than it was, is that
        # partial beating or fluctuating, not a new event
        sounding = [w for w in sounding if w.onset + w.duration > onset]
        whistles = [
            w
            for w in whistles
            if not any(
                abs(freq_to_erb(w.freq) - freq_to_erb(s.freq)) <= 0.3 and w.level_db <= s.level_db + 3.0
                for s in sounding
            )
        ]
        sounding += whistles
        if whistles:
            explanations = _explanations(tuple(whistles), f0_range, tolerance_erb, cochleagram.f_hi)
            groups.append(OnsetGroup(onset, tuple(explanations)))
    return groups
