"""Experiment P3: does the model hear a pair of Shepard tones go up or down?

Per pair (decision P4), enumerated hypotheses, each a source with two
events, one per tone:

- ``harmonic up`` and ``harmonic down``: one harmonic source whose first
  event starts at ``--reading``'s component of the first tone (default: the
  one between 52 and 104 Hz, experiment P5's reading) and whose second event
  starts at the nearest component of the second tone above it (up) or below
  it (down). Harmonics start at the first tone's levels where it has a
  component; with ``--sparse-spectrum`` the others start absent, as on P5,
  whose octave tones all preferred that start.
- ``whistles up`` and ``whistles down``: one whistle per component of the
  first tone at or above 30 dB and above the floor, each moving to the
  nearest component of the second tone above (up) or below (down) it.

With ``--octave k`` the harmonic hypotheses start the first tone's f0 at its
component k octaves from that reading (experiment P8), and the whistle
hypotheses, which do not depend on the reading, are left out.

Each hypothesis is fitted and scored as on P5. Its direction is read at its
fitted mode, not from its start: up when the second event's mean f0 (or,
for whistles, the mean of the whistles' frequency shifts, in ERB number) is
higher than the first's. P(up) (decision P5 (A)) is the posterior mass of
the hypotheses whose fitted direction is up, each weighted by its
variational evidence plus structure prior (the ELBO where importance
sampling fails).

    python tools/relative_pitch/pair.py --first 1 --seconds 2 3 --sparse-spectrum --floor 30
"""

import argparse
import math
import sys
import time

import numpy as np
import torch

from sonore_infer.cochleagram import FFTCochleagram
from sonore_infer.erb import freq_to_erb_float
from sonore_infer.hearing import dominance_sigma
from sonore_infer.priors import ABSENT_OFFSET_DB
from sonore_infer.scene import GRID_STEP, Harmonic, Scene, SceneTiming, Whistle, evaluate
from sonore_infer.stimuli import shepard as sh

TOTAL = 2 * sh.DURATION + sh.GAP + 2 * sh.PADDING
TIMING = SceneTiming(fs=sh.FS, total_duration=TOTAL, ramp=sh.RAMP)
ONSETS = [sh.PADDING, sh.PADDING + sh.DURATION + sh.GAP]
DURATIONS = [sh.DURATION, sh.DURATION]
ABSENT_DB = 30.0
DTYPE = torch.float64
READING_RANGE = (52.0, 104.0)


def default_reading(freqs):
    inside = freqs[(freqs >= READING_RANGE[0]) & (freqs < READING_RANGE[1])]
    return float(inside[0])


def nearest(freqs, f, up):
    """The component of ``freqs`` nearest to ``f`` strictly above it (``up``) or at or below it."""
    candidates = freqs[freqs > f * 1.0001] if up else freqs[freqs <= f * 1.0001]
    return float(candidates[0] if up else candidates[-1])


def harmonic_start(source, f0s, freqs, levels):
    """Two events at ``f0s``; harmonics at the first tone's levels where it has a component."""
    f0 = f0s[0]
    found = []
    for n in range(1, source.n_harmonics + 1):
        match = abs(freqs - n * f0) < 0.01 * n
        found.append(float(levels[match][0]) if match.any() else None)
    loudest = max(level for level in found if level is not None)
    present = [n for n, level in enumerate(found) if level is not None]
    extra = {}
    if source.sparse_spectrum:
        erbs = [freq_to_erb_float((n + 1) * f0) for n in range(source.n_harmonics)]
        spectrum = torch.tensor(
            np.interp(erbs, [erbs[n] for n in present], [found[n] - loudest for n in present]), dtype=DTYPE
        )
        offsets = torch.tensor(
            [0.0 if level is not None else ABSENT_OFFSET_DB for level in found], dtype=DTYPE
        )
        fraction = len(present) / source.n_harmonics
        extra = dict(spectrum_offset_db=offsets, presence_logit=math.log(fraction / (1 - fraction)))
    else:
        spectrum = torch.tensor(
            [(level - loudest) if level is not None else -ABSENT_DB for level in found], dtype=DTYPE
        )
    return source.initial(
        TIMING,
        f0=list(f0s),
        level_db=loudest,
        onset=ONSETS,
        duration=DURATIONS,
        spectrum_db=spectrum,
        dtype=DTYPE,
        **extra,
    )


def hypotheses(first, second, reading, sparse, floor, whistle_floor, octave=0, harmonic_only=False):
    """(label, scene, start, frequency parameter names) for every hypothesis of this pair."""
    freqs1, levels1 = sh.components(first)
    freqs2, _ = sh.components(second)
    reading = reading or default_reading(freqs1)
    if octave:
        # the first tone's component nearest the reading moved by ``octave`` octaves
        reading = float(freqs1[np.argmin(abs(np.log2(freqs1 / reading) - octave))])
    out = []
    for direction in ("up", "down"):
        f0s = (reading, nearest(freqs2, reading, direction == "up"))
        if floor and min(f0s) < floor:
            continue  # starts outside the f0 prior
        source = Harmonic(
            "h0",
            int(sh.F_MAX // min(f0s)),
            n_events=2,
            sparse_spectrum=sparse,
            frequency_floor_hz=floor,
        )
        label = f"harmonic {direction} {f0s[0]:.1f}->{f0s[1]:.1f}"
        out.append((label, Scene((source,), TIMING), harmonic_start(source, f0s, freqs1, levels1), ["h0.f0"]))
    if harmonic_only:
        return out
    lowest = floor or 87.5
    keep = (freqs1 >= lowest) & (levels1 >= whistle_floor)
    for direction in ("up", "down"):
        whistles, start = [], {}
        for index, (f, level) in enumerate(zip(freqs1[keep], levels1[keep], strict=True)):
            target = nearest(freqs2, f, direction == "up")
            if target < lowest:
                continue
            whistle = Whistle(f"w{index}", n_events=2, frequency_floor_hz=floor)
            whistles.append(whistle)
            start |= whistle.initial(
                TIMING,
                freq=[float(f), target],
                level_db=float(level),
                onset=ONSETS,
                duration=DURATIONS,
                dtype=DTYPE,
            )
        label = f"whistles×{len(whistles)} {direction}"
        out.append((label, Scene(tuple(whistles), TIMING), start, [f"{w.name}.freq" for w in whistles]))
    return out


def event_means(params, name):
    """Mean of a trajectory [ERB number] over each tone's span."""
    trajectory = params[f"{name}_mean"] + params[f"{name}_deviation"]
    t = torch.arange(len(trajectory), dtype=DTYPE) * GRID_STEP
    return [
        trajectory[(t >= onset) & (t < onset + duration)].mean().item()
        for onset, duration in zip(ONSETS, DURATIONS, strict=True)
    ]


def shift(params, names):
    """The second tone's mean frequency minus the first's [ERB number], averaged over sources."""
    shifts = []
    for name in names:
        first, second = event_means(params, name)
        shifts.append(second - first)
    return sum(shifts) / len(shifts)


def status(prefix):
    return lambda message: print(f"status: {prefix}: {message}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--first", type=int, default=1, help="the first tone (1 to 10)")
    parser.add_argument("--seconds", type=int, nargs="+", help="second tones; default every other tone")
    parser.add_argument("--reading", type=float, help="the first tone's starting f0 [Hz]; default 52-104 Hz")
    parser.add_argument(
        "--octave",
        type=int,
        default=0,
        help="start the first tone's f0 this many octaves from the reading; whistles are then not fitted",
    )
    parser.add_argument("--sparse-spectrum", action="store_true", help="design relative-pitch-prior P10")
    parser.add_argument("--floor", type=float, help="lowest f0 and whistle frequency [Hz] (P12)")
    parser.add_argument(
        "--whistle-floor", type=float, default=30.0, help="components below this [dB] get none"
    )
    parser.add_argument("--only", nargs="+", help="fit only hypotheses whose label starts with one of these")
    parser.add_argument(
        "--sigma",
        type=float,
        default=10.0,
        help="the likelihood's SD [dB] (design relative-pitch-listeners P18)",
    )
    parser.add_argument(
        "--threshold-floor",
        action="store_true",
        help="floor each channel at the threshold of hearing (design relative-pitch-dominance P19)",
    )
    parser.add_argument(
        "--dominance",
        action="store_true",
        help="likelihood SD per channel by spectral dominance (design relative-pitch-dominance P20)",
    )
    parser.add_argument("--max-steps", type=int, default=15000)
    parser.add_argument("--tolerance", type=float, default=0.5)
    parser.add_argument("--variational-steps", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    seconds = args.seconds or [s for s in range(1, sh.STEPS_PER_OCTAVE + 1) if s != args.first]
    cochleagram = FFTCochleagram(threshold_floor=args.threshold_floor)
    sigma = dominance_sigma(cochleagram.cfs, args.sigma) if args.dominance else args.sigma
    print(
        f"Shepard pairs from tone {args.first}; up to {args.max_steps} Adam steps, stop at a rise under "
        f"{args.tolerance} over 1000; {args.variational_steps} variational steps, {args.samples} samples, "
        f"seed {args.seed}; fft-bass-gain cochleagram, sigma {args.sigma:g} dB; "
        f"threshold floor {args.threshold_floor}, dominance {args.dominance}; "
        f"floor {args.floor or 87.5:g} Hz, {'sparse' if args.sparse_spectrum else 'GP'} spectrum"
    )
    print(
        "first  second  steps_clockwise  hypothesis  n_params  structure  log_joint_at_mode  laplace"
        "  elbo  vi_importance  vi_ess  shift_erb  seconds"
    )
    for second in seconds:
        sound = sh.pair(args.first, second)
        observed = cochleagram(torch.tensor(sound.data[:, 0], dtype=DTYPE))
        clockwise = sh.steps_clockwise(args.first, second)
        posteriors, shifts = {}, {}
        for label, scene, start, names in hypotheses(
            args.first,
            second,
            args.reading,
            args.sparse_spectrum,
            args.floor,
            args.whistle_floor,
            args.octave,
            harmonic_only=args.octave != 0,
        ):
            if args.only and not any(label.startswith(prefix) for prefix in args.only):
                continue
            begin = time.perf_counter()
            result = evaluate(
                scene,
                [start],
                observed,
                cochleagram,
                steps=args.max_steps,
                tolerance=args.tolerance,
                n_samples=args.samples,
                generator=torch.Generator().manual_seed(args.seed),
                variational_steps=args.variational_steps,
                sigma=sigma,
                report=status(f"{args.first}->{second} {label}"),
            )
            e, v = result.evidence, result.variational
            evidence = v.importance if math.isfinite(v.importance) else v.elbo
            posteriors[label] = evidence + result.structure_log_prior
            shifts[label] = shift(result.fit.params, names)
            n_params = sum(value.numel() for value in start.values())
            print(
                f"{args.first:5d}  {second:6d}  {clockwise:15d}  {label:34s}  {n_params:5d}"
                f"  {result.structure_log_prior:7.2f}  {e.log_joint_at_mode:9.1f}  {e.laplace:9.1f}"
                f"  {v.elbo:9.1f}  {v.importance:9.1f}  {v.effective_sample_size:5.0f}"
                f"  {shifts[label]:+8.3f}  {time.perf_counter() - begin:7.1f}",
                flush=True,
            )
        if posteriors:
            top = max(posteriors.values())
            weights = {label: math.exp(value - top) for label, value in posteriors.items()}
            total = sum(weights.values())
            p_up = sum(w for label, w in weights.items() if shifts[label] > 0) / total
            text = ", ".join(f"{label} {value - top:.1f}" for label, value in posteriors.items())
            print(f"{args.first}->{second}: relative to the best: {text}", flush=True)
            print(f"{args.first}->{second}: steps clockwise {clockwise}, P(up) {p_up:.3f}", flush=True)


if __name__ == "__main__":
    main()
