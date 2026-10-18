"""How the model hears one Shepard tone: experiments P1 and P2 (docs/design/relative-pitch.md).

For each tone, these hypotheses (decision P4), each fitted from a start built
from the stimulus and scored by its variational evidence:

- ``harmonic@F``: one harmonic source with f0 started at component F [Hz],
  one reading per component from ``--lowest`` to ``--highest`` Hz. It has
  every harmonic up to 5 kHz; the stimulus's components start at their own
  levels and the harmonics it lacks 30 dB below the loudest (as in
  ``tools/bass/compare_two_notes.py``).
- ``whistles``: one whistle per component at or above ``--whistle-floor`` dB
  and inside the frequency prior.
- ``harmonic@F+whistle@G`` (with ``--mixed``): the harmonic reading at F plus
  a whistle at the component an octave below it, G = F/2, the structure the
  scene search found best on the octave tone 1 (``results/P1-shepard-one-tone.md``).
- ``chord×K`` (with ``--chord``): K harmonic sources, one at each of the K
  lowest readings, K being the components per octave, so each source takes
  one octave-spaced series of components. With minor-third spacing, four
  Shepard-like tones a minor third apart: the chord listeners hear [Shepard §V].

The f0 and whistle-frequency prior is uniform on 3 to 33.19 ERB number, from
87.5 Hz, so no hypothesis can explain components below 87.5 Hz (Shepard's
tones have components down to 4.9 Hz; those from 20 Hz reach the
cochleagram). By default readings start at the lowest component above the
floor. ``--floor`` lowers it, and ``--sparse-spectrum`` lets each harmonic be
absent (design relative-pitch-prior P10 to P12). With a sparse spectrum the
start puts the Gaussian-process spectrum through the stimulus's levels,
interpolated in ERB number between components (held flat beyond them), and
starts the offsets at 0 dB for harmonics the stimulus has and at
``ABSENT_OFFSET_DB`` for the others, with the presence probability at the
fraction present. A second start has every harmonic present, as without a
sparse spectrum (its 30 dB-down harmonics lie between the two parts of the
offset prior, so a fit from one start rarely reaches the other's mode); the
fit keeps the better.

Prints one line per fit and, per tone, the log posterior of each hypothesis
relative to the best (0 is the best).

    python tools/relative_pitch/one_tone.py --steps-of-circle 1 --max-steps 300
    python tools/relative_pitch/one_tone.py --minor-third
"""

import argparse
import math
import sys
import time

import numpy as np
import torch

from sonore_infer.cochleagram import FFTCochleagram
from sonore_infer.erb import erb_to_freq, freq_to_erb_float
from sonore_infer.priors import ABSENT_OFFSET_DB, F0_TRAJECTORY
from sonore_infer.scene import Harmonic, Scene, SceneTiming, Whistle, evaluate
from sonore_infer.stimuli import shepard as sh

TOTAL = sh.DURATION + 2 * sh.PADDING
TIMING = SceneTiming(fs=sh.FS, total_duration=TOTAL, ramp=sh.RAMP)
ABSENT_DB = 30.0
# the f0 and whistle-frequency prior is uniform on 3 to 33.19 ERB number:
# 87.5 Hz to 8.0 kHz; nothing in the model has a fundamental below 87.5 Hz
PRIOR_LOW_HZ = float(erb_to_freq(F0_TRAJECTORY.mean_range[0]))
DTYPE = torch.float64


def harmonic_starts(source, f0, freqs, levels):
    """Starts for a harmonic source at ``f0``: one, or with a sparse spectrum two (see above)."""
    if not source.sparse_spectrum:
        return [harmonic_start(source, f0, freqs, levels)]
    return [harmonic_start(source, f0, freqs, levels), harmonic_start(source, f0, freqs, levels, dense=True)]


def harmonic_start(source, f0, freqs, levels, dense=False):
    """A harmonic source at ``f0`` whose harmonics take the stimulus's levels where it has a component;
    with ``dense``, a sparse source starts with every harmonic present."""
    found = []
    for n in range(1, source.n_harmonics + 1):
        match = abs(freqs - n * f0) < 0.01
        found.append(float(levels[match][0]) if match.any() else None)
    loudest = max(level for level in found if level is not None)
    present = [n for n, level in enumerate(found) if level is not None]
    if source.sparse_spectrum and not dense:
        # the GP spectrum through the present harmonics, interpolated in ERB number
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
        extra = {"presence_logit": 2.0} if source.sparse_spectrum else {}
    return source.initial(
        TIMING,
        f0=f0,
        level_db=loudest,
        onset=sh.PADDING,
        duration=sh.DURATION,
        spectrum_db=spectrum,
        dtype=DTYPE,
        **extra,
    )


def whistle_start(whistle, f, level):
    return whistle.initial(
        TIMING, freq=float(f), level_db=float(level), onset=sh.PADDING, duration=sh.DURATION, dtype=DTYPE
    )


def hypotheses(step, components_per_octave, lowest, highest, whistle_floor, floor, sparse, mixed, chord):
    """(label, scene, starts) for every hypothesis fitted to this tone."""
    freqs, levels = sh.components(step, components_per_octave=components_per_octave)
    out = []
    readings = freqs[(freqs >= lowest) & (freqs <= highest)]
    for f0 in readings:
        source = Harmonic("h0", int(sh.F_MAX // f0), sparse_spectrum=sparse, frequency_floor_hz=floor)
        out.append(
            (f"harmonic@{f0:.1f}", Scene((source,), TIMING), harmonic_starts(source, f0, freqs, levels))
        )
    if mixed:
        for f0 in readings:
            below = abs(freqs - f0 / 2) < 0.01
            if not below.any() or f0 / 2 < (floor or PRIOR_LOW_HZ):
                continue
            source = Harmonic("h0", int(sh.F_MAX // f0), sparse_spectrum=sparse, frequency_floor_hz=floor)
            whistle = Whistle("w0", frequency_floor_hz=floor)
            whistle_params = whistle_start(whistle, freqs[below][0], levels[below][0])
            starts = [start | whistle_params for start in harmonic_starts(source, f0, freqs, levels)]
            label = f"harmonic@{f0:.1f}+whistle@{f0 / 2:.1f}"
            out.append((label, Scene((source, whistle), TIMING), starts))
    if chord and len(readings) >= components_per_octave:
        sources, starts = [], [{}, {}] if sparse else [{}]
        for index, f0 in enumerate(readings[:components_per_octave]):
            source = Harmonic(
                f"h{index}", int(sh.F_MAX // f0), sparse_spectrum=sparse, frequency_floor_hz=floor
            )
            sources.append(source)
            for start, part in zip(starts, harmonic_starts(source, f0, freqs, levels), strict=True):
                start |= part
        label = f"chord×{components_per_octave}@" + "/".join(
            f"{f:.1f}" for f in readings[:components_per_octave]
        )
        out.append((label, Scene(tuple(sources), TIMING), starts))
    keep = (freqs >= (floor or PRIOR_LOW_HZ)) & (levels >= whistle_floor)
    whistles = [Whistle(f"w{index}", frequency_floor_hz=floor) for index in range(int(keep.sum()))]
    start = {}
    for whistle, f, level in zip(whistles, freqs[keep], levels[keep], strict=True):
        start |= whistle_start(whistle, f, level)
    out.append((f"whistles×{len(whistles)}", Scene(tuple(whistles), TIMING), [start]))
    return out


def status(prefix):
    return lambda message: print(f"status: {prefix}: {message}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--steps-of-circle", type=int, nargs="+", default=list(range(1, 11)), help="which tones (1 to 10)"
    )
    parser.add_argument(
        "--minor-third",
        action="store_true",
        help="components a minor third apart (experiment P2) instead of an octave",
    )
    parser.add_argument(
        "--floor",
        type=float,
        help="floor of the f0 and whistle-frequency prior [Hz] (design relative-pitch-prior P12); "
        "default Table A.2's 87.5 Hz",
    )
    parser.add_argument(
        "--sparse-spectrum",
        action="store_true",
        help="harmonics may be absent (design relative-pitch-prior P10)",
    )
    parser.add_argument(
        "--mixed", action="store_true", help="also fit each reading plus a whistle an octave below it"
    )
    parser.add_argument(
        "--chord", action="store_true", help="also fit one harmonic source per component of an octave"
    )
    parser.add_argument("--lowest", type=float, help="lowest component tried as f0 [Hz]; default the floor")
    parser.add_argument("--highest", type=float, default=1300.0, help="highest component tried as f0 [Hz]")
    parser.add_argument(
        "--whistle-floor", type=float, default=30.0, help="components below this level [dB] get no whistle"
    )
    parser.add_argument("--only", nargs="+", help="fit only hypotheses whose label starts with one of these")
    parser.add_argument("--exact", action="store_true", help="--only labels must match exactly")
    parser.add_argument("--list", action="store_true", help="print each tone's hypothesis labels and stop")
    parser.add_argument("--max-steps", type=int, default=15000, help="Adam steps at most")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
        help="stop once the log joint rises by less than this over 1000 steps",
    )
    parser.add_argument("--variational-steps", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    per_octave = 4 if args.minor_third else 1
    lowest = args.lowest if args.lowest is not None else (args.floor or PRIOR_LOW_HZ)
    cochleagram = FFTCochleagram()
    print(
        f"Shepard tones, {'minor-third' if args.minor_third else 'octave'} spacing; up to {args.max_steps} "
        f"Adam steps, stop at a rise under {args.tolerance} over 1000; {args.variational_steps} variational "
        f"steps, {args.samples} samples, seed {args.seed}; fft-bass-gain cochleagram, sigma 10 dB; "
        f"floor {args.floor or PRIOR_LOW_HZ:.1f} Hz, {'sparse' if args.sparse_spectrum else 'GP'} spectrum"
    )
    print(
        "step  hypothesis  n_params  structure  log_joint_at_mode  laplace"
        "  elbo  vi_importance  vi_ess  seconds"
    )
    for step in args.steps_of_circle:
        sound = sh.single_tone(step, components_per_octave=per_octave)
        observed = cochleagram(torch.tensor(sound.data[:, 0], dtype=DTYPE))
        posteriors = {}
        for label, scene, starts in hypotheses(
            step,
            per_octave,
            lowest,
            args.highest,
            args.whistle_floor,
            args.floor,
            args.sparse_spectrum,
            args.mixed,
            args.chord,
        ):
            if args.only and not any(
                label == prefix if args.exact else label.startswith(prefix) for prefix in args.only
            ):
                continue
            if args.list:
                print(f"{step} {label}")
                continue
            begin = time.perf_counter()
            result = evaluate(
                scene,
                starts,
                observed,
                cochleagram,
                steps=args.max_steps,
                tolerance=args.tolerance,
                n_samples=args.samples,
                generator=torch.Generator().manual_seed(args.seed),
                variational_steps=args.variational_steps,
                report=status(f"step {step} {label}"),
            )
            e, v = result.evidence, result.variational
            n_params = sum(value.numel() for value in starts[0].values())
            print(
                f"{step:4d}  {label:34s}  {n_params:5d}  {result.structure_log_prior:7.2f}"
                f"  {e.log_joint_at_mode:9.1f}  {e.laplace:9.1f}"
                + (f"  {v.elbo:9.1f}  {v.importance:9.1f}  {v.effective_sample_size:5.0f}" if v else "")
                + f"  {time.perf_counter() - begin:7.1f}",
                flush=True,
            )
            posteriors[label] = result.log_posterior_variational if v else result.log_posterior_laplace
        if posteriors:
            best = max(posteriors.values())
            text = ", ".join(f"{label} {value - best:.1f}" for label, value in posteriors.items())
            print(f"step {step}: log posterior relative to the best: {text}", flush=True)


if __name__ == "__main__":
    main()
