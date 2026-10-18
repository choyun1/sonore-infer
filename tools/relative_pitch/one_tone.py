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

The f0 and whistle-frequency prior is uniform on 3 to 33.19 ERB number, from
87.5 Hz, so no hypothesis can explain components below 87.5 Hz (Shepard's
tones have components down to 4.9 Hz; those from 20 Hz reach the
cochleagram). By default readings start at the lowest component above 87.5 Hz.

Prints one line per fit and, per tone, the log posterior of each hypothesis
relative to the best (0 is the best).

    python tools/relative_pitch/one_tone.py --steps-of-circle 1 --max-steps 300
    python tools/relative_pitch/one_tone.py --minor-third
"""

import argparse
import sys
import time

import torch

from sonore_infer.cochleagram import FFTCochleagram
from sonore_infer.erb import erb_to_freq
from sonore_infer.priors import F0_TRAJECTORY
from sonore_infer.scene import Harmonic, Scene, SceneTiming, Whistle, evaluate
from sonore_infer.stimuli import shepard as sh

TOTAL = sh.DURATION + 2 * sh.PADDING
TIMING = SceneTiming(fs=sh.FS, total_duration=TOTAL, ramp=sh.RAMP)
ABSENT_DB = 30.0
# the f0 and whistle-frequency prior is uniform on 3 to 33.19 ERB number:
# 87.5 Hz to 8.0 kHz; nothing in the model has a fundamental below 87.5 Hz
PRIOR_LOW_HZ = float(erb_to_freq(F0_TRAJECTORY.mean_range[0]))
DTYPE = torch.float64


def harmonic_start(source, f0, freqs, levels):
    """A harmonic source at ``f0`` whose harmonics take the stimulus's levels where it has a component."""
    found = []
    for n in range(1, source.n_harmonics + 1):
        match = abs(freqs - n * f0) < 0.01
        found.append(float(levels[match][0]) if match.any() else None)
    loudest = max(level for level in found if level is not None)
    spectrum = torch.tensor(
        [(level - loudest) if level is not None else -ABSENT_DB for level in found], dtype=DTYPE
    )
    return source.initial(
        TIMING,
        f0=f0,
        level_db=loudest,
        onset=sh.PADDING,
        duration=sh.DURATION,
        spectrum_db=spectrum,
        dtype=DTYPE,
    )


def hypotheses(step, components_per_octave, lowest, highest, whistle_floor):
    """(label, scene, starts) for every hypothesis fitted to this tone."""
    freqs, levels = sh.components(step, components_per_octave=components_per_octave)
    out = []
    for f0 in freqs[(freqs >= lowest) & (freqs <= highest)]:
        source = Harmonic("h0", int(sh.F_MAX // f0))
        out.append(
            (f"harmonic@{f0:.1f}", Scene((source,), TIMING), [harmonic_start(source, f0, freqs, levels)])
        )
    keep = (freqs >= PRIOR_LOW_HZ) & (levels >= whistle_floor)
    whistles = [Whistle(f"w{index}") for index in range(int(keep.sum()))]
    start = {}
    for whistle, f, level in zip(whistles, freqs[keep], levels[keep], strict=True):
        start |= whistle.initial(
            TIMING, freq=float(f), level_db=float(level), onset=sh.PADDING, duration=sh.DURATION, dtype=DTYPE
        )
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
        "--lowest", type=float, default=PRIOR_LOW_HZ, help="lowest component tried as f0 [Hz]"
    )
    parser.add_argument("--highest", type=float, default=1300.0, help="highest component tried as f0 [Hz]")
    parser.add_argument(
        "--whistle-floor", type=float, default=30.0, help="components below this level [dB] get no whistle"
    )
    parser.add_argument("--only", nargs="+", help="fit only hypotheses whose label starts with one of these")
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
    cochleagram = FFTCochleagram()
    print(
        f"Shepard tones, {'minor-third' if args.minor_third else 'octave'} spacing; up to {args.max_steps} "
        f"Adam steps, stop at a rise under {args.tolerance} over 1000; {args.variational_steps} variational "
        f"steps, {args.samples} samples, seed {args.seed}; fft-bass-gain cochleagram, sigma 10 dB"
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
            step, per_octave, args.lowest, args.highest, args.whistle_floor
        ):
            if args.only and not any(label.startswith(prefix) for prefix in args.only):
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
                f"{step:4d}  {label:16s}  {n_params:5d}  {result.structure_log_prior:7.2f}"
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
