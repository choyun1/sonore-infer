"""Where a Shepard pair's harmonic up and down hypotheses differ, term by term.

For each pair, fits the two harmonic hypotheses of pair.py (from the 52 to
104 Hz reading, as P3) to their modes with the same Adam settings, then
splits each mode's log joint into the likelihood of the first tone, the
likelihood of the second tone (windows before and after the middle of the
gap) and the four prior terms (frequency, level, spectrum, timing). The
second tone's likelihood is also split into channels below the down
hypothesis's second f0, between it and the up hypothesis's, and above, the
same channels for both hypotheses. With ``--channels``, also prints the
second tone's log likelihood, observed level and predicted level [dB, mean
over its windows] in every channel. Prints up minus down for each part, to show which part makes the
model's log odds of up against down (P9) grow with the steps.

    python tools/relative_pitch/decompose.py --first 1 --seconds 5 6 7 --sparse-spectrum --floor 30
"""

import argparse
import math
import sys

import torch
from pair import DTYPE, ONSETS, TOTAL, hypotheses

from sonore_infer.cochleagram import FFTCochleagram
from sonore_infer.fit import fit
from sonore_infer.hearing import dominance_sigma
from sonore_infer.stimuli import shepard as sh

PARTS = [
    "like_tone1",
    "like_tone2_below",
    "like_tone2_between",
    "like_tone2_above",
    "frequency",
    "level",
    "spectrum",
    "timing",
]


def parts(scene, params, observed, cochleagram, bounds, sigma):
    """{part: value} of the log joint at ``params``, and the second tone channel by channel."""
    with torch.no_grad():
        predicted = cochleagram(scene.render(params))
        n_windows = observed.shape[-1]
        middle = int(n_windows * (ONSETS[1] - sh.GAP / 2) / TOTAL)
        # log density of each channel and window; sigma may be one SD per channel
        sigma = torch.as_tensor(sigma, dtype=observed.dtype).expand(observed.shape[-2], 1)
        density = -0.5 * ((observed - predicted) / sigma) ** 2 - sigma.log() - 0.5 * math.log(2 * math.pi)
        cfs = torch.tensor(cochleagram.cfs)
        bands = {
            "below": cfs < bounds[0],
            "between": (cfs >= bounds[0]) & (cfs < bounds[1]),
            "above": cfs >= bounds[1],
        }
        out = {"like_tone1": density[:, :middle].sum().item()}
        for band, channels in bands.items():
            out[f"like_tone2_{band}"] = density[channels, middle:].sum().item()
        terms = [torch.zeros((), dtype=DTYPE)] * 4
        for source in scene.sources:
            for index, term in enumerate(source.log_prior_terms(params, scene.timing)):
                terms[index] = terms[index] + term
        out |= dict(zip(PARTS[4:], (term.item() for term in terms), strict=True))
        channels = {
            "loglik": density[:, middle:].sum(dim=-1),
            "observed_db": observed[:, middle:].mean(dim=-1),
            "predicted_db": predicted[:, middle:].mean(dim=-1),
        }
    return out, channels


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--seconds", type=int, nargs="+", required=True)
    parser.add_argument("--sparse-spectrum", action="store_true")
    parser.add_argument("--floor", type=float)
    parser.add_argument("--whistle-floor", type=float, default=30.0)
    parser.add_argument("--max-steps", type=int, default=15000)
    parser.add_argument("--tolerance", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument(
        "--threshold-floor", action="store_true", help="floor channels at the threshold of hearing (P19)"
    )
    parser.add_argument(
        "--dominance", action="store_true", help="likelihood SD per channel by dominance (P20)"
    )
    parser.add_argument("--channels", action="store_true", help="print the second tone channel by channel")
    parser.add_argument("--only", choices=["up", "down"], help="fit only this direction")
    args = parser.parse_args()
    cochleagram = FFTCochleagram(threshold_floor=args.threshold_floor)
    sigma = dominance_sigma(cochleagram.cfs, args.sigma) if args.dominance else args.sigma
    print(f"threshold floor {args.threshold_floor}, dominance {args.dominance}, sigma {args.sigma:g} dB")
    print(f"first  second  steps  hypothesis  {'  '.join(PARTS)}  total")
    for second in args.seconds:
        sound = sh.pair(args.first, second)
        observed = cochleagram(torch.tensor(sound.data[:, 0], dtype=DTYPE))
        steps = sh.steps_clockwise(args.first, second)
        found = {}
        harmonics = hypotheses(
            args.first, second, None, args.sparse_spectrum, args.floor, args.whistle_floor, harmonic_only=True
        )
        bounds = sorted(float(label.split("->")[1]) for label, *_ in harmonics)
        for label, scene, start, _ in harmonics:
            if args.only and label.split()[1] != args.only:
                continue
            print(f"status: {args.first}->{second} {label}", file=sys.stderr, flush=True)
            result = fit(
                scene.render,
                start,
                observed,
                cochleagram,
                learning_rates=scene.learning_rates(),
                steps=args.max_steps,
                sigma=sigma,
                log_prior=scene.log_prior,
                patience=1000,
                tolerance=args.tolerance,
            )
            values, channels = parts(scene, result.params, observed, cochleagram, bounds, sigma)
            print(
                f"{args.first:5d}  {second:6d}  {steps:5d}  {label:28s}  "
                + "  ".join(f"{values[part]:10.1f}" for part in PARTS)
                + f"  {sum(values.values()):10.1f}",
                flush=True,
            )
            found[label.split()[1]] = values
            if args.channels:
                for name, row in channels.items():
                    print(
                        f"  {label.split()[1]} tone 2 {name}: " + " ".join(f"{v:.1f}" for v in row.tolist())
                    )
        if len(found) == 2:
            difference = {part: found["up"][part] - found["down"][part] for part in PARTS}
            print(
                f"{args.first:5d}  {second:6d}  {steps:5d}  {'up minus down':28s}  "
                + "  ".join(f"{difference[part]:+10.1f}" for part in PARTS)
                + f"  {sum(difference.values()):+10.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
