"""How far one ABA hypothesis's MAP fit has got after a given number of Adam steps.

Fits one hypothesis of one stimulus (as ``compare_aba.py`` builds it, GP
sigma and lengthscale inferred) for ``--steps`` steps and prints the best log
joint seen by each checkpoint:

    python tools/grammar/aba_fit_steps.py --semitones 6 --interval 0.1 --hypothesis one --steps 15000
"""

import argparse
import time

import compare_aba
import torch

from sonore_infer.cochleagram import FFTCochleagram
from sonore_infer.fit import fit

CHECKPOINTS = (300, 1000, 2000, 3000, 5000, 7500, 10000, 15000)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--semitones", type=float, required=True)
    parser.add_argument("--interval", type=float, required=True, help="onset-to-onset [s]")
    parser.add_argument("--hypothesis", choices=["one", "two"], required=True)
    parser.add_argument("--steps", type=int, default=15000)
    args = parser.parse_args()
    cochleagram = FFTCochleagram()
    sound = compare_aba.ab.bistability_stimulus(args.interval, args.semitones, lead=compare_aba.LEAD)
    observed = cochleagram(torch.tensor(sound.data[:, 0], dtype=compare_aba.DTYPE))
    hypotheses = compare_aba.hypotheses(args.semitones, args.interval, infer_kernel=True)
    _, scene, starts = next(h for h in hypotheses if h[0] == args.hypothesis)
    begin = time.perf_counter()
    result = fit(
        scene.render,
        starts[0],
        observed,
        cochleagram,
        learning_rates=scene.learning_rates(),
        steps=args.steps,
        log_prior=scene.log_prior,
    )
    best, report = float("-inf"), []
    for step, value in enumerate(result.history):
        best = max(best, value)
        if step in CHECKPOINTS or step == len(result.history) - 1:
            report.append(f"{step}: {best:.1f}")
    print(
        f"{args.semitones:g} st {1000 * args.interval:g} ms {args.hypothesis}  "
        + "  ".join(report)
        + f"  ({time.perf_counter() - begin:.0f} s)"
    )


if __name__ == "__main__":
    main()
