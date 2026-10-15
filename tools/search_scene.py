"""Search for a sound's best hypothesis without being given its structure (design C9, C10).

Runs :func:`sonore_infer.search.search` on one stimulus of milestone (b)
or the ABA map and prints the detector's onset groups, every hypothesis fitted
(stage, sound length, Laplace log posterior, structure), the finalists'
variational log posteriors, and the number of fits and the time.

Stimuli are named ``aba:<semitones>:<interval s>``, ``two:<interval>:<asynchrony s>``
(``tritone``, ``tritone_different_spectra``, ``just_fifth``, ``octave``) or
``control:<f0>``:

    python tools/search_scene.py two:tritone:0.04 --variational-steps 2000
    python tools/search_scene.py aba:6:0.1
"""

import argparse
import sys

import torch

from sonore_infer.cochleagram import FFTCochleagram
from sonore_infer.search import describe, search
from sonore_infer.stimuli import bistability as ab
from sonore_infer.stimuli import two_notes as tn

DTYPE = torch.float64
LEAD = 0.05  # as tools/compare_aba.py


def stimulus(name):
    """(waveform, ramp) of a named stimulus."""
    kind, *args = name.split(":")
    if kind == "aba":
        sound = ab.bistability_stimulus(float(args[1]), float(args[0]), lead=LEAD)
        ramp = ab.RAMP
    elif kind == "two":
        sound = tn.two_notes_stimulus(args[0], float(args[1]))
        ramp = tn.RAMP
    elif kind == "control":
        sound = tn.single_note_control(float(args[0]))
        ramp = tn.RAMP
    else:
        raise SystemExit(f"unknown stimulus {name!r}")
    return torch.tensor(sound.data[:, 0], dtype=DTYPE), ramp


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stimulus")
    parser.add_argument(
        "--steps", type=int, default=15000, help="Adam steps per fit, a maximum with --tolerance"
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.5, help="stop a fit at a rise under this over 1000 steps"
    )
    parser.add_argument("--variational-steps", type=int, default=2000)
    parser.add_argument("--keep", type=int, default=5, help="finalists given variational evidence")
    parser.add_argument("--max-cleanup", type=int, default=8)
    parser.add_argument("--infer-kernel", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0: torch's default)")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    waveform, ramp = stimulus(args.stimulus)
    generator = torch.Generator().manual_seed(args.seed)
    result = search(
        waveform,
        FFTCochleagram(),
        ramp,
        steps=args.steps,
        tolerance=args.tolerance or None,
        keep=args.keep,
        max_cleanup=args.max_cleanup,
        variational_steps=args.variational_steps,
        infer_kernel=args.infer_kernel,
        generator=generator,
        report=lambda message: print(f"status: {message}", file=sys.stderr, flush=True),
    )
    print(f"{args.stimulus}: steps {args.steps}, tolerance {args.tolerance}, seed {args.seed}")
    print(f"onset groups: {len(result.groups)}")
    for group in result.groups:
        options = " | ".join(describe(tuple(_single(c) for c in e)) for e in group.explanations)
        print(f"  {group.onset:.3f} s: {options}")
    print("stage                 end_s  fit_steps  laplace_posterior  structure")
    for stage in result.stages:
        for s in stage:
            if s.result is None:
                print(f"{s.stage:20s} {s.end:6.3f}  {'-':>9}  {'left out':>17}  {describe(s.plan)}")
                continue
            steps = len(s.result.fit.history)
            print(f"{s.stage:20s} {s.end:6.3f}  {steps:9d}  {s.screen:17.1f}  {describe(s.plan)}")
    print("final               laplace_posterior  variational_posterior  elbo_posterior  structure")
    for s in result.final:
        variational = s.result.variational
        vi = f"{s.result.log_posterior_variational:21.1f}" if variational else f"{'-':>21}"
        elbo = f"{s.result.log_posterior_elbo:14.1f}" if variational else f"{'-':>14}"
        print(f"{'':20s} {s.screen:17.1f}  {vi}  {elbo}  {describe(s.plan)}")
    print(f"best: {describe(result.best.plan)}")
    print(f"hypotheses fitted: {result.n_fits}, seconds: {result.seconds:.0f}")


def _single(candidate):
    from sonore_infer.search import SourcePlan

    return SourcePlan(candidate.kind, (candidate,))


if __name__ == "__main__":
    main()
