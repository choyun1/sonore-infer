"""One stream or two for ABA sequences, by marginal likelihood (paper App. C.10; design C8).

The tones start 50 ms into the 3.1 s (``LEAD``). For each stimulus, two
hypotheses, each fitted from the stimulus's own tones (frequencies, onsets
and durations), as the paper did:

- ``one``: one whistle source with all 12 tones as its events;
- ``two``: one whistle source with the 8 A tones and one with the 4 B tones.

Prints one line per fit and, per stimulus, the log posterior odds of ``two``
over ``one`` (positive favours two streams) by each evidence estimate.

    python tools/compare_aba.py --semitones 6 --intervals 0.1 --variational-steps 2000
"""

import argparse
import time

import torch

from sonore_inference.cochleagram import FFTCochleagram
from sonore_inference.scene import Scene, SceneTiming, Whistle, evaluate
from sonore_inference.stimuli import bistability as ab

TIMING = SceneTiming(fs=ab.FS, total_duration=ab.TOTAL_DURATION, ramp=ab.RAMP)
DTYPE = torch.float64
# silence before the first tone, out of the 3.1 s: the onset prior starts at 0,
# and a fit cannot move an onset that starts exactly there
LEAD = 0.05


def hypotheses(semitones, onset_interval, infer_kernel=False):
    """(label, scene, starts) for the two hypotheses of one stimulus."""
    freqs = {"A": ab.A_FREQ, "B": ab.A_FREQ * 2 ** (semitones / 12)}
    tones = ab.tone_onsets(onset_interval, lead=LEAD)

    def whistle(name, selected):
        source = Whistle(name, n_events=len(selected), infer_kernel=infer_kernel)
        start = source.initial(
            TIMING,
            freq=[freqs[kind] for _, kind in selected],
            level_db=ab.TONE_LEVEL_DB,
            onset=[onset for onset, _ in selected],
            duration=[ab.TONE_DURATION] * len(selected),
            dtype=DTYPE,
        )
        return source, start

    one, one_start = whistle("w0", tones)
    a, a_start = whistle("w0", [tone for tone in tones if tone[1] == "A"])
    b, b_start = whistle("w1", [tone for tone in tones if tone[1] == "B"])
    return [
        ("one", Scene((one,), TIMING), [one_start]),
        ("two", Scene((a, b), TIMING), [a_start | b_start]),
    ]


def log_odds(results, estimate):
    """Log posterior odds of two streams over one by one evidence estimate."""
    two, one = (getattr(results[h], f"log_posterior_{estimate}") for h in ("two", "one"))
    return two - one


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--semitones", type=float, nargs="+", default=list(ab.SEMITONES))
    parser.add_argument(
        "--intervals", type=float, nargs="+", default=list(ab.ONSET_INTERVALS), help="onset-to-onset [s]"
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--variational-steps",
        type=int,
        default=0,
        help="also estimate the evidence by variational inference with this many steps (extra columns)",
    )
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument(
        "--seed", type=int, default=0, help="seed for the evidence's sampling (fits are deterministic)"
    )
    parser.add_argument(
        "--infer-kernel",
        action="store_true",
        help="infer each GP's sigma and lengthscale instead of fixing them at the medians (design C3 (B))",
    )
    args = parser.parse_args()
    cochleagram = FFTCochleagram()
    print(
        f"{args.steps} Adam steps, {args.samples} samples, seed {args.seed}, "
        "fft-bass-gain cochleagram, trajectories, all timing inferred"
        + (", GP sigma and lengthscale inferred" if args.infer_kernel else "")
    )
    print(
        "stimulus  hypothesis  n_params  structure  laplace  importance  ess  floored"
        "  log_joint_at_mode  seconds" + ("  elbo  vi_importance  vi_ess" if args.variational_steps else "")
    )
    for semitones in args.semitones:
        for interval in args.intervals:
            name = f"{semitones:g} st {1000 * interval:g} ms"
            sound = ab.bistability_stimulus(interval, semitones, lead=LEAD)
            observed = cochleagram(torch.tensor(sound.data[:, 0], dtype=DTYPE))
            results = {}
            for hypothesis, scene, starts in hypotheses(semitones, interval, args.infer_kernel):
                begin = time.perf_counter()
                result = evaluate(
                    scene,
                    starts,
                    observed,
                    cochleagram,
                    steps=args.steps,
                    n_samples=args.samples,
                    generator=torch.Generator().manual_seed(args.seed),
                    variational_steps=args.variational_steps,
                )
                results[hypothesis] = result
                e = result.evidence
                n_params = sum(value.numel() for value in starts[0].values())
                print(
                    f"{name:16s}  {hypothesis:4s}  {n_params:5d}  {result.structure_log_prior:7.2f}"
                    f"  {e.laplace:9.1f}  {e.importance:9.1f}  {e.effective_sample_size:4.0f}"
                    f"  {e.floored_eigenvalues:3d}"
                    f"  {e.log_joint_at_mode:9.1f}  {time.perf_counter() - begin:7.1f}"
                    + (
                        f"  {result.variational.elbo:9.1f}  {result.variational.importance:9.1f}"
                        f"  {result.variational.effective_sample_size:4.0f}"
                        if result.variational
                        else ""
                    ),
                    flush=True,
                )
            estimates = ["laplace", "importance"] + (
                ["elbo", "variational"] if args.variational_steps else []
            )
            odds = ", ".join(f"{estimate} {log_odds(results, estimate):.1f}" for estimate in estimates)
            print(f"{name:16s}  log odds two vs one: {odds}", flush=True)


if __name__ == "__main__":
    main()
