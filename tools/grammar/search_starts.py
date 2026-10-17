"""How the search's two-note fit to the just fifth (0 ms) depends on where it starts.

The search's best for the just fifth at 0 ms was a note plus a whistle,
and its own two-note fit came out about 10 below milestone (b)'s
(results/C3-search.md). This fits one hypothesis to that stimulus from one
start and prints its log posteriors, shifted by the likelihood's
normalising constant as on the C3 page. Starts:

- ``search``: two notes from the detector's candidates (197 and 304 Hz,
  spectra from the peaks), as the search fitted them;
- ``true-f0``: the same, with the f0s set to the notes' 200 and 300 Hz;
- ``no-floor``: the detector's f0s, with the harmonics no peak gave (-30 dB)
  started at 0 dB instead;
- ``b2``: two notes from milestone (b)'s start (tools/bass/compare_two_notes.py);
- ``whistle``: the search's best, the 197 Hz candidate plus the 304 Hz one
  as a whistle.

    python tools/grammar/search_starts.py true-f0 --seed 1
"""

import argparse
import dataclasses
import math
import sys
from pathlib import Path

import torch

from sonore_infer.cochleagram import FFTCochleagram
from sonore_infer.detect import detect
from sonore_infer.erb import freq_to_erb
from sonore_infer.scene import evaluate
from sonore_infer.search import SourcePlan, build

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bass"))
import compare_two_notes as b2  # noqa: E402
from search_scene import DTYPE, FS, stimulus  # noqa: E402

STARTS = ("search", "true-f0", "no-floor", "b2", "whistle")
# log(10 * sqrt(2 pi)), the Gaussian likelihood's normalising term per bin at SD 10 dB
BIN_CONSTANT = math.log(10 * math.sqrt(2 * math.pi))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("start", choices=STARTS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    waveform, ramp = stimulus("two:just_fifth:0.0")
    cochleagram = FFTCochleagram()
    observed = cochleagram(waveform)
    group = detect(observed, cochleagram, waveform.shape[-1])[0]
    pair = next(e for e in group.explanations if [c.kind for c in e] == ["harmonic", "harmonic"])
    upper = max(pair, key=lambda c: c.freq)
    plan = tuple(SourcePlan(c.kind, (c,)) for c in pair)
    if args.start == "whistle":
        whistle = dataclasses.replace(upper, kind="whistle", spectrum_db=None)
        plan = tuple(
            SourcePlan("whistle", (whistle,)) if c is upper else SourcePlan(c.kind, (c,)) for c in pair
        )
    scene, init = build(plan, waveform.shape[-1] / FS, FS, ramp, dtype=DTYPE)
    if args.start == "b2":
        condition = {"interval": "just_fifth", "asynchrony": 0.0}
        _, scene, starts = next(h for h in b2.hypotheses(condition) if h[0] == "two")
        init = starts[0]
    elif args.start == "true-f0":
        for index, c in enumerate(pair):
            f0 = 300.0 if c is upper else 200.0
            init[f"h{index}.f0_mean"] = freq_to_erb(torch.tensor(f0, dtype=DTYPE))
    elif args.start == "no-floor":
        for index in range(len(pair)):
            spectrum = init[f"h{index}.spectrum_db"].clone()
            spectrum[spectrum <= -30.0] = 0.0
            init[f"h{index}.spectrum_db"] = spectrum
    result = evaluate(
        scene,
        [init],
        observed,
        cochleagram,
        steps=15000,
        tolerance=0.5,
        variational_steps=2000,
        generator=torch.Generator().manual_seed(args.seed),
    )
    shift = observed.numel() * BIN_CONSTANT
    print("start  seed  fit_steps  laplace_posterior  variational_posterior  elbo_posterior")
    print(
        f"{args.start:8s} {args.seed:4d}  {len(result.fit.history):9d}"
        f"  {result.log_posterior_laplace + shift:17.1f}"
        f"  {result.log_posterior_variational + shift:21.1f}  {result.log_posterior_elbo + shift:14.1f}"
    )


if __name__ == "__main__":
    main()
