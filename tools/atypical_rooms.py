"""Experiment (b): an exponential, ecological room model heard through rooms
that are not (docs/design/source-reverb.md, R7).

Each sound is a 400 ms ``sonore.gaussian_spectrogram`` at sonore's defaults
on a Gaussian noise carrier in a ``synth_ir`` room at the typical DRR (9.9
dB), with one atypical property: a decay shape (time_reversed,
linear_matched_start, linear_matched_end) or an RT60 profile across
frequency (inverted, exaggerated, reduced). The exponential, ecological
room is the control. Seeds of spectrogram, carrier and room are shared
across conditions (and distinct from each other), so each condition hears
the same source through a room drawn from the same noise.

The posterior is :func:`sonore_infer.sourceroom.infer` at floors of 20,
40 and 60 dB below the observation's peak and the measured sigma.

Each line written holds: the condition, true RT60, floor, the posterior's
5/50/95% RT60 quantiles, the log marginal likelihood, and
``measure_rt60``'s Schroeder T20 x 3 reading of the same IR (median over
bands, after the direct sound; tools/rt60_prior_fit.py's ``measure``).
``--summary`` prints, per condition and floor, the median over seeds of
posterior median / true RT60, next to T20 x 3 / true RT60, and the log
marginal likelihood relative to the control (same seed, RT60 and floor).

Default: 5 seeds x RT60 {0.4, 0.8} s x 7 conditions x 3 floors.

    python tools/atypical_rooms.py --out rooms.jsonl [--shard 0 --shards 1] [--floors 20]
    python tools/atypical_rooms.py --summary rooms.jsonl
"""

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import sonore as so
import torch
from rt60_prior_fit import measure
from scipy.signal import fftconvolve

from sonore_infer.blockpower import BlockPower
from sonore_infer.room import TYPICAL_DRR_DB
from sonore_infer.sourceroom import Model, infer

FS = 20_000.0
RT60S = (0.4, 0.8)
FLOORS = (60.0, 40.0, 20.0)
# Posterior 5/50/95% RT60 quantiles when every RT60 has the same evidence
# (the gamma prior on R5's grid, 0.1-2 s): an inference whose level fit
# settled where no cell is above the floor returns exactly these
PRIOR_QUANTILES = (0.1305, 0.3907, 1.0627)
CONDITIONS = {
    "control": {},
    "time_reversed": {"decay_shape": "time_reversed"},
    "linear_matched_start": {"decay_shape": "linear_matched_start"},
    "linear_matched_end": {"decay_shape": "linear_matched_end"},
    "inverted": {"rt60_profile": "inverted"},
    "exaggerated": {"rt60_profile": "exaggerated"},
    "reduced": {"rt60_profile": "reduced"},
}


def sounds(n_seeds):
    """(sound index, RT60, seed) in a fixed order; conditions share them."""
    out = []
    for rt60 in RT60S:
        for seed in range(n_seeds):
            out.append((len(out), rt60, seed))
    return out


def observe(power: BlockPower, index: int, rt60: float, condition: str) -> tuple[torch.Tensor, float]:
    """Block energies of one sound in one condition's room, and T20 x 3 of that room."""
    # an offset of 10**4 keeps these seeds apart from tools/recovery.py's
    seeds = np.random.SeedSequence(10**4 + index).generate_state(3)
    env_seed, carrier_seed, ir_seed = (int(s) for s in seeds)
    env = so.gaussian_spectrogram(power.duration, FS, rng=env_seed)
    x = env.to_sound(so.gaussian_noise(power.duration, FS, rng=carrier_seed)).ramp(power.ramp)
    ir = so.synth_ir(rt60, FS, drr_db=TYPICAL_DRR_DB, rng=ir_seed, **CONDITIONS[condition])
    t20, _, _ = measure(ir)
    return power.analyze(fftconvolve(x.data.ravel(), ir.data.ravel())), t20


def run_all(args):
    power = BlockPower()
    jobs = [
        (index, rt60, seed, condition) for index, rt60, seed in sounds(args.seeds) for condition in CONDITIONS
    ]
    jobs = [job for i, job in enumerate(jobs) if i % args.shards == args.shard]
    with open(args.out, "a") as out:
        for index, rt60, seed, condition in jobs:
            observed, t20 = observe(power, index, rt60, condition)
            for floor in args.floors:
                start = time.time()
                posterior = infer(Model(observed, power, floor_db=floor))
                quantiles = posterior.rt60_quantiles()
                record = {
                    "sound": index,
                    "seed": seed,
                    "rt60": rt60,
                    "condition": condition,
                    "floor_db": floor,
                    "quantiles": quantiles.tolist(),
                    "log_marginal": posterior.log_marginal,
                    "t20x3": t20,
                    "points": len(posterior.points),
                    "seconds": round(time.time() - start, 1),
                }
                out.write(json.dumps(record) + "\n")
                out.flush()
                print(
                    f"sound {index} rt60 {rt60} {condition} floor {floor:.0f}: rt60 "
                    + " ".join(f"{q:.3f}" for q in quantiles)
                    + f", T20x3 {t20:.3f} ({record['seconds']} s)",
                    flush=True,
                )


def summary(path):
    records = [json.loads(line) for line in open(path)]
    prior = [r for r in records if np.allclose(r["quantiles"], PRIOR_QUANTILES, rtol=1e-4)]
    broken = [r for r in records if not np.isfinite(r["quantiles"]).all()]
    for label, bad in (("returned the prior exactly", prior), ("has no finite posterior", broken)):
        for r in bad:
            print(f"left out: sound {r['sound']} {r['condition']} floor {r['floor_db']:.0f} ({label})")
    records = [r for r in records if r not in prior and r not in broken]
    control = {(r["sound"], r["floor_db"]): r for r in records if r["condition"] == "control"}
    groups = defaultdict(list)
    for r in records:
        groups[(r["condition"], r["rt60"], r["floor_db"])].append(r)
    print(f"{len(records)} records; medians over seeds")
    print("condition              rt60 floor  n   posterior/true  T20x3/true  log marginal re control [nats]")
    for condition in CONDITIONS:
        for rt60 in RT60S:
            for floor in FLOORS:
                group = groups.get((condition, rt60, floor), [])
                if not group:
                    continue
                ratio = np.median([r["quantiles"][1] / r["rt60"] for r in group])
                t20 = np.median([r["t20x3"] / r["rt60"] for r in group])
                gaps = [
                    r["log_marginal"] - control[(r["sound"], floor)]["log_marginal"]
                    for r in group
                    if (r["sound"], floor) in control
                ]
                gap = f"{np.median(gaps):9.1f}" if gaps else "        -"
                label = f"{condition:22s} {rt60:4.1f} {floor:5.0f} {len(group):2d}"
                print(f"{label}   {ratio:10.3f}  {t20:10.3f}  {gap}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="atypical_rooms.jsonl")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--floors", type=float, nargs="*", default=list(FLOORS))
    parser.add_argument("--summary", default=None, help="summarize this JSON-lines file instead")
    args = parser.parse_args()
    if args.summary:
        summary(args.summary)
        return
    if args.threads:
        torch.set_num_threads(args.threads)
    run_all(args)


if __name__ == "__main__":
    main()
