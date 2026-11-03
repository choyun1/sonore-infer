"""Average pair.py outputs over seeds (design relative-pitch-seeds, P27).

For every file name present in each seed's folder, writes one file in
pair.py's format to ``--out`` in which each hypothesis's evidence (its
importance estimate, or its ELBO where importance sampling failed, as
pair_summary.py reads it) is the log of the mean over seeds of exp(evidence):
the importance estimate one would get by pooling the seeds' samples, with
equal weight per seed. The ``vi_importance`` column holds that value, the
``elbo`` column the mean ELBO and ``vi_ess`` the summed effective sample
size; the mode, Laplace and shift columns are the first seed's (the mode fit
does not use the seed). It also prints, per pair, the log odds of up against
down on each seed, their range, and the same summed over steps clockwise.

    python tools/relative_pitch/average_seeds.py results/data/P09/seeds/s0 results/data/P09/seeds/s1 \\
        results/data/P09/seeds/s2 --out results/data/P09/seeds/mean
"""

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROW = re.compile(
    r"(?P<head>\s*\d+\s+\d+\s+\d+\s+\S.*?\s{2,}\d+\s+\S+\s+\S+\s+\S+)"
    r"\s+(?P<elbo>\S+)\s+(?P<importance>\S+)\s+(?P<ess>\S+)\s+(?P<shift>\S+)\s+(?P<seconds>\S+)$"
)
LABEL = re.compile(r"\s*(?P<first>\d+)\s+(?P<second>\d+)\s+(?P<steps>\d+)\s+(?P<label>\S.*?)\s{2,}\d+")


def evidence(row):
    importance = float(row["importance"])
    return importance if math.isfinite(importance) else float(row["elbo"])


def log_mean_exp(values):
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values) / len(values))


def rows(path):
    """[(line match, label match)] for the hypothesis rows of one pair.py output."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = ROW.match(line)
        if row:
            out.append((row, LABEL.match(line)))
    return out


def log_odds(entries):
    """Log posterior mass of upward hypotheses minus the rest, from (evidence + structure, shift)."""
    up = [value for value, shift in entries if shift > 0]
    down = [value for value, shift in entries if shift <= 0]
    if not up or not down:
        return None
    return log_sum_exp(up) - log_sum_exp(down)


def log_sum_exp(values):
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("folders", nargs="+", type=Path, help="one folder of pair.py outputs per seed")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = sorted(set.intersection(*({p.name for p in folder.glob("*.txt")} for folder in args.folders)))
    # per pair and seed: [(evidence + structure, shift)] over hypotheses
    per_pair = defaultdict(lambda: [[] for _ in args.folders])
    steps_of = {}
    for name in names:
        per_seed = [rows(folder / name) for folder in args.folders]
        header = (args.folders[0] / name).read_text(encoding="utf-8").splitlines()[:2]
        lines = [*header]
        for index, (row, label) in enumerate(per_seed[0]):
            matches = [seed_rows[index] for seed_rows in per_seed]
            assert all(m[1]["label"] == label["label"] for m in matches), name
            values = [evidence(m[0]) for m in matches]
            mean = log_mean_exp(values)
            elbo = float(np.mean([float(m[0]["elbo"]) for m in matches]))
            ess = sum(float(m[0]["ess"]) for m in matches)
            lines.append(
                f"{row['head']}  {elbo:9.1f}  {mean:9.1f}  {ess:5.0f}"
                f"  {row['shift']:>8s}  {row['seconds']:>7s}"
            )
            structure = float(row["head"].split()[-3])
            key = (int(label["first"]), int(label["second"]))
            steps_of[key] = int(label["steps"])
            for seed, value in enumerate(values):
                per_pair[key][seed].append((value + structure, float(row["shift"])))
        (args.out / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("first  second  steps  log odds on each seed  range")
    ranges = defaultdict(list)
    for key in sorted(per_pair):
        odds = [log_odds(entries) for entries in per_pair[key]]
        if any(value is None for value in odds):
            continue
        spread = max(odds) - min(odds)
        ranges[steps_of[key]].append(spread)
        print(
            f"{key[0]:5d}  {key[1]:6d}  {steps_of[key]:5d}  "
            + " ".join(f"{value:+7.1f}" for value in odds)
            + f"  {spread:5.1f}"
        )
    print()
    print("steps  pairs  median range  largest range")
    for steps in sorted(ranges):
        values = ranges[steps]
        print(f"{steps:5d}  {len(values):5d}  {np.median(values):12.1f}  {max(values):13.1f}")
    print(f"wrote {len(names)} files to {args.out}")


if __name__ == "__main__":
    main()
