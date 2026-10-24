"""Eqn 2 mistuning thresholds by all four evidence estimates, from scene_mistuned_harmonic.py
outputs run with ``--variational-steps``.

Rows for the same f0 and harmonic are merged across files and sorted by
mistuning. Every row is used, floored Hessians included (as on the A7 page);
the four log-odds columns (Laplace, importance sampling from Laplace, ELBO,
importance sampling from the variational fit) are turned into p(two
sources) and Eqn 2's threshold (``sonore_infer.thresholds``). A row whose
estimate is not finite is left out of that estimate's threshold and counted.

    python tools/bass/variational_thresholds.py results/data/A7/converged/*.txt
"""

import argparse
import math
import re
from collections import defaultdict

from sonore_infer.thresholds import probability_from_log_odds, threshold

HEADER = re.compile(r"f0 (\S+) Hz, harmonic (\d+) mistuned")
ESTIMATES = ("laplace", "is", "elbo", "vi_is")


def read(paths):
    """{(f0, harmonic): {percent: (four log odds)}}"""
    rows = defaultdict(dict)
    for path in paths:
        key = None
        for line in open(path, encoding="utf-8"):
            header = HEADER.match(line)
            if header:
                key = (float(header[1]), int(header[2]))
                continue
            fields = line.split()
            if key is None or len(fields) != 18 or not re.fullmatch(r"\d+(\.\d+)?", fields[0]):
                continue
            values = [float(value) for value in fields]
            rows[key][values[0]] = (values[7], values[8], values[16], values[17])
    return rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    print(
        "f0_hz  harmonic  mistunings  " + "  ".join(f"threshold_{name}" for name in ESTIMATES) + "  dropped"
    )
    for (f0, harmonic), by_percent in sorted(read(args.paths).items()):
        percents = sorted(by_percent)
        out, dropped = [], 0
        for index in range(len(ESTIMATES)):
            kept = [(p, by_percent[p][index]) for p in percents if math.isfinite(by_percent[p][index])]
            dropped += len(percents) - len(kept)
            odds = [value for _, value in kept]
            out.append(threshold([p for p, _ in kept], probability_from_log_odds(odds)))
        print(
            f"{f0:5g}  {harmonic:8d}  {len(percents):10d}  "
            + "  ".join(
                f"{value:{len(f'threshold_{name}')}.1f}" for value, name in zip(out, ESTIMATES, strict=True)
            )
            + f"  {dropped:7d}"
        )


if __name__ == "__main__":
    main()
