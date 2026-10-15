"""Leaky log odds (design online, decision O10) from the output of ``tools/online/online_buildup.py``.

Reads the ``window:`` lines of one or more outputs and prints, at the end
of each triplet, the log odds of two streams over one with the full sum of
window terms and with each ``--tau``, plus the structure prior's log odds.
No inference is run.

    python tools/online/leaky_odds.py st8_w50_s150_lag1.txt --tau 1 2.5 5
"""

import argparse
import math
from collections import defaultdict

from sonore_infer.online import leaky_log_odds


def read_windows(path):
    """{(lag, hypothesis): [(end, term, structure prior)]} from an output file."""
    runs = defaultdict(list)
    with open(path) as lines:
        for line in lines:
            if line.startswith("window: ") and not line.startswith("window: lag"):
                fields = line.split()
                lag, hypothesis, end, term, structure = fields[1], fields[2], fields[3], fields[5], fields[7]
                runs[lag, hypothesis].append((float(end), float(term), float(structure)))
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("outputs", nargs="+")
    parser.add_argument("--tau", type=float, nargs="+", default=[1.0, 2.5, 5.0], help="s")
    parser.add_argument("--lead", type=float, default=0.05, help="s before the first triplet")
    args = parser.parse_args()
    for path in args.outputs:
        runs = read_windows(path)
        for lag in sorted({lag for lag, _ in runs}):
            one, two = runs[lag, "one"], runs[lag, "two"]
            # windows before the first tone have no terms; align on the ends both have
            ends = sorted({end for end, _, _ in one} & {end for end, _, _ in two})
            one_by_end = {end: (term, structure) for end, term, structure in one}
            two_by_end = {end: (term, structure) for end, term, structure in two}
            taus = [math.inf] + args.tau
            curves = {
                tau: leaky_log_odds(
                    ends, [two_by_end[e][0] for e in ends], [one_by_end[e][0] for e in ends], tau
                )
                for tau in taus
            }
            print(f"{path}, lag {lag} s: log odds two vs one at each triplet's end")
            print("triplets  structure  full_sum  " + "  ".join(f"tau_{tau:g}" for tau in args.tau))
            for n in range(1, 100):
                end = round(args.lead + 0.5 * n, 9)
                if end not in ends:
                    break
                k = ends.index(end)
                structure = two_by_end[end][1] - one_by_end[end][1]
                values = "  ".join(f"{curves[tau][k] + structure:9.2f}" for tau in taus)
                print(f"{n}  {structure:7.2f}  {values}")


if __name__ == "__main__":
    main()
