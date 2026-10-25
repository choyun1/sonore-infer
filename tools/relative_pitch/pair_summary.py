"""Summarize pair.py outputs: P(up) per pair beside Shepard's listeners (Fig. 3).

A pair's hypotheses may sit in several files (pair.py run with ``--only``);
rows are grouped by first and second tone across all the files given. Each
hypothesis's log posterior is its variational importance estimate plus its
structure prior, the ELBO where importance sampling failed, and P(up) is the
posterior mass of the hypotheses whose fitted shift is upward, as in pair.py.

Per pair: steps clockwise, the best hypothesis, harmonic up minus harmonic
down (log posterior), the better whistle hypothesis relative to the best, the
fitted shifts of the two harmonic hypotheses [ERB number], and P(up). Shepard
(1964, Fig. 3, read by eye) gives the fraction of presentations judged
higher, pooled over all 90 pairs, for his most accurate (Group I) and least
accurate (Group III) listeners. With ``--by-steps``, P(up) is also averaged
over the starting tones at each number of steps.

    python tools/relative_pitch/pair_summary.py results/data/P3/*.txt
"""

import argparse
import math
import re
from collections import defaultdict

# Shepard (1964) Fig. 3, fraction "2nd tone judged higher" at 1 to 9 steps clockwise
GROUP_I = (1.00, 1.00, 0.89, 0.75, 0.53, 0.25, 0.09, 0.02, 0.02)
GROUP_III = (0.74, 0.78, 0.83, 0.75, 0.52, 0.29, 0.24, 0.26, 0.40)
ROW = re.compile(
    r"\s*(?P<first>\d+)\s+(?P<second>\d+)\s+(?P<steps>\d+)\s+(?P<label>\S.*?)\s{2,}\d+"
    r"\s+(?P<structure>\S+)\s+\S+\s+\S+\s+(?P<elbo>\S+)\s+(?P<importance>\S+)\s+\S+"
    r"\s+(?P<shift>\S+)\s+\S+$"
)
HYPOTHESES = 4


def pairs(paths):
    """{(first, second): (steps, {label: (log posterior, fitted shift)})}"""
    found = defaultdict(dict)
    steps = {}
    for path in paths:
        for line in open(path, encoding="utf-8"):
            row = ROW.match(line)
            if not row:
                continue
            key = (int(row["first"]), int(row["second"]))
            importance = float(row["importance"])
            evidence = importance if math.isfinite(importance) else float(row["elbo"])
            found[key][row["label"]] = (evidence + float(row["structure"]), float(row["shift"]))
            steps[key] = int(row["steps"])
    return {key: (steps[key], hypotheses) for key, hypotheses in found.items()}


def p_up(hypotheses):
    top = max(value for value, _ in hypotheses.values())
    weights = [(math.exp(value - top), shift) for value, shift in hypotheses.values()]
    return sum(w for w, shift in weights if shift > 0) / sum(w for w, _ in weights)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="+", help="pair.py outputs")
    parser.add_argument("--by-steps", action="store_true", help="also average P(up) over starting tones")
    args = parser.parse_args()
    found = pairs(args.paths)
    print(
        "first  second  steps  best                        harmonic_up_minus_down  best_whistles"
        "  shift_up_erb  shift_down_erb  p_up  shepard_I  shepard_III"
    )
    by_steps = defaultdict(list)
    for (first, second), (steps, hypotheses) in sorted(found.items()):
        if len(hypotheses) < HYPOTHESES:
            print(
                f"{first:5d}  {second:6d}  {steps:5d}  incomplete:"
                f" {len(hypotheses)} of {HYPOTHESES} hypotheses"
            )
            continue
        relative = {
            label: value - max(v for v, _ in hypotheses.values()) for label, (value, _) in hypotheses.items()
        }
        best = max(relative, key=relative.get)
        up = next(label for label in relative if label.startswith("harmonic up"))
        down = next(label for label in relative if label.startswith("harmonic down"))
        whistles = max(value for label, value in relative.items() if label.startswith("whistles"))
        probability = p_up(hypotheses)
        by_steps[steps].append(probability)
        print(
            f"{first:5d}  {second:6d}  {steps:5d}  {best:26s}  {relative[up] - relative[down]:22.1f}"
            f"  {whistles:13.1f}  {hypotheses[up][1]:+12.3f}  {hypotheses[down][1]:+14.3f}"
            f"  {probability:4.3f}"
            f"  {GROUP_I[steps - 1]:9.2f}  {GROUP_III[steps - 1]:11.2f}"
        )
    if args.by_steps:
        print()
        print("steps  pairs  mean_p_up  min_p_up  max_p_up  shepard_I  shepard_III")
        for steps, values in sorted(by_steps.items()):
            print(
                f"{steps:5d}  {len(values):5d}  {sum(values) / len(values):9.3f}  {min(values):8.3f}"
                f"  {max(values):8.3f}  {GROUP_I[steps - 1]:9.2f}  {GROUP_III[steps - 1]:11.2f}"
            )


if __name__ == "__main__":
    main()
