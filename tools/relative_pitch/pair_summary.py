"""Summarize pair.py outputs: P(up) per pair beside Shepard's listeners (Fig. 3).

Per pair: steps clockwise, the best hypothesis, harmonic up minus harmonic
down (log posterior), the best whistle hypothesis relative to the best, the
fitted shifts of the two harmonic hypotheses [ERB number], and P(up). Shepard
(1964, Fig. 3, read by eye) gives the fraction of presentations judged
higher, pooled over all 90 pairs, for his most accurate (Group I) and least
accurate (Group III) listeners.

    python tools/relative_pitch/pair_summary.py results/data/P3/*.txt
"""

import argparse
import re

# Shepard (1964) Fig. 3, fraction "2nd tone judged higher" at 1 to 9 steps clockwise
GROUP_I = (1.00, 1.00, 0.89, 0.75, 0.53, 0.25, 0.09, 0.02, 0.02)
GROUP_III = (0.74, 0.78, 0.83, 0.75, 0.52, 0.29, 0.24, 0.26, 0.40)
ROW = re.compile(r"\s*(\d+)\s+(\d+)\s+(\d+)\s+(\S.*?)\s{2,}\d+\s+.*\s(\S+)\s+(\S+)$")
RELATIVE = re.compile(r"(\S.*?) (-?\d+\.\d)(?:, |$)")
P_UP = re.compile(r"(\d+)->(\d+): steps clockwise (\d+), P\(up\) (\S+)")


def read(paths):
    pairs = {}
    for path in paths:
        shifts, relative = {}, {}
        for line in open(path, encoding="utf-8"):
            row = ROW.match(line)
            if row:
                shifts[row[4]] = float(row[5])
            if ": relative to the best: " in line:
                relative = dict(RELATIVE.findall(line.split(": relative to the best: ")[1].strip()))
            up = P_UP.match(line)
            if up:
                key = (int(up[1]), int(up[2]))
                pairs[key] = (int(up[3]), float(up[4]), shifts, {k: float(v) for k, v in relative.items()})
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="+", help="pair.py outputs")
    args = parser.parse_args()
    print(
        "first  second  steps  best                        harmonic_up_minus_down  best_whistles"
        "  shift_up_erb  shift_down_erb  p_up  shepard_I  shepard_III"
    )
    for (first, second), (steps, p_up, shifts, relative) in sorted(read(args.paths).items()):
        best = max(relative, key=relative.get)
        up = next(label for label in relative if label.startswith("harmonic up"))
        down = next(label for label in relative if label.startswith("harmonic down"))
        whistles = max(value for label, value in relative.items() if label.startswith("whistles"))
        print(
            f"{first:5d}  {second:6d}  {steps:5d}  {best:26s}  {relative[up] - relative[down]:22.1f}"
            f"  {whistles:13.1f}  {shifts[up]:+12.3f}  {shifts[down]:+14.3f}  {p_up:4.3f}"
            f"  {GROUP_I[steps - 1]:9.2f}  {GROUP_III[steps - 1]:11.2f}"
        )


if __name__ == "__main__":
    main()
