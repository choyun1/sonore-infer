"""Combine one_tone.py outputs into one log posterior per tone and hypothesis.

A hypothesis with a sparse spectrum is fitted from two starts, which reach
different modes (see one_tone.py). Its fits may sit in several files: one
run with both starts (rows ``label [start k]``), or an earlier run that kept
only the mode with the higher log joint plus a rerun of one start (``--start``).
Rows whose log joint at the mode agree within 1 are taken as the same mode
and counted once; the hypothesis's log posterior is the log-sum-exp of its
distinct modes. Each mode's log posterior is the variational importance
estimate plus the structure prior, or the ELBO plus the structure prior
where importance sampling failed (marked ELBO).

Log posteriors are shifted by the likelihood's normalizing term, 4329.7 for
a 0.225 s tone (64 channels x 21 windows x 3.222), as on the P1 page.

    python tools/relative_pitch/combine_starts.py results/data/P5/*.txt
"""

import argparse
import math
import re
from collections import defaultdict

SHIFT = 4329.7
SAME_MODE = 1.0
ROW = re.compile(
    r"\s*(?P<step>\d+)\s+(?P<label>\S.*?)\s{2,}(?P<n_params>\d+)\s+(?P<structure>-?[\d.]+)"
    r"\s+(?P<joint>-?[\d.]+)\s+(?P<laplace>-?[\d.]+)\s+(?P<elbo>-?[\d.]+)\s+(?P<importance>-?[\d.]+|-inf)"
    r"\s+(?P<ess>\S+)\s+(?P<seconds>[\d.]+)$"
)


def read_modes(paths):
    """{(step, label): [(log joint, shifted log posterior, importance succeeded, seconds)]}"""
    modes = defaultdict(list)
    for path in paths:
        text = open(path, encoding="utf-8").read()
        if "relative to the best" not in text:
            continue
        for line in text.splitlines():
            row = ROW.match(line)
            if not row:
                continue
            label = re.sub(r" \[start \d\]$", "", row["label"])
            importance = float(row["importance"])
            evidence = importance if math.isfinite(importance) else float(row["elbo"])
            modes[(int(row["step"]), label)].append(
                (
                    float(row["joint"]),
                    evidence + float(row["structure"]) + SHIFT,
                    math.isfinite(importance),
                    float(row["seconds"]),
                )
            )
    return modes


def combine(fits):
    """(log posterior, distinct modes' log posteriors, every importance estimate succeeded)"""
    distinct = []
    for fit in fits:
        if not any(abs(fit[0] - other[0]) < SAME_MODE for other in distinct):
            distinct.append(fit)
    top = max(mode[1] for mode in distinct)
    total = top + math.log(sum(math.exp(mode[1] - top) for mode in distinct))
    return total, [mode[1] for mode in distinct], all(mode[2] for mode in distinct)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="+", help="one_tone.py outputs")
    args = parser.parse_args()
    modes = read_modes(args.paths)
    seconds = [fit[3] for fits in modes.values() for fit in fits]
    for step in sorted({step for step, _ in modes}):
        combined = {label: combine(fits) for (s, label), fits in modes.items() if s == step}
        best = max(value[0] for value in combined.values())
        print(f"step {step}")
        print("  hypothesis                          log_posterior  relative  modes")
        for label, (total, each, succeeded) in sorted(combined.items(), key=lambda item: -item[1][0]):
            each_text = ", ".join(f"{value:.1f}" for value in each)
            print(
                f"  {label:34s}  {total:12.1f}  {total - best:8.1f}  {each_text}"
                + ("" if succeeded else "  (ELBO)")
            )
    print(f"{len(seconds)} fits, {min(seconds):.0f} to {max(seconds):.0f} s each")


if __name__ == "__main__":
    main()
