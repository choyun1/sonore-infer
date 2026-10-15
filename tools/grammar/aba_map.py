"""The ABA map: log posterior odds of two streams over one per stimulus, from compare_aba.py outputs.

Each hypothesis is read from the first folder that has it, in the order
given, so a rerun of one hypothesis (``compare_aba.py --hypotheses one``)
can be combined with the other from an earlier run:

    python tools/grammar/aba_map.py results/aba/pc_one_converged results/aba/pc_300steps

Prints, for each evidence estimate, a table of semitones by onset-to-onset
interval (positive favors two streams), then the fit steps and evidence
of each hypothesis and the folder it came from.
"""

import argparse
from pathlib import Path

ESTIMATES = {"variational": "vi_importance", "elbo": "elbo", "laplace": "laplace"}


def read(path):
    """{(semitones, interval_ms, hypothesis): {column: value}} from one compare_aba.py output."""
    rows = {}
    header = None
    for line in path.read_text().splitlines():
        tokens = line.split()
        if tokens[:2] == ["stimulus", "hypothesis"]:
            header = tokens[1:]
        elif header and len(tokens) > 4 and tokens[1] == "st" and tokens[3] == "ms" and tokens[4] != "log":
            values = dict(zip(header, tokens[4:], strict=True))
            key = (float(tokens[0]), float(tokens[2]), values["hypothesis"])
            rows[key] = {
                name: value if name == "hypothesis" else float(value) for name, value in values.items()
            }
    return rows


def collect(folders):
    """Each (semitones, interval_ms, hypothesis) from the first folder that has it, with that folder."""
    found = {}
    for folder in folders:
        for path in sorted(Path(folder).glob("*.txt")):
            for key, row in read(path).items():
                found.setdefault(key, (row, folder))
    return found


def log_odds(found, semitones, interval, column):
    """Log posterior odds of two streams over one by one estimate, or None if a hypothesis is missing."""
    try:
        two, one = (found[(semitones, interval, h)][0] for h in ("two", "one"))
    except KeyError:
        return None
    return (two[column] + two["structure"]) - (one[column] + one["structure"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="+", type=Path, help="compare_aba.py output folders, first wins")
    args = parser.parse_args()
    found = collect(args.folders)
    semitones = sorted({key[0] for key in found})
    intervals = sorted({key[1] for key in found})
    for estimate, column in ESTIMATES.items():
        print(f"log odds two streams vs one, {estimate}")
        print("semitones " + "".join(f"{interval:>9g}" for interval in intervals) + "  (onset-to-onset, ms)")
        for st in semitones:
            cells = (log_odds(found, st, interval, column) for interval in intervals)
            print(f"{st:9g} " + "".join("        -" if c is None else f"{c:9.1f}" for c in cells))
        print()
    print("stimulus           hypothesis  fit_steps  laplace  vi_importance  folder")
    for (st, interval, hypothesis), (row, folder) in sorted(found.items()):
        steps = row.get("fit_steps")
        print(
            f"{st:3g} st {interval:4g} ms  {hypothesis:4s}  {'?' if steps is None else int(steps):>9}"
            f"  {row['laplace']:9.1f}  {row['vi_importance']:9.1f}  {folder}"
        )


if __name__ == "__main__":
    main()
