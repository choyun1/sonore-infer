"""Summary of ``tools/online/online_reset.py`` outputs (design order-effects, step 4 of §7).

Reads the ``odds:`` lines of each output; no inference is run. For each
run it prints the evidence for two streams over one gathered during the
test (the second part), the leaky decision (O10, τ 2.5 s) at the end, and
each hypothesis's gain from restarting its sources after the silence. For
Cusack et al.'s sequences it also prints the leaky decision through the
second sequence beside the first sequence's at the same times from its
start, and the two-over-one window terms of each 0.5 s. Log odds are
without the structure prior, as ``online_reset.py`` prints them (it adds
under 2).

    python tools/online/reset_summary.py results/data/O3/*.txt
"""

import argparse
import os
import re

LEAD = 0.05  # s before the first tone, as online_reset.py
TAU_INDEX = 1  # online_reset.py's default --tau 1 2.5 5: the column for 2.5 s
CUSACK_TIMES = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0)  # s from a sequence's first onset


def read(path):
    """(gap, rows) from one output: gap is (end of the first part, first onset of the second) [s];
    each row is {"t", "post": {hypothesis: log posterior}, "structure", "full", "tau": [...]}."""
    with open(path, encoding="utf-8") as file:
        lines = file.read().splitlines()
    gap = tuple(float(x) for x in re.search(r"gap ([\d.]+)-([\d.]+)", lines[0]).groups())
    names, rows = [], []
    for line in lines:
        if line.startswith("odds: end"):
            names = [field[len("logpost_") :] for field in line.split() if field.startswith("logpost_")]
        elif line.startswith("odds: "):
            values = [float(field) for field in line.split()[1:]]
            n = len(names)
            rows.append(
                {
                    "t": values[0],
                    "post": dict(zip(names, values[1 : 1 + n], strict=True)),
                    "structure": values[1 + n],
                    "full": values[2 + n],
                    "tau": values[3 + n :],
                }
            )
    return gap, rows


def at(rows, t):
    """The row whose evidence time is nearest ``t``."""
    return min(rows, key=lambda row: abs(row["t"] - t))


def summary(path):
    """The numbers the O3 page reports for one output."""
    gap, rows = read(path)
    end = rows[-1]
    out = {
        "gap": gap,
        "test_evidence": end["full"] - at(rows, gap[1])["full"],
        "leaky_end": end["tau"][TAU_INDEX],
    }
    if "one_restart" in end["post"]:
        post = end["post"]
        out["restart_gain"] = {
            "one": post["one_restart"] - post["one"],
            "two": post["two_restart"] - post["two"],
        }
        out["best"] = max(post, key=post.get)
    if os.path.basename(path).startswith("cu"):
        out["second"] = [(d, at(rows, gap[1] + d)["tau"][TAU_INDEX]) for d in CUSACK_TIMES]
        out["first"] = [(d, at(rows, LEAD + d)["tau"][TAU_INDEX] if d else 0.0) for d in CUSACK_TIMES]
        out["first_evidence"] = at(rows, LEAD + 4.0)["full"]
        pairs = list(zip(rows, rows[1:], strict=False))
        out["terms_second"] = [b["full"] - a["full"] for a, b in pairs if b["t"] > gap[1] + 1e-6]
        out["terms_first"] = [
            b["full"] - a["full"] for a, b in [({"full": 0.0}, rows[0]), *pairs] if b["t"] <= LEAD + 4.0
        ]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("outputs", nargs="+")
    args = parser.parse_args()
    for path in args.outputs:
        s = summary(path)
        name = os.path.splitext(os.path.basename(path))[0]
        line = f"{name:9s} test evidence {s['test_evidence']:7.1f}  leaky at end {s['leaky_end']:6.1f}"
        if "restart_gain" in s:
            gain = s["restart_gain"]
            line += f"  restart gain: one {gain['one']:6.1f}, two {gain['two']:6.1f}  best {s['best']}"
        print(line)
        if "second" in s:
            times = " ".join(f"{d:g}" for d, _ in s["second"])
            print(f"  leaky at {times} s: second " + " ".join(f"{v:6.1f}" for _, v in s["second"]))
            print(f"  {' ' * (len(times) + 11)}first  " + " ".join(f"{v:6.1f}" for _, v in s["first"]))
            print(f"  first 4 s evidence {s['first_evidence']:6.1f}")
            print("  terms per 0.5 s, second: " + " ".join(f"{v:6.1f}" for v in s["terms_second"]))
            print("  terms per 0.5 s, first:  " + " ".join(f"{v:6.1f}" for v in s["terms_first"]))


if __name__ == "__main__":
    main()
