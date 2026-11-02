"""Summary of ``tools/online/online_reset.py`` outputs (design order-effects, step 4 of §7).

Reads the ``odds:`` lines of each output; no inference is run. For each
run it prints the evidence for two streams over one gathered during the
test (the second part), the leaky decision (O10, τ 2.5 s) at the end, and
each hypothesis's gain from restarting its sources after the silence. For
Cusack et al.'s sequences it also prints the leaky decision through the
second sequence beside the first sequence's at the same times from its
start, the same decision gated (design order-effects-reset E12: emptied at
the end of the silence, so only the second sequence's terms count), the
gated decision between the restart hypotheses (E12 (C)), and
the two-over-one window terms of each 0.5 s. Log odds are
without the structure prior, as ``online_reset.py`` prints them (it adds
under 2).

    python tools/online/reset_summary.py results/data/O03/*.txt
"""

import argparse
import math
import os
import re

LEAD = 0.05  # s before the first tone, as online_reset.py
TAU_INDEX = 1  # online_reset.py's default --tau 1 2.5 5: the column for 2.5 s
TAU = 2.5  # s, the leak of that column (O10)
CUSACK_TIMES = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0)  # s from a sequence's first onset


def read(path):
    """(gap, rows) from one output: gap is (end of the first part, first onset of the second) [s];
    each row is {"t", "post": {hypothesis: log posterior}, "structure", "full", "tau": [...]}."""
    with open(path, encoding="utf-8") as file:
        lines = file.read().splitlines()
    gap = tuple(
        float(x)
        for x in re.search(r"gap ([\d.]+)-([\d.]+)", next(line for line in lines if " gap " in line)).groups()
    )
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
    """The numbers the O03 page reports for one output."""
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
        gated, value, previous = {0.0: 0.0}, 0.0, gap[1]
        for a, b in pairs:
            if b["t"] > gap[1] + 1e-6:
                value = value * math.exp(-(b["t"] - previous) / TAU) + b["full"] - a["full"]
                previous = b["t"]
                gated[round(b["t"] - gap[1], 9)] = value
        out["second_gated"] = [(d, gated[min(gated, key=lambda t: abs(t - d))]) for d in CUSACK_TIMES]
        if "one_restart" in end["post"]:
            # the same gated decision between the restart hypotheses: two restart over one restart,
            # from their log posteriors (so the structure prior's small change over time is included)
            restart, value, previous = {0.0: 0.0}, 0.0, gap[1]
            for a, b in pairs:
                if b["t"] > gap[1] + 1e-6:
                    term = (b["post"]["two_restart"] - b["post"]["one_restart"]) - (
                        a["post"]["two_restart"] - a["post"]["one_restart"]
                    )
                    value = value * math.exp(-(b["t"] - previous) / TAU) + term
                    previous = b["t"]
                    restart[round(b["t"] - gap[1], 9)] = value
            out["restart_gated"] = [
                (d, restart[min(restart, key=lambda t: abs(t - d))]) for d in CUSACK_TIMES
            ]
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
            pad = " " * (len(times) + 11)
            print(f"  {pad}gated  " + " ".join(f"{v:6.1f}" for _, v in s["second_gated"]))
            if "restart_gated" in s:
                print(f"  {pad}restart" + " ".join(f"{v:6.1f}" for _, v in s["restart_gated"]))
            print(f"  {pad}first  " + " ".join(f"{v:6.1f}" for _, v in s["first"]))
            print(f"  first 4 s evidence {s['first_evidence']:6.1f}")
            print("  terms per 0.5 s, second: " + " ".join(f"{v:6.1f}" for v in s["terms_second"]))
            print("  terms per 0.5 s, first:  " + " ".join(f"{v:6.1f}" for v in s["terms_first"]))


if __name__ == "__main__":
    main()
