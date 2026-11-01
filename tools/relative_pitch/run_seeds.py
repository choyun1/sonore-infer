"""Rerun P9's last section on several seeds (design relative-pitch-group3, P26).

One ``pair.py --seeds`` job per file of ``results/data/P9/options/all/``
(first tone, second tone, reading's octave, direction), each seed's rows to
``<out>/s<seed>/<the same name>``. Jobs run ``--jobs`` at a time, each with
one thread (several at once with more threads each run several times
slower). A job whose files for every seed already end in a P(up) line is
skipped, so the run can be stopped and restarted.

    python tools/relative_pitch/run_seeds.py --seeds 0 1 2 --jobs 6 --out results/data/P9/seeds
"""

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
SOURCE = HERE.parent.parent / "results" / "data" / "P9" / "options" / "all"
NAME = re.compile(
    r"(?P<first>\d+)-(?P<second>\d+)-octave(?P<octave>-?\d+)-harmonic-(?P<direction>up|down)\.txt"
)
OPTIONS = ["--sparse-spectrum", "--floor", "30", "--threshold-floor", "--dominance"]


def done(paths):
    """Whether every seed's file is complete (or the job has no hypothesis to fit)."""
    for path in paths:
        if not path.exists():
            return False
        text = path.read_text(encoding="utf-8")
        if "P(up)" not in text and text.count("\n") != 2:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--out", type=Path, default=SOURCE.parent.parent / "seeds")
    args = parser.parse_args()
    for seed in args.seeds:
        (args.out / f"s{seed}").mkdir(parents=True, exist_ok=True)
    jobs = []
    for path in sorted(SOURCE.glob("*.txt")):
        match = NAME.fullmatch(path.name)
        if not match:
            continue
        outputs = [args.out / f"s{seed}" / path.name for seed in args.seeds]
        if done(outputs):
            continue
        jobs.append((match, path.name))
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="-1")
    begin = time.perf_counter()
    print(f"{len(jobs)} jobs to run, {args.jobs} at a time, seeds {args.seeds}", flush=True)

    def run(job):
        match, name = job
        command = [
            sys.executable,
            str(HERE / "pair.py"),
            "--first",
            match["first"],
            "--seconds",
            match["second"],
            "--octave",
            match["octave"],
            "--only",
            f"harmonic {match['direction']}",
            *OPTIONS,
            "--seeds",
            *map(str, args.seeds),
            "--output",
            str(args.out / "s{seed}" / name),
        ]
        log = args.out / "logs" / name
        log.parent.mkdir(exist_ok=True)
        with open(log, "w", encoding="utf-8") as stderr:
            code = subprocess.run(command, env=env, stderr=stderr, stdout=subprocess.DEVNULL).returncode
        hours = (time.perf_counter() - begin) / 3600
        print(f"{name}: {'ok' if code == 0 else f'FAILED ({code})'} at {hours:.2f} h", flush=True)
        return code

    with ThreadPoolExecutor(args.jobs) as pool:
        codes = list(pool.map(run, jobs))
    failed = sum(code != 0 for code in codes)
    print(f"finished: {len(codes) - failed} ok, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
