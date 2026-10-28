"""Reset of streaming after silence, online (docs/design/order-effects.md, step 3 of §7).

The stimulus is one of :mod:`sonore_infer.stimuli.reset`'s: ``bm`` for
Beauvois & Meddis (1997), an inducer of A tones, ``--silence`` s, then an
ABAB test (``--silence none`` for the test alone); ``cusack`` for Cusack et
al. (2004, Exp. 3), a 10 s ABA- sequence at ``--semitones``, ``--silence``
s, then the start of another. 50 ms of silence come first (``LEAD``, as
``tools/online/online_buildup.py``).

Hypotheses (design E3), each run online with
:func:`sonore_infer.online.run_online`:

- ``one``: one whistle source with every tone;
- ``two``: one source with the A tones and one with the B tones;
- ``one_restart`` and ``two_restart``: the same within each part, but the
  sources of the first part end at the silence and new ones start after it
  (``--restart`` adds them; the test alone has no restart).

Each tone starts its event; where tones fill their slots (Cusack et al.),
the event starts half a ramp shorter, since one source's events need a
positive rest between them.

Window ends are multiples of ``--window`` from the start of each part,
with one window spanning the silence (design E5). The evidence is taken
every ``--evidence-every`` s within each part and at the ends of the parts
and of the silence. Prints one line per evidence window and hypothesis, as
``online_buildup.py`` does, then at each evidence time each hypothesis's
log posterior and the log odds of two streams over one: the full sum and
the leaky decision (O10) for each ``--tau``, without the structure prior,
as ``leaky_odds.py`` prints them, and the structure prior's log odds.

    python tools/online/online_reset.py bm --silence 2 --restart
    python tools/online/online_reset.py cusack --semitones 6 --silence 1 --restart
"""

import argparse
import math
import time

import torch

from sonore_infer.cochleagram import FFTCochleagram
from sonore_infer.online import Tone, WhistleSpec, leaky_log_odds, run_online
from sonore_infer.stimuli import reset
from sonore_infer.stimuli.levels import FS

DTYPE = torch.float64
LEAD = 0.05


def hypotheses(stimulus: reset.ResetStimulus, restart: bool) -> dict:
    # Cusack et al.'s tones fill their 125 ms slots, so one source's events would touch, and the
    # scene needs a positive rest between them: such events start half a ramp shorter, centred
    # on the tone (an initial value only; the fit moves it)
    interval = min(b.onset - a.onset for a, b in zip(stimulus.tones, stimulus.tones[1:], strict=False))
    trim = stimulus.ramp / 2 if stimulus.tone_duration >= interval else 0.0

    def tone(placed):
        return Tone(
            placed.onset + trim / 2,
            stimulus.tone_duration - trim,
            stimulus.freqs[placed.kind],
            stimulus.level_db,
        )

    def sources(tones, prefix):
        one = (WhistleSpec(f"{prefix}0", tuple(tone(t) for t in tones)),)
        two = tuple(
            WhistleSpec(f"{prefix}{kind}", tuple(tone(t) for t in tones if t.kind == kind))
            for kind in "AB"
            if any(t.kind == kind for t in tones)
        )
        return one, two

    one, two = sources(stimulus.tones, "w")
    specs = {"one": one, "two": two}
    parts = [[t for t in stimulus.tones if t.part == part] for part in ("first", "second")]
    if restart and parts[0]:
        (one_a, two_a), (one_b, two_b) = sources(parts[0], "f"), sources(parts[1], "s")
        specs["one_restart"] = one_a + one_b
        specs["two_restart"] = two_a + two_b
    return specs


def part_grid(start: float, end: float, step: float) -> list[float]:
    """Multiples of ``step`` after ``start`` and before ``end``, then ``end``."""
    n = int((end - start) / step + 1e-9)
    return sorted({round(start + (k + 1) * step, 9) for k in range(n)} | {round(end, 9)})


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("study", choices=("bm", "cusack"))
    parser.add_argument("--silence", default="2", help="s, or none for Beauvois & Meddis's test alone")
    parser.add_argument("--semitones", type=float, default=6, help="Cusack et al. only")
    parser.add_argument("--test-duration", type=float, default=4.0, help="s; Cusack et al. only")
    parser.add_argument("--restart", action="store_true", help="add the two restart hypotheses (E3)")
    parser.add_argument("--only", nargs="*", default=None, help="run only these hypotheses")
    parser.add_argument("--window", type=float, default=0.1, help="s between updates")
    parser.add_argument("--lag", type=float, default=1.0, help="s")
    parser.add_argument("--steps", type=int, default=30, help="Adam steps per window")
    parser.add_argument("--heard", type=float, default=0.02, help="s of a tone heard before it enters")
    parser.add_argument("--evidence-every", type=float, default=0.5, help="s, within each part")
    parser.add_argument("--tau", type=float, nargs="+", default=[1.0, 2.5, 5.0], help="s")
    parser.add_argument(
        "--first-duration", type=float, default=None, help="s; shortens the first part, for timing runs"
    )
    args = parser.parse_args()
    started = time.perf_counter()
    silence = None if args.silence == "none" else float(args.silence)
    if args.study == "bm":
        inducer = reset.BM_INDUCER_TONES
        if args.first_duration is not None:
            inducer = int((args.first_duration - reset.BM_TONE_DURATION) / reset.BM_ONSET_INTERVAL + 1e-9) + 1
        stimulus = reset.beauvois_meddis(silence, inducer_tones=inducer, lead=LEAD)
    else:
        first = reset.CU_SEQUENCE_DURATION if args.first_duration is None else args.first_duration
        stimulus = reset.cusack(
            silence, args.semitones, test_duration=args.test_duration, first_duration=first, lead=LEAD
        )
    waveform = torch.tensor(stimulus.sound.data[:, 0], dtype=DTYPE)
    specs = hypotheses(stimulus, args.restart)
    if args.only:
        specs = {name: spec for name, spec in specs.items() if name in args.only}
    gap_start, gap_end = stimulus.gap
    has_first = any(t.part == "first" for t in stimulus.tones)
    if has_first:
        # each part on its own grid; the silence is one window, from the first part's end to the second's
        first_ends = part_grid(0.0, gap_start, args.window)
        ends = sorted(
            set(first_ends) | set(part_grid(gap_end, stimulus.duration, args.window)) | {round(gap_end, 9)}
        )
        evidence = sorted(
            set(part_grid(0.0, gap_start, args.evidence_every))
            | {round(gap_end, 9)}
            | set(part_grid(gap_end, stimulus.duration, args.evidence_every))
        )
    else:
        ends = part_grid(0.0, stimulus.duration, args.window)
        evidence = part_grid(0.0, stimulus.duration, args.evidence_every)
    print(
        f"{args.study}, silence {args.silence} s"
        + (
            f", {args.semitones:g} st, second sequence {args.test_duration:g} s"
            if args.study == "cusack"
            else ""
        )
        + (f", first part cut to {args.first_duration:g} s" if args.first_duration is not None else "")
        + f", gap {gap_start:.3f}-{gap_end:.3f} s, {stimulus.duration:.3f} s in all"
        + f"; window {1000 * args.window:g} ms, lag {args.lag:g} s, {args.steps} Adam steps,"
        + f" evidence every {args.evidence_every:g} s, past summarized, fft-bass-gain cochleagram"
    )
    print("window: hypothesis  end  n_free  term  total  structure  floored  log_joint  seconds  steps")
    cochleagram = FFTCochleagram()
    runs = {}
    for name, spec in specs.items():

        def report(step, name=name):
            w = step.window
            print(
                f"window: {name}  {w.end:.3f}  {w.n_free:4d}  {w.term:9.2f}  {w.total:10.2f}"
                f"  {step.structure_log_prior:6.2f}  {w.floored:2d}  {w.log_joint:10.2f}"
                f"  {w.seconds:7.2f}  {w.adam_steps}",
                flush=True,
            )

        runs[name] = {
            round(step.window.end, 9): step
            for step in run_online(
                waveform,
                spec,
                cochleagram,
                fs=FS,
                ramp=stimulus.ramp,
                window=args.window,
                lag=args.lag,
                steps=args.steps,
                heard=args.heard,
                ends=ends,
                evidence_ends=evidence,
                summarize_past=True,
                report=report,
            )
        }
    names = list(runs)
    common = sorted(set.intersection(*(set(run) for run in runs.values())))
    print(
        "odds: end  "
        + "  ".join(f"logpost_{n}" for n in names)
        + "  structure  full_sum  "
        + "  ".join(f"tau_{tau:g}" for tau in args.tau)
        + "  (two vs one)"
    )
    if "one" in runs and "two" in runs:
        terms = {n: [runs[n][e].window.term for e in common] for n in ("one", "two")}
        curves = [leaky_log_odds(common, terms["two"], terms["one"], tau) for tau in [math.inf] + args.tau]
    else:
        curves = []
    for k, end in enumerate(common):
        posts = "  ".join(f"{runs[n][end].log_posterior:11.2f}" for n in names)
        if curves:
            structure = runs["two"][end].structure_log_prior - runs["one"][end].structure_log_prior
            leaky = "  ".join(f"{curve[k]:9.2f}" for curve in curves)
            print(f"odds: {end:.3f}  {posts}  {structure:7.2f}  {leaky}", flush=True)
        else:
            print(f"odds: {end:.3f}  {posts}", flush=True)
    print(f"wall {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
