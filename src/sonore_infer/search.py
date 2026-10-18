"""Search over scene structures: hypotheses built from the sound, not written by hand (design C9, C10).

The paper's sequential inference [App. B.2 of Cusimano et al., 2024], with
its segmentation network replaced by :mod:`sonore_infer.detect` (design C1,
C9 (A)):

1. **Rounds.** The detector's onset groups are taken in time order, one per
   round. Round ``j`` sees the sound up to the next group's onset (the last
   round sees all of it). Each hypothesis kept from the round before is
   extended by one explanation of the group (see
   :class:`~sonore_infer.detect.OnsetGroup`), each of its events joining an
   existing source of the same kind whose events it does not overlap, or
   starting a new source; or the group is left out (design C9). The paper's limits apply [App. B.2.2]: at most
   ``max_sources`` sources, ``max_new_sources`` new sources after the first
   round and ``max_new_events`` new events per round, and
   ``max_per_round`` hypotheses per round, fewest sources first.
2. **Screening.** Every hypothesis is fitted (MAP, Adam) on the round's part of
   the sound, from its candidates, and ranked by its Laplace log posterior
   (evidence plus structure prior; design C10 (B)). The best ``beam`` go on to
   the next round [the paper keeps two, App. B.2.3].
3. **Residual proposals** (design C9 (B)). After the last round, the best
   hypothesis is rendered and the detector is run again on what it leaves
   unexplained (the observed level above the rendered level); each group found
   there makes one more round on the whole sound.
4. **Cleanup** [App. B.2.4]: from the best hypotheses, remove a source, merge
   two sources, change a source's kind, merge two of a source's events, or
   remove an event; up to ``max_cleanup`` such hypotheses, in that order.
5. **Final ranking.** Of every hypothesis fitted to the whole sound, the best
   ``keep`` by the screen get variational evidence (as in
   :func:`~sonore_infer.scene.evaluate`), which ranks them.

Departures from the paper, besides the detector: hypotheses are fitted by
MAP and screened by Laplace instead of 250 steps of variational inference
[paper App. B.2; docs/design/milestone-c.md, decision C9] (decision C10);
each round's fits start from the candidates, not from the previous round's
fit; one onset group per round.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from sonore_infer.cochleagram import Cochleagram
from sonore_infer.detect import Candidate, OnsetGroup, detect
from sonore_infer.erb import erb_to_freq, freq_to_erb
from sonore_infer.evidence import variational_evidence
from sonore_infer.scene import Harmonic, Params, Scene, SceneResult, SceneTiming, Whistle, evaluate

# events of one source must start at least this far apart [s], and each ends
# at least GAP before the next one starts
MIN_SPACING = 0.02
# an event joins a source only if it overlaps the source's events by at most this
# fraction of their union [the paper's IoU limit, App. B.2.2]
MAX_OVERLAP = 0.25
# MIN_SPACING, GAP, MIN_DURATION [s] and ABSENT_DB [dB, the starting level, relative
# to the source, of a harmonic no candidate gave] are not measured: they were set
# when the search was written (commit e9f5649), and no tool or decision records why
GAP = 0.005
MIN_DURATION = 0.01
ABSENT_DB = -30.0


@dataclass(frozen=True)
class SourcePlan:
    """One source of a hypothesis: its kind and the candidate events it emits, in time order."""

    kind: str
    events: tuple[Candidate, ...]


Plan = tuple[SourcePlan, ...]


def plan_key(plan: Plan) -> tuple:
    """A hypothesis's structure, the same whatever the order of its sources."""
    return tuple(
        sorted(
            (source.kind, tuple((round(e.onset, 3), round(e.freq)) for e in source.events)) for source in plan
        )
    )


def describe(plan: Plan) -> str:
    """A short text form, e.g. ``"w[1000x8] + w[1414x4]"``: kind, first frequency [Hz], events."""
    parts = []
    for source in sorted(plan, key=lambda s: (s.events[0].onset, s.events[0].freq)):
        freqs = sorted({round(e.freq) for e in source.events})
        shown = "/".join(str(f) for f in freqs[:3]) + ("/..." if len(freqs) > 3 else "")
        parts.append(f"{source.kind[0]}[{shown}x{len(source.events)}]")
    return " + ".join(parts)


def build(
    plan: Plan,
    end: float,
    fs: float,
    ramp: float,
    *,
    infer_kernel: bool = False,
    dtype: torch.dtype = torch.float64,
) -> tuple[Scene, Params]:
    """The hypothesis as a scene on the sound's first ``end`` seconds, and its start.

    Events that start at or after ``end`` are left out. Each event's duration
    is cut so it ends ``GAP`` before the source's next event and before
    ``end``. A harmonic source's spectrum is the largest of its events'.
    """
    timing = SceneTiming(fs=fs, total_duration=end, ramp=ramp)
    sources: list[Harmonic | Whistle] = []
    init: Params = {}
    for index, source_plan in enumerate(plan):
        events = [e for e in source_plan.events if e.onset < end - MIN_DURATION]
        if not events:
            continue
        onsets = [e.onset for e in events]
        limits = [*(onsets[1:]), end + GAP]
        durations = [
            max(min(e.duration, limit - GAP - e.onset), MIN_DURATION)
            for e, limit in zip(events, limits, strict=True)
        ]

        level_db = max(e.level_db for e in events)
        if source_plan.kind == "whistle":
            whistle = Whistle(f"w{index}", n_events=len(events), infer_kernel=infer_kernel)
            sources.append(whistle)
            init |= whistle.initial(
                timing,
                freq=_value([e.freq for e in events]),
                level_db=level_db,
                onset=_value(onsets),
                duration=_value(durations),
                dtype=dtype,
            )
        else:
            n_harmonics = max(len(e.spectrum_db or ()) for e in events) or 1
            spectrum = torch.full((n_harmonics,), ABSENT_DB, dtype=dtype)
            for e in events:
                values = torch.tensor(e.spectrum_db or (0.0,), dtype=dtype) + e.level_db - level_db
                spectrum[: len(values)] = torch.maximum(spectrum[: len(values)], values)
            harmonic = Harmonic(f"h{index}", n_harmonics, n_events=len(events), infer_kernel=infer_kernel)
            sources.append(harmonic)
            init |= harmonic.initial(
                timing,
                f0=_value([e.freq for e in events]),
                level_db=level_db,
                onset=_value(onsets),
                duration=_value(durations),
                spectrum_db=spectrum,
                dtype=dtype,
            )
    return Scene(tuple(sources), timing), init


def _scaled_starts(scene: Scene, init: Params, f0_scales: tuple[float, ...]) -> list[Params]:
    """``init`` once per scale, every harmonic source's f0 multiplied by it; just
    ``init`` if the scene has no harmonic source."""
    harmonics = [source for source in scene.sources if isinstance(source, Harmonic)]
    if not harmonics or tuple(f0_scales) == (1.0,):
        return [init]
    starts = []
    for scale in f0_scales:
        start = dict(init)
        for source in harmonics:
            key = source.key("f0_mean")
            start[key] = freq_to_erb(erb_to_freq(init[key]) * scale)
        starts.append(start)
    return starts


def overlap(a: Candidate, b: Candidate) -> float:
    """The overlap of two events' time spans as a fraction of their union (IoU)."""
    shared = min(a.onset + a.duration, b.onset + b.duration) - max(a.onset, b.onset)
    union = max(a.onset + a.duration, b.onset + b.duration) - min(a.onset, b.onset)
    return max(shared, 0.0) / union


def _fits(source: SourcePlan, event: Candidate) -> bool:
    """Whether ``event`` can be ``source``'s next event: same kind, starting
    ``MIN_SPACING`` after its last, overlapping none of its events by more than ``MAX_OVERLAP``."""
    return (
        source.kind == event.kind
        and source.events[-1].onset + MIN_SPACING <= event.onset
        and all(overlap(e, event) <= MAX_OVERLAP for e in source.events)
    )


def _value(values: list[float]) -> float | list[float]:
    """One event's value as a number, several events' as a list (as ``initial`` takes them)."""
    return values if len(values) > 1 else values[0]


def _assignments(
    sources: list[SourcePlan],
    explanation: tuple[Candidate, ...],
    used: set[int],
    new: int,
    max_sources: int,
    max_new_sources: int | None,
) -> list[Plan]:
    """Every way to place ``explanation``'s events, each in an unused existing source of
    its kind or a new one."""
    if not explanation:
        return [tuple(sources)]
    event, rest = explanation[0], explanation[1:]
    out: list[Plan] = []
    for i, source in enumerate(sources):
        if i in used or not _fits(source, event):
            continue
        joined = SourcePlan(source.kind, (*source.events, event))
        out += _assignments(
            [*sources[:i], joined, *sources[i + 1 :]], rest, used | {i}, new, max_sources, max_new_sources
        )
    if len(sources) < max_sources and (max_new_sources is None or new < max_new_sources):
        grown = [*sources, SourcePlan(event.kind, (event,))]
        out += _assignments(grown, rest, used | {len(sources)}, new + 1, max_sources, max_new_sources)
    return out


def extensions(
    plan: Plan,
    group: OnsetGroup,
    *,
    max_sources: int,
    max_new_sources: int | None,
    max_new_events: int,
) -> list[Plan]:
    """Every way to add one of ``group``'s explanations to ``plan``, and ``plan`` itself
    (the group left out) unless it is empty.

    Each new event joins an existing source it fits (see :func:`_fits`), or
    starts a new source. ``max_new_sources``
    ``None`` leaves the number of new sources free (the first round).
    """
    out: list[Plan] = [plan] if plan else []  # the group left out
    for explanation in group.explanations:
        if len(explanation) <= max_new_events:
            out += _assignments(list(plan), explanation, set(), 0, max_sources, max_new_sources)
    return out


def cleanup_moves(plan: Plan) -> list[Plan]:
    """The paper's cleanup proposals for one hypothesis, in the order they are tried:
    merge two sources of a kind, change a source's kind, remove a source, merge
    two consecutive events of a source, remove an event."""
    moves: list[Plan] = []
    n = len(plan)
    for i in range(n):
        for j in range(i + 1, n):
            if plan[i].kind != plan[j].kind:
                continue
            events = tuple(sorted(plan[i].events + plan[j].events, key=lambda e: e.onset))
            if all(b.onset - a.onset >= MIN_SPACING for a, b in zip(events, events[1:], strict=False)):
                merged = SourcePlan(plan[i].kind, events)
                moves.append((*plan[:i], merged, *plan[i + 1 : j], *plan[j + 1 :]))
    for i, source in enumerate(plan):
        if source.kind == "whistle":
            events = tuple(dataclasses.replace(e, kind="harmonic", spectrum_db=(0.0,)) for e in source.events)
            changed = SourcePlan("harmonic", events)
        else:
            events = tuple(dataclasses.replace(e, kind="whistle", spectrum_db=None) for e in source.events)
            changed = SourcePlan("whistle", events)
        moves.append((*plan[:i], changed, *plan[i + 1 :]))
    if n > 1:
        moves += [(*plan[:i], *plan[i + 1 :]) for i in range(n)]
    for i, source in enumerate(plan):
        for j in range(len(source.events) - 1):
            a, b = source.events[j], source.events[j + 1]
            joined = dataclasses.replace(a, duration=b.onset + b.duration - a.onset)
            events = (*source.events[:j], joined, *source.events[j + 2 :])
            moves.append((*plan[:i], SourcePlan(source.kind, events), *plan[i + 1 :]))
    for i, source in enumerate(plan):
        if len(source.events) > 1:
            for j in range(len(source.events)):
                events = (*source.events[:j], *source.events[j + 1 :])
                moves.append((*plan[:i], SourcePlan(source.kind, events), *plan[i + 1 :]))
    return moves


@dataclass
class Scored:
    """One hypothesis fitted on the sound's first ``end`` seconds."""

    plan: Plan
    end: float
    result: SceneResult | None
    stage: str
    error: str = ""

    @property
    def screen(self) -> float:
        """The screening score: Laplace log evidence plus the structure prior
        (minus infinity when the evidence could not be estimated)."""
        return -math.inf if self.result is None else self.result.log_posterior_laplace

    @property
    def fitted(self) -> SceneResult:
        """The fit and evidence (only for hypotheses that were not left out)."""
        if self.result is None:
            raise ValueError(f"left out: {self.error}")
        return self.result


@dataclass
class SearchResult:
    """Everything the search fitted, and the final ranking."""

    groups: list[OnsetGroup]
    stages: list[list[Scored]] = field(default_factory=list)
    final: list[Scored] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def n_fits(self) -> int:
        """The number of hypotheses fitted."""
        return sum(len(stage) for stage in self.stages)

    @property
    def best(self) -> Scored:
        """The final ranking's first: highest variational log posterior (by the screen
        without variational steps)."""
        return self.final[0]


def search(
    waveform: torch.Tensor,
    cochleagram: Cochleagram,
    ramp: float,
    *,
    steps: int = 15000,
    tolerance: float | None = 0.5,
    beam: int = 2,
    keep: int = 5,
    max_sources: int = 4,
    max_new_sources: int = 2,
    max_new_events: int = 3,
    max_per_round: int = 20,
    max_residual_groups: int = 2,
    max_cleanup: int = 8,
    variational_steps: int = 2000,
    n_samples: int = 32,
    sigma: float = 10.0,
    infer_kernel: bool = False,
    generator: torch.Generator | None = None,
    report: Callable[[str], None] | None = None,
    detect_options: dict | None = None,
    f0_scales: tuple[float, ...] = (1.0,),
) -> SearchResult:
    """Search for the hypothesis that best explains ``waveform`` (see the module docstring).

    ``steps`` and ``tolerance`` set each MAP fit as in
    :func:`~sonore_infer.scene.evaluate`; ``n_samples`` is its importance
    sampling's (unused in the ranking). ``report`` receives progress messages.
    A hypothesis with a harmonic source is fitted once from each of
    ``f0_scales``, every harmonic source's f0 start multiplied by it, and the
    fit with the highest log joint is kept (as :func:`~sonore_infer.scene.evaluate`
    does with several starts). The just fifth's two-note fit reaches the
    notes' f0s only from about 1% above the detector's (results/C3-search.md).
    """
    started = time.perf_counter()
    fs = cochleagram.fs
    dtype = waveform.dtype
    total = waveform.shape[-1] / fs
    observed_all = cochleagram(waveform)
    groups = detect(observed_all, cochleagram, waveform.shape[-1], **(detect_options or {}))
    out = SearchResult(groups)
    say = report or (lambda message: None)

    def observed_until(end: float) -> torch.Tensor:
        return cochleagram(waveform[..., : round(end * fs)])

    def fit_all(plans: list[Plan], end: float, stage: str) -> list[Scored]:
        observed = observed_until(end)
        scored = []
        for number, plan in enumerate(plans, start=1):
            scene, init = build(plan, end, fs, ramp, infer_kernel=infer_kernel, dtype=dtype)
            say(f"{stage}: fitting {number} of {len(plans)}: {describe(plan)}")
            try:
                result = evaluate(
                    scene,
                    _scaled_starts(scene, init, f0_scales),
                    observed,
                    cochleagram,
                    steps=steps,
                    tolerance=tolerance,
                    n_samples=n_samples,
                    generator=generator,
                    sigma=sigma,
                )
            except (ValueError, RuntimeError) as error:
                # the Laplace estimate needs a peak; a fit that ends off one (a saddle,
                # or too few steps) leaves the hypothesis out, as in milestones (a) and (b)
                say(f"{stage}: left out ({type(error).__name__}): {describe(plan)}")
                scored.append(Scored(plan, end, None, stage, str(error).splitlines()[0]))
                continue
            scored.append(Scored(plan, end, result, stage))
        scored.sort(key=lambda s: -s.screen)
        out.stages.append(scored)
        return scored

    def unique(plans: list[Plan], seen: set | None = None) -> list[Plan]:
        seen = set() if seen is None else seen
        kept = []
        for plan in plans:
            key = plan_key(plan)
            if key not in seen:
                seen.add(key)
                kept.append(plan)
        return kept

    def capped(plans: list[Plan]) -> list[Plan]:
        ranked = sorted(plans, key=lambda p: (len(p), -sum(e.salience for s in p for e in s.events)))
        return ranked[:max_per_round]

    kept: list[Plan] = [()]
    whole: list[Scored] = []
    for index, group in enumerate(groups):
        end = groups[index + 1].onset if index + 1 < len(groups) else total
        plans = [
            extended
            for plan in kept
            for extended in extensions(
                plan,
                group,
                max_sources=max_sources,
                max_new_sources=None if index == 0 else max_new_sources,
                max_new_events=max_new_events,
            )
        ]
        plans = capped(unique(plans))
        if not plans:
            continue
        scored = fit_all(plans, end, f"round {index + 1} of {len(groups)}")
        kept = [s.plan for s in scored[:beam] if s.result is not None] or kept
        if end == total:
            whole += scored

    seen = {plan_key(s.plan) for s in whole}
    whole = [s for s in whole if s.result is not None]
    if whole:
        best = max(whole, key=lambda s: s.screen)
        scene, _ = build(best.plan, total, fs, ramp, infer_kernel=infer_kernel, dtype=dtype)
        rendered = cochleagram(scene.render(best.fitted.fit.params).detach())
        residual = (observed_all - rendered).clamp_min(0.0) + cochleagram.floor_db
        residual_groups = detect(residual, cochleagram, waveform.shape[-1], **(detect_options or {}))
        residual_groups = sorted(
            residual_groups, key=lambda g: -max(c.salience for e in g.explanations for c in e)
        )[:max_residual_groups]
        for number, group in enumerate(residual_groups, start=1):
            plans = [
                extended
                for s in sorted(whole, key=lambda s: -s.screen)[:beam]
                for extended in extensions(
                    s.plan,
                    group,
                    max_sources=max_sources,
                    max_new_sources=max_new_sources,
                    max_new_events=max_new_events,
                )
            ]
            plans = capped(unique(plans, seen))
            if plans:
                fitted = fit_all(plans, total, f"residual {number} of {len(residual_groups)}")
                whole += [s for s in fitted if s.result is not None]
        moves = [
            move for s in sorted(whole, key=lambda s: -s.screen)[:beam] for move in cleanup_moves(s.plan)
        ]
        moves = [move for move in unique([m for m in moves if m], seen)][:max_cleanup]
        if moves:
            whole += fit_all(moves, total, "cleanup")
    whole = [s for s in whole if s.result is not None]

    observed = observed_all
    finalists = sorted(whole, key=lambda s: -s.screen)[:keep]
    if not variational_steps:
        out.final = finalists
        out.seconds = time.perf_counter() - started
        return out
    for number, s in enumerate(finalists, start=1):
        say(f"variational evidence {number} of {len(finalists)}: {describe(s.plan)}")
        scene, _ = build(s.plan, total, fs, ramp, infer_kernel=infer_kernel, dtype=dtype)
        s.fitted.variational = variational_evidence(
            scene.log_joint(observed, cochleagram, sigma),
            s.fitted.fit.params,
            steps=variational_steps,
            generator=generator,
        )
    out.final = sorted(finalists, key=lambda s: -s.fitted.log_posterior_variational)
    out.seconds = time.perf_counter() - started
    return out
