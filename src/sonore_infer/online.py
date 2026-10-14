"""Online inference: the scene's belief updated window by window (docs/design/online.md).

The sound arrives in windows of ``window`` seconds. After each, the
hypothesis is extended to the sound so far (its grid grows, and sources and
events enter when their tones' onsets arrive, design O7), a few Adam steps
move the latents that are still free, and the window's term of the evidence
is added to a running total.

**Fixed lag (O3 (B)).** Grid deviations older than ``lag`` seconds, and the
timing of events that ended before then, are frozen at their current values;
their prior terms stay in the log joint, so the free latents are conditioned
on them exactly (O4 (A)). Trajectory means, and GP kernel latents when
inferred, are never frozen: they belong to the whole sound, like a filter's
static parameters [Särkkä 2013, §12.1].

**Evidence (O5 (B)).** The evidence of the sound so far is the product of
each window's predictive likelihood, ``log p(x_1:k) = sum_k log p(x_k |
x_1:k-1)`` [Särkkä 2013, Thm 12.1]. With ``F`` the latents free at window
``k`` and the rest held at their values, each term is estimated as

    log Z_k(F) - log Z_k-1(F'),     log Z(F) = Laplace estimate of
                                    log integral p(x, theta) d theta_F,

where ``F'`` is ``F`` without the latents that window ``k`` added. ``Z_k-1``
is evaluated at the previous window's mode, where its Hessian is already
known: it is the previous window's Hessian restricted to ``F'``. So each
window costs one Hessian, over the free latents only. With an infinite lag
the terms telescope, and the total is the batch Laplace estimate of the
whole prefix at the final mode; with a finite lag the frozen past enters as
a point value instead of being integrated out, which is the approximation
whose cost the experiment measures.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from sonore_infer.cochleagram import Cochleagram, gaussian_log_likelihood
from sonore_infer.priors import GRID_STEP
from sonore_infer.scene import Params, Scene, SceneTiming, Whistle

Free = dict[str, torch.Tensor]
"""Which entries of each parameter are free: a boolean tensor of the parameter's
shape per name. A name that is missing is frozen."""

Coordinate = tuple[str, int]
"""One free entry: the parameter's name and its index in the flattened tensor.
Tensors only grow at the end as the sound arrives, so a coordinate keeps its
key from window to window."""


def coordinates(params: Params, free: Free) -> list[Coordinate]:
    keys = []
    for name, mask in free.items():
        if params[name].shape != mask.shape:
            raise ValueError(f"{name}: mask shape {tuple(mask.shape)} != {tuple(params[name].shape)}")
        keys += [(name, int(i)) for i in torch.nonzero(mask.reshape(-1)).flatten()]
    return keys


def free_vector(
    params: Params, free: Free
) -> tuple[torch.Tensor, list[Coordinate], Callable[[torch.Tensor], Params]]:
    """The free entries as one vector, their keys, and the function putting a vector back."""
    keys = coordinates(params, free)
    base = {name: value.detach() for name, value in params.items()}
    if not keys:
        dtype = next(iter(base.values())).dtype if base else torch.float64
        return torch.zeros(0, dtype=dtype), keys, lambda flat: dict(base)
    vector = torch.stack([base[name].reshape(-1)[i] for name, i in keys])
    by_name: dict[str, list[tuple[int, int]]] = {}
    for position, (name, i) in enumerate(keys):
        by_name.setdefault(name, []).append((position, i))

    def unflatten(flat: torch.Tensor) -> Params:
        out = dict(base)
        for name, pairs in by_name.items():
            positions = torch.tensor([p for p, _ in pairs])
            indices = torch.tensor([i for _, i in pairs])
            out[name] = (
                base[name].reshape(-1).index_put((indices,), flat[positions]).reshape(base[name].shape)
            )
        return out

    return vector, keys, unflatten


@dataclass
class LaplaceState:
    """A Laplace estimate over the free latents, with what the next window needs from it."""

    peak: float
    keys: list[Coordinate]
    hessian: torch.Tensor  # of the negative log joint over ``keys``
    min_eigenvalue: float = 1e-6

    @property
    def value(self) -> float:
        return self.restricted(self.keys)[0]

    def restricted(self, keys: Sequence[Coordinate]) -> tuple[float, int]:
        """The Laplace estimate over ``keys`` only, the others held at the mode, and how
        many eigenvalues were raised to the floor."""
        position = {key: p for p, key in enumerate(self.keys)}
        missing = [key for key in keys if key not in position]
        if missing:
            raise ValueError(f"not free in the previous window: {missing[:3]}")
        index = torch.tensor([position[key] for key in keys], dtype=torch.long)
        if index.numel() == 0:
            return self.peak, 0
        hessian = self.hessian[index][:, index]
        eigenvalues = torch.linalg.eigvalsh(0.5 * (hessian + hessian.T))
        floored = int((eigenvalues < self.min_eigenvalue).sum())
        log_det = eigenvalues.clamp_min(self.min_eigenvalue).log().sum().item()
        return self.peak + 0.5 * len(keys) * math.log(2 * math.pi) - 0.5 * log_det, floored


def laplace(
    log_joint: Callable[[Params], torch.Tensor], mode: Params, free: Free, min_eigenvalue: float = 1e-6
) -> LaplaceState:
    """The Laplace state of ``log_joint`` at ``mode`` over the free entries."""
    vector, keys, unflatten = free_vector(mode, free)

    def negative(flat):
        return -log_joint(unflatten(flat))

    with torch.no_grad():
        peak = -negative(vector).item()
    if keys:
        hessian = torch.autograd.functional.hessian(negative, vector)
    else:
        hessian = torch.zeros(0, 0, dtype=vector.dtype)
    return LaplaceState(peak, keys, hessian.detach(), min_eigenvalue)


def adam_free(
    log_joint: Callable[[Params], torch.Tensor],
    init: Params,
    free: Free,
    learning_rates: dict[str, float],
    steps: int,
    patience: int | None = None,
    tolerance: float = 1.0,
) -> tuple[Params, list[float]]:
    """Adam on the free entries of ``init`` for ``steps`` steps; the best point seen and the history.

    A fresh optimizer each call: frozen entries get no gradient, so they never move.
    With ``patience``, ``steps`` is a maximum, as in :func:`sonore_infer.fit.fit`:
    the fit stops once the best value has risen by less than ``tolerance``
    over the last ``patience`` steps, so a surprising window gets more steps
    than a predictable one (design decision O9).
    """
    params = {name: value.detach().clone() for name, value in init.items()}
    names = [name for name in free if bool(free[name].any())]
    for name in names:
        params[name].requires_grad_()
    best, best_value, history = {k: v.detach().clone() for k, v in params.items()}, -math.inf, []
    if not names:
        return best, [log_joint(params).item()]
    optimizer = torch.optim.Adam([{"params": [params[name]], "lr": learning_rates[name]} for name in names])
    for _ in range(steps + 1):
        optimizer.zero_grad()
        value = log_joint(params)
        history.append(value.item())
        if value.item() > best_value:
            best_value = value.item()
            best = {k: v.detach().clone() for k, v in params.items()}
        if patience is not None and len(history) > patience:
            if best_value - max(history[:-patience]) < tolerance:
                break
        (-value).backward()
        for name in names:
            params[name].grad.mul_(free[name].to(params[name].dtype))
        optimizer.step()
    return best, history


@dataclass
class WindowTerm:
    end: float  # s
    term: float  # log p(x_k | x_1:k-1), estimated
    total: float  # the running log evidence
    n_free: int
    floored: int
    log_joint: float  # at the mode
    seconds: float
    adam_steps: int = 0


class FixedLagEvidence:
    """The running sum of the windows' terms (O5 (B)); see the module docstring."""

    def __init__(self, initial: float = 0.0):
        self.total = initial
        self.previous: LaplaceState | None = None
        self.initial = initial

    def add(self, state: LaplaceState) -> tuple[float, int]:
        """Add window ``k``'s term, given its Laplace state; returns the term and floored eigenvalues."""
        if self.previous is None:
            before = self.initial
        else:
            old = set(self.previous.keys)
            before, _ = self.previous.restricted([key for key in state.keys if key in old])
        after, floored = state.restricted(state.keys)
        self.previous = state
        term = after - before
        self.total += term
        return term, floored


# --- scenes ---------------------------------------------------------------


@dataclass(frozen=True)
class Tone:
    onset: float  # s
    duration: float  # s
    freq: float  # Hz
    level_db: float


@dataclass(frozen=True)
class WhistleSpec:
    """A whistle source of a hypothesis and the tones it is started from, in time order."""

    name: str
    tones: tuple[Tone, ...]


Hypothesis = tuple[WhistleSpec, ...]


def _scalar_or_list(values: list[float]):
    return values[0] if len(values) == 1 else values


def prefix_scene(hypothesis: Hypothesis, end: float, fs: float, ramp: float) -> tuple[Scene | None, Params]:
    """The hypothesis on the sound's first ``end`` seconds, with the tones whose onsets
    have arrived, and its batch-style initialization (as ``tools/compare_aba.py``)."""
    timing = SceneTiming(fs=fs, total_duration=end, ramp=ramp)
    sources, init = [], {}
    for spec in hypothesis:
        tones = [tone for tone in spec.tones if tone.onset < end]
        if not tones:
            continue
        source = Whistle(spec.name, n_events=len(tones))
        sources.append(source)
        init |= source.initial(
            timing,
            freq=_scalar_or_list([tone.freq for tone in tones]),
            level_db=tones[0].level_db,
            onset=_scalar_or_list([tone.onset for tone in tones]),
            duration=_scalar_or_list([tone.duration for tone in tones]),
        )
    if not sources:
        return None, {}
    return Scene(tuple(sources), timing), init


def extend(scene: Scene, fresh: Params, previous: Params, hypothesis: Hypothesis) -> Params:
    """The new window's starting point: the previous mode where it exists, ``fresh`` elsewhere.

    New grid points start at the fresh trajectory's value (the current tone's
    frequency and level), measured from the previous mean. A new event's rest
    is chosen so it starts at its tone's onset given the previous events'
    current onsets and durations.
    """
    params = {}
    for source in scene.sources:
        p = source.key
        if p("onset") not in previous:  # a new source
            params |= {name: value for name, value in fresh.items() if name.startswith(source.name + ".")}
            continue
        for gp in ("freq", "level"):
            mean, deviation = previous[p(f"{gp}_mean")], previous[p(f"{gp}_deviation")]
            fresh_total = fresh[p(f"{gp}_mean")] + fresh[p(f"{gp}_deviation")]
            n_old = deviation.shape[0]
            params[p(f"{gp}_mean")] = mean.clone()
            params[p(f"{gp}_deviation")] = torch.cat([deviation, fresh_total[n_old:] - mean])
        params[p("onset")] = previous[p("onset")].clone()
        old_durations = previous[p("log_duration")].reshape(-1)
        n_old = old_durations.shape[0]
        if source.n_events == 1:
            params[p("log_duration")] = previous[p("log_duration")].clone()
            continue
        durations = torch.cat([old_durations, fresh[p("log_duration")][n_old:]])
        old_rests = previous.get(p("log_rest"), torch.zeros(0, dtype=durations.dtype))
        rests = [old_rests]
        tones = next(spec for spec in hypothesis if spec.name == source.name).tones
        onset = previous[p("onset")].item()
        for j in range(n_old - 1):
            onset += old_durations[j].exp().item() + old_rests[j].exp().item()
        for j in range(n_old - 1, source.n_events - 1):
            end_of_j = onset + durations[j].exp().item()
            rest = tones[j + 1].onset - end_of_j
            if rest <= 0:
                rest = fresh[p("log_rest")][j].exp().item()
            rests.append(torch.tensor([math.log(rest)], dtype=durations.dtype))
            onset = end_of_j + rest
        params[p("log_duration")] = durations
        params[p("log_rest")] = torch.cat(rests)
    return params


def lag_free(scene: Scene, params: Params, end: float, lag: float) -> Free:
    """What is free at ``end`` with a lag of ``lag`` seconds (``math.inf``: everything)."""
    free = {}
    start = end - lag
    for source in scene.sources:
        p = source.key
        names = source.learning_rates()
        n_grid = params[p("freq_deviation")].shape[0]
        recent = torch.arange(n_grid, dtype=torch.float64) * GRID_STEP >= start - 1e-9
        onsets, durations = source.event_times(params, scene.timing)
        event_free = [float(o + d) >= start for o, d in zip(onsets, durations, strict=True)]
        for name in names:
            short = name.split(".", 1)[1]
            if short.endswith("_deviation"):
                free[name] = recent.clone()
            elif short == "onset":
                free[name] = torch.tensor(event_free[0])
            elif short == "log_duration":
                mask = torch.tensor(event_free)
                free[name] = mask.reshape(params[name].shape)
            elif short == "log_rest":
                free[name] = torch.tensor(event_free[1:])
            else:  # means and kernel latents: the whole sound's
                free[name] = torch.ones(params[name].shape, dtype=torch.bool)
    return free


def silence_log_likelihood(observed: torch.Tensor, cochleagram: Cochleagram, n_samples: int, sigma: float):
    """The log likelihood of ``observed`` under a scene with no sources."""
    if observed.shape[-1] == 0:
        return 0.0
    silent = cochleagram(torch.zeros(n_samples, dtype=observed.dtype))
    return gaussian_log_likelihood(observed, silent, sigma).item()


@dataclass
class OnlineStep:
    """One window of one hypothesis's online run."""

    window: WindowTerm
    structure_log_prior: float
    params: Params = field(repr=False)

    @property
    def log_posterior(self) -> float:
        """Unnormalized log posterior of the hypothesis on the sound so far."""
        return self.window.total + self.structure_log_prior


def run_online(
    waveform: torch.Tensor,
    hypothesis: Hypothesis,
    cochleagram: Cochleagram,
    *,
    fs: float,
    ramp: float,
    window: float = 0.05,
    lag: float = 1.0,
    steps: int = 30,
    patience: int | None = None,
    tolerance: float = 1.0,
    sigma: float = 10.0,
    ends: Sequence[float] | None = None,
    report: Callable[[OnlineStep], None] | None = None,
) -> list[OnlineStep]:
    """One pass over ``waveform``, updating ``hypothesis`` every ``window`` seconds.

    ``ends`` overrides the window ends [s]; by default they are multiples of
    ``window`` up to the waveform's duration (and the duration itself).
    Before the first tone the scene is silence, with no latents, so its log
    likelihood is the starting total. With ``patience``, ``steps`` is the
    most Adam steps a window may take (see :func:`adam_free`).
    """
    total = waveform.shape[-1] / fs
    if ends is None:
        n = int(math.floor(total / window + 1e-9))
        ends = [round((k + 1) * window, 9) for k in range(n)]
        if total - ends[-1] > 1e-9:
            ends.append(total)
    evidence: FixedLagEvidence | None = None
    previous: Params = {}
    steps_out: list[OnlineStep] = []
    last_silent_end = 0.0
    for end in ends:
        begin = time.perf_counter()
        n_samples = round(end * fs)
        if n_samples < cochleagram.frame_samples:
            continue  # not one whole frame yet: nothing heard, nothing to update
        observed = cochleagram(waveform[:n_samples])
        scene, fresh = prefix_scene(hypothesis, end, fs, ramp)
        if scene is None:
            last_silent_end = end
            continue
        if evidence is None:
            n_silent = round(last_silent_end * fs)
            observed_silent = (
                cochleagram(waveform[:n_silent]) if n_silent >= cochleagram.frame_samples else None
            )
            initial = (
                0.0
                if observed_silent is None
                else silence_log_likelihood(observed_silent, cochleagram, n_silent, sigma)
            )
            evidence = FixedLagEvidence(initial)
        init = extend(scene, fresh, previous, hypothesis) if previous else fresh
        free = lag_free(scene, init, end, lag)
        log_joint = scene.log_joint(observed, cochleagram, sigma)
        mode, history = adam_free(log_joint, init, free, scene.learning_rates(), steps, patience, tolerance)
        # what is free is decided before the fit and kept: the evidence's terms need one set
        state = laplace(log_joint, mode, free)
        term, floored = evidence.add(state)
        step = OnlineStep(
            WindowTerm(
                end,
                term,
                evidence.total,
                len(state.keys),
                floored,
                state.peak,
                time.perf_counter() - begin,
                len(history) - 1,
            ),
            scene.structure_log_prior(),
            mode,
        )
        steps_out.append(step)
        if report:
            report(step)
        previous = mode
    return steps_out


def leaky_log_odds(
    ends: Sequence[float], terms_a: Sequence[float], terms_b: Sequence[float], tau: float
) -> list[float]:
    """The leaky decision variable of design O10, after each window: the
    differences of the two hypotheses' window terms (a minus b), each
    weighted by ``exp(-age / tau)``, summed. ``math.inf`` gives the full sum.
    The structure prior's log odds are not included."""
    out, value, previous = [], 0.0, None
    for end, a, b in zip(ends, terms_a, terms_b, strict=True):
        if previous is not None and math.isfinite(tau):
            value *= math.exp(-(end - previous) / tau)
        value += a - b
        out.append(value)
        previous = end
    return out


def batch_laplace(
    scene: Scene, mode: Params, observed: torch.Tensor, cochleagram: Cochleagram, sigma: float = 10.0
) -> LaplaceState:
    """The batch Laplace state of the whole prefix (every latent free), for O5 (A)."""
    free = {name: torch.ones(value.shape, dtype=torch.bool) for name, value in mode.items()}
    return laplace(scene.log_joint(observed, cochleagram, sigma), mode, free)


__all__ = [
    "FixedLagEvidence",
    "LaplaceState",
    "OnlineStep",
    "Tone",
    "WhistleSpec",
    "adam_free",
    "batch_laplace",
    "extend",
    "lag_free",
    "leaky_log_odds",
    "laplace",
    "prefix_scene",
    "run_online",
]
