"""Listener noise against Shepard's (1964) groups (design relative-pitch-listeners, P15 to P18).

Per pair, L is the model's log odds of up against down: the log posterior
mass of the hypotheses whose fitted shift is upward minus that of the rest
(as pair_summary.py's P(up)). A listener group judges up with probability

    λ/2 + (1 − λ) · logistic(β · L)

(decision P15 (A)), and the predicted fraction at a number of steps
clockwise is the mean over its 10 pairs. β and λ are fitted per group by
maximum binomial likelihood to Shepard's Fig. 3 fractions (read by eye),
with 26 listeners × 10 pairs per step for Group I and 14 × 10 for Group III
(decision P16). Variants: (A) β and λ per group, (B) β shared and λ per
group, (C) β per group and no lapse. Guards against overfitting (P16):
a baseline that ignores the model, λ/2 + (1 − λ) · logistic(a − b · steps);
leave-one-step-out prediction for (A) and the baseline; and Group I's and
III's predicted accuracy on adjacent pairs, against Shepard §IV's figures
for his first sequence of complex tones.

The fit is a grid search, so it is deterministic: β on a log grid from 1e-3
to 1, λ from 0 to 0.99, the baseline's slope b on a log grid and its
intercept a on a linear grid. β · L reads as a likelihood SD of 10/√β dB
(P15 (A)); that SD is printed for P18's check.

With ``--interval-noise`` (design relative-pitch-group-iii, P24 to P26) it
also fits noise on the interval between the two tones, after the scene
model. The listener hears a pair k steps clockwise as k + ε steps, with ε
Gaussian of SD s steps, wrapped on the circle of 10 steps. Two versions:

- **On nominal steps** (a second baseline): judged up when the heard
  interval is under 5 steps clockwise. One parameter, s.
- **On the model's log odds**: the heard interval moves the pair along the
  model's own L against steps from the same starting tone (linear between
  whole steps, L = 0 at 0 and 10 steps, where the two tones are the same),
  then λ/2 + (1 − λ) · logistic(β · L). Three parameters; s = 0 is (A).

Both are fitted on grids (s from 0 to 4 steps), with leave one step out and
the adjacent-pair check. It also prints the size of the dip at small steps:
accuracy folded over both directions and its standard error if every
presentation were an independent trial.

    python tools/relative_pitch/listener_noise.py results/data/P3/[0-9]*.txt results/data/P8/[0-9]*.txt
"""

import argparse
import math
from collections import defaultdict

import numpy as np
import pair_summary

GROUPS = {
    "I": (pair_summary.GROUP_I, 26),
    "III": (pair_summary.GROUP_III, 14),
}
PAIRS_PER_STEP = 10
STEPS = range(1, 10)
# Shepard §IV, first sequence of complex tones (adjacent pairs): fraction judged as the circle predicts
ADJACENT = {"I": "over 0.99", "II": "0.87 to 1.00 per pair", "III": "0.63 to 0.82 per pair"}
BETAS = np.exp(np.linspace(math.log(1e-3), 0.0, 401))
LAPSES = np.linspace(0.0, 0.99, 199)
SLOPES = np.exp(np.linspace(math.log(0.05), math.log(20.0), 241))
INTERCEPTS = np.linspace(-10.0, 30.0, 401)
LIKELIHOOD_SD = 10.0
CIRCLE = 10  # steps per octave
INTERVAL_SDS = np.linspace(0.0, 4.0, 81)
COARSE_BETAS = np.exp(np.linspace(math.log(1e-3), math.log(10.0), 161))
COARSE_LAPSES = np.linspace(0.0, 0.99, 100)
# Gauss–Hermite nodes and weights for averaging over ε ~ N(0, 1)
NODES, WEIGHTS = np.polynomial.hermite_e.hermegauss(41)
WEIGHTS = WEIGHTS / WEIGHTS.sum()


def logistic(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def pair_log_odds(paths):
    """{(first, second): (steps, L)} from pair.py outputs."""
    out = {}
    for key, (steps, hypotheses) in pair_summary.pairs(paths).items():
        up = [value for value, shift in hypotheses.values() if shift > 0]
        down = [value for value, shift in hypotheses.values() if shift <= 0]
        out[key] = (steps, logsumexp(up) - logsumexp(down))
    return out


def log_odds(paths):
    """{steps: [L of each pair]} from pair.py outputs."""
    by_steps = defaultdict(list)
    for steps, value in pair_log_odds(paths).values():
        by_steps[steps].append(value)
    return dict(by_steps)


def logsumexp(values):
    if not values:
        return -math.inf
    top = max(values)
    return top + math.log(sum(math.exp(value - top) for value in values))


def model_fractions(odds, steps, beta, lapse):
    """Predicted fraction judged up at each of ``steps``, for arrays of β and λ broadcast together."""
    out = []
    for step in steps:
        values = np.array(odds[step])
        p = logistic(np.multiply.outer(beta, values)).mean(axis=-1)
        out.append(lapse / 2 + (1 - lapse) * p)
    return np.stack(out, axis=-1)


def baseline_fractions(steps, intercept, slope, lapse):
    steps = np.array(list(steps), dtype=float)
    p = logistic(np.asarray(intercept)[..., None] - np.asarray(slope)[..., None] * steps)
    return np.asarray(lapse)[..., None] / 2 + (1 - np.asarray(lapse)[..., None]) * p


def log_likelihood(fractions, observed, trials):
    """Binomial log likelihood (without the constant) of ``observed`` fractions over the last axis."""
    p = np.clip(fractions, 1e-9, 1 - 1e-9)
    k = np.asarray(observed) * trials
    return (k * np.log(p) + (trials - k) * np.log(1 - p)).sum(axis=-1)


def fit_model(odds, observed, trials, steps=STEPS, lapse_free=True):
    """(β, λ, log likelihood) maximizing the binomial likelihood on the grid."""
    steps = list(steps)
    observed = np.array([observed[step - 1] for step in steps])
    lapses = LAPSES if lapse_free else np.zeros(1)
    fractions = model_fractions(odds, steps, BETAS[:, None], lapses[None, :])
    ll = log_likelihood(fractions, observed, trials)
    i, j = np.unravel_index(np.argmax(ll), ll.shape)
    return float(BETAS[i]), float(lapses[j]), float(ll[i, j])


def fit_shared_beta(odds, groups):
    """(β, {group: λ}, total log likelihood) with one β for all groups and a λ each."""
    total = np.zeros(len(BETAS))
    best_lapse = {}
    for name, (observed, listeners) in groups.items():
        fractions = model_fractions(odds, STEPS, BETAS[:, None], LAPSES[None, :])
        ll = log_likelihood(fractions, np.array(observed), listeners * PAIRS_PER_STEP)
        best_lapse[name] = LAPSES[np.argmax(ll, axis=1)]
        total += ll.max(axis=1)
    i = int(np.argmax(total))
    return float(BETAS[i]), {name: float(lapses[i]) for name, lapses in best_lapse.items()}, float(total[i])


def fit_baseline(observed, trials, steps=STEPS):
    """(a, b, λ, log likelihood) for the baseline that ignores the model."""
    steps = list(steps)
    observed = np.array([observed[step - 1] for step in steps])
    best = (0.0, 0.0, 0.0, -math.inf)
    for lapse in LAPSES:
        fractions = baseline_fractions(steps, INTERCEPTS[:, None], SLOPES[None, :], np.full((1, 1), lapse))
        ll = log_likelihood(fractions, observed, trials)
        i, j = np.unravel_index(np.argmax(ll), ll.shape)
        if ll[i, j] > best[3]:
            best = (float(INTERCEPTS[i]), float(SLOPES[j]), float(lapse), float(ll[i, j]))
    return best


def held_out(odds, observed, trials):
    """Summed log likelihood of each step predicted from fits to the other 8: (model A, baseline)."""
    model_total = baseline_total = 0.0
    for left_out in STEPS:
        kept = [step for step in STEPS if step != left_out]
        beta, lapse, _ = fit_model(odds, observed, trials, kept)
        intercept, slope, base_lapse, _ = fit_baseline(observed, trials, kept)
        target = np.array([observed[left_out - 1]])
        model_total += float(
            log_likelihood(model_fractions(odds, [left_out], np.array(beta), np.array(lapse)), target, trials)
        )
        baseline_total += float(
            log_likelihood(baseline_fractions([left_out], intercept, slope, base_lapse), target, trials)
        )
    return model_total, baseline_total


def normal_cdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def circle_fractions(steps, sd):
    """P(judged up) at each of ``steps`` when the heard interval is steps + N(0, sd), wrapped."""
    steps = np.array(list(steps), dtype=float)
    if sd == 0:
        return np.where((steps % CIRCLE > 0) & (steps % CIRCLE < CIRCLE / 2), 1.0, 0.0)
    total = np.zeros_like(steps)
    for turn in range(-5, 6):
        low = (turn * CIRCLE - steps) / sd
        high = (turn * CIRCLE + CIRCLE / 2 - steps) / sd
        total += normal_cdf(high) - normal_cdf(low)
    return total


def odds_curves(paths):
    """Array [starting tone, steps 0..10] of the model's L, 0 at 0 and 10 steps."""
    curves = np.zeros((CIRCLE, CIRCLE + 1))
    for (first, _second), (steps, value) in pair_log_odds(paths).items():
        curves[first - 1, steps] = value
    return curves


def blurred_odds(curves, steps, sd):
    """L at heard intervals step + sd · node: array [steps, starting tone, node]."""
    grid = np.arange(CIRCLE + 1)
    out = []
    for step in steps:
        heard = (step + sd * NODES) % CIRCLE
        out.append(np.array([np.interp(heard, grid, curve) for curve in curves]))
    return np.array(out)


def blurred_fractions(curves, steps, beta, lapse, sd):
    """Predicted fraction judged up at each of ``steps`` with interval noise on the model's L."""
    values = blurred_odds(curves, list(steps), sd)
    p = (logistic(beta * values) * WEIGHTS).sum(axis=-1).mean(axis=-1)
    return lapse / 2 + (1 - lapse) * p


def fit_circle(observed, trials, steps=STEPS):
    """(s, log likelihood) for interval noise on nominal steps."""
    steps = list(steps)
    observed = np.array([observed[step - 1] for step in steps])
    ll = [float(log_likelihood(circle_fractions(steps, sd), observed, trials)) for sd in INTERVAL_SDS]
    i = int(np.argmax(ll))
    return float(INTERVAL_SDS[i]), ll[i]


def fit_blurred(curves, observed, trials, steps=STEPS):
    """(β, λ, s, log likelihood) for interval noise on the model's L."""
    steps = list(steps)
    observed = np.array([observed[step - 1] for step in steps])
    best = (0.0, 0.0, 0.0, -math.inf)
    for sd in INTERVAL_SDS:
        values = blurred_odds(curves, steps, sd)
        # [β, steps]
        p = (logistic(np.multiply.outer(COARSE_BETAS, values)) * WEIGHTS).sum(axis=-1).mean(axis=-1)
        fractions = COARSE_LAPSES[None, :, None] / 2 + (1 - COARSE_LAPSES[None, :, None]) * p[:, None, :]
        ll = log_likelihood(fractions, observed, trials)
        i, j = np.unravel_index(np.argmax(ll), ll.shape)
        if ll[i, j] > best[3]:
            best = (float(COARSE_BETAS[i]), float(COARSE_LAPSES[j]), float(sd), float(ll[i, j]))
    return best


def held_out_interval(curves, observed, trials):
    """Summed held-out log likelihood: (noise on nominal steps, noise on L)."""
    circle_total = blurred_total = 0.0
    for left_out in STEPS:
        kept = [step for step in STEPS if step != left_out]
        target = np.array([observed[left_out - 1]])
        sd, _ = fit_circle(observed, trials, kept)
        circle_total += float(log_likelihood(circle_fractions([left_out], sd), target, trials))
        beta, lapse, sd, _ = fit_blurred(curves, observed, trials, kept)
        blurred_total += float(
            log_likelihood(blurred_fractions(curves, [left_out], beta, lapse, sd), target, trials)
        )
    return circle_total, blurred_total


def folded_accuracy(observed):
    """Fraction judged as the shorter way round predicts at 1 to 4 steps, both directions pooled."""
    return [(observed[step - 1] + 1 - observed[CIRCLE - step - 1]) / 2 for step in range(1, 5)]


def print_interval_noise(paths):
    curves = odds_curves(paths)
    print()
    print("interval noise after the scene model (design relative-pitch-group-iii, P24 to P26)")
    print(
        "version            group  s_steps  beta     lapse  log_likelihood  held_out"
        "  fitted fractions at 1-9 steps"
    )
    fits = {}
    for name, (observed, listeners) in GROUPS.items():
        trials = listeners * PAIRS_PER_STEP
        circle_held, blurred_held = held_out_interval(curves, observed, trials)
        sd, ll = fit_circle(observed, trials)
        fractions = circle_fractions(STEPS, sd)
        fits[("nominal steps", name)] = fractions
        print(
            f"{'nominal steps':17s}  {name:5s}  {sd:7.2f}  {'':6s}  {'':5s}  {ll:14.1f}  {circle_held:8.1f}  "
            + " ".join(f"{value:.2f}" for value in fractions)
        )
        beta, lapse, sd, ll = fit_blurred(curves, observed, trials)
        fractions = blurred_fractions(curves, STEPS, beta, lapse, sd)
        fits[("model log odds", name)] = fractions
        print(
            f"{'model log odds':17s}  {name:5s}  {sd:7.2f}  {beta:.4f}  {lapse:.3f}  {ll:14.1f}"
            f"  {blurred_held:8.1f}  " + " ".join(f"{value:.2f}" for value in fractions)
        )
    print()
    print("adjacent pairs, predicted fraction as the circle predicts (1 step up, 1 step down):")
    for (version, name), fractions in fits.items():
        print(
            f"  {version:14s}  Group {name}: {fractions[0]:.2f}, {1 - fractions[8]:.2f};"
            f" Shepard {ADJACENT[name]}"
        )
    print()
    print("accuracy at 1-4 steps, both directions pooled, and z of each against 1 step if every")
    print("presentation were independent (2 x 10 pairs x listeners per step):")
    for name, (observed, listeners) in GROUPS.items():
        trials = 2 * listeners * PAIRS_PER_STEP
        accuracy = folded_accuracy(observed)
        first = accuracy[0]
        parts = []
        for value in accuracy:
            se = math.sqrt(first * (1 - first) / trials + value * (1 - value) / trials)
            parts.append(f"{value:.3f} (z {(value - first) / se:+.1f})" if se > 0 else f"{value:.3f}")
        print(f"  Group {name}: " + ", ".join(parts))
    observed = GROUPS["III"][0]
    trials = GROUPS["III"][1] * PAIRS_PER_STEP
    p8, p9 = observed[7], observed[8]
    se = math.sqrt(p8 * (1 - p8) / trials + p9 * (1 - p9) / trials)
    print(f"  Group III, 9 against 8 steps: {p9:.2f} against {p8:.2f}, z {(p9 - p8) / se:+.1f}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="+", help="pair.py outputs (P3, and P8 for the extra readings)")
    parser.add_argument(
        "--refits",
        nargs="+",
        default=[],
        help="pair.py outputs of the same hypotheses at Group I's implied likelihood SD (decision P18)",
    )
    parser.add_argument(
        "--interval-noise",
        action="store_true",
        help="also fit noise on the interval (design relative-pitch-group-iii, P24 to P26)",
    )
    args = parser.parse_args()
    odds = log_odds(args.paths)
    print("model log odds of up against down, by steps clockwise (each pair):")
    for step in STEPS:
        print(f"  {step}: " + " ".join(f"{value:+.1f}" for value in sorted(odds[step])))
    print()
    print("variant  group  beta     lapse  likelihood_sd_db  log_likelihood  fitted fractions at 1-9 steps")
    fits = {}
    for name, (observed, listeners) in GROUPS.items():
        trials = listeners * PAIRS_PER_STEP
        for variant, lapse_free in (("A", True), ("C", False)):
            beta, lapse, ll = fit_model(odds, observed, trials, lapse_free=lapse_free)
            fits[(variant, name)] = (beta, lapse)
            fractions = model_fractions(odds, STEPS, np.array(beta), np.array(lapse))
            print(
                f"{variant:7s}  {name:5s}  {beta:.4f}  {lapse:.3f}  {LIKELIHOOD_SD / math.sqrt(beta):16.1f}"
                f"  {ll:14.1f}  " + " ".join(f"{value:.2f}" for value in fractions)
            )
    beta, lapses, total = fit_shared_beta(odds, GROUPS)
    for name, lapse in lapses.items():
        fractions = model_fractions(odds, STEPS, np.array(beta), np.array(lapse))
        print(
            f"{'B':7s}  {name:5s}  {beta:.4f}  {lapse:.3f}  {LIKELIHOOD_SD / math.sqrt(beta):16.1f}"
            f"  {'':14s}  " + " ".join(f"{value:.2f}" for value in fractions)
        )
    print(f"B total log likelihood (both groups): {total:.1f}")
    print()
    print("baseline (ignores the model): group  a  b  lapse  log_likelihood  fitted fractions")
    for name, (observed, listeners) in GROUPS.items():
        intercept, slope, lapse, ll = fit_baseline(observed, listeners * PAIRS_PER_STEP)
        fractions = baseline_fractions(STEPS, intercept, slope, lapse)
        print(
            f"  {name:5s}  {intercept:.2f}  {slope:.3f}  {lapse:.3f}  {ll:.1f}  "
            + " ".join(f"{value:.2f}" for value in fractions)
        )
    print()
    print("leave one step out, summed held-out log likelihood: group  model_A  baseline")
    for name, (observed, listeners) in GROUPS.items():
        model_total, baseline_total = held_out(odds, observed, listeners * PAIRS_PER_STEP)
        print(f"  {name:5s}  {model_total:.1f}  {baseline_total:.1f}")
    print()
    print(
        "adjacent pairs (1 step either way), predicted fraction as the circle predicts, against Shepard §IV:"
    )
    for name in GROUPS:
        beta, lapse = fits[("A", name)]
        up = model_fractions(odds, [1], np.array(beta), np.array(lapse))[0]
        down = 1 - model_fractions(odds, [9], np.array(beta), np.array(lapse))[0]
        print(f"  Group {name}: 1 step up {up:.2f}, 1 step down {down:.2f}; Shepard {ADJACENT[name]}")
    print(f"  Group II: not fitted; Shepard {ADJACENT['II']}")
    if args.refits:
        beta = fits[("A", "I")][0]
        print()
        print(f"P18: log odds at {LIKELIHOOD_SD:g} dB times Group I's beta {beta:.4f}, against a refit at")
        print(
            f"{LIKELIHOOD_SD / math.sqrt(beta):.1f} dB: first  second  steps  L_10db  beta_times_L  L_refit"
        )
        base = pair_log_odds(args.paths)
        for key, (steps, value) in sorted(pair_log_odds(args.refits).items()):
            print(
                f"  {key[0]:5d}  {key[1]:6d}  {steps:5d}  {base[key][1]:+7.1f}"
                f"  {beta * base[key][1]:+12.2f}  {value:+7.2f}"
            )
    if args.interval_noise:
        print_interval_noise(args.paths)


if __name__ == "__main__":
    main()
