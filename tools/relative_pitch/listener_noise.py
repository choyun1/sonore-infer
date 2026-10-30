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


if __name__ == "__main__":
    main()
