"""Where an online window's time goes, as the sound so far grows (design online, step 4).

For the buildup stimulus (App. C.11) and each hypothesis, at several ends of
the sound so far, times the parts of one online window as ``run_online``
runs them: the log joint (render, cochleagram, likelihood, prior), one Adam
step (log joint and its gradient), and the Laplace Hessian over the latents
free with the given lag. The scene is at its batch initialization from the
tones, which costs the same as any other point. Each time is the median of
``--repeats`` runs, on however many threads torch uses.

    python tools/online/online_cost.py --semitones 8 --ends 0.55 1.05 2.05 3.05
"""

import argparse
import statistics
import time

import torch
from online_buildup import LEAD, hypotheses

from sonore_infer.cochleagram import FFTCochleagram, gaussian_log_likelihood
from sonore_infer.online import free_vector, lag_free, laplace, prefix_scene
from sonore_infer.stimuli import buildup as bu

DTYPE = torch.float64


def median_seconds(function, repeats):
    times = []
    for _ in range(repeats):
        begin = time.perf_counter()
        function()
        times.append(time.perf_counter() - begin)
    return statistics.median(times)


def window_costs(scene, params, observed, cochleagram, end, lag, repeats):
    """Seconds for the parts of one window at ``params``, and the number of free latents."""
    params = {key: value.detach().clone() for key, value in params.items()}
    free = lag_free(scene, params, end, lag)
    log_joint = scene.log_joint(observed, cochleagram)
    with torch.no_grad():
        rendered = scene.render(params)
        heard = cochleagram(rendered)
        render = median_seconds(lambda: scene.render(params), repeats)
        analysis = median_seconds(lambda: cochleagram(rendered), repeats)
        likelihood = median_seconds(lambda: gaussian_log_likelihood(observed, heard, 10.0), repeats)
        prior = median_seconds(lambda: scene.log_prior(params), repeats)
        forward = median_seconds(lambda: log_joint(params), repeats)

    def adam_step():
        leaves = {key: value.clone().requires_grad_(True) for key, value in params.items()}
        log_joint(leaves).backward()

    step = median_seconds(adam_step, repeats)
    hessian = median_seconds(lambda: laplace(log_joint, params, free), max(1, repeats // 5))
    n_free = free_vector(params, free)[0].numel()
    return n_free, render, analysis, likelihood, prior, forward, step, hessian


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--semitones", type=float, default=8)
    parser.add_argument("--ends", type=float, nargs="+", default=[0.55, 1.05, 2.05, 3.05], help="s")
    parser.add_argument("--lag", type=float, default=1.0, help="s")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    triplets = 6
    cochleagram = FFTCochleagram()
    waveform = torch.tensor(bu.buildup_stimulus(triplets, args.semitones, lead=LEAD).data[:, 0], dtype=DTYPE)
    print(
        f"{args.semitones:g} st, lag {args.lag:g} s, {torch.get_num_threads()} threads, "
        f"median of {args.repeats}; times in ms (Hessian in s)"
    )
    print(
        "cost: hypothesis  end  n_grid  n_free  render  cochleagram  likelihood  prior"
        "  log_joint  adam_step  hessian"
    )
    for end in args.ends:
        observed = cochleagram(waveform[: round(end * bu.FS)])
        for name, spec in hypotheses(args.semitones, triplets).items():
            scene, params = prefix_scene(spec, end, bu.FS, bu.RAMP, heard=0.02)
            costs = window_costs(scene, params, observed, cochleagram, end, args.lag, args.repeats)
            n_free, render, analysis, likelihood, prior, forward, step, hessian = costs
            n_grid = next(value.shape[0] for key, value in params.items() if key.endswith("freq_deviation"))
            print(
                f"cost: {name}  {end:.2f}  {n_grid}  {n_free}  {1000 * render:.1f}  {1000 * analysis:.1f}"
                f"  {1000 * likelihood:.2f}  {1000 * prior:.1f}  {1000 * forward:.1f}  {1000 * step:.1f}"
                f"  {hessian:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
