"""Online inference (docs/design/online.md): the evidence's window terms and the fixed lag."""

import math

import pytest
import torch

from sonore_infer.cochleagram import FFTCochleagram
from sonore_infer.online import (
    FixedLagEvidence,
    Tone,
    WhistleSpec,
    batch_laplace,
    free_vector,
    laplace,
    prefix_scene,
    run_online,
)
from sonore_infer.stimuli import buildup as bu

F64 = torch.float64


def toy_run(lag, n_grid=120, window=5, lengthscale=10.0, noise=0.5, jitter=1e-6):
    """A Gaussian process observed with noise, point by point, in windows of ``window`` points.

    Every window, the free points are moved to the mode by one Newton step
    (exact for this quadratic log joint), and the window's term added.
    Returns (accumulated, exact) log evidence after each window.
    """
    t = torch.arange(n_grid, dtype=F64)
    kernel = torch.exp(-0.5 * ((t[:, None] - t[None, :]) / lengthscale) ** 2) + jitter * torch.eye(n_grid)
    generator = torch.Generator().manual_seed(0)
    z = torch.linalg.cholesky(kernel) @ torch.randn(n_grid, generator=generator, dtype=F64)
    y = z + noise * torch.randn(n_grid, generator=generator, dtype=F64)
    evidence, current, out = FixedLagEvidence(0.0), torch.zeros(0, dtype=F64), []
    for n in range(window, n_grid + 1, window):
        prior = torch.distributions.MultivariateNormal(torch.zeros(n, dtype=F64), kernel[:n, :n])

        def log_joint(params, n=n, prior=prior):
            likelihood = torch.distributions.Normal(params["z"], noise).log_prob(y[:n]).sum()
            return prior.log_prob(params["z"]) + likelihood

        params = {"z": torch.cat([current, torch.zeros(window, dtype=F64)])}
        free = {"z": t[:n] >= n - lag}
        state = laplace(log_joint, params, free)
        vector, _, unflatten = free_vector(params, free)
        gradient = torch.func.grad(lambda flat, unflatten=unflatten: log_joint(unflatten(flat)))(vector)
        params = unflatten(vector + torch.linalg.solve(state.hessian, gradient))
        evidence.add(laplace(log_joint, params, free))
        current = params["z"]
        marginal = torch.distributions.MultivariateNormal(
            torch.zeros(n, dtype=F64), kernel[:n, :n] + noise**2 * torch.eye(n)
        )
        out.append((evidence.total, marginal.log_prob(y[:n]).item()))
    return out


def test_window_terms_sum_to_the_exact_evidence_with_no_lag():
    for accumulated, exact in toy_run(math.inf):
        assert accumulated == pytest.approx(exact, abs=1e-8)


def test_a_lag_of_several_lengthscales_is_nearly_exact():
    # lengthscale 10 points: a lag of 6 lengthscales is exact to rounding, 3 to 0.1
    for lag, tolerance in ((60, 1e-3), (30, 0.1)):
        for accumulated, exact in toy_run(lag):
            assert accumulated == pytest.approx(exact, abs=tolerance)


def test_a_lag_shorter_than_the_lengthscale_is_not():
    # the frozen past enters as a point, and the squared-exponential kernel
    # extrapolates it almost exactly: the window terms are overconfident
    accumulated, exact = toy_run(5)[-1]
    assert accumulated < exact - 50


TONES = (Tone(0.05, 0.05, 1000.0, 60.0), Tone(0.175, 0.05, 1000.0, 60.0))


def short_run(lag, steps=5):
    waveform = torch.tensor(bu.buildup_stimulus(1, 8, lead=0.05).data[:, 0], dtype=F64)[: round(0.3 * bu.FS)]
    cochleagram = FFTCochleagram()
    spec = (WhistleSpec("w0", TONES),)
    steps_out = run_online(
        waveform, spec, cochleagram, fs=bu.FS, ramp=bu.RAMP, window=0.05, lag=lag, steps=steps
    )
    return waveform, cochleagram, spec, steps_out


def test_with_no_lag_the_online_total_is_the_batch_laplace_at_the_final_mode():
    waveform, cochleagram, spec, steps = short_run(math.inf)
    last = steps[-1]
    scene, _ = prefix_scene(spec, last.window.end, bu.FS, bu.RAMP)
    state = batch_laplace(scene, last.params, cochleagram(waveform), cochleagram)
    assert last.window.total == pytest.approx(state.value, abs=1e-6)


def test_frozen_latents_do_not_move():
    _, _, _, steps = short_run(0.1)
    for before, after in zip(steps, steps[1:], strict=False):
        old = before.params["w0.freq_deviation"]
        new = after.params["w0.freq_deviation"][: old.shape[0]]
        end = after.window.end
        frozen = torch.arange(old.shape[0], dtype=F64) * 0.01 < end - 0.1 - 1e-9
        assert torch.equal(new[frozen], old[frozen])
        assert not torch.equal(new[~frozen], old[~frozen])


def test_events_enter_when_their_onsets_arrive():
    _, _, _, steps = short_run(math.inf, steps=0)
    counts = [step.params["w0.log_duration"].numel() for step in steps]
    ends = [step.window.end for step in steps]
    assert counts == [1 if end <= 0.175 else 2 for end in ends]
    # a new event starts at its tone's onset, whatever the earlier events' fitted timing
    last = steps[-1].params
    onset = last["w0.onset"] + last["w0.log_duration"][0].exp() + last["w0.log_rest"][0].exp()
    assert onset.item() == pytest.approx(0.175)


def test_buildup_stimuli_are_prefixes_of_the_longest():
    longest = bu.buildup_stimulus(6, 8, lead=0.05).data[:, 0]
    for n in range(1, 6):
        shorter = bu.buildup_stimulus(n, 8, lead=0.05).data[:, 0]
        assert shorter.shape[0] == round((0.05 + 0.5 * n) * bu.FS)
        assert (shorter == longest[: shorter.shape[0]]).all()
    assert len(bu.buildup_set()) == 12


def test_a_fit_with_patience_stops_when_it_stops_improving():
    from sonore_infer.fit import fit

    waveform = torch.tensor(bu.buildup_stimulus(1, 8, lead=0.05).data[:, 0], dtype=F64)[: round(0.3 * bu.FS)]
    cochleagram = FFTCochleagram()
    scene, init = prefix_scene((WhistleSpec("w0", TONES[:1]),), 0.3, bu.FS, bu.RAMP)
    observed = cochleagram(waveform)
    kwargs = dict(learning_rates=scene.learning_rates(), steps=400, log_prior=scene.log_prior)
    full = fit(scene.render, init, observed, cochleagram, **kwargs)
    stopped = fit(scene.render, init, observed, cochleagram, patience=20, tolerance=1.0, **kwargs)
    n = len(stopped.history)
    assert n < len(full.history)
    assert stopped.history == full.history[:n]  # the same path, cut short
    assert max(stopped.history) - max(stopped.history[: n - 20]) < 1.0


def test_leaky_log_odds():
    from sonore_infer.online import leaky_log_odds

    ends = [0.05 * k for k in range(1, 201)]
    ones, zeros = [1.0] * 200, [0.0] * 200
    assert leaky_log_odds(ends, ones, zeros, math.inf)[-1] == pytest.approx(200)
    # a constant gain of 1 per 50 ms levels off at 1 / (1 - exp(-0.05 / tau)), about 20 tau
    assert leaky_log_odds(ends, ones, zeros, 0.5)[-1] == pytest.approx(1 / (1 - math.exp(-0.1)))


def test_online_windows_with_patience_stop_early_and_report_their_steps():
    _, _, _, fixed = short_run(math.inf, steps=200)
    waveform = torch.tensor(bu.buildup_stimulus(1, 8, lead=0.05).data[:, 0], dtype=F64)[: round(0.3 * bu.FS)]
    adaptive = run_online(
        waveform,
        (WhistleSpec("w0", TONES),),
        FFTCochleagram(),
        fs=bu.FS,
        ramp=bu.RAMP,
        window=0.05,
        lag=math.inf,
        steps=200,
        patience=10,
    )
    assert all(step.window.adam_steps == 200 for step in fixed)
    assert all(step.window.adam_steps <= 200 for step in adaptive)
    assert any(step.window.adam_steps < 200 for step in adaptive)


def test_windows_shorter_than_a_cochleagram_window_are_skipped():
    waveform = torch.tensor(bu.buildup_stimulus(1, 8, lead=0.05).data[:, 0], dtype=F64)[: round(0.1 * bu.FS)]
    steps = run_online(
        waveform,
        (WhistleSpec("w0", TONES[:1]),),
        FFTCochleagram(),
        fs=bu.FS,
        ramp=bu.RAMP,
        window=0.01,
        lag=math.inf,
        steps=2,
    )
    assert [round(step.window.end, 3) for step in steps] == [0.06, 0.07, 0.08, 0.09, 0.1]


def test_with_short_windows_an_event_enters_once_its_tone_is_heard():
    waveform = torch.tensor(bu.buildup_stimulus(1, 8, lead=0.05).data[:, 0], dtype=F64)[: round(0.1 * bu.FS)]
    steps = run_online(
        waveform,
        (WhistleSpec("w0", TONES[:1]),),
        FFTCochleagram(),
        fs=bu.FS,
        ramp=bu.RAMP,
        window=0.01,
        lag=math.inf,
        steps=2,
        heard=0.02,
    )
    # the tone starts at 0.05: nothing enters until 0.07, so the scene is silence before
    assert [round(step.window.end, 3) for step in steps] == [0.07, 0.08, 0.09, 0.1]


def short_run_with(**kwargs):
    waveform = torch.tensor(bu.buildup_stimulus(1, 8, lead=0.05).data[:, 0], dtype=F64)[: round(0.3 * bu.FS)]
    options = dict(fs=bu.FS, ramp=bu.RAMP, window=0.05, lag=math.inf, steps=5) | kwargs
    return run_online(waveform, (WhistleSpec("w0", TONES),), FFTCochleagram(), **options)


def test_evidence_taken_less_often_keeps_the_fits_and_still_telescopes():
    """Design online-speed S2: the fits are the same windows' fits; only the
    evidence is taken at fewer ends (and the last), and with no lag the total
    is still the batch Laplace estimate at the final mode."""
    every = short_run_with()
    sparse = short_run_with(evidence_ends=[0.15])
    assert [round(step.window.end, 3) for step in sparse] == [0.15, 0.3]
    final = every[-1]
    assert torch.equal(sparse[-1].params["w0.freq_deviation"], final.params["w0.freq_deviation"])
    assert sparse[-1].window.total == pytest.approx(final.window.total, abs=1e-6)
    later = [step for step in every if step.window.end > 0.15 + 1e-9]
    assert sparse[-1].window.adam_steps == sum(step.window.adam_steps for step in later)
