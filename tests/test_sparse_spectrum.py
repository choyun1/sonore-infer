"""The two options of design relative-pitch-prior: a lower frequency floor and sparse harmonic spectra."""

import math

import numpy as np
import pytest
import torch

from sonore_infer.erb import freq_to_erb_float
from sonore_infer.priors import (
    ABSENT_OFFSET_DB,
    ABSENT_OFFSET_SD_DB,
    F0_TRAJECTORY,
    PRESENT_OFFSET_SD_DB,
    StandardLogistic,
    presence_offset_distribution,
)
from sonore_infer.scene import Harmonic, Scene, SceneTiming, Whistle, _Recorder

TIMING = SceneTiming(fs=20_000, total_duration=0.3, ramp=0.01)
EVENT = dict(onset=0.05, duration=0.2)


def normal_log_pdf(x, mean, sd):
    return -0.5 * ((x - mean) / sd) ** 2 - math.log(sd * math.sqrt(2 * math.pi))


def test_off_by_default_the_sources_are_unchanged():
    plain, options_off = Harmonic("h0", 6), Harmonic("h0", 6, sparse_spectrum=False, frequency_floor_hz=None)
    params = plain.initial(TIMING, f0=200.0, level_db=60.0, **EVENT)
    assert set(params) == set(options_off.initial(TIMING, f0=200.0, level_db=60.0, **EVENT))
    assert "h0.presence_logit" not in plain.learning_rates()
    assert (
        plain.log_prior_terms(params, TIMING)[2].item()
        == options_off.log_prior_terms(params, TIMING)[2].item()
    )


@pytest.mark.parametrize(
    "source", [Harmonic("s", 4, frequency_floor_hz=30.0), Whistle("s", frequency_floor_hz=30.0)]
)
def test_the_floor_admits_lower_frequencies_and_widens_the_uniform(source):
    kwargs = dict(f0=50.0) if isinstance(source, Harmonic) else dict(freq=50.0)
    params = source.initial(TIMING, level_db=60.0, **EVENT, **kwargs)
    first = source.log_prior_terms(params, TIMING)[0].item()
    default = type(source)("s", **({"n_harmonics": 4} if isinstance(source, Harmonic) else {}))
    assert default.log_prior_terms(params, TIMING)[0].item() == -math.inf
    params_100 = source.initial(TIMING, level_db=60.0, **EVENT, **{k: 100.0 for k in kwargs})
    widened = source.log_prior_terms(params_100, TIMING)[0].item()
    original = default.log_prior_terms(params_100, TIMING)[0].item()
    high = F0_TRAJECTORY.mean_range[1]
    expected = math.log((high - F0_TRAJECTORY.mean_range[0]) / (high - freq_to_erb_float(30.0)))
    assert widened - original == pytest.approx(expected)
    assert math.isfinite(first)


def test_standard_logistic_is_beta_one_one_in_logit_coordinates():
    x = torch.linspace(-6, 6, 13, dtype=torch.float64)
    p = torch.sigmoid(x)
    np.testing.assert_allclose(
        StandardLogistic().log_prob(x).numpy(), (p * (1 - p)).log().numpy(), atol=1e-12
    )
    # integrates to one
    grid = torch.linspace(-40, 40, 80_001, dtype=torch.float64)
    assert torch.trapezoid(StandardLogistic().log_prob(grid).exp(), grid).item() == pytest.approx(
        1.0, abs=1e-9
    )


def test_presence_mixture_sums_present_and_absent_per_harmonic():
    logit = torch.tensor(0.7, dtype=torch.float64)
    offsets = torch.tensor([0.3, -38.0, -5.0], dtype=torch.float64)
    p = 1 / (1 + math.exp(-0.7))
    expected = sum(
        math.log(
            p * math.exp(normal_log_pdf(d, 0.0, PRESENT_OFFSET_SD_DB))
            + (1 - p) * math.exp(normal_log_pdf(d, ABSENT_OFFSET_DB, ABSENT_OFFSET_SD_DB))
        )
        for d in offsets.tolist()
    )
    assert presence_offset_distribution(3, logit).log_prob(offsets).item() == pytest.approx(
        expected, rel=1e-12
    )


def test_sparse_source_renders_offsets_and_scores_them():
    sparse = Harmonic("h0", 5, sparse_spectrum=True)
    dense = Harmonic("h0", 5)
    offsets = torch.tensor([0.0, -40.0, 0.0, -40.0, 0.0], dtype=torch.float64)
    params = sparse.initial(
        TIMING, f0=200.0, level_db=60.0, spectrum_offset_db=offsets, presence_logit=0.5, **EVENT
    )
    dense_params = {
        k: v for k, v in params.items() if k in dense.initial(TIMING, f0=200.0, level_db=60.0, **EVENT)
    }
    dense_params["h0.spectrum_db"] = params["h0.spectrum_db"] + offsets
    np.testing.assert_allclose(
        sparse.render(params, TIMING).numpy(), dense.render(dense_params, TIMING).numpy()
    )
    terms = sparse.log_prior_terms(params, TIMING)
    dense_terms = dense.log_prior_terms({**params, "h0.spectrum_db": params["h0.spectrum_db"]}, TIMING)
    logit = params["h0.presence_logit"]
    extra = StandardLogistic().log_prob(logit) + presence_offset_distribution(5, logit).log_prob(offsets)
    assert terms[2].item() == pytest.approx((dense_terms[2] + extra).item())
    assert set(sparse.learning_rates()) == set(params)


def test_a_sparse_scene_samples_from_its_prior():
    sparse = Harmonic("h0", 8, sparse_spectrum=True, frequency_floor_hz=30.0)
    scene = Scene((sparse,), TIMING)
    recorder = _Recorder()
    sparse.program(recorder, TIMING)
    assert recorder.params["h0.spectrum_offset_db"].shape == (8,)
    assert math.isfinite(scene.log_prior(recorder.params).item())


def test_the_pyro_model_scores_a_sparse_scene_like_the_scene():
    pyro = pytest.importorskip("pyro")
    sparse = Harmonic("h0", 4, sparse_spectrum=True, frequency_floor_hz=30.0)
    scene = Scene((sparse,), TIMING)
    params = sparse.initial(TIMING, f0=60.0, level_db=60.0, **EVENT)
    trace = pyro.poutine.trace(pyro.poutine.condition(scene.model, data=params)).get_trace()
    assert trace.log_prob_sum().item() == pytest.approx(scene.log_prior(params).item())
