"""tools/relative_pitch/listener_noise.py: the grid fit recovers known values."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

FOLDER = Path(__file__).parent.parent / "tools" / "relative_pitch"


@pytest.fixture(scope="module")
def noise():
    sys.path.insert(0, str(FOLDER))
    spec = importlib.util.spec_from_file_location("listener_noise", FOLDER / "listener_noise.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    sys.path.remove(str(FOLDER))


def test_fit_recovers_temperature_and_lapse(noise):
    rng = np.random.default_rng(0)
    # made-up log odds falling with steps, 10 pairs each, as the model's do
    odds = {step: list(30 - 7 * step + rng.normal(0, 5, 10)) for step in noise.STEPS}
    beta, lapse = 0.2, 0.1
    observed = list(noise.model_fractions(odds, noise.STEPS, np.array(beta), np.array(lapse)))
    fitted_beta, fitted_lapse, _ = noise.fit_model(odds, observed, trials=100_000)
    assert fitted_beta == pytest.approx(beta, rel=0.03)
    assert fitted_lapse == pytest.approx(lapse, abs=0.01)


def test_baseline_recovers_its_curve(noise):
    observed = list(noise.baseline_fractions(noise.STEPS, 6.0, 1.2, 0.05))
    intercept, slope, lapse, _ = noise.fit_baseline(observed, trials=100_000)
    assert (intercept, slope, lapse) == pytest.approx((6.0, 1.2, 0.05), rel=0.05, abs=0.1)


def test_circle_recovers_interval_noise(noise):
    observed = list(noise.circle_fractions(noise.STEPS, 1.5))
    sd, _ = noise.fit_circle(observed, trials=100_000)
    assert sd == pytest.approx(1.5, abs=0.05)


def test_circle_without_noise_is_the_shorter_way_round(noise):
    assert list(noise.circle_fractions(noise.STEPS, 0.0)) == [1, 1, 1, 1, 0, 0, 0, 0, 0]


def test_interval_noise_on_odds_without_noise_is_the_plain_fit(noise):
    rng = np.random.default_rng(1)
    curves = np.zeros((10, 11))
    curves[:, 1:10] = 30 - 7 * np.arange(1, 10) + rng.normal(0, 5, (10, 9))
    odds = {step: list(curves[:, step]) for step in noise.STEPS}
    plain = noise.model_fractions(odds, noise.STEPS, np.array(0.2), np.array(0.1))
    blurred = noise.blurred_fractions(curves, noise.STEPS, 0.2, 0.1, 0.0)
    assert np.allclose(plain, blurred)


def test_interval_noise_makes_the_smallest_steps_harder(noise):
    curves = np.zeros((10, 11))
    curves[:, 1:10] = 30 - 7 * np.arange(1, 10)
    fractions = noise.blurred_fractions(curves, noise.STEPS, 1.0, 0.0, 1.5)
    assert fractions[0] < fractions[1]
    assert fractions[8] > fractions[7]
