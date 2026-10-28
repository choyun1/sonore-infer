"""tools/relative_pitch/listener_noise.py (design relative-pitch-listeners): the grid fit recovers known values."""

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
