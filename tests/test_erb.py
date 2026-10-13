"""The ERB scale: equal to sonore's, the same for NumPy and torch, and the float
version unchanged from what the scene and the tools computed before."""

import math

import numpy as np
import pytest
import sonore as so
import torch
from sonore.core.utils import erb_bandwidth as sonore_erb_bandwidth  # not re-exported as so.*

from sonore_infer.erb import erb_bandwidth, erb_to_freq, freq_to_erb, freq_to_erb_float

FREQS = np.geomspace(1.0, 20_000.0, 1001)
ERBS = np.linspace(0.0, 42.0, 1001)


def test_equal_to_sonore():
    np.testing.assert_array_equal(freq_to_erb(FREQS), so.freq_to_erb(FREQS))
    np.testing.assert_array_equal(erb_to_freq(ERBS), so.erb_to_freq(ERBS))
    np.testing.assert_array_equal(erb_bandwidth(FREQS), sonore_erb_bandwidth(FREQS))


def test_torch_matches_numpy():
    for function, values in [(freq_to_erb, FREQS), (erb_to_freq, ERBS), (erb_bandwidth, FREQS)]:
        tensor_result = function(torch.tensor(values, dtype=torch.float64))
        assert isinstance(tensor_result, torch.Tensor)
        np.testing.assert_allclose(tensor_result.numpy(), function(values), rtol=1e-15, atol=0)


def test_inverse_and_derivative():
    np.testing.assert_allclose(erb_to_freq(freq_to_erb(FREQS)), FREQS, rtol=1e-12)
    # d(ERB number)/df = 1 / ERB bandwidth, to the rounding of 9.265 (module docstring)
    freqs = torch.tensor(FREQS, dtype=torch.float64, requires_grad=True)
    (gradient,) = torch.autograd.grad(freq_to_erb(freqs).sum(), freqs)
    np.testing.assert_allclose(gradient.numpy(), 1 / erb_bandwidth(FREQS), rtol=6e-5)


def test_float_version():
    # what scene.py and tools/compare_mistuned_harmonic.py computed before this module
    for freq in FREQS:
        freq = float(freq)
        assert freq_to_erb_float(freq) == 9.265 * math.log1p(freq / (24.7 * 9.265))
        assert freq_to_erb_float(freq) == pytest.approx(float(freq_to_erb(freq)), rel=1e-15)
