"""hearing.py (design relative-pitch-dominance P19, P20): Terhardt's formulas and the cochleagram options."""

import math

import numpy as np
import pytest
import torch

from sonore_infer.cochleagram import FFTCochleagram, gaussian_log_likelihood
from sonore_infer.hearing import dominance_sigma, dominance_weight, threshold_of_hearing_db


def test_threshold_matches_eq_8():
    expected = 3.64 * 1.0 - 6.5 * math.exp(-0.6 * (1.0 - 3.3) ** 2) + 1e-3
    assert threshold_of_hearing_db(1000.0) == pytest.approx(expected)
    # steep below 100 Hz
    assert threshold_of_hearing_db(20.0) > threshold_of_hearing_db(51.0) > threshold_of_hearing_db(100.0)


def test_dominance_peaks_at_700_hz_and_is_symmetric_in_log_frequency():
    assert dominance_weight(700.0) == pytest.approx(1.0)
    assert dominance_weight(350.0) == pytest.approx(dominance_weight(1400.0))
    assert dominance_weight(55.0) < dominance_weight(200.0) < dominance_weight(700.0)


def test_dominance_sigma_shape_and_value():
    sigma = dominance_sigma(np.array([700.0, 100.0]))
    assert sigma.shape == (2, 1)
    assert sigma[0, 0].item() == pytest.approx(10.0)
    assert sigma[1, 0].item() == pytest.approx(10.0 / math.sqrt(dominance_weight(100.0)))


def test_constant_tensor_sigma_matches_scalar():
    generator = torch.Generator().manual_seed(0)
    observed = torch.randn(4, 6, generator=generator, dtype=torch.float64) * 10
    predicted = torch.randn(4, 6, generator=generator, dtype=torch.float64) * 10
    scalar = gaussian_log_likelihood(observed, predicted, 7.0)
    tensor = gaussian_log_likelihood(observed, predicted, torch.full((4, 1), 7.0, dtype=torch.float64))
    assert tensor.item() == pytest.approx(scalar.item())


def test_threshold_floor_only_raises_low_channels():
    plain, floored = FFTCochleagram(), FFTCochleagram(threshold_floor=True)
    floors = floored.channel_floor_db()
    assert np.all(floors >= plain.floor_db)
    assert floors[0] > plain.floor_db  # 20 Hz: far above the 20 dB floor
    assert floors[-1] == plain.floor_db
    t = torch.arange(4000, dtype=torch.float64) / plain.fs
    tone = math.sqrt(2) * 1e-6 * 10 ** (60 / 20) * torch.cos(2 * math.pi * 1000 * t)
    a, b = plain(tone), floored(tone)
    assert torch.equal(torch.maximum(a, torch.as_tensor(floors)[:, None]), b)
