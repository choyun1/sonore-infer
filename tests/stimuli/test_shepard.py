"""Shepard tones, checked against Shepard's (1964) Eqs. 1-3 written out directly."""

import numpy as np
import pytest

from sonore_infer.stimuli import shepard as sh

FS = 20_000


def shepard_eqs(t, t_max=10, c_max=10, f_min=4.863, l_min=22.0, l_max=56.0):
    """Eqs. 1-3 of Shepard (1964) as printed, every component."""
    c = np.arange(1, c_max + 1)
    f = f_min * 2 ** (((c - 1) * t_max + t - 1) / t_max)
    theta = 2 * np.pi * ((c - 1) * t_max + t - 1) / (t_max * c_max)
    return f, l_min + (l_max - l_min) * (1 - np.cos(theta)) / 2


@pytest.mark.parametrize("step", range(1, 11))
def test_components_follow_eqs_1_to_3(step):
    freqs, levels = sh.components(step)
    expected_f, expected_l = shepard_eqs(step)
    np.testing.assert_allclose(freqs, expected_f, rtol=1e-12)
    np.testing.assert_allclose(levels, expected_l, atol=1e-12)


def test_envelope_peak():
    freqs, levels = sh.components(1)
    assert freqs[np.argmax(levels)] == pytest.approx(155.616)
    assert levels.max() == pytest.approx(56.0)


def test_minor_third_control_has_four_per_octave():
    freqs, levels = sh.components(1, components_per_octave=4)
    assert len(freqs) == 40
    np.testing.assert_allclose(freqs[1:] / freqs[:-1], 2**0.25)
    # the same envelope: at the octave-spaced components it matches the octave tone
    octave_freqs, octave_levels = sh.components(1)
    np.testing.assert_allclose(freqs[::4], octave_freqs)
    np.testing.assert_allclose(levels[::4], octave_levels)


def test_components_stop_at_f_max():
    freqs, _ = sh.components(10, steps=10, components_per_octave=4)
    assert freqs.max() <= sh.F_MAX


def test_tone_waveform():
    freqs, levels = sh.components(3)
    n = round(sh.DURATION * FS)
    t = np.arange(n) / FS
    expected = sum(
        np.sqrt(2) * 1e-6 * 10 ** (level / 20) * np.sin(2 * np.pi * f * t)
        for f, level in zip(freqs, levels, strict=True)
    )
    n_ramp = round(sh.RAMP * FS)
    phase = (np.arange(n_ramp) + 0.5) / n_ramp
    gain = np.ones(n)
    gain[:n_ramp] = (1 - np.cos(np.pi * phase)) / 2
    gain[-n_ramp:] = gain[:n_ramp][::-1]
    np.testing.assert_allclose(sh.tone(3).data[:, 0], expected * gain, atol=1e-12)


def test_pair_timing():
    sound = sh.pair(2, 7)
    pad, dur, gap = (round(x * FS) for x in (sh.PADDING, sh.DURATION, sh.GAP))
    data = sound.data[:, 0]
    assert len(data) == 2 * pad + 2 * dur + gap
    np.testing.assert_allclose(data[pad : pad + dur], sh.tone(2).data[:, 0])
    np.testing.assert_allclose(data[pad + dur + gap : pad + 2 * dur + gap], sh.tone(7).data[:, 0])
    assert not data[:pad].any() and not data[pad + dur : pad + dur + gap].any()


def test_steps_clockwise():
    assert sh.steps_clockwise(2, 7) == 5
    assert sh.steps_clockwise(9, 1) == 2
    assert sh.steps_clockwise(4, 4) == 0
