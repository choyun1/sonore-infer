"""The two reset studies' numbers, checked against the stimuli (docs/design/order-effects.md, E2)."""

import numpy as np
import pytest

from sonore_infer.stimuli import reset

FS = 20_000


def level_db(samples):
    return 20 * np.log10(np.sqrt(np.mean(samples**2)) / 1e-6)


def test_beauvois_meddis_timing():
    stimulus = reset.beauvois_meddis(2.0)
    inducer = [t for t in stimulus.tones if t.part == "first"]
    test = [t for t in stimulus.tones if t.part == "second"]
    assert len(inducer) == 111 and {t.kind for t in inducer} == {"A"}
    assert inducer[-1].onset + 0.040 <= 10.0 < inducer[-1].onset + 0.090 + 0.040  # fills 10 s
    assert [t.kind for t in test] == ["A", "B"] * 8
    assert np.diff([t.onset for t in test]) == pytest.approx([0.09] * 15)
    assert stimulus.gap == pytest.approx((inducer[-1].onset + 0.040, inducer[-1].onset + 2.040))
    assert test[0].onset == pytest.approx(stimulus.gap[1])
    assert stimulus.duration - test[0].onset == pytest.approx(1.44)
    assert stimulus.sound.n_samples == round(stimulus.duration * FS)
    assert stimulus.freqs == {"A": 1000.0, "B": 1420.0}


def test_beauvois_meddis_silence_is_silent_and_levels_are_65_db():
    stimulus = reset.beauvois_meddis(1.0)
    data = stimulus.sound.data[:, 0]
    begin, end = (round(t * FS) for t in stimulus.gap)
    assert not data[begin:end].any()
    first = data[100:700]  # inside the first tone, past its 5 ms ramp: 30 periods of 1000 Hz
    assert level_db(first) == pytest.approx(65, abs=1e-9)


def test_beauvois_meddis_test_alone():
    alone = reset.beauvois_meddis(None)
    with_inducer = reset.beauvois_meddis(0.5)
    assert all(t.part == "second" for t in alone.tones) and len(alone.tones) == 16
    tail = with_inducer.sound.data[-alone.sound.n_samples :, 0]
    np.testing.assert_allclose(tail, alone.sound.data[:, 0], atol=1e-12)


@pytest.mark.parametrize("semitones, b_printed", [(4, 504), (6, 566)])
def test_cusack_frequencies_round_to_printed(semitones, b_printed):
    stimulus = reset.cusack(1.0, semitones)
    assert stimulus.freqs["A"] == 400.0
    assert round(stimulus.freqs["B"]) == b_printed


def test_cusack_timing():
    stimulus = reset.cusack(5.0, 6, test_duration=4.0)
    first = [t for t in stimulus.tones if t.part == "first"]
    second = [t for t in stimulus.tones if t.part == "second"]
    assert len(first) == 60 and len(second) == 24  # 20 and 8 ABA- triplets
    assert [t.kind for t in first[:3]] == ["A", "B", "A"]
    assert first[3].onset - first[0].onset == pytest.approx(0.5)
    assert stimulus.gap == pytest.approx((10.0, 15.0))
    assert second[0].onset == pytest.approx(15.0)
    assert stimulus.duration == pytest.approx(19.0)
    data = stimulus.sound.data[:, 0]
    assert not data[round(first[-1].onset * FS) + 2500 : round(15.0 * FS)].any()
    steady = data[400:2100]  # inside the first 125 ms tone, past its 20 ms ramp: 34 periods of 400 Hz
    assert level_db(steady) == pytest.approx(55, abs=1e-9)


def test_lead_shifts_both_studies():
    for make in (lambda **k: reset.beauvois_meddis(1.0, **k), lambda **k: reset.cusack(1.0, 4, **k)):
        plain, led = make(), make(lead=0.05)
        assert led.tones[0].onset == pytest.approx(0.05)
        assert led.gap == pytest.approx(tuple(t + 0.05 for t in plain.gap))
