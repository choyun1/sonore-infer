"""The detector finds the stimuli's events, and the search builds, extends and cleans up hypotheses."""

import math

import pytest
import torch

from sonore_infer.cochleagram import FFTCochleagram
from sonore_infer.detect import Candidate, OnsetGroup, channel_gains, detect
from sonore_infer.search import SourcePlan, build, cleanup_moves, extensions, plan_key
from sonore_infer.stimuli import bistability as ab
from sonore_infer.stimuli import two_notes as tn

COCHLEAGRAM = FFTCochleagram()


def groups_of(sound):
    waveform = torch.tensor(sound.data[:, 0], dtype=torch.float64)
    return detect(COCHLEAGRAM(waveform), COCHLEAGRAM, waveform.shape[-1])


def test_channel_gains_calibrate_a_tone():
    gains = channel_gains(COCHLEAGRAM)
    # BASS's uncalibrated channels read a tone low, by more at low frequencies
    assert gains.shape == (COCHLEAGRAM.n_channels,)
    assert gains[5] > gains[50] > 0


def test_detector_finds_every_aba_tone():
    groups = groups_of(ab.bistability_stimulus(0.1, 6, lead=0.05))
    true = ab.tone_onsets(0.1, lead=0.05)
    assert len(groups) == len(true)
    b_freq = ab.A_FREQ * 2 ** (6 / 12)
    for group, (onset, kind) in zip(groups, true, strict=True):
        assert abs(group.onset - onset) <= 0.01
        (whistles,) = group.explanations  # a pure tone has no harmonic explanation
        (tone,) = whistles
        expected = ab.A_FREQ if kind == "A" else b_freq
        assert tone.kind == "whistle"
        assert abs(math.log2(tone.freq / expected)) < 0.02
        assert abs(tone.level_db - ab.TONE_LEVEL_DB) < 3
        assert abs(tone.duration - ab.TONE_DURATION) <= 0.02


def test_detector_offers_a_harmonic_source_and_its_octaves_for_a_note():
    (group,) = groups_of(tn.single_note_control(200.0))
    f0s = [e[0].freq for e in group.explanations if e[0].kind == "harmonic"]
    assert len(f0s) == 3
    assert abs(math.log2(f0s[0] / 200.0)) < 0.03
    assert f0s[1] == pytest.approx(2 * f0s[0]) and f0s[2] == pytest.approx(f0s[0] / 2)


def test_detector_finds_the_later_note_of_an_asynchronous_pair():
    groups = groups_of(tn.two_notes_stimulus("just_fifth", 0.04))
    assert len(groups) == 2
    assert abs(groups[1].onset - (tn.PADDING + 0.04)) <= 0.01
    f0s = [e[0].freq for e in groups[1].explanations if e[0].kind == "harmonic"]
    assert abs(math.log2(f0s[0] / tn.UPPER_F0S["just_fifth"])) < 0.03


def tone(onset, freq, kind="whistle", index=0):
    spectrum = (0.0, -6.0) if kind == "harmonic" else None
    return Candidate(kind, onset, 0.05, freq, 70.0, 10.0, index, spectrum)


def test_extensions_join_or_start_sources_or_leave_the_group_out():
    first = (SourcePlan("whistle", (tone(0.05, 1000.0),)),)

    def added(group, max_sources=4):
        plans = extensions(first, group, max_sources=max_sources, max_new_sources=2, max_new_events=3)
        assert plans[0] == first  # the group left out
        return plans[1:]

    group = OnsetGroup(0.15, ((tone(0.15, 1400.0, index=1),),))
    assert sorted(len(plan) for plan in added(group)) == [1, 2]
    # no new source when the hypothesis is full
    assert len(added(group, max_sources=1)) == 1
    # a harmonic candidate cannot join a whistle
    assert [len(p) for p in added(OnsetGroup(0.15, ((tone(0.15, 200.0, "harmonic", 1),),)))] == [2]
    # two simultaneous events never join the same source
    pair = OnsetGroup(0.15, ((tone(0.15, 1400.0, index=1), tone(0.15, 1800.0, index=1)),))
    plans = added(pair)
    assert sorted(len(p) for p in plans) == [2, 2, 3]
    assert len({plan_key(p) for p in plans}) == 3
    # the first group is never left out
    assert extensions((), group, max_sources=4, max_new_sources=None, max_new_events=3) == [
        (SourcePlan("whistle", group.explanations[0]),)
    ]


def test_cleanup_moves_merge_change_and_remove():
    plan = (
        SourcePlan("whistle", (tone(0.05, 1000.0), tone(0.25, 1000.0))),
        SourcePlan("whistle", (tone(0.15, 1400.0),)),
    )
    moves = cleanup_moves(plan)
    sizes = [tuple(len(s.events) for s in move) for move in moves]
    assert (3,) in sizes  # the two sources merged
    assert ((2,) in sizes) and ((1,) in sizes)  # a source removed
    assert any(s.kind == "harmonic" for move in moves for s in move)  # a kind changed
    assert (1, 1) in sizes  # an event merged into the next, or removed


def test_build_cuts_overlaps_and_leaves_out_later_events():
    long = Candidate("whistle", 0.05, 0.3, 1000.0, 70.0, 10.0, 0)
    plan = (SourcePlan("whistle", (long, tone(0.15, 1000.0, index=1), tone(0.45, 1000.0, index=2))),)
    scene, params = build(plan, 0.4, 20_000, 0.01)
    (source,) = scene.sources
    assert source.n_events == 2
    onsets, durations = source.event_times(params, scene.timing)
    assert float(onsets[0] + durations[0]) <= 0.15
    assert torch.isfinite(scene.log_prior(params))
