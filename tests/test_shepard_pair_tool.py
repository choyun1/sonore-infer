"""tools/relative_pitch/pair.py (experiment P3): each hypothesis's start moves the way its label says."""

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).parent.parent / "tools" / "relative_pitch" / "pair.py"


@pytest.fixture(scope="module")
def pair():
    spec = importlib.util.spec_from_file_location("pair", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("sparse", [False, True])
def test_starts_move_as_labelled(pair, sparse):
    for label, _scene, start, names in pair.hypotheses(1, 3, None, sparse, 30.0, 30.0):
        moved = pair.shift(start, names)
        assert (moved > 0) == (" up" in label), (label, moved)


def test_octave_moves_the_reading_and_respects_the_floor(pair):
    base = {label.split()[1]: label for label, *_ in pair.hypotheses(1, 5, None, True, 30.0, 30.0, 0, True)}
    lower = [label for label, *_ in pair.hypotheses(1, 5, None, True, 30.0, 30.0, -1, True)]
    upper = [label for label, *_ in pair.hypotheses(1, 5, None, True, 30.0, 30.0, 1, True)]
    reading = float(base["up"].split()[2].split("->")[0])
    starts = [float(label.split()[2].split("->")[0]) for label in upper]
    assert starts == pytest.approx([reading * 2] * len(upper), rel=0.01)
    # an octave down, the down step would start below the 30 Hz floor and is left out
    assert [label.split()[1] for label in lower] == ["up"]
