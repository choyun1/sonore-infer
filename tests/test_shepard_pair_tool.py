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
