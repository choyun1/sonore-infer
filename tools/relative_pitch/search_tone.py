"""The scene search (design C9, C10) on one Shepard tone, as a check on the hypotheses of
``one_tone.py`` (docs/design/relative-pitch.md, decision P4).

Runs ``tools/grammar/search_scene.py`` with Shepard tones added to its stimuli, named
``shepard:<step>`` (octave spacing) or ``shepard:<step>:minor_third``; every other
argument is that tool's:

    python tools/relative_pitch/search_tone.py shepard:1 --variational-steps 2000
    python tools/relative_pitch/search_tone.py shepard:1:minor_third
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "grammar"))
import search_scene  # noqa: E402

from sonore_infer.stimuli import shepard as sh  # noqa: E402

_named = search_scene.stimulus


def stimulus(name):
    """(waveform, ramp) of a Shepard tone, or of any stimulus ``search_scene`` names."""
    kind, *args = name.split(":")
    if kind != "shepard":
        return _named(name)
    per_octave = 4 if args[1:] == ["minor_third"] else 1
    sound = sh.single_tone(int(args[0]), components_per_octave=per_octave)
    return torch.tensor(sound.data[:, 0], dtype=search_scene.DTYPE), sh.RAMP


search_scene.stimulus = stimulus

if __name__ == "__main__":
    search_scene.main()
