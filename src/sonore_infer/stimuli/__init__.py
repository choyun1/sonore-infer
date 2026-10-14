"""Stimuli from Appendix C of Cusimano, Hewitt & McDermott (2024), made with sonore.

Each module rebuilds one experiment's stimuli from the paper's text alone
(design D3, milestone (a); D7). Where the text leaves a detail open, the
module docstring says what is assumed, and the choice is an argument with
that default. ``two_notes`` makes milestone (b)'s stimuli, which are not
from the paper (docs/design/milestone-b.md).

This subpackage needs the ``stimuli`` extra (``pip install
sonore-infer[stimuli]``); the rest of the package never imports it.
"""

from sonore_infer.stimuli.asynchronous_onsets import asynchronous_onsets_set, asynchronous_onsets_stimulus
from sonore_infer.stimuli.bistability import bistability_set, bistability_stimulus
from sonore_infer.stimuli.levels import FS, REFERENCE_RMS, rms_from_db
from sonore_infer.stimuli.mistuned_harmonic import mistuned_harmonic_set, mistuned_harmonic_stimulus
from sonore_infer.stimuli.two_notes import single_note_control, two_notes_set, two_notes_stimulus

__all__ = [
    "FS",
    "REFERENCE_RMS",
    "asynchronous_onsets_set",
    "asynchronous_onsets_stimulus",
    "bistability_set",
    "bistability_stimulus",
    "mistuned_harmonic_set",
    "mistuned_harmonic_stimulus",
    "rms_from_db",
    "single_note_control",
    "two_notes_set",
    "two_notes_stimulus",
]
