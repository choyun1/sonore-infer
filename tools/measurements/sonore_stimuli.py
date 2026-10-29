"""Every stimulus this repository makes with sonore, written out, so two sonore versions can be compared.

``dump`` writes, with the installed sonore, every stimulus set's sounds (two
notes, bistability, buildup, mistuned harmonic, asynchronous onsets, all
Shepard single tones and pairs) and, for the source-and-room tools, 20
``synth_ir`` rooms, a Gaussian-spectrogram noise heard in each, and
``band_rt60s``, to one ``.npz``. ``compare`` prints, for each group, how many
arrays are bit-identical between two such files and the largest difference
relative to each array's peak.

    python tools/measurements/sonore_stimuli.py dump old.npz      # with one sonore installed
    python tools/measurements/sonore_stimuli.py dump new.npz      # with the other
    python tools/measurements/sonore_stimuli.py compare old.npz new.npz
"""

import argparse

import numpy as np
from scipy.signal import fftconvolve

ROOM_RT60S = (0.2, 0.4, 0.8, 1.6)
ROOM_SEEDS = range(5)


def dump(path):
    import sonore as so

    from sonore_infer.stimuli import (
        asynchronous_onsets,
        bistability,
        buildup,
        mistuned_harmonic,
        shepard,
        two_notes,
    )
    from sonore_infer.stimuli.levels import FS

    out = {}
    for group, make in [
        ("two_notes", two_notes.two_notes_set),
        ("bistability", bistability.bistability_set),
        ("buildup", buildup.buildup_set),
        ("mistuned_harmonic", mistuned_harmonic.mistuned_harmonic_set),
        ("asynchronous_onsets", asynchronous_onsets.asynchronous_onsets_set),
    ]:
        for i, (params, sound) in enumerate(make()):
            out[f"{group}|{i}|{sorted(params.items())}"] = sound.data
    steps = range(1, shepard.STEPS_PER_OCTAVE + 1)
    for first in steps:
        for per_octave in (1, 2):
            out[f"shepard|single|{first}|{per_octave}"] = shepard.single_tone(
                first, components_per_octave=per_octave
            ).data
        for second in steps:
            out[f"shepard|pair|{first}|{second}"] = shepard.pair(first, second).data
    cfs = so.cosine_filterbank(30, 50.0, 8000.0).cfs[1:-1]
    for rt60 in ROOM_RT60S:
        for seed in ROOM_SEEDS:
            ir = so.synth_ir(rt60, FS, drr_db=5.0, rng=seed).data
            env = so.gaussian_spectrogram(
                0.4, FS, band_correlation_erb=4.0, time_correlation=0.05, sd_db=6.0, rng=seed
            )
            noise = env.to_sound(so.gaussian_noise(0.4, FS, rng=seed + 100)).ramp(0.01)
            out[f"synth_ir|{rt60}|{seed}"] = ir
            out[f"room_noise|{rt60}|{seed}"] = fftconvolve(noise.data.ravel(), ir.ravel())
        out[f"band_rt60s|{rt60}"] = so.band_rt60s(rt60, cfs)
    np.savez(path, **out)
    print(f"sonore {so.__version__}: {len(out)} arrays written to {path}")


def compare(path_a, path_b):
    a, b = np.load(path_a), np.load(path_b)
    if set(a.files) != set(b.files):
        raise SystemExit(f"the files hold different arrays: {sorted(set(a.files) ^ set(b.files))[:5]} ...")
    groups: dict[str, list[float]] = {}
    for key in a.files:
        x, y = a[key], b[key]
        if np.array_equal(x, y):
            diff = 0.0
        elif x.shape != y.shape:
            diff = np.inf
        else:
            diff = float(np.abs(x - y).max() / np.abs(x).max())
        groups.setdefault(key.split("|")[0], []).append(diff)
    for group, diffs in groups.items():
        same = sum(d == 0.0 for d in diffs)
        print(f"{group:22s} {same:4d}/{len(diffs):<4d} bit-identical; worst {max(diffs):.1e} of the peak")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("dump").add_argument("path")
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    args = parser.parse_args()
    if args.command == "dump":
        dump(args.path)
    else:
        compare(args.a, args.b)


if __name__ == "__main__":
    main()
