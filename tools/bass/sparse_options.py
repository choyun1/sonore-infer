"""The relative-pitch-prior options (sparse spectra, a lower frequency floor) for the milestone (a)
and (b) tools, and the scoring of each start's mode separately (design relative-pitch-prior P14).

With a sparse spectrum a harmonic source has two kinds of mode: harmonics the start sets far
down either stay present but quiet, or are marked absent. A fit from one start rarely reaches
the other kind, and the mode with the higher log joint can have the lower evidence. So each
start is fitted and scored on its own, and a hypothesis's evidence is the log-sum-exp over its
distinct modes (log joints within 1 count as one mode), as in
``tools/relative_pitch/combine_starts.py``.
"""

import math
from types import SimpleNamespace

import numpy as np
import torch

from sonore_infer.priors import ABSENT_OFFSET_DB
from sonore_infer.scene import Harmonic, evaluate

# a harmonic whose start is more than this far below the source's level is marked absent
ABSENT_BELOW_DB = 20.0
SAME_MODE = 1.0


def add_arguments(parser):
    parser.add_argument(
        "--sparse-spectrum",
        action="store_true",
        help="harmonics may be absent; each start is scored separately (design relative-pitch-prior P10)",
    )
    parser.add_argument(
        "--floor", type=float, help="lowest f0 and whistle frequency [Hz] (design relative-pitch-prior P12)"
    )


def harmonic_options(args):
    return dict(sparse_spectrum=args.sparse_spectrum, frequency_floor_hz=args.floor)


def whistle_options(args):
    return dict(frequency_floor_hz=args.floor)


def describe(args):
    """The options for a run's header line."""
    return (", sparse spectra" if args.sparse_spectrum else "") + (
        f", frequency floor {args.floor:g} Hz" if args.floor else ""
    )


def with_sparse_starts(scene, starts):
    """``starts`` (every harmonic present), each followed by its sparse twin when it has one: the
    harmonics more than ABSENT_BELOW_DB down marked absent, and the spectrum through the others,
    interpolated over log harmonic number."""
    out = []
    for start in starts:
        out.append(start)
        twin = dict(start)
        for source in scene.sources:
            if not (isinstance(source, Harmonic) and source.sparse_spectrum):
                continue
            spectrum = start[f"{source.name}.spectrum_db"]
            absent = spectrum < -ABSENT_BELOW_DB
            if not absent.any() or absent.all():
                continue
            x = np.log(np.arange(1, len(spectrum) + 1))
            present = (~absent).numpy()
            twin[f"{source.name}.spectrum_db"] = torch.tensor(
                np.interp(x, x[present], spectrum.numpy()[present]), dtype=spectrum.dtype
            )
            twin[f"{source.name}.spectrum_offset_db"] = torch.where(
                absent, torch.tensor(ABSENT_OFFSET_DB, dtype=spectrum.dtype), torch.zeros_like(spectrum)
            )
            fraction = float(present.mean())
            twin[f"{source.name}.presence_logit"] = torch.tensor(
                math.log(fraction / (1 - fraction)), dtype=spectrum.dtype
            )
        if any(twin[name] is not start[name] for name in start):
            out.append(twin)
    return out


def logsumexp(values):
    top = max(values)
    if top == -math.inf:
        return top
    return top + math.log(sum(math.exp(value - top) for value in values))


def evaluate_modes(scene, starts, observed, cochleagram, **kwargs):
    """Fit and score each start on its own; a result like ``evaluate``'s whose estimates are the
    log-sum-exp over the distinct modes, with ``modes`` holding each mode's result."""
    results = [evaluate(scene, [start], observed, cochleagram, **kwargs) for start in starts]
    modes = []
    for result in results:
        joint = result.evidence.log_joint_at_mode
        if not any(abs(joint - other.evidence.log_joint_at_mode) < SAME_MODE for other in modes):
            modes.append(result)
    structure = modes[0].structure_log_prior
    evidence = SimpleNamespace(
        laplace=logsumexp([m.evidence.laplace for m in modes]),
        importance=logsumexp([m.evidence.importance for m in modes]),
        effective_sample_size=min(m.evidence.effective_sample_size for m in modes),
        floored_eigenvalues=max(m.evidence.floored_eigenvalues for m in modes),
        log_joint_at_mode=max(m.evidence.log_joint_at_mode for m in modes),
    )
    variational = None
    if modes[0].variational is not None:
        # where importance sampling failed for a mode, its ELBO stands in (a lower bound)
        variational = SimpleNamespace(
            elbo=logsumexp([m.variational.elbo for m in modes]),
            importance=logsumexp(
                [
                    m.variational.importance
                    if math.isfinite(m.variational.importance)
                    else m.variational.elbo
                    for m in modes
                ]
            ),
            effective_sample_size=min(m.variational.effective_sample_size for m in modes),
        )
    return SimpleNamespace(
        evidence=evidence,
        variational=variational,
        structure_log_prior=structure,
        modes=modes,
        log_posterior_laplace=evidence.laplace + structure,
        log_posterior_importance=evidence.importance + structure,
        log_posterior_elbo=(variational.elbo + structure) if variational else None,
        log_posterior_variational=(variational.importance + structure) if variational else None,
    )


def mode_lines(result, indent):
    """One line per distinct mode of an ``evaluate_modes`` result, for the raw output."""
    lines = []
    for index, mode in enumerate(result.modes):
        e, v = mode.evidence, mode.variational
        lines.append(
            f"{indent}mode {index + 1}: log_joint_at_mode {e.log_joint_at_mode:.1f}, laplace {e.laplace:.1f},"
            f" importance {e.importance:.1f}"
            + (f", elbo {v.elbo:.1f}, vi_importance {v.importance:.1f}" if v else "")
        )
    return lines
