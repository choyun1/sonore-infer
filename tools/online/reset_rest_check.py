"""E8's diagnostic: online_reset.py with long rests left out of the rhythm prior (order-effects-reset.md).

Each source's log rests share one normal-gamma, integrated out
(:class:`sonore_infer.priors.NormalGammaMarginal`). Design §2 infers that
the rest across a silence, very unlikely under a regular source's pooled
rests, is what makes the first triplet after a gap favour one stream. This
tool runs ``online_reset.py`` unchanged except that rests longer than
``--cutoff`` s are dropped from that pool: they add nothing to the log
prior, so a source crossing a silence pays for it in nothing but its other
latents. Durations, and rests shorter than the cutoff, are scored as
before. A diagnostic only; the library is untouched.

    python tools/online/reset_rest_check.py --cutoff 0.75 cusack --semitones 4 --silence 5 \\
        --first-duration 2 --restart

Compare with the same command through ``online_reset.py``.
"""

import math
import os
import runpy
import sys

import torch

from sonore_infer import scene
from sonore_infer.priors import NormalGammaMarginal


class _ShortRestsOnly(NormalGammaMarginal):
    """The normal-gamma marginal of the values below ``log(cutoff)``; longer ones add 0."""

    log_cutoff = math.inf

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        keep = value < self.log_cutoff
        if bool(keep.all()):
            return super().log_prob(value)
        kept = value[keep]
        if kept.numel() == 0:
            return value.sum() * 0.0
        return NormalGammaMarginal(kept.numel(), self.dtype).log_prob(kept)


def main():
    if len(sys.argv) < 3 or sys.argv[1] != "--cutoff":
        sys.exit(__doc__)
    _ShortRestsOnly.log_cutoff = math.log(float(sys.argv[2]))
    scene.NormalGammaMarginal = _ShortRestsOnly  # the scene's sampler and scorer look it up here
    print(f"rest check: rests longer than {sys.argv[2]} s left out of the rhythm prior")
    sys.argv = ["online_reset.py", *sys.argv[3:]]
    runpy.run_path(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "online_reset.py"), run_name="__main__"
    )


if __name__ == "__main__":
    main()
