# sonore-infer

[![tests](https://github.com/choyun1/sonore-infer/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/choyun1/sonore-infer/actions/workflows/tests.yml) [![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**Hearing a scene by inverting a generative model of sound.**

sonore-infer infers what is in an auditory scene (which sources there are,
and what each one is doing) by searching for the sounds that would have
produced what was heard: analysis by synthesis. It builds on Cusimano,
Hewitt & McDermott (2024), *Listening with generative models*, and is
written in PyTorch so that the renderers and the cochleagram are
differentiable. [sonore](https://github.com/choyun1/sonore) makes the
stimuli and is the NumPy reference that the package's signal processing is
tested against.

The aim is a model of human auditory perception: one that reproduces
listening experiments, including listeners' errors. So far it covers the
paper's mistuned-harmonic experiment, two simultaneous notes, ABA
streaming, online inference window by window, and a source heard together
with its room.

sonore-infer is research code, pre-alpha. Its API may change without
warning, and its results are under review by Adrian Cho and may change.

<!-- Contents: one entry per ## section, in order (tests/test_readme.py checks it). -->
<table><tr><td>

<small><b>Contents</b><br>
1&ensp;Using it<br>
&emsp;1.1&ensp;[Install](#install)<br>
&emsp;1.2&ensp;[Where things are](#where-things-are)<br>
&emsp;1.3&ensp;[What's in it](#whats-in-it)<br>
2&ensp;Background<br>
&emsp;2.1&ensp;[Directions and design documents](#directions-and-design-documents)<br>
&emsp;2.2&ensp;[Results](#results)<br>
&emsp;2.3&ensp;[References](#references)<br>
3&ensp;The project<br>
&emsp;3.1&ensp;[Development](#development)<br>
&emsp;3.2&ensp;[How sonore-infer was developed](#how-sonore-infer-was-developed)<br>
&emsp;3.3&ensp;[License and citation](#license-and-citation)</small>

</td></tr></table>

## Install

    pip install -e .              # the differentiable core: numpy, scipy, torch
    pip install -e ".[stimuli]"   # adds sonore, which makes the stimuli
    pip install -e ".[pyro]"      # adds Pyro, for Scene.model() (design C7)

The core never imports sonore (`tests/test_layers.py` checks this). The
scripts in `tools/` and the notebook need the `stimuli` extra.

## Where things are

| Path | What it holds |
|---|---|
| `docs/design/` | Design documents: each lists numbered decisions, the options, and what Cho accepted. Start with [docs/README.md](docs/README.md). |
| `src/sonore_infer/` | The package (see "What's in it" below). |
| `src/sonore_infer/stimuli/` | Stimuli from App. C of the paper (mistuned harmonic, asynchronous onsets, ABA bistability and buildup), and milestone (b)'s two notes, made with sonore. |
| `tests/` | One test file per module, plus `tests/stimuli/`. |
| `tools/` | Standalone scripts: experiment runners and measurements. Every number in the design documents comes from one of them or is marked as an estimate. Index: [tools/README.md](tools/README.md). |
| `results/` | One page per experiment, with the committed outputs and plots behind it. |
| `notebooks/` | `appendix_c_stimuli.ipynb`: listen to and plot the stimuli. |

## What's in it

The package holds two models that share no code beyond the ERB scale
(`erb.py`, sonore's conversions written for torch too). Its modules are
arranged by what they are, not by direction.

**Harmonic and whistle sources heard through a cochleagram** (milestones (a)
to (c), and online inference: mistuned harmonic, two notes, ABA streaming)

| Module | Role |
|---|---|
| `cochleagram.py` | Differentiable cochleagram (gammatone, and BASS's FFT approximation) and the Gaussian likelihood |
| `sources.py` | Renderers: harmonic tone, whistle, events with inferred timing, trajectories |
| `priors.py` | The paper's priors (App. A, Tables A.1-A.2) |
| `scene.py` | A hypothesis as a list of sources; fitting and evidence for it |
| `detect.py` | Candidate events from the cochleagram: onsets, spectral peaks, harmonic f0s |
| `search.py` | Search over hypotheses in rounds of onsets, with a residual pass and cleanup moves |
| `fit.py` | Posterior mode by Adam |
| `evidence.py` | Marginal likelihood: Laplace, importance sampling, variational |
| `thresholds.py` | Thresholds from posterior probabilities (the paper's Eqn 2) |
| `online.py` | The scene's evidence accumulated window by window, with a fixed lag |

**A random spectrogram heard through a room** (source and room inference,
`docs/design/source-reverb.md`)

| Module | Role |
|---|---|
| `spectrotemporal.py` | Prior on the spectrogram (McDermott, Wrobleski & Oxenham, 2011) |
| `room.py` | Expected band energy of a `synth_ir` room, and the RT60 prior |
| `blockpower.py` | Band energy in 10 ms blocks, and its expected value in a room |
| `sourceroom.py` | Posterior over RT60 and the spectrogram's statistics |

## Directions and design documents

The work runs in four directions, each with its own design documents in
`docs/design/`: reproducing the paper's model (BASS) and testing it on new
stimuli (milestones (a) and (b)); our own scene grammar (milestone (c));
online inference; and inferring a source together with its room.

Every direction is designed before it is built. A design document states
the problem, puts forward numbered decisions with options and a
recommendation, and records which option Cho accepted and when; code and
docstrings cite decisions by their number. [docs/README.md](docs/README.md)
lists the documents by direction and defines the names used across the
code.

## Results

[`results/`](results/README.md) has one page per experiment: what it asks,
the part of the paper it is compared with, the paper's values as we read
them, ours, and the command that made them. Every number on a page comes
from a committed output under `results/data/` or is marked as an estimate;
the exceptions, values by the rule in BASS's analysis code, are said so on
their page. The plots are drawn from the outputs by `tools/results_plots.py`. The pages are a working record,
under review by Adrian Cho, and may change.

## References

The works the package implements or rebuilds stimuli from. Those the
stimuli come from are cited as in Cusimano et al.'s (2024) reference
list. Each design document lists the further works it draws on.

Bregman, A. S., & Ahad, P. (1996). *Demonstrations of auditory scene
analysis: The perceptual organization of sound* [Audio CD]. McGill University.
https://themusiclab.github.io/bregman-archive/downloadstoc.htm

Cusimano, M., Hewitt, L. B., & McDermott, J. H. (2024). Listening with
generative models. *Cognition*, *253*, 105874.
https://doi.org/10.1016/j.cognition.2024.105874

Darwin, C. J., & Sutherland, N. S. (1984). Grouping frequency components
of vowels: When is a harmonic not a harmonic? *The Quarterly Journal of
Experimental Psychology*, *36*(2), 193–208.

Ellis, D. P. W. (2009). *Gammatone-like spectrograms* [Computer software].
https://www.ee.columbia.edu/~dpwe/resources/matlab/gammatonegram/

Glasberg, B. R., & Moore, B. C. J. (1990). Derivation of auditory filter
shapes from notched-noise data. *Hearing Research*, *47*(1–2), 103–138.
https://doi.org/10.1016/0378-5955(90)90170-T

Klatt, D. H. (1980). Software for a cascade/parallel formant synthesizer.
*The Journal of the Acoustical Society of America*, *67*(3), 971–995.
https://doi.org/10.1121/1.383940

McDermott, J. H., Wrobleski, D., & Oxenham, A. J. (2011). Recovering sound
sources from embedded repetition. *Proceedings of the National Academy of
Sciences*, *108*(3), 1188–1193. https://doi.org/10.1073/pnas.1004765108

Moore, B. C. J., Glasberg, B. R., & Peters, R. W. (1986). Thresholds for
hearing mistuned partials as separate tones in harmonic complexes. *The
Journal of the Acoustical Society of America*, *80*(2), 479–483.
https://doi.org/10.1121/1.394043

Thompson, S. K., Carlyon, R. P., & Cusack, R. (2011). An objective
measurement of the build-up of auditory streaming and of its modulation by
attention. *Journal of Experimental Psychology: Human Perception and
Performance*, *37*(4), 1253–1262.

Traer, J., & McDermott, J. H. (2016). Statistics of natural reverberation
enable perceptual separation of sound and space. *Proceedings of the
National Academy of Sciences*, *113*(48), E7856–E7865.
https://doi.org/10.1073/pnas.1612524113

van Noorden, L. P. A. S. (1975). *Temporal coherence in the perception of
tone sequences* [Doctoral dissertation, Eindhoven University of
Technology].

## Development

```
pip install -e ".[dev]"     # the core, plus sonore, Pyro, pytest, ruff and mypy
ruff check .
ruff format --check .
mypy                        # src/ only
python -m pytest -q
```

CI runs the same commands on every pull request, on Linux and Windows,
with Python 3.11 and 3.14 (`.github/workflows/tests.yml`). After changing
any output under `results/data/`, rerun `python tools/results_plots.py`
(`tests/test_results_plots.py` fails until then).

## How sonore-infer was developed

sonore-infer is developed the same way as sonore, together with Claude,
Anthropic's AI assistant (Claude Code), during 2026.

**What Claude did.** Wrote most of the code, tests and documentation,
delivered as pull requests; drafted the design documents; and wrote and
ran the scripts in `tools/`, including the experiments on the result pages.

**What Cho did.** Decided what the project is for and which directions and
milestones it takes, chose among the options each design document put
forward, and reviewed and merged each pull request.

**How it is verified.** Each module has its own tests, run on every pull
request; the cochleagram, the ERB scale, the room model and the
spectrogram prior are tested against sonore's implementations, and the
fits against stimuli sonore makes. Every number a design or a result page rests on is made by a
script in `tools/` from a committed output, or is marked as an estimate.
These are measurements of what the package does, not independent checks
of it, and the result pages have not yet been checked carefully.

## License and citation

MIT; see [LICENSE](LICENSE). There is no citation file yet. If this code
is useful to you, please cite Cusimano, Hewitt & McDermott (2024), whose
model it builds on, and [sonore](https://github.com/choyun1/sonore#license-and-citation),
which makes its stimuli.
