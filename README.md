# sonore-infer

[![tests](https://github.com/choyun1/sonore-infer/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/choyun1/sonore-infer/actions/workflows/tests.yml) [![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Inferring what is in an auditory scene by inverting a generative model of
sound (analysis by synthesis), building on Cusimano, Hewitt & McDermott (2024),
*Listening with generative models*, Cognition 253, 105874, and using
[sonore](https://github.com/choyun1/sonore) as the NumPy reference that its
renderers are tested against.

The aim is a model of human auditory perception: one that reproduces
listening experiments, including listeners' errors. Research code, pre-alpha.
This project is developed with AI assistance (Claude).

## Install and test

    pip install -e ".[dev]"     # numpy, scipy, torch, plus sonore, pyro, pytest, ruff, mypy
    ruff check .
    ruff format --check .
    mypy
    python -m pytest -q

`pip install -e .` alone gives the differentiable core (numpy, scipy and torch). The
`stimuli` extra adds sonore, which makes the stimuli and serves as the test
oracle; the core never imports it (`tests/test_layers.py` checks this). CI
runs the same commands on Linux and Windows, Python 3.11 and 3.14.

## Directions

The work runs in four directions, each with its own design document in
`docs/design/`: reproducing the paper's model (BASS) and testing it on new
stimuli (milestones (a) and (b)); our own scene grammar (milestone (c));
online inference; and inferring a source together with its room.
[docs/README.md](docs/README.md) lists the documents by direction.

## Where things are

| Path | What it holds |
|---|---|
| `docs/` | [overview.md](docs/overview.md): the project in a page; [methods.md](docs/methods.md): the models and inference, defined mathematically. |
| `docs/design/` | Design documents: each lists numbered decisions, the options, and what Cho accepted. Start with [docs/README.md](docs/README.md). |
| `src/sonore_infer/` | The package (map below). |
| `src/sonore_infer/stimuli/` | Stimuli from App. C of the paper (mistuned harmonic, asynchronous onsets, ABA bistability and buildup), and milestone (b)'s two notes, made with sonore. |
| `tests/` | One test file per module, plus `tests/stimuli/`. |
| `tools/` | Standalone scripts: experiment runners and measurements, one folder per direction. Every number in the design documents comes from one of them or is marked as an estimate. Index: [tools/README.md](tools/README.md). |
| `website/` | Builds the project website from this Markdown ([website/README.md](website/README.md)). |
| `notebooks/` | `appendix_c_stimuli.ipynb`: listen to and plot the stimuli. |

The package holds two models that share no code beyond the ERB scale
(`erb.py`, sonore's conversions written for torch too):

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

## Results

[`results/`](results/README.md) has one page per experiment, with the
committed outputs each number comes from. These pages are under review
and may change.

## License

MIT.
