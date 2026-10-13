# sonore-infer

Inferring what is in an auditory scene by inverting a generative model of
sound (analysis by synthesis), building on Cusimano, Hewitt & McDermott (2024),
*Listening with generative models*, Cognition 253, 105874, and using
[sonore](https://github.com/choyun1/sonore) as the NumPy reference that its
renderers are tested against.

The aim is a model of human auditory perception: one that reproduces
listening experiments, including listeners' errors. Research code, pre-alpha.
This project is developed with AI assistance (Claude).

## Install and test

    pip install -e ".[dev]"     # numpy, torch, plus sonore, pyro, pytest, ruff
    ruff check .
    python -m pytest -q

`pip install -e .` alone gives the differentiable core (numpy and torch). The
`stimuli` extra adds sonore, which makes the stimuli and serves as the test
oracle; the core never imports it (`tests/test_layers.py` checks this). CI
runs the same commands on Linux and Windows, Python 3.10 and 3.14.

## Directions

The work runs in four directions, each with its own design document in
`docs/design/`: reproducing the paper's model (BASS) and testing it on new
stimuli (milestones (a) and (b)); our own scene grammar (milestone (c));
online inference; and inferring a source together with its room.
[docs/README.md](docs/README.md) lists the documents by direction.

## Where things are

| Path | What it holds |
|---|---|
| `docs/design/` | Design documents: each lists numbered decisions, the options, and what Cho accepted. Start with [docs/README.md](docs/README.md). |
| `src/sonore_infer/` | The package (map below). |
| `src/sonore_infer/stimuli/` | Stimuli from App. C of the paper, and milestone (b)'s two notes, made with sonore. |
| `tests/` | One test file per module, plus `tests/stimuli/`. |
| `tools/` | Standalone scripts: experiment runners, checks of the model against sonore renders, and measurements. Every number in the design documents comes from one of them or is marked as an estimate. Index: [tools/README.md](tools/README.md). |
| `notebooks/` | `appendix_c_stimuli.ipynb`: listen to and plot the stimuli. |

The package holds two models that share no code beyond ERB conversions:

**Harmonic and whistle sources heard through a cochleagram** (milestones (a)
to (c): mistuned harmonic, two notes, ABA streaming)

| Module | Role |
|---|---|
| `cochleagram.py` | Differentiable cochleagram (gammatone, and BASS's FFT approximation) and the Gaussian likelihood |
| `sources.py` | Renderers: harmonic tone, whistle, events with inferred timing, trajectories |
| `priors.py` | The paper's priors (App. A, Tables A.1-A.2) |
| `scene.py` | A hypothesis as a list of sources; fitting and evidence for it |
| `fit.py` | Posterior mode by Adam |
| `evidence.py` | Marginal likelihood: Laplace, importance sampling, variational |
| `thresholds.py` | Thresholds from posterior probabilities (the paper's Eqn 2) |

**A random spectrogram heard through a room** (source and room inference,
`docs/design/source-reverb.md`)

| Module | Role |
|---|---|
| `spectrotemporal.py` | Prior on the spectrogram (McDermott, Wrobleski & Oxenham, 2011) |
| `room.py` | Expected band energy of a `synth_ir` room, and the RT60 prior |
| `blockpower.py` | Band energy in 10 ms blocks, and its expected value in a room |
| `sourceroom.py` | Posterior over RT60 and the spectrogram's statistics |

## Results

Raw outputs of the experiments are not in this repository yet; the design
documents cite them by path (`results/...`).

## License

MIT.
