# tools

Standalone scripts. Each one's docstring says what it does, how to run it
and what it prints; numbers in the design documents cite the script that
measured them. Most need the `stimuli` extra (sonore).

In sonore's terms (its docs/design/philosophy.md), the scripts that import
`sonore_infer` measure what this package does; none of them is an
independent check of it. `table_a2_quartiles.py` and `rt60_prior_fit.py` do
not import the package.

## Milestone (a): the mistuned harmonic (App. C.6)

| Script | What it does |
|---|---|
| `fit_mistuned_harmonic.py` | First step: best fits of one source and of two, without priors |
| `compare_mistuned_harmonic.py` | One source or two by marginal likelihood. Its flags add the model's parts one at a time (FFT cochleagram, inferred timing, trajectories), as the steps of milestone (a) were run |
| `scene_mistuned_harmonic.py` | The full model of the previous script re-run through `sonore_infer.scene`, printing the same table so the two can be compared line by line |
| `mistuning_thresholds.py` | Thresholds (Eqn 2) from the output of `compare_mistuned_harmonic.py` |

## Milestone (b): two simultaneous notes

| Script | What it does |
|---|---|
| `compare_two_notes.py` | One note or two (or one plus whistles) by marginal likelihood |
| `two_notes_summary.py` | Log-odds and threshold tables from `compare_two_notes.py` outputs |
| `run_seeds.py` | Re-runs milestones (a) and (b) for more seeds, in parallel and resumably |

## Source and room (docs/design/source-reverb.md)

| Script | What it does |
|---|---|
| `rt60_prior_fit.py` | RT60 prior and typical DRR from Traer & McDermott's IR survey (R2, R8) |
| `room_gain_check.py` | Measures `RoomGain` against `synth_ir` renders (R3) |
| `block_power_check.py` | Measures `BlockPower`'s expected energies against sonore renders (R3-R4) |
| `likelihood_slices.py` | Likelihood along one parameter at a time through the truth (R4-R5) |
| `recovery.py` | Experiment (a) of R7: recover RT60 and the source's statistics |
| `atypical_rooms.py` | Experiment (b) of R7: the model heard through atypical rooms |

## Result pages (results/)

| Script | What it does |
|---|---|
| `results_plots.py` | Draws the SVG plots in `results/plots/` from the committed outputs in `results/data/` |

## Measurements of the cochleagram, and readings of the paper

| Script | What it does |
|---|---|
| `channel_bandwidths.py` | Measures the effective bandwidth of the cochleagram's channels, gammatone against FFT |
| `table_a2_quartiles.py` | Which reading of Table A.2's parameters reproduces its quartiles (C3) |

## Machine and timing

| Script | What it does |
|---|---|
| `machine_specs.py` | Prints the specs of this machine that matter for timing (D6) |
| `time_cochleagram.py` | Seconds per cochleagram forward and backward pass (D6) |
