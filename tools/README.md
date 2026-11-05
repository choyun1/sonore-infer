# tools

Standalone scripts. Each one's docstring says what it does, how to run it
and what it prints; numbers in the design documents cite the script that
measured them. Most need the `stimuli` extra (sonore).

A script is defined by the experiment it runs, so the scripts are in one
folder per direction of the project, as sonore's philosophy puts it,
"folders follow meaning". The package in `src/` is not split this way,
because its modules serve every direction. Run each script from the
repository root, for example `python tools/grammar/compare_aba.py`.

| Folder | Direction |
|---|---|
| `bass/` | Milestones (a) and (b): the paper's model (BASS) on the mistuned harmonic and on two notes |
| `grammar/` | Milestone (c): scene grammar |
| `source_room/` | Source and room |
| `online/` | Online inference |
| `measurements/` | Measurements of the cochleagram and the machine, and readings of the paper |
| (top level) | `results_plots.py` and `run_seeds.py`, which serve several directions |

In sonore's terms (its docs/design/philosophy.md), the scripts that import
`sonore_infer` measure what this package does; none of them is an
independent check of it. `table_a2_quartiles.py` and `rt60_prior_fit.py` do
not import the package.

## Milestone (a): the mistuned harmonic (App. C.6)

| Script | What it does |
|---|---|
| `bass/fit_mistuned_harmonic.py` | First step: best fits of one source and of two, without priors |
| `bass/compare_mistuned_harmonic.py` | One source or two by marginal likelihood. Its flags add the model's parts one at a time (FFT cochleagram, inferred timing, trajectories), as the steps of milestone (a) were run |
| `bass/scene_mistuned_harmonic.py` | The full model of the previous script re-run through `sonore_infer.scene`, printing the same table so the two can be compared line by line |
| `bass/mistuning_thresholds.py` | Thresholds (Eqn 2) from the output of `compare_mistuned_harmonic.py` |
| `bass/variational_thresholds.py` | Thresholds (Eqn 2) by all four evidence estimates, from `scene_mistuned_harmonic.py` run with `--variational-steps` |
| `bass/sparse_options.py` | The relative-pitch-prior options (sparse spectra, a lower f0 floor) for the milestone (a) and (b) tools, scoring each start's mode separately (decision P14) |

## Milestone (b): two simultaneous notes

| Script | What it does |
|---|---|
| `bass/compare_two_notes.py` | One note or two (or one plus whistles) by marginal likelihood |
| `bass/two_notes_summary.py` | Log-odds and threshold tables from `compare_two_notes.py` outputs |
| `run_seeds.py` | Re-runs milestones (a) and (b) for more seeds, and the ABA map (`--only aba`), in parallel and resumably |

## Milestone (c): scene grammar

| Script | What it does |
|---|---|
| `grammar/compare_aba.py` | One stream or two for an ABA sequence by marginal likelihood (App. C.10) |
| `grammar/aba_map.py` | The ABA map of log odds from `compare_aba.py` outputs, combining reruns of one hypothesis |
| `grammar/aba_fit_steps.py` | How far one ABA hypothesis's fit has got after a given number of Adam steps |
| `grammar/search_scene.py` | Searches for a stimulus's best hypothesis without being given its structure (decisions C9, C10) |
| `grammar/search_starts.py` | Fits the search's two-note hypothesis to the just fifth (0 ms) from one of five starts |

## Source and room (docs/design/source-reverb.md)

| Script | What it does |
|---|---|
| `source_room/rt60_prior_fit.py` | RT60 prior and typical DRR from Traer & McDermott's IR survey (decisions R2, R8) |
| `source_room/room_gain_error.py` | Measures `RoomGain` against `synth_ir` renders (decision R3) |
| `source_room/block_power_error.py` | Measures `BlockPower`'s expected energies against sonore renders (decisions R3-R4) |
| `source_room/likelihood_slices.py` | Likelihood along one parameter at a time through the truth (decisions R4-R5) |
| `source_room/recovery.py` | Experiment (a) of decision R7: recover RT60 and the source's statistics |
| `source_room/atypical_rooms.py` | Experiment (b) of decision R7: the model heard through atypical rooms |

## Result pages (results/)

| Script | What it does |
|---|---|
| `results_plots.py` | Draws the SVG plots in `results/plots/` from the committed outputs in `results/data/` |

## Relative pitch (docs/design/relative-pitch.md)

| Script | What it does |
|---|---|
| `relative_pitch/one_tone.py` | How the model hears one Shepard tone: one harmonic source per octave reading, or whistles (P01, P02, P05, P06) |
| `relative_pitch/combine_starts.py` | One log posterior per tone and hypothesis from `one_tone.py` outputs fitted from several starts |
| `relative_pitch/search_tone.py` | The scene search (`grammar/search_scene.py`) on one Shepard tone, as a check on `one_tone.py`'s hypotheses |
| `relative_pitch/pair.py` | Whether a pair of Shepard tones is heard going up or down (P03, P08) |
| `relative_pitch/pair_summary.py` | P(up) per pair from `pair.py` outputs, beside Shepard's listeners (Fig. 3) |
| `relative_pitch/decompose.py` | Where a pair's up and down hypotheses differ, term by term (P09) |
| `relative_pitch/listener_noise.py` | Decision noise and lapse on the model's log odds fitted to Shepard's groups, and interval noise (P09, P11) |
| `relative_pitch/run_seeds.py` | Reruns P09's pairs with both options on several seeds (decision P27, P12) |
| `relative_pitch/average_seeds.py` | Averages `pair.py` outputs over seeds (P12) |

## Online inference (docs/design/online.md)

| Script | What it does |
|---|---|
| `online/online_buildup.py` | Buildup of streaming (App. C.11) in one online pass, against batch fits at each triplet's end |
| `online/leaky_odds.py` | Leaky log odds (decision O10) from `online_buildup.py` outputs, no new inference |
| `online/online_reset.py` | Reset of streaming after silence: Beauvois & Meddis (1997) and Cusack et al. (2004) sequences online, with restart hypotheses (order-effects E3) |
| `online/reset_summary.py` | O03's numbers from `online_reset.py` outputs: evidence over the test, leaky decision, restart gains, second sequence against the first, no new inference |
| `online/reset_rest_check.py` | Order-effects E8's diagnostic: `online_reset.py` with rests longer than a cutoff left out of the rhythm prior |
| `online/online_cost.py` | Where an online window's time goes (render, cochleagram, prior, Adam step, Hessian) as the sound grows |

## Measurements of the cochleagram, and readings of the paper

| Script | What it does |
|---|---|
| `measurements/channel_bandwidths.py` | Measures the effective bandwidth of the cochleagram's channels, gammatone against FFT |
| `measurements/table_a2_quartiles.py` | Which reading of Table A.2's parameters reproduces its quartiles (decision C3) |
| `measurements/sonore_stimuli.py` | Writes every stimulus made with sonore to a file, and compares two such files, to check a sonore upgrade |
| `measurements/streamed_cochleagram.py` | Our cochleagram from sonore's streamed gammatone envelopes, 10 ms at a time, against `Cochleagram`, and its time per block |

## Machine and timing

| Script | What it does |
|---|---|
| `measurements/machine_specs.py` | Prints the specs of this machine that matter for timing (decision D6) |
| `measurements/time_cochleagram.py` | Seconds per cochleagram forward and backward pass (decision D6) |
