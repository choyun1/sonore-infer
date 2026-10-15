# Documentation

Every design document follows the same pattern: the problem, numbered
decisions with options and a recommendation, and a closing section recording
what Cho accepted and when. Code and docstrings cite decisions by their
number (for example "design C3 (B)").

## Design documents, by direction

The project has four directions. They share the package (whose modules are
arranged by what they are, not by direction; see the main README), but each
has its own design documents, tools and results.

| Direction | Document | Decisions | Accepted |
|---|---|---|---|
| **Reproducing the paper's model (BASS)** and testing it on new stimuli | [design/design-v1.md](design/design-v1.md): the project as a whole, and milestone (a), the mistuned harmonic | D1-D8 | 2026-10-05 |
| | [design/bass-code-check.md](design/bass-code-check.md): three items from design-v1 §1.1 checked against BASS's code at 48e3755 | none | n/a |
| | [design/milestone-b.md](design/milestone-b.md): milestone (b), two simultaneous notes | B1-B6 | 2026-10-07 |
| **Our own scene grammar** | [design/milestone-c.md](design/milestone-c.md): milestone (c), several events per source, inferred GP kernels, search over structures, ABA streaming | C1-C11 | 2026-10-11 |
| **Online inference** | [design/online.md](design/online.md): evidence accumulated window by window, and a phone-app roadmap | O1-O10 | 2026-10-13, O10 2026-10-14 |
| | [design/online-realtime.md](design/online-realtime.md): what real time would take, measured on the buildup runs | none | n/a |
| | [design/online-speed.md](design/online-speed.md): faster online inference, each change checked against O1 | S1-S6 | draft |
| | [design/order-effects.md](design/order-effects.md): reset after silence and the previous trial's context, run online | E1-E6 | draft |
| **Source and room** | [design/source-reverb.md](design/source-reverb.md): a random spectrogram and its room's RT60, inferred together | R1-R8 | 2026-10-10 |

## Names used across the code

- **Milestones (a), (b), (c)**: the mistuned harmonic (App. C.6 of the
  paper), two simultaneous notes, and the scene grammar with ABA streaming
  (App. C.10). Defined in design-v1 D3 and the milestone documents.
- **Experiments (a) and (b)** in `tools/source_room/recovery.py` and
  `tools/source_room/atypical_rooms.py` are the two experiments of source-reverb.md R7,
  not milestones (a) and (b).
- **Decision numbers** carry the letter of their document: D (design-v1),
  B (milestone-b), R (source-reverb), C (milestone-c), O (online),
  S (online-speed), E (order-effects).
- **Experiment IDs** (A1–A8, B1–B4, C1, R1, R2; pages in
  [`results/`](../results/README.md)) share letters with decision numbers,
  so outside their own document both are written with their word: "experiment B1" is a result
  page, "decision B1" is a section of milestone-b.md. Inside a design
  document a bare number is one of its decisions; on a result page a bare
  ID is an experiment.
