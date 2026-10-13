# Documentation

Every design document follows the same pattern: the problem, numbered
decisions with options and a recommendation, and a closing section recording
what Cho accepted and when. Code and docstrings cite decisions by their
number (for example "design C3 (B)").

## Design documents, in the order they were written

| Document | Covers | Decisions | Accepted |
|---|---|---|---|
| [design/design-v1.md](design/design-v1.md) | The project as a whole: framework, relation to sonore, milestones, likelihood, inference, compute, priors | D1-D8 | 2026-10-05 |
| [design/bass-code-check.md](design/bass-code-check.md) | Three items from design-v1 §1.1 checked against BASS's code at 48e3755 | none | n/a |
| [design/milestone-b.md](design/milestone-b.md) | Milestone (b): two simultaneous notes | B1-B6 | 2026-10-07 |
| [design/source-reverb.md](design/source-reverb.md) | Inferring a source and its room (RT60) | R1-R8 | 2026-10-10 |
| [design/milestone-c.md](design/milestone-c.md) | Milestone (c): scene grammar (several events per source, inferred GP kernels, search over structures) | C1-C11 | 2026-10-11 |
| [design/online.md](design/online.md) | Online inference: evidence accumulated window by window, and a phone-app roadmap | O1-O9 | 2026-10-13 |

## Names used across the code

- **Milestones (a), (b), (c)**: the mistuned harmonic (App. C.6 of the
  paper), two simultaneous notes, and the scene grammar with ABA streaming
  (App. C.10). Defined in design-v1 D3 and the milestone documents.
- **Experiments (a) and (b)** in `tools/recovery.py` and
  `tools/atypical_rooms.py` are the two experiments of source-reverb.md R7,
  not milestones (a) and (b).
- **Decision numbers** carry the letter of their document: D (design-v1),
  B (milestone-b), R (source-reverb), C (milestone-c), O (online).
