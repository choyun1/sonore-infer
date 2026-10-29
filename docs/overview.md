# sonore-infer

> **Under review.** This is a working record, not reviewed results. The
> work was done with AI assistance (Claude) and has not yet been checked
> carefully. Numbers, methods and interpretations may change.

sonore-infer is research code for a model of human hearing: one that,
given a sound, infers which sources made it, and that should hear what
listeners hear in listening experiments, including their errors. It builds
on Cusimano, Hewitt & McDermott (2024), *Listening with generative models*
(Cognition 253, 105874), and is written from the paper, not from its code.
Pre-alpha.

## The idea

Hearing is treated as inference in a generative model of sound
(analysis by synthesis). The model says how a scene is made: a few
sources, each a harmonic tone or a whistle, each with a frequency and a
level that change smoothly over time, sounding in one or more events. A
renderer turns a scene into a waveform and a cochleagram turns the
waveform into a time-frequency picture, roughly what the ear passes on.

To hear a sound, the model asks which scenes could have made it. Each
answer to a question such as "one source or two?" is a separate
hypothesis. For each one the model finds the best-fitting scene and
estimates its **evidence**: how well the hypothesis explains the sound,
averaged over everything it leaves unknown. A hypothesis with more sources
always fits at least as well, but spreads its prior over more unknowns,
so the evidence has an Occam penalty built in. The posterior odds of two
hypotheses then say what the model hears, and how sure it is.

The definitions are on the [Methods](methods.md) page: the priors, the
renderer, the cochleagram and likelihood, and how the evidence is
estimated.

## Why it is built this way

- **Listeners are the reference, not the paper.** The paper's model is a
  strong starting point, and the first experiments reproduce it. A
  difference from the paper is chased only if it is larger than seed and
  machine noise and either shows our inference is not doing what we claim
  or changes how the model compares with listeners. Every results page
  says where we match the paper, where we don't, and where we chose not to
  care ([Against the paper](../results/README.md#against-the-paper)).
- **Our own code.** The paper's code (BASS) has no license, so none of it
  is in this repository. We read it only to check details, and ran it
  locally as a check ([A6](../results/A6-bass-own-code.md)).
- **Decisions first.** Every change starts as a design document with
  numbered decisions, the options and a recommendation, and is built only
  once the decisions are accepted ([Design](README.md)).
- **Every number has a source.** Each number on a results page comes from
  a committed output and the script that made it, or is marked as an
  estimate.
- **Differentiable throughout.** The renderer and cochleagram are written
  in PyTorch, so a scene is fitted by gradient and the evidence's
  curvature comes from automatic differentiation. sonore, a separate NumPy
  library, makes the stimuli and serves as the reference the renderers are
  tested against; the core never imports it.

## What has been done

The work runs in five directions. Each has its design documents and its
results pages.

| Direction | What it asks | Results |
|---|---|---|
| Reproducing the paper's model | Do we get the paper's mistuned-harmonic thresholds (Fig. 7D)? And what does the same model hear in two simultaneous notes? | [A1–A8](../results/README.md#experiments), [B1–B4](../results/B2-two-notes-variational.md) |
| Our own scene grammar | Sources with several events, inferred Gaussian-process kernels, a search over structures, and ABA streaming (Fig. 8D) | [C1–C3](../results/C2-aba-streaming.md) |
| Online inference | The evidence accumulated window by window as the sound arrives, and the buildup of streaming (Fig. 8F) | [O1](../results/O1-online-buildup.md), [O2](../results/O2-online-speed.md) |
| Relative pitch | Whether the model hears the direction of a step between Shepard tones as Shepard's (1964) listeners did | [P1–P7](../results/P3-shepard-pairs.md) |
| Source and room | Whether a room's reverberation time can be told apart from the source from one sound | [R1](../results/R1-source-and-room.md), [R2](../results/R2-atypical-rooms.md) |

The [results index](../results/README.md) lists every experiment with its
result in one line, and what is planned next: order effects in streaming,
and a likelihood that keeps timing and phase.

## Where things are

- [Methods](methods.md): the model and the inference engine, defined.
- [Results](../results/README.md): one page per experiment.
- [Design](README.md): the design documents and their decisions.
- [Code](../README.md): installing, testing, and a map of the package.
- Source: [github.com/choyun1/sonore-infer](https://github.com/choyun1/sonore-infer), MIT license.

This project is developed with AI assistance (Claude).
