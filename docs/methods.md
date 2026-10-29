# Methods

> **Under review.** This is a working record, not reviewed results. The
> work was done with AI assistance (Claude) and has not yet been checked
> carefully. Definitions may change with the code.

This page defines the models and the inference engine mathematically, as
the code on `main` implements them. Each definition names the module that
computes it. Where the code departs from Cusimano, Hewitt & McDermott
(2024), *Listening with generative models* (Cognition 253, 105874), the
departure is marked **Departure** and points to the decision that made it.
Section, table and equation numbers in square brackets, such as [App. A.2],
are the paper's.

There are two models. They share only the ERB scale:

1. **Harmonic sources and whistles heard through a cochleagram**
   (experiments A, B, C, O and P): §1 to §3.
2. **A random spectrogram heard through a room** (experiments R1 and R2):
   §4.

## Notation

| Symbol | Meaning |
|---|---|
| $\mathbf{y}$ | the observation: the cochleagram of the sound heard |
| $H$ | a hypothesis: how many sources, of which kinds, with how many events each (the discrete structure) |
| $\boldsymbol\theta$ | the continuous latents of $H$ (frequencies, levels, spectrum, timing), $d$ numbers in all |
| $T$ | the sound's duration [s] |
| $t_g = g\,\Delta$ | the trajectory grid, $\Delta = 10$ ms, $g = 0, \dots, G$ with $G = \mathrm{round}(T/\Delta)$ |
| $E(f)$ | ERB number of frequency $f$ [Cams]: $E(f) = 9.265 \ln\left(1 + \frac{f}{24.7 \times 9.265}\right)$ (Glasberg & Moore, 1990) |
| $B(f)$ | equivalent rectangular bandwidth at $f$ [Hz]: $B(f) = 24.7\,(4.37 f / 1000 + 1)$ |
| $r_0$ | the level reference, an RMS of $10^{-6}$; levels are in dB re $r_0$ |
| $\mathcal{N}(x; \mu, s^2)$ | the normal density |

## 1. The generative model

A scene is a set of sources. Each source is a **harmonic** source or a
**whistle** (a single frequency), and emits one or more **events**: stretches
of time when it sounds. A source's frequency and level follow smooth
trajectories over time, shared by its events. The sound is the sum of the
sources' waveforms.

### 1.1 Prior on the structure

A hypothesis $H$ has $n \ge 1$ sources, source $i$ of kind $\kappa_i$ with
$m_i \ge 1$ events. Its prior (`priors.log_prior_structure`) is

$$
p(H) = \underbrace{\frac{(\lambda T)^n e^{-\lambda T}}{n!\,\left(1 - e^{-\lambda T}\right)}}_{\text{number of sources}}
\cdot \underbrace{n!}_{\text{labelings}}
\cdot \underbrace{\left(\tfrac{1}{3}\right)^n}_{\text{kinds}}
\cdot \prod_{i=1}^{n} \underbrace{\tfrac{1}{2}\left(\tfrac{1}{2}\right)^{m_i - 1}}_{\text{events}},
\qquad \lambda = 1\ \text{s}^{-1}.
$$

The number of sources is Poisson with one source per second, conditioned
on at least one [Table A.1; design D8]. Each kind is uniform over noise,
harmonic and whistle, and the number of events is Geometric(1/2) [Table
A.1]. A fitted hypothesis is one mode with labeled sources; the $n!$
relabelings are other modes of equal mass, so the prior counts them
(design C6).

- **Departure.** Noise sources are not implemented (design C5); the factor
  1/3 still counts them as a possible kind.
- **Departure from BASS's code, not the paper.** BASS scores the number of
  sources as Poisson(1) per scene, without the duration or the truncation
  ([bass-code-check.md](design/bass-code-check.md), item 1). We follow the
  paper.

### 1.2 Event timing

For a source with events $j = 1, \dots, m$: the first onset $a_1$, the
durations $d_j$ and the rests $r_j$ between events, so that
$a_{j+1} = a_j + d_j + r_j$ [App. A.2, Eqns A.8–A.12]
(`scene._Events`).

$$
a_1 \sim \mathcal{U}(0, T).
$$

The log durations share an unknown mean and precision $(\mu, \tau)$ drawn
from a normal-gamma with $(\mu_0, \lambda_0, \alpha_0, \beta_0) =
(-1, 0.5, 2.5, 1)$ [Table A.1]; so do the log rests, with their own
$(\mu, \tau)$. Both are integrated out in closed form (design C3 (D)). For
$k$ values $x_1, \dots, x_k$ with mean $\bar x$ and sum of squares
$S = \sum_i (x_i - \bar x)^2$ (`priors.NormalGammaMarginal`):

$$
\log p(x_{1:k}) = \log\Gamma(\alpha_k) - \log\Gamma(\alpha_0) + \alpha_0 \log\beta_0 - \alpha_k \log\beta_k
+ \tfrac12 \log\frac{\lambda_0}{\lambda_k} - \tfrac{k}{2}\log 2\pi,
$$

$$
\lambda_k = \lambda_0 + k, \qquad \alpha_k = \alpha_0 + \tfrac{k}{2}, \qquad
\beta_k = \beta_0 + \tfrac{S}{2} + \frac{\lambda_0 k (\bar x - \mu_0)^2}{2 \lambda_k},
$$

with $x_j = \log d_j$ ($k = m$) and, separately, $x_j = \log r_j$
($k = m - 1$). With one event this is a Student-t on $\log d_1$ with
$2\alpha_0$ degrees of freedom, location $\mu_0$ and squared scale
$\beta_0 (\lambda_0 + 1) / (\alpha_0 \lambda_0)$.

Each event's **gate** is a raised-cosine ramp up at the onset and down at
the offset, with ramp duration $\tau_r$ (the stimulus's own ramp):

$$
\gamma_j(t) = \rho(t - a_j)\,\rho(a_j + d_j - t), \qquad
\rho(u) = \tfrac12\left(1 - \cos\left(\pi \min(\max(u / \tau_r, 0), 1)\right)\right).
$$

- **Departure.** The paper's ramps are sigmoids, 18 ms in all [App. A.5].
  The gate is a smooth function of $a_j$ and $d_j$, so timing can be
  fitted by gradient.

### 1.3 Trajectories

Each source has a **frequency trajectory** $\phi(t_g)$ in ERB number (the
f0 of a harmonic source, the frequency of a whistle) and a **level
trajectory** $A(t_g)$ in dB, each a mean plus a Gaussian-process deviation
on the grid [App. A.2, Eqns A.13–A.17] (`priors.TrajectoryPrior`):

$$
\phi(t_g) = \bar\phi + \delta_g, \qquad A(t_g) = \bar A + \eta_g,
$$

$$
\bar\phi \sim \mathcal{U}(3, 33.19)\ \text{Cams}, \qquad \bar A \sim \mathcal{U}(0, 120)\ \text{dB}, \qquad
\boldsymbol\delta \sim \mathcal{N}(\mathbf{0}, K_\phi), \qquad \boldsymbol\eta \sim \mathcal{N}(\mathbf{0}, K_A).
$$

The kernel of each process, on grid points $g, h$, is

$$
K_{gh} = \sigma^2 \exp\left(-\frac{(t_g - t_h)^2}{2\ell^2}\right)
+ \beta^2 \sum_{j=1}^{m} \gamma_j(t_g)\,\gamma_j(t_h)
+ \left(\epsilon^2 + 0.001\,\sigma\right) \mathbb{1}[g = h].
$$

The middle term correlates points within an event more strongly than
points in different events. With one event it is $\beta^2$ everywhere:
grid points outside the event are never heard, so they integrate out.

| Process | $\sigma$ | $\ell$ [s] | $\beta$ | $\epsilon$ |
|---|---|---|---|---|
| frequency (harmonic f0 and whistle) | 5.9 | 2.5 | 0.437 | 0.1 |
| harmonic level | 7.8 | 0.18 | 3.12 | 0.5 |
| whistle level | 1.3 | 6.9 | 0.542 | 0.1 |

$\sigma$ and $\ell$ are the medians of Table A.2; $\beta$ and $\epsilon$
are as in BASS's configuration.

- **Departure.** The paper infers $\sigma$ and $\ell$ for every sound; by
  default we fix them at the medians. With `infer_kernel` (experiments C1
  and C2), each is a latent under Table A.2's prior: $\psi =
  \operatorname{softplus}(u)$ with $u$ normal and truncated so that $\psi$
  stays in the table's bounds, parameterized by $v$ with $u = u_{\min} +
  (u_{\max} - u_{\min})\,\operatorname{sigmoid}(v)$ so that every real $v$
  is allowed (`priors.KernelParameterPrior`). The f0 lengthscale's lower
  bound is 0.01 s, not the 0.1 s printed, because the table's own quartiles
  need 0.01 ([C1](../results/C1-kernel-inferred.md)).
- **Departure.** With several events the within-event term uses the gates
  $\gamma_j$, which lie between 0 and 1, instead of a hard 0 or 1, so the
  prior moves smoothly with the timing (design C4).
- **Option, off by default.** `frequency_floor_hz` lowers the floor of
  $\bar\phi$ from 3 Cams (87.5 Hz) to $E(f_{\text{floor}})$; experiments
  P3 and P5 to P7 use 30 Hz (design P12).

### 1.4 Harmonic spectrum

A harmonic source with $K$ harmonics (given per experiment) has a spectrum
$\mathbf{s} \in \mathbb{R}^K$ in dB, harmonic $k$'s level relative to the
source's. Its prior is a Gaussian process over the harmonics' ERB numbers
[App. A.3] (`priors.spectrum_distribution`):

$$
\mathbf{s} \sim \mathcal{N}(\mathbf{0}, \Sigma), \qquad
\Sigma_{kk'} = \sigma_s^2 \exp\left(-\frac{\left(E(k \tilde f_0) - E(k' \tilde f_0)\right)^2}{2\ell_s^2}\right) + \epsilon_s^2\,\mathbb{1}[k = k'],
$$

with $\sigma_s = 11.8$ dB, $\ell_s = 4.7$ Cams (Table A.2 medians),
$\epsilon_s = 0.5$, and $\tilde f_0 = E^{-1}\left(\frac{1}{G+1}\sum_g \phi(t_g)\right)$,
the f0 at the source's mean ERB number.

- **Departure.** The paper's harmonic sources have every harmonic below
  the Nyquist frequency; ours have $K$, set by the experiment.
- **Option, off by default: sparse spectra** (design P10 (B), P11 (B),
  used by P3 and P5 to P7). Each harmonic's level gets an offset $o_k$
  whose prior says the harmonic is present (near 0 dB) or absent (far
  below), with the probability of present a latent too. The three numbers
  (1, −40 and 10 dB) are assumptions, not fits:

$$
\pi = \operatorname{sigmoid}(z), \qquad z \sim \operatorname{Logistic}(0, 1)\ \ (\text{so } \pi \sim \mathcal{U}(0, 1)),
$$

$$
o_k \overset{\text{iid}}{\sim} \pi\,\mathcal{N}(0, 1^2) + (1 - \pi)\,\mathcal{N}(-40, 10^2)\ \text{dB}.
$$

### 1.5 Rendering

The renderer maps $(H, \boldsymbol\theta)$ to a waveform, sampled at
$f_s = 20$ kHz (`sources.trajectory_event`, `scene.Scene.render`). For
each component of each source, frequency and level are interpolated
linearly from the grid to the sample times. A harmonic source's component
$k$ has

$$
f_k(t) = k\,E^{-1}(\phi(t)), \qquad L_k(t) = A(t) + s_k\ \ (+\,o_k \text{ if sparse});
$$

a whistle has one component, $f_1(t) = E^{-1}(\phi(t))$ and $L_1(t) = A(t)$.
Each event $j$ of a source contributes

$$
x_j(t) = \gamma_j(t) \sum_k \sqrt2\, r_0\, 10^{L_k(t)/20} \cos\left(2\pi \int_{a_j}^{t} f_k(u)\,du\right),
$$

with the phase integral computed by a cumulative sum over samples, so every
component starts in cosine phase at the event's onset. The sound is

$$
x(t) = \sum_{\text{sources}} \sum_{j} x_j(t).
$$

A component at steady level $L$ has RMS $r_0 10^{L/20}$, so $L$ is its
level in dB re $r_0$.

### 1.6 Likelihood

The observation is a cochleagram $\mathbf{y}$, an array of $C = 64$
channels by $M$ time windows. Given $(H, \boldsymbol\theta)$, the
renderer's sound goes through the same cochleagram, $\mathbf{c}(\boldsymbol\theta) = \mathcal{C}(x)$, and every
cell is an independent Gaussian around it [§2.2] (`cochleagram.gaussian_log_likelihood`):

$$
\log p(\mathbf{y} \mid \boldsymbol\theta, H) = \sum_{c=1}^{C} \sum_{m=1}^{M}
\log \mathcal{N}\left(y_{cm};\ c_{cm}(\boldsymbol\theta),\ \sigma_y^2\right), \qquad \sigma_y = 10\ \text{dB}.
$$

All the randomness in the observation is this noise: the renderer and the
cochleagram are deterministic and differentiable, so gradients of the log
likelihood with respect to $\boldsymbol\theta$ come from automatic
differentiation. Experiment B3 uses $\sigma_y = 5$ dB.

### 1.7 The cochleagram

**FFT cochleagram** (the default from experiment A2 on;
`cochleagram.FFTCochleagram`). BASS computes its cochleagram with Ellis's
(2009) approximation: a short-time spectrum pooled into channels shaped
like gammatone filters [App. A.5]. With a symmetric Hann window $w$ of
$W = 500$ samples (25 ms), a hop of $R = 200$ samples (10 ms) and an FFT of
$N = 1024$ points,

$$
X_m[q] = \sum_{n=0}^{W-1} w[n]\, x[mR + n]\, e^{-2\pi i q n / N}, \qquad q = 0, \dots, N/2.
$$

Channel $c$ has center frequency $f_c$, the 64 centers equally spaced in
ERB number from 20 to 9423 Hz, and the magnitude response of a fourth-order
gammatone with bandwidth parameter $b_c = 0.5 \times 1.019\,B(f_c)$ (half an
ERB), normalized to 1 at $f_c$:

$$
G_c(f) = \frac{|\Gamma_c(f)|}{|\Gamma_c(f_c)|}, \qquad
\Gamma_c(f) = \left(b_c + i(f - f_c)\right)^{-4} + \left(b_c + i(f + f_c)\right)^{-4}.
$$

The cell's value is the pooled magnitude in dB, floored at 20 dB:

$$
c_{cm} = \max\left(20 \log_{10} \frac{\frac{1}{N} \sum_q G_c(f_q)\, |X_m[q]|}{r_0},\ 20\right).
$$

The factor $1/N$ is BASS's: the channels are not calibrated, so a
sinusoid reads below its level, by about 12.8 dB in the channel at 104 Hz
and 3.3 dB in the one at 2974 Hz (default from experiment A4 on, "BASS's
gain"). With `bass_gain=False` each channel is divided instead by its
response to a sinusoid of RMS 1 at $f_c$, so that sinusoid reads its level.

**Gammatone cochleagram** (experiment A1; `cochleagram.Cochleagram`). The
same 64 gammatones applied as filters: $e_c(t)$ is the magnitude of the
analytic signal of channel $c$ (its Hilbert envelope), and with
$\tilde w$ the Hann window normalized to sum to 1,

$$
c_{cm} = \max\left(10 \log_{10} \frac{\frac12 \sum_n \tilde w[n]\, e_c^2(mR + n)}{r_0^2},\ 20\right),
$$

so a sinusoid at $f_c$ and level $L$ reads $L$.

- **Departure, deliberate.** BASS's FFT cochleagram has channels 2 to 3.6
  times wider than true half-ERB filters below 600 Hz, and uncalibrated
  gains. Each moved our thresholds toward the paper's, so both are used to
  reproduce the paper ([A2](../results/A2-fft-cochleagram.md),
  [A4](../results/A4-bass-gain.md)). Neither is a claim about the ear.
- **What it leaves out.** The cochleagram keeps no phase, and smooths
  envelope modulation above a few tens of Hz. See "Timing in the
  likelihood" under [Planned](../results/README.md#planned).

## 2. Inference for one hypothesis

For a hypothesis $H$, the log joint density is

$$
\log p(\mathbf{y}, \boldsymbol\theta \mid H) = \log p(\mathbf{y} \mid \boldsymbol\theta, H) + \log p(\boldsymbol\theta \mid H),
$$

the second term being the sum of §1.2 to §1.4's log densities
(`scene.Scene.log_joint`). Inference asks for the **evidence** (marginal
likelihood) of $H$ [App. B.1.1]:

$$
Z_H = p(\mathbf{y} \mid H) = \int p(\mathbf{y} \mid \boldsymbol\theta, H)\, p(\boldsymbol\theta \mid H)\, d\boldsymbol\theta.
$$

A hypothesis with more sources can always fit at least as well; $Z_H$
averages the fit over the prior, which builds in the Occam penalty.

### 2.1 Posterior mode

$$
\boldsymbol\theta^* = \arg\max_{\boldsymbol\theta} \log p(\mathbf{y}, \boldsymbol\theta \mid H),
$$

found by Adam with a step size per parameter in its own units (ERB number,
dB, seconds), from one or more starting points, keeping the best
(`fit.fit`, `scene.evaluate`). The starts are the stimulus's own parameters
as the paper gives them [App. C] for enumerated hypotheses, or the
detector's candidates in the search (§3.3). Milestones (a) and (b) ran 300
steps; from 2026-10-15 fits run until the objective rises by less than a
tolerance over 1000 steps.

### 2.2 Laplace estimate

With $\mathbf{Q}$ the Hessian of $-\log p(\mathbf{y}, \boldsymbol\theta \mid H)$
at $\boldsymbol\theta^*$ and $\lambda_1, \dots, \lambda_d$ its eigenvalues
(`evidence.log_evidence`),

$$
\log \hat Z_H^{\text{Laplace}} = \log p(\mathbf{y}, \boldsymbol\theta^* \mid H) + \frac{d}{2}\log 2\pi
- \frac12 \sum_{i=1}^{d} \log \max(\lambda_i, 10^{-6}).
$$

An eigenvalue raised to $10^{-6}$ means the mode is not a true peak (a flat
direction, or a saddle); the estimate is then marked unreliable and left
out of the results.

### 2.3 Importance sampling

The Laplace Gaussian $q = \mathcal{N}(\boldsymbol\theta^*, \mathbf{Q}^{-1})$
(with the same floor) is the proposal for $S = 128$ draws:

$$
\hat Z_H^{\text{IS}} = \frac1S \sum_{s=1}^{S} w_s, \qquad
w_s = \frac{p(\mathbf{y}, \boldsymbol\theta_s \mid H)}{q(\boldsymbol\theta_s)}, \qquad
\boldsymbol\theta_s \sim q,
$$

unbiased for $Z_H$. The effective sample size
$\left(\sum_s w_s\right)^2 / \sum_s w_s^2$ says how well $q$ covers the
posterior.

### 2.4 Variational estimate

From experiment A7 on, as the paper does [App. B.3], a Gaussian is fitted
to the posterior by maximizing the evidence lower bound, then used as the
importance-sampling proposal (`evidence.variational_evidence`). The
coordinates are whitened by the Hessian at the mode: with
$\mathbf{Q} = V \operatorname{diag}(\lambda) V^\top$ and
$W = V \operatorname{diag}\left(\max(|\lambda_i|, 10^{-2})^{-1/2}\right)$,

$$
\boldsymbol\theta = \boldsymbol\theta^* + W\mathbf{z}, \qquad
q(\mathbf{z}) = \mathcal{N}\left(\boldsymbol\mu, \operatorname{diag}(e^{2\boldsymbol\omega})\right),
$$

$$
\mathcal{L}(\boldsymbol\mu, \boldsymbol\omega) = \mathbb{E}_{q}\left[\log p(\mathbf{y}, \boldsymbol\theta^* + W\mathbf{z} \mid H)\right]
+ \mathbb{H}[q] + \log|\det W| \ \le\ \log Z_H .
$$

$\mathcal{L}$ is maximized by Adam (2000 steps, rate 0.01, 10
reparameterized draws per step; draws outside the prior's support are left
out of a step). The reported ELBO is the mean of the last tenth of the
steps; the importance estimate uses $q$ with $S = 256$ draws.

- **Departure.** The paper ran 250 steps of variational inference for each
  hypothesis in its search and longer fits for the best ones [App. B.2–B.3],
  starting from its proposals. We start from the posterior mode and whiten
  by its Hessian. A7 found the same thresholds by Laplace and by this.

## 3. From evidence to what the model hears

### 3.1 Posterior over hypotheses

$$
\log p(H \mid \mathbf{y}) = \log Z_H + \log p(H) - \log p(\mathbf{y}),
$$

so for a set of hypotheses, $p(H_i \mid \mathbf{y}) \propto Z_{H_i}\, p(H_i)$,
normalized over the set. Two hypotheses are compared by their **log
posterior odds**:

$$
\log O_{21} = \left(\log Z_{H_2} + \log p(H_2)\right) - \left(\log Z_{H_1} + \log p(H_1)\right).
$$

$\log Z_H$ is whichever estimate the page says: Laplace, importance from
Laplace, or importance from the variational fit (the ELBO where
importance sampling fails).

### 3.2 Thresholds

When hypothesis $H_0$ (say, one source) explains low values of a stimulus
parameter $v$ (say, mistuning) and $H_1$ high values, the paper defines the
threshold $\tau$ by Eqn 2. Read as integrals over $[v_{\min}, v_{\max}]$,
with the posterior interpolated linearly between the values run, it is the
area under the $H_0$ curve (`thresholds.threshold`):

$$
\tau = v_{\min} + \int_{v_{\min}}^{v_{\max}} p(H_0 \mid \mathbf{y}(v))\, dv.
$$

When $H_0$ wins everywhere, $\tau = v_{\max}$, which the paper reports as
not measurable. Some pages also give the rule in BASS's analysis code,
computed outside this repository because BASS has no license
([A6](../results/A6-bass-own-code.md)).

For Shepard pairs ([P3](../results/P3-shepard-pairs.md)), P(up) is the
posterior mass of the hypotheses whose fitted f0 (or whistle frequency)
rises from the first tone to the second.

### 3.3 Search over structures

Enumeration needs the hypotheses written by hand. The search
(`search.search`, design C9 and C10, experiment
[C3](../results/C3-search.md)) builds them from the sound instead, after
the paper's sequential inference [App. B.2]:

1. A detector (`detect.detect`) finds onset groups in the cochleagram and,
   for each, candidate explanations: whistles at spectral peaks, and
   harmonic sources at f0s found by subharmonic summation.
2. Onset groups are taken in time order, one per round. Each kept
   hypothesis is extended by each explanation of the group (each event
   joining a compatible existing source or starting a new one), or by
   leaving the group out, within the paper's limits on sources and events
   [App. B.2.2].
3. Every extension is fitted (§2.1) to the sound up to the next group and
   ranked by $\log \hat Z^{\text{Laplace}} + \log p(H)$; the best few go on.
4. The detector is run again on what the best hypothesis leaves
   unexplained, and cleanup moves (remove, merge, change kind) are tried
   [App. B.2.4].
5. The best by the screen get the variational estimate (§2.4), which ranks
   them.

- **Departure.** The paper's proposals come from a trained segmentation
  network and its screening fits are variational; ours come from a
  hand-built detector and screening uses MAP and Laplace (design C9 (A),
  C10 (B)).

### 3.4 Online inference

For streaming and buildup ([O1](../results/O1-online-buildup.md),
[O2](../results/O2-online-speed.md); design O1 to O10), the sound arrives in
windows $\mathbf{y}_1, \mathbf{y}_2, \dots$ ending at times $t_1, t_2,
\dots$ (`online.run_online`). The evidence of the sound so far is the
product of predictive likelihoods,

$$
\log p(\mathbf{y}_{1:k} \mid H) = \sum_{j=1}^{k} \log p(\mathbf{y}_j \mid \mathbf{y}_{1:j-1}, H),
$$

each term estimated as a difference of two Laplace estimates,
$\log \hat Z_j(F_j) - \log \hat Z_{j-1}(F_j')$, where $F_j$ is the set of
latents still free at window $j$ and $F_j'$ is $F_j$ without the latents
that window added. After each window a few Adam steps move only the free
latents.

- **Fixed lag** (O3 (B)). Grid deviations older than a lag $L$, and the
  timing of events that ended before $t_k - L$, are frozen at their values.
  Their prior terms stay in the log joint, so the free latents are
  conditioned on them exactly. Trajectory means and kernel latents are never
  frozen. With $L = \infty$ the terms telescope to the batch Laplace
  estimate of the whole prefix. A latent is frozen only after it has been
  free in one window term, so a source born more than $L$ after the start
  (or a grid point inside a long silence) is integrated once before its past
  freezes.
- **Leaky decision variable** (O10). With $\Delta_j^{(a)}$ and
  $\Delta_j^{(b)}$ two hypotheses' window terms,
  $D_k = \sum_{j \le k} e^{-(t_k - t_j)/\tau_D} \left(\Delta_j^{(a)} - \Delta_j^{(b)}\right)$;
  $\tau_D = \infty$ is the full log odds without the structure prior.
- **Summarizing the past** (S4, default on). The likelihood of time
  windows that only frozen and whole-sound latents reach is replaced by
  its second-order Taylor expansion in the whole-sound latents
  (assumed-density filtering), so each step renders only the recent sound.

## 4. The source-and-room model

Experiments [R1](../results/R1-source-and-room.md) and
[R2](../results/R2-atypical-rooms.md) ask whether a room's reverberation
time can be told apart from the source from one sound (design R1 to R8).
The model is separate from §1 to §3.

### 4.1 Latents and priors

- **The room**: its median RT60 $r$ [s], with a gamma prior fitted to
  Traer & McDermott's (2016) survey of everyday spaces
  (`room.log_prior_rt60`), shape $k = 1.9668$ and scale $\vartheta = 0.2203$ s:
  $p(r) = r^{k-1} e^{-r/\vartheta} / \left(\Gamma(k)\, \vartheta^k\right)$.
  The direct-to-reverberant ratio is fixed at the survey's median, 9.9 dB.
- **The source's statistics** $\boldsymbol\psi = (\kappa_b, \kappa_t, s)$:
  correlation lengths across bands [Cams] and time [s], and level SD [dB];
  each log-uniform on $[\psi_0/\sqrt{10}, \psi_0\sqrt{10}]$ around
  $\psi_0 = (8.78, 0.154)$ for the two lengths, and on $[3, 30]$ dB for $s$.
- **The source**: a grid of levels $\mathbf{l}$ in dB, 39 bands by 41 time
  steps of 10 ms, Gaussian with separable exponential correlation
  (McDermott, Wrobleski & Oxenham, 2011; `spectrotemporal.SpectrogramPrior`):

$$
\mathbf{l} \sim \mathcal{N}\left(\boldsymbol\mu,\ s^2\, R_b \otimes R_t\right), \qquad
(R_b)_{bb'} = \rho_b^{|b - b'|}, \quad (R_t)_{kk'} = \rho_t^{|k - k'|},
$$

$$
\rho_b = e^{-\Delta_E / \kappa_b}, \qquad \rho_t = e^{-\Delta / \kappa_t},
$$

  with $\Delta_E$ the band spacing in ERB number and $\mu_b$ rising with
  each band's width in Hz so the long-term spectrum is flat on average.
  Both factors are first-order autoregressions, so the density needs no
  dense covariance matrix.

### 4.2 Likelihood

The representation is the energy of each of the 39 half-cosine ERB bands
(20 to 4000 Hz) in 10 ms blocks (`blockpower.BlockPower`). The model
predicts its *expected* value: the source's expected block energies
$S_b(j)$, from $\mathbf{l}$, convolved band by band with the room's
expected gain $g_b(k; r)$, averaged over the room's random noise
(`room.RoomGain`):

$$
\mathbb{E}[e_b(k)] = \sum_j S_b(j)\, g_b(k - j; r).
$$

With $Y_{bk}$ the observed block energy in dB, $P_{bk} = 10\log_{10}\mathbb{E}[e_b(k)] + \beta_e$,
a floor $F = \max_{b,k} Y_{bk} - 60$ dB, and $\beta_e = -1.3$ dB,
$\sigma_e = 3.6$ dB measured against renders (`tools/source_room/block_power_error.py`),

$$
\log p(\mathbf{Y} \mid \mathbf{l}, r) = \sum_{b,k} \log \mathcal{N}\left(\max(Y_{bk}, F);\ \max(P_{bk}, F),\ \sigma_e^2\right).
$$

### 4.3 Inference

(`sourceroom.infer`.) At fixed $(r, \boldsymbol\psi)$ the 1,599 levels are
integrated out by Laplace, at the mode found by Newton's method with the
exact Hessian $P$:

$$
\log p(\mathbf{Y} \mid r, \boldsymbol\psi) \approx \log p(\mathbf{Y} \mid \mathbf{l}^*, r)
+ \log \mathcal{N}(\mathbf{l}^*; \boldsymbol\mu, \Sigma_{\boldsymbol\psi}) + \frac{D}{2}\log 2\pi - \frac12 \log\det P .
$$

$\boldsymbol\psi$ is maximized and integrated out by Laplace in log
coordinates, and $r$ is placed on a grid of 15 points from 0.1 to 2 s,
refined where the posterior has mass, and integrated on those points to
give $p(r \mid \mathbf{Y})$.

## References

- Cusimano, M., Hewitt, L. B., & McDermott, J. H. (2024). Listening with
  generative models. *Cognition*, 253, 105874.
- Ellis, D. P. W. (2009). Gammatone-like spectrograms. Web resource.
- Glasberg, B. R., & Moore, B. C. J. (1990). Derivation of auditory filter
  shapes from notched-noise data. *Hearing Research*, 47, 103–138.
- McDermott, J. H., Wrobleski, D., & Oxenham, A. J. (2011). Recovering
  sound sources from embedded repetition. *PNAS*, 108, 1188–1193.
- Särkkä, S. (2013). *Bayesian Filtering and Smoothing*. Cambridge
  University Press.
- Traer, J., & McDermott, J. H. (2016). Statistics of natural reverberation
  enable perceptual separation of sound and space. *PNAS*, 113,
  E7856–E7865.
