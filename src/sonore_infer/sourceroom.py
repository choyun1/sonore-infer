"""Posterior over a room's RT60 and a random spectrogram's statistics from
one reverberant sound (docs/design/source-reverb.md, R5).

The unknowns are the room's RT60 ``r``, the source prior's hyperparameters
``theta`` (band and time correlation lengths and level sd, each
log-uniform over a box) and the 39 x 41 grid of source cell levels ``l``.
The cell levels are integrated out by Laplace: at fixed ``(r, theta)`` the
posterior mode of ``l`` is found by Newton's method with the exact Hessian
(by L-BFGS in coordinates whitened by the prior when there is no nearby
mode to start from), and

    log p(y | r, theta) ~ log p(y | l*) + log N(l*; mu, Sigma_theta)
                          + d/2 log 2 pi - 1/2 log det P,

with ``P`` the Hessian of the negative log joint density at the mode. This
is what :func:`sonore_infer.evidence.log_evidence` computes, but its
autograd Hessian takes about a minute at this size, so the Hessian here is
exact and analytic (:meth:`BlockPower.source_energy_jacobian` and
``source_energy_curvature``) and takes under a second; a test checks the two
agree.

The posterior is sharp in every one of ``r, theta`` (1,599 cells): on one
sound at RT60 0.4 s its sd is about 0.015 in log r, 0.12 in log sd and 0.2
in log correlation length, against 0.58 between R5's 5 grid points per
hyperparameter and 0.21 between its 15 RT60s, so a fixed grid would put
all the mass on one point. Instead:

- At each ``r``, the log likelihood is expanded to second order in ``l``
  around the mode, which makes ``log p(y | r, theta)`` a closed-form
  Gaussian integral for every ``theta`` (:meth:`Expansion.log_evidence`;
  equal to Laplace at the expansion's own ``theta``). It is maximized over
  ``theta``, the mode is refitted there, and ``theta`` is integrated out by
  Laplace in log coordinates.
- ``r`` is placed on R5's grid (15 points, 0.1 to 2 s), then refined
  around the best point until the posterior's bulk is covered by points a
  fraction of its width apart, and integrated on those points.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from .blockpower import BIAS_DB, SIGMA_DB, BlockPower
from .room import TYPICAL_DRR_DB, RoomGain, log_prior_rt60
from .spectrotemporal import SpectrogramPrior

DB = 10 / math.log(10)  # d(dB)/d(ln energy)

# Log-uniform prior box of theta = (band correlation [ERB], time correlation
# [s], sd [dB]): a factor of 10 around sonore's correlation lengths, and
# 3-30 dB for the sd (design R1).
THETA_DEFAULT = (8.78, 0.154, 14.1)
THETA_LOW = (8.78 / math.sqrt(10), 0.154 / math.sqrt(10), 3.0)
THETA_HIGH = (8.78 * math.sqrt(10), 0.154 * math.sqrt(10), 30.0)
LOG_PRIOR_THETA = -sum(math.log(math.log(h / lo)) for lo, h in zip(THETA_LOW, THETA_HIGH, strict=True))
RT60_GRID = tuple(np.geomspace(0.1, 2.0, 15))


def ar1_cholesky(rho: torch.Tensor, size: int) -> torch.Tensor:
    """Cholesky factor of the correlation ``rho^|i - j|``, ``size`` by ``size``."""
    index = torch.arange(size, dtype=torch.float64)
    lag = index[:, None] - index[None, :]
    first = rho ** lag.clamp_min(0)
    rest = torch.sqrt(1 - rho**2) * rho ** lag.clamp_min(0)
    factor = torch.where(lag >= 0, torch.where(index[None, :] == 0, first, rest), torch.zeros_like(lag))
    return factor


def ar1_precision(rho: torch.Tensor, size: int) -> torch.Tensor:
    """Inverse of the correlation ``rho^|i - j|``: tridiagonal."""
    end = torch.ones(1, dtype=torch.float64)
    diagonal = torch.cat([end, (1 + rho**2) * torch.ones(size - 2, dtype=torch.float64), end])
    off_diagonal = -rho * torch.ones(size - 1, dtype=torch.float64)
    return (torch.diag(diagonal) + torch.diag(off_diagonal, 1) + torch.diag(off_diagonal, -1)) / (1 - rho**2)


@dataclass
class Model:
    """The observer: :class:`BlockPower` for the source, :class:`RoomGain` at
    the typical DRR for the room, and the block-power likelihood."""

    observed: torch.Tensor
    power: BlockPower = field(default_factory=BlockPower)
    sigma_db: float = SIGMA_DB
    floor_db: float = 60.0
    bias_db: float = BIAS_DB
    drr_db: float = TYPICAL_DRR_DB

    def __post_init__(self) -> None:
        self.observed_db = DB * torch.log(self.observed.to(torch.float64).clamp_min(1e-300))
        self.floor = self.observed_db.max() - self.floor_db
        self.target = torch.maximum(self.observed_db, self.floor)
        self.mean = torch.from_numpy(SpectrogramPrior().mean_db()).reshape(-1, 1)
        self.room = RoomGain(fs=self.power.fs, n_blocks=self.power.n_blocks, drr_db=self.drr_db)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the level grid: ``(n_bands, n_windows)``."""
        return self.power.n_bands, self.power.n_windows

    def gain(self, rt60: float) -> torch.Tensor:
        """The room's expected gain at median RT60 ``rt60`` [s], see :meth:`RoomGain.gain`."""
        return self.room.gain(rt60)

    def log_likelihood(self, levels: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        """Same as :func:`blockpower.log_likelihood` with this model's settings."""
        expected = self.power.reverberant(self.power.source_energy(levels), gain)
        predicted = DB * torch.log(expected.clamp_min(1e-300)) + self.bias_db
        residual = self.target - torch.maximum(predicted, self.floor)
        n_cells = residual.numel()
        return -0.5 * (residual**2).sum() / self.sigma_db**2 - n_cells * (
            math.log(self.sigma_db) + 0.5 * math.log(2 * math.pi)
        )

    def likelihood_derivatives(
        self, levels: torch.Tensor, gain: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Log likelihood, its gradient and the Hessian of its negative with
        respect to the flattened levels, all exact."""
        power = self.power
        source = power.source_energy(levels).detach()
        expected = power.reverberant(source, gain)
        predicted = DB * torch.log(expected.clamp_min(1e-300)) + self.bias_db
        active = predicted > self.floor
        residual = self.target - torch.maximum(predicted, self.floor)
        value = -0.5 * (residual**2).sum() / self.sigma_db**2 - residual.numel() * (
            math.log(self.sigma_db) + 0.5 * math.log(2 * math.pi)
        )
        variance = self.sigma_db**2
        # loss = sum residual^2 / (2 variance); derivatives in the expected energy E
        first = torch.where(active, -residual * DB / (variance * expected), 0.0)
        second = torch.where(active, (DB**2 + residual * DB) / (variance * expected**2), 0.0)
        # gradient through the room's convolution: its transpose applied to `first`
        source_leaf = source.clone().requires_grad_(True)
        (back,) = torch.autograd.grad((power.reverberant(source_leaf, gain) * first).sum(), source_leaf)
        source_jacobian = power.source_energy_jacobian(levels)  # (bands, blocks, cells)
        jacobian = power.reverberant(source_jacobian.permute(2, 0, 1), gain).permute(1, 2, 0)
        jacobian = jacobian.reshape(-1, jacobian.shape[-1])
        keep = active.reshape(-1)
        rows = jacobian[keep]
        gradient = -(rows.T @ first.reshape(-1)[keep])
        hessian = rows.T @ (second.reshape(-1)[keep, None] * rows)
        hessian = hessian + power.source_energy_curvature(levels, back)
        return value, gradient, 0.5 * (hessian + hessian.T)


@dataclass
class Prior:
    """The source prior at log theta, in the forms the fit and evidence need."""

    log_theta: torch.Tensor
    n_bands: int
    n_windows: int

    def __post_init__(self) -> None:
        base = SpectrogramPrior()
        band_correlation, time_correlation, sd = torch.exp(self.log_theta)
        self.rho_band = torch.exp(-base.band_spacing_erb / band_correlation)
        self.rho_time = torch.exp(-base.step / time_correlation)
        self.sd = sd

    def color(self, whitened: torch.Tensor) -> torch.Tensor:
        """Deviations [dB] from whitened coordinates ``(n_bands, n_windows)``."""
        band_factor = ar1_cholesky(self.rho_band, self.n_bands)
        time_factor = ar1_cholesky(self.rho_time, self.n_windows)
        return self.sd * band_factor @ whitened @ time_factor.T

    def precision(self) -> torch.Tensor:
        """Inverse covariance of the flattened (band-major) levels."""
        band_precision = ar1_precision(self.rho_band, self.n_bands)
        time_precision = ar1_precision(self.rho_time, self.n_windows)
        return torch.kron(band_precision, time_precision) / self.sd**2

    def log_det_covariance(self) -> torch.Tensor:
        """Log determinant of the covariance of the flattened levels."""
        n_bands, n_windows = self.n_bands, self.n_windows
        return (
            2 * n_bands * n_windows * torch.log(self.sd)
            + n_windows * (n_bands - 1) * torch.log(1 - self.rho_band**2)
            + n_bands * (n_windows - 1) * torch.log(1 - self.rho_time**2)
        )


def initial_levels(model: Model, gain: torch.Tensor, limit_db: float = 60.0) -> torch.Tensor:
    """Levels to start a fit from: the prior mean, raised (by up to
    ``limit_db``) in cells whose block is observed above the floor, by the
    observed over the predicted energy at the prior mean. A cell predicted
    below the floor where the observation is above it gets no gradient (the
    floor is flat), so from the prior mean alone a fit can stay silent where
    a sound is loud; lowering the other cells only costs prior density."""
    n_bands, n_windows = model.shape
    mean = model.mean.expand(n_bands, n_windows)
    expected = model.power.reverberant(model.power.source_energy(mean), gain)[:, :n_windows]
    shift = DB * torch.log(model.observed[:, :n_windows].clamp_min(1e-300) / expected.clamp_min(1e-300))
    audible = model.observed_db[:, :n_windows] > model.floor
    return mean + torch.where(audible, shift.clamp(0.0, limit_db), 0.0)


def fit_mode(
    model: Model,
    gain: torch.Tensor,
    prior: Prior,
    start: torch.Tensor | None = None,
) -> torch.Tensor:
    """Posterior mode of the levels at one (r, theta), by L-BFGS on the
    whitened coordinates; ``start`` is a level grid to start from (by default
    :func:`initial_levels`)."""
    n_bands, n_windows = model.shape
    band_factor = ar1_cholesky(prior.rho_band.detach(), n_bands)
    time_factor = ar1_cholesky(prior.rho_time.detach(), n_windows)
    sd = prior.sd.detach()
    if start is None:
        start = initial_levels(model, gain)
    deviation = (start - model.mean) / sd
    start_whitened = torch.linalg.solve_triangular(band_factor, deviation, upper=False)
    start_whitened = torch.linalg.solve_triangular(time_factor, start_whitened.T, upper=False).T
    whitened = start_whitened.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [whitened],
        lr=1,
        max_iter=2000,
        history_size=20,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-4,
        tolerance_change=1e-12,
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        levels = model.mean + sd * band_factor @ whitened @ time_factor.T
        loss = -model.log_likelihood(levels, gain) + 0.5 * (whitened**2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return (model.mean + sd * band_factor @ whitened.detach() @ time_factor.T).detach()


@dataclass
class Expansion:
    """The log likelihood to second order in the levels around ``levels``
    (at one RT60): value, gradient and the Hessian of its negative."""

    levels: torch.Tensor
    value: float
    gradient: torch.Tensor
    hessian: torch.Tensor
    mean: torch.Tensor

    @classmethod
    def at(cls, model: Model, gain: torch.Tensor, levels: torch.Tensor) -> Expansion:
        """The expansion of ``model``'s log likelihood at ``levels``, with room gain ``gain``."""
        value, gradient, hessian = model.likelihood_derivatives(levels, gain)
        return cls(levels, float(value), gradient, hessian, model.mean.expand_as(levels).reshape(-1))

    def log_evidence(self, log_theta: torch.Tensor) -> torch.Tensor:
        """``log integral exp(quadratic log likelihood) N(l; mu, Sigma_theta) dl``,
        differentiable in ``log_theta``. At the theta whose mode is
        ``levels`` this is the Laplace approximation. ``-inf`` where the
        quadratic is not concave enough to integrate."""
        n_bands, n_windows = self.levels.shape
        prior = Prior(log_theta, n_bands, n_windows)
        precision = prior.precision()
        offset = self.mean - self.levels.reshape(-1)  # prior mean minus expansion point
        linear = self.gradient + precision @ offset
        posterior = self.hessian + precision
        factor, info = torch.linalg.cholesky_ex(posterior)
        if info.item() != 0:
            return torch.tensor(-math.inf, dtype=torch.float64)
        solved = torch.cholesky_solve(linear[:, None], factor)[:, 0]
        return (
            self.value
            - 0.5 * offset @ precision @ offset
            + 0.5 * linear @ solved
            - 0.5 * prior.log_det_covariance()
            - torch.log(torch.diagonal(factor)).sum()
        )

    def value_and_gradient(self, log_theta: np.ndarray) -> tuple[float, np.ndarray]:
        """:meth:`log_evidence` and its gradient in ``log_theta``, the gradient
        from the derivatives of the prior precision ``Q`` (autograd through
        the Cholesky factor is several times slower):
        ``dG = -1/2 (m - v)' dQ (m - v) - 1/2 tr(P^-1 dQ) - 1/2 d log det Sigma``,
        with ``m`` the prior mean minus the expansion point and ``v = P^-1 h``."""
        n_bands, n_windows = self.levels.shape
        log_theta_tensor = torch.tensor(log_theta, dtype=torch.float64)

        def factors(log_theta_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            prior = Prior(log_theta_tensor, n_bands, n_windows)
            return ar1_precision(prior.rho_band, n_bands), ar1_precision(prior.rho_time, n_windows), prior.sd

        band_precision, time_precision, sd = factors(log_theta_tensor)
        band_derivative, time_derivative, _ = torch.func.jacfwd(factors)(log_theta_tensor)
        precision = torch.kron(band_precision, time_precision) / sd**2
        offset = self.mean - self.levels.reshape(-1)
        linear = self.gradient + precision @ offset
        factor, info = torch.linalg.cholesky_ex(self.hessian + precision)
        if info.item() != 0:
            return -math.inf, np.zeros(3)
        solved = torch.cholesky_solve(linear[:, None], factor)[:, 0]
        prior = Prior(log_theta_tensor, n_bands, n_windows)
        value = (
            self.value
            - 0.5 * offset @ precision @ offset
            + 0.5 * linear @ solved
            - 0.5 * prior.log_det_covariance()
            - torch.log(torch.diagonal(factor)).sum()
        )
        inverse = torch.cholesky_inverse(factor)
        offset_minus_solved = offset - solved  # m - v in the formula above
        d_log_det = torch.func.jacfwd(
            lambda log_theta_tensor: Prior(log_theta_tensor, n_bands, n_windows).log_det_covariance()
        )(log_theta_tensor)
        # dQ for each of the three log theta: band correlation, time correlation, sd
        derivatives = (
            torch.kron(band_derivative[:, :, 0], time_precision) / sd**2,
            torch.kron(band_precision, time_derivative[:, :, 1]) / sd**2,
            -2 * precision,
        )
        gradient = [
            (
                -0.5 * offset_minus_solved @ precision_derivative @ offset_minus_solved
                - 0.5 * (inverse * precision_derivative).sum()
                - 0.5 * d_log_det[index]
            ).item()
            for index, precision_derivative in enumerate(derivatives)
        ]
        return value.item(), np.array(gradient)


def newton_mode(
    model: Model,
    gain: torch.Tensor,
    prior: Prior,
    start: torch.Tensor,
    max_steps: int = 20,
    tolerance: float = 1e-3,
) -> tuple[torch.Tensor, Expansion] | None:
    """Posterior mode of the levels by Newton's method with the exact
    Hessian and a backtracking line search, from ``start``, and the
    expansion there. Stops when the expected gain of a full step is below
    ``tolerance`` nats; ``None`` where the Hessian is not positive definite
    or it does not converge (the caller falls back to :func:`fit_mode`)."""
    shape = model.shape
    precision = prior.precision().detach()
    mean = model.mean.expand(shape).reshape(-1)

    def log_joint(levels: torch.Tensor, value: torch.Tensor | float) -> torch.Tensor:
        return value - 0.5 * (levels - mean) @ precision @ (levels - mean)

    levels = start.reshape(-1).clone()
    for _ in range(max_steps):
        expansion = Expansion.at(model, gain, levels.reshape(shape))
        gradient = expansion.gradient - precision @ (levels - mean)
        factor, info = torch.linalg.cholesky_ex(expansion.hessian + precision)
        if info.item() != 0:
            return None
        step = torch.cholesky_solve(gradient[:, None], factor)[:, 0]
        decrement = (gradient @ step).item()
        if decrement < 2 * tolerance:
            return levels.reshape(shape), expansion
        current = log_joint(levels, expansion.value)
        size = 1.0
        while size > 1e-3:
            candidate = levels + size * step
            value = model.log_likelihood(candidate.reshape(shape), gain).detach()
            if log_joint(candidate, value) >= current + 0.25 * size * decrement:
                break
            size /= 2
        else:
            return None
        levels = candidate
    return None


def _bounds() -> list[tuple[float, float]]:
    return [(math.log(lo), math.log(hi)) for lo, hi in zip(THETA_LOW, THETA_HIGH, strict=True)]


def _theta_hessian(expansion: Expansion, log_theta: np.ndarray, step: float = 0.01) -> np.ndarray:
    """Hessian of the expansion's log evidence in log theta, by central
    differences of its exact gradient."""
    hessian = np.zeros((3, 3))
    for index in range(3):
        shift = np.zeros(3)
        shift[index] = step
        gradient_up = expansion.value_and_gradient(log_theta + shift)[1]
        gradient_down = expansion.value_and_gradient(log_theta - shift)[1]
        hessian[index] = (gradient_up - gradient_down) / (2 * step)
    return 0.5 * (hessian + hessian.T)


def _maximize_theta(expansion: Expansion, start: np.ndarray, step: float = 0.25) -> np.ndarray:
    """Maximize the expansion's evidence over log theta by Newton's method
    with a backtracking line search, each move within ``step`` of ``start``
    and inside the prior box. The expansion is only good near its own theta,
    and far from it ``P`` can stop being positive definite (the evidence is
    then ``-inf``, and the line search backs off); the caller refits and
    calls again until theta settles."""
    low, high = (np.array(bounds) for bounds in zip(*_bounds(), strict=True))
    low, high = np.maximum(low, start - step), np.minimum(high, start + step)
    log_theta = np.asarray(start, float)
    value, gradient = expansion.value_and_gradient(log_theta)
    for _ in range(20):
        eigenvalues, vectors = np.linalg.eigh(-_theta_hessian(expansion, log_theta))
        eigenvalues = np.maximum(eigenvalues, max(1e-3 * eigenvalues.max(), 1e-6))
        direction = vectors @ ((vectors.T @ gradient) / eigenvalues)
        size = 1.0
        while size > 1e-3:
            candidate = np.clip(log_theta + size * direction, low, high)
            new_value, new_gradient = expansion.value_and_gradient(candidate)
            if new_value > value:
                break
            size /= 2
        else:
            break
        moved = np.abs(candidate - log_theta).max()
        log_theta, value, gradient = candidate, new_value, new_gradient
        if moved < 1e-3:
            break
    return log_theta


@dataclass
class RT60Point:
    """Results at one RT60, theta integrated out."""

    rt60: float
    log_evidence: float  # log p(y | r), theta integrated by Laplace (prior box included)
    laplace_at_mode: float  # log p(y | r, theta_hat), Laplace over levels
    log_theta: np.ndarray  # theta_hat, log coordinates
    log_theta_cov: np.ndarray  # Laplace covariance in log coordinates
    levels: torch.Tensor  # level mode at theta_hat
    refits: int


def evaluate_rt60(
    model: Model,
    rt60: float,
    log_theta: np.ndarray,
    levels: torch.Tensor | None = None,
    max_refits: int = 8,
    tolerance: float = 0.05,
) -> RT60Point:
    """Fit the levels at (r, theta), maximize the expansion's evidence over
    theta, refit there, and repeat until theta moves less than
    ``tolerance`` (in log units); then Laplace over log theta. ``levels``
    is the returned point's last mode, at a theta within ``tolerance``."""
    gain = model.gain(rt60)
    n_bands, n_windows = model.shape
    current = np.asarray(log_theta, float)
    refits = 0
    while refits < max_refits:
        refits += 1
        prior = Prior(torch.tensor(current, dtype=torch.float64), n_bands, n_windows)
        found = None if levels is None else newton_mode(model, gain, prior, levels)
        if found is None:
            fitted = fit_mode(model, gain, prior, levels)
            found = newton_mode(model, gain, prior, fitted)
            if found is None:
                found = fitted, Expansion.at(model, gain, fitted)
        levels, expansion = found
        new_log_theta = _maximize_theta(expansion, current)
        moved = np.abs(new_log_theta - current).max()
        current = new_log_theta
        if moved < tolerance:
            break
    if levels is None:
        raise ValueError("max_refits must be at least 1")
    # theta has moved less than `tolerance` from the expansion's own theta,
    # where the expansion is Laplace; it is not refitted again
    peak = expansion.log_evidence(torch.tensor(current, dtype=torch.float64)).item()
    curvature = -_theta_hessian(expansion, current)
    eigenvalues, vectors = np.linalg.eigh(curvature)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    covariance = (vectors / eigenvalues) @ vectors.T
    log_evidence = peak + LOG_PRIOR_THETA + 1.5 * math.log(2 * math.pi) - 0.5 * np.log(eigenvalues).sum()
    return RT60Point(rt60, log_evidence, peak, current, covariance, levels, refits)


@dataclass
class Posterior:
    """The posterior over RT60 from the evidence at a set of RT60 points, and the
    Laplace Gaussians in log theta at each."""

    points: list[RT60Point]

    def _weights(self) -> tuple[list[RT60Point], np.ndarray, float]:
        """Posterior mass of each point (trapezoid in log r) and the log
        marginal likelihood log p(y)."""
        points = sorted(self.points, key=lambda point: point.rt60)
        log_r = np.log([point.rt60 for point in points])
        log_density = np.array(
            [
                point.log_evidence + log_prior_rt60(point.rt60).item() + math.log(point.rt60)
                for point in points
            ]
        )
        if len(points) == 1:
            return points, np.ones(1), float(log_density[0])
        widths = np.zeros(len(points))
        widths[:-1] += np.diff(log_r) / 2
        widths[1:] += np.diff(log_r) / 2
        log_mass = log_density + np.log(widths)
        total = np.logaddexp.reduce(log_mass)
        return points, np.exp(log_mass - total), float(total)

    @property
    def log_marginal(self) -> float:
        """The log marginal likelihood ``log p(y)``."""
        return self._weights()[2]

    def rt60_quantiles(self, qs: Sequence[float] = (0.05, 0.5, 0.95)) -> np.ndarray:
        """Quantiles of r, linear in log r between points."""
        points, weights, _ = self._weights()
        log_r = np.log([point.rt60 for point in points])
        cdf = np.cumsum(weights) - weights / 2
        return np.exp(np.interp(qs, cdf, log_r))

    def theta_quantiles(self, qs: Sequence[float] = (0.05, 0.5, 0.95)) -> np.ndarray:
        """Quantiles ``(3, len(qs))`` of each theta, from the mixture over r
        of the Laplace Gaussians in log theta."""
        from scipy.stats import norm

        points, weights, _ = self._weights()
        keep = weights > 1e-6
        result = np.zeros((3, len(qs)))
        for index in range(3):
            means = np.array([point.log_theta[index] for point in points])[keep]
            sds = np.sqrt([point.log_theta_cov[index, index] for point in points])[keep]
            grid = np.linspace((means - 6 * sds).min(), (means + 6 * sds).max(), 4001)
            cdf = (weights[keep][:, None] * norm.cdf((grid[None, :] - means[:, None]) / sds[:, None])).sum(0)
            cdf /= cdf[-1]
            result[index] = np.exp(np.interp(qs, cdf, grid))
        return result


def importance_estimate(
    model: Model, point: RT60Point, n_samples: int = 128, generator: torch.Generator | None = None
) -> tuple[float, float]:
    """Importance-sampling estimate of ``log p(y | r, theta_hat)`` at one
    point, with the Laplace Gaussian over the levels as the proposal (as
    :func:`evidence.log_evidence` does), and its effective sample size."""
    n_bands, n_windows = model.shape
    gain = model.gain(point.rt60)
    prior = Prior(torch.tensor(point.log_theta, dtype=torch.float64), n_bands, n_windows)
    precision = prior.precision()
    expansion = Expansion.at(model, gain, point.levels)
    factor = torch.linalg.cholesky(expansion.hessian + precision)
    mode = point.levels.reshape(-1)
    noise = torch.randn(n_samples, mode.numel(), generator=generator, dtype=torch.float64)
    # draws = mode + P^(-1/2) noise, with P = factor factor'
    draws = mode + torch.linalg.solve_triangular(factor.T, noise.T, upper=True).T
    half_log_2pi = 0.5 * mode.numel() * math.log(2 * math.pi)
    log_proposal = -0.5 * (noise**2).sum(1) + torch.log(torch.diagonal(factor)).sum() - half_log_2pi
    mean = expansion.mean
    log_prior = (
        -0.5 * torch.einsum("si,ij,sj->s", draws - mean, precision, draws - mean)
        - 0.5 * prior.log_det_covariance()
        - half_log_2pi
    )
    with torch.no_grad():
        log_likelihood = torch.stack(
            [model.log_likelihood(draw.reshape(n_bands, n_windows), gain) for draw in draws]
        )
    log_weights = log_likelihood + log_prior - log_proposal
    estimate = (torch.logsumexp(log_weights, 0) - math.log(n_samples)).item()
    effective = (1 / (torch.softmax(log_weights, 0) ** 2).sum()).item()
    return estimate, effective


def infer(model: Model, *, refine_to: float = 0.5, max_points: int = 36, verbose: bool = False) -> Posterior:
    """Posterior over RT60 and theta for ``model``'s observation.

    R5's coarse grid first, each point warm-started from its neighbor; then
    points are added in log r where the posterior mass is, until neighboring
    points with mass in between are less than ``refine_to`` posterior
    standard deviations (of log r) apart.
    """
    points: list[RT60Point] = []
    log_theta = np.log(THETA_DEFAULT)
    levels = None
    for rt60 in RT60_GRID:
        point = evaluate_rt60(model, rt60, log_theta, levels)
        points.append(point)
        log_theta, levels = point.log_theta, point.levels
        if verbose:
            print(f"  r {rt60:.3f}  {point.log_evidence:.1f}", flush=True)
    while len(points) < max_points:
        posterior = Posterior(points)
        ordered, weights, _ = posterior._weights()
        log_r = np.log([point.rt60 for point in ordered])
        mean = (weights * log_r).sum()
        sd = math.sqrt(max((weights * (log_r - mean) ** 2).sum(), 1e-12))
        gaps = np.diff(log_r)
        mass = weights[:-1] + weights[1:]
        wide = (gaps > refine_to * sd) & (mass > 1e-3)
        if not wide.any():
            break
        index = int(np.argmax(np.where(wide, mass, -1)))
        added_rt60 = float(np.exp(0.5 * (log_r[index] + log_r[index + 1])))
        start = ordered[index] if weights[index] >= weights[index + 1] else ordered[index + 1]
        point = evaluate_rt60(model, added_rt60, start.log_theta, start.levels)
        points.append(point)
        if verbose:
            print(f"  r {added_rt60:.4f}  {point.log_evidence:.1f}  (sd of log r {sd:.4f})", flush=True)
    return Posterior(points)
