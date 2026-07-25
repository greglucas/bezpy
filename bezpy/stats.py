"""Tail statistics for geoelectric / GIC hazard: return levels with CIs.

The default method is peaks-over-threshold with a Generalized Pareto tail
(``method="gpd"``) -- the exceedances produced by the continuous scan
(``bezpy.pipeline.scan``) and declustered by ``bezpy.storms.find_events``
feed it directly. Lognormal fits of storm maxima (the 2019 hazard paper's
method) are kept, by name, as ``method="lognormal-2019"``; block (annual)
maxima with a GEV are available as ``method="gev"``.

Every estimate carries a confidence interval: **a return level without a CI
is not a result.**
"""

__all__ = ["ReturnLevel", "return_level"]

import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.optimize
import scipy.stats

_BOOTSTRAP_DEFAULT = 999


@dataclass
class ReturnLevel:
    """A return-level estimate with its confidence interval."""

    value: float
    ci_low: float
    ci_high: float
    ci_level: float
    return_period: float
    method: str
    params: dict = field(default_factory=dict)

    def __str__(self):
        return (
            f"{self.return_period:g}-year return level: {self.value:.4g} "
            f"[{self.ci_low:.4g}, {self.ci_high:.4g}] "
            f"({self.ci_level:.0%} CI, {self.method})"
        )


def _gpd_level(u, sigma, xi, rate, return_period):
    """GPD return level: the value exceeded once per ``return_period`` years."""
    m = rate * return_period  # expected exceedance clusters in the period
    if abs(xi) < 1e-9:
        return u + sigma * np.log(m)
    return u + sigma / xi * (m**xi - 1.0)


def _shape_penalty(xi, prior_scale):
    """Negative-log of a mean-zero Gaussian prior on the shape (0 if disabled).

    The prior shrinks xi toward 0. Being independent of the sample size, its
    weight relative to the data log-likelihood fades as the number of
    exceedances grows -- it regularizes small samples (where the MLE of xi is
    wild and blows the return level and its CI up) and all but vanishes on
    large ones.
    """
    return 0.0 if prior_scale is None else xi * xi / (2.0 * prior_scale * prior_scale)


def _fit_gpd(excesses, prior_scale=None):
    """(Penalized) ML fit of GPD(shape xi, scale sigma) above the threshold.

    With ``prior_scale`` set, maximizes the likelihood times a Gaussian shape
    prior (a MAP estimate); otherwise the plain scipy MLE.
    """
    xi0, _, sigma0 = scipy.stats.genpareto.fit(excesses, floc=0)
    if prior_scale is None:
        return xi0, sigma0

    def obj(p):
        xi, log_sigma = p
        return _gpd_nll(excesses, xi, np.exp(log_sigma)) + _shape_penalty(xi, prior_scale)

    res = scipy.optimize.minimize(
        obj,
        [np.clip(xi0, -0.9, 1.5), np.log(max(sigma0, 1e-9))],
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-8},
    )
    xi, log_sigma = res.x
    return float(xi), float(np.exp(log_sigma))


def _gpd_nll(excesses, xi, sigma):
    if sigma <= 0:
        return np.inf
    logpdf = scipy.stats.genpareto.logpdf(excesses, xi, loc=0, scale=sigma)
    return np.inf if not np.all(np.isfinite(logpdf)) else -logpdf.sum()


def _gpd_profile_ci(excesses, u, rate, return_period, level, x_hat, nll_hat, prior_scale=None):
    """Profile-likelihood CI on the GPD return level (Coles 2001, 4.3.3).

    The likelihood is re-parametrized in (return level, shape); for each
    candidate level the shape is profiled out, and the CI is where the
    deviance crosses the chi-square(1) quantile. The exceedance rate is held
    at its estimate (Poisson variability in the count is not profiled). With a
    ``prior_scale`` the same Gaussian shape penalty used for the fit is carried
    through, giving a penalized-likelihood interval that stays finite when the
    unpenalized profile would run away on a heavy-tailed small sample.
    """
    m = rate * return_period
    log_m = np.log(m)

    def profile_nll(x_t):
        if x_t <= u:
            return np.inf

        def nll_of_xi(xi):
            sigma = (x_t - u) / log_m if abs(xi) < 1e-9 else (x_t - u) * xi / (m**xi - 1.0)
            return _gpd_nll(excesses, xi, sigma) + _shape_penalty(xi, prior_scale)

        result = scipy.optimize.minimize_scalar(nll_of_xi, bounds=(-1.0, 2.0), method="bounded")
        return result.fun

    target = nll_hat + scipy.stats.chi2.ppf(level, df=1) / 2.0

    def deviance_gap(x_t):
        return profile_nll(x_t) - target

    span = x_hat - u
    # Lower bound: bracket between the threshold and the estimate
    lo_bracket = u + 1e-9 * span
    ci_low = (
        scipy.optimize.brentq(deviance_gap, lo_bracket, x_hat)
        if deviance_gap(lo_bracket) > 0
        else lo_bracket
    )
    # Upper bound: expand until the deviance crosses; heavy tails can make
    # the upper limit effectively unbounded
    hi = x_hat + span
    for _ in range(60):
        if deviance_gap(hi) > 0:
            ci_high = scipy.optimize.brentq(deviance_gap, x_hat, hi)
            break
        hi = u + (hi - u) * 2.0
    else:
        ci_high = np.inf
    return ci_low, ci_high


def _bootstrap_ci(data, estimator, level, n_boot, rng):
    estimates = []
    n = len(data)
    for _ in range(n_boot):
        sample = data[rng.integers(0, n, n)]
        try:
            estimates.append(estimator(sample))
        except (ValueError, RuntimeError):
            continue
    if not estimates:
        raise RuntimeError("all bootstrap replicates failed to fit")
    alpha = (1.0 - level) / 2.0
    return tuple(np.nanquantile(estimates, [alpha, 1.0 - alpha]))


def return_level(
    data,
    years_observed,
    return_period=100.0,
    method="gpd",
    threshold=None,
    ci="profile",
    ci_level=0.95,
    n_bootstrap=_BOOTSTRAP_DEFAULT,
    seed=None,
    shape_prior_scale=None,
):
    """Estimate the T-year return level of a hazard quantity, with a CI.

    :param data: the sample the tail is fit to. For ``method="gpd"``:
        declustered exceedance peaks (e.g. ``peak_value`` from
        ``bezpy.storms.find_events``). For ``method="lognormal-2019"``:
        per-storm maxima (the 2019 paper's input). For ``method="gev"``:
        block (annual) maxima.
    :param years_observed: length of the observation span in years -- sets
        the event rate (GPD, lognormal) or is checked against the number of
        blocks (GEV)
    :param return_period: return period in years (default 100)
    :param method:

        * ``"gpd"`` (default, D5): peaks-over-threshold with a Generalized
          Pareto tail -- fits the tail, not the body.
        * ``"lognormal-2019"``: lognormal fit of storm maxima, reproducing
          the published 2019 workflow (body-dominated; kept for
          reproduction and comparison).
        * ``"gev"``: GEV fit of the annual maxima.
    :param threshold: for ``"gpd"``, the threshold the exceedances refer to;
        defaults to the smallest data value (correct when ``data`` are the
        exceedances of a known scan threshold -- pass it explicitly)
    :param ci: ``"profile"`` (profile likelihood; GPD only) or
        ``"bootstrap"`` (any method)
    :param ci_level: confidence level (default 0.95)
    :param n_bootstrap: bootstrap replicates for ``ci="bootstrap"``
    :param seed: random seed for the bootstrap
    :param shape_prior_scale: for ``method="gpd"``, the standard deviation of a
        mean-zero Gaussian prior on the tail shape xi (a penalized/MAP fit).
        ``None`` (default) is the plain MLE. A value like ``0.5`` regularizes
        the notoriously unstable small-sample shape estimate -- it shrinks xi
        toward 0 when there are few exceedances (taming runaway return levels
        and their profile CIs) while fading to the MLE as the sample grows.
        Recommended for per-site hazard fits with only a dozen-odd exceedances.
    :return: :class:`ReturnLevel` -- value, CI, and fit parameters
    """
    if not np.isfinite(years_observed) or years_observed <= 0:
        raise ValueError("years_observed must be finite and positive")
    if not np.isfinite(return_period) or return_period <= 0:
        raise ValueError("return_period must be finite and positive")
    if not np.isfinite(ci_level) or not 0 < ci_level < 1:
        raise ValueError("ci_level must lie strictly between 0 and 1")
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if shape_prior_scale is not None and (
        not np.isfinite(shape_prior_scale) or shape_prior_scale <= 0
    ):
        raise ValueError("shape_prior_scale must be finite and positive or None")
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 5:
        raise ValueError(f"need at least 5 values to fit a tail, got {len(data)}")
    if ci not in ("profile", "bootstrap"):
        raise ValueError("ci must be one of: profile, bootstrap")
    rng = np.random.default_rng(seed)

    if method == "gpd":
        u = float(np.min(data)) if threshold is None else float(threshold)
        if not np.isfinite(u):
            raise ValueError("threshold must be finite")
        excesses = data[data > u] - u
        if len(excesses) < 5:
            raise ValueError("fewer than 5 exceedances above the threshold")
        rate = len(excesses) / years_observed
        if rate * return_period <= 1:
            raise ValueError(
                "return_period is too short for the fitted exceedance rate: "
                "the requested level would not lie above the GPD threshold"
            )
        xi, sigma = _fit_gpd(excesses, prior_scale=shape_prior_scale)
        value = _gpd_level(u, sigma, xi, rate, return_period)
        params = {"shape": xi, "scale": sigma, "threshold": u, "rate": rate}
        if ci == "profile":
            nll_hat = _gpd_nll(excesses, xi, sigma) + _shape_penalty(xi, shape_prior_scale)
            ci_low, ci_high = _gpd_profile_ci(
                excesses,
                u,
                rate,
                return_period,
                ci_level,
                value,
                nll_hat,
                prior_scale=shape_prior_scale,
            )
        else:

            def estimate(sample):
                xi_b, sigma_b = _fit_gpd(sample, prior_scale=shape_prior_scale)
                return _gpd_level(u, sigma_b, xi_b, len(sample) / years_observed, return_period)

            ci_low, ci_high = _bootstrap_ci(excesses, estimate, ci_level, n_bootstrap, rng)

    elif method == "lognormal-2019":
        if ci == "profile":
            raise ValueError(
                "profile CIs are implemented for method='gpd'; "
                "use ci='bootstrap' with lognormal-2019"
            )
        rate = len(data) / years_observed  # storms per year
        if np.any(data <= 0):
            raise ValueError("lognormal-2019 requires strictly positive data")
        if rate * return_period <= 1:
            raise ValueError("return_period is too short for the observed storm rate")

        def estimate(sample):
            shape, _, scale = scipy.stats.lognorm.fit(sample, floc=0)
            return scipy.stats.lognorm.ppf(
                1.0 - 1.0 / (rate * return_period), shape, loc=0, scale=scale
            )

        value = estimate(data)
        shape, _, scale = scipy.stats.lognorm.fit(data, floc=0)
        params = {"shape": shape, "scale": scale, "rate": rate}
        ci_low, ci_high = _bootstrap_ci(data, estimate, ci_level, n_bootstrap, rng)

    elif method == "gev":
        if ci == "profile":
            raise ValueError(
                "profile CIs are implemented for method='gpd'; use ci='bootstrap' with gev"
            )
        if abs(len(data) - years_observed) > 1:
            warnings.warn(
                f"method='gev' expects one block (annual) maximum per observed "
                f"year, but got {len(data)} maxima for {years_observed:g} years "
                "-- check that the input is annual maxima, not exceedances",
                stacklevel=2,
            )
        if return_period <= 1:
            raise ValueError("method='gev' requires return_period > 1 year")

        def estimate(sample):
            c, loc, scale = scipy.stats.genextreme.fit(sample)
            return scipy.stats.genextreme.ppf(1.0 - 1.0 / return_period, c, loc, scale)

        value = estimate(data)
        c, loc, scale = scipy.stats.genextreme.fit(data)
        params = {"shape": -c, "loc": loc, "scale": scale}  # Coles sign
        ci_low, ci_high = _bootstrap_ci(data, estimate, ci_level, n_bootstrap, rng)

    else:
        raise ValueError("method must be one of: gpd, lognormal-2019, gev")

    if not np.isfinite(value) or np.isnan(ci_low) or np.isnan(ci_high):
        raise RuntimeError("tail fit produced a non-finite return-level estimate")
    if ci_low > value or ci_high < value:
        warnings.warn(
            "bootstrap/profile interval does not contain the point estimate; "
            "the tail fit may be unstable for this sample",
            stacklevel=2,
        )
    return ReturnLevel(
        value=float(value),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        ci_level=ci_level,
        return_period=float(return_period),
        method=method,
        params=params,
    )
