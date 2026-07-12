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
        return (f"{self.return_period:g}-year return level: {self.value:.4g} "
                f"[{self.ci_low:.4g}, {self.ci_high:.4g}] "
                f"({self.ci_level:.0%} CI, {self.method})")


def _gpd_level(u, sigma, xi, rate, return_period):
    """GPD return level: the value exceeded once per ``return_period`` years."""
    m = rate * return_period  # expected exceedance clusters in the period
    if abs(xi) < 1e-9:
        return u + sigma*np.log(m)
    return u + sigma/xi*(m**xi - 1.)


def _fit_gpd(excesses):
    """MLE fit of GPD(shape xi, scale sigma) to excesses above the threshold."""
    xi, _, sigma = scipy.stats.genpareto.fit(excesses, floc=0)
    return xi, sigma


def _gpd_nll(excesses, xi, sigma):
    if sigma <= 0:
        return np.inf
    logpdf = scipy.stats.genpareto.logpdf(excesses, xi, loc=0, scale=sigma)
    return np.inf if not np.all(np.isfinite(logpdf)) else -logpdf.sum()


def _gpd_profile_ci(excesses, u, rate, return_period, level, x_hat, nll_hat):
    """Profile-likelihood CI on the GPD return level (Coles 2001, 4.3.3).

    The likelihood is re-parametrized in (return level, shape); for each
    candidate level the shape is profiled out, and the CI is where the
    deviance crosses the chi-square(1) quantile. The exceedance rate is held
    at its estimate (Poisson variability in the count is not profiled).
    """
    m = rate * return_period
    log_m = np.log(m)

    def profile_nll(x_t):
        if x_t <= u:
            return np.inf

        def nll_of_xi(xi):
            sigma = (x_t - u)/log_m if abs(xi) < 1e-9 else \
                (x_t - u)*xi/(m**xi - 1.)
            return _gpd_nll(excesses, xi, sigma)
        result = scipy.optimize.minimize_scalar(nll_of_xi, bounds=(-1., 2.),
                                                method="bounded")
        return result.fun

    target = nll_hat + scipy.stats.chi2.ppf(level, df=1)/2.

    def deviance_gap(x_t):
        return profile_nll(x_t) - target

    span = x_hat - u
    # Lower bound: bracket between the threshold and the estimate
    lo_bracket = u + 1e-9*span
    ci_low = scipy.optimize.brentq(deviance_gap, lo_bracket, x_hat) \
        if deviance_gap(lo_bracket) > 0 else lo_bracket
    # Upper bound: expand until the deviance crosses; heavy tails can make
    # the upper limit effectively unbounded
    hi = x_hat + span
    for _ in range(60):
        if deviance_gap(hi) > 0:
            ci_high = scipy.optimize.brentq(deviance_gap, x_hat, hi)
            break
        hi = u + (hi - u)*2.
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
    alpha = (1. - level)/2.
    return tuple(np.nanquantile(estimates, [alpha, 1. - alpha]))


def return_level(data, years_observed, return_period=100., method="gpd",
                 threshold=None, ci="profile", ci_level=0.95,
                 n_bootstrap=_BOOTSTRAP_DEFAULT, seed=None):
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
    :return: :class:`ReturnLevel` -- value, CI, and fit parameters
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 5:
        raise ValueError(f"need at least 5 values to fit a tail, got {len(data)}")
    if ci not in ("profile", "bootstrap"):
        raise ValueError("ci must be one of: profile, bootstrap")
    rng = np.random.default_rng(seed)

    if method == "gpd":
        u = float(np.min(data)) if threshold is None else float(threshold)
        excesses = data[data > u] - u
        if len(excesses) < 5:
            raise ValueError("fewer than 5 exceedances above the threshold")
        rate = len(excesses)/years_observed
        xi, sigma = _fit_gpd(excesses)
        value = _gpd_level(u, sigma, xi, rate, return_period)
        params = {"shape": xi, "scale": sigma, "threshold": u, "rate": rate}
        if ci == "profile":
            nll_hat = _gpd_nll(excesses, xi, sigma)
            ci_low, ci_high = _gpd_profile_ci(excesses, u, rate, return_period,
                                              ci_level, value, nll_hat)
        else:
            def estimate(sample):
                xi_b, sigma_b = _fit_gpd(sample)
                return _gpd_level(u, sigma_b, xi_b, len(sample)/years_observed,
                                  return_period)
            ci_low, ci_high = _bootstrap_ci(excesses, estimate, ci_level,
                                            n_bootstrap, rng)

    elif method == "lognormal-2019":
        if ci == "profile":
            raise ValueError("profile CIs are implemented for method='gpd'; "
                             "use ci='bootstrap' with lognormal-2019")
        rate = len(data)/years_observed  # storms per year

        def estimate(sample):
            shape, _, scale = scipy.stats.lognorm.fit(sample, floc=0)
            return scipy.stats.lognorm.ppf(1. - 1./(rate*return_period),
                                           shape, loc=0, scale=scale)
        value = estimate(data)
        shape, _, scale = scipy.stats.lognorm.fit(data, floc=0)
        params = {"shape": shape, "scale": scale, "rate": rate}
        ci_low, ci_high = _bootstrap_ci(data, estimate, ci_level,
                                        n_bootstrap, rng)

    elif method == "gev":
        if ci == "profile":
            raise ValueError("profile CIs are implemented for method='gpd'; "
                             "use ci='bootstrap' with gev")
        if abs(len(data) - years_observed) > 1:
            warnings.warn(
                f"method='gev' expects one block (annual) maximum per observed "
                f"year, but got {len(data)} maxima for {years_observed:g} years "
                "-- check that the input is annual maxima, not exceedances",
                stacklevel=2)

        def estimate(sample):
            c, loc, scale = scipy.stats.genextreme.fit(sample)
            return scipy.stats.genextreme.ppf(1. - 1./return_period,
                                              c, loc, scale)
        value = estimate(data)
        c, loc, scale = scipy.stats.genextreme.fit(data)
        params = {"shape": -c, "loc": loc, "scale": scale}  # Coles sign
        ci_low, ci_high = _bootstrap_ci(data, estimate, ci_level,
                                        n_bootstrap, rng)

    else:
        raise ValueError("method must be one of: gpd, lognormal-2019, gev")

    return ReturnLevel(value=float(value), ci_low=float(ci_low),
                       ci_high=float(ci_high), ci_level=ci_level,
                       return_period=float(return_period), method=method,
                       params=params)
