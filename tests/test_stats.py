"""Tests for bezpy.stats.return_level: recovery, CIs, method comparison."""

import numpy as np
import pytest
import scipy.stats

from bezpy.stats import return_level

RNG = np.random.default_rng(21)


def gpd_truth(u, sigma, xi, rate, T):
    m = rate * T
    return u + sigma / xi * (m**xi - 1.0) if abs(xi) > 1e-9 else u + sigma * np.log(m)


def make_gpd_sample(n=400, u=50.0, sigma=10.0, xi=0.1):
    return u + scipy.stats.genpareto.rvs(xi, scale=sigma, size=n, random_state=RNG)


def test_gpd_parameter_recovery():
    u, sigma, xi, years = 50.0, 10.0, 0.1, 40.0
    data = make_gpd_sample(2000, u, sigma, xi)
    rl = return_level(data, years_observed=years, return_period=100.0, threshold=u, ci="profile")
    truth = gpd_truth(u, sigma, xi, len(data) / years, 100.0)
    assert abs(rl.value - truth) / truth < 0.1
    assert abs(rl.params["shape"] - xi) < 0.1
    assert abs(rl.params["scale"] - sigma) / sigma < 0.15


def test_shape_prior_stabilizes_small_samples():
    """A Gaussian shape prior regularizes the notoriously unstable small-sample
    GPD fit: on a dozen-odd heavy-tailed exceedances it delivers a lower-RMSE
    return level and much tighter CIs, without biasing a large clean sample."""
    u, sigma, xi, years = 50.0, 10.0, 0.3, 30.0
    rate = 13 / years
    truth = gpd_truth(u, sigma, xi, rate, 100.0)
    rng = np.random.default_rng(7)

    def relwidth(rl):
        return (rl.ci_high - rl.ci_low) / rl.value if np.isfinite(rl.ci_high) else np.inf

    mle_err, pri_err, mle_w, pri_w = [], [], [], []
    for _ in range(120):
        x = u + scipy.stats.genpareto.rvs(xi, scale=sigma, size=13, random_state=rng)
        for scale, errs, widths in ((None, mle_err, mle_w), (0.5, pri_err, pri_w)):
            try:
                rl = return_level(x, years, threshold=u, ci="profile", shape_prior_scale=scale)
            except ValueError:
                continue
            errs.append((rl.value - truth) / truth)
            widths.append(relwidth(rl))

    # lower RMSE and tighter median CI with the prior
    assert np.sqrt(np.mean(np.square(pri_err))) < 0.6 * np.sqrt(np.mean(np.square(mle_err)))
    assert np.median(pri_w) < 0.5 * np.median(mle_w)

    # non-breaking on a large clean sample: prior ~ MLE, both recover truth
    big = make_gpd_sample(1500, u, sigma, xi)
    mle = return_level(big, years_observed=40.0, threshold=u, ci="profile")
    pri = return_level(big, years_observed=40.0, threshold=u, ci="profile", shape_prior_scale=0.5)
    assert abs(pri.value - mle.value) / mle.value < 0.05


def test_gpd_ci_brackets_and_orders():
    data = make_gpd_sample(300)
    profile = return_level(data, years_observed=30.0, threshold=50.0, ci="profile")
    boot = return_level(
        data, years_observed=30.0, threshold=50.0, ci="bootstrap", n_bootstrap=150, seed=1
    )
    for rl in (profile, boot):
        assert rl.ci_low < rl.value < rl.ci_high
    # Profile intervals for return levels are famously asymmetric: the
    # upper arm must be longer than the lower one
    assert (profile.ci_high - profile.value) > (profile.value - profile.ci_low)


def test_gpd_profile_ci_coverage():
    """~95% profile CIs must actually cover the truth (loose bounds to stay
    deterministic-fast: 60 replicates)."""
    u, sigma, xi, n, years = 50.0, 10.0, 0.1, 150, 30.0
    truth = gpd_truth(u, sigma, xi, n / years, 100.0)
    rng = np.random.default_rng(5)
    hits = 0
    trials = 60
    for _ in range(trials):
        data = u + scipy.stats.genpareto.rvs(xi, scale=sigma, size=n, random_state=rng)
        rl = return_level(data, years_observed=years, threshold=u, ci="profile")
        hits += rl.ci_low <= truth <= rl.ci_high
    assert hits / trials > 0.8


def test_lognormal_2019_recovery():
    years = 31.0
    # ~20 events/year keeps the extreme-quantile estimator variance modest
    maxima = scipy.stats.lognorm.rvs(0.8, scale=30.0, size=620, random_state=RNG)
    rl = return_level(
        maxima,
        years_observed=years,
        return_period=100.0,
        method="lognormal-2019",
        ci="bootstrap",
        n_bootstrap=300,
        seed=2,
    )
    truth = scipy.stats.lognorm.ppf(1.0 - 1.0 / (20.0 * 100.0), 0.8, scale=30.0)
    assert abs(rl.value - truth) / truth < 0.3
    assert rl.ci_low < rl.value < rl.ci_high


def test_gev_recovery():
    annual_max = scipy.stats.genextreme.rvs(-0.1, loc=100.0, scale=20.0, size=60, random_state=RNG)
    rl = return_level(
        annual_max, years_observed=60.0, method="gev", ci="bootstrap", n_bootstrap=60, seed=3
    )
    truth = scipy.stats.genextreme.ppf(0.99, -0.1, loc=100.0, scale=20.0)
    assert abs(rl.value - truth) / truth < 0.25


def test_methods_diverge_on_heavy_tail():
    """The D5 point: a lognormal body-fit and a GPD tail-fit give different
    answers on tail-heavy data -- the choice matters."""
    u = 50.0
    data = u + scipy.stats.genpareto.rvs(0.3, scale=10.0, size=300, random_state=RNG)
    gpd = return_level(data, years_observed=30.0, threshold=u, ci="profile")
    logn = return_level(
        data, years_observed=30.0, method="lognormal-2019", ci="bootstrap", n_bootstrap=100, seed=4
    )
    assert abs(gpd.value - logn.value) / gpd.value > 0.1


def test_validation_errors():
    data = make_gpd_sample(100)
    with pytest.raises(ValueError, match="method must be"):
        return_level(data, 30.0, method="weibull")
    with pytest.raises(ValueError, match="ci must be"):
        return_level(data, 30.0, ci="magic")
    with pytest.raises(ValueError, match="profile CIs"):
        return_level(data, 30.0, method="gev", ci="profile")
    with pytest.raises(ValueError, match="at least 5"):
        return_level(data[:3], 30.0)
    with pytest.raises(ValueError, match="exceedances above"):
        return_level(data, 30.0, threshold=data.max() + 1)
    with pytest.raises(ValueError, match="years_observed"):
        return_level(data, 0)
    with pytest.raises(ValueError, match="return_period"):
        return_level(data, 30, return_period=0)
    with pytest.raises(ValueError, match="ci_level"):
        return_level(data, 30, ci_level=1)
    with pytest.raises(ValueError, match="shape_prior_scale"):
        return_level(data, 30, shape_prior_scale=0)


def test_str_output():
    rl = return_level(
        make_gpd_sample(200),
        years_observed=30.0,
        threshold=50.0,
        ci="bootstrap",
        n_bootstrap=100,
        seed=6,
    )
    text = str(rl)
    assert "100-year return level" in text
    assert "95% CI" in text


def test_gev_warns_on_block_count_mismatch(rng):
    """GEV expects one annual maximum per observed year (B4)."""
    data = scipy.stats.genextreme.rvs(-0.1, loc=10, scale=2, size=60, random_state=42)
    with pytest.warns(UserWarning, match="annual"):
        return_level(data, years_observed=30, method="gev", ci="bootstrap", n_bootstrap=25, seed=0)
