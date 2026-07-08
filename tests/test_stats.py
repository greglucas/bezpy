"""Tests for bezpy.stats.return_level: recovery, CIs, method comparison."""
import numpy as np
import pytest
import scipy.stats

from bezpy.stats import return_level

RNG = np.random.default_rng(21)


def gpd_truth(u, sigma, xi, rate, T):
    m = rate*T
    return u + sigma/xi*(m**xi - 1.) if abs(xi) > 1e-9 else u + sigma*np.log(m)


def make_gpd_sample(n=400, u=50., sigma=10., xi=0.1):
    return u + scipy.stats.genpareto.rvs(xi, scale=sigma, size=n,
                                         random_state=RNG)


def test_gpd_parameter_recovery():
    u, sigma, xi, years = 50., 10., 0.1, 40.
    data = make_gpd_sample(2000, u, sigma, xi)
    rl = return_level(data, years_observed=years, return_period=100.,
                      threshold=u, ci="profile")
    truth = gpd_truth(u, sigma, xi, len(data)/years, 100.)
    assert abs(rl.value - truth) / truth < 0.1
    assert abs(rl.params["shape"] - xi) < 0.1
    assert abs(rl.params["scale"] - sigma) / sigma < 0.15


def test_gpd_ci_brackets_and_orders():
    data = make_gpd_sample(300)
    profile = return_level(data, years_observed=30., threshold=50.,
                           ci="profile")
    boot = return_level(data, years_observed=30., threshold=50.,
                        ci="bootstrap", n_bootstrap=150, seed=1)
    for rl in (profile, boot):
        assert rl.ci_low < rl.value < rl.ci_high
    # Profile intervals for return levels are famously asymmetric: the
    # upper arm must be longer than the lower one
    assert (profile.ci_high - profile.value) > (profile.value - profile.ci_low)


def test_gpd_profile_ci_coverage():
    """~95% profile CIs must actually cover the truth (loose bounds to stay
    deterministic-fast: 60 replicates)."""
    u, sigma, xi, n, years = 50., 10., 0.1, 150, 30.
    truth = gpd_truth(u, sigma, xi, n/years, 100.)
    rng = np.random.default_rng(5)
    hits = 0
    trials = 60
    for _ in range(trials):
        data = u + scipy.stats.genpareto.rvs(xi, scale=sigma, size=n,
                                             random_state=rng)
        rl = return_level(data, years_observed=years, threshold=u,
                          ci="profile")
        hits += rl.ci_low <= truth <= rl.ci_high
    assert hits / trials > 0.8


def test_lognormal_2019_recovery():
    years = 31.
    # ~20 events/year keeps the extreme-quantile estimator variance modest
    maxima = scipy.stats.lognorm.rvs(0.8, scale=30., size=620, random_state=RNG)
    rl = return_level(maxima, years_observed=years, return_period=100.,
                      method="lognormal-2019", ci="bootstrap",
                      n_bootstrap=300, seed=2)
    truth = scipy.stats.lognorm.ppf(1. - 1./(20.*100.), 0.8, scale=30.)
    assert abs(rl.value - truth)/truth < 0.3
    assert rl.ci_low < rl.value < rl.ci_high


def test_gev_recovery():
    annual_max = scipy.stats.genextreme.rvs(-0.1, loc=100., scale=20.,
                                            size=60, random_state=RNG)
    rl = return_level(annual_max, years_observed=60., method="gev",
                      ci="bootstrap", n_bootstrap=60, seed=3)
    truth = scipy.stats.genextreme.ppf(0.99, -0.1, loc=100., scale=20.)
    assert abs(rl.value - truth)/truth < 0.25


def test_methods_diverge_on_heavy_tail():
    """The D5 point: a lognormal body-fit and a GPD tail-fit give different
    answers on tail-heavy data -- the choice matters."""
    u = 50.
    data = u + scipy.stats.genpareto.rvs(0.3, scale=10., size=300,
                                         random_state=RNG)
    gpd = return_level(data, years_observed=30., threshold=u, ci="profile")
    logn = return_level(data, years_observed=30., method="lognormal-2019",
                        ci="bootstrap", n_bootstrap=100, seed=4)
    assert abs(gpd.value - logn.value) / gpd.value > 0.1


def test_validation_errors():
    data = make_gpd_sample(100)
    with pytest.raises(ValueError, match="method must be"):
        return_level(data, 30., method="weibull")
    with pytest.raises(ValueError, match="ci must be"):
        return_level(data, 30., ci="magic")
    with pytest.raises(ValueError, match="profile CIs"):
        return_level(data, 30., method="gev", ci="profile")
    with pytest.raises(ValueError, match="at least 5"):
        return_level(data[:3], 30.)
    with pytest.raises(ValueError, match="exceedances above"):
        return_level(data, 30., threshold=data.max() + 1)


def test_str_output():
    rl = return_level(make_gpd_sample(200), years_observed=30., threshold=50.,
                      ci="bootstrap", n_bootstrap=100, seed=6)
    text = str(rl)
    assert "100-year return level" in text
    assert "95% CI" in text
