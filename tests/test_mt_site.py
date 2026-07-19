"""Tests for Site3d interpolation/caching, convolution, and collections."""
import numpy as np
import pytest

import bezpy


def test_calcz_shape(site3d):
    freqs = np.logspace(-3, -1, 20)
    Z = site3d.calcZ(freqs)
    assert Z.shape == (4, len(freqs))
    assert np.isfinite(Z).all()


def test_calcz_cache_returns_same_object(site3d):
    freqs = np.logspace(-3, -1, 20)
    first = site3d.calcZ(freqs)
    second = site3d.calcZ(freqs)
    # The cached result is returned without recomputation.
    assert first is second


def test_calcz_cache_not_poisoned_by_mutation(site3d):
    # Regression: the cache must store a copy of the frequency array so that
    # in-place mutation of the caller's array cannot silently return stale Z.
    freqs = np.logspace(-3, -1, 20).copy()
    Z1 = site3d.calcZ(freqs).copy()
    freqs[5] = 999.0
    Z2 = site3d.calcZ(freqs)
    assert not np.array_equal(Z1[:, 5], Z2[:, 5])


def test_convolve_fft_shape_and_finiteness(site3d, rng):
    n = 1024
    mag_x = rng.standard_normal(n)
    mag_y = rng.standard_normal(n)
    Ex, Ey = site3d.convolve_fft(mag_x, mag_y, dt=60)
    assert Ex.shape == (n,)
    assert Ey.shape == (n,)
    assert np.isfinite(Ex).all()
    assert np.isfinite(Ey).all()


def test_convolve_fft_zero_input_gives_zero_field(site3d):
    n = 512
    zeros = np.zeros(n)
    Ex, Ey = site3d.convolve_fft(zeros, zeros, dt=60)
    assert np.allclose(Ex, 0.0)
    assert np.allclose(Ey, 0.0)


def test_site_collection_matches_individual_sites(rng):
    sites = [bezpy.mt.get_1d_site(n) for n in ["CP-1", "IP-3"]]
    coll = bezpy.mt.SiteCollection(sites)
    n = 1024
    mag_x = rng.standard_normal(n)
    mag_y = rng.standard_normal(n)
    Ex, Ey = coll.convolve_fft(mag_x, mag_y, dt=60)
    assert Ex.shape == (len(sites), n)
    # The batched result must equal the per-site computation.
    for i, site in enumerate(sites):
        ex_i, ey_i = site.convolve_fft(mag_x, mag_y, dt=60)
        assert np.allclose(Ex[i], ex_i)
        assert np.allclose(Ey[i], ey_i)


def test_site_collection_cache_respects_dt(rng):
    # Regression: same input length but a different dt must recompute the
    # impedances rather than returning a cache keyed only on length.
    sites = [bezpy.mt.get_1d_site("CP-1")]
    coll = bezpy.mt.SiteCollection(sites)
    n = 1024
    mag_x = rng.standard_normal(n)
    mag_y = rng.standard_normal(n)
    ex_60, _ = coll.convolve_fft(mag_x, mag_y, dt=60)
    ex_1, _ = coll.convolve_fft(mag_x, mag_y, dt=1)
    assert not np.allclose(ex_60, ex_1)


def test_dtir_runs_on_site(site3d):
    dtir = bezpy.mt.DTIR(nmin=-10, nmax=60)
    ns, zn = dtir.calculate(site=site3d)
    assert zn.shape[0] == 4
    assert len(ns) == zn.shape[1]
    assert np.isfinite(zn).all()


def test_calc_resistivity_deprecated_alias(site3d):

    site3d.calc_resistivity()
    expected = site3d.resistivity.copy()
    with pytest.warns(DeprecationWarning, match="calc_resistivity"):
        site3d.calc_resisitivity()
    np.testing.assert_array_equal(site3d.resistivity, expected)


def test_list_1d_sites():
    names = bezpy.mt.list_1d_sites()
    assert "CP1" in names
    assert names == sorted(names)
    # Every listed name resolves
    assert all(bezpy.mt.get_1d_site(name) is not None for name in names)


def test_short_period_list_interpolates():
    """Sites with fewer than 4 periods (e.g. downsampled gridded products)
    must fall back to a lower spline degree instead of failing in scipy."""
    import numpy as np

    from bezpy.mt.site import Site3d, SiteCollection
    for nper in (2, 3):
        periods = np.logspace(1, 3, nper)
        Z = np.zeros((4, nper), dtype=complex)
        Z[1] = (1 + 1j) * np.sqrt(1. / periods)
        Z[2] = -Z[1]
        site = Site3d("short", latitude=40., longitude=-105.,
                      periods=periods, Z=Z)
        ex, ey = SiteCollection([site]).convolve_fft(
            np.sin(np.arange(256) / 20.), np.cos(np.arange(256) / 20.), dt=60)
        assert np.isfinite(ex).all() and np.isfinite(ey).all()

    # A single valid estimate is not interpolable: clear error, names the site
    import pytest
    bad = Site3d("lonely", periods=np.array([10., 100.]),
                 Z=np.array([[1 + 1j, np.nan]] * 4, dtype=complex))
    with pytest.raises(ValueError, match="lonely"):
        bad.calcZ(np.array([0.01, 0.001]))


def test_spline_ignores_bad_variance_periods():
    """Non-finite or zero Z_var must not poison the interpolation (A1).

    A gridded-product row without an error column leaves Z_var NaN at a
    period whose Z is valid; previously that made the whole component NaN.
    """
    rng = np.random.default_rng(0)
    periods = np.logspace(1, 4, 12)
    Z = (rng.normal(size=(4, 12)) + 1j * rng.normal(size=(4, 12)))
    Z_var = np.full((4, 12), 0.01)
    Z_var[:, 5] = np.nan   # missing error column on one row
    Z_var[:, 7] = 0.0      # degenerate (infinite-weight) variance
    site = bezpy.mt.Site3d("mixed", latitude=40.0, longitude=-105.0,
                           periods=periods, Z=Z, Z_var=Z_var)
    Z_interp = site.calcZ(1.0 / np.logspace(1.2, 3.8, 7))
    assert np.all(np.isfinite(Z_interp))

    # Same Z without variances still interpolates (baseline sanity)
    plain = bezpy.mt.Site3d("plain", latitude=40.0, longitude=-105.0,
                            periods=periods, Z=Z)
    assert np.all(np.isfinite(plain.calcZ(1.0 / np.logspace(1.2, 3.8, 7))))


def test_perturbed_no_variance_returns_self():
    """Site1d (and a Site3d without Z_var) has nothing to propagate, so
    every ensemble realization must be identical (roadmap 1.2 phase 1)."""
    rng = np.random.default_rng(0)
    site1d = bezpy.mt.Site1d("hs", thicknesses=[1e9], resistivities=[100.])
    assert site1d.perturbed(rng) is site1d

    site3d = bezpy.mt.Site3d("plain", latitude=40., longitude=-105.,
                             periods=np.logspace(1, 4, 8),
                             Z=np.ones((4, 8), dtype=complex))
    assert site3d.perturbed(rng) is site3d


def test_perturbed_draws_within_reported_variance(site3d):
    """Site3d.perturbed must actually move Z, by roughly the reported std,
    and leave the original site untouched (a fresh copy each draw)."""
    rng = np.random.default_rng(0)
    original_Z = site3d.Z.copy()
    draws = np.stack([site3d.perturbed(rng).Z for _ in range(200)])
    np.testing.assert_array_equal(site3d.Z, original_Z)  # unmutated

    valid = np.isfinite(site3d.Z_var) & (site3d.Z_var > 0.)
    empirical_var = draws.real.var(axis=0) + draws.imag.var(axis=0)
    # Loose tolerance: 200 draws gives a noisy variance estimate, this only
    # needs to catch a badly wrong scale factor (e.g. missing the /2 split).
    ratio = empirical_var[valid] / site3d.Z_var[valid]
    assert np.median(ratio) == pytest.approx(1.0, rel=0.5)
    # Components without a usable Z_var are never perturbed.
    np.testing.assert_array_equal(
        draws[:, ~valid], np.broadcast_to(original_Z[~valid], draws[:, ~valid].shape))


def test_z_param_jacobian_matches_independent_finite_difference(site3d):
    """_z_param_jacobian must match a hand-rolled central difference through
    calcZ directly (roadmap 1.2 phase 2) -- and must hold Z_var identical
    between the two (Z_var None vs given selects different spline smoothing,
    see spline_interp; comparing across that mismatch is the wrong test)."""
    freqs = np.logspace(-4, -1, 20)
    component, period_idx, part, eps = 1, 3, "real", 1e-4
    jac = site3d._z_param_jacobian(freqs, "asymptotic", component, period_idx, part, eps)

    Zp, Zm = site3d.Z.copy(), site3d.Z.copy()
    Zp[component, period_idx] += eps
    Zm[component, period_idx] -= eps
    plus = bezpy.mt.Site3d("p", periods=site3d.periods, Z=Zp, Z_var=site3d.Z_var)
    minus = bezpy.mt.Site3d("m", periods=site3d.periods, Z=Zm, Z_var=site3d.Z_var)
    brute = (plus.calcZ(freqs, extrapolate="asymptotic")[component]
            - minus.calcZ(freqs, extrapolate="asymptotic")[component]) / (2 * eps)

    np.testing.assert_allclose(jac, brute)
    # Real/imag are independently-fit splines: perturbing the real part must
    # leave the output's imaginary part exactly unperturbed.
    np.testing.assert_allclose(jac.imag, 0., atol=1e-12)


def test_z_sensitivities_empty_without_variance():
    site = bezpy.mt.Site3d("plain", latitude=40., longitude=-105.,
                           periods=np.logspace(1, 4, 6),
                           Z=np.ones((4, 6), dtype=complex))
    assert site.z_sensitivities(np.logspace(-4, -1, 10)) == []


def test_convolve_fft_return_var_matches_ensemble_in_linear_regime(rng):
    """Analytic Ex_var/Ey_var (roadmap 1.2 phase 2) must agree with the
    Monte Carlo ensemble spread (phase 1) in the small-relative-variance
    regime where the production spline's nonlinearity is negligible --
    the two propagation methods derive from the same Z_var and should
    converge as the perturbation shrinks."""
    from bezpy.mt.site import Site3d, SiteCollection
    periods = np.logspace(1, 4, 10)
    Z = np.zeros((4, len(periods)), dtype=complex)
    Z[1, :] = (1. + 0.5j) * (1. + 0.1 * np.arange(len(periods)))
    Z[2, :] = -Z[1, :]
    Z_var = (0.02 * np.abs(Z))**2 + 1e-8
    site = Site3d("S0", latitude=40., longitude=-100., periods=periods, Z=Z, Z_var=Z_var)

    mag_x, mag_y = rng.standard_normal(1024) * 20, rng.standard_normal(1024) * 20
    ex, ey, ex_var, ey_var = SiteCollection([site]).convolve_fft(
        mag_x, mag_y, dt=60., return_var=True)
    assert (ex_var > 0).all() and (ey_var > 0).all()

    draws = np.stack([SiteCollection([site.perturbed(rng)]).convolve_fft(
        mag_x, mag_y, dt=60.)[0][0] for _ in range(300)])
    mc_var = draws.var(axis=0)
    ratio = ex_var[0].mean() / mc_var.mean()
    assert ratio == pytest.approx(1.0, rel=0.3)


def test_convolve_fft_return_var_zero_without_variance():
    """Site1d sites (and Site3d without Z_var) have nothing to propagate,
    so their analytic variance must be exactly zero, not fabricated."""
    from bezpy.mt.site import SiteCollection
    site = bezpy.mt.Site1d("hs", thicknesses=[1e9], resistivities=[100.])
    site.latitude, site.longitude = 40., -100.
    ex, ey, ex_var, ey_var = SiteCollection([site]).convolve_fft(
        np.sin(np.arange(256) / 20.), np.cos(np.arange(256) / 20.),
        dt=60, return_var=True)
    np.testing.assert_array_equal(ex_var, 0.)
    np.testing.assert_array_equal(ey_var, 0.)


def test_collection_site_chunk_equivalence():
    """Blocked processing returns the same fields as one block (P1)."""
    rng = np.random.default_rng(3)
    sites = []
    for i in range(5):
        site = bezpy.mt.Site1d(f"s{i}", [20e3, 1e9], [100.0 * (i + 1), 1000.0])
        site.latitude, site.longitude = 40.0 + i, -105.0
        sites.append(site)
    collection = bezpy.mt.SiteCollection(sites)
    bx = rng.normal(size=256)
    by = rng.normal(size=256)
    ex_all, ey_all = collection.convolve_fft(bx, by, dt=60)
    ex_blk, ey_blk = bezpy.mt.SiteCollection(sites).convolve_fft(
        bx, by, dt=60, site_chunk=2)
    np.testing.assert_allclose(ex_blk, ex_all)
    np.testing.assert_allclose(ey_blk, ey_all)

    # Per-site fields through the blocked path as well
    bx2 = rng.normal(size=(256, 5))
    by2 = rng.normal(size=(256, 5))
    ex_all, ey_all = collection.convolve_fft(bx2, by2, dt=60)
    ex_blk, ey_blk = bezpy.mt.SiteCollection(sites).convolve_fft(
        bx2, by2, dt=60, site_chunk=2)
    np.testing.assert_allclose(ex_blk, ex_all)
    np.testing.assert_allclose(ey_blk, ey_all)
