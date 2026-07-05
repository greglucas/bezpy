"""Tests for Site3d interpolation/caching, convolution, and collections."""
import numpy as np

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
