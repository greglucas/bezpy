"""Tests for the precomputed voltage functionals and telluric response kernels.

The load-bearing property is *exact reassociation*: the kernels must
reproduce the existing calc_voltages / convolve_fft pipeline to rounding
error, because they perform the same linear operations in a different order.
"""

import numpy as np
import pytest
import shapely.geometry

from bezpy.mt.site import Site1d, Site3d, SiteCollection
from bezpy.tl import TransmissionLine
from bezpy.tl.kernels import (
    TelluricResponse,
    apply_voltage_matrix,
    voltage_functional,
    voltage_matrix,
)

RNG = np.random.default_rng(42)


def make_site(name, lon, lat):
    """A synthetic Site3d with a smooth, site-specific impedance tensor."""
    site = Site3d(name)
    site.longitude, site.latitude = lon, lat
    site.periods = np.logspace(0.5, 4.3, 25)
    logt = np.log10(site.periods)
    # Smooth log-period variation, different per component and per site
    phases = RNG.uniform(0, 2 * np.pi, 4)
    amps = RNG.uniform(0.5, 2.0, 4)
    site.Z = np.array(
        [
            a * (1 + 0.3 * np.sin(logt + p)) * np.exp(1j * (0.5 + 0.2 * np.cos(logt + p)))
            for a, p in zip(amps, phases)
        ]
    )
    site.Z_var = None
    return site


@pytest.fixture(scope="module")
def site_grid():
    """5x5 grid of synthetic MT sites covering the test lines."""
    lons, lats = np.meshgrid(np.linspace(-110, -100, 5), np.linspace(35, 45, 5))
    return [
        make_site(f"S{i}", lon, lat) for i, (lon, lat) in enumerate(zip(lons.ravel(), lats.ravel()))
    ]


@pytest.fixture(scope="module")
def site_xys(site_grid):
    return np.array([[s.latitude, s.longitude] for s in site_grid])


def make_lines():
    """A few multi-vertex lines inside the site grid's convex hull."""
    coords = [
        [(-108.0, 37.0), (-107.2, 37.9), (-106.1, 38.3), (-105.0, 39.5)],
        [(-104.0, 42.0), (-103.1, 41.2), (-102.5, 40.1)],
        [(-109.0, 43.5), (-107.5, 43.0), (-106.0, 42.5), (-104.5, 42.0), (-103.0, 41.5)],
    ]
    return [TransmissionLine(shapely.geometry.LineString(c)) for c in coords]


def test_voltage_functional_matches_nn(site_grid, site_xys):
    E = RNG.standard_normal((50, len(site_grid), 2))
    for line in make_lines():
        line.set_nearest_sites(site_xys)
        L, valid = voltage_functional(line, len(site_grid), how="nn")
        assert valid
        expected = line.calc_voltages(E, how="nn")
        np.testing.assert_allclose(np.sum(E * L, axis=(1, 2)), expected, rtol=1e-12)


def test_voltage_functional_matches_delaunay(site_grid, site_xys):
    E = RNG.standard_normal((50, len(site_grid), 2))
    for line in make_lines():
        line.set_delaunay_weights(site_xys)
        L, valid = voltage_functional(line, len(site_grid), how="delaunay")
        assert valid
        expected = line.calc_voltages(E, how="delaunay")
        np.testing.assert_allclose(np.sum(E * L, axis=(1, 2)), expected, rtol=1e-10)


def test_voltage_functional_matches_1d():
    line = TransmissionLine(shapely.geometry.LineString([(0.5, 0.5), (1.5, 1.5), (3.0, 3.0)]))
    polys = [shapely.geometry.box(0, 0, 2, 2), shapely.geometry.box(2, 2, 5, 5)]
    line.set_1d_regions(polys, default_region=3)
    n_regions = 4
    E = RNG.standard_normal((20, n_regions, 2))
    L, valid = voltage_functional(line, n_regions, how="1d")
    assert valid
    expected = line.calc_voltages(E, how="1d")
    np.testing.assert_allclose(np.sum(E * L, axis=(1, 2)), expected, rtol=1e-12)


def test_voltage_functional_out_of_hull_flags_invalid(site_grid, site_xys):
    # A line starting outside the site grid: calc_voltages returns all-NaN.
    # (The out-of-hull vertex must not be the final one -- the last vertex's
    # weights are dropped by the [:-1] segment convention.)
    line = TransmissionLine(shapely.geometry.LineString([(-120.0, 37.0), (-108.0, 37.0)]))
    line.set_delaunay_weights(site_xys)
    _, valid = voltage_functional(line, len(site_grid), how="delaunay")
    assert not valid
    E = RNG.standard_normal((5, len(site_grid), 2))
    assert np.all(np.isnan(line.calc_voltages(E, how="delaunay")))


def test_apply_voltage_matrix(site_grid, site_xys):
    lines = make_lines()
    for line in lines:
        line.set_delaunay_weights(site_xys)
    L, valid = voltage_matrix(lines, len(site_grid), how="delaunay")
    E = RNG.standard_normal((30, len(site_grid), 2))
    V = apply_voltage_matrix(E, L, valid)
    assert V.shape == (30, len(lines))
    for i, line in enumerate(lines):
        np.testing.assert_allclose(V[:, i], line.calc_voltages(E, how="delaunay"), rtol=1e-10)


def current_pipeline_voltages(lines, sites, mag_x, mag_y, dt, how="delaunay"):
    """The existing bezpy chain: per-site FFT convolution + E interpolation."""
    ex, ey = SiteCollection(sites).convolve_fft(mag_x, mag_y, dt=dt)
    E = np.stack([ex.T, ey.T], axis=-1)  # (ntimes, nsites, 2)
    return np.column_stack([line.calc_voltages(E, how=how) for line in lines])


def test_telluric_exact_mode_matches_pipeline(site_grid, site_xys):
    lines = make_lines()
    for line in lines:
        line.set_delaunay_weights(site_xys)
    mag_x = RNG.standard_normal(2000)
    mag_y = RNG.standard_normal(2000)

    expected = current_pipeline_voltages(lines, site_grid, mag_x, mag_y, dt=60)
    tr = TelluricResponse.from_lines(lines, site_grid, how="delaunay")
    V = tr.calc_voltages(mag_x, mag_y, dt=60)

    scale = np.max(np.abs(expected))
    np.testing.assert_allclose(V, expected, atol=1e-9 * scale)


def test_telluric_exact_mode_matches_pipeline_nn(site_grid, site_xys):
    lines = make_lines()
    for line in lines:
        line.set_nearest_sites(site_xys)
    mag_x = RNG.standard_normal(1000)
    mag_y = RNG.standard_normal(1000)

    expected = current_pipeline_voltages(lines, site_grid, mag_x, mag_y, dt=60, how="nn")
    tr = TelluricResponse.from_lines(lines, site_grid, how="nn")
    V = tr.calc_voltages(mag_x, mag_y, dt=60)

    scale = np.max(np.abs(expected))
    np.testing.assert_allclose(V, expected, atol=1e-9 * scale)


def test_telluric_master_grid_close_to_exact(site_grid, site_xys):
    lines = make_lines()
    mag_x = RNG.standard_normal(2000)
    mag_y = RNG.standard_normal(2000)

    exact = TelluricResponse.from_lines(lines, site_grid, how="delaunay")
    V_exact = exact.calc_voltages(mag_x, mag_y, dt=60)

    # Master grid at the sites' own periods: the only differences are the
    # spline flavor (CubicSpline on tau vs UnivariateSpline per site) and
    # band-edge handling.
    master = TelluricResponse(exact.L, site_grid, valid=exact.valid, periods=site_grid[0].periods)
    V_master = master.calc_voltages(mag_x, mag_y, dt=60)

    scale = np.sqrt(np.mean(V_exact**2))
    err = np.sqrt(np.mean((V_master - V_exact) ** 2))
    assert err < 0.02 * scale


def test_telluric_invalid_line_is_nan(site_grid, site_xys):
    lines = make_lines() + [
        TransmissionLine(shapely.geometry.LineString([(-120.0, 37.0), (-108.0, 37.0)]))
    ]
    tr = TelluricResponse.from_lines(lines, site_grid, how="delaunay")
    V = tr.calc_voltages(RNG.standard_normal(500), RNG.standard_normal(500), dt=60)
    assert np.all(np.isnan(V[:, -1]))
    assert not np.any(np.isnan(V[:, :-1]))


def test_telluric_line_chunking_consistent(site_grid, site_xys):
    lines = make_lines()
    mag_x = RNG.standard_normal(1000)
    mag_y = RNG.standard_normal(1000)
    tr = TelluricResponse.from_lines(lines, site_grid, how="delaunay")
    V1 = tr.calc_voltages(mag_x, mag_y, dt=60, line_chunk=1)
    V2 = tr.calc_voltages(mag_x, mag_y, dt=60, line_chunk=1000)
    np.testing.assert_allclose(V1, V2, rtol=1e-12)


# ---------------------------------------------------------------------
# Line resampling (build-time quadrature refinement)
# ---------------------------------------------------------------------


def test_resampled_geometry():
    line = make_lines()[0]
    dense = line.resampled(1.0)

    # More vertices, same endpoints, same total length (great-circle
    # subdivision preserves arc length), and no segment over the limit.
    assert dense.npts > line.npts
    np.testing.assert_allclose(dense.pts[0], line.pts[0], atol=1e-12)
    np.testing.assert_allclose(dense.pts[-1], line.pts[-1], atol=1e-12)
    np.testing.assert_allclose(dense.length, line.length, rtol=1e-9)
    seg_km = np.linalg.norm(dense.dl, axis=1)
    assert np.all(seg_km <= 1.0 * (1 + 1e-6))


def test_resampled_keeps_metadata():
    line = TransmissionLine(make_lines()[0].pts, metadata={"voltage": 345})
    assert line.resampled(5.0).metadata == {"voltage": 345}


def test_resampled_constant_field_voltage(site_grid, site_xys):
    # A field that is identical at every site interpolates to a constant, so
    # the line integral must not depend on the quadrature.
    E = np.tile(RNG.standard_normal((10, 1, 2)), (1, len(site_grid), 1))
    for line in make_lines():
        dense = line.resampled(1.0)
        line.set_delaunay_weights(site_xys)
        dense.set_delaunay_weights(site_xys)
        # The (dl_lat, dl_lon) decomposition of the sub-segments does not
        # telescope exactly, so allow the ~1e-5 quadrature difference
        # (relative to the voltage scale -- elements can be near zero).
        expected = line.calc_voltages(E, how="delaunay")
        np.testing.assert_allclose(
            dense.calc_voltages(E, how="delaunay"), expected, atol=1e-4 * np.max(np.abs(expected))
        )


def test_from_lines_resample_km(site_grid, site_xys):
    lines = make_lines()
    mag_x, mag_y = RNG.standard_normal(500), RNG.standard_normal(500)
    coarse = TelluricResponse.from_lines(lines, site_grid, how="delaunay")
    dense = TelluricResponse.from_lines(lines, site_grid, how="delaunay", resample_km=1.0)
    # The input lines must be untouched, and the refined quadrature should
    # give voltages close to (but not identical with) the coarse ones.
    assert all(line.npts <= 5 for line in lines)
    V_coarse = coarse.calc_voltages(mag_x, mag_y, dt=60)
    V_dense = dense.calc_voltages(mag_x, mag_y, dt=60)
    scale = np.sqrt(np.mean(V_coarse**2))
    assert np.sqrt(np.mean((V_dense - V_coarse) ** 2)) < 0.1 * scale


# ---------------------------------------------------------------------
# FFT-grid tau caching
# ---------------------------------------------------------------------


def test_tau_fft_grid_cache_reused(site_grid, site_xys, monkeypatch):
    lines = make_lines()
    tr = TelluricResponse.from_lines(lines, site_grid, how="delaunay", periods=site_grid[0].periods)
    mag_x, mag_y = RNG.standard_normal(1000), RNG.standard_normal(1000)
    V1 = tr.calc_voltages(mag_x, mag_y, dt=60)
    assert tr._tau_grid_key is not None

    # A second equal-length chunk must not rebuild tau at all.
    def boom(*args, **kwargs):
        raise AssertionError("tau was rebuilt despite an identical FFT grid")

    monkeypatch.setattr(tr, "calc_tau", boom)
    V2 = tr.calc_voltages(mag_x, mag_y, dt=60)
    np.testing.assert_array_equal(V1, V2)
    monkeypatch.undo()

    # A different chunk length (or dt) invalidates the cache.
    tr.calc_voltages(mag_x[:500], mag_y[:500], dt=60)
    assert tr._tau_grid_key == (1024, 60)


# ---------------------------------------------------------------------
# netCDF persistence of master-grid responses
# ---------------------------------------------------------------------


def test_netcdf_roundtrip(site_grid, site_xys, tmp_path):
    pytest.importorskip("xarray")
    lines = make_lines() + [
        TransmissionLine(shapely.geometry.LineString([(-120.0, 37.0), (-108.0, 37.0)]))
    ]  # one invalid line
    tr = TelluricResponse.from_lines(
        lines, site_grid, how="delaunay", periods=site_grid[0].periods, extrapolate="asymptotic"
    )
    path = tmp_path / "tau.nc"
    tr.to_netcdf(path)
    restored = TelluricResponse.from_netcdf(path)

    assert restored.nlines == tr.nlines
    assert restored.extrapolate == "asymptotic"
    np.testing.assert_array_equal(restored.valid, tr.valid)
    freqs = np.fft.rfftfreq(4096, d=60)
    np.testing.assert_allclose(restored.calc_tau(freqs), tr.calc_tau(freqs))

    mag_x, mag_y = RNG.standard_normal(1000), RNG.standard_normal(1000)
    np.testing.assert_allclose(
        restored.calc_voltages(mag_x, mag_y, dt=60), tr.calc_voltages(mag_x, mag_y, dt=60)
    )


def test_netcdf_requires_master_grid(site_grid, tmp_path):
    tr = TelluricResponse.from_lines(make_lines(), site_grid, how="delaunay")
    with pytest.raises(ValueError, match="master-grid"):
        tr.to_netcdf(tmp_path / "tau.nc")


# ---------------------------------------------------------------------
# D6: band-edge extrapolation
# ---------------------------------------------------------------------


def make_halfspace_site3d(resistivity=100.0, periods=None):
    """A Site3d holding half-space impedances over a truncated band, so the
    analytic continuation outside the band is known exactly."""
    reference = Site1d("halfspace", [1000.0], [resistivity])
    site = Site3d("HS")
    site.longitude, site.latitude = -105.0, 40.0
    site.periods = np.logspace(1, 3, 15) if periods is None else periods
    site.Z = reference.calcZ(1.0 / site.periods)
    site.Z_var = None
    return site, reference


def test_calcz_asymptotic_matches_halfspace():
    site, reference = make_halfspace_site3d()
    # Frequencies well outside the measured band on both sides
    freqs = np.array([1e-4, 3e-4, 0.3, 1.0])
    assert np.all(np.abs(site.calcZ(freqs, extrapolate="zeros")) == 0.0)
    Z_asym = site.calcZ(freqs, extrapolate="asymptotic")
    # For a half space |Z| ~ sqrt(f) with constant 45 deg phase, so the
    # anchored asymptote reproduces the analytic values.
    np.testing.assert_allclose(Z_asym, reference.calcZ(freqs), rtol=1e-6)


def test_calcz_asymptotic_zero_at_dc():
    site, _ = make_halfspace_site3d()
    Z = site.calcZ(np.array([0.0, 1e-5]), extrapolate="asymptotic")
    assert np.all(Z[:, 0] == 0.0)


def test_calcz_cache_keyed_on_extrapolate():
    site, _ = make_halfspace_site3d()
    freqs = np.array([1.0, 2.0])
    Z_asym = site.calcZ(freqs, extrapolate="asymptotic").copy()
    Z_zero = site.calcZ(freqs, extrapolate="zeros").copy()
    assert np.any(Z_asym != Z_zero)
    np.testing.assert_array_equal(site.calcZ(freqs, extrapolate="asymptotic"), Z_asym)


def test_telluric_extrapolation_recovers_short_periods(site_grid, site_xys):
    # The D6 comparison: with a band-stop, out-of-band signal content is
    # truncated from the voltages; the asymptotic continuation keeps it.
    lines = make_lines()
    t = np.arange(2000) * 60.0
    period_in, period_out = 3000.0, 60.0  # site band is ~3-20000 s
    mag_x = np.sin(2 * np.pi * t / period_in) + np.sin(2 * np.pi * t / period_out)
    mag_y = np.zeros_like(t)

    V = {}
    for extrapolate in ("zeros", "asymptotic"):
        tr = TelluricResponse.from_lines(lines, site_grid, how="delaunay", extrapolate=extrapolate)
        V[extrapolate] = tr.calc_voltages(mag_x, mag_y, dt=60)

    scale = np.sqrt(np.mean(V["zeros"] ** 2))
    diff = np.sqrt(np.mean((V["asymptotic"] - V["zeros"]) ** 2))
    assert diff > 0.01 * scale


def test_telluric_master_grid_asymptotic_close_to_exact(site_grid, site_xys):
    lines = make_lines()
    mag_x = RNG.standard_normal(2000)
    mag_y = RNG.standard_normal(2000)

    exact = TelluricResponse.from_lines(lines, site_grid, how="delaunay", extrapolate="asymptotic")
    master = TelluricResponse(
        exact.L,
        site_grid,
        valid=exact.valid,
        periods=site_grid[0].periods,
        extrapolate="asymptotic",
    )
    V_exact = exact.calc_voltages(mag_x, mag_y, dt=60)
    V_master = master.calc_voltages(mag_x, mag_y, dt=60)
    scale = np.sqrt(np.mean(V_exact**2))
    assert np.sqrt(np.mean((V_master - V_exact) ** 2)) < 0.05 * scale


def test_telluric_response_rejects_nan_input():
    """NaN B must raise, not silently produce all-NaN voltages (A3)."""
    sites = []
    for lat, lon in [(39.0, -105.0), (41.0, -105.0), (40.0, -103.0), (40.0, -107.0)]:
        site = Site1d(f"s{lat}_{lon}", [20e3, 1e9], [100.0, 1000.0])
        site.latitude, site.longitude = lat, lon
        sites.append(site)
    line = TransmissionLine(np.array([[-105.5, 39.5], [-104.0, 40.5]]))
    response = TelluricResponse.from_lines([line], sites, how="nn")
    bad = np.ones(256)
    bad[10] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        response.calc_voltages(bad, np.ones(256))
    with pytest.raises(ValueError, match="same shape"):
        response.calc_voltages(np.ones(256), np.ones(128))


# ---------------------------------------------------------------------
# Multi-input kernels: tau gains an obs axis via a SECS transfer matrix
# (constant observatory-to-site SECS transfer matrix)
# ---------------------------------------------------------------------


def test_telluric_multi_obs_identity_matches_single_input(site_grid, site_xys):
    """One observatory (identity M)
    must reproduce the classic single-input path to rounding error."""
    lines = make_lines()
    mag_x = RNG.standard_normal(1000)
    mag_y = RNG.standard_normal(1000)
    M = np.tile(np.eye(2), (len(site_grid), 1, 1, 1))  # (nsites, 1, 2, 2)

    classic = TelluricResponse.from_lines(lines, site_grid, how="delaunay")
    multi = TelluricResponse.from_lines(lines, site_grid, how="delaunay", M=M)
    assert multi.n_obs == 1

    V_classic = classic.calc_voltages(mag_x, mag_y, dt=60)
    V_multi = multi.calc_voltages(mag_x[:, None], mag_y[:, None], dt=60)
    np.testing.assert_allclose(V_multi, V_classic, atol=1e-9 * np.max(np.abs(V_classic)))


def test_telluric_multi_obs_identity_matches_single_input_master_grid(site_grid, site_xys):
    lines = make_lines()
    mag_x = RNG.standard_normal(1000)
    mag_y = RNG.standard_normal(1000)
    M = np.tile(np.eye(2), (len(site_grid), 1, 1, 1))
    periods = site_grid[0].periods

    classic = TelluricResponse.from_lines(lines, site_grid, how="delaunay", periods=periods)
    multi = TelluricResponse.from_lines(lines, site_grid, how="delaunay", periods=periods, M=M)
    V_classic = classic.calc_voltages(mag_x, mag_y, dt=60)
    V_multi = multi.calc_voltages(mag_x[:, None], mag_y[:, None], dt=60)
    np.testing.assert_allclose(V_multi, V_classic, atol=1e-9 * np.max(np.abs(V_classic)))


def test_telluric_multi_obs_matches_secs_plus_single_input(site_grid, site_xys):
    """Independent proof of the _build_tau einsum: compose SECSInterpolator
    .predict() with the existing single-input kernel per sample and check the
    multi-input kernel reproduces it, without sharing any code path."""
    pytest.importorskip("pysecs")
    from bezpy.mag import SECSInterpolator

    lines = make_lines()
    for line in lines:
        line.set_delaunay_weights(site_xys)
    obs_latlons = np.array([[36.0, -109.0], [44.0, -109.0], [40.0, -99.0], [40.0, -101.0]])
    ntimes = 500
    mag_x = RNG.standard_normal((ntimes, len(obs_latlons))) * 20
    mag_y = RNG.standard_normal((ntimes, len(obs_latlons))) * 20
    B_obs = np.stack([mag_x, mag_y], axis=-1)
    secs = SECSInterpolator().fit(B_obs, obs_latlons=obs_latlons)

    B_pred = secs.predict(site_xys)[:, :, :2]
    ex, ey = SiteCollection(site_grid).convolve_fft(B_pred[:, :, 0], B_pred[:, :, 1], dt=60)
    E = np.stack([ex.T, ey.T], axis=-1)
    L, valid = voltage_matrix(lines, len(site_grid), how="delaunay")
    V_ref = apply_voltage_matrix(E, L, valid)

    multi = TelluricResponse.from_lines(lines, site_grid, how="delaunay", secs=secs)
    assert multi.n_obs == len(obs_latlons)
    assert multi.obs_names is None  # bare-array fit: no obs names available
    V_new = multi.calc_voltages(mag_x, mag_y, dt=60)

    scale = np.max(np.abs(V_ref))
    np.testing.assert_allclose(V_new, V_ref, atol=1e-8 * scale)


def test_telluric_master_grid_multi_obs(site_grid, site_xys):
    pytest.importorskip("pysecs")
    from bezpy.mag import SECSInterpolator

    lines = make_lines()
    obs_latlons = np.array([[36.0, -109.0], [44.0, -109.0], [40.0, -99.0]])
    ntimes = 1000
    mag_x = RNG.standard_normal((ntimes, len(obs_latlons)))
    mag_y = RNG.standard_normal((ntimes, len(obs_latlons)))
    B_obs = np.stack([mag_x, mag_y], axis=-1)
    secs = SECSInterpolator().fit(B_obs, obs_latlons=obs_latlons)

    exact = TelluricResponse.from_lines(lines, site_grid, how="delaunay", secs=secs)
    master = TelluricResponse.from_lines(
        lines, site_grid, how="delaunay", secs=secs, periods=site_grid[0].periods
    )
    V_exact = exact.calc_voltages(mag_x, mag_y, dt=60)
    V_master = master.calc_voltages(mag_x, mag_y, dt=60)
    scale = np.sqrt(np.mean(V_exact**2))
    assert np.sqrt(np.mean((V_master - V_exact) ** 2)) < 0.02 * scale


def test_telluric_multi_obs_shape_validation(site_grid, site_xys):
    lines = make_lines()
    M = np.tile(np.eye(2), (len(site_grid), 2, 1, 1))  # nobs=2
    multi = TelluricResponse.from_lines(lines, site_grid, how="delaunay", M=M)
    with pytest.raises(ValueError, match="observatory"):
        multi.calc_voltages(RNG.standard_normal(200), RNG.standard_normal(200), dt=60)

    classic = TelluricResponse.from_lines(lines, site_grid, how="delaunay")
    with pytest.raises(ValueError, match="single-input"):
        classic.calc_voltages(RNG.standard_normal((200, 2)), RNG.standard_normal((200, 2)), dt=60)


def test_telluric_rejects_mismatched_M_shape(site_grid):
    lines = make_lines()
    bad_M = np.tile(np.eye(2), (len(site_grid) - 1, 1, 1, 1))
    with pytest.raises(ValueError, match="site axis"):
        TelluricResponse.from_lines(lines, site_grid, how="delaunay", M=bad_M)


def test_from_lines_rejects_both_secs_and_M(site_grid):
    lines = make_lines()
    M = np.tile(np.eye(2), (len(site_grid), 1, 1, 1))
    with pytest.raises(ValueError, match="at most one"):
        TelluricResponse.from_lines(lines, site_grid, how="delaunay", secs="placeholder", M=M)


def test_netcdf_roundtrip_multi_obs(site_grid, site_xys, tmp_path):
    pytest.importorskip("xarray")
    pytest.importorskip("pysecs")
    from bezpy.mag import SECSInterpolator

    lines = make_lines()
    obs_latlons = np.array([[36.0, -109.0], [44.0, -109.0]])
    ntimes = 500
    mag_x = RNG.standard_normal((ntimes, len(obs_latlons)))
    mag_y = RNG.standard_normal((ntimes, len(obs_latlons)))
    B_obs = np.stack([mag_x, mag_y], axis=-1)
    codes = [f"OB{i}" for i in range(len(obs_latlons))]
    secs = SECSInterpolator().fit(B_obs, obs_latlons=obs_latlons)
    secs.obs_names_ = np.array(codes)

    tr = TelluricResponse.from_lines(
        lines, site_grid, how="delaunay", secs=secs, periods=site_grid[0].periods
    )
    path = tmp_path / "tau_multi.nc"
    tr.to_netcdf(path)
    restored = TelluricResponse.from_netcdf(path)

    assert restored.n_obs == tr.n_obs == len(obs_latlons)
    assert restored.obs_names == codes
    np.testing.assert_allclose(
        restored.calc_voltages(mag_x, mag_y, dt=60), tr.calc_voltages(mag_x, mag_y, dt=60)
    )


def test_netcdf_from_netcdf_old_single_input_file_still_works(site_grid, tmp_path):
    pytest.importorskip("xarray")
    tr = TelluricResponse.from_lines(
        make_lines(), site_grid, how="delaunay", periods=site_grid[0].periods
    )
    path = tmp_path / "tau_classic.nc"
    tr.to_netcdf(path)
    restored = TelluricResponse.from_netcdf(path)
    assert restored.n_obs is None
    mag_x, mag_y = RNG.standard_normal(500), RNG.standard_normal(500)
    np.testing.assert_allclose(
        restored.calc_voltages(mag_x, mag_y, dt=60), tr.calc_voltages(mag_x, mag_y, dt=60)
    )
