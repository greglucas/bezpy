"""Tests for the precomputed voltage functionals and telluric response kernels.

The load-bearing property is *exact reassociation*: the kernels must
reproduce the existing calc_voltages / convolve_fft pipeline to rounding
error, because they perform the same linear operations in a different order.
"""
import numpy as np
import pytest
import shapely.geometry

from bezpy.mt.site import Site3d, SiteCollection
from bezpy.tl import TransmissionLine
from bezpy.tl.kernels import (TelluricResponse, apply_voltage_matrix,
                              voltage_functional, voltage_matrix)

RNG = np.random.default_rng(42)


def make_site(name, lon, lat):
    """A synthetic Site3d with a smooth, site-specific impedance tensor."""
    site = Site3d(name)
    site.longitude, site.latitude = lon, lat
    site.periods = np.logspace(0.5, 4.3, 25)
    logt = np.log10(site.periods)
    # Smooth log-period variation, different per component and per site
    phases = RNG.uniform(0, 2*np.pi, 4)
    amps = RNG.uniform(0.5, 2.0, 4)
    site.Z = np.array([a * (1 + 0.3*np.sin(logt + p)) *
                       np.exp(1j * (0.5 + 0.2*np.cos(logt + p)))
                       for a, p in zip(amps, phases)])
    site.Z_var = None
    return site


@pytest.fixture(scope="module")
def site_grid():
    """5x5 grid of synthetic MT sites covering the test lines."""
    lons, lats = np.meshgrid(np.linspace(-110, -100, 5), np.linspace(35, 45, 5))
    return [make_site(f"S{i}", lon, lat)
            for i, (lon, lat) in enumerate(zip(lons.ravel(), lats.ravel()))]


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
    line = TransmissionLine(shapely.geometry.LineString(
        [(0.5, 0.5), (1.5, 1.5), (3.0, 3.0)]))
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
    line = TransmissionLine(shapely.geometry.LineString(
        [(-120.0, 37.0), (-108.0, 37.0)]))
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
        np.testing.assert_allclose(V[:, i], line.calc_voltages(E, how="delaunay"),
                                   rtol=1e-10)


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

    expected = current_pipeline_voltages(lines, site_grid, mag_x, mag_y,
                                         dt=60, how="nn")
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
    master = TelluricResponse(exact.L, site_grid, valid=exact.valid,
                              periods=site_grid[0].periods)
    V_master = master.calc_voltages(mag_x, mag_y, dt=60)

    scale = np.sqrt(np.mean(V_exact**2))
    err = np.sqrt(np.mean((V_master - V_exact)**2))
    assert err < 0.02 * scale


def test_telluric_invalid_line_is_nan(site_grid, site_xys):
    lines = make_lines() + [TransmissionLine(shapely.geometry.LineString(
        [(-120.0, 37.0), (-108.0, 37.0)]))]
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
