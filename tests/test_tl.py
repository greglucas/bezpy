"""Tests for the transmission line module and its geospatial helpers."""
import numpy as np
import pytest
import shapely.geometry

from bezpy.tl import TransmissionLine
from bezpy.tl.utils import haversine_distance, haversine_dl


class _FakeLine:
    """Minimal stand-in for a geopandas row with a .geometry.coords attribute."""
    def __init__(self, coords):
        self.geometry = shapely.geometry.LineString(coords)


def make_line(coords):
    return TransmissionLine(_FakeLine(coords))


def test_haversine_distance_one_degree_latitude():
    # One degree of latitude is ~111.19 km for a sphere of R=6371 km.
    d = haversine_distance(0.0, 0.0, 0.0, 1.0)
    assert d == pytest.approx(111.19, abs=0.5)


def test_haversine_distance_zero():
    assert haversine_distance(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0)


def test_haversine_distance_symmetric():
    a = haversine_distance(-105.0, 40.0, -104.0, 41.0)
    b = haversine_distance(-104.0, 41.0, -105.0, 40.0)
    assert a == pytest.approx(b)


def test_haversine_dl_direction_signs():
    # Moving north-east: dl_lat > 0 (x, North) and dl_lon > 0 (y, East).
    dl = haversine_dl(np.array([-105.0]), np.array([40.0]),
                      np.array([-104.0]), np.array([41.0]))
    assert dl.shape == (1, 2)
    assert dl[0, 0] > 0  # north
    assert dl[0, 1] > 0  # east


def test_transmission_line_length():
    # A line spanning one degree of longitude at the equator.
    tl = make_line([(0.0, 0.0), (1.0, 0.0)])
    assert tl.length == pytest.approx(111.19, abs=0.5)
    assert tl.npts == 2
    assert len(tl) == 2


def test_set_nearest_sites():
    tl = make_line([(-105.0, 40.0), (-104.0, 40.5)])
    # site_xys stored as (lat, lon) pairs.
    site_xys = np.array([[40.0, -105.0], [40.5, -104.0], [50.0, -90.0]])
    tl.set_nearest_sites(site_xys)
    assert tl.nearest_sites[0] == 0
    assert tl.nearest_sites[1] == 1


def test_set_1d_regions_assigns_lowest_index():
    tl = make_line([(0.5, 0.5), (0.5, 0.6), (20.0, 20.0)])
    polys = [
        shapely.geometry.box(10, 10, 11, 11),  # 0: no match for first points
        shapely.geometry.box(0, 0, 2, 2),       # 1: contains the point
        shapely.geometry.box(0, 0, 5, 5),       # 2: also contains the point
    ]
    tl.set_1d_regions(polys, default_region=18)
    # Lowest matching polygon index wins; outside points get the default.
    assert tl.regions1d.tolist() == [1, 1, 18]


def test_set_1d_regions_default_for_all_outside():
    tl = make_line([(100.0, 100.0), (101.0, 101.0)])
    polys = [shapely.geometry.box(0, 0, 1, 1)]
    tl.set_1d_regions(polys, default_region=7)
    assert tl.regions1d.tolist() == [7, 7]


def test_calc_voltages_nn_shape():
    tl = make_line([(-105.0, 40.0), (-104.0, 40.5), (-103.0, 41.0)])
    site_xys = np.array([[40.0, -105.0], [40.5, -104.0], [41.0, -103.0]])
    tl.set_nearest_sites(site_xys)
    ntimes = 10
    E = np.random.default_rng(0).standard_normal((ntimes, len(site_xys), 2))
    v = tl.calc_voltages(E, how="nn")
    assert v.shape == (ntimes,)


def test_calc_voltages_single_timestep_returns_float():
    # Regression: a single time step must return a Python float, not crash on
    # float() of a size-1 numpy array (NumPy 2.0 behavior).
    tl = make_line([(-105.0, 40.0), (-104.0, 40.5), (-103.0, 41.0)])
    site_xys = np.array([[40.0, -105.0], [40.5, -104.0], [41.0, -103.0]])
    tl.set_nearest_sites(site_xys)
    E = np.random.default_rng(0).standard_normal((1, len(site_xys), 2))
    v = tl.calc_voltages(E, how="nn")
    assert isinstance(v, float)


def test_calc_voltages_zero_field_is_zero():
    tl = make_line([(-105.0, 40.0), (-104.0, 40.5), (-103.0, 41.0)])
    site_xys = np.array([[40.0, -105.0], [40.5, -104.0], [41.0, -103.0]])
    tl.set_nearest_sites(site_xys)
    E = np.zeros((5, len(site_xys), 2))
    v = tl.calc_voltages(E, how="nn")
    assert np.allclose(v, 0.0)


def test_calc_voltages_requires_setup():
    tl = make_line([(-105.0, 40.0), (-104.0, 40.5)])
    with pytest.raises(ValueError):
        tl.calc_voltages(np.zeros((3, 2, 2)), how="nn")


def test_calc_voltages_bad_how_raises():
    tl = make_line([(-105.0, 40.0), (-104.0, 40.5)])
    with pytest.raises(ValueError):
        tl.calc_voltages(np.zeros((3, 2, 2)), how="bogus")


def test_calc_voltages_delaunay():
    tl = make_line([(-104.5, 40.2), (-104.3, 40.4)])
    # Need enough surrounding sites for a Delaunay triangulation.
    site_xys = np.array([[40.0, -105.0], [40.5, -104.0], [41.0, -105.0],
                         [40.0, -104.0], [40.6, -104.6]])
    tl.set_delaunay_weights(site_xys)
    E = np.random.default_rng(0).standard_normal((4, len(site_xys), 2))
    v = tl.calc_voltages(E, how="delaunay")
    assert v.shape == (4,)
