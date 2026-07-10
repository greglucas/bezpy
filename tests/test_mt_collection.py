"""Tests for SiteCollection sequence behavior and the canonical dataset round trip."""
import numpy as np
import pytest

from bezpy.mt.site import (
    Site1d,
    Site3d,
    SiteCollection,
    sites_from_dataset,
    sites_to_dataset,
)

RNG = np.random.default_rng(7)


def make_site3d(name, lat, lon, periods, with_var=True, nan_entries=()):
    """A synthetic Site3d with a smooth, site-specific impedance tensor."""
    periods = np.asarray(periods, dtype=float)
    logt = np.log10(periods)
    phases = RNG.uniform(0, 2*np.pi, 4)
    amps = RNG.uniform(0.5, 2.0, 4)
    Z = np.array([a * (1 + 0.3*np.sin(logt + p)) *
                  np.exp(1j * (0.5 + 0.2*np.cos(logt + p)))
                  for a, p in zip(amps, phases)])
    for comp, iperiod in nan_entries:
        Z[comp, iperiod] = np.nan + 1j*np.nan
    Z_var = RNG.uniform(0.01, 0.1, Z.shape) if with_var else None
    return Site3d(name, latitude=lat, longitude=lon, periods=periods, Z=Z, Z_var=Z_var)


def make_3d_sites(periods_b=None):
    """Two 3d sites: one with Z_var and NaN Z entries, one without Z_var."""
    periods_a = np.logspace(0.5, 4.0, 12)
    if periods_b is None:
        periods_b = periods_a
    site_a = make_site3d("A01", 40.0, -105.0, periods_a,
                         with_var=True, nan_entries=[(1, 3), (2, 7)])
    site_b = make_site3d("B02", 42.5, -101.2, periods_b, with_var=False)
    return [site_a, site_b]


def assert_sites_equal(restored, original):
    """Exact equality of the canonical site fields (NaN positions included)."""
    assert len(restored) == len(original)
    for new, old in zip(restored, original):
        assert new.name == old.name
        assert new.latitude == old.latitude
        assert new.longitude == old.longitude
        np.testing.assert_array_equal(new.periods, old.periods)
        np.testing.assert_array_equal(new.Z, old.Z)
        if old.Z_var is None:
            assert new.Z_var is None
        else:
            np.testing.assert_array_equal(new.Z_var, old.Z_var)


# ---------------------------------------------------------------------
# Sequence protocol
# ---------------------------------------------------------------------

def test_sequence_protocol():
    sites = make_3d_sites()
    coll = SiteCollection(sites)
    assert len(coll) == 2
    assert list(coll) == sites
    assert coll[0] is sites[0]
    assert coll[-1] is sites[-1]
    sliced = coll[1:]
    assert isinstance(sliced, SiteCollection)
    assert len(sliced) == 1
    assert sliced[0] is sites[1]


def test_repr_counts_per_class():
    sites = make_3d_sites()
    assert repr(SiteCollection(sites)) == "SiteCollection(2 sites: 2 Site3d)"
    mixed = sites + [Site1d("CP", [1000.0], [100.0])]
    assert repr(SiteCollection(mixed)) == "SiteCollection(3 sites: 1 Site1d, 2 Site3d)"


def test_latitudes_longitudes():
    sites = make_3d_sites()
    sites[1].latitude = None  # a site without a location becomes NaN
    coll = SiteCollection(sites)
    np.testing.assert_array_equal(coll.latitudes, [40.0, np.nan])
    np.testing.assert_array_equal(coll.longitudes, [-105.0, -101.2])
    assert coll.latitudes.dtype == float
    assert coll.longitudes.dtype == float


# ---------------------------------------------------------------------
# Canonical dataset round trip
# ---------------------------------------------------------------------

def test_3d_dataset_convention():
    pytest.importorskip("xarray")
    sites = make_3d_sites()
    sites[1].longitude = None
    ds = sites_to_dataset(sites)

    assert ds.attrs == {"title": "bezpy MT site collection", "site_type": "3d"}
    assert list(ds["site"].values) == ["A01", "B02"]
    assert list(ds["component"].values) == ["xx", "xy", "yx", "yy"]
    assert ds["period"].attrs == {"units": "s"}
    assert np.all(np.diff(ds["period"].values) > 0)
    assert ds["Z_real"].dims == ("site", "component", "period")
    assert ds["Z_real"].attrs == {"units": "(mV/km)/nT"}
    assert np.isnan(ds["longitude"].values[1])
    # Site B has no Z_var: its rows are all NaN, site A's are all present
    assert np.isnan(ds["Z_var"].values[1]).all()
    assert not np.isnan(ds["Z_var"].values[0]).any()
    # The NaN Z entries of site A survive in both parts
    assert np.isnan(ds["Z_real"].values[0, 1, 3])
    assert np.isnan(ds["Z_imag"].values[0, 1, 3])


def test_3d_roundtrip_exact():
    pytest.importorskip("xarray")
    sites = make_3d_sites()
    sites[1].latitude = None
    restored = sites_from_dataset(sites_to_dataset(sites))
    assert all(isinstance(site, Site3d) for site in restored)
    assert_sites_equal(restored, sites)


def test_3d_no_var_at_all_omits_variable():
    pytest.importorskip("xarray")
    sites = [make_site3d(f"S{i}", 40.0 + i, -105.0, np.logspace(1, 3, 8), with_var=False)
             for i in range(2)]
    ds = sites_to_dataset(sites)
    assert "Z_var" not in ds
    assert_sites_equal(sites_from_dataset(ds), sites)


def test_union_period_grid_roundtrip():
    xr = pytest.importorskip("xarray")
    # Different, partially overlapping period grids
    sites = make_3d_sites(periods_b=np.logspace(1.0, 3.5, 9))
    ds = sites_to_dataset(sites)

    union = np.unique(np.concatenate([sites[0].periods, sites[1].periods]))
    np.testing.assert_array_equal(ds["period"].values, union)
    # Site A is NaN at site B's exclusive periods and vice versa
    only_b = ~np.isin(union, sites[0].periods)
    assert np.isnan(ds["Z_real"].values[0][:, only_b]).all()
    only_a = ~np.isin(union, sites[1].periods)
    assert np.isnan(ds["Z_real"].values[1][:, only_a]).all()

    # The dataset representation is a fixed point of the round trip
    restored = sites_from_dataset(ds)
    for site in restored:
        np.testing.assert_array_equal(site.periods, union)
    xr.testing.assert_identical(sites_to_dataset(restored), ds)


def test_1d_roundtrip_exact():
    pytest.importorskip("xarray")
    sites = [Site1d("CP", [1000.0, 5000.0, 1e5], [100.0, 20.0, 5.0]),
             Site1d("IP", [500.0, 2.5e4], [300.0, 10.0])]
    sites[0].latitude, sites[0].longitude = 38.0, -90.0
    coll = SiteCollection(sites)
    ds = coll.to_dataset()

    assert ds.attrs["site_type"] == "1d"
    assert ds["thickness"].dims == ("site", "layer")
    assert ds["thickness"].attrs == {"units": "m"}
    assert ds["resistivity"].attrs == {"units": "ohm-m"}
    # Padded to the deepest profile with NaN
    assert ds.sizes["layer"] == 3
    assert np.isnan(ds["thickness"].values[1, 2])

    restored = SiteCollection.from_dataset(ds)
    assert isinstance(restored, SiteCollection)
    for new, old in zip(restored, sites):
        assert isinstance(new, Site1d)
        assert new.name == old.name
        assert new.latitude == old.latitude
        assert new.longitude == old.longitude
        np.testing.assert_array_equal(new.thicknesses, old.thicknesses)
        np.testing.assert_array_equal(new.resistivities, old.resistivities)


def test_mixed_collection_raises():
    pytest.importorskip("xarray")
    sites = make_3d_sites() + [Site1d("CP", [1000.0], [100.0])]
    with pytest.raises(ValueError, match="mix"):
        sites_to_dataset(sites)


def test_empty_collection_raises():
    pytest.importorskip("xarray")
    with pytest.raises(ValueError, match="empty"):
        sites_to_dataset([])


def test_netcdf_roundtrip(tmp_path):
    pytest.importorskip("xarray")
    sites = make_3d_sites(periods_b=np.logspace(1.0, 3.5, 9))
    sites[1].latitude = None
    coll = SiteCollection(sites)
    path = tmp_path / "sites.nc"
    coll.to_netcdf(path)

    restored = SiteCollection.from_netcdf(path)
    assert isinstance(restored, SiteCollection)
    # The file stores the union grid, so compare against the dataset fixed point
    assert_sites_equal(list(restored), sites_from_dataset(coll.to_dataset()))


def test_convolve_fft_equivalent_after_roundtrip(rng):
    pytest.importorskip("xarray")
    sites = make_3d_sites(periods_b=np.logspace(1.0, 3.5, 9))
    coll = SiteCollection(sites)
    restored = SiteCollection.from_dataset(coll.to_dataset())

    n = 1024
    mag_x = rng.standard_normal(n)
    mag_y = rng.standard_normal(n)
    ex_before, ey_before = coll.convolve_fft(mag_x, mag_y, dt=60)
    ex_after, ey_after = restored.convolve_fft(mag_x, mag_y, dt=60)
    np.testing.assert_allclose(ex_after, ex_before)
    np.testing.assert_allclose(ey_after, ey_before)
