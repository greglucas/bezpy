"""Tests for bezpy.pipeline: B -> E -> V, continuous scan, storm scan.

Synthetic setup: a grid of Site1d half spaces (physical impedances valid at
all frequencies), observatory B from a smooth SEC source so SECS
interpolation is near-exact, and multi-vertex lines inside the site hull.
"""
import numpy as np
import pandas as pd
import pytest
import shapely.geometry

xr = pytest.importorskip("xarray")
pysecs = pytest.importorskip("pysecs")

import bezpy  # noqa: E402
from bezpy.mt.site import Site1d, SiteCollection  # noqa: E402
from bezpy.pipeline import scan, storm_scan  # noqa: E402
from bezpy.tl import TransmissionLine, filter_bbox, filter_polygon  # noqa: E402

RNG = np.random.default_rng(3)

OBS_LATLONS = np.array([[lat, lon]
                        for lat in np.linspace(34, 46, 4)
                        for lon in np.linspace(-110, -96, 4)], dtype=float)


def make_sites(n_lat=4, n_lon=4):
    sites = []
    for i, lat in enumerate(np.linspace(36, 44, n_lat)):
        for j, lon in enumerate(np.linspace(-108, -98, n_lon)):
            site = Site1d(f"S{i}{j}", [1000.], [100. + 20.*(i + j)])
            site.latitude, site.longitude = lat, lon
            sites.append(site)
    return sites


def make_lines():
    coords = [
        [(-106.0, 38.0), (-104.8, 38.9), (-103.5, 39.8)],
        [(-104.0, 42.0), (-102.9, 41.0)],
    ]
    return [TransmissionLine(shapely.geometry.LineString(c)) for c in coords]


def make_b_obs(ntimes=1440, start="2024-05-10", freq="60s", amp=1e13):
    """Observatory Dataset driven by a smooth, time-varying SEC source."""
    from bezpy.mag.secs import R_EARTH
    poles = np.array([[lat, lon, R_EARTH + 110e3]
                      for lat in np.linspace(30, 50, 4)
                      for lon in np.linspace(-115, -90, 5)])
    source = pysecs.SECS(sec_df_loc=poles)
    t = np.arange(ntimes)[:, None]
    phases = RNG.uniform(0, 2*np.pi, len(poles))
    # A couple of spectral lines well inside any band + smooth envelope
    source.sec_amps = amp * (np.sin(2*np.pi*t/600. + phases)
                             + 0.5*np.sin(2*np.pi*t/57. + 2*phases))
    B = source.predict(np.column_stack([OBS_LATLONS,
                                        np.full(len(OBS_LATLONS), R_EARTH)]))
    times = pd.date_range(start, periods=ntimes, freq=freq)
    return xr.Dataset(
        {name: (("time", "obs"), B[:, :, k]) for k, name in
         enumerate(("Bx", "By", "Bz"))},
        coords={"time": times, "obs": [f"OB{i}" for i in range(len(OBS_LATLONS))],
                "latitude": ("obs", OBS_LATLONS[:, 0]),
                "longitude": ("obs", OBS_LATLONS[:, 1])})


@pytest.fixture(scope="module")
def setup():
    return make_b_obs(), make_sites(), make_lines()


def test_calc_geoelectric_field_secs(setup):
    b_obs, sites, _ = setup
    e = bezpy.calc_geoelectric_field(b_obs, sites, interp="secs", pad="1h")
    assert set(e.data_vars) == {"Ex", "Ey"}
    assert e["Ex"].dims == ("time", "site")
    assert e.sizes["site"] == len(sites)
    assert not np.isnan(e["Ex"].values).any()
    assert "pad_left" not in e.attrs  # trimmed


def test_calc_geoelectric_field_uniform_matches_convolve(setup):
    b_obs, sites, _ = setup
    one_obs = b_obs.isel(obs=[5])
    e = bezpy.calc_geoelectric_field(one_obs, sites, interp="uniform",
                                     pad="1h", detrend=None, taper=None)
    # Manual reference: same B through SiteCollection directly
    prepared = bezpy.mag.prepare(one_obs, pad="1h", detrend=None, taper=None)
    ex, ey = SiteCollection(sites).convolve_fft(
        prepared["Bx"].values[:, 0], prepared["By"].values[:, 0], dt=60)
    trimmed = bezpy.mag.trim_pad(
        xr.Dataset({"Ex": (("time", "site"), ex.T)},
                   coords={"time": prepared["time"].values},
                   attrs=prepared.attrs))
    np.testing.assert_allclose(e["Ex"].values, trimmed["Ex"].values, rtol=1e-10)


def test_calc_geoelectric_field_validation(setup):
    b_obs, sites, _ = setup
    with pytest.raises(ValueError, match="exactly one observatory"):
        bezpy.calc_geoelectric_field(b_obs, sites, interp="uniform", pad="1h")
    with pytest.raises(ValueError, match="interp must be"):
        bezpy.calc_geoelectric_field(b_obs, sites, interp="magic", pad="1h")
    with pytest.raises(ValueError, match="not one of"):
        bezpy.calc_geoelectric_field(b_obs, sites, leave_one_out="NOPE", pad="1h")


def test_leave_one_out_changes_fit(setup):
    b_obs, sites, _ = setup
    e_all = bezpy.calc_geoelectric_field(b_obs, sites, pad="1h")
    e_loo = bezpy.calc_geoelectric_field(b_obs, sites, pad="1h",
                                         leave_one_out="OB5")
    assert not np.allclose(e_all["Ex"].values, e_loo["Ex"].values)


def test_calc_geoelectric_field_sitecollection(setup):
    """A SiteCollection passed as sites= must match the list path exactly."""
    b_obs, sites, _ = setup
    e_list = bezpy.calc_geoelectric_field(b_obs, sites, pad="1h")
    e_coll = bezpy.calc_geoelectric_field(b_obs, SiteCollection(sites), pad="1h")
    np.testing.assert_array_equal(e_coll["Ex"].values, e_list["Ex"].values)
    np.testing.assert_array_equal(e_coll["Ey"].values, e_list["Ey"].values)
    np.testing.assert_array_equal(e_coll["site"].values, e_list["site"].values)


def test_calc_voltages(setup):
    b_obs, sites, lines = setup
    e = bezpy.calc_geoelectric_field(b_obs, sites, pad="1h")
    v = bezpy.calc_voltages(lines, e)
    assert v.dims == ("time", "line")
    assert v.shape == (e.sizes["time"], len(lines))
    assert not np.isnan(v.values).any()

    # Must equal the per-line legacy path exactly
    E = np.stack([e["Ex"].values, e["Ey"].values], axis=-1)
    for i, line in enumerate(lines):
        np.testing.assert_allclose(v.values[:, i],
                                   line.calc_voltages(E, how="delaunay"),
                                   rtol=1e-10)


def test_scan_reducers(setup):
    b_obs, sites, lines = setup
    result = scan(b_obs, sites, lines=lines, chunk="6h",
                  reduce=("max", "exceedances", "quantiles"),
                  threshold={"E": 1e-9, "V": 1e-9}, pad="1h")
    assert result["E_max"].dims == ("site",)
    assert result["V_max"].dims == ("line",)
    assert (result["E_max"].values > 0).all()
    # q50 <= q99 <= max for every site
    assert (result["E_q50"].values <= result["E_q99"].values + 1e-12).all()
    assert (result["E_q99"].values <= result["E_max"].values * 1.05).all()
    assert result["E_exceedance_time"].size > 0


def test_scan_chunk_invariance(setup):
    """The D4 guarantee: chunking must not change the statistics.

    Detrending is per-chunk by construction, so it is disabled here. The
    residual chunk sensitivity is the finite-pad convolution edge effect
    (half-space responses decay slowly), which must shrink as the pad grows
    and be small at a practical pad length.
    """
    b_obs, sites, _ = setup
    kwargs = dict(reduce=("max",), detrend=None, taper=None)

    def disagreement(pad):
        fine = scan(b_obs, sites, chunk="4h", pad=pad, **kwargs)
        coarse = scan(b_obs, sites, chunk="12h", pad=pad, **kwargs)
        rel = np.abs(fine["E_max"].values - coarse["E_max"].values)
        return (rel / coarse["E_max"].values).max(), fine, coarse

    short_diff, _, _ = disagreement("2h")
    long_diff, fine, coarse = disagreement("12h")
    assert long_diff < short_diff        # pad controls the edge effect
    assert long_diff < 5e-3              # and it is small at a real pad
    np.testing.assert_array_equal(fine["E_max_time"].values,
                                  coarse["E_max_time"].values)


def test_scan_finds_index_invisible_event():
    """The D4 comparison: a local burst outside any storm window is seen by
    the continuous scan but missed by a storm-window scan."""
    b_obs = make_b_obs(ntimes=1440)
    sites = make_sites()
    # Inject a strong local burst at one observatory, mid-record, away from
    # the (synthetic) storm window at the start
    burst = b_obs.copy(deep=True)
    bump = np.zeros(1440)
    bump[700:720] = 500.  # nT excursion
    burst["Bx"].loc[{"obs": "OB5"}] = burst["Bx"].sel(obs="OB5") + \
        xr.DataArray(bump, dims="time")

    continuous = scan(burst, sites, chunk="12h", reduce=("max",), pad="1h")
    storm_windows = [(pd.Timestamp("2024-05-10 01:00"),
                      pd.Timestamp("2024-05-10 05:00"))]  # miss the burst
    storms = storm_scan(storm_windows, burst, sites, pad="1h")

    # The continuous maxima at the site nearest the burst dwarf the
    # storm-window maxima there
    site_idx = 5
    assert continuous["E_max"].values[site_idx] > \
        3 * storms["E_max"].values[0, site_idx]
    # And the continuous max occurs at the burst time
    burst_time = pd.Timestamp(continuous["E_max_time"].values[site_idx])
    assert pd.Timestamp("2024-05-10 11:00") < burst_time < \
        pd.Timestamp("2024-05-10 12:30")


def test_scan_resume(setup, tmp_path, monkeypatch):
    b_obs, sites, _ = setup
    path = tmp_path / "scan.nc"
    kwargs = dict(chunk="6h", reduce=("max",), pad="1h")
    full = scan(b_obs, sites, **kwargs)

    # Simulate a crash after the second chunk, then resume from the file
    import bezpy.pipeline as pipeline
    real = pipeline.calc_geoelectric_field
    calls = {"n": 0}

    def flaky(*args, **kw):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("interrupted")
        return real(*args, **kw)

    monkeypatch.setattr(pipeline, "calc_geoelectric_field", flaky)
    with pytest.raises(RuntimeError, match="interrupted"):
        scan(b_obs, sites, out_path=str(path), **kwargs)
    monkeypatch.undo()

    resumed = scan(b_obs, sites, out_path=str(path), **kwargs)
    np.testing.assert_allclose(resumed["E_max"].values, full["E_max"].values,
                               rtol=1e-9)
    np.testing.assert_array_equal(resumed["E_max_time"].values,
                                  full["E_max_time"].values)


def test_storm_scan(setup):
    b_obs, sites, lines = setup
    windows = [(pd.Timestamp("2024-05-10 02:00"), pd.Timestamp("2024-05-10 06:00")),
               (pd.Timestamp("2024-05-10 14:00"), pd.Timestamp("2024-05-10 20:00"))]
    catalog = pd.DataFrame(windows, columns=["start", "end"])
    result = storm_scan(catalog, b_obs, sites, lines=lines, pad="1h")
    assert result["E_max"].shape == (2, len(sites))
    assert result["V_max"].shape == (2, len(lines))
    assert (result["E_max"].values > 0).all()


def test_storm_scan_sitecollection(setup):
    b_obs, sites, _ = setup
    windows = [(pd.Timestamp("2024-05-10 02:00"),
                pd.Timestamp("2024-05-10 06:00"))]
    from_list = storm_scan(windows, b_obs, sites, pad="1h")
    from_collection = storm_scan(windows, b_obs, SiteCollection(sites), pad="1h")
    np.testing.assert_array_equal(from_collection["E_max"].values,
                                  from_list["E_max"].values)


def test_filter_bbox_and_polygon():
    lines = make_lines() + [TransmissionLine(shapely.geometry.LineString(
        [(-90., 30.), (-89., 31.)]))]
    inside = filter_bbox(lines, (-110, 35, -100, 45))
    assert len(inside) == 2
    poly = shapely.geometry.box(-91, 29, -88, 32)
    assert filter_polygon(lines, poly) == [lines[-1]]
