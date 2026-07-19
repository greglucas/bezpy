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
from bezpy.tl.network import Network  # noqa: E402

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


def make_var_sites(n_lat=4, n_lon=4, relative_std=0.1):
    """Site3d sites with Z_var (roadmap 1.2 ensemble tests need real
    impedance uncertainty; make_sites' Site1d sites have none by design)."""
    from bezpy.mt.site import Site3d
    periods = np.logspace(1, 4, 10)
    sites = []
    for i, lat in enumerate(np.linspace(36, 44, n_lat)):
        for j, lon in enumerate(np.linspace(-108, -98, n_lon)):
            Z = np.zeros((4, len(periods)), dtype=complex)
            Z[1, :] = (1. + 0.5j) * (1. + 0.1*(i + j))
            Z[2, :] = -Z[1, :]
            Z_var = (relative_std * np.abs(Z))**2 + 1e-6
            site = Site3d(f"S{i}{j}", latitude=lat, longitude=lon,
                         periods=periods, Z=Z, Z_var=Z_var)
            sites.append(site)
    return sites


def make_lines():
    coords = [
        [(-106.0, 38.0), (-104.8, 38.9), (-103.5, 39.8)],
        [(-104.0, 42.0), (-102.9, 41.0)],
    ]
    return [TransmissionLine(shapely.geometry.LineString(c)) for c in coords]


def make_network(lines):
    """One independent grounded-GSU-to-GSU loop per line, so each line
    drives exactly one transformer's neutral current (see
    test_network.py::test_toy_series_loop for the same, hand-verified
    topology)."""
    for i, line in enumerate(lines):
        line.metadata["id"] = f"branch{i}"
    substations = [{"id": f"sub{i}a", "resistance": 1.0} for i in range(len(lines))] + \
        [{"id": f"sub{i}b", "resistance": 1.0} for i in range(len(lines))]
    transformers = []
    for i in range(len(lines)):
        transformers.append({"id": f"T{i}a", "type": "gsu", "hi_bus": f"bus{i}a",
                             "lo_bus": None, "substation": f"sub{i}a",
                             "r_hi": 0.5, "r_lo": None})
        transformers.append({"id": f"T{i}b", "type": "gsu", "hi_bus": f"bus{i}b",
                             "lo_bus": None, "substation": f"sub{i}b",
                             "r_hi": 0.5, "r_lo": None})
    branches = [{"id": f"branch{i}", "from_bus": f"bus{i}a", "to_bus": f"bus{i}b",
                "resistance": 1.0} for i in range(len(lines))]
    return Network(substations, transformers, branches)


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


def test_despike_obs_recovers_field_from_spiked_observatory(setup):
    """A single injected spike in one observatory corrupts the SECS E field;
    despike_obs=True removes it and recovers the clean-input field."""
    b_obs, sites, _ = setup
    clean = bezpy.calc_geoelectric_field(b_obs, sites, interp="secs", pad="1h")

    spiked = b_obs.copy(deep=True)
    spiked["Bx"].values[700, 3] += 5000.0        # one huge sample at one obs
    corrupted = bezpy.calc_geoelectric_field(spiked, sites, interp="secs",
                                             pad="1h")
    recovered = bezpy.calc_geoelectric_field(spiked, sites, interp="secs",
                                             pad="1h", despike_obs=True)

    peak = lambda e: np.nanmax(np.abs(e["Ex"].values))  # noqa: E731
    assert peak(corrupted) > 3 * peak(clean)            # spike inflates E
    assert peak(recovered) < 1.05 * peak(clean)         # inflation removed
    # Field restored everywhere (despike masks only the one spiked instant, so
    # the tiny residual is a localized SECS gap, not the 3x corruption).
    resid = np.abs(recovered["Ex"].values - clean["Ex"].values)
    assert np.median(resid) < 1e-3 * peak(clean)


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


def test_calc_geoelectric_field_n_realizations_no_variance_is_deterministic(setup):
    """Site1d sites carry no Z_var, so n_realizations must add the dimension
    but every draw is identical (Site.perturbed's base-class no-op) and
    matches the plain point-estimate call exactly."""
    b_obs, sites, _ = setup
    point = bezpy.calc_geoelectric_field(b_obs, sites, pad="1h")
    ensemble = bezpy.calc_geoelectric_field(b_obs, sites, pad="1h",
                                            n_realizations=4, seed=0)
    assert ensemble["Ex"].dims == ("realization", "time", "site")
    assert ensemble.sizes["realization"] == 4
    for r in range(4):
        np.testing.assert_allclose(ensemble["Ex"].values[r], point["Ex"].values)


def test_calc_geoelectric_field_n_realizations_spreads_with_variance():
    """Sites with Z_var must produce a nonzero ensemble spread centered
    near the point estimate, and be reproducible under a fixed seed."""
    b_obs = make_b_obs()
    sites = make_var_sites()
    point = bezpy.calc_geoelectric_field(b_obs, sites, pad="1h")
    ensemble = bezpy.calc_geoelectric_field(b_obs, sites, pad="1h",
                                            n_realizations=8, seed=1)
    std = ensemble["Ex"].std("realization").values
    assert (std > 0).all()
    mean = ensemble["Ex"].mean("realization").values
    assert np.abs(mean - point["Ex"].values).max() < 3 * std.max()

    repeat = bezpy.calc_geoelectric_field(b_obs, sites, pad="1h",
                                          n_realizations=8, seed=1)
    np.testing.assert_array_equal(ensemble["Ex"].values, repeat["Ex"].values)


def test_calc_geoelectric_field_perturb_b_validation(setup):
    b_obs, sites, _ = setup
    with pytest.raises(ValueError, match="requires interp='secs'"):
        bezpy.calc_geoelectric_field(b_obs, sites, interp="uniform",
                                     n_realizations=3, perturb_b=True, pad="1h")
    with pytest.raises(ValueError, match="no effect without n_realizations"):
        bezpy.calc_geoelectric_field(b_obs, sites, perturb_b=True, pad="1h")


def test_calc_geoelectric_field_perturb_b_widens_spread():
    b_obs = make_b_obs()
    sites = make_var_sites()
    z_only = bezpy.calc_geoelectric_field(b_obs, sites, pad="1h",
                                          n_realizations=8, seed=2)
    z_and_b = bezpy.calc_geoelectric_field(b_obs, sites, pad="1h",
                                           n_realizations=8, seed=2,
                                           perturb_b=True)
    assert z_and_b["Ex"].std("realization").mean() > \
        z_only["Ex"].std("realization").mean()


def test_calc_geoelectric_field_return_var():
    """roadmap 1.2 phase 2: return_var=True adds Ex_var/Ey_var, zero for
    sites without Z_var, mutually exclusive with n_realizations."""
    b_obs = make_b_obs()
    sites = make_var_sites(n_lat=2, n_lon=2)
    e = bezpy.calc_geoelectric_field(b_obs, sites, pad="1h", return_var=True)
    assert {"Ex_var", "Ey_var"}.issubset(e.data_vars)
    assert (e["Ex_var"].values > 0).all()
    assert (e["Ey_var"].values > 0).all()

    e_no_var = bezpy.calc_geoelectric_field(
        make_b_obs(), make_sites(n_lat=2, n_lon=2), pad="1h", return_var=True)
    np.testing.assert_array_equal(e_no_var["Ex_var"].values, 0.)


def test_calc_voltages_return_var():
    b_obs = make_b_obs()
    sites = make_var_sites(n_lat=2, n_lon=2)
    lines = make_lines()
    e = bezpy.calc_geoelectric_field(b_obs, sites, pad="1h", return_var=True)
    v, v_var = bezpy.calc_voltages(lines, e, return_var=True)
    assert v.dims == v_var.dims == ("time", "line")
    finite = np.isfinite(v.values)
    assert finite.any()
    assert (v_var.values[finite] >= 0).all()

    e_point = bezpy.calc_geoelectric_field(b_obs, sites, pad="1h")
    with pytest.raises(ValueError, match="Ex_var/Ey_var"):
        bezpy.calc_voltages(lines, e_point, return_var=True)


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


def test_scan_n_realizations_spreads_and_matches_single_run():
    """Ensemble scan (roadmap 1.2 phase 1): the realization dimension lands
    on every reduced variable, the perturbed-sites-fixed-for-the-whole-scan
    contract holds (each realization is one internally consistent E_max
    series, not per-chunk noise), and the *_max_std spread is nonzero for
    sites with reported Z_var."""
    b_obs = make_b_obs()
    sites = make_var_sites()
    result = scan(b_obs, sites, lines=make_lines(), chunk="6h",
                  reduce=("max", "quantiles"), n_realizations=5, seed=0,
                  pad="1h")
    assert result["E_max"].dims == ("realization", "site")
    assert result["V_max"].dims == ("realization", "line")
    assert result.sizes["realization"] == 5
    assert (result["E_max_std"].values > 0).all()
    np.testing.assert_allclose(result["E_max_mean"].values,
                               result["E_max"].mean("realization").values)

    rng = np.random.default_rng(0)
    single = scan(b_obs, [site.perturbed(rng) for site in sites], chunk="6h",
                 reduce=("max",), pad="1h")
    # realization 0 of the ensemble must be exactly the scan of the same
    # rng-drawn perturbed sites (both draw with a fresh Generator(seed=0)).
    np.testing.assert_allclose(result["E_max"].isel(realization=0).values,
                               single["E_max"].values)


def test_scan_n_realizations_validation():
    b_obs, sites = make_b_obs(), make_var_sites()
    with pytest.raises(ValueError, match="exceedances"):
        scan(b_obs, sites, reduce=("exceedances",), threshold=1e-9,
            n_realizations=3, pad="1h")
    with pytest.raises(ValueError, match="out_path"):
        scan(b_obs, sites, out_path="unused.nc", n_realizations=3, pad="1h")


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


def test_scan_workers_matches_sequential(setup):
    """The D4 guarantee extends to workers: chunks are independent, so
    splitting the work across processes must not change the statistics."""
    b_obs, sites, lines = setup
    kwargs = dict(lines=lines, chunk="6h",
                  reduce=("max", "exceedances", "quantiles"),
                  threshold={"E": 1e-9, "V": 1e-9}, pad="1h")
    sequential = scan(b_obs, sites, **kwargs)
    parallel = scan(b_obs, sites, workers=2, **kwargs)

    np.testing.assert_array_equal(sequential["E_max"].values, parallel["E_max"].values)
    np.testing.assert_array_equal(sequential["E_max_time"].values,
                                  parallel["E_max_time"].values)
    np.testing.assert_array_equal(sequential["V_max"].values, parallel["V_max"].values)
    np.testing.assert_array_equal(sequential["E_q50"].values, parallel["E_q50"].values)
    np.testing.assert_array_equal(sequential["E_exceedance_value"].values,
                                  parallel["E_exceedance_value"].values)


def test_scan_workers_resume(setup, tmp_path):
    """A scan interrupted midway (sequentially) resumes correctly when the
    second half runs in parallel -- checkpoints are chunk-order-exact either
    way, so which mode finishes the job doesn't matter."""
    b_obs, sites, _ = setup
    path = tmp_path / "scan_parallel.nc"
    kwargs = dict(chunk="6h", reduce=("max",), pad="1h")
    full = scan(b_obs, sites, **kwargs)

    import bezpy.pipeline as pipeline
    real = pipeline.calc_geoelectric_field
    calls = {"n": 0}

    def flaky(*args, **kw):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("interrupted")
        return real(*args, **kw)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pipeline, "calc_geoelectric_field", flaky)
        with pytest.raises(RuntimeError, match="interrupted"):
            scan(b_obs, sites, out_path=str(path), **kwargs)

    resumed = scan(b_obs, sites, out_path=str(path), workers=2, **kwargs)
    np.testing.assert_allclose(resumed["E_max"].values, full["E_max"].values,
                               rtol=1e-9)


def test_scan_workers_invalid():
    with pytest.raises(ValueError, match="workers"):
        scan(make_b_obs(), make_sites(), workers=0)


def test_scan_network_reduces_gic(setup):
    """scan(..., network=) adds an I reducer group alongside E and V,
    reporting one current per transformer, chunk-invariant like E and V
    (D4 -- see test_scan_chunk_invariance for the same check on E)."""
    b_obs, sites, lines = setup
    network = make_network(lines)
    kwargs = dict(lines=lines, network=network, reduce=("max",),
                  detrend=None, taper=None, pad="12h")
    fine = scan(b_obs, sites, chunk="4h", **kwargs)
    coarse = scan(b_obs, sites, chunk="12h", **kwargs)

    assert fine["I_max"].dims == ("transformer",)
    assert fine.sizes["transformer"] == len(network.transformers)
    assert (fine["I_max"].values > 0).all()
    assert fine["I_max_time"].values.dtype.kind == "M"
    np.testing.assert_allclose(fine["I_max"].values, coarse["I_max"].values, rtol=5e-3)

    # The two ends of each independent series loop (see make_network) carry
    # the same current magnitude -- the same invariant test_network.py
    # checks on the EPRI-21 benchmark, now exercised end to end through scan().
    np.testing.assert_allclose(fine["I_max"].values[0::2], fine["I_max"].values[1::2],
                               rtol=1e-10)


def test_scan_network_requires_lines():
    with pytest.raises(ValueError, match="network= requires lines="):
        scan(make_b_obs(), make_sites(), network=make_network(make_lines()))


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
