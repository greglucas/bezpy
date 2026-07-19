"""Tests for bezpy.pipeline.Nowcast (roadmap 2.2).

Synthetic setup: Site1d half spaces (physical impedances valid at all
frequencies, so the period band for DTIR can be chosen freely rather than
tied to a fixture's embedded band) and **broadband** multi-tone observatory
B -- many active frequencies at once, like a real storm. This matters: DTIR's
regularized inversion reconstructs broadband B much better than narrowband
B (a couple of dominant tones), even entirely within a period band that is
otherwise well-sampled relative to dt (see the accuracy note in
``Nowcast``'s docstring, and ``ROADMAP.md`` 2.2 for the measured numbers). A
narrowband synthetic would understate error at unlucky frequencies and give
a misleadingly optimistic picture -- these tests deliberately don't do that.
"""
import numpy as np
import pandas as pd
import pytest
import shapely.geometry

xr = pytest.importorskip("xarray")
pysecs = pytest.importorskip("pysecs")

from bezpy.mag.secs import SECSInterpolator  # noqa: E402
from bezpy.mt.impulse import DTIR  # noqa: E402
from bezpy.mt.site import Site1d  # noqa: E402
from bezpy.pipeline import Nowcast, calc_geoelectric_field  # noqa: E402
from bezpy.tl import TransmissionLine  # noqa: E402
from bezpy.tl.kernels import apply_voltage_matrix, voltage_matrix  # noqa: E402

RNG = np.random.default_rng(11)
DT = 60.
# A well-sampled band (>=10 samples/cycle at dt=60s -- see the Nyquist
# aliasing note in Nowcast's docstring) matching this module's Site1d sites.
PERIODS = np.logspace(np.log10(900), np.log10(18000), 20)

OBS_LATLONS = np.array([[lat, lon]
                        for lat in np.linspace(34, 46, 4)
                        for lon in np.linspace(-110, -96, 4)], dtype=float)


def make_sites(n_lat=3, n_lon=3):
    sites = []
    for i, lat in enumerate(np.linspace(36, 44, n_lat)):
        for j, lon in enumerate(np.linspace(-108, -98, n_lon)):
            site = Site1d(f"S{i}{j}", [1e9], [100. + 20.*(i + j)])
            site.latitude, site.longitude = lat, lon
            sites.append(site)
    return sites


def make_lines():
    coords = [
        [(-106.0, 38.0), (-104.8, 38.9), (-103.5, 39.8)],
        [(-104.0, 42.0), (-102.9, 41.0)],
    ]
    return [TransmissionLine(shapely.geometry.LineString(c)) for c in coords]


def broadband_series(ntimes, dt, rng, periods_pool=None, ntones=15):
    """A sum of many sinusoids across a wide period range -- storm-like."""
    if periods_pool is None:
        periods_pool = np.logspace(np.log10(600), np.log10(20000), 40)
    t = np.arange(ntimes) * dt
    chosen = rng.choice(periods_pool, size=ntones, replace=False)
    b = np.zeros(ntimes)
    for p in chosen:
        b += rng.uniform(0.3, 1.0) * np.sin(2*np.pi*t/p + rng.uniform(0, 2*np.pi))
    return b


def make_broadband_b_obs(ntimes=4000, dt=DT, start="2024-05-10", rng=None):
    rng = rng if rng is not None else RNG
    nobs = len(OBS_LATLONS)
    bx = np.stack([broadband_series(ntimes, dt, rng) for _ in range(nobs)], axis=1)
    by = np.stack([broadband_series(ntimes, dt, rng) for _ in range(nobs)], axis=1)
    times = pd.date_range(start, periods=ntimes, freq=f"{int(dt)}s")
    return xr.Dataset(
        {"Bx": (("time", "obs"), bx), "By": (("time", "obs"), by),
         "Bz": (("time", "obs"), np.zeros_like(bx))},
        coords={"time": times, "obs": [f"OB{i}" for i in range(nobs)],
                "latitude": ("obs", OBS_LATLONS[:, 0]),
                "longitude": ("obs", OBS_LATLONS[:, 1])})


def feed_nowcast(nc, b_obs, chunk):
    """Feed a Dataset through update() in fixed-size chunks; concatenate
    the newly-finalized results into one Dataset."""
    ntimes = b_obs.sizes["time"]
    results = []
    for i0 in range(0, ntimes, chunk):
        result = nc.update(b_obs.isel(time=slice(i0, min(i0 + chunk, ntimes))))
        if result.sizes["time"]:
            results.append(result)
    return xr.concat(results, dim="time")


# ---------------------------------------------------------------------
# 1. Pure DTIR reconstruction fidelity (no buffering, no SECS)
# ---------------------------------------------------------------------

def test_dtir_vs_fft_reconstruction_tolerance():
    site = Site1d("hs", [1e9], [100.0])
    dt = DT
    nmax = max(1, int(np.ceil(2.0 * PERIODS.max() / dt)))
    nmin = -max(1, int(np.ceil(0.1 * PERIODS.min() / dt)))
    dtir = DTIR(dt=dt, nmin=nmin, nmax=nmax, decay_factor=1e-2)
    dtir.calculate(periods=PERIODS, Z=site.calcZ(1. / PERIODS))

    ntimes = nmax * 6
    rng = np.random.default_rng(21)
    bx = broadband_series(ntimes, dt, rng)
    by = broadband_series(ntimes, dt, rng)
    ex_dtir, ey_dtir = dtir.convolve(bx, by)
    ex_fft, ey_fft = site.convolve_fft(bx, by, dt=dt)

    max_n = max(abs(nmin), nmax)
    interior = slice(max_n, ntimes - max_n)
    rms = np.sqrt(np.mean((ex_dtir[interior] - ex_fft[interior])**2))
    scale = np.sqrt(np.mean(ex_fft[interior]**2))
    assert rms / scale < 0.1  # measured ~0.4-1% for this broadband signal


# ---------------------------------------------------------------------
# 2. Pure buffering correctness (bit-identical to a whole-array DTIR call)
# ---------------------------------------------------------------------

def test_nowcast_buffer_matches_whole_array_dtir():
    """The chunked append/finalize/trim algorithm inside Nowcast.update()
    must reproduce a single whole-array dtir.convolve() call: both are
    direct-form convolution (no FFT), so agreement should be near machine
    precision, not merely "small"."""
    site = Site1d("hs", [1e9], [100.0])
    dt = DT
    nmax = max(1, int(np.ceil(2.0 * PERIODS.max() / dt)))
    nmin = -max(1, int(np.ceil(0.1 * PERIODS.min() / dt)))
    dtir = DTIR(dt=dt, nmin=nmin, nmax=nmax, decay_factor=1e-2)
    dtir.calculate(periods=PERIODS, Z=site.calcZ(1. / PERIODS))

    ntimes = 2000
    rng = np.random.default_rng(22)
    bx = broadband_series(ntimes, dt, rng)
    by = broadband_series(ntimes, dt, rng)
    ex_full, ey_full = dtir.convolve(bx, by)

    warmup = 2*max(nmax, abs(nmin)) + 1
    buf_bx = np.empty(0)
    buf_by = np.empty(0)
    buffer_start = 0
    last_finalized_abs = -1
    ex_stream = np.full(ntimes, np.nan)
    chunk = 37  # not a divisor of ntimes
    i = 0
    while i < ntimes:
        j = min(i + chunk, ntimes)
        buf_bx = np.concatenate([buf_bx, bx[i:j]])
        buf_by = np.concatenate([buf_by, by[i:j]])
        buf_len = len(buf_bx)
        if buf_len >= warmup:
            prev_local = last_finalized_abs - buffer_start + 1
            new_last_local = buf_len - 1 - abs(nmin)
            if new_last_local >= prev_local:
                exf, _ = dtir.convolve(buf_bx, buf_by)
                ex_stream[buffer_start + prev_local:buffer_start + new_last_local + 1] = \
                    exf[prev_local:new_last_local + 1]
                last_finalized_abs = buffer_start + new_last_local
            target_start_local = (last_finalized_abs + 1 - nmax) - buffer_start
            keep_from = max(0, min(target_start_local, buf_len - warmup))
            if keep_from > 0:
                buf_bx, buf_by = buf_bx[keep_from:], buf_by[keep_from:]
                buffer_start += keep_from
        i = j

    valid = ~np.isnan(ex_stream)
    assert valid.sum() > ntimes - warmup - 5  # nearly everything finalizes
    np.testing.assert_allclose(ex_stream[valid], ex_full[valid], atol=1e-9)


# ---------------------------------------------------------------------
# 3. The literal roadmap acceptance test
# ---------------------------------------------------------------------

def test_nowcast_matches_calc_geoelectric_field_interior():
    sites = make_sites()
    b_obs = make_broadband_b_obs(ntimes=4000, rng=np.random.default_rng(31))

    e_batch = calc_geoelectric_field(b_obs, sites, interp="secs", pad="6h",
                                     detrend=None, taper=None)

    nc = Nowcast(sites, SECSInterpolator(), dt=DT, periods=PERIODS)
    result = feed_nowcast(nc, b_obs, chunk=41)

    common = np.intersect1d(result["time"].values, e_batch["time"].values)
    assert len(common) > 2000  # a real overlap, not an edge-case sliver
    s1 = result.sel(time=common)["Ex"].values
    s2 = e_batch.sel(time=common)["Ex"].values
    rms = np.sqrt(np.mean((s1 - s2)**2))
    scale = np.sqrt(np.mean(s2**2))
    # Measured ~5% for this broadband signal (DTIR reconstruction fidelity,
    # see test 1) plus headroom for the causal/acausal taper difference and
    # DTIR having no extrapolate= band-edge continuation.
    assert rms / scale < 0.15


# ---------------------------------------------------------------------
# 4. SECS fast path vs. fallback
# ---------------------------------------------------------------------

def test_nowcast_secs_fast_path_matches_fallback():
    sites = make_sites()
    b_obs = make_broadband_b_obs(ntimes=2000, rng=np.random.default_rng(41))

    fast = Nowcast(sites, SECSInterpolator(), dt=DT, periods=PERIODS)
    slow = Nowcast(sites, SECSInterpolator(), dt=DT, periods=PERIODS,
                  force_slow_secs=True)
    r_fast = feed_nowcast(fast, b_obs, chunk=37)
    r_slow = feed_nowcast(slow, b_obs, chunk=37)

    assert fast._M is not None
    assert slow._M is None
    common = np.intersect1d(r_fast["time"].values, r_slow["time"].values)
    np.testing.assert_allclose(r_fast.sel(time=common)["Ex"].values,
                               r_slow.sel(time=common)["Ex"].values, rtol=1e-8)


# ---------------------------------------------------------------------
# 5. Chunk-size invariance
# ---------------------------------------------------------------------

def test_nowcast_chunk_size_invariance():
    sites = make_sites()
    b_obs = make_broadband_b_obs(ntimes=1500, rng=np.random.default_rng(51))

    nc1 = Nowcast(sites, SECSInterpolator(), dt=DT, periods=PERIODS)
    nc100 = Nowcast(sites, SECSInterpolator(), dt=DT, periods=PERIODS)
    r1 = feed_nowcast(nc1, b_obs, chunk=1)
    r100 = feed_nowcast(nc100, b_obs, chunk=100)

    common = np.intersect1d(r1["time"].values, r100["time"].values)
    assert len(common) > 500
    # DTIR convolution is direct-form, not FFT-padding-sensitive the way
    # SiteCollection.convolve_fft is -- near machine precision is the right
    # bar here, tighter than test_scan_chunk_invariance's 5e-3.
    np.testing.assert_allclose(r1.sel(time=common)["Ex"].values,
                               r100.sel(time=common)["Ex"].values, atol=1e-9)


# ---------------------------------------------------------------------
# 6. Early calls return empty, not an error
# ---------------------------------------------------------------------

def test_nowcast_early_calls_return_empty_not_error():
    sites = make_sites()
    nc = Nowcast(sites, SECSInterpolator(), dt=DT, periods=PERIODS)
    b_obs = make_broadband_b_obs(ntimes=nc.warmup_samples + 50,
                                 rng=np.random.default_rng(61))

    # Feed up to (but not including) warmup_samples in small steps: every
    # call in this phase must return empty, not raise.
    fed, step = 0, 25
    while fed + step <= nc.warmup_samples - 1:
        result = nc.update(b_obs.isel(time=slice(fed, fed + step)))
        assert result.sizes["time"] == 0
        fed += step

    # The next update crosses warmup_samples: finalization starts, and it
    # is retroactive to absolute index 0.
    result = nc.update(b_obs.isel(time=slice(fed, nc.warmup_samples)))
    assert result.sizes["time"] > 0
    assert result["time"].values[0] == b_obs["time"].values[0]


# ---------------------------------------------------------------------
# 7. Kalman fallback: never engages the fast path
# ---------------------------------------------------------------------

def test_nowcast_kalman_fallback_used():
    sites = make_sites()
    secs = SECSInterpolator(method="kalman", smoother=False)
    nc = Nowcast(sites, secs, dt=DT, periods=PERIODS)
    b_obs = make_broadband_b_obs(ntimes=1800, rng=np.random.default_rng(71))
    result = feed_nowcast(nc, b_obs, chunk=100)

    assert nc._M is None
    assert result.sizes["time"] > 0
    assert np.all(np.isfinite(result["Ex"].values))


# ---------------------------------------------------------------------
# 8. Consistency validation
# ---------------------------------------------------------------------

def test_nowcast_rejects_obs_set_change():
    sites = make_sites()
    nc = Nowcast(sites, SECSInterpolator(), dt=DT, periods=PERIODS)
    b_obs = make_broadband_b_obs(ntimes=200, rng=np.random.default_rng(81))
    nc.update(b_obs.isel(time=slice(0, 100)))

    other_latlons = OBS_LATLONS.copy()
    other_latlons[0, 0] += 5.
    bad = b_obs.isel(time=slice(100, 200)).assign_coords(
        latitude=("obs", other_latlons[:, 0]))
    with pytest.raises(ValueError, match="observatory set"):
        nc.update(bad)


def test_nowcast_rejects_stream_gap():
    sites = make_sites()
    nc = Nowcast(sites, SECSInterpolator(), dt=DT, periods=PERIODS)
    b_obs = make_broadband_b_obs(ntimes=200, rng=np.random.default_rng(82))
    nc.update(b_obs.isel(time=slice(0, 100)))

    gappy = b_obs.isel(time=slice(101, 200))  # skips sample 100
    with pytest.raises(ValueError, match="gap or overlap"):
        nc.update(gappy)


# ---------------------------------------------------------------------
# 9. Period-less sites need periods= or dtirs=
# ---------------------------------------------------------------------

def test_nowcast_periods_required_for_site1d():
    sites = make_sites()
    with pytest.raises(ValueError, match="S00"):
        Nowcast(sites, SECSInterpolator(), dt=DT)


# ---------------------------------------------------------------------
# 10. Voltages
# ---------------------------------------------------------------------

def test_nowcast_voltages():
    sites = make_sites()
    lines = make_lines()
    site_xys = np.array([[s.latitude, s.longitude] for s in sites])
    for line in lines:
        line.set_delaunay_weights(site_xys)
    L, valid = voltage_matrix(lines, len(sites), how="delaunay")

    nc = Nowcast(sites, SECSInterpolator(), dt=DT, periods=PERIODS, lines=lines)
    b_obs = make_broadband_b_obs(ntimes=1800, rng=np.random.default_rng(101))
    result = feed_nowcast(nc, b_obs, chunk=97)

    assert "V" in result
    E = np.stack([result["Ex"].values, result["Ey"].values], axis=-1)
    expected_V = apply_voltage_matrix(E, L, valid)
    np.testing.assert_allclose(result["V"].values, expected_V, rtol=1e-10)


# ---------------------------------------------------------------------
# 11. Undersampled-period warning
# ---------------------------------------------------------------------

def test_nowcast_warns_on_undersampled_periods():
    sites = [Site1d("s1", [1e9], [100.0])]
    sites[0].latitude, sites[0].longitude = 40., -105.
    short_periods = np.logspace(np.log10(50), np.log10(500), 10)  # << 10*dt
    with pytest.warns(UserWarning, match="Nyquist"):
        Nowcast(sites, SECSInterpolator(), dt=DT, periods=short_periods)
