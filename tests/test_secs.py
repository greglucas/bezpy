"""Tests for bezpy.mag.SECSInterpolator.

Synthetic truth comes from pysecs itself: a time-varying set of SEC pole
amplitudes generates B at "observatory" locations, the interpolator fits
those observations, and predictions are compared with the same source
evaluated at held-out locations.
"""
import numpy as np
import pytest

pysecs = pytest.importorskip("pysecs")

from bezpy.mag import SECSInterpolator  # noqa: E402
from bezpy.mag.secs import R_EARTH, _distance_to_gap  # noqa: E402

RNG = np.random.default_rng(7)

OBS_LATLONS = np.array([[lat, lon]
                        for lat in np.linspace(34, 46, 6)
                        for lon in np.linspace(-110, -96, 6)], dtype=float)
TEST_LATLONS = np.array([[38.7, -104.1], [41.3, -101.5], [39.9, -106.2]])


@pytest.fixture(scope="module")
def synthetic():
    """(B_obs, B_true_at_test) generated from a smooth SEC source."""
    poles = np.array([[lat, lon, R_EARTH + 110e3]
                      for lat in np.linspace(30, 50, 4)
                      for lon in np.linspace(-115, -90, 5)])
    source = pysecs.SECS(sec_df_loc=poles)
    ntimes = 40
    # Smoothly varying amplitudes scaled to give ~nT fields at the ground
    t = np.linspace(0, 2*np.pi, ntimes)[:, None]
    phases = RNG.uniform(0, 2*np.pi, len(poles))
    source.sec_amps = 1e13 * np.sin(t + phases) * RNG.uniform(0.5, 1.5, len(poles))

    def eval_at(latlons):
        locs = np.column_stack([latlons, np.full(len(latlons), R_EARTH)])
        return source.predict(locs)

    return eval_at(OBS_LATLONS), eval_at(TEST_LATLONS)


def test_predict_recovers_synthetic_field(synthetic):
    B_obs, B_true = synthetic
    interp = SECSInterpolator(grid="auto")
    interp.fit(B_obs, obs_latlons=OBS_LATLONS)
    B_pred = interp.predict(TEST_LATLONS)

    scale = np.sqrt(np.mean(B_true[:, :, :2]**2))
    err = np.sqrt(np.mean((B_pred[:, :, :2] - B_true[:, :, :2])**2))
    assert err < 0.15 * scale


def test_gap_masking_beats_drop(synthetic):
    """The D1 comparison: masking gappy samples must beat discarding the
    observatory (the legacy interpolate-then-drop endpoint)."""
    B_obs, B_true = synthetic
    B_gappy = B_obs.copy()
    # Knock a short gap into the observatory nearest the first test point
    B_gappy[10:14, 5, :] = np.nan

    errs = {}
    for gaps in ("mask", "drop"):
        interp = SECSInterpolator(grid="auto")
        interp.fit(B_gappy, obs_latlons=OBS_LATLONS, gaps=gaps)
        B_pred = interp.predict(TEST_LATLONS)
        # Score only samples where the gappy observatory has data
        good = np.ones(len(B_obs), dtype=bool)
        good[10:14] = False
        errs[gaps] = np.sqrt(np.mean(
            (B_pred[good, :, :2] - B_true[good, :, :2])**2))

    assert errs["mask"] < errs["drop"]


def test_gap_masking_matches_full_fit_outside_gap(synthetic):
    """Samples without gaps must be completely unaffected by masking."""
    B_obs, _ = synthetic
    B_gappy = B_obs.copy()
    B_gappy[20:25, 3, :] = np.nan

    full = SECSInterpolator().fit(B_obs, obs_latlons=OBS_LATLONS)
    masked = SECSInterpolator().fit(B_gappy, obs_latlons=OBS_LATLONS)
    P_full = full.predict(TEST_LATLONS)
    P_masked = masked.predict(TEST_LATLONS)

    good = np.ones(len(B_obs), dtype=bool)
    good[20:25] = False
    np.testing.assert_allclose(P_masked[good], P_full[good], rtol=1e-8)


def test_weight_gaps_inflates_near_edges(synthetic):
    B_obs, _ = synthetic
    B_gappy = B_obs.copy()
    B_gappy[15:20, 0, :] = np.nan
    interp = SECSInterpolator()
    # Just verifies the option runs and changes the fit
    interp.fit(B_gappy, obs_latlons=OBS_LATLONS, weight_gaps=True,
               gap_inflation=10., gap_window=5)
    plain = SECSInterpolator().fit(B_gappy, obs_latlons=OBS_LATLONS)
    assert np.any(interp._secs.sec_amps != plain._secs.sec_amps)


def test_distance_to_gap():
    missing = np.zeros((7, 1), dtype=bool)
    missing[3, 0] = True
    np.testing.assert_array_equal(_distance_to_gap(missing)[:, 0],
                                  [3, 2, 1, 0, 1, 2, 3])


def test_xarray_roundtrip(synthetic):
    xr = pytest.importorskip("xarray")
    B_obs, _ = synthetic
    codes = [f"OB{i}" for i in range(len(OBS_LATLONS))]
    times = np.datetime64("2024-05-10") + np.arange(len(B_obs))*np.timedelta64(60, "s")
    ds = xr.Dataset(
        {name: (("time", "obs"), B_obs[:, :, i]) for i, name in
         enumerate(("Bx", "By", "Bz"))},
        coords={"time": times, "obs": codes,
                "latitude": ("obs", OBS_LATLONS[:, 0]),
                "longitude": ("obs", OBS_LATLONS[:, 1])})

    interp = SECSInterpolator()
    interp.fit(ds)
    out = interp.predict(TEST_LATLONS)
    assert set(out.data_vars) == {"Bx", "By", "Bz"}
    assert out["Bx"].dims == ("time", "loc")
    np.testing.assert_array_equal(out["time"].values, times)
    np.testing.assert_allclose(out["latitude"].values, TEST_LATLONS[:, 0])

    # Same numbers as the bare-array path
    plain = SECSInterpolator().fit(B_obs, obs_latlons=OBS_LATLONS)
    np.testing.assert_allclose(
        np.stack([out[v].values for v in ("Bx", "By", "Bz")], axis=-1),
        plain.predict(TEST_LATLONS))


def test_explicit_bbox_and_pole_grid(synthetic):
    B_obs, B_true = synthetic
    bbox = SECSInterpolator(grid=(-113, 31, -93, 49))
    bbox.fit(B_obs, obs_latlons=OBS_LATLONS)
    err = np.sqrt(np.mean((bbox.predict(TEST_LATLONS)[:, :, :2]
                           - B_true[:, :, :2])**2))
    assert err < 0.15 * np.sqrt(np.mean(B_true[:, :, :2]**2))

    poles = np.array([[40., -105.], [42., -100.]])
    explicit = SECSInterpolator(grid=poles)
    explicit.fit(B_obs, obs_latlons=OBS_LATLONS)
    assert explicit._secs.sec_df_loc.shape == (2, 3)


def test_loo_score(synthetic):
    B_obs, _ = synthetic
    interp = SECSInterpolator()
    scores = interp.loo_score(B_obs, obs_latlons=OBS_LATLONS)
    assert len(scores) == len(OBS_LATLONS)
    assert all(np.isfinite(s) and s >= 0 for s in scores.values())
    # A hugely over-regularized fit must score worse
    bad = interp.loo_score(B_obs, obs_latlons=OBS_LATLONS, epsilon=0.9)
    assert np.mean(list(bad.values())) > np.mean(list(scores.values()))


def test_kalman_beats_lstsq_on_noisy_gappy_data(synthetic):
    """The 3.1 promise: the temporal filter/smoother must beat independent
    snapshot fits when the observations are noisy and gappy."""
    B_obs, B_true = synthetic
    rng = np.random.default_rng(11)
    # Noise at ~40% of the (few-nT) synthetic field scale
    noisy = B_obs + rng.normal(0., 1.5, B_obs.shape)
    noisy[18:22, 7, :] = np.nan  # a dropout on top of the noise
    times = np.arange(len(B_obs))*60.  # 1-minute cadence, seconds

    errs = {}
    for method in ("lstsq", "kalman"):
        interp = SECSInterpolator(method=method, tau=600.)
        interp.fit(noisy, obs_latlons=OBS_LATLONS, times=times)
        B_pred = interp.predict(TEST_LATLONS)
        errs[method] = np.sqrt(np.mean((B_pred[:, :, :2] - B_true[:, :, :2])**2))

    assert errs["kalman"] < errs["lstsq"]


def test_kalman_needs_times():
    interp = SECSInterpolator(method="kalman")
    with pytest.raises(ValueError, match="times"):
        interp.fit(np.zeros((5, 4, 3)), obs_latlons=OBS_LATLONS[:4])


def test_kalman_xarray_uses_dataset_times(synthetic):
    xr = pytest.importorskip("xarray")
    B_obs, B_true = synthetic
    times = (np.datetime64("2024-05-10")
             + np.arange(len(B_obs))*np.timedelta64(60, "s"))
    ds = xr.Dataset(
        {name: (("time", "obs"), B_obs[:, :, i]) for i, name in
         enumerate(("Bx", "By", "Bz"))},
        coords={"time": times, "obs": [f"OB{i}" for i in range(len(OBS_LATLONS))],
                "latitude": ("obs", OBS_LATLONS[:, 0]),
                "longitude": ("obs", OBS_LATLONS[:, 1])})
    interp = SECSInterpolator(method="kalman", tau=600.)
    interp.fit(ds)
    out = interp.predict(TEST_LATLONS, return_var=True)
    assert {"Bx", "By", "Bz", "Bx_var", "By_var", "Bz_var"} <= set(out.data_vars)
    scale = np.sqrt(np.mean(B_true[:, :, :2]**2))
    err = np.sqrt(np.mean(
        (np.stack([out["Bx"].values, out["By"].values], axis=-1)
         - B_true[:, :, :2])**2))
    assert err < 0.2 * scale
    assert (out["Bx_var"].values > 0).all()


def test_robust_rejects_single_station_spike(synthetic):
    """IRLS must downweight a localized artifact that ordinary least
    squares would spread across the whole solution."""
    B_obs, B_true = synthetic
    spiked = B_obs.copy()
    spiked[12:15, 20, :2] += 400.  # gross baseline jump at one station

    errs = {}
    for robust in (None, "bisquare"):
        interp = SECSInterpolator(robust=robust)
        interp.fit(spiked, obs_latlons=OBS_LATLONS)
        B_pred = interp.predict(TEST_LATLONS)
        errs[robust] = np.sqrt(np.mean(
            (B_pred[12:15, :, :2] - B_true[12:15, :, :2])**2))

    assert errs["bisquare"] < 0.5 * errs[None]


def test_robust_requires_lstsq():
    with pytest.raises(ValueError, match="gate_sigma"):
        SECSInterpolator(method="kalman", robust="huber")


def test_prediction_variance(synthetic):
    """Variances must be positive and follow the linear-Gaussian scaling:
    doubling every observation's standard error quadruples them."""
    B_obs, _ = synthetic
    interp = SECSInterpolator().fit(B_obs, obs_latlons=OBS_LATLONS)
    B, var = interp.predict(TEST_LATLONS, return_var=True)
    assert B.shape == var.shape == (len(B_obs), len(TEST_LATLONS), 3)
    assert (var[:, :, :2] > 0).all()

    doubled = SECSInterpolator().fit(B_obs, obs_latlons=OBS_LATLONS, obs_std=2.)
    _, var2 = doubled.predict(TEST_LATLONS, return_var=True)
    np.testing.assert_allclose(var2[:, :, :2], 4.*var[:, :, :2], rtol=1e-8)


def test_predict_single_time_step():
    """A one-sample fit must keep the (ntimes, nloc, 3) prediction shape."""
    B = RNG.normal(0., 50., (1, len(OBS_LATLONS), 3))
    interp = SECSInterpolator().fit(B, obs_latlons=OBS_LATLONS)
    out = interp.predict(TEST_LATLONS)
    assert out.shape == (1, len(TEST_LATLONS), 3)


def test_predict_before_fit_raises():
    with pytest.raises(ValueError, match="not been fit"):
        SECSInterpolator().predict(TEST_LATLONS)
