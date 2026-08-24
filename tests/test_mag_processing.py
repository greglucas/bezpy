"""Tests for bezpy.mag preprocessing (combine_observatories/prepare/trim_pad)."""

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

from bezpy.mag import combine_observatories, despike, prepare, trim_pad  # noqa: E402

RNG = np.random.default_rng(11)


def make_obs(start, periods, dt="60s", lat=40.0, lon=-105.0, name_style="XYZ"):
    times = pd.date_range(start, periods=periods, freq=dt)
    names = ("X", "Y", "Z") if name_style == "XYZ" else ("Bx", "By", "Bz")
    ds = xr.Dataset(
        {n: ("time", RNG.standard_normal(periods)) for n in names},
        coords={"time": times},
        attrs={"latitude": lat, "longitude": lon},
    )
    return ds


def test_combine_union_and_coords():
    ds1 = make_obs("2024-05-10", 100, lat=40.0, lon=-105.0)
    ds2 = make_obs("2024-05-10 00:30", 100, lat=45.0, lon=-100.0, name_style="BXYZ")
    combined = combine_observatories({"BOU": ds1, "XYZ": ds2})

    assert set(combined.data_vars) == {"Bx", "By", "Bz"}
    assert combined["Bx"].dims == ("time", "obs")
    assert combined.sizes["obs"] == 2
    # Union time axis: 30 min offset with 100 minutes each -> 130 samples
    assert combined.sizes["time"] == 130
    np.testing.assert_array_equal(combined["latitude"].values, [40.0, 45.0])
    # Samples outside each observatory's coverage are NaN
    assert np.isnan(combined["Bx"].sel(obs="XYZ").values[:30]).all()
    assert np.isnan(combined["Bx"].sel(obs="BOU").values[-30:]).all()


def test_combine_accepts_dataframes():
    times = pd.date_range("2024-05-10", periods=50, freq="60s")
    df = pd.DataFrame(
        {"X": RNG.standard_normal(50), "Y": RNG.standard_normal(50), "Z": RNG.standard_normal(50)},
        index=times,
    )
    combined = combine_observatories({"BOU": df})
    assert combined.sizes == {"time": 50, "obs": 1}
    # Coordinates fall back to the bundled IAGA table
    assert abs(combined["latitude"].item() - 40.14) < 0.5


def test_prepare_nonuniform_time_raises():
    ds = make_obs("2024-05-10", 50)
    broken = ds.isel(time=np.r_[0:20, 25:50])  # cut a chunk out
    with pytest.raises(ValueError, match="not uniformly sampled"):
        prepare(combine_observatories({"BOU": broken}), pad="1h")


def test_prepare_trim_pad_roundtrip():
    b = combine_observatories({"BOU": make_obs("2024-05-09", 3 * 1440)})
    window = ("2024-05-10", "2024-05-11")
    prepared = prepare(b, window=window, pad="12h")

    assert prepared.attrs["pad_left"] == 12 * 60
    assert prepared.attrs["dt_seconds"] == 60.0
    trimmed = trim_pad(prepared)
    assert trimmed["time"].values[0] == np.datetime64("2024-05-10")
    assert trimmed["time"].values[-1] == np.datetime64("2024-05-11")
    assert "pad_left" not in trimmed.attrs


def test_prepare_auto_pad_scales_with_max_period():
    b = combine_observatories({"BOU": make_obs("2024-05-01", 20 * 1440)})
    window = ("2024-05-08", "2024-05-09")
    short = prepare(b, window=window, max_period=10000.0)
    long = prepare(b, window=window, max_period=30000.0)
    # pad = 3 * max_period on each side
    assert short.attrs["pad_left"] == 3 * 10000 / 60
    assert long.attrs["pad_left"] == 3 * short.attrs["pad_left"]


def test_taper_zero_at_edges_one_inside():
    b = combine_observatories({"BOU": make_obs("2024-05-09", 3 * 1440)})
    prepared = prepare(
        b, window=("2024-05-10", "2024-05-11"), pad="6h", detrend=None, taper="tukey"
    )
    values = prepared["Bx"].values[:, 0]
    original = b["Bx"].sel(time=prepared["time"]).values[:, 0]
    pad = prepared.attrs["pad_left"]
    # Fully suppressed at the outer edge, untouched inside the window
    assert values[0] == 0.0
    np.testing.assert_allclose(values[pad:-pad], original[pad:-pad])


def test_d2_linear_taper_beats_median_on_leakage():
    """The D2 comparison: for a trending record, linear detrend + taper must
    leak far less broadband power than the legacy median subtraction."""
    n = 4 * 1440  # 4 days at 1 min
    t = np.arange(n) * 60.0
    trend = 5e-3 * t  # secular drift, ~430 nT over the span
    signal = 20 * np.sin(2 * np.pi * t / 86400.0)  # diurnal
    times = pd.date_range("2024-05-09", periods=n, freq="60s")
    ds = xr.Dataset(
        {"X": ("time", trend + signal), "Y": ("time", np.zeros(n)), "Z": ("time", np.zeros(n))},
        coords={"time": times},
        attrs={"latitude": 40.0, "longitude": -105.0},
    )
    b = combine_observatories({"BOU": ds})

    def high_freq_power(prepared):
        x = np.nan_to_num(prepared["Bx"].values[:, 0])
        spectrum = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(len(x), d=60.0)
        return spectrum[freqs > 1 / 3600.0].sum()  # periods under an hour

    window = ("2024-05-10", "2024-05-12")
    legacy = prepare(b, window=window, pad="2D", detrend="median", taper=None)
    modern = prepare(b, window=window, pad="2D", detrend="linear", taper="tukey")
    assert high_freq_power(modern) < 0.01 * high_freq_power(legacy)


def test_gap_policies():
    ds = make_obs("2024-05-10", 200)
    ds["X"][50:53] = np.nan  # short gap (3 min)
    ds["X"][100:160] = np.nan  # long gap (1 h)
    b = combine_observatories({"BOU": ds})

    masked = prepare(b, pad="10min", detrend=None, taper=None, gaps="mask")
    assert np.isnan(masked["Bx"].values[:, 0]).sum() == 63

    filled = prepare(b, pad="10min", detrend=None, taper=None, gaps="interpolate", max_gap="10min")
    x = filled["Bx"].values[:, 0]
    assert not np.isnan(x[50:53]).any()  # short gap filled
    assert np.isnan(x[100:160]).all()  # long gap respected


def test_prepare_option_validation():
    b = combine_observatories({"BOU": make_obs("2024-05-10", 50)})
    with pytest.raises(ValueError, match="detrend"):
        prepare(b, pad="5min", detrend="cubic")
    with pytest.raises(ValueError, match="gaps"):
        prepare(b, pad="5min", gaps="drop")


def _clean_field(n=6 * 1440, seed=3):
    """A smooth diurnal + trend + mild noise -- physically plausible B(t)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 60.0
    return (
        30 * np.sin(2 * np.pi * t / 86400.0)  # diurnal
        + 2e-3 * t  # secular drift
        + rng.standard_normal(n)
    )  # ~1 nT measurement noise


def test_despike_removes_spikes_and_rails_keeps_signal():
    """Isolated spikes and multi-minute rails are masked; the field survives."""
    clean = _clean_field()
    x = clean.copy()
    x[1000] += 5000.0  # isolated spike
    x[2000:2006] = 99800.0  # multi-minute saturation rail
    times = pd.date_range("2024-05-09", periods=len(x), freq="60s")
    ds = xr.Dataset(
        {"X": ("time", x), "Y": ("time", np.zeros_like(x)), "Z": ("time", np.zeros_like(x))},
        coords={"time": times},
        attrs={"latitude": 40.0, "longitude": -105.0},
    )

    out, flags = despike(ds, return_flags=True)
    fx = flags["X"]
    assert fx[1000]  # spike flagged
    assert fx[2000:2006].all()  # whole rail flagged
    assert out.attrs["n_despiked"] == int(fx.sum()) > 0
    # no physically-impossible minute-to-minute rate survives
    assert np.nanmax(np.abs(np.diff(out["X"].values))) < 500.0
    # away from the artifacts the smooth field is untouched
    quiet = np.ones(len(x), bool)
    for lo, hi in [(980, 1020), (1980, 2030)]:
        quiet[lo:hi] = False
    kept = np.isfinite(out["X"].values) & quiet
    np.testing.assert_allclose(out["X"].values[kept], clean[kept])


def test_despike_leaves_pure_dc_step_in_place():
    """A persistent re-level with no spike is not an outlier against a rolling
    median -- despike must not mangle it (the per-window detrend owns it)."""
    x = _clean_field(n=4000)
    x[2000:] += 26000.0
    out, flags = despike(np.asarray(x), window=31, return_flags=True)
    # the step edge is not turned into a scatter of masked samples
    assert flags[1900:2100].sum() <= 2
    # both plateaus survive intact
    np.testing.assert_allclose(out[1000], x[1000])
    np.testing.assert_allclose(out[3000], x[3000])


def test_despike_leaves_clean_data_alone():
    """A smooth record with no artifacts is essentially never flagged."""
    x = _clean_field()
    _, flags = despike(pd.DataFrame({"X": x}), return_flags=True)
    assert flags.mean() < 1e-3


def test_despike_preserves_real_transient():
    """A rapid but real (multi-sample, evolving) excursion is not a spike and
    must survive -- despike keys on deviation from a rolling median, and a
    smoothly-varying storm tracks that median."""
    n = 3 * 1440
    t = np.arange(n) * 60.0
    x = 20 * np.sin(2 * np.pi * t / 86400.0)
    burst = np.exp(-0.5 * ((np.arange(n) - 1500) / 40.0) ** 2) * 300  # ~1h storm
    x = x + burst
    out = despike(np.asarray(x), window=31)
    assert np.isfinite(out).all()  # nothing masked
    assert out[1500] == pytest.approx(x[1500])


def test_despike_clears_short_rail_flanked_by_spikes():
    """A short saturation rail flanked by rail-valued spikes -- the real
    variometer morphology -- is fully removed, leaving no impossible rate."""
    x = _clean_field(n=2000)
    x[898] = -5e4  # spike just before ...
    x[899:906] = 5e4  # ... a 7-min rail ...
    x[906] = 8e4  # ... spike just after
    out = despike(np.asarray(x), window=31)
    assert np.isnan(out[898:907]).all()
    assert np.nanmax(np.abs(np.diff(out))) < 500.0


def test_despike_input_types_and_window_units():
    x = _clean_field(n=1000)
    x[500] += 5000.0
    times = pd.date_range("2024-05-09", periods=len(x), freq="60s")

    # DataFrame: duration window resolved from the index cadence
    df_out = despike(pd.DataFrame({"X": x}, index=times), window="31min")
    assert np.isnan(df_out["X"].to_numpy()[500])

    # bare array: window must be an integer sample count
    arr_out = despike(np.asarray(x), window=31)
    assert np.isnan(arr_out[500])
    with pytest.raises(ValueError, match="integer sample count"):
        despike(np.asarray(x), window="31min")


def test_prepare_record_shorter_than_pad_raises():
    """A record consumed entirely by the pad raises, not empties (B2)."""
    import xarray as xr

    times = pd.date_range("2024-05-10", periods=1440, freq="1min")  # 1 day
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {
            "Bx": (("time", "obs"), rng.normal(size=(1440, 1))),
            "By": (("time", "obs"), rng.normal(size=(1440, 1))),
        },
        coords={
            "time": times,
            "obs": ["AAA"],
            "latitude": ("obs", [40.0]),
            "longitude": ("obs", [-105.0]),
        },
    )
    # Default pad="auto" is 25 h per side -- more than the whole record
    with pytest.raises(ValueError, match="pad"):
        prepare(ds)
    # An explicit short pad works on the same record
    out = prepare(ds, pad="1h")
    assert trim_pad(out).sizes["time"] == 1440 - 2 * 60
