"""Tests for bezpy.mag preprocessing (combine_observatories/prepare/trim_pad)."""
import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

from bezpy.mag import combine_observatories, prepare, trim_pad  # noqa: E402

RNG = np.random.default_rng(11)


def make_obs(start, periods, dt="60s", lat=40., lon=-105., name_style="XYZ"):
    times = pd.date_range(start, periods=periods, freq=dt)
    names = ("X", "Y", "Z") if name_style == "XYZ" else ("Bx", "By", "Bz")
    ds = xr.Dataset({n: ("time", RNG.standard_normal(periods)) for n in names},
                    coords={"time": times},
                    attrs={"latitude": lat, "longitude": lon})
    return ds


def test_combine_union_and_coords():
    ds1 = make_obs("2024-05-10", 100, lat=40., lon=-105.)
    ds2 = make_obs("2024-05-10 00:30", 100, lat=45., lon=-100., name_style="BXYZ")
    combined = combine_observatories({"BOU": ds1, "XYZ": ds2})

    assert set(combined.data_vars) == {"Bx", "By", "Bz"}
    assert combined["Bx"].dims == ("time", "obs")
    assert combined.sizes["obs"] == 2
    # Union time axis: 30 min offset with 100 minutes each -> 130 samples
    assert combined.sizes["time"] == 130
    np.testing.assert_array_equal(combined["latitude"].values, [40., 45.])
    # Samples outside each observatory's coverage are NaN
    assert np.isnan(combined["Bx"].sel(obs="XYZ").values[:30]).all()
    assert np.isnan(combined["Bx"].sel(obs="BOU").values[-30:]).all()


def test_combine_accepts_dataframes():
    times = pd.date_range("2024-05-10", periods=50, freq="60s")
    df = pd.DataFrame({"X": RNG.standard_normal(50),
                       "Y": RNG.standard_normal(50),
                       "Z": RNG.standard_normal(50)}, index=times)
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
    b = combine_observatories({"BOU": make_obs("2024-05-09", 3*1440)})
    window = ("2024-05-10", "2024-05-11")
    prepared = prepare(b, window=window, pad="12h")

    assert prepared.attrs["pad_left"] == 12*60
    assert prepared.attrs["dt_seconds"] == 60.
    trimmed = trim_pad(prepared)
    assert trimmed["time"].values[0] == np.datetime64("2024-05-10")
    assert trimmed["time"].values[-1] == np.datetime64("2024-05-11")
    assert "pad_left" not in trimmed.attrs


def test_prepare_auto_pad_scales_with_max_period():
    b = combine_observatories({"BOU": make_obs("2024-05-01", 20*1440)})
    window = ("2024-05-08", "2024-05-09")
    short = prepare(b, window=window, max_period=10000.)
    long = prepare(b, window=window, max_period=30000.)
    # pad = 3 * max_period on each side
    assert short.attrs["pad_left"] == 3*10000/60
    assert long.attrs["pad_left"] == 3*short.attrs["pad_left"]


def test_taper_zero_at_edges_one_inside():
    b = combine_observatories({"BOU": make_obs("2024-05-09", 3*1440)})
    prepared = prepare(b, window=("2024-05-10", "2024-05-11"), pad="6h",
                       detrend=None, taper="tukey")
    values = prepared["Bx"].values[:, 0]
    original = b["Bx"].sel(time=prepared["time"]).values[:, 0]
    pad = prepared.attrs["pad_left"]
    # Fully suppressed at the outer edge, untouched inside the window
    assert values[0] == 0.
    np.testing.assert_allclose(values[pad:-pad], original[pad:-pad])


def test_d2_linear_taper_beats_median_on_leakage():
    """The D2 comparison: for a trending record, linear detrend + taper must
    leak far less broadband power than the legacy median subtraction."""
    n = 4*1440  # 4 days at 1 min
    t = np.arange(n)*60.
    trend = 5e-3*t                     # secular drift, ~430 nT over the span
    signal = 20*np.sin(2*np.pi*t/86400.)  # diurnal
    times = pd.date_range("2024-05-09", periods=n, freq="60s")
    ds = xr.Dataset({"X": ("time", trend + signal),
                     "Y": ("time", np.zeros(n)), "Z": ("time", np.zeros(n))},
                    coords={"time": times},
                    attrs={"latitude": 40., "longitude": -105.})
    b = combine_observatories({"BOU": ds})

    def high_freq_power(prepared):
        x = np.nan_to_num(prepared["Bx"].values[:, 0])
        spectrum = np.abs(np.fft.rfft(x))**2
        freqs = np.fft.rfftfreq(len(x), d=60.)
        return spectrum[freqs > 1/3600.].sum()  # periods under an hour

    window = ("2024-05-10", "2024-05-12")
    legacy = prepare(b, window=window, pad="2D", detrend="median", taper=None)
    modern = prepare(b, window=window, pad="2D", detrend="linear", taper="tukey")
    assert high_freq_power(modern) < 0.01 * high_freq_power(legacy)


def test_gap_policies():
    ds = make_obs("2024-05-10", 200)
    ds["X"][50:53] = np.nan   # short gap (3 min)
    ds["X"][100:160] = np.nan  # long gap (1 h)
    b = combine_observatories({"BOU": ds})

    masked = prepare(b, pad="10min", detrend=None, taper=None, gaps="mask")
    assert np.isnan(masked["Bx"].values[:, 0]).sum() == 63

    filled = prepare(b, pad="10min", detrend=None, taper=None,
                     gaps="interpolate", max_gap="10min")
    x = filled["Bx"].values[:, 0]
    assert not np.isnan(x[50:53]).any()      # short gap filled
    assert np.isnan(x[100:160]).all()        # long gap respected


def test_prepare_option_validation():
    b = combine_observatories({"BOU": make_obs("2024-05-10", 50)})
    with pytest.raises(ValueError, match="detrend"):
        prepare(b, pad="5min", detrend="cubic")
    with pytest.raises(ValueError, match="gaps"):
        prepare(b, pad="5min", gaps="drop")


def test_prepare_record_shorter_than_pad_raises():
    """A record consumed entirely by the pad raises, not empties (B2)."""
    import xarray as xr
    times = pd.date_range("2024-05-10", periods=1440, freq="1min")  # 1 day
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {"Bx": (("time", "obs"), rng.normal(size=(1440, 1))),
         "By": (("time", "obs"), rng.normal(size=(1440, 1)))},
        coords={"time": times, "obs": ["AAA"],
                "latitude": ("obs", [40.0]), "longitude": ("obs", [-105.0])})
    # Default pad="auto" is 25 h per side -- more than the whole record
    with pytest.raises(ValueError, match="pad"):
        prepare(ds)
    # An explicit short pad works on the same record
    out = prepare(ds, pad="1h")
    assert trim_pad(out).sizes["time"] == 1440 - 2 * 60
