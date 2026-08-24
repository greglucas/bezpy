"""Tests for bezpy.storms and bezpy.data.fetch_index parsing."""

import numpy as np
import pandas as pd
import pytest

from bezpy.storms import find_events, find_storms


def hourly(start, values):
    times = pd.date_range(start, periods=len(values), freq="1h")
    return pd.Series(values, index=times)


# ---------------------------------------------------------------------
# find_events
# ---------------------------------------------------------------------


def test_find_events_basic():
    values = np.zeros(200)
    values[30:35] = [5, 8, 12, 9, 6]  # event 1, peak 12
    values[120:122] = [7, 15]  # event 2, peak 15
    series = hourly("2024-01-01", values)

    catalog = find_events(series, threshold=4.0, gap="12h")
    assert len(catalog) == 2
    np.testing.assert_array_equal(catalog["peak_value"], [12, 15])
    assert catalog["start"].iloc[0] == series.index[30]
    assert catalog["end"].iloc[0] == series.index[34]
    assert catalog["peak_time"].iloc[0] == series.index[32]


def test_find_events_runs_declustering():
    values = np.zeros(100)
    values[10] = 10.0
    values[15] = 8.0  # 5 h later: same event under a 12 h gap
    values[40] = 9.0  # 25 h later: separate event
    series = hourly("2024-01-01", values)

    assert len(find_events(series, threshold=4.0, gap="12h")) == 2
    # With declustering off, contiguity decides: three separate events
    assert len(find_events(series, threshold=4.0, decluster=None)) == 3


def test_find_events_below_direction():
    values = np.zeros(100)
    values[20:24] = [-150, -220, -180, -145]
    series = hourly("2024-01-01", values)
    catalog = find_events(series, threshold=-140, direction="below")
    assert len(catalog) == 1
    assert catalog["peak_value"].iloc[0] == -220
    assert catalog["peak_time"].iloc[0] == series.index[21]


def test_find_events_min_duration():
    values = np.zeros(100)
    values[10] = 10.0  # single-sample spike
    values[50:56] = 10.0  # sustained event (5 h span)
    series = hourly("2024-01-01", values)
    catalog = find_events(series, threshold=4.0, min_duration="3h")
    assert len(catalog) == 1
    assert catalog["start"].iloc[0] == series.index[50]


def test_find_events_empty_and_xarray():
    xr = pytest.importorskip("xarray")
    series = hourly("2024-01-01", np.zeros(50))
    assert find_events(series, threshold=1.0).empty
    da = xr.DataArray(series.values, coords={"time": series.index}, dims="time")
    assert find_events(da, threshold=-1.0).shape[0] == 1  # all above -> one run


# ---------------------------------------------------------------------
# find_storms
# ---------------------------------------------------------------------


def make_index():
    """10 days of hourly Dst with two storms, 3-hourly Kp with a third.

    Trigger clusters are separated by more than 2 x 1.5 days so the three
    storms stay distinct.
    """
    n = 240
    dst = hourly("1989-03-01", np.full(n, -20.0))
    dst.iloc[30:40] = -200.0  # storm 1 (Dst), hours 30-39
    dst.iloc[130:135] = -160.0  # storm 2 (Dst), hours 130-134
    kp = pd.Series(3.0, index=pd.date_range("1989-03-01", periods=n // 3, freq="3h"))
    kp.iloc[73] = 8.5  # storm 3 (Kp only, Dst-quiet), hour 219
    return pd.concat([dst.rename("dst"), kp.rename("kp")], axis=1)


def test_find_storms_catalog():
    catalog = find_storms(make_index(), dst_threshold=-140, kp_threshold=8, window="1.5D")
    assert len(catalog) == 3
    # Window edges are +-1.5 days around the triggering samples
    first = catalog.iloc[0]
    assert first["start"] == first["trigger_start"] - pd.Timedelta("1.5D")
    assert first["end"] == first["trigger_end"] + pd.Timedelta("1.5D")
    assert first["min_dst"] == -200.0
    # The Kp-only storm is found even though Dst never crossed
    assert (catalog["max_kp"] == 8.5).any()


def test_find_storms_merges_overlaps():
    n = 240
    dst = hourly("1989-03-01", np.full(n, -20.0))
    dst.iloc[30] = -200.0
    dst.iloc[60] = -180.0  # 30 h later: windows overlap (2*36 h) -> one storm
    catalog = find_storms(dst.rename("dst").to_frame(), kp_threshold=None, window="1.5D")
    assert len(catalog) == 1
    assert catalog["trigger_start"].iloc[0] == dst.index[30]
    assert catalog["trigger_end"].iloc[0] == dst.index[60]


def test_find_storms_single_series_and_validation():
    dst = hourly("1989-03-01", np.full(100, -20.0))
    dst.iloc[10] = -300.0
    catalog = find_storms(dst.rename("dst"))
    assert len(catalog) == 1

    with pytest.raises(ValueError, match="neither"):
        find_storms(hourly("1989-03-01", np.zeros(10)).rename("foo").to_frame())


# ---------------------------------------------------------------------
# index parsers (bezpy.data.indices)
# ---------------------------------------------------------------------

GFZ_SAMPLE = """\
# PURPOSE: This file distributes the geomagnetic planetary three-hour index Kp
#          YYY MM DD  days  days_m  Bsr dB   Kp1 ... more header
1989 03 13 20890 20890.5 2116  6  5.333  6.000  7.667  8.667  9.000  8.667  8.000  7.667  48  80 132 207 300 236 179 132 164.0 158 219.3 217.6 1
1989 03 14 20891 20891.5 2116  7  6.667  4.667  3.667  4.333  5.667  5.333  4.667  4.000  95  32  22  32  56  48  32  27  43.0 168 227.1 225.3 1
1989 03 15 20892 20892.5 2116  8 -1.000  2.000  2.333  2.667  3.000  3.333  3.667  4.000  -1   7   9  12  15  18  22  27  16.4 150 230.0 228.1 1
"""


def omni2_row(year, doy, hour, dst):
    """A synthetic OMNI2 record: Dst is word 41 (words 39/40 are Kp*10/R)."""
    words = [str(year), str(doy), str(hour)] + ["999.9"] * 35  # words 4-38
    words += ["87", "999", str(dst)] + ["9999"] * 5  # 39, 40, 41, ...
    return " ".join(words)


OMNI2_SAMPLE = (
    "\n".join(
        [omni2_row(1989, 72, 0, -30), omni2_row(1989, 72, 1, -589), omni2_row(1989, 72, 2, 99999)]
    )
    + "\n"
)


def test_parse_gfz():
    indices = pytest.importorskip("bezpy.data.indices")
    frame = indices._parse_gfz(GFZ_SAMPLE)
    assert frame.index[0] == pd.Timestamp("1989-03-13 00:00")
    assert frame.index[1] == pd.Timestamp("1989-03-13 03:00")
    assert frame["kp"].loc["1989-03-13 12:00"] == 9.0
    assert frame["ap"].loc["1989-03-13 12:00"] == 300
    # Missing values (negative) become NaN
    assert np.isnan(frame["kp"].loc["1989-03-15 00:00"])
    assert np.isnan(frame["ap"].loc["1989-03-15 00:00"])
    # 3 days x 8 intervals
    assert len(frame) == 24


def test_parse_omni2():
    indices = pytest.importorskip("bezpy.data.indices")
    frame = indices._parse_omni2(OMNI2_SAMPLE)
    # Day-of-year 72 of 1989 is March 13
    assert frame.index[0] == pd.Timestamp("1989-03-13 00:00")
    assert frame["dst"].iloc[0] == -30.0
    assert frame["dst"].iloc[1] == -589.0
    assert np.isnan(frame["dst"].iloc[2])  # fill value


def test_fetch_index_slicing(monkeypatch):
    indices = pytest.importorskip("bezpy.data.indices")
    import bezpy.data

    monkeypatch.setattr(
        indices, "_fetch_gfz_table", lambda refresh=False: indices._parse_gfz(GFZ_SAMPLE)
    )
    kp = bezpy.data.fetch_index("kp", "1989-03-13 06:00", "1989-03-14")
    assert kp.name == "kp"
    assert kp.index[0] == pd.Timestamp("1989-03-13 06:00")
    assert kp.index[-1] == pd.Timestamp("1989-03-14 00:00")

    monkeypatch.setattr(
        indices, "_fetch_omni2_year", lambda year: indices._parse_omni2(OMNI2_SAMPLE)
    )
    dst = bezpy.data.fetch_index("dst", "1989-03-13", "1989-03-14")
    assert dst.name == "dst"
    assert dst.min() == -589.0

    with pytest.raises(ValueError, match="name must be"):
        bezpy.data.fetch_index("ae", "1989", "1990")
