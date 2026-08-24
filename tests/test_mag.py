"""Tests for bezpy.mag IAGA-2002 IO and signal utilities."""

import numpy as np
import pandas as pd
import pytest

import bezpy.mag
from bezpy.mag.utils import detrend_polynomial, filter_signal

IAGA_HDZF = """ Format                 IAGA-2002                                    |
 Source of Data         Test                                         |
 Station Name           Boulder                                      |
 IAGA CODE              BOU                                          |
 Geodetic Latitude      40.137                                       |
 Geodetic Longitude     254.764                                      |
 Elevation              1682                                         |
 Reported               HDZF                                         |
 Sensor Orientation     HDZF                                         |
 Digital Sampling       0.01 second                                  |
 Data Interval Type     filtered 1-minute (00:15-01:45)              |
 Data Type              definitive                                   |
DATE       TIME         DOY     BOUH      BOUD      BOUZ      BOUF   |
2024-05-10 00:00:00.000 131     20800.10      0.00  47000.00  52000.00
2024-05-10 00:01:00.000 131     20801.20      0.00  47001.00  52001.00
2024-05-10 00:02:00.000 131     99999.00  99999.00  99999.00  99999.00
"""


@pytest.fixture
def iaga_file(tmp_path):
    fname = tmp_path / "bou20240510.min"
    fname.write_text(IAGA_HDZF)
    return fname


def test_read_iaga_converts_hd_to_xy(iaga_file):
    df = bezpy.mag.read_iaga(str(iaga_file))
    # With declination D = 0, X == H and Y == 0.
    assert "X" in df.columns and "Y" in df.columns
    assert df["X"].iloc[0] == pytest.approx(20800.10, abs=1e-2)
    assert df["Y"].iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_read_iaga_fill_values_become_nan(iaga_file):
    df = bezpy.mag.read_iaga(str(iaga_file))
    # The 99999.0 fill row must be parsed as NaN.
    assert np.isnan(df["Z"].iloc[2])


def test_read_iaga_header(iaga_file):
    df, header = bezpy.mag.read_iaga(str(iaga_file), return_header=True)
    assert header["iaga code"] == "BOU"
    assert header["reported"] == "HDZF"


def test_write_iaga_round_trips_columns(tmp_path):
    idx = pd.date_range("2024-05-10", periods=3, freq="1min")
    df = pd.DataFrame(index=idx, data={"Ex": [1.0, 2.0, 3.0], "Ey": [4.0, 5.0, 6.0]})
    out = tmp_path / "out.min"
    bezpy.mag.write_iaga_2002(df, str(out))
    text = out.read_text()
    assert "IAGA-2002" in text
    # Values are written with the 2-decimal formatter.
    assert "1.00" in text and "6.00" in text


def test_detrend_polynomial_removes_linear_trend():
    x = np.arange(100)
    signal = 3.0 * x + 5.0
    detrended = detrend_polynomial(signal, deg=1)
    assert np.allclose(detrended, 0.0, atol=1e-6)


def test_filter_signal_removes_dc_offset():
    # A constant (DC) signal should be strongly attenuated by the highpass/bandpass.
    data = np.ones(2048) * 10.0 + np.sin(np.linspace(0, 20 * np.pi, 2048))
    filtered = filter_signal(data, sample_freq=1.0 / 60)
    assert abs(np.mean(filtered)) < abs(np.mean(data))


def test_observatory_lookup():
    codes = bezpy.mag.get_iaga_observatory_codes()
    assert len(codes) > 0
    obs = bezpy.mag.get_iaga_observatory(codes[0])
    assert "latitude" in obs and "longitude" in obs


def test_observatory_lookup_unknown_raises():
    with pytest.raises(ValueError):
        bezpy.mag.get_iaga_observatory("ZZZ")
