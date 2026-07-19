"""Tests for bezpy.cli: bezpy download/update/scan/nowcast subcommands.

Argument-parsing and dispatch are checked with the underlying bezpy.data /
bezpy.pipeline calls mocked -- no network access here. ``nowcast --once``
gets a lighter integration check: real (tiny synthetic) Nowcast machinery
fed by an injected fake data source, doubling as a wiring test for the
CLI's poll-loop body and CSV output.
"""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
pysecs = pytest.importorskip("pysecs")

from bezpy import cli  # noqa: E402
from bezpy.mt.site import Site1d, Site3d  # noqa: E402


def test_download_dispatches_per_code():
    with patch("bezpy.data.download_observatory", return_value=["a.nc"]) as m:
        cli.main(["download", "BOU", "FRD", "--start", "2024-01-01", "--end", "2024-01-02"])
    assert m.call_count == 2
    codes = [c.args[0] for c in m.call_args_list]
    assert codes == ["BOU", "FRD"]


def test_update_dispatches_per_code():
    with patch("bezpy.data.update_observatory", return_value=[]) as m:
        cli.main(["update", "BOU"])
    m.assert_called_once()
    assert m.call_args.args[0] == "BOU"


def test_scan_requires_threshold_for_exceedances():
    with pytest.raises(SystemExit, match="threshold"):
        cli.main(["scan", "--obs", "BOU", "--start", "2024-01-01", "--end", "2024-01-02",
                 "--bbox", "-110", "35", "-100", "45", "--reduce", "exceedances",
                 "--out", "out.nc"])


def test_scan_no_sites_raises():
    with patch("bezpy.data.load_mt_sites", return_value=[]):
        with pytest.raises(SystemExit, match="no cached MT sites"):
            cli.main(["scan", "--obs", "BOU", "--start", "2024-01-01", "--end", "2024-01-02",
                     "--bbox", "-110", "35", "-100", "45", "--out", "out.nc"])


def test_scan_calls_pipeline_scan(tmp_path):
    site = Site3d("S00", latitude=40., longitude=-105.,
                 periods=np.logspace(1, 4, 10), Z=np.zeros((4, 10), dtype=complex))
    out = str(tmp_path / "out.nc")
    with patch("bezpy.data.load_mt_sites", return_value=[site]), \
         patch("bezpy.data.load_observatory_group", return_value="B_OBS_STUB"), \
         patch("bezpy.pipeline.scan") as m:
        cli.main(["scan", "--obs", "BOU", "--start", "2024-01-01", "--end", "2024-01-02",
                 "--bbox", "-110", "35", "-100", "45", "--out", out])
    m.assert_called_once()
    assert m.call_args.args[0] == "B_OBS_STUB"
    assert m.call_args.kwargs["out_path"] == out


# ---------------------------------------------------------------------
# nowcast --once: light integration test against real Nowcast machinery
# ---------------------------------------------------------------------

def _broadband_multi_obs(obs_latlons, ntimes, dt, rng, ntones=15):
    t = np.arange(ntimes) * dt
    periods_pool = np.logspace(np.log10(700), np.log10(2000), 30)

    def series():
        chosen = rng.choice(periods_pool, size=ntones, replace=False)
        b = np.zeros(ntimes)
        for p in chosen:
            b += rng.uniform(0.3, 1.0) * np.sin(2*np.pi*t/p + rng.uniform(0, 2*np.pi))
        return b

    nobs = len(obs_latlons)
    bx = np.stack([series() for _ in range(nobs)], axis=1)
    by = np.stack([series() for _ in range(nobs)], axis=1)
    times = pd.date_range("2024-05-10", periods=ntimes, freq=f"{int(dt)}s")
    return xr.Dataset(
        {"Bx": (("time", "obs"), bx), "By": (("time", "obs"), by),
         "Bz": (("time", "obs"), np.zeros_like(bx))},
        coords={"time": times, "obs": [f"OB{i}" for i in range(nobs)],
                "latitude": ("obs", obs_latlons[:, 0]),
                "longitude": ("obs", obs_latlons[:, 1])})


def test_nowcast_once_writes_csv(tmp_path):
    dt = 60.
    # A small, well-sampled period band keeps warmup_samples small so a
    # single --once call (fed enough samples in one shot below) finalizes.
    periods = np.logspace(np.log10(700), np.log10(2000), 8)
    site = Site1d("s00", [1e9], [100.0])
    site3d = Site3d("s00", latitude=40., longitude=-105.,
                    periods=periods, Z=site.calcZ(1./periods))
    obs_latlons = np.array([[36., -109.], [44., -109.], [40., -99.], [40., -101.]])

    rng = np.random.default_rng(1)
    nmax = max(1, int(np.ceil(2.0 * periods.max() / dt)))
    ntimes = 2 * (2*nmax + 1) + 100  # comfortably past warmup
    b_obs = _broadband_multi_obs(obs_latlons, ntimes, dt, rng)

    out = str(tmp_path / "nowcast_out.csv")
    with patch("bezpy.data.load_mt_sites", return_value=[site3d]), \
         patch("bezpy.data.update_observatory", return_value=[]), \
         patch("bezpy.data.load_observatory_group", return_value=b_obs):
        cli.main(["nowcast", "--obs", "OB0", "OB1", "OB2", "OB3",
                 "--bbox", "-110", "35", "-100", "45", "--dt", str(dt),
                 "--out", out, "--once"])

    df = pd.read_csv(out, index_col="time")
    assert len(df) > 0
    assert "s00_Ex" in df.columns and "s00_Ey" in df.columns


def test_nowcast_once_only_polls_once():
    site = Site1d("s00", [1e9], [100.0])
    periods = np.logspace(np.log10(700), np.log10(2000), 8)
    site3d = Site3d("s00", latitude=40., longitude=-105.,
                    periods=periods, Z=site.calcZ(1./periods))
    empty = xr.Dataset({"Bx": (("time", "obs"), np.empty((0, 1))),
                        "By": (("time", "obs"), np.empty((0, 1))),
                        "Bz": (("time", "obs"), np.empty((0, 1)))},
                       coords={"time": np.array([], dtype="datetime64[ns]"),
                               "obs": ["OB0"], "latitude": ("obs", [40.]),
                               "longitude": ("obs", [-105.])})
    with patch("bezpy.data.load_mt_sites", return_value=[site3d]), \
         patch("bezpy.data.update_observatory", return_value=[]) as m_update, \
         patch("bezpy.data.load_observatory_group", return_value=empty):
        cli.main(["nowcast", "--obs", "OB0", "--bbox", "-110", "35", "-100", "45",
                 "--out", "unused.csv", "--once"])
    m_update.assert_called_once()
