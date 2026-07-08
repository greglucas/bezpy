"""Geomagnetic activity indices (Kp, ap, Dst).

Sources:

* **Kp / ap** -- the definitive GFZ Potsdam record
  (``Kp_ap_Ap_SN_F107_since_1932.txt``), one 3-hourly series since 1932.
* **Dst** -- the hourly Dst index as distributed in the NASA/SPDF OMNI2
  low-resolution files (one file per year).

Files are downloaded once into the bezpy cache (see ``bezpy.data._cache``)
and parsed into pandas Series ready for ``bezpy.storms.find_storms``.
"""

__all__ = ["fetch_index"]

import numpy as np
import pandas as pd
import pooch

from ._cache import get_cache_dir

GFZ_KP_URL = ("https://kp.gfz-potsdam.de/app/files/"
              "Kp_ap_Ap_SN_F107_since_1932.txt")
OMNI2_URL = ("https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/"
             "omni2_{year}.dat")


def _parse_gfz(text):
    """Parse the GFZ Kp/ap file into a 3-hourly DataFrame (kp, ap columns).

    Data rows: YYYY MM DD days days_m Bsr dB Kp1..Kp8 ap1..ap8 Ap SN F10.7o
    F10.7a D; missing values are negative.
    """
    rows = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        day = pd.Timestamp(int(fields[0]), int(fields[1]), int(fields[2]))
        kps = np.array(fields[7:15], dtype=float)
        aps = np.array(fields[15:23], dtype=float)
        for i in range(8):
            rows.append((day + pd.Timedelta(hours=3*i), kps[i], aps[i]))
    frame = pd.DataFrame(rows, columns=["time", "kp", "ap"]).set_index("time")
    return frame.where(frame >= 0)  # negative = missing


def _parse_omni2(text):
    """Parse OMNI2 yearly records into an hourly DataFrame (dst column)."""
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 41:
            continue
        time = (pd.Timestamp(int(fields[0]), 1, 1)
                + pd.Timedelta(days=int(fields[1]) - 1, hours=int(fields[2])))
        dst = float(fields[40])
        rows.append((time, np.nan if dst >= 99999 else dst))
    return pd.DataFrame(rows, columns=["time", "dst"]).set_index("time")


def _fetch_gfz_table(refresh=False):
    fname = "Kp_ap_Ap_SN_F107_since_1932.txt"
    cached = get_cache_dir("indices") / fname
    if refresh and cached.exists():
        cached.unlink()
    path = pooch.retrieve(GFZ_KP_URL, known_hash=None, fname=fname,
                          path=get_cache_dir("indices"))
    with open(path) as f:
        return _parse_gfz(f.read())


def _fetch_omni2_year(year):
    path = pooch.retrieve(OMNI2_URL.format(year=year), known_hash=None,
                          fname=f"omni2_{year}.dat",
                          path=get_cache_dir("indices"))
    with open(path) as f:
        return _parse_omni2(f.read())


def fetch_index(name, start, end, refresh=False):
    """Fetch a geomagnetic activity index as a pandas Series.

    :param name: ``"kp"``, ``"ap"`` (3-hourly, GFZ Potsdam) or ``"dst"``
        (hourly, from the NASA/SPDF OMNI2 files)
    :param start: start time (anything ``pandas.Timestamp`` accepts,
        e.g. ``"1989"`` or ``"2024-05-10"``)
    :param end: end time (inclusive)
    :param refresh: re-download the (growing) Kp/ap source file instead of
        using the cached copy -- needed to pick up recent months
    :return: Series named after the index, indexed by UTC time. Combine
        several with ``pandas.concat(..., axis=1)`` for
        ``bezpy.storms.find_storms``.
    """
    name = str(name).lower()
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if name in ("kp", "ap"):
        series = _fetch_gfz_table(refresh=refresh)[name]
    elif name == "dst":
        frames = [_fetch_omni2_year(year)
                  for year in range(start.year, end.year + 1)]
        series = pd.concat(frames)["dst"]
    else:
        raise ValueError("name must be one of: kp, ap, dst")
    return series.loc[start:end].rename(name)
