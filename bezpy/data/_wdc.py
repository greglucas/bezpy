"""Client for the BGS World Data Centre (WDC) geomagnetic API.

The World Data Centre for Geomagnetism, Edinburgh (British Geological Survey)
holds the *definitive* observatory archive, including the century-long records
of observatories such as Kakioka (KAK, back to 1913) that long predate the
real-time web services. It serves them from a REST API at ``wdcapi.bgs.ac.uk``
as zipped IAGA-2002 files, which bezpy already parses
(:func:`bezpy.mag.read_iaga`) -- so this client is URL construction, an unzip,
and reuse of that reader.

The archive covers many observatories worldwide (not only Japan); it is the
source to reach for whenever the record needed predates USGS/INTERMAGNET
coverage.

Network-free, unit-testable pieces:

* :func:`build_url` -- assemble a zipped-data request URL (pure).
* :func:`parse_zip` -- turn a returned zip archive into an XYZF DataFrame
  (pure).
"""

__all__ = ["build_url", "parse_zip", "fetch_dataframe", "DEFAULT_BASE_URL"]

import io
import tempfile
import zipfile

import pandas as pd

from ._http import http_get

DEFAULT_BASE_URL = "https://wdcapi.bgs.ac.uk"
PROVIDER = "WDC for Geomagnetism, Edinburgh (wdcapi.bgs.ac.uk)"

# Sampling period (s) -> API cadence path and the years-per-request used to
# bound each download. Minute files are large (~0.5 MB/month), so a request is
# capped at one year; hourly data is tiny and fetched a decade at a time.
_CADENCE = {60: ("minute", 1), 3600: ("hour", 10)}


def build_url(
    code,
    start,
    end,
    cadence="minute",
    orientation="XYZF",
    file_format="iaga",
    base_url=DEFAULT_BASE_URL,
):
    """Build a WDC zipped-data request URL.

    ``start``/``end`` are inclusive dates (the service returns whole monthly
    files overlapping the range). ``cadence`` is ``"minute"`` or ``"hour"``.
    """
    params = {
        "start_dt": pd.Timestamp(start).strftime("%Y-%m-%d"),
        "end_dt": pd.Timestamp(end).strftime("%Y-%m-%d"),
        "file_format": file_format,
        "orientation": orientation,
    }
    if cadence == "minute":
        params["group_by"] = "month"
    query = "&".join(f"{key}={value}" for key, value in params.items())
    return f"{base_url.rstrip('/')}/data/zipped-data/{cadence}/{code.upper()}?{query}"


def parse_zip(content):
    """Parse a WDC zip archive of IAGA-2002 files into one XYZF DataFrame.

    The archive holds one IAGA data file per month (plus ``_REQUEST_SUMMARY``
    and acknowledgement files, which are skipped). Station metadata from the
    first file's header is attached to ``df.attrs``.
    """
    from ..mag import read_iaga

    archive = zipfile.ZipFile(io.BytesIO(content))
    data_names = sorted(
        n
        for n in archive.namelist()
        if not n.split("/")[-1].startswith("_")
        and n.lower().endswith((".min", ".hor", ".sec", ".day"))
    )
    frames, header = [], None
    for name in data_names:
        # read_iaga reads from a path; stage the member in a temp file and
        # reuse the well-tested IAGA reader rather than duplicating it.
        with tempfile.NamedTemporaryFile("wb", suffix=".min") as handle:
            handle.write(archive.read(name))
            handle.flush()
            frame, this_header = read_iaga(handle.name, return_xyzf=True, return_header=True)
        header = header or this_header
        frames.append(frame)

    if not frames:
        raise ValueError("WDC archive contained no IAGA-2002 data files")
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    if header is not None:
        df.attrs.update(
            {
                "station_name": header.get("station name"),
                "latitude": _float(header.get("geodetic latitude")),
                "longitude": _float(header.get("geodetic longitude")),
                "provider": PROVIDER,
            }
        )
    return df


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_dataframe(
    code,
    start,
    end,
    sampling_period=60,
    level="definitive",
    base_url=DEFAULT_BASE_URL,
    timeout=180,
    session=None,
    verbose=False,
):
    """Fetch an interval from the WDC API, chunking long requests.

    Returns a :class:`pandas.DataFrame` indexed by UTC time (XYZF columns),
    trimmed to ``[start, end]``. ``sampling_period`` selects the cadence: 60 s
    (minute, the default) or 3600 s (hourly, for the longest historical spans).
    """
    if int(sampling_period) not in _CADENCE:
        raise ValueError(
            f"WDC source supports minute (60 s) and hourly (3600 s) data; "
            f"got sampling_period={sampling_period}."
        )
    cadence, chunk_years = _CADENCE[int(sampling_period)]

    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end <= start:
        raise ValueError(f"end ({end}) must be after start ({start})")

    frames = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + pd.DateOffset(years=chunk_years), end)
        url = build_url(code, chunk_start, chunk_end, cadence=cadence, base_url=base_url)
        if verbose:
            print(f"Fetching {code} [WDC {cadence}] {chunk_start.date()} -> {chunk_end.date()}")
        response = http_get(url, timeout=timeout, session=session)
        frames.append(parse_zip(response.content))
        chunk_start = chunk_end

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.loc[(df.index >= start) & (df.index <= end)]
    if frames:
        df.attrs.update(frames[0].attrs)
    return df
