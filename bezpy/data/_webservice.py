"""Client for the USGS/NRCan geomagnetic web service.

The USGS Geomagnetism Program serves observatory data at several quality levels
through a simple HTTP web service (https://geomag.usgs.gov/ws/data/). The same
server also relays a set of Natural Resources Canada (NRCan) observatories, so a
single client covers most of the North-American network.

This module is deliberately split so that the network-free pieces (URL
construction and JSON parsing) can be unit tested without hitting the live
service:

* :func:`build_url` -- assemble a query URL (pure).
* :func:`parse_payload` -- turn the JSON response into arrays (pure).
* :func:`fetch_interval` -- perform the HTTP request and parse it.
* :func:`fetch_dataframe` -- fetch an arbitrary interval, chunking long
  requests, and return a tidy :class:`pandas.DataFrame`.
"""

__all__ = [
    "build_url",
    "parse_payload",
    "fetch_interval",
    "fetch_dataframe",
    "wire_type",
    "default_source",
    "USGS_OBSERVATORIES",
    "NRCAN_OBSERVATORIES",
]

import datetime

import numpy as np
import pandas as pd

from ._http import http_get
from ._levels import canonical_level

DEFAULT_BASE_URL = "https://geomag.usgs.gov"

# USGS-operated observatories (real-time relayed under location code "R0").
USGS_OBSERVATORIES = (
    "BRW",
    "DED",
    "CMO",
    "SIT",
    "SHU",
    "NEW",
    "BOU",
    "FRD",
    "FRN",
    "TUC",
    "BSL",
    "HON",
    "SJG",
    "GUA",
)
# NRCan observatories relayed through the USGS server (location code "R1").
NRCAN_OBSERVATORIES = (
    "ALE",
    "BLC",
    "BRD",
    "CBB",
    "FCC",
    "IQA",
    "MEA",
    "OTT",
    "RES",
    "STJ",
    "SNK",
    "VIC",
    "YKC",
    "EUA",
)

# The service caps a single JSON response at this many samples, where
# samples = n_elements * interval_seconds / sampling_period (IAGA2002 allows
# 576000, but we request JSON). Long intervals are chunked to stay under it.
_JSON_SAMPLE_LIMIT = 144_000

# The service only offers these sampling periods (seconds).
_VALID_SAMPLING_PERIODS = (1, 60, 3600)


def _max_chunk(elements, sampling_period):
    """Largest time chunk that stays under the JSON sample limit.

    Scales with the number of elements and the cadence: e.g. 4 elements of
    one-minute data -> ~22 days per request, but one-second data -> hours.
    """
    max_seconds = _JSON_SAMPLE_LIMIT / len(elements) * int(sampling_period)
    # Keep a 10% safety margin; never smaller than one hour.
    return datetime.timedelta(seconds=max(3600, int(max_seconds * 0.9)))


def default_source(code):
    """Return the provenance label for an observatory's operator."""
    return "nrcan" if code.upper() in NRCAN_OBSERVATORIES else "usgs"


def wire_type(code, level):
    """Map a canonical quality level to the service's ``type`` parameter.

    Real-time ``variation`` data is fetched via the EDGE location code
    (``R0`` for USGS, ``R1`` for relayed NRCan observatories), which is the
    proven path for near-real-time data. Higher quality levels are requested by
    name (``adjusted``, ``quasi-definitive``, ``definitive``); ``provisional``
    is not distinctly served here and falls back to the location code.
    """
    level = canonical_level(level)
    if level in ("variation", "provisional"):
        return "R1" if code.upper() in NRCAN_OBSERVATORIES else "R0"
    if level == "quasi-definitive":
        return "quasi-definitive"
    return level  # "adjusted" or "definitive"


def _isoformat(when):
    """Format a timestamp as the ``...Z`` UTC string the service expects."""
    ts = pd.Timestamp(when)
    # Treat naive timestamps as UTC; convert aware ones to UTC.
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_url(
    code,
    start,
    end,
    elements=("X", "Y", "Z", "F"),
    sampling_period=60,
    level="variation",
    base_url=DEFAULT_BASE_URL,
):
    """Build a geomagnetic web-service query URL.

    Parameters
    ----------
    code : str
        IAGA observatory code (e.g. ``"BOU"``).
    start, end : datetime-like
        Interval to request. Naive timestamps are interpreted as UTC.
    elements : sequence of str, optional
        Field components to request. Default ``("X", "Y", "Z", "F")``.
    sampling_period : int, optional
        Sampling period in seconds (60 for one-minute data, 1 for one-second).
    level : str, optional
        Data quality level (see :mod:`._levels`). Default ``"variation"``.
    base_url : str, optional
        Root URL of the service.
    """
    params = {
        "id": code.upper(),
        "type": wire_type(code, level),
        "elements": ",".join(elements),
        "sampling_period": int(sampling_period),
        "starttime": _isoformat(start),
        "endtime": _isoformat(end),
        "format": "json",
    }
    query = "&".join(f"{key}={value}" for key, value in params.items())
    return f"{base_url.rstrip('/')}/ws/data/?{query}"


def parse_payload(payload):
    """Parse a web-service JSON response into times, values, and metadata.

    Returns
    -------
    times : numpy.ndarray of datetime64[ns]
        Sample times (UTC).
    values : dict of {str: numpy.ndarray}
        Element id (``"X"``, ``"Y"``, ...) mapped to float arrays. The service's
        null fill values are converted to ``NaN``.
    metadata : dict
        Station metadata pulled from the response where available.
    """
    times = pd.to_datetime(payload["times"]).to_numpy(dtype="datetime64[ns]")

    values = {}
    for element in payload.get("values", []):
        # Prefer the explicit element id; fall back to the metadata channel.
        name = element.get("id")
        if name is None:
            name = element.get("metadata", {}).get("element")
        if name is None:
            continue
        # None -> NaN so downstream code has a consistent missing-data marker.
        arr = np.array([np.nan if v is None else v for v in element["values"]], dtype=float)
        values[name.upper()] = arr

    meta_in = payload.get("metadata", {})
    intermag = meta_in.get("intermagnet", {})
    geo = intermag.get("imo", {}).get("coordinates", [None, None, None])
    metadata = {
        "station_name": intermag.get("imo", {}).get("name"),
        "longitude": geo[0] if len(geo) > 0 else None,
        "latitude": geo[1] if len(geo) > 1 else None,
        "elevation": geo[2] if len(geo) > 2 else None,
        "sampling_period": meta_in.get("sampling_period"),
    }
    return times, values, metadata


def _payload_to_frame(payload):
    """Convert a parsed payload into a DataFrame indexed by time (UTC)."""
    times, values, metadata = parse_payload(payload)
    df = pd.DataFrame(values, index=pd.DatetimeIndex(times, name="time"))
    df.attrs.update({k: v for k, v in metadata.items() if v is not None})
    return df


def fetch_interval(
    code,
    start,
    end,
    elements=("X", "Y", "Z", "F"),
    sampling_period=60,
    level="variation",
    base_url=DEFAULT_BASE_URL,
    timeout=60,
    session=None,
):
    """Fetch a single interval from the web service (no chunking).

    Returns a :class:`pandas.DataFrame` indexed by UTC time with one column per
    requested element. Requires the ``requests`` library.
    """
    url = build_url(
        code,
        start,
        end,
        elements=elements,
        sampling_period=sampling_period,
        level=level,
        base_url=base_url,
    )
    response = http_get(url, timeout=timeout, session=session)
    return _payload_to_frame(response.json())


def fetch_dataframe(
    code,
    start,
    end,
    elements=("X", "Y", "Z", "F"),
    sampling_period=60,
    level="variation",
    base_url=DEFAULT_BASE_URL,
    timeout=60,
    session=None,
    chunk=None,
    verbose=False,
):
    """Fetch an arbitrary interval, chunking long requests.

    Long intervals are split into pieces sized to stay under the service's JSON
    sample limit (or ``chunk`` if given) and concatenated. Duplicate timestamps
    at chunk boundaries are removed. Returns a :class:`pandas.DataFrame` indexed
    by UTC time.
    """
    if int(sampling_period) not in _VALID_SAMPLING_PERIODS:
        raise ValueError(
            f"sampling_period must be one of {_VALID_SAMPLING_PERIODS} seconds, "
            f"got {sampling_period}."
        )
    if chunk is None:
        chunk = _max_chunk(elements, sampling_period)

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if end <= start:
        raise ValueError(f"end ({end}) must be after start ({start})")

    frames = []
    metadata = {}
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + chunk, end)
        if verbose:
            print(f"Fetching {code} [{level}] {chunk_start} -> {chunk_end}")
        frame = fetch_interval(
            code,
            chunk_start,
            chunk_end,
            elements=elements,
            sampling_period=sampling_period,
            level=level,
            base_url=base_url,
            timeout=timeout,
            session=session,
        )
        frames.append(frame)
        metadata.update(frame.attrs)
        chunk_start = chunk_end

    if not frames:
        combined = pd.DataFrame(columns=list(elements))
        combined.index = pd.DatetimeIndex([], name="time")
    else:
        combined = pd.concat(frames)
        # Drop duplicate boundary samples, keeping the first occurrence, sorted.
        combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    combined.attrs.update(metadata)
    return combined
