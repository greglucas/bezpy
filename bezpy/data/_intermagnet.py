"""Client for the INTERMAGNET GIN web service.

INTERMAGNET is the global network of geomagnetic observatories (~150 stations
worldwide) and is the most complete single source of observatory data. Its
Geomagnetic Information Node (GIN) serves data as IAGA-2002 text, which bezpy
already knows how to parse (:func:`bezpy.mag.read_iaga`), so this client is
mostly URL construction plus reuse of that reader.

Network-free, unit-testable pieces:

* :func:`build_url` -- assemble a GIN query URL (pure).
* :func:`parse_iaga_text` -- parse an IAGA-2002 response body (pure).

Note
----
The GIN endpoint and its exact parameter vocabulary have changed over the
years and access has at times required registration. The defaults here follow
the current public GINServices interface; pass ``base_url=`` to point at a
different node if needed.
"""
__all__ = ["build_url", "parse_iaga_text", "fetch_dataframe",
           "publication_state", "DEFAULT_BASE_URL"]

import datetime
import math
import tempfile

import pandas as pd

from ._levels import BEST, canonical_level

DEFAULT_BASE_URL = "https://imag-data.bgs.ac.uk/GIN_V1/GINServices"

# GIN request-duration caps. The service *spec* allows 366 days of minute
# means (31 for one-second data), but in practice the GIN resets the
# connection on large responses: empirically (2026-07) 92 days of minute data
# (~10 MB) is served reliably while 183+ days is reset mid-transfer. Chunks
# are also halved adaptively on transport errors (see fetch_dataframe), so
# these are starting points, not hard assumptions.
_MAX_CHUNK_DAYS_MINUTE = 92
_MAX_CHUNK_DAYS_SECOND = 1


def publication_state(level):
    """Map a bezpy quality level to a GIN ``publicationState`` value.

    The GIN vocabulary is ``reported`` (raw), ``adjusted`` (edited real-time),
    ``quasi-def``, ``definitive``, and ``best-avail``. bezpy's ``variation``
    maps to ``reported`` and ``provisional`` to ``adjusted`` (the GIN has no
    distinct provisional state).
    """
    if level == BEST:
        return "best-avail"
    level = canonical_level(level)
    return {
        "variation": "reported",
        "adjusted": "adjusted",
        "provisional": "adjusted",
        "quasi-definitive": "quasi-def",
        "definitive": "definitive",
    }[level]


def _samples_per_day(sampling_period):
    """Translate a sampling period in seconds to the GIN cadence keyword."""
    if int(sampling_period) == 1:
        return "Second"
    if int(sampling_period) == 60:
        return "Minute"
    raise ValueError(
        "INTERMAGNET GIN serves 'Minute' (60 s) or 'Second' (1 s) data; "
        f"got sampling_period={sampling_period}.")


def _max_chunk_days(sampling_period):
    """Per-cadence request-duration cap in whole days."""
    return _MAX_CHUNK_DAYS_SECOND if int(sampling_period) == 1 else _MAX_CHUNK_DAYS_MINUTE


def build_url(code, start, duration_days, sampling_period=60,
              level="best", base_url=DEFAULT_BASE_URL):
    """Build a GIN GetData query URL.

    Parameters
    ----------
    code : str
        IAGA observatory code.
    start : datetime-like
        Start date (UTC). Only the date portion is used by the GIN.
    duration_days : int
        Number of days to request.
    sampling_period : int, optional
        Sampling period in seconds (60 = minute, 1 = second).
    level : str, optional
        Quality level, or ``"best"`` for the GIN "Best available" merge.
    base_url : str, optional
        GIN service endpoint.
    """
    params = {
        "Request": "GetData",
        "format": "iaga2002",
        "observatoryIagaCode": code.upper(),
        "samplesPerDay": _samples_per_day(sampling_period),
        "publicationState": publication_state(level),
        "dataStartDate": pd.Timestamp(start).strftime("%Y-%m-%d"),
        "dataDuration": int(duration_days),
    }
    query = "&".join(f"{key}={value}" for key, value in params.items())
    return f"{base_url}?{query}"


def parse_iaga_text(text):
    """Parse an IAGA-2002 response body into an XYZF DataFrame.

    Raises
    ------
    ValueError
        If the body is not IAGA-2002 (e.g. a GIN error page).
    """
    if "IAGA-2002" not in text[:2000]:
        snippet = text.strip()[:200]
        raise ValueError(
            "INTERMAGNET response is not IAGA-2002 data. The service returned:\n"
            f"{snippet!r}")

    # read_iaga reads from a path; write the body to a temporary file and reuse
    # the well-tested IAGA reader rather than duplicating its parsing logic.
    from ..mag import read_iaga

    with tempfile.NamedTemporaryFile("w", suffix=".min", delete=True) as handle:
        handle.write(text)
        handle.flush()
        return read_iaga(handle.name, return_xyzf=True)


def fetch_dataframe(code, start, end, sampling_period=60, level="best",
                    base_url=DEFAULT_BASE_URL, timeout=120, session=None,
                    verbose=False):
    """Fetch an interval from the GIN, chunking long requests.

    Requests are chunked to the empirically reliable duration and the chunk
    size is halved (down to one day) whenever the GIN drops the connection
    mid-transfer, which it does for large responses regardless of the
    documented duration limits.

    Returns a :class:`pandas.DataFrame` indexed by UTC time (XYZF columns).
    """
    import requests

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if end <= start:
        raise ValueError(f"end ({end}) must be after start ({start})")

    getter = session.get if session is not None else requests.get
    chunk_days = _max_chunk_days(sampling_period)

    frames = []
    chunk_start = start
    failures = 0
    while chunk_start < end:
        # GIN duration is whole days; request enough days to cover the chunk.
        remaining = (end - chunk_start).total_seconds() / 86400
        duration = min(chunk_days, max(1, math.ceil(remaining)))
        url = build_url(code, chunk_start, duration,
                        sampling_period=sampling_period, level=level,
                        base_url=base_url)
        if verbose:
            print(f"Fetching {code} [{level}] {chunk_start.date()} +{duration}d")
        try:
            response = getter(url, timeout=timeout)
            response.raise_for_status()
            text = response.text
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and 400 <= status < 500:
                # A client error is the GIN saying "no such data" (unknown
                # observatory, no data in the window, bad state) -- shrinking
                # the request cannot help, so fail fast.
                raise
            if duration > 1:
                # The GIN resets connections on responses it deems too large
                # (and occasionally on transient load); retry the same window
                # with half the duration.
                chunk_days = max(1, duration // 2)
                if verbose:
                    print(f"  {type(exc).__name__}; retrying with "
                          f"{chunk_days}-day chunks")
                continue
            failures += 1
            if failures < 3:  # transient failure at the 1-day floor
                if verbose:
                    print(f"  {type(exc).__name__}; retry {failures}/2")
                continue
            raise
        failures = 0
        frames.append(parse_iaga_text(text))
        chunk_start = chunk_start + datetime.timedelta(days=duration)

    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    # Trim to the exact requested window (chunks are whole days).
    combined = combined.loc[(combined.index >= start) & (combined.index < end)]
    combined.index.name = "time"
    return combined
