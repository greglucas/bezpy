"""Shared HTTP helper for the bezpy.data download clients."""

__all__ = ["http_get"]


def http_get(url, timeout=60, session=None):
    """GET a URL and raise on HTTP errors; returns the ``requests`` response.

    ``requests`` is imported lazily so the rest of bezpy.data works without
    it; pass a ``requests.Session`` to reuse connections across calls.
    """
    import requests

    getter = session.get if session is not None else requests.get
    response = getter(url, timeout=timeout)
    response.raise_for_status()
    return response
