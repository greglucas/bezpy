"""Download and load high-voltage transmission line geometries.

Transmission line geometries for the United States are published by HIFLD
(Homeland Infrastructure Foundation-Level Data) as an ArcGIS feature service.
This module queries such a service, caches the result as GeoJSON, and returns
:class:`bezpy.tl.TransmissionLine` objects ready for voltage calculations.

Cached as::

    <cache>/tl/<name>.geojson

Network-free, unit-testable pieces:

* :func:`build_query_url` -- assemble an ArcGIS query URL (pure).
* :func:`parse_geojson` -- turn a GeoJSON FeatureCollection into
  :class:`~bezpy.tl.TransmissionLine` objects (pure).

Note
----
The default service URL points at a public HIFLD-derived layer. ArcGIS
endpoints are re-hosted from time to time; pass ``url=`` to target a different
service (any ArcGIS feature layer with LineString geometry works). Check the
dataset's license before redistribution.
"""
__all__ = [
    "download_transmission_lines", "load_transmission_lines", "read_geojson",
    "parse_geojson", "build_query_url", "DEFAULT_SERVICE_URL",
]

import json

from ._cache import get_cache_dir

# A public HIFLD-derived Electric Power Transmission Lines feature layer.
# Fields include VOLTAGE, VOLT_CLASS, OWNER, ID, SUB_1, SUB_2.
DEFAULT_SERVICE_URL = (
    "https://arcgis.netl.doe.gov/server/rest/services/Hosted/"
    "Energy_Transition_Atlas_493d6/FeatureServer/18"
)

# ArcGIS feature services page results; this is a safe default page size.
_PAGE_SIZE = 2000


def build_query_url(service_url, where="1=1", out_fields="*", offset=0,
                    count=_PAGE_SIZE, bbox=None):
    """Build an ArcGIS ``/query`` URL returning GeoJSON.

    Parameters
    ----------
    service_url : str
        The feature layer URL (ending in a layer number).
    where : str
        SQL ``WHERE`` clause. Default ``"1=1"`` (all features).
    out_fields : str
        Comma-separated attribute fields, or ``"*"`` for all.
    offset, count : int
        Pagination controls (``resultOffset`` / ``resultRecordCount``).
    bbox : tuple, optional
        ``(west, south, east, north)`` in degrees, to spatially filter.
    """
    params = {
        "where": where,
        "outFields": out_fields,
        "f": "geojson",
        "returnGeometry": "true",
        "outSR": "4326",
        "resultOffset": int(offset),
        "resultRecordCount": int(count),
    }
    if bbox is not None:
        west, south, east, north = bbox
        params["geometry"] = f"{west},{south},{east},{north}"
        params["geometryType"] = "esriGeometryEnvelope"
        params["inSR"] = "4326"
        params["spatialRel"] = "esriSpatialRelIntersects"

    # URL-encode the WHERE clause (spaces, >=, etc.) minimally.
    from urllib.parse import urlencode
    return f"{service_url.rstrip('/')}/query?{urlencode(params)}"


def parse_geojson(geojson):
    """Turn a GeoJSON FeatureCollection into ``TransmissionLine`` objects.

    ``LineString`` features become one line each; ``MultiLineString`` features
    become one line per part (all sharing the feature's attributes).
    """
    from ..tl import TransmissionLine

    lines = []
    for feature in geojson.get("features", []):
        geometry = feature.get("geometry") or {}
        properties = feature.get("properties", {})
        gtype = geometry.get("type")
        coords = geometry.get("coordinates")
        if gtype == "LineString":
            lines.append(TransmissionLine(coords, metadata=properties))
        elif gtype == "MultiLineString":
            for part in coords:
                lines.append(TransmissionLine(part, metadata=properties))
        # Points/polygons/None are silently skipped -- not transmission lines.
    return lines


def _tl_path(name):
    return get_cache_dir("tl") / f"{name}.geojson"


def read_geojson(path):
    """Read a local GeoJSON file into a list of ``TransmissionLine`` objects."""
    with open(path) as handle:
        return parse_geojson(json.load(handle))


def load_transmission_lines(name="transmission_lines"):
    """Load a previously downloaded transmission-line GeoJSON from the cache."""
    path = _tl_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached transmission lines named {name!r}. "
            f"Download them first with bezpy.data.download_transmission_lines().")
    return read_geojson(path)


def download_transmission_lines(bbox=None, min_voltage=None, where="1=1",
                                url=DEFAULT_SERVICE_URL, name="transmission_lines",
                                out_fields="*", session=None, timeout=120,
                                max_features=None, verbose=False):
    """Download transmission line geometries and cache them as GeoJSON.

    Parameters
    ----------
    bbox : tuple, optional
        ``(west, south, east, north)`` in degrees; only lines intersecting the
        box are returned. Strongly recommended -- the full US dataset is large.
    min_voltage : float, optional
        Keep only lines with ``VOLTAGE >=`` this (kV). Added to ``where``.
    where : str, optional
        Additional SQL ``WHERE`` clause.
    url : str, optional
        ArcGIS feature layer URL (see :data:`DEFAULT_SERVICE_URL`).
    name : str, optional
        Cache file name (``<cache>/tl/<name>.geojson``).
    out_fields, session, timeout, max_features, verbose
        Query/HTTP controls.

    Returns
    -------
    list of bezpy.tl.TransmissionLine
    """
    import requests

    getter = session.get if session is not None else requests.get

    clauses = [where] if where and where != "1=1" else []
    if min_voltage is not None:
        clauses.append(f"VOLTAGE>={min_voltage}")
    where_clause = " AND ".join(clauses) if clauses else "1=1"

    features = []
    offset = 0
    while True:
        query_url = build_query_url(url, where=where_clause,
                                    out_fields=out_fields, offset=offset,
                                    count=_PAGE_SIZE, bbox=bbox)
        if verbose:
            print(f"Fetching transmission lines offset={offset}")
        response = getter(query_url, timeout=timeout)
        response.raise_for_status()
        page = response.json()

        page_features = page.get("features", [])
        features.extend(page_features)

        # Stop when the server signals the last page (or returns a short page).
        exceeded = page.get("exceededTransferLimit") or page.get(
            "properties", {}).get("exceededTransferLimit")
        if not page_features or not exceeded:
            break
        if max_features is not None and len(features) >= max_features:
            features = features[:max_features]
            break
        offset += len(page_features)

    collection = {"type": "FeatureCollection", "features": features}
    path = _tl_path(name)
    path.write_text(json.dumps(collection))
    if verbose:
        print(f"Stored {len(features)} features to {path}")
    return parse_geojson(collection)
