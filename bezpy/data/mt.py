"""Download, store, and load magnetotelluric transfer functions (EMTF XML).

Magnetotelluric transfer functions from the IRIS/EarthScope SPUD EMTF archive
are distributed as EMTF XML files -- the same format :func:`bezpy.mt.read_xml`
already parses. This module fetches those files and organizes them on disk by
*survey* (the field campaign / deployment they belong to), which is the natural
provenance grouping for MT data::

    <cache>/mt/<survey>/<CODE>.xml

For example::

    mt/USArray_TA/TEST01.xml
    mt/SnakeRiverPlain/SRP12.xml

The survey is read from the XML's ``<Site><Survey>`` element (falling back to
``<Project>`` and then the IRIS network code). Loading a stored site returns a
fully-parsed :class:`bezpy.mt.Site3d`.

Notes
-----
``download_mt`` works with any direct EMTF-XML URL, including IRIS SPUD data
URLs (see :func:`spud_data_url`). Programmatic bounding-box *discovery* against
the live SPUD service is intentionally left thin (:func:`spud_search_url`)
because that query interface has changed over time and should be confirmed
against the current service before relying on it; downloading and organizing
known files is fully supported.
"""
__all__ = [
    "download_mt", "download_mt_spud", "ingest_mt_xml", "store_mt_xml",
    "load_mt_site", "load_mt_sites", "mt_inventory", "stored_mt_surveys",
    "mt_survey_of", "spud_data_url", "spud_bundle_url", "spud_search_url",
]

import html
import re
import warnings
import xml.etree.ElementTree as ET

import pandas as pd

from ._cache import get_cache_dir

# IRIS SPUD service URL templates.
#   - a TF "bundle" (what the ds.iris.edu/spud/emtf/<id> page shows) has a
#     metadata document at /spudservice/emtf/<bundle_id> listing its parts;
#   - the EMTF XML itself is one of those parts, at /spudservice/data/<data_id>.
# The bundle's "primarydata" is a PNG plot, NOT the XML -- the XML is the
# SourceData/Attachment whose MIME type is application/xml, which is why
# download_mt_spud resolves the bundle rather than guessing a data id.
_SPUD_DATA_URL = "https://ds.iris.edu/spudservice/data/{data_id}"
_SPUD_BUNDLE_URL = "https://ds.iris.edu/spudservice/emtf/{bundle_id}"
_SPUD_SEARCH_URL = "https://ds.iris.edu/spud/emtf"


def spud_data_url(data_id):
    """Return the IRIS SPUD URL that serves the EMTF XML at a given *data* id.

    ``data_id`` is the numeric SPUD data id of the XML attachment (not the
    bundle/page id shown at ``ds.iris.edu/spud/emtf/<id>``, and not the product
    id). Use :func:`download_mt_spud` to resolve a bundle id automatically.
    """
    return _SPUD_DATA_URL.format(data_id=data_id)


def spud_bundle_url(bundle_id):
    """Return the SPUD metadata URL for a TF bundle (page) id."""
    return _SPUD_BUNDLE_URL.format(bundle_id=bundle_id)


def spud_search_url(bbox=None):
    """Return a SPUD EMTF search URL for a bounding box (best-effort).

    ``bbox`` is ``(west, south, east, north)`` in degrees. The live SPUD query
    vocabulary should be confirmed before depending on this; it is provided as a
    starting point for discovery.
    """
    if bbox is None:
        return _SPUD_SEARCH_URL
    west, south, east, north = bbox
    return (f"{_SPUD_SEARCH_URL}?minlat={south}&maxlat={north}"
            f"&minlon={west}&maxlon={east}")


# ----------------------------------------------------------------------------
# XML header parsing + path helpers
# ----------------------------------------------------------------------------
def _text(base, path):
    """Return the text of a child element, or None."""
    if base is None:
        return None
    element = base.find(path)
    return element.text if element is not None else None


def _parse_header(root):
    """Extract the survey/code/location metadata used for filing and inventory."""
    site = root.find("Site")
    if site is None:
        raise ValueError("EMTF XML has no <Site> element; not a valid TF file.")

    # Network code lives in the external metadata URL, e.g. .../EM/TEST01 -> EM.
    network = None
    external = root.find("ExternalUrl")
    url = _text(external, "Url") if external is not None else None
    if url:
        parts = url.rstrip("/").split("/")
        network = parts[-2] if len(parts) >= 2 else None

    loc = site.find("Location")
    quality = site.find("DataQualityNotes")

    def _float(value):
        return float(value) if value not in (None, "") else None

    def _int(value):
        return int(value) if value not in (None, "") else None

    return {
        "code": _text(site, "Id"),
        "survey": _text(site, "Survey") or _text(site, "Project"),
        "project": _text(site, "Project"),
        "network": network,
        "country": _text(site, "Country"),
        "year": _text(site, "YearCollected"),
        "latitude": _float(_text(loc, "Latitude")),
        "longitude": _float(_text(loc, "Longitude")),
        "rating": _int(_text(quality, "Rating")),
        "product_id": _text(root, "ProductId"),
    }


def _sanitize(name):
    """Make a filesystem-safe survey directory name."""
    cleaned = "".join(c if (c.isalnum() or c in "-._") else "_" for c in str(name))
    return cleaned.strip("_") or "unknown"


def mt_survey_of(xml_text):
    """Return the survey name an EMTF XML string would be filed under."""
    header = _parse_header(ET.fromstring(xml_text.encode("utf-8")))
    return _sanitize(header["survey"] or header["network"] or "unknown")


def _survey_dir(survey, create=True):
    return get_cache_dir(f"mt/{_sanitize(survey)}", create=create)


def _site_path(survey, code):
    return _survey_dir(survey) / f"{code}.xml"


# ----------------------------------------------------------------------------
# Store / download
# ----------------------------------------------------------------------------
def store_mt_xml(xml_text, survey=None):
    """Store an EMTF XML string in the survey-organized archive.

    Parameters
    ----------
    xml_text : str
        The EMTF XML document.
    survey : str, optional
        Override the survey directory. By default it is read from the XML.

    Returns
    -------
    pathlib.Path
        Where the file was written.
    """
    root = ET.fromstring(xml_text.encode("utf-8"))
    header = _parse_header(root)
    code = header["code"]
    if not code:
        raise ValueError("EMTF XML has no <Site><Id>; cannot determine a code.")

    survey = survey or header["survey"] or header["network"] or "unknown"
    path = _site_path(survey, code)
    path.write_text(xml_text, encoding="utf-8")
    return path


def ingest_mt_xml(paths, survey=None):
    """Store existing local EMTF XML files into the archive.

    Returns the list of written paths.
    """
    if isinstance(paths, str):
        paths = [paths]
    written = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            written.append(store_mt_xml(handle.read(), survey=survey))
    return written


def download_mt(url, survey=None, session=None, timeout=60):
    """Download one EMTF XML file from a URL and store it by survey.

    ``url`` may be any direct EMTF-XML URL, including an IRIS SPUD data URL
    (see :func:`spud_data_url`). To download from a SPUD *bundle/page* id
    instead, use :func:`download_mt_spud`. Returns the stored path.
    """
    import requests

    getter = session.get if session is not None else requests.get
    response = getter(url, timeout=timeout)
    response.raise_for_status()
    return store_mt_xml(response.text, survey=survey)


def _resolve_spud_xml_url(bundle_id, getter, timeout):
    """Resolve a SPUD bundle id to the download URL of its EMTF XML attachment."""
    response = getter(spud_bundle_url(bundle_id), timeout=timeout)
    response.raise_for_status()
    # The service HTML-escapes the metadata XML inside <pre> for browser viewing;
    # its DataLinks also embed HTML <a> tags, so this document does not parse
    # cleanly as XML -- resolve the XML attachment's data id by pattern instead.
    text = html.unescape(response.text)

    # Each attachment is a <SourceData>/<Attachment id="..."> block; pick the one
    # whose MIME type is application/xml (or whose filename ends in .xml). The
    # bundle's "primarydata" is a plot image, not the transfer function XML.
    block_re = re.compile(r'<(SourceData|Attachment)\b[^>]*\bid="(\d+)"[^>]*>(.*?)</\1>',
                          re.DOTALL | re.IGNORECASE)
    for match in block_re.finditer(text):
        block = match.group(3)
        is_xml = ("application/xml" in block.lower()
                  or re.search(r'<Filename>[^<]*\.xml\s*<', block, re.IGNORECASE))
        if is_xml:
            data_id = match.group(2)
            link = re.search(r'/spudservice/data/(\d+)', block)
            if link:
                data_id = link.group(1)
            return spud_data_url(data_id)
    raise ValueError(
        f"Could not find an EMTF XML attachment in SPUD bundle {bundle_id}.")


def download_mt_spud(bundle_id, survey=None, session=None, timeout=60):
    """Download an MT transfer function from an IRIS SPUD bundle/page id.

    ``bundle_id`` is the numeric id from a SPUD EMTF page URL
    (``ds.iris.edu/spud/emtf/<bundle_id>``). This resolves the bundle's XML
    attachment and stores it by survey. Returns the stored path.
    """
    import requests

    getter = session.get if session is not None else requests.get
    xml_url = _resolve_spud_xml_url(bundle_id, getter, timeout)
    response = getter(xml_url, timeout=timeout)
    response.raise_for_status()
    return store_mt_xml(response.text, survey=survey)


# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------
def stored_mt_surveys():
    """Return the sorted list of survey directories present in the archive."""
    root = get_cache_dir("mt", create=False)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _find_site_files(code=None, survey=None):
    """Yield stored XML paths, optionally filtered by code and/or survey."""
    root = get_cache_dir("mt", create=False)
    if not root.exists():
        return
    surveys = [survey] if survey is not None else stored_mt_surveys()
    for survey_name in surveys:
        survey_dir = _survey_dir(survey_name, create=False)
        if not survey_dir.exists():
            continue
        pattern = f"{code}.xml" if code is not None else "*.xml"
        yield from sorted(survey_dir.glob(pattern))


def load_mt_site(code, survey=None):
    """Load one stored MT site as a :class:`bezpy.mt.Site3d`.

    Raises ``FileNotFoundError`` if the code is not stored. If the same code
    exists in multiple surveys, pass ``survey`` to disambiguate (otherwise the
    first match is returned with a warning).
    """
    from ..mt import read_xml

    matches = list(_find_site_files(code=code, survey=survey))
    if not matches:
        raise FileNotFoundError(
            f"No stored MT site {code!r}"
            + (f" in survey {survey!r}" if survey else "")
            + ". Download it first with bezpy.data.download_mt().")
    if len(matches) > 1:
        warnings.warn(
            f"MT site {code!r} exists in multiple surveys "
            f"({', '.join(p.parent.name for p in matches)}); loading "
            f"{matches[0].parent.name}. Pass survey=... to choose.",
            stacklevel=2)
    return read_xml(str(matches[0]))


def load_mt_sites(survey=None, bbox=None, min_rating=None):
    """Load stored MT sites as a list of :class:`bezpy.mt.Site3d`.

    Parameters
    ----------
    survey : str, optional
        Restrict to a single survey.
    bbox : tuple, optional
        ``(west, south, east, north)`` in degrees; only sites inside are
        returned.
    min_rating : int, optional
        Only return sites with data-quality rating >= this value.
    """
    from ..mt import read_xml

    sites = []
    for path in _find_site_files(survey=survey):
        site = read_xml(str(path))
        if min_rating is not None and (site.rating is None or site.rating < min_rating):
            continue
        if bbox is not None:
            west, south, east, north = bbox
            if not (west <= site.longitude <= east and south <= site.latitude <= north):
                continue
        sites.append(site)
    return sites


def mt_inventory(survey=None):
    """Summarize the stored MT archive as a :class:`pandas.DataFrame`.

    One row per stored site (code, survey, network, country, year, location,
    rating, path). Empty (with the expected columns) if nothing is stored.
    """
    columns = ["code", "survey", "network", "country", "year",
               "latitude", "longitude", "rating", "path"]
    rows = []
    for path in _find_site_files(survey=survey):
        header = _parse_header(ET.parse(path).getroot())
        rows.append({
            "code": header["code"],
            "survey": path.parent.name,
            "network": header["network"],
            "country": header["country"],
            "year": header["year"],
            "latitude": header["latitude"],
            "longitude": header["longitude"],
            "rating": header["rating"],
            "path": str(path),
        })
    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df = df.sort_values(["survey", "code"]).reset_index(drop=True)
    return df
