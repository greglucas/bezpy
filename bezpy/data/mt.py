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

Discovery
---------
:func:`search_mt` finds transfer functions by region (and survey / quality)
against the live SPUD archive, and :func:`download_mt_search` downloads the
matches -- so a whole region's MT data is one call, no bundle ids needed::

    bezpy.data.download_mt_search(bbox=(128, 30, 146, 46))   # e.g. Japan

``download_mt``/:func:`download_mt_spud` remain for fetching a single known URL
or SPUD bundle id.
"""
__all__ = [
    "download_mt", "download_mt_spud", "download_mt_search", "search_mt",
    "ingest_mt_xml", "store_mt_xml", "load_mt_site", "load_mt_sites",
    "mt_inventory", "stored_mt_surveys", "mt_survey_of",
    "spud_data_url", "spud_bundle_url", "spud_search_url",
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
# The service lists every EM_TF product (id, location, and full <Site> header)
# as one document. It has no reliable server-side geographic filter, so
# search_mt caches this index and filters it locally.
_SPUD_INDEX_URL = "https://ds.iris.edu/spudservice/emtf"


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
    """Return the human SPUD EMTF *browser* URL for a bounding box.

    For programmatic discovery use :func:`search_mt` / :func:`download_mt_search`
    instead; this just builds the ``ds.iris.edu/spud/emtf`` page URL a person
    would open. ``bbox`` is ``(west, south, east, north)`` in degrees.
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
# Search (discovery against the live SPUD archive)
# ----------------------------------------------------------------------------
def _spud_index_path():
    return get_cache_dir("mt") / "spud_emtf_index.xml"


def fetch_spud_index(refresh=False, session=None, timeout=300):
    """Download (and cache) the SPUD EMTF index of every transfer function.

    The index is one large document listing all ~8000 products with their
    location and site header; :func:`search_mt` filters it locally. It is
    cached on disk and reused until ``refresh=True`` re-downloads it. Returns
    the cached path.
    """
    path = _spud_index_path()
    if refresh or not path.exists():
        import requests
        getter = session.get if session is not None else requests.get
        response = getter(_SPUD_INDEX_URL, timeout=timeout)
        response.raise_for_status()
        path.write_text(response.text, encoding="utf-8")
    return path


# One <EM_TF id="..."> ... </EM_TF> product block.
_EMTF_BLOCK_RE = re.compile(r'<EM_TF\b[^>]*\bid="(\d+)".*?</EM_TF>', re.DOTALL)
# Embedded <a href=...>...</a> links (in <Link>, <OrgUrl>, ...) break XML parsing.
_ANCHOR_RE = re.compile(r'</?a\b[^>]*>', re.IGNORECASE)
# Bare ampersands in free-text fields (e.g. "Nagamachi, & Owada") are not valid
# XML; re-escape any that are not already part of an entity before parsing.
_BARE_AMP_RE = re.compile(r'&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)')


def _parse_index_block(spud_id, block):
    """One search row from an EM_TF index block, or None if it isn't a site TF."""
    clean = _BARE_AMP_RE.sub("&amp;", _ANCHOR_RE.sub("", block))
    try:
        root = ET.fromstring(clean)
    except ET.ParseError:
        return None
    if _text(root, "SubType") not in ("MT_TF", None):
        return None
    header = _parse_header(root)
    if header["latitude"] is None or header["longitude"] is None:
        return None
    return {
        "spud_id": int(spud_id),
        "code": header["code"],
        "survey": header["survey"] or header["network"] or "unknown",
        "country": header["country"],
        "year": header["year"],
        "latitude": header["latitude"],
        "longitude": header["longitude"],
        "rating": header["rating"],
        "product_id": header["product_id"],
    }


def search_mt(bbox=None, survey=None, min_rating=None, country=None,
              refresh=False, session=None, timeout=300):
    """Search the SPUD EMTF archive for transfer functions to download.

    Parses the cached SPUD index (see :func:`fetch_spud_index`) and filters it;
    the returned ``spud_id`` column feeds :func:`download_mt_spud` (or use
    :func:`download_mt_search` to fetch every match in one call).

    Parameters
    ----------
    bbox : tuple, optional
        ``(west, south, east, north)`` in degrees; keep sites inside it.
    survey : str, optional
        Case-insensitive substring match on the survey/project name
        (e.g. ``"JMA"`` or ``"INTERMAGNET"`` for the Japanese observatories).
    min_rating : int, optional
        Drop sites rated below this (1-5); sites with no rating are kept.
    country : str, optional
        Case-insensitive substring match on the country field.
    refresh : bool, optional
        Re-download the index instead of using the cached copy.

    Returns
    -------
    pandas.DataFrame
        One row per matching transfer function (``spud_id``, ``code``,
        ``survey``, ``country``, ``year``, ``latitude``, ``longitude``,
        ``rating``, ``product_id``), sorted by survey then code.
    """
    text = html.unescape(fetch_spud_index(refresh=refresh, session=session,
                                          timeout=timeout).read_text(encoding="utf-8"))
    rows = []
    for match in _EMTF_BLOCK_RE.finditer(text):
        row = _parse_index_block(match.group(1), match.group(0))
        if row is None:
            continue
        if survey is not None and survey.lower() not in \
                f"{row['survey']} {row['product_id']}".lower():
            continue
        if country is not None and country.lower() not in str(row["country"]).lower():
            continue
        if min_rating is not None and row["rating"] is not None \
                and row["rating"] < min_rating:
            continue
        if bbox is not None:
            west, south, east, north = bbox
            if not (west <= row["longitude"] <= east
                    and south <= row["latitude"] <= north):
                continue
        rows.append(row)
    columns = ["spud_id", "code", "survey", "country", "year",
               "latitude", "longitude", "rating", "product_id"]
    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df = df.sort_values(["survey", "code"]).reset_index(drop=True)
    return df


def download_mt_search(bbox=None, survey=None, min_rating=None, country=None,
                       refresh=False, session=None, timeout=60, verbose=False):
    """Search the SPUD archive and download every matching transfer function.

    Runs :func:`search_mt` with the given filters, then downloads each result
    (skipping sites already stored under the same survey). Returns the list of
    stored paths. See :func:`search_mt` for the filter arguments.
    """
    import requests

    catalog = search_mt(bbox=bbox, survey=survey, min_rating=min_rating,
                        country=country, refresh=refresh, session=session)
    getter_session = session if session is not None else requests.Session()
    stored = []
    for row in catalog.itertuples():
        dest = _site_path(row.survey, row.code)
        if dest.exists():
            stored.append(dest)
            continue
        if verbose:
            print(f"downloading {row.code} ({row.survey}) [spud {row.spud_id}]")
        stored.append(download_mt_spud(row.spud_id, survey=row.survey,
                                       session=getter_session, timeout=timeout))
    return stored


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
