"""Tests for the bezpy.data acquisition/storage subpackage.

These tests never touch the network: HTTP calls are replaced with synthetic-data
generators, and everything else (netCDF store, provenance layout, cross-level
merge, inventory, IAGA/MT ingest) runs against a temporary cache directory.
"""
import urllib.parse as urlparse

import numpy as np
import pandas as pd
import pytest

# The whole subpackage is optional; skip cleanly if the extras aren't installed.
data = pytest.importorskip("bezpy.data")
pytest.importorskip("h5netcdf")

from bezpy.data import _intermagnet as im  # noqa: E402
from bezpy.data import _levels as levels  # noqa: E402
from bezpy.data import _webservice as ws  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def cache(tmp_path):
    """Point the bezpy cache at a temporary directory for the test."""
    data.set_cache_dir(tmp_path)
    yield tmp_path
    data.set_cache_dir(None)


def _synthetic_payload(url):
    """Build a web-service-shaped JSON payload for a URL.

    The X component encodes the requested data ``type`` so tests can tell levels
    apart: definitive data is valued 99 and has its last 5 samples culled (NaN);
    everything else is valued 11 and complete.
    """
    query = urlparse.parse_qs(urlparse.urlparse(url).query)
    start = pd.Timestamp(query["starttime"][0].replace("Z", ""))
    end = pd.Timestamp(query["endtime"][0].replace("Z", ""))
    elements = query["elements"][0].split(",")
    definitive = query["type"][0] == "definitive"

    idx = pd.date_range(start, end, freq="1min", inclusive="left")
    n = len(idx)
    component = {}
    component["X"] = np.full(n, 99.0 if definitive else 11.0)
    if definitive:
        component["X"][-5:] = np.nan  # simulate points culled by definitive QC
    return {
        "times": [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in idx],
        "metadata": {"sampling_period": 60},
        "values": [{"id": e, "values": list(component.get(e, np.zeros(n)))}
                   for e in elements],
    }


class _FakeResponse:
    def __init__(self, *, payload=None, text=None):
        self._payload = payload
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture
def fake_ws(monkeypatch):
    """Replace the USGS HTTP fetch with a deterministic generator."""
    class FakeSession:
        def get(self, url, timeout=None):
            return _FakeResponse(payload=_synthetic_payload(url))

    original = ws.fetch_interval

    def patched(code, start, end, **kwargs):
        kwargs["session"] = FakeSession()
        return original(code, start, end, **kwargs)

    monkeypatch.setattr(ws, "fetch_interval", patched)


# ---------------------------------------------------------------------------
# Data levels
# ---------------------------------------------------------------------------
def test_canonical_level_aliases():
    assert levels.canonical_level("Definitive") == "definitive"
    assert levels.canonical_level("quasidefinitive") == "quasi-definitive"
    assert levels.canonical_level("reported") == "variation"
    assert levels.canonical_level("best", allow_best=True) == "best"


def test_canonical_level_unknown_raises():
    with pytest.raises(ValueError):
        levels.canonical_level("bogus")


def test_level_rank_orders_quality():
    assert levels.level_rank("definitive") > levels.level_rank("quasi-definitive")
    assert levels.level_rank("quasi-definitive") > levels.level_rank("variation")


# ---------------------------------------------------------------------------
# USGS/NRCan web-service helpers (pure)
# ---------------------------------------------------------------------------
def test_build_url_level_to_type():
    # Real-time variation uses the EDGE location code (R0 USGS / R1 NRCan).
    assert "type=R0" in ws.build_url("BOU", "2024-05-10", "2024-05-11")
    assert "type=R1" in ws.build_url("OTT", "2024-05-10", "2024-05-11")
    # Quality levels are requested by name.
    assert "type=definitive" in ws.build_url("BOU", "2024-05-10", "2024-05-11",
                                             level="definitive")
    assert "type=quasi-definitive" in ws.build_url("BOU", "2024-05-10",
                                                   "2024-05-11", level="qd")


def test_parse_payload_nulls_and_metadata():
    payload = {
        "times": ["2024-05-10T00:00:00Z", "2024-05-10T00:01:00Z"],
        "metadata": {"intermagnet": {"imo": {"name": "Boulder",
                                             "coordinates": [254.764, 40.137, 1682]}}},
        "values": [{"id": "X", "values": [20800.1, None]}],
    }
    times, values, metadata = ws.parse_payload(payload)
    assert len(times) == 2 and np.isnan(values["X"][1])
    assert metadata["station_name"] == "Boulder"
    assert metadata["latitude"] == pytest.approx(40.137)


# ---------------------------------------------------------------------------
# Magnetic download / store / load / provenance
# ---------------------------------------------------------------------------
def test_download_stores_under_source_and_level(cache, fake_ws):
    data.download_observatory("BOU", "2024-05-10", "2024-05-11", level="variation")
    path = cache / "magnetic" / "usgs" / "variation" / "BOU" / "BOU_2024.nc"
    assert path.exists()
    assert data.stored_partitions("BOU") == [("usgs", "variation")]


def test_load_roundtrip(cache, fake_ws):
    data.download_observatory("BOU", "2024-05-10", "2024-05-12", level="variation")
    ds = data.load_observatory("BOU", level="variation")
    assert ds.sizes["time"] == 2 * 24 * 60
    assert list(ds.dims) == ["time"]
    assert ds.attrs["latitude"] == pytest.approx(40.2, abs=0.5)


def test_nrcan_auto_source(cache, fake_ws):
    data.download_observatory("OTT", "2024-05-10", "2024-05-11")
    assert data.stored_partitions("OTT") == [("nrcan", "variation")]


def test_year_boundary_splits(cache, fake_ws):
    data.download_observatory("NEW", "2023-12-31", "2024-01-02", elements=("X", "Y"))
    assert data.observatory_years("NEW") == [2023, 2024]


def test_update_is_incremental(cache, fake_ws):
    data.download_observatory("BOU", "2024-05-10", "2024-05-12", level="variation")
    data.update_observatory("BOU", level="variation", until="2024-05-13")
    ds = data.load_observatory("BOU", level="variation")
    assert ds.sizes["time"] == 3 * 24 * 60
    assert len(np.unique(ds["time"].values)) == ds.sizes["time"]


def test_load_missing_raises(cache):
    with pytest.raises(FileNotFoundError):
        data.load_observatory("XYZ")


# ---------------------------------------------------------------------------
# Cross-level merge (the headline provenance feature)
# ---------------------------------------------------------------------------
def test_best_merge_prefers_higher_level_and_backfills(cache, fake_ws):
    # Same day at two levels: variation (11, complete) and definitive (99, with
    # its last 5 samples culled to NaN).
    data.download_observatory("BOU", "2024-05-10", "2024-05-11", level="variation")
    data.download_observatory("BOU", "2024-05-10", "2024-05-11", level="definitive")

    # Highest quality is listed first.
    assert data.stored_partitions("BOU")[0] == ("usgs", "definitive")

    variation = data.load_observatory("BOU", level="variation", as_dataframe=True)
    definitive = data.load_observatory("BOU", level="definitive", as_dataframe=True)
    assert variation["X"].iloc[0] == 11.0
    assert definitive["X"].iloc[0] == 99.0
    assert definitive["X"].isna().sum() == 5

    best = data.load_observatory("BOU", as_dataframe=True)  # level="best" default
    # Definitive wins where present...
    assert best["X"].iloc[0] == 99.0
    # ...but the culled tail is recovered from variation, leaving no gaps.
    assert best["X"].iloc[-1] == 11.0
    assert best["X"].isna().sum() == 0
    assert "definitive" in best.attrs["provenance"]


def test_load_specific_level_only(cache, fake_ws):
    data.download_observatory("BOU", "2024-05-10", "2024-05-11", level="variation")
    data.download_observatory("BOU", "2024-05-10", "2024-05-11", level="definitive")
    only_def = data.load_observatory("BOU", level="definitive", as_dataframe=True)
    assert (only_def["X"].dropna() == 99.0).all()


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------
def test_inventory_reports_source_and_level(cache, fake_ws):
    data.download_observatory("BOU", "2024-05-10", "2024-05-11", level="variation")
    data.download_observatory("BOU", "2024-05-10", "2024-05-11", level="definitive")
    inv = data.observatory_inventory(detailed=True)
    assert set(inv["level"]) == {"variation", "definitive"}
    assert set(inv["source"]) == {"usgs"}
    assert (inv["n_samples"] == 24 * 60).all()


def test_inventory_empty_has_columns(cache):
    inv = data.observatory_inventory()
    assert inv.empty and "level" in inv.columns and "source" in inv.columns


# ---------------------------------------------------------------------------
# INTERMAGNET GIN client
# ---------------------------------------------------------------------------
def test_intermagnet_url_and_publication_state():
    url = im.build_url("ESK", "2024-05-10", 2, level="definitive")
    assert "observatoryIagaCode=ESK" in url
    assert "publicationState=definitive" in url
    assert "dataDuration=2" in url and "samplesPerDay=Minute" in url
    assert "format=iaga2002" in url
    assert "publicationState=best-avail" in im.build_url("ESK", "2024-05-10",
                                                        1, level="best")
    # The GIN vocabulary (verified against the live service).
    assert im.publication_state("quasi-definitive") == "quasi-def"
    assert im.publication_state("variation") == "reported"
    assert im.publication_state("definitive") == "definitive"


def test_intermagnet_second_data_uses_capitalized_keyword():
    url = im.build_url("ESK", "2024-05-10", 1, sampling_period=1, level="definitive")
    assert "samplesPerDay=Second" in url


IAGA_ESK = """ Format                 IAGA-2002                                    |
 Source of Data         INTERMAGNET                                  |
 Station Name           Eskdalemuir                                  |
 IAGA CODE              ESK                                          |
 Geodetic Latitude      55.314                                       |
 Geodetic Longitude     356.794                                      |
 Elevation              245                                          |
 Reported               XYZF                                         |
 Sensor Orientation     HDZF                                         |
 Digital Sampling       0.01 second                                  |
 Data Interval Type     filtered 1-minute                            |
 Data Type              definitive                                   |
DATE       TIME         DOY     ESKX      ESKY      ESKZ      ESKF   |
2024-05-10 00:00:00.000 131     17000.00   -1500.00  45000.00  48000.00
2024-05-10 00:01:00.000 131     17001.00   -1499.00  45001.00  48001.00
"""


def test_intermagnet_parse_iaga_text():
    df = im.parse_iaga_text(IAGA_ESK)
    assert list(df.columns) == ["X", "Y", "Z", "F"]
    assert len(df) == 2


def test_intermagnet_error_page_raises():
    with pytest.raises(ValueError):
        im.parse_iaga_text("<html>Error: no data available</html>")


def test_intermagnet_end_to_end(cache, monkeypatch):
    class FakeSession:
        def get(self, url, timeout=None):
            return _FakeResponse(text=IAGA_ESK)

    original = im.fetch_dataframe

    def patched(code, start, end, **kwargs):
        kwargs["session"] = FakeSession()
        return original(code, start, end, **kwargs)

    monkeypatch.setattr(im, "fetch_dataframe", patched)

    data.download_observatory("ESK", "2024-05-10", "2024-05-11",
                              source="intermagnet", level="definitive")
    assert ("intermagnet", "definitive") in data.stored_partitions("ESK")
    ds = data.load_observatory("ESK")
    assert ds.sizes["time"] == 2


# ---------------------------------------------------------------------------
# IAGA ingest
# ---------------------------------------------------------------------------
def test_ingest_iaga(cache, tmp_path):
    iaga_path = tmp_path / "esk.min"
    iaga_path.write_text(IAGA_ESK)
    written = data.ingest_iaga(str(iaga_path), level="definitive")
    assert [p.name for p in written] == ["ESK_2024.nc"]
    assert ("iaga", "definitive") in data.stored_partitions("ESK")


# ---------------------------------------------------------------------------
# MT (EMTF XML)
# ---------------------------------------------------------------------------
def test_mt_survey_detection(cache, sample_emtf_path):
    xml = sample_emtf_path.read_text()
    assert data.mt.mt_survey_of(xml) == "TEST"


def test_mt_ingest_and_load(cache, sample_emtf_path):
    written = data.ingest_mt_xml(str(sample_emtf_path))
    assert written[0].parent.name == "TEST"
    assert written[0].name == "TEST01.xml"

    site = data.load_mt_site("TEST01", survey="TEST")
    assert site.name == "TEST01"
    assert site.latitude == pytest.approx(40.0)
    Z = site.calcZ(np.array([1e-2, 1e-3]))
    assert Z.shape == (4, 2)


def test_mt_download_with_survey_override(cache, sample_emtf_path):
    xml = sample_emtf_path.read_text()

    class FakeSession:
        def get(self, url, timeout=None):
            return _FakeResponse(text=xml)

    path = data.download_mt("http://example/spud/data/1", session=FakeSession(),
                            survey="MyCampaign")
    assert path.parent.name == "MyCampaign"
    assert "MyCampaign" in data.stored_mt_surveys()


def test_mt_inventory_and_filters(cache, sample_emtf_path):
    data.ingest_mt_xml(str(sample_emtf_path))
    inv = data.mt_inventory()
    assert inv.iloc[0]["code"] == "TEST01"
    assert inv.iloc[0]["network"] == "EM"
    assert inv.iloc[0]["survey"] == "TEST"

    inside = data.load_mt_sites(bbox=(-110, 35, -100, 45), min_rating=5)
    assert [s.name for s in inside] == ["TEST01"]
    outside = data.load_mt_sites(bbox=(0, 0, 10, 10))
    assert outside == []


def test_mt_duplicate_code_warns(cache, sample_emtf_path):
    data.ingest_mt_xml(str(sample_emtf_path), survey="A")
    data.ingest_mt_xml(str(sample_emtf_path), survey="B")
    with pytest.warns(UserWarning, match="multiple surveys"):
        data.load_mt_site("TEST01")


# ---------------------------------------------------------------------------
# USGS request chunking (JSON sample limit)
# ---------------------------------------------------------------------------
def test_max_chunk_scales_with_elements_and_cadence():
    minute = ws._max_chunk(("X", "Y", "Z", "F"), 60)
    second = ws._max_chunk(("X", "Y", "Z", "F"), 1)
    # Minute data -> tens of days; one-second data -> only hours.
    assert 20 < minute.total_seconds() / 86400 < 26
    assert second.total_seconds() / 3600 < 10


def test_invalid_sampling_period_raises(fake_ws):
    with pytest.raises(ValueError, match="sampling_period"):
        ws.fetch_dataframe("BOU", "2024-05-10", "2024-05-11", sampling_period=30)


# ---------------------------------------------------------------------------
# prefer= override on merge
# ---------------------------------------------------------------------------
def test_prefer_lets_variation_override_definitive(cache, fake_ws):
    data.download_observatory("BOU", "2024-05-10", "2024-05-11", level="variation")
    data.download_observatory("BOU", "2024-05-10", "2024-05-11", level="definitive")

    default = data.load_observatory("BOU", as_dataframe=True)
    overridden = data.load_observatory("BOU", prefer="variation", as_dataframe=True)
    # Default: definitive wins. prefer="variation": real-time wins instead.
    assert default["X"].iloc[0] == 99.0
    assert overridden["X"].iloc[0] == 11.0
    # And the full (un-culled) variation series is used.
    assert overridden["X"].isna().sum() == 0


# ---------------------------------------------------------------------------
# SPUD bundle resolver
# ---------------------------------------------------------------------------
_SPUD_METADATA = """<html><body><pre>&lt;EM_TF id="999"&gt;
  &lt;SourceData id="555"&gt;
    &lt;Filename&gt;WVO55.xml&lt;/Filename&gt;
    &lt;MimeType&gt;application/xml&lt;/MimeType&gt;
    &lt;DataLink&gt;&lt;a href="https://ds.iris.edu/spudservice/data/555"&gt;link&lt;/a&gt;&lt;/DataLink&gt;
  &lt;/SourceData&gt;
  &lt;Attachment id="556"&gt;
    &lt;MimeType&gt;image/png&lt;/MimeType&gt;
  &lt;/Attachment&gt;
&lt;/EM_TF&gt;</pre></body></html>"""


def test_spud_resolver_picks_xml_not_png(cache, sample_emtf_path, monkeypatch):
    xml = sample_emtf_path.read_text()

    class FakeSession:
        def get(self, url, timeout=None):
            # Bundle metadata request vs. the resolved data request.
            if "/emtf/" in url:
                return _FakeResponse(text=_SPUD_METADATA)
            assert url.endswith("/data/555")  # the XML attachment, not the PNG
            return _FakeResponse(text=xml)

    path = data.download_mt_spud(999, session=FakeSession())
    assert path.name == "TEST01.xml"
    assert data.load_mt_site("TEST01").name == "TEST01"


# ---------------------------------------------------------------------------
# Transmission lines
# ---------------------------------------------------------------------------
def test_tl_build_query_url():
    url = data.tl.build_query_url(data.tl.DEFAULT_SERVICE_URL, where="VOLTAGE>=345",
                                  bbox=(-110, 35, -100, 45))
    assert "/query?" in url and "f=geojson" in url
    assert "esriGeometryEnvelope" in url and "VOLTAGE" in url


def test_tl_parse_geojson_linestring_and_multi():
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"VOLTAGE": 345.0},
             "geometry": {"type": "LineString",
                          "coordinates": [[-105, 40], [-104, 41]]}},
            {"type": "Feature", "properties": {"VOLTAGE": 500.0},
             "geometry": {"type": "MultiLineString",
                          "coordinates": [[[-100, 38], [-99, 38.5]],
                                          [[-98, 39], [-97, 39.2]]]}},
        ],
    }
    lines = data.tl.parse_geojson(geojson)
    # One line for the LineString, two for the MultiLineString parts.
    assert len(lines) == 3
    assert lines[0].metadata["VOLTAGE"] == 345.0
    assert lines[0].npts == 2 and lines[0].length > 0


def test_tl_download_and_reload(cache):
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"VOLTAGE": 345.0, "ID": "1"},
             "geometry": {"type": "LineString",
                          "coordinates": [[-105, 40], [-104.5, 40.5]]}},
        ],
    }

    class FakeSession:
        def get(self, url, timeout=None):
            return _FakeResponse(payload={**geojson, "exceededTransferLimit": False})

    lines = data.download_transmission_lines(bbox=(-110, 35, -100, 45),
                                             min_voltage=345, session=FakeSession())
    assert len(lines) == 1
    reloaded = data.load_transmission_lines()
    assert len(reloaded) == 1 and reloaded[0].metadata["ID"] == "1"


def test_transmission_line_accepts_linestring():
    # bezpy.tl.TransmissionLine should accept a bare coordinate array now.
    from bezpy.tl import TransmissionLine
    line = TransmissionLine([[-105, 40], [-104, 41]], metadata={"VOLTAGE": 500})
    assert line.npts == 2 and line.length > 0 and line.metadata["VOLTAGE"] == 500


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------
def test_cache_override_and_clear(cache, fake_ws):
    data.download_observatory("BOU", "2024-05-10", "2024-05-11")
    assert str(data.get_cache_dir()).startswith(str(cache))
    removed = data.clear_cache("magnetic")
    assert not removed.exists()


def test_download_all_nan_response_not_stored(cache, monkeypatch):
    """A service returning a time axis of nulls must not create a partition.

    The USGS/GIN services return full-length all-null responses for quality
    levels they do not hold (e.g. definitive before publication); storing
    them created misleading all-NaN archive files.
    """
    def all_nan_payload(url):
        payload = _synthetic_payload(url)
        for entry in payload["values"]:
            entry["values"] = [None] * len(entry["values"])
        return payload

    class NaNSession:
        def get(self, url, timeout=None):
            return _FakeResponse(payload=all_nan_payload(url))

    original = ws.fetch_interval

    def patched(code, start, end, **kwargs):
        kwargs["session"] = NaNSession()
        return original(code, start, end, **kwargs)

    monkeypatch.setattr(ws, "fetch_interval", patched)
    with pytest.warns(UserWarning, match="nothing stored"):
        paths = data.download_observatory("BOU", "2024-05-10", "2024-05-11",
                                          level="definitive")
    assert paths == []
    inventory = data.observatory_inventory()
    assert inventory.empty
