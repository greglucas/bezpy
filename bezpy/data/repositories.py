"""Download versioned research files from scientific data repositories.

The helpers in this module are intentionally small and provenance-oriented:
Zenodo's record metadata supplies the canonical filename, persistent record
identifier, and checksum; bezpy verifies that checksum before placing a file
in its data cache.
"""

__all__ = ["download_zenodo_file", "zenodo_record"]

import hashlib
import pathlib

from ._cache import _atomic_write_bytes, get_cache_dir
from ._http import http_get

_ZENODO_RECORD_URL = "https://zenodo.org/api/records/{record_id}"


def _record_id(value):
    """Return a validated positive Zenodo record identifier."""
    try:
        record_id = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Zenodo record_id must be a positive integer; got {value!r}") from exc
    if record_id <= 0 or str(record_id) != str(value).strip():
        raise ValueError(f"Zenodo record_id must be a positive integer; got {value!r}")
    return record_id


def zenodo_record(record_id, session=None, timeout=60):
    """Return the public metadata for a Zenodo record.

    Parameters
    ----------
    record_id : int or str
        Numeric record identifier, such as ``15107366`` from
        ``https://doi.org/10.5281/zenodo.15107366``.
    session : requests.Session, optional
        Reusable HTTP session (also useful for testing).
    timeout : float, optional
        Request timeout in seconds.
    """
    record_id = _record_id(record_id)
    return http_get(
        _ZENODO_RECORD_URL.format(record_id=record_id),
        timeout=timeout,
        session=session,
    ).json()


def _file_metadata(record, filename):
    """Find one exact, safe filename in a Zenodo API response."""
    if not isinstance(filename, str) or not filename:
        raise ValueError("filename must be a non-empty string")
    path = pathlib.PurePath(filename)
    if path.name != filename or filename in {".", ".."}:
        raise ValueError(f"filename must not contain path components; got {filename!r}")

    matches = [item for item in record.get("files", []) if item.get("key") == filename]
    if not matches:
        available = ", ".join(sorted(item.get("key", "") for item in record.get("files", [])))
        raise FileNotFoundError(
            f"Zenodo record does not contain {filename!r}. Available files: {available or '(none)'}"
        )
    return matches[0]


def _matches_checksum(content, checksum):
    """Whether bytes match a Zenodo ``algorithm:hex-digest`` checksum."""
    try:
        algorithm, expected = checksum.split(":", 1)
        digest = hashlib.new(algorithm)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"Unsupported repository checksum {checksum!r}") from exc
    digest.update(content)
    return digest.hexdigest().lower() == expected.lower()


def download_zenodo_file(
    record_id,
    filename,
    *,
    refresh=False,
    session=None,
    timeout=120,
):
    """Download and checksum one named file from a Zenodo record.

    Files are cached under
    ``<BEZPY_DATA_DIR>/repositories/zenodo/<record_id>/``. The checksum in
    Zenodo's live record metadata is verified for both cached and newly
    downloaded content. A changed or corrupt cached file is fetched again;
    corrupt downloaded content is never installed in the cache.

    Parameters
    ----------
    record_id : int or str
        Numeric Zenodo record identifier.
    filename : str
        Exact filename shown by the record.
    refresh : bool, optional
        Download again even if a checksum-valid cached copy exists.
    session : requests.Session, optional
        Reusable HTTP session.
    timeout : float, optional
        Timeout for each metadata/file request in seconds.

    Returns
    -------
    pathlib.Path
        Checksum-verified cached file.
    """
    record_id = _record_id(record_id)
    record = zenodo_record(record_id, session=session, timeout=timeout)
    metadata = _file_metadata(record, filename)
    checksum = metadata.get("checksum")
    if not checksum:
        raise ValueError(f"Zenodo did not provide a checksum for {filename!r}")

    destination = get_cache_dir(f"repositories/zenodo/{record_id}") / filename
    if destination.exists() and not refresh:
        if _matches_checksum(destination.read_bytes(), checksum):
            return destination

    url = metadata.get("links", {}).get("self")
    if not url:
        raise ValueError(f"Zenodo did not provide a content URL for {filename!r}")
    content = http_get(url, timeout=timeout, session=session).content
    if not _matches_checksum(content, checksum):
        raise OSError(
            f"Checksum mismatch for Zenodo record {record_id}, file {filename!r}; "
            "the cache was not changed"
        )
    _atomic_write_bytes(destination, content)
    return destination
