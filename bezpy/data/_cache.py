"""Cache directory management for downloaded data.

bezpy caches everything it downloads (magnetic-field time series, MT transfer
functions, region polygons, ...) in a single on-disk directory so that repeated
analyses do not re-download the same data. The location is, in priority order:

1. The path passed to :func:`set_cache_dir` during the session.
2. The ``BEZPY_DATA_DIR`` environment variable.
3. A platform-appropriate default from :func:`pooch.os_cache` (e.g.
   ``~/.cache/bezpy`` on Linux, ``~/Library/Caches/bezpy`` on macOS).

The directory layout underneath the cache root is::

    <cache>/magnetic/<source>/<level>/<CODE>/<CODE>_<YEAR>.nc
    <cache>/mt/<survey>/<CODE>.xml
    <cache>/tl/<name>.geojson

"""

__all__ = ["get_cache_dir", "set_cache_dir", "clear_cache"]

import os
import pathlib
import shutil
import tempfile

import pooch

# Session-level override set by set_cache_dir(). None means "fall back to the
# environment variable or the pooch default".
_CACHE_OVERRIDE = None


def _atomic_write_text(path, text, encoding="utf-8"):
    """Replace a cache text file atomically after a successful full write."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding=encoding,
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    tmp = pathlib.Path(handle.name)
    try:
        with handle:
            handle.write(text)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_write_bytes(path, content):
    """Replace a cache file atomically after a successful binary download."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    tmp = pathlib.Path(handle.name)
    try:
        with handle:
            handle.write(content)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def get_cache_dir(subdir=None, create=True):
    """Return the bezpy data cache directory.

    Parameters
    ----------
    subdir : str, optional
        A sub-directory beneath the cache root to return (e.g. ``"magnetic"``).
        Nested paths such as ``"magnetic/BOU"`` are also accepted.
    create : bool, optional
        Whether to create the directory (and any parents) if it does not yet
        exist. Default ``True``.

    Returns
    -------
    pathlib.Path
        The absolute path to the requested cache directory.
    """
    if _CACHE_OVERRIDE is not None:
        root = pathlib.Path(_CACHE_OVERRIDE)
    elif os.environ.get("BEZPY_DATA_DIR"):
        root = pathlib.Path(os.environ["BEZPY_DATA_DIR"])
    else:
        # pooch.os_cache returns the conventional per-user cache location.
        root = pathlib.Path(pooch.os_cache("bezpy"))

    root = root.expanduser().resolve()
    path = root if subdir is None else (root / subdir).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"cache subdir must stay within {root}; got {subdir!r}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def set_cache_dir(path):
    """Override the cache directory for the rest of the session.

    Parameters
    ----------
    path : str, pathlib.Path, or None
        The new cache root. Pass ``None`` to clear a previous override and fall
        back to ``BEZPY_DATA_DIR`` / the pooch default.
    """
    global _CACHE_OVERRIDE
    _CACHE_OVERRIDE = None if path is None else pathlib.Path(path).expanduser()


def clear_cache(subdir=None):
    """Delete cached data.

    Parameters
    ----------
    subdir : str, optional
        Only clear this sub-directory (e.g. ``"magnetic"`` or
        ``"magnetic/BOU"``). If omitted, the entire bezpy cache is removed.

    Returns
    -------
    pathlib.Path
        The directory that was removed.
    """
    # Resolve without creating so we don't recreate a directory we're deleting.
    target = get_cache_dir(subdir, create=False)
    if target.exists():
        shutil.rmtree(target)
    return target
