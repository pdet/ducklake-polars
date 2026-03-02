"""Storage abstraction for local and remote (S3/GCS/Azure) file operations.

All public functions transparently handle both local paths and remote URIs
(``s3://``, ``gs://``, ``az://``, ``abfs://``, ``abfss://``) via *fsspec*
when it is installed.  For local-only usage, no extra dependencies are required.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pyarrow.parquet as pq

if TYPE_CHECKING:
    import pyarrow as pa

# Module-level cache for fsspec filesystem instances, keyed by protocol.
_fs_cache: dict[str, object] = {}


def _is_remote(path: str) -> bool:
    """Return True if *path* looks like a remote URI."""
    return path.startswith(("s3://", "gs://", "az://", "abfs://", "abfss://"))


def _get_fs(path: str):
    """Return an fsspec filesystem for *path*, or None for local paths."""
    if not _is_remote(path):
        return None
    try:
        import fsspec
    except ImportError as exc:
        msg = (
            f"fsspec is required to access remote paths ({path!r}). "
            "Install it with: pip install 'ducklake-dataframe[cloud]'"
        )
        raise ImportError(msg) from exc
    protocol = path.split("://")[0]
    if protocol not in _fs_cache:
        _fs_cache[protocol] = fsspec.filesystem(protocol)
    return _fs_cache[protocol]


# ------------------------------------------------------------------
# Public helpers
# ------------------------------------------------------------------


def makedirs(path: str, *, exist_ok: bool = True) -> None:
    """Create directories, including parents.  No-op for remote paths."""
    if _is_remote(path):
        # Object stores don't need explicit directory creation.
        return
    os.makedirs(path, exist_ok=exist_ok)


def write_parquet(table: pa.Table, path: str) -> None:
    """Write a PyArrow table to a Parquet file (local or remote)."""
    fs = _get_fs(path)
    if fs is None:
        pq.write_table(table, path)
    else:
        with fs.open(path, "wb") as f:
            pq.write_table(table, f)


def read_parquet(path: str) -> pa.Table:
    """Read a Parquet file into a PyArrow table (local or remote)."""
    fs = _get_fs(path)
    if fs is None:
        return pq.ParquetFile(path).read()
    with fs.open(path, "rb") as f:
        return pq.ParquetFile(f).read()


def get_file_size(path: str) -> int:
    """Return the file size in bytes."""
    fs = _get_fs(path)
    if fs is None:
        return os.path.getsize(path)
    return fs.info(path)["size"]


def read_parquet_footer_size(path: str) -> int:
    """Read the 4-byte Parquet footer length from the last 8 bytes of a file."""
    fs = _get_fs(path)
    if fs is None:
        with open(path, "rb") as f:
            f.seek(-8, 2)
            return int.from_bytes(f.read(4), "little")
    size = fs.info(path)["size"]
    with fs.open(path, "rb") as f:
        f.seek(size - 8)
        return int.from_bytes(f.read(4), "little")


def join_path(base: str, *parts: str) -> str:
    """Join path components.  Uses ``/`` for remote URIs, ``os.path.join`` locally."""
    if _is_remote(base):
        # Strip trailing slashes from base, leading slashes from parts
        result = base.rstrip("/")
        for p in parts:
            p = p.strip("/")
            if p:
                result = result + "/" + p
        return result
    return os.path.join(base, *parts)


def file_exists(path: str) -> bool:
    """Check whether a file exists."""
    fs = _get_fs(path)
    if fs is None:
        return os.path.exists(path)
    return fs.exists(path)


def normalize_path(path: str) -> str:
    """Normalize a path for consistent comparison.

    For local paths uses ``os.path.normpath``.  For remote URIs collapses
    duplicate slashes (preserving the ``protocol://`` prefix) and strips
    trailing slashes.
    """
    if _is_remote(path):
        protocol, rest = path.split("://", 1)
        parts = [p for p in rest.split("/") if p]
        return protocol + "://" + "/".join(parts)
    return os.path.normpath(path)


def list_directory(path: str, *, suffix: str | None = None) -> list[str]:
    """Recursively list all files under *path*.

    If *suffix* is given, only files ending with that suffix are returned.
    Returns fully-qualified paths (absolute for local, full URI for remote).
    """
    fs = _get_fs(path)
    if fs is None:
        results: list[str] = []
        for dirpath, _dirnames, filenames in os.walk(path):
            for fname in filenames:
                if suffix is None or fname.endswith(suffix):
                    results.append(os.path.normpath(os.path.join(dirpath, fname)))
        return results
    # Remote: use fsspec's find() for recursive listing
    try:
        all_files = fs.find(path)
    except FileNotFoundError:
        return []
    protocol = path.split("://")[0]
    results = []
    for f in all_files:
        full = f if f.startswith(protocol + "://") else protocol + "://" + f
        if suffix is None or full.endswith(suffix):
            results.append(full)
    return results


def delete_file(path: str) -> None:
    """Delete a single file.

    Raises ``FileNotFoundError`` if the file does not exist (local) or
    the remote object is missing.
    """
    fs = _get_fs(path)
    if fs is None:
        os.remove(path)
    else:
        fs.rm(path)
