"""Storage abstraction for local and remote (S3/GCS/Azure) file operations.

All public functions transparently handle both local paths and remote URIs
(``s3://``, ``gs://``, ``az://``) via *fsspec* when it is installed.
For local-only usage, no extra dependencies are required.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pyarrow.parquet as pq

if TYPE_CHECKING:
    import pyarrow as pa


def _is_remote(path: str) -> bool:
    """Return True if *path* looks like a remote URI."""
    return path.startswith(("s3://", "gs://", "az://", "abfs://"))


def _get_fs(path: str):
    """Return an fsspec filesystem for *path*, or None for local paths."""
    if not _is_remote(path):
        return None
    try:
        import fsspec
    except ImportError as exc:
        msg = (
            f"fsspec is required to access remote paths ({path!r}). "
            "Install it with: pip install 'ducklake-polars[cloud]'"
        )
        raise ImportError(msg) from exc
    return fsspec.filesystem(path.split("://")[0])


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
