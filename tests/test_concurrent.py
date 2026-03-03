"""Concurrent write tests for ducklake-dataframe.

Tests optimistic concurrency control: conflict detection, retry logic,
and safe concurrent operations.
"""

from __future__ import annotations

import threading

import polars as pl
import pytest

from ducklake_polars import read_ducklake, write_ducklake
from ducklake_core._writer import TransactionConflictError


class TestConcurrentInserts:
    """Concurrent inserts to the same table should succeed (append-only).

    SQLite-only: Postgres uses READ COMMITTED isolation and doesn't support
    the IMMEDIATE locking that prevents concurrent ID races. DuckDB's native
    extension handles Postgres concurrency at the C++ level.
    """

    def test_concurrent_inserts_same_table(self, ducklake_catalog_sqlite):
        """Two threads inserting into the same table simultaneously."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.close()

        errors = []

        def insert_batch(start: int):
            try:
                df = pl.DataFrame({
                    "a": pl.Series(list(range(start, start + 50)), dtype=pl.Int32),
                    "b": [f"val_{i}" for i in range(start, start + 50)],
                })
                write_ducklake(
                    df, cat.metadata_path, "test",
                    data_path=cat.data_path, mode="append",
                )
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=insert_batch, args=(0,))
        t2 = threading.Thread(target=insert_batch, args=(50,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0, f"Unexpected errors: {errors}"
        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 100

    def test_concurrent_inserts_different_tables(self, ducklake_catalog_sqlite):
        """Two threads inserting into different tables."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("CREATE TABLE ducklake.t2 (a INTEGER)")
        cat.close()

        errors = []

        def insert_t1():
            try:
                write_ducklake(
                    pl.DataFrame({"a": pl.Series([1, 2, 3], dtype=pl.Int32)}),
                    cat.metadata_path, "t1",
                    data_path=cat.data_path, mode="append",
                )
            except Exception as e:
                errors.append(e)

        def insert_t2():
            try:
                write_ducklake(
                    pl.DataFrame({"a": pl.Series([4, 5, 6], dtype=pl.Int32)}),
                    cat.metadata_path, "t2",
                    data_path=cat.data_path, mode="append",
                )
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=insert_t1)
        t2 = threading.Thread(target=insert_t2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0
        r1 = read_ducklake(cat.metadata_path, "t1", data_path=cat.data_path)
        r2 = read_ducklake(cat.metadata_path, "t2", data_path=cat.data_path)
        assert r1.shape[0] == 3
        assert r2.shape[0] == 3

    def test_sequential_after_concurrent(self, ducklake_catalog_sqlite):
        """After concurrent inserts, sequential reads/writes work correctly."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        def insert(val):
            write_ducklake(
                pl.DataFrame({"a": pl.Series([val], dtype=pl.Int32)}),
                cat.metadata_path, "test",
                data_path=cat.data_path, mode="append",
            )

        threads = [threading.Thread(target=insert, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        write_ducklake(
            pl.DataFrame({"a": pl.Series([100], dtype=pl.Int32)}),
            cat.metadata_path, "test",
            data_path=cat.data_path, mode="append",
        )
        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 6


class TestRetryLogic:
    """Test that retry parameters are accepted."""

    def test_writer_accepts_retry_params(self, ducklake_catalog_sqlite):
        """Writer constructor accepts retry parameters."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        write_ducklake(
            pl.DataFrame({"a": pl.Series([1], dtype=pl.Int32)}),
            cat.metadata_path, "test",
            data_path=cat.data_path, mode="append",
            max_retries=5, retry_wait_ms=50, retry_backoff=1.5,
        )
        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 1

    def test_transaction_conflict_error_class(self):
        """TransactionConflictError is importable and well-formed."""
        assert issubclass(TransactionConflictError, Exception)
        err = TransactionConflictError("test conflict")
        assert str(err) == "test conflict"
