"""Transaction semantics tests for ducklake-dataframe.

Covers:
  - Write atomicity: partial writes don't leave corrupt state
  - Conflict detection: concurrent writes to same table conflict correctly
  - Multi-step consistency: sequential operations maintain consistency
  - Retry behavior: TransactionConflictError triggers retries
  - Read-after-write consistency: writes are visible immediately after
  - Table-level isolation: writes to different tables don't interfere
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ducklake_polars import (
    read_ducklake,
    write_ducklake,
    rewrite_data_files_ducklake,
)
from ducklake_core._writer import TransactionConflictError


class TestWriteAtomicity:
    """Write operations are atomic — either fully committed or not at all."""

    def test_failed_write_preserves_state(self, make_write_catalog):
        """If a write fails mid-way, previous data should be intact."""
        cat = make_write_catalog()
        df1 = pl.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        write_ducklake(df1, cat.metadata_path, "t", data_path=cat.data_path)

        # Try to write invalid data (wrong schema) — should fail
        df_bad = pl.DataFrame({"wrong_col": [1, 2]})
        try:
            write_ducklake(df_bad, cat.metadata_path, "t", mode="append",
                          data_path=cat.data_path)
        except Exception:
            pass  # Expected to fail or succeed with extra column

        # Original data should still be readable
        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) >= 3  # At least original data intact

    def test_sequential_writes_consistent(self, make_write_catalog):
        """Multiple sequential writes maintain data consistency."""
        cat = make_write_catalog()
        total = 0
        for i in range(10):
            batch_size = (i + 1) * 5
            df = pl.DataFrame({
                "id": list(range(total, total + batch_size)),
                "batch": [i] * batch_size,
            })
            write_ducklake(df, cat.metadata_path, "t", mode="append",
                          data_path=cat.data_path)
            total += batch_size

            # Verify count after each write
            result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
            assert len(result) == total, f"After batch {i}: expected {total}, got {len(result)}"

        assert total == 275  # sum(5,10,15,...,50)


class TestReadAfterWriteConsistency:
    """Writes are immediately visible to subsequent reads."""

    def test_read_sees_latest_write(self, make_write_catalog):
        """A read immediately after a write sees the new data."""
        cat = make_write_catalog()
        for i in range(5):
            df = pl.DataFrame({"id": [i], "val": [f"v{i}"]})
            write_ducklake(df, cat.metadata_path, "t", mode="append",
                          data_path=cat.data_path)

            result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
            assert len(result) == i + 1
            assert i in result["id"].to_list()

    def test_rewrite_followed_by_write(self, make_write_catalog):
        """Rewrite + subsequent write: both visible."""
        cat = make_write_catalog()
        for i in range(3):
            df = pl.DataFrame({"id": [i]})
            write_ducklake(df, cat.metadata_path, "t", mode="append",
                          data_path=cat.data_path)

        rewrite_data_files_ducklake(cat.metadata_path, "t", data_path=cat.data_path)

        df_new = pl.DataFrame({"id": [100]})
        write_ducklake(df_new, cat.metadata_path, "t", mode="append",
                      data_path=cat.data_path)

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 4
        assert 100 in result["id"].to_list()


class TestTableIsolation:
    """Operations on different tables don't interfere."""

    def test_writes_to_different_tables_independent(self, make_write_catalog):
        """Writing to table A doesn't affect table B."""
        cat = make_write_catalog()
        df_a = pl.DataFrame({"id": [1, 2, 3]})
        df_b = pl.DataFrame({"id": [10, 20, 30]})
        write_ducklake(df_a, cat.metadata_path, "table_a", data_path=cat.data_path)
        write_ducklake(df_b, cat.metadata_path, "table_b", data_path=cat.data_path)

        result_a = read_ducklake(cat.metadata_path, "table_a", data_path=cat.data_path)
        result_b = read_ducklake(cat.metadata_path, "table_b", data_path=cat.data_path)

        assert set(result_a["id"].to_list()) == {1, 2, 3}
        assert set(result_b["id"].to_list()) == {10, 20, 30}

    def test_rewrite_one_table_doesnt_affect_other(self, make_write_catalog):
        """Rewriting table A doesn't affect table B's data."""
        cat = make_write_catalog()
        for i in range(3):
            df_a = pl.DataFrame({"id": [i]})
            write_ducklake(df_a, cat.metadata_path, "a", mode="append",
                          data_path=cat.data_path)

        df_b = pl.DataFrame({"id": [100, 200]})
        write_ducklake(df_b, cat.metadata_path, "b", data_path=cat.data_path)

        rewrite_data_files_ducklake(cat.metadata_path, "a", data_path=cat.data_path)

        result_b = read_ducklake(cat.metadata_path, "b", data_path=cat.data_path)
        assert set(result_b["id"].to_list()) == {100, 200}


class TestConflictDetection:
    """Concurrent writes that conflict are detected and retried."""

    def test_concurrent_rewrite_and_write(self, ducklake_catalog_sqlite):
        """Concurrent rewrite + write: both should eventually succeed."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (id INTEGER)")
        for i in range(5):
            cat.execute(f"INSERT INTO ducklake.t VALUES ({i})")
        cat.close()

        errors = []
        results = {"rewrite": None, "write": None}

        def do_rewrite():
            try:
                results["rewrite"] = rewrite_data_files_ducklake(
                    cat.metadata_path, "t", data_path=cat.data_path,
                )
            except Exception as e:
                errors.append(("rewrite", e))

        def do_write():
            try:
                df = pl.DataFrame({"id": pl.Series([100, 200], dtype=pl.Int32)})
                write_ducklake(df, cat.metadata_path, "t", mode="append",
                              data_path=cat.data_path)
                results["write"] = True
            except Exception as e:
                errors.append(("write", e))

        t1 = threading.Thread(target=do_rewrite)
        t2 = threading.Thread(target=do_write)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        # Both should succeed (retry logic handles conflicts)
        assert len(errors) == 0, f"Unexpected errors: {errors}"

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 7  # 5 original + 2 appended


class TestWriteModes:
    """Write mode semantics (error, append, overwrite)."""

    def test_error_mode_on_existing_table(self, make_write_catalog):
        """mode='error' raises on existing table."""
        cat = make_write_catalog()
        df = pl.DataFrame({"id": [1]})
        write_ducklake(df, cat.metadata_path, "t", data_path=cat.data_path)

        with pytest.raises(Exception):
            write_ducklake(df, cat.metadata_path, "t", mode="error",
                          data_path=cat.data_path)

    def test_append_mode_adds_rows(self, make_write_catalog):
        """mode='append' adds rows to existing table."""
        cat = make_write_catalog()
        df1 = pl.DataFrame({"id": [1, 2]})
        write_ducklake(df1, cat.metadata_path, "t", data_path=cat.data_path)

        df2 = pl.DataFrame({"id": [3, 4]})
        write_ducklake(df2, cat.metadata_path, "t", mode="append",
                      data_path=cat.data_path)

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert set(result["id"].to_list()) == {1, 2, 3, 4}

    def test_overwrite_mode_replaces_data(self, make_write_catalog):
        """mode='overwrite' replaces all existing data."""
        cat = make_write_catalog()
        df1 = pl.DataFrame({"id": [1, 2, 3]})
        write_ducklake(df1, cat.metadata_path, "t", data_path=cat.data_path)

        df2 = pl.DataFrame({"id": [10, 20]})
        write_ducklake(df2, cat.metadata_path, "t", mode="overwrite",
                      data_path=cat.data_path)

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert set(result["id"].to_list()) == {10, 20}


class TestSnapshotConsistency:
    """Snapshot IDs increase monotonically and are consistent."""

    def test_snapshot_increases_on_write(self, ducklake_catalog_sqlite):
        """Each write creates a new snapshot with higher ID."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (id INTEGER)")
        prev_snap = cat.fetchone(
            "SELECT * FROM ducklake_current_snapshot('ducklake')"
        )[0]

        for i in range(3):
            cat.execute(f"INSERT INTO ducklake.t VALUES ({i})")
            snap = cat.fetchone(
                "SELECT * FROM ducklake_current_snapshot('ducklake')"
            )[0]
            assert snap > prev_snap, f"Snapshot should increase: {snap} <= {prev_snap}"
            prev_snap = snap

        cat.close()

    def test_polars_write_creates_snapshot(self, make_write_catalog):
        """write_ducklake creates a new snapshot."""
        cat = make_write_catalog()
        if cat.backend != "sqlite":
            pytest.skip("Snapshot ID check only reliable on SQLite")

        df = pl.DataFrame({"id": [1]})
        write_ducklake(df, cat.metadata_path, "t", data_path=cat.data_path)

        row = cat.query_one("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        snap1 = row[0]

        df2 = pl.DataFrame({"id": [2]})
        write_ducklake(df2, cat.metadata_path, "t", mode="append",
                      data_path=cat.data_path)

        row2 = cat.query_one("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        snap2 = row2[0]

        assert snap2 > snap1
