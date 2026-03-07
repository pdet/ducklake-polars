"""Optimistic Concurrency Control (OCC) tests for ducklake-dataframe.

Tests snapshot ID collision recovery, change-set tracking, conflict
detection for concurrent schema changes, retry exhaustion, and
multi-writer scenarios.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from unittest.mock import MagicMock, patch

import polars as pl
import pyarrow as pa
import pytest

from ducklake_polars import (
    alter_ducklake_add_column,
    alter_ducklake_drop_column,
    delete_ducklake,
    read_ducklake,
    update_ducklake,
    write_ducklake,
)
from ducklake_core._writer import (
    DuckLakeCatalogWriter,
    TransactionConflictError,
)


# ── helpers ──────────────────────────────────────────────────────────────


def _snapshot_count(metadata_path: str) -> int:
    con = sqlite3.connect(metadata_path)
    try:
        return con.execute("SELECT COUNT(*) FROM ducklake_snapshot").fetchone()[0]
    finally:
        con.close()


def _latest_snapshot_id(metadata_path: str) -> int:
    con = sqlite3.connect(metadata_path)
    try:
        row = con.execute(
            "SELECT COALESCE(MAX(snapshot_id), -1) FROM ducklake_snapshot"
        ).fetchone()
        return row[0]
    finally:
        con.close()


def _snapshot_changes(metadata_path: str) -> list[str]:
    """Return all changes_made strings from ducklake_snapshot_changes."""
    con = sqlite3.connect(metadata_path)
    try:
        rows = con.execute(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


# ── Test: snapshot ID collision recovery ────────────────────────────────


class TestSnapshotCollisionRecovery:
    """_create_snapshot retries on UNIQUE constraint violations."""

    def test_snapshot_id_retry_on_collision(self, ducklake_catalog_sqlite):
        """Simulate a snapshot ID collision and verify recovery."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        writer = DuckLakeCatalogWriter(
            cat.metadata_path,
            data_path_override=cat.data_path,
            max_snapshot_retries=5,
            snapshot_retry_wait_ms=1,
        )
        try:
            writer._connect()
            writer._acquire_write_lock()

            # Get the next snapshot_id that _create_snapshot would try
            row = writer._con.execute(
                "SELECT COALESCE(MAX(snapshot_id), -1) + 1 FROM ducklake_snapshot"
            ).fetchone()
            next_id = row[0]

            # Pre-insert that snapshot_id to simulate a collision
            writer._con.execute(
                "INSERT INTO ducklake_snapshot "
                "(snapshot_id, snapshot_time, schema_version, "
                "next_catalog_id, next_file_id) "
                "VALUES (?, '2025-01-01 00:00:00.000000+00', 1, 1, 1)",
                [next_id],
            )
            writer._con.commit()

            # Re-acquire lock after commit
            writer._acquire_write_lock()

            # _create_snapshot should recover: it tries next_id, gets a
            # collision, re-reads, and succeeds with next_id + 1
            new_id = writer._create_snapshot(2, 2, 2)
            assert new_id == next_id + 1
        finally:
            writer.close()

    def test_snapshot_collision_exhaustion(self, ducklake_catalog_sqlite):
        """When all snapshot retries are exhausted, TransactionConflictError is raised."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        writer = DuckLakeCatalogWriter(
            cat.metadata_path,
            data_path_override=cat.data_path,
            max_snapshot_retries=2,
            snapshot_retry_wait_ms=1,
        )
        try:
            writer._connect()
            writer._acquire_write_lock()

            call_count = 0
            original_execute = writer._con._con.execute

            def mock_execute(sql, params=None):
                nonlocal call_count
                # Normalise placeholder before matching
                sql_norm = sql.replace("%s", "?")
                if "INSERT INTO ducklake_snapshot" in sql_norm:
                    call_count += 1
                    raise sqlite3.IntegrityError(
                        "UNIQUE constraint failed: ducklake_snapshot.snapshot_id"
                    )
                if params is not None:
                    return original_execute(sql, params)
                return original_execute(sql)

            with patch.object(writer._con, "_con", wraps=writer._con._con) as mock_con:
                mock_con.execute = mock_execute
                with pytest.raises(TransactionConflictError, match="Snapshot ID collision"):
                    writer._create_snapshot(1, 1, 1)

            # Should have tried max_snapshot_retries + 1 times
            assert call_count == 3  # initial + 2 retries
        finally:
            writer.close()

    def test_non_integrity_error_propagates(self, ducklake_catalog_sqlite):
        """Non-integrity errors in _create_snapshot propagate immediately."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        writer = DuckLakeCatalogWriter(
            cat.metadata_path,
            data_path_override=cat.data_path,
            max_snapshot_retries=3,
        )
        try:
            writer._connect()
            writer._acquire_write_lock()

            original_execute = writer._con._con.execute

            def mock_execute(sql, params=None):
                sql_norm = sql.replace("%s", "?")
                if "INSERT INTO ducklake_snapshot" in sql_norm:
                    raise RuntimeError("disk full")
                if params is not None:
                    return original_execute(sql, params)
                return original_execute(sql)

            with patch.object(writer._con, "_con", wraps=writer._con._con) as mock_con:
                mock_con.execute = mock_execute
                with pytest.raises(RuntimeError, match="disk full"):
                    writer._create_snapshot(1, 1, 1)
        finally:
            writer.close()

    def test_snapshot_collision_with_concurrent_writer(self, ducklake_catalog_sqlite):
        """Two concurrent writers both successfully create snapshots."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        snap_before = _snapshot_count(cat.metadata_path)

        errors = []

        def do_insert(start: int):
            try:
                df = pl.DataFrame({
                    "a": pl.Series(list(range(start, start + 10)), dtype=pl.Int32),
                })
                write_ducklake(
                    df, cat.metadata_path, "test",
                    data_path=cat.data_path, mode="append",
                    max_retries=5, retry_wait_ms=10,
                )
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=do_insert, args=(0,))
        t2 = threading.Thread(target=do_insert, args=(100,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0, f"Errors: {errors}"

        snap_after = _snapshot_count(cat.metadata_path)
        assert snap_after == snap_before + 2  # Both created a snapshot

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 20


# ── Test: concurrent appends (should succeed) ──────────────────────────


class TestConcurrentAppends:
    """Two concurrent appends to the same table should both succeed."""

    def test_two_appends_same_table(self, ducklake_catalog_sqlite):
        """Both appends complete and all rows are present."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, val VARCHAR)")
        cat.close()

        barrier = threading.Barrier(2, timeout=10)
        errors = []

        def append(prefix: str, start: int):
            try:
                barrier.wait()
                df = pl.DataFrame({
                    "id": pl.Series(list(range(start, start + 20)), dtype=pl.Int32),
                    "val": [f"{prefix}_{i}" for i in range(start, start + 20)],
                })
                write_ducklake(
                    df, cat.metadata_path, "test",
                    data_path=cat.data_path, mode="append",
                    max_retries=5, retry_wait_ms=10,
                )
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=append, args=("A", 0))
        t2 = threading.Thread(target=append, args=("B", 100))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0, f"Errors: {errors}"
        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 40

    def test_conflict_check_insert_vs_insert_no_conflict(self):
        """_check_conflicts: concurrent inserts to the same table don't conflict."""
        writer = DuckLakeCatalogWriter.__new__(DuckLakeCatalogWriter)
        writer._backend = None
        writer._con = None

        with patch.object(
            writer, "_get_concurrent_changes",
            return_value=[(2, "inserted_into_table:1")],
        ):
            # Should NOT raise
            writer._check_conflicts(0, {1: "insert"})

    def test_three_concurrent_appends(self, ducklake_catalog_sqlite):
        """Three concurrent writers all appending to the same table."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        barrier = threading.Barrier(3, timeout=10)
        errors = []

        def append(start: int):
            try:
                barrier.wait()
                df = pl.DataFrame({
                    "a": pl.Series(list(range(start, start + 10)), dtype=pl.Int32),
                })
                write_ducklake(
                    df, cat.metadata_path, "test",
                    data_path=cat.data_path, mode="append",
                    max_retries=5, retry_wait_ms=20,
                )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=append, args=(i * 100,))
            for i in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 30


# ── Test: concurrent deletes (should conflict) ─────────────────────────


class TestConcurrentDeletes:
    """Two concurrent deletes on the same table should conflict."""

    def test_conflict_check_delete_vs_delete(self):
        """_check_conflicts: concurrent deletes on the same table conflict."""
        writer = DuckLakeCatalogWriter.__new__(DuckLakeCatalogWriter)
        writer._backend = None
        writer._con = None

        with patch.object(
            writer, "_get_concurrent_changes",
            return_value=[(2, "deleted_from_table:1")],
        ):
            with pytest.raises(
                TransactionConflictError, match="concurrent deletes"
            ):
                writer._check_conflicts(0, {1: "delete"})

    def test_conflict_check_update_vs_delete(self):
        """_check_conflicts: update conflicts with concurrent delete."""
        writer = DuckLakeCatalogWriter.__new__(DuckLakeCatalogWriter)
        writer._backend = None
        writer._con = None

        with patch.object(
            writer, "_get_concurrent_changes",
            return_value=[(2, "deleted_from_table:1")],
        ):
            with pytest.raises(
                TransactionConflictError, match="concurrent deletes"
            ):
                writer._check_conflicts(0, {1: "update"})

    def test_concurrent_deletes_different_tables_ok(self):
        """Concurrent deletes on different tables don't conflict."""
        writer = DuckLakeCatalogWriter.__new__(DuckLakeCatalogWriter)
        writer._backend = None
        writer._con = None

        with patch.object(
            writer, "_get_concurrent_changes",
            return_value=[(2, "deleted_from_table:99")],
        ):
            # Deleting from table 1, concurrent delete from table 99 -> OK
            writer._check_conflicts(0, {1: "delete"})


# ── Test: concurrent append + schema change (should conflict) ──────────


class TestConcurrentAppendAndSchemaChange:
    """Concurrent append + schema change should conflict."""

    def test_conflict_check_insert_vs_alter(self):
        """_check_conflicts: insert conflicts with concurrent ALTER."""
        writer = DuckLakeCatalogWriter.__new__(DuckLakeCatalogWriter)
        writer._backend = None
        writer._con = None

        with patch.object(
            writer, "_get_concurrent_changes",
            return_value=[(2, "altered_table:1")],
        ):
            with pytest.raises(
                TransactionConflictError, match="schema was altered"
            ):
                writer._check_conflicts(0, {1: "insert"})

    def test_conflict_check_alter_vs_insert(self):
        """_check_conflicts: DDL conflicts with concurrent insert."""
        writer = DuckLakeCatalogWriter.__new__(DuckLakeCatalogWriter)
        writer._backend = None
        writer._con = None

        with patch.object(
            writer, "_get_concurrent_changes",
            return_value=[(2, "inserted_into_table:1")],
        ):
            with pytest.raises(
                TransactionConflictError, match="was modified"
            ):
                writer._check_conflicts(0, {1: "ddl"})

    def test_conflict_check_ddl_vs_ddl(self):
        """_check_conflicts: concurrent DDL on same table conflicts."""
        writer = DuckLakeCatalogWriter.__new__(DuckLakeCatalogWriter)
        writer._backend = None
        writer._con = None

        with patch.object(
            writer, "_get_concurrent_changes",
            return_value=[(2, "altered_table:1")],
        ):
            with pytest.raises(
                TransactionConflictError, match="was modified"
            ):
                writer._check_conflicts(0, {1: "ddl"})

    def test_conflict_check_ddl_vs_delete(self):
        """_check_conflicts: DDL conflicts with concurrent delete."""
        writer = DuckLakeCatalogWriter.__new__(DuckLakeCatalogWriter)
        writer._backend = None
        writer._con = None

        with patch.object(
            writer, "_get_concurrent_changes",
            return_value=[(2, "deleted_from_table:1")],
        ):
            with pytest.raises(
                TransactionConflictError, match="was modified"
            ):
                writer._check_conflicts(0, {1: "ddl"})


# ── Test: concurrent schema changes ────────────────────────────────────


class TestConcurrentSchemaChanges:
    """Two concurrent ALTER TABLE operations should conflict."""

    def test_two_add_columns_conflict(self):
        """Two concurrent ADD COLUMN on the same table conflict."""
        writer = DuckLakeCatalogWriter.__new__(DuckLakeCatalogWriter)
        writer._backend = None
        writer._con = None

        with patch.object(
            writer, "_get_concurrent_changes",
            return_value=[(2, "altered_table:1")],
        ):
            with pytest.raises(TransactionConflictError):
                writer._check_conflicts(0, {1: "ddl"})

    def test_concurrent_ddl_on_different_tables_ok(self):
        """DDL on different tables doesn't conflict."""
        writer = DuckLakeCatalogWriter.__new__(DuckLakeCatalogWriter)
        writer._backend = None
        writer._con = None

        with patch.object(
            writer, "_get_concurrent_changes",
            return_value=[(2, "altered_table:99")],
        ):
            # DDL on table 1, concurrent DDL on table 99 -> OK
            writer._check_conflicts(0, {1: "ddl"})

    def test_drop_table_vs_alter_conflict(self):
        """Drop table conflicts with concurrent ALTER on the same table."""
        writer = DuckLakeCatalogWriter.__new__(DuckLakeCatalogWriter)
        writer._backend = None
        writer._con = None

        with patch.object(
            writer, "_get_concurrent_changes",
            return_value=[(2, "altered_table:1")],
        ):
            with pytest.raises(TransactionConflictError, match="was modified"):
                writer._check_conflicts(0, {1: "drop_table"})

    def test_overwrite_vs_alter_conflict(self):
        """Overwrite conflicts with concurrent ALTER."""
        writer = DuckLakeCatalogWriter.__new__(DuckLakeCatalogWriter)
        writer._backend = None
        writer._con = None

        with patch.object(
            writer, "_get_concurrent_changes",
            return_value=[(2, "altered_table:1")],
        ):
            with pytest.raises(TransactionConflictError, match="was modified"):
                writer._check_conflicts(0, {1: "overwrite"})


# ── Test: retry exhaustion ──────────────────────────────────────────────


class TestRetryExhaustion:
    """Retry limits are respected and TransactionConflictError propagates."""

    def test_retryable_exhaustion_with_custom_settings(self):
        """Custom retry settings are honored."""
        from ducklake_core._writer import _retryable

        call_count = 0

        class MockWriter:
            _max_retries = 1
            _retry_wait_ms = 1
            _retry_backoff = 1.0

            def _reset_connection(self):
                pass

        @_retryable
        def always_fail(self):
            nonlocal call_count
            call_count += 1
            raise TransactionConflictError("always fails")

        with pytest.raises(TransactionConflictError, match="always fails"):
            always_fail(MockWriter())

        assert call_count == 2  # initial + 1 retry

    def test_retryable_zero_retries(self):
        """With max_retries=0, no retry is attempted."""
        from ducklake_core._writer import _retryable

        call_count = 0

        class MockWriter:
            _max_retries = 0
            _retry_wait_ms = 1
            _retry_backoff = 1.0

            def _reset_connection(self):
                pass

        @_retryable
        def fail(self):
            nonlocal call_count
            call_count += 1
            raise TransactionConflictError("no retry")

        with pytest.raises(TransactionConflictError, match="no retry"):
            fail(MockWriter())

        assert call_count == 1  # Only the initial attempt

    def test_backoff_increases_wait(self):
        """Exponential backoff doubles the wait between retries."""
        from ducklake_core._writer import _retryable

        waits = []

        class MockWriter:
            _max_retries = 3
            _retry_wait_ms = 10
            _retry_backoff = 2.0

            def _reset_connection(self):
                pass

        @_retryable
        def always_fail(self):
            raise TransactionConflictError("fail")

        with patch("time.sleep", side_effect=lambda s: waits.append(s)):
            with pytest.raises(TransactionConflictError):
                always_fail(MockWriter())

        # Waits: 10ms, 20ms, 40ms -> 0.01, 0.02, 0.04
        assert len(waits) == 3
        assert abs(waits[0] - 0.01) < 0.001
        assert abs(waits[1] - 0.02) < 0.001
        assert abs(waits[2] - 0.04) < 0.001

    def test_successful_on_last_retry(self):
        """Operation succeeds on the last allowed retry attempt."""
        from ducklake_core._writer import _retryable

        call_count = 0

        class MockWriter:
            _max_retries = 3
            _retry_wait_ms = 1
            _retry_backoff = 1.0

            def _reset_connection(self):
                pass

        @_retryable
        def succeed_on_last(self):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:  # Fail 3 times, succeed on 4th
                raise TransactionConflictError("not yet")
            return "done"

        result = succeed_on_last(MockWriter())
        assert result == "done"
        assert call_count == 4


# ── Test: change-set tracking ───────────────────────────────────────────


class TestChangeSetTracking:
    """Change-set tracking records tables and columns modified."""

    def test_track_table_write_basic(self):
        """_track_table_write records the operation."""
        writer = DuckLakeCatalogWriter.__new__(DuckLakeCatalogWriter)
        writer._txn_conflict_tables = {}
        writer._txn_touched_columns = {}

        writer._track_table_write(1, "insert")
        assert writer._txn_conflict_tables == {1: "insert"}

    def test_track_table_write_with_columns(self):
        """_track_table_write records column names for DDL operations."""
        writer = DuckLakeCatalogWriter.__new__(DuckLakeCatalogWriter)
        writer._txn_conflict_tables = {}
        writer._txn_touched_columns = {}

        writer._track_table_write(1, "ddl", columns=["new_col"])
        assert writer._txn_conflict_tables == {1: "ddl"}
        assert writer._txn_touched_columns == {1: {"new_col"}}

    def test_track_multiple_columns(self):
        """Multiple column tracking calls accumulate."""
        writer = DuckLakeCatalogWriter.__new__(DuckLakeCatalogWriter)
        writer._txn_conflict_tables = {}
        writer._txn_touched_columns = {}

        writer._track_table_write(1, "ddl", columns=["col_a"])
        writer._track_table_write(1, "ddl", columns=["col_b"])
        assert writer._txn_touched_columns[1] == {"col_a", "col_b"}

    def test_start_write_transaction_clears_tracking(self):
        """_start_write_transaction clears all tracking state."""
        writer = DuckLakeCatalogWriter.__new__(DuckLakeCatalogWriter)
        writer._txn_start_snapshot = 5
        writer._txn_conflict_tables = {1: "insert"}
        writer._txn_touched_columns = {1: {"col_a"}}

        writer._start_write_transaction(10)
        assert writer._txn_start_snapshot == 10
        assert writer._txn_conflict_tables == {}
        assert writer._txn_touched_columns == {}

    def test_reset_connection_clears_columns(self, ducklake_catalog_sqlite):
        """_reset_connection clears column tracking state."""
        cat = ducklake_catalog_sqlite
        cat.close()

        writer = DuckLakeCatalogWriter(
            cat.metadata_path, data_path_override=cat.data_path,
        )
        try:
            writer._connect()
            writer._txn_touched_columns = {1: {"col_a"}}
            writer._reset_connection()
            assert writer._txn_touched_columns == {}
        finally:
            writer.close()

    def test_record_change_contains_operation(self, ducklake_catalog_sqlite):
        """After insert, snapshot_changes records the correct operation string."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        write_ducklake(
            pl.DataFrame({"a": pl.Series([1], dtype=pl.Int32)}),
            cat.metadata_path, "test",
            data_path=cat.data_path, mode="append",
        )

        changes = _snapshot_changes(cat.metadata_path)
        # Find the insert change
        insert_changes = [c for c in changes if "inserted_into_table" in c]
        assert len(insert_changes) >= 1


# ── Test: three-way concurrent writes ───────────────────────────────────


class TestThreeWayConcurrentWrites:
    """Three concurrent writers: various combinations."""

    def test_three_appends_all_succeed(self, ducklake_catalog_sqlite):
        """Three concurrent appenders all succeed."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.close()

        barrier = threading.Barrier(3, timeout=10)
        errors = []

        def append(label: str, start: int):
            try:
                barrier.wait()
                df = pl.DataFrame({
                    "a": pl.Series(list(range(start, start + 5)), dtype=pl.Int32),
                    "b": [f"{label}_{i}" for i in range(5)],
                })
                write_ducklake(
                    df, cat.metadata_path, "test",
                    data_path=cat.data_path, mode="append",
                    max_retries=5, retry_wait_ms=20,
                )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=append, args=(f"W{i}", i * 100))
            for i in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 15

    def test_append_append_overwrite_mixed(self, ducklake_catalog_sqlite):
        """Two appenders + one overwriter: all eventually complete or conflict."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (0)")
        cat.close()

        errors = []
        successes = []
        lock = threading.Lock()

        def do_append(val: int):
            try:
                write_ducklake(
                    pl.DataFrame({"a": pl.Series([val], dtype=pl.Int32)}),
                    cat.metadata_path, "test",
                    data_path=cat.data_path, mode="append",
                    max_retries=5, retry_wait_ms=20,
                )
                with lock:
                    successes.append(("append", val))
            except Exception as e:
                with lock:
                    errors.append(("append", e))

        def do_overwrite():
            try:
                write_ducklake(
                    pl.DataFrame({"a": pl.Series([999], dtype=pl.Int32)}),
                    cat.metadata_path, "test",
                    data_path=cat.data_path, mode="overwrite",
                    max_retries=5, retry_wait_ms=20,
                )
                with lock:
                    successes.append(("overwrite", 999))
            except Exception as e:
                with lock:
                    errors.append(("overwrite", e))

        t1 = threading.Thread(target=do_append, args=(1,))
        t2 = threading.Thread(target=do_append, args=(2,))
        t3 = threading.Thread(target=do_overwrite)
        t1.start()
        t2.start()
        t3.start()
        t1.join()
        t2.join()
        t3.join()

        # At least the sequential operations should work
        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] >= 1


# ── Test: OCC with partitioned tables ───────────────────────────────────


class TestOCCPartitioned:
    """OCC works correctly with partitioned tables."""

    def test_concurrent_appends_partitioned_table(self, ducklake_catalog_sqlite):
        """Two concurrent appends to a partitioned table both succeed."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (region VARCHAR, val INTEGER)")
        cat.close()

        # Set partitioning
        with DuckLakeCatalogWriter(
            cat.metadata_path, data_path_override=cat.data_path,
        ) as writer:
            writer.set_partitioned_by("test", ["region"])

        errors = []

        def append(region: str, start: int):
            try:
                df = pl.DataFrame({
                    "region": [region] * 5,
                    "val": pl.Series(list(range(start, start + 5)), dtype=pl.Int32),
                })
                write_ducklake(
                    df, cat.metadata_path, "test",
                    data_path=cat.data_path, mode="append",
                    max_retries=5, retry_wait_ms=20,
                )
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=append, args=("US", 0))
        t2 = threading.Thread(target=append, args=("EU", 100))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0, f"Errors: {errors}"
        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 10
        assert set(result["region"].to_list()) == {"US", "EU"}

    def test_concurrent_appends_same_partition(self, ducklake_catalog_sqlite):
        """Two concurrent appends to the same partition value both succeed."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (part VARCHAR, val INTEGER)")
        cat.close()

        with DuckLakeCatalogWriter(
            cat.metadata_path, data_path_override=cat.data_path,
        ) as writer:
            writer.set_partitioned_by("test", ["part"])

        errors = []
        barrier = threading.Barrier(2, timeout=10)

        def append(start: int):
            try:
                barrier.wait()
                df = pl.DataFrame({
                    "part": ["same"] * 3,
                    "val": pl.Series(list(range(start, start + 3)), dtype=pl.Int32),
                })
                write_ducklake(
                    df, cat.metadata_path, "test",
                    data_path=cat.data_path, mode="append",
                    max_retries=5, retry_wait_ms=20,
                )
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=append, args=(0,))
        t2 = threading.Thread(target=append, args=(100,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0, f"Errors: {errors}"
        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 6


# ── Test: conflict matrix comprehensive ─────────────────────────────────


class TestConflictMatrix:
    """Comprehensive conflict matrix testing."""

    @pytest.mark.parametrize(
        "our_op,their_change,should_conflict",
        [
            # Inserts
            ("insert", "inserted_into_table:1", False),
            ("insert", "deleted_from_table:1", False),
            ("insert", "altered_table:1", True),
            ("insert", "dropped_table:1", True),
            # Deletes
            ("delete", "inserted_into_table:1", False),
            ("delete", "deleted_from_table:1", True),
            ("delete", "altered_table:1", True),
            ("delete", "dropped_table:1", True),
            # Updates
            ("update", "inserted_into_table:1", False),
            ("update", "deleted_from_table:1", True),
            ("update", "altered_table:1", True),
            ("update", "dropped_table:1", True),
            # Overwrite
            ("overwrite", "inserted_into_table:1", True),
            ("overwrite", "deleted_from_table:1", True),
            ("overwrite", "altered_table:1", True),
            ("overwrite", "dropped_table:1", True),
            # DDL
            ("ddl", "inserted_into_table:1", True),
            ("ddl", "deleted_from_table:1", True),
            ("ddl", "altered_table:1", True),
            ("ddl", "dropped_table:1", True),
            # Drop table
            ("drop_table", "inserted_into_table:1", True),
            ("drop_table", "deleted_from_table:1", True),
            ("drop_table", "altered_table:1", True),
            ("drop_table", "dropped_table:1", True),
        ],
    )
    def test_conflict_matrix(self, our_op, their_change, should_conflict):
        """Verify each cell of the conflict matrix."""
        writer = DuckLakeCatalogWriter.__new__(DuckLakeCatalogWriter)
        writer._backend = None
        writer._con = None

        with patch.object(
            writer, "_get_concurrent_changes",
            return_value=[(2, their_change)],
        ):
            if should_conflict:
                with pytest.raises(TransactionConflictError):
                    writer._check_conflicts(0, {1: our_op})
            else:
                # Should not raise
                writer._check_conflicts(0, {1: our_op})

    def test_no_concurrent_changes_no_conflict(self):
        """No concurrent changes -> no conflict for any operation."""
        writer = DuckLakeCatalogWriter.__new__(DuckLakeCatalogWriter)
        writer._backend = None
        writer._con = None

        with patch.object(
            writer, "_get_concurrent_changes",
            return_value=[],
        ):
            for op in ("insert", "delete", "update", "overwrite", "ddl", "drop_table"):
                writer._check_conflicts(0, {1: op})

    def test_empty_conflict_tables_no_check(self):
        """Empty conflict tables -> no conflict check at all."""
        writer = DuckLakeCatalogWriter.__new__(DuckLakeCatalogWriter)
        writer._backend = None
        writer._con = None

        # Even with concurrent changes, empty conflict_tables -> no conflict
        with patch.object(
            writer, "_get_concurrent_changes",
            return_value=[(2, "deleted_from_table:1")],
        ) as mock:
            writer._check_conflicts(0, {})
            mock.assert_not_called()


# ── Test: parse_table_changes edge cases ────────────────────────────────


class TestParseTableChanges:
    """Edge cases in _parse_table_changes."""

    def test_multiple_operations_same_table(self):
        """Multiple operations on the same table in one change string."""
        result = DuckLakeCatalogWriter._parse_table_changes(
            "inserted_into_table:1,deleted_from_table:1"
        )
        assert result == {1: {"inserted_into_table", "deleted_from_table"}}

    def test_multiple_tables(self):
        """Operations on multiple tables."""
        result = DuckLakeCatalogWriter._parse_table_changes(
            "inserted_into_table:1,altered_table:2,dropped_table:3"
        )
        assert result == {
            1: {"inserted_into_table"},
            2: {"altered_table"},
            3: {"dropped_table"},
        }

    def test_invalid_table_id_ignored(self):
        """Non-integer table IDs are silently ignored."""
        result = DuckLakeCatalogWriter._parse_table_changes(
            "inserted_into_table:abc"
        )
        assert result == {}

    def test_unrecognized_prefix_ignored(self):
        """Unrecognized change prefixes are silently ignored."""
        result = DuckLakeCatalogWriter._parse_table_changes(
            "unknown_change:1"
        )
        assert result == {}

    def test_whitespace_handling(self):
        """Whitespace around entries is handled."""
        result = DuckLakeCatalogWriter._parse_table_changes(
            "  inserted_into_table:1 , altered_table:2  "
        )
        assert result == {
            1: {"inserted_into_table"},
            2: {"altered_table"},
        }


# ── Test: writer constructor parameters ─────────────────────────────────


class TestWriterConstructorOCC:
    """Writer constructor accepts new OCC parameters."""

    def test_default_snapshot_retry_settings(self, ducklake_catalog_sqlite):
        """Default snapshot retry settings are applied."""
        cat = ducklake_catalog_sqlite
        cat.close()

        writer = DuckLakeCatalogWriter(
            cat.metadata_path, data_path_override=cat.data_path,
        )
        try:
            assert writer._max_snapshot_retries == 5
            assert writer._snapshot_retry_wait_ms == 50
        finally:
            writer.close()

    def test_custom_snapshot_retry_settings(self, ducklake_catalog_sqlite):
        """Custom snapshot retry settings are applied."""
        cat = ducklake_catalog_sqlite
        cat.close()

        writer = DuckLakeCatalogWriter(
            cat.metadata_path,
            data_path_override=cat.data_path,
            max_snapshot_retries=10,
            snapshot_retry_wait_ms=100,
        )
        try:
            assert writer._max_snapshot_retries == 10
            assert writer._snapshot_retry_wait_ms == 100
        finally:
            writer.close()

    def test_default_retry_settings(self, ducklake_catalog_sqlite):
        """Default OCC retry settings are applied."""
        cat = ducklake_catalog_sqlite
        cat.close()

        writer = DuckLakeCatalogWriter(
            cat.metadata_path, data_path_override=cat.data_path,
        )
        try:
            assert writer._max_retries == 3
            assert writer._retry_wait_ms == 100
            assert writer._retry_backoff == 2.0
        finally:
            writer.close()


# ── Test: end-to-end OCC integration ───────────────────────────────────


class TestOCCIntegration:
    """End-to-end tests that exercise the full OCC path."""

    def test_sequential_operations_maintain_consistency(
        self, ducklake_catalog_sqlite
    ):
        """Sequential insert -> delete -> insert maintains correct state."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, val VARCHAR)")
        cat.close()

        # Insert batch 1
        write_ducklake(
            pl.DataFrame({
                "id": pl.Series([1, 2, 3], dtype=pl.Int32),
                "val": ["a", "b", "c"],
            }),
            cat.metadata_path, "test",
            data_path=cat.data_path, mode="append",
        )

        # Delete some rows
        deleted = delete_ducklake(
            cat.metadata_path, "test",
            pl.col("id") == 2,
            data_path=cat.data_path,
        )
        assert deleted == 1

        # Insert batch 2
        write_ducklake(
            pl.DataFrame({
                "id": pl.Series([4, 5], dtype=pl.Int32),
                "val": ["d", "e"],
            }),
            cat.metadata_path, "test",
            data_path=cat.data_path, mode="append",
        )

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 4
        assert 2 not in result["id"].to_list()

    def test_concurrent_insert_and_read(self, ducklake_catalog_sqlite):
        """Concurrent insert and read don't interfere."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        cat.close()

        errors = []
        read_results = []

        def do_insert():
            try:
                write_ducklake(
                    pl.DataFrame({"a": pl.Series([2, 3], dtype=pl.Int32)}),
                    cat.metadata_path, "test",
                    data_path=cat.data_path, mode="append",
                )
            except Exception as e:
                errors.append(e)

        def do_read():
            try:
                result = read_ducklake(
                    cat.metadata_path, "test", data_path=cat.data_path,
                )
                read_results.append(result.shape[0])
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=do_insert)
        t2 = threading.Thread(target=do_read)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0, f"Errors: {errors}"
        # Read got either pre-insert (1 row) or post-insert (3 rows)
        assert read_results[0] in (1, 3)

    def test_schema_evolution_with_concurrent_insert(
        self, ducklake_catalog_sqlite
    ):
        """Schema change followed by insert with new schema works."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        # Insert initial data
        write_ducklake(
            pl.DataFrame({"a": pl.Series([1], dtype=pl.Int32)}),
            cat.metadata_path, "test",
            data_path=cat.data_path, mode="append",
        )

        # Add column
        alter_ducklake_add_column(
            cat.metadata_path, "test", "b", pl.Utf8,
            data_path=cat.data_path,
        )

        # Insert with new schema
        write_ducklake(
            pl.DataFrame({
                "a": pl.Series([2], dtype=pl.Int32),
                "b": ["hello"],
            }),
            cat.metadata_path, "test",
            data_path=cat.data_path, mode="append",
        )

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 2
        assert "b" in result.columns
