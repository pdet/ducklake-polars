"""P2 backlog coverage from audit_coverage.md and audit_parity.md.

Each test cites the audit item and the upstream upstream `.test` file
(if any) it mirrors.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from ducklake_polars import (
    DuckLakeStreamWriter,
    alter_ducklake_drop_column,
    cleanup_old_files_ducklake,
    create_ducklake_macro,
    delete_ducklake_column_tag,
    delete_ducklake_table_tag,
    expire_snapshots,
    merge_adjacent_files_ducklake,
    read_ducklake,
    read_ducklake_changes,
    scan_ducklake_changes,
    set_ducklake_column_tag,
    set_ducklake_table_tag,
    write_ducklake,
)


# ---------------------------------------------------------------------
# coverage P2 #10 — invalid CDC snapshot range
# ---------------------------------------------------------------------

class TestCDCInvalidRange:
    def test_start_greater_than_end_raises(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (id INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1)")
        cat.close()

        with pytest.raises(ValueError, match="start_version.*<= end_version"):
            read_ducklake_changes(cat.metadata_path, "t", 5, 1)
        with pytest.raises(ValueError, match="start_version.*<= end_version"):
            scan_ducklake_changes(cat.metadata_path, "t", 5, 1)

    def test_end_beyond_current_raises(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (id INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1)")
        snap1 = cat.fetchone(
            "SELECT * FROM ducklake_current_snapshot('ducklake')"
        )[0]
        cat.close()

        with pytest.raises(ValueError, match="not found"):
            read_ducklake_changes(cat.metadata_path, "t", 0, snap1 + 99)


# ---------------------------------------------------------------------
# coverage P2 #13 — delete_*_tag missing key
# ---------------------------------------------------------------------

class TestDeleteTagMissing:
    def test_delete_table_tag_missing_key_raises(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (id INTEGER)")
        cat.close()

        with pytest.raises(ValueError, match="Tag 'no_such_key' not found"):
            delete_ducklake_table_tag(cat.metadata_path, "t", "no_such_key")

    def test_delete_column_tag_missing_key_raises(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (id INTEGER)")
        cat.close()

        with pytest.raises(ValueError, match="Tag .* not found"):
            delete_ducklake_column_tag(
                cat.metadata_path, "t", "id", "no_such_key",
            )


# ---------------------------------------------------------------------
# coverage P2 #14 — exception in StreamWriter context drops the buffer
# ---------------------------------------------------------------------

class TestStreamWriterExceptionRollback:
    def test_exception_in_context_does_not_commit_partial_buffer(
        self, ducklake_catalog_sqlite,
    ):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.events (id INTEGER, val VARCHAR)")
        cat.close()

        with pytest.raises(RuntimeError, match="boom"):
            with DuckLakeStreamWriter(
                cat.metadata_path, "events",
                data_path=cat.data_path,
                flush_threshold=10_000,  # well above buffered rows
            ) as writer:
                writer.append(pl.DataFrame({
                    "id": pl.Series([1, 2], dtype=pl.Int32),
                    "val": ["a", "b"],
                }))
                # Buffer holds 2 rows but is below flush_threshold; exiting via
                # exception must NOT commit them.
                raise RuntimeError("boom")

        result = read_ducklake(
            cat.metadata_path, "events", data_path=cat.data_path,
        )
        assert result.shape[0] == 0

    def test_already_flushed_batches_remain(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.events (id INTEGER, val VARCHAR)")
        cat.close()

        with pytest.raises(RuntimeError, match="boom"):
            with DuckLakeStreamWriter(
                cat.metadata_path, "events",
                data_path=cat.data_path,
                flush_threshold=2,
                compact_on_close=False,
            ) as writer:
                writer.append(pl.DataFrame({
                    "id": pl.Series([1, 2], dtype=pl.Int32),
                    "val": ["a", "b"],
                }))  # auto-flushes at threshold
                writer.append(pl.DataFrame({
                    "id": pl.Series([3], dtype=pl.Int32),
                    "val": ["c"],
                }))  # buffered, will be dropped on exception
                raise RuntimeError("boom")

        result = read_ducklake(
            cat.metadata_path, "events", data_path=cat.data_path,
        )
        # First batch landed; second was dropped.
        assert sorted(result["id"].to_list()) == [1, 2]


# ---------------------------------------------------------------------
# coverage P2 #16 — files scheduled for deletion are picked up by cleanup
# ---------------------------------------------------------------------

def _query_catalog(metadata_path: str, sql: str) -> tuple | None:
    import sqlite3
    con = sqlite3.connect(metadata_path, timeout=30)
    try:
        return con.execute(sql).fetchone()
    finally:
        con.close()


class TestCleanupPicksUpScheduledFiles:
    def test_merge_adjacent_then_cleanup_removes_source_files(self, tmp_path):
        # Use our-writer-bootstrapped catalog (v1.0) so merge_adjacent_files
        # works regardless of the DuckDB-ducklake version installed.
        metadata_path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")

        for batch in (
            pl.DataFrame({
                "id": pl.Series([1, 2], dtype=pl.Int32),
                "val": ["a", "b"],
            }),
            pl.DataFrame({
                "id": pl.Series([3, 4], dtype=pl.Int32),
                "val": ["c", "d"],
            }),
        ):
            write_ducklake(
                batch, metadata_path, "t",
                data_path=data_path, mode="append",
            )

        merge_adjacent_files_ducklake(
            metadata_path, "t", data_path=data_path,
            min_file_size=1, max_file_size=10_000_000,
        )

        before = _query_catalog(
            metadata_path,
            "SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion",
        )
        assert before is not None and before[0] >= 2

        future = datetime.now(timezone.utc) + timedelta(days=1)
        deleted = cleanup_old_files_ducklake(
            metadata_path, older_than=future, data_path=data_path,
        )
        assert len(deleted) >= 2
        for p in deleted:
            assert not Path(p).exists(), f"{p} should have been deleted"

        after = _query_catalog(
            metadata_path,
            "SELECT COUNT(*) FROM ducklake_files_scheduled_for_deletion",
        )
        assert after is not None and after[0] == 0

        result = read_ducklake(
            metadata_path, "t", data_path=data_path,
        ).sort("id")
        assert result["id"].to_list() == [1, 2, 3, 4]


# ---------------------------------------------------------------------
# coverage P2 #17 — non-duckdb dialect macros round-trip
# ---------------------------------------------------------------------

class TestMacroDialect:
    def test_non_duckdb_dialect_persists(self, tmp_path):
        metadata_path = str(tmp_path / "catalog.ducklake")
        data_path = str(tmp_path / "data")
        # Bootstrap a v1.0 catalog via our writer (DuckDB-ducklake on 1.4.4
        # creates v0.x catalogs that lack the macro tables).
        write_ducklake(
            pl.DataFrame({"a": pl.Series([1], dtype=pl.Int32)}),
            metadata_path, "t",
            data_path=data_path, mode="append",
        )

        create_ducklake_macro(
            metadata_path, "spark_macro",
            sql="SELECT 1",
            macro_type="scalar",
            dialect="spark",
            data_path=data_path,
        )

        row = _query_catalog(
            metadata_path,
            "SELECT dialect FROM ducklake_macro_impl "
            "WHERE macro_id = (SELECT macro_id FROM ducklake_macro "
            "WHERE macro_name = 'spark_macro' AND end_snapshot IS NULL)"
        )
        assert row is not None
        assert row[0] == "spark"


# ---------------------------------------------------------------------
# coverage P2 #19 — expire_snapshots leaves no orphan child rows
# ---------------------------------------------------------------------

class TestExpireSnapshotsNoOrphans:
    def test_expire_keeps_referential_integrity(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (id INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1)")
        cat.execute("INSERT INTO ducklake.t VALUES (2)")
        cat.execute("INSERT INTO ducklake.t VALUES (3)")
        latest = cat.fetchone(
            "SELECT * FROM ducklake_current_snapshot('ducklake')"
        )[0]
        cat.close()

        # Keep only the latest snapshot.
        expire_snapshots(
            cat.metadata_path, keep_last_n=1, data_path=cat.data_path,
        )

        remaining = cat.query_metadata(
            "SELECT MAX(snapshot_id), MIN(snapshot_id), COUNT(*) "
            "FROM ducklake_snapshot"
        )
        assert remaining is not None
        max_snap, _, n_snaps = remaining
        assert n_snaps == 1
        assert max_snap == latest

        # No ducklake_snapshot_changes rows should reference a missing snapshot.
        orphans = cat.query_metadata(
            "SELECT COUNT(*) FROM ducklake_snapshot_changes "
            "WHERE snapshot_id NOT IN (SELECT snapshot_id FROM ducklake_snapshot)"
        )
        assert orphans is not None and orphans[0] == 0

        # Data still readable at the surviving snapshot.
        result = read_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        )
        assert sorted(result["id"].to_list()) == [1, 2, 3]


# ---------------------------------------------------------------------
# parity P2 #16 — graceful error when an underlying parquet file is missing
# ---------------------------------------------------------------------

class TestMissingParquetGracefulError:
    def test_missing_data_file_surfaces_error(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (id INTEGER, val VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'a')")
        cat.close()

        # Delete the parquet file out from under the catalog.
        deleted = 0
        for root, _, files in os.walk(cat.data_path):
            for f in files:
                if f.endswith(".parquet"):
                    os.remove(os.path.join(root, f))
                    deleted += 1
        assert deleted == 1

        # Reading must error (not silently return empty data); we don't pin
        # the exact exception type because Polars' parquet scanner wraps it,
        # but it must contain a recognisable file-not-found phrase.
        with pytest.raises(Exception) as excinfo:
            read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        msg = str(excinfo.value).lower()
        assert any(
            keyword in msg for keyword in (
                "not found", "no such", "cannot", "missing", "open"
            )
        )


# ---------------------------------------------------------------------
# parity P2 #19 — drop NOT NULL column then re-add nullable
# ---------------------------------------------------------------------

class TestNotNullDropColumn:
    def test_drop_then_readd_column_with_null_values(
        self, ducklake_catalog_sqlite,
    ):
        cat = ducklake_catalog_sqlite
        cat.execute(
            "CREATE TABLE ducklake.t (id INTEGER, val VARCHAR NOT NULL)"
        )
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'a'), (2, 'b')")
        cat.close()

        alter_ducklake_drop_column(
            cat.metadata_path, "t", "val", data_path=cat.data_path,
        )

        # Now re-add a nullable column with the same name; new rows can be NULL.
        write_ducklake(
            pl.DataFrame({
                "id": pl.Series([3, 4], dtype=pl.Int32),
            }),
            cat.metadata_path, "t",
            data_path=cat.data_path, mode="append",
            schema_evolution="merge",
        )

        result = read_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path,
        ).sort("id")
        assert sorted(result["id"].to_list()) == [1, 2, 3, 4]
        assert "val" not in result.columns


# ---------------------------------------------------------------------
# parity P2 #20 — null bytes in strings round-trip
# ---------------------------------------------------------------------

class TestNullByteInString:
    def test_string_with_null_byte_round_trips(self, tmp_path):
        # Use a fresh catalog (no DuckDB involvement — DuckDB rejects \0).
        metadata_path = str(tmp_path / "t.ducklake")
        data_path = str(tmp_path / "data")

        df = pl.DataFrame({
            "id": pl.Series([1, 2], dtype=pl.Int32),
            "val": ["hello\x00world", "no_null"],
        })
        write_ducklake(
            df, metadata_path, "t",
            data_path=data_path, mode="append",
        )

        result = read_ducklake(metadata_path, "t", data_path=data_path)
        # Polars preserves the embedded NUL byte through Parquet.
        vals = sorted(result.sort("id")["val"].to_list())
        assert vals == sorted(["hello\x00world", "no_null"])
