"""Remaining test-parity gaps with ducklake-ref.

Covers:
  1. Constraints — unsupported constraint errors (UNIQUE, PK, CHECK, FK)
  2. General/misc — attach_at_snapshot (time travel on open), read_only mode,
     missing_parquet error handling, database_size / catalog_info reporting
  3. Comments/tags edge cases — comment schema versioning after ALTER,
     comment on column after rename, comment survives drop+recreate
"""

from __future__ import annotations

import os
import stat
from datetime import datetime, timezone

import duckdb
import polars as pl
import pytest

from ducklake_polars import (
    DuckLakeCatalog,
    DuckLakeCatalogReader,
    alter_ducklake_add_column,
    alter_ducklake_drop_column,
    alter_ducklake_rename_column,
    catalog_info,
    create_ducklake_table,
    drop_ducklake_table,
    read_ducklake,
    scan_ducklake,
    set_ducklake_column_tag,
    set_ducklake_table_tag,
    write_ducklake,
)


# =====================================================================
# 1. CONSTRAINTS — unsupported constraint error handling
# =====================================================================


class TestUnsupportedConstraints:
    """DuckLake rejects UNIQUE, PRIMARY KEY, CHECK, and FOREIGN KEY via DuckDB."""

    def test_unique_constraint_rejected(self, ducklake_catalog_sqlite):
        """UNIQUE constraint should raise NotImplementedException in DuckDB."""
        cat = ducklake_catalog_sqlite
        with pytest.raises(duckdb.NotImplementedException, match="(?i)unique"):
            cat.execute("CREATE TABLE ducklake.t (a INTEGER UNIQUE)")

    def test_primary_key_constraint_rejected(self, ducklake_catalog_sqlite):
        """PRIMARY KEY constraint should raise NotImplementedException in DuckDB."""
        cat = ducklake_catalog_sqlite
        with pytest.raises(duckdb.NotImplementedException, match="(?i)primary key"):
            cat.execute("CREATE TABLE ducklake.t (a INTEGER PRIMARY KEY)")

    def test_check_constraint_rejected(self, ducklake_catalog_sqlite):
        """CHECK constraint should raise NotImplementedException in DuckDB."""
        cat = ducklake_catalog_sqlite
        with pytest.raises(duckdb.NotImplementedException, match="(?i)check"):
            cat.execute("CREATE TABLE ducklake.t (a INTEGER CHECK(a > 0))")

    def test_composite_primary_key_rejected(self, ducklake_catalog_sqlite):
        """Composite PRIMARY KEY should also be rejected."""
        cat = ducklake_catalog_sqlite
        with pytest.raises(duckdb.NotImplementedException):
            cat.execute(
                "CREATE TABLE ducklake.t (a INTEGER, b INTEGER, PRIMARY KEY(a, b))"
            )

    def test_not_null_is_allowed(self, ducklake_catalog_sqlite):
        """NOT NULL should still be accepted — only structural constraints rejected."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER NOT NULL, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'ok')")
        cat.close()
        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.shape[0] == 1


# =====================================================================
# 2. GENERAL / MISC — attach_at_snapshot, read_only, missing_parquet,
#    database_size
# =====================================================================


class TestAttachAtSnapshot:
    """Time-travel on open: read table at a specific snapshot version."""

    def test_read_at_snapshot_version(self, ducklake_catalog_sqlite):
        """Read data at different snapshot versions via read_ducklake."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER, b VARCHAR)")

        cat.execute("INSERT INTO ducklake.t VALUES (1, 'v1')")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("INSERT INTO ducklake.t VALUES (2, 'v2')")
        v2 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("INSERT INTO ducklake.t VALUES (3, 'v3')")
        cat.close()

        # At v1 — 1 row
        r1 = read_ducklake(
            cat.metadata_path, "t", snapshot_version=v1, data_path=cat.data_path
        )
        assert r1.shape[0] == 1
        assert r1["a"].to_list() == [1]

        # At v2 — 2 rows
        r2 = read_ducklake(
            cat.metadata_path, "t", snapshot_version=v2, data_path=cat.data_path
        )
        assert r2.shape[0] == 2
        assert sorted(r2["a"].to_list()) == [1, 2]

        # Latest — 3 rows
        r_latest = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert r_latest.shape[0] == 3

    def test_snapshot_with_schema_evolution(self, ducklake_catalog_sqlite):
        """Time travel across schema changes — columns appear/disappear."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1)")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("ALTER TABLE ducklake.t ADD COLUMN b VARCHAR")
        cat.execute("INSERT INTO ducklake.t VALUES (2, 'hello')")
        v2 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("ALTER TABLE ducklake.t ADD COLUMN c DOUBLE")
        cat.execute("INSERT INTO ducklake.t VALUES (3, 'world', 3.14)")
        cat.close()

        # At v1: only column 'a'
        r1 = read_ducklake(
            cat.metadata_path, "t", snapshot_version=v1, data_path=cat.data_path
        )
        assert r1.columns == ["a"]
        assert r1.shape[0] == 1

        # At v2: columns 'a', 'b'
        r2 = read_ducklake(
            cat.metadata_path, "t", snapshot_version=v2, data_path=cat.data_path
        )
        assert set(r2.columns) == {"a", "b"}
        assert r2.shape[0] == 2

    def test_snapshot_version_and_time_mutually_exclusive(self, ducklake_catalog_sqlite):
        """Cannot specify both snapshot_version and snapshot_time."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1)")
        cat.close()

        with pytest.raises(ValueError, match="(?i)both"):
            read_ducklake(
                cat.metadata_path,
                "t",
                snapshot_version=1,
                snapshot_time="2024-01-01T00:00:00",
                data_path=cat.data_path,
            )

    def test_invalid_snapshot_version_raises(self, ducklake_catalog_sqlite):
        """Reading at a non-existent snapshot version should raise."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1)")
        cat.close()

        with pytest.raises((ValueError, Exception)):
            read_ducklake(
                cat.metadata_path,
                "t",
                snapshot_version=99999,
                data_path=cat.data_path,
            )

    def test_scan_at_snapshot_lazy(self, ducklake_catalog_sqlite):
        """Lazy scan at a specific snapshot version with filter."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1), (2)")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.execute("INSERT INTO ducklake.t VALUES (3), (4)")
        cat.close()

        result = (
            scan_ducklake(
                cat.metadata_path,
                "t",
                snapshot_version=v1,
                data_path=cat.data_path,
            )
            .filter(pl.col("a") > 1)
            .collect()
        )
        assert result.shape[0] == 1
        assert result["a"].to_list() == [2]

    def test_snapshot_after_delete(self, ducklake_catalog_sqlite):
        """Time travel to before a delete still shows deleted rows."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1), (2), (3)")
        v_before = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.execute("DELETE FROM ducklake.t WHERE a = 2")
        cat.close()

        # Before delete: 3 rows
        r_before = read_ducklake(
            cat.metadata_path, "t", snapshot_version=v_before, data_path=cat.data_path
        )
        assert r_before.shape[0] == 3

        # After delete: 2 rows
        r_latest = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert r_latest.shape[0] == 2
        assert 2 not in r_latest["a"].to_list()


class TestReadOnlyMode:
    """Read-only catalog — reads succeed, writes fail."""

    def test_read_only_catalog_allows_reads(self, ducklake_catalog_sqlite):
        """A read-only SQLite file should still allow reads."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'hello'), (2, 'world')")
        cat.close()

        # Make read-only
        os.chmod(cat.metadata_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        try:
            result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
            assert result.shape[0] == 2
            assert sorted(result["a"].to_list()) == [1, 2]
        finally:
            os.chmod(cat.metadata_path, stat.S_IRUSR | stat.S_IWUSR)

    def test_read_only_catalog_rejects_writes(self, ducklake_catalog_sqlite):
        """Writing to a read-only catalog should raise an error."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1)")
        cat.close()

        os.chmod(cat.metadata_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        try:
            with pytest.raises(Exception):
                write_ducklake(
                    pl.DataFrame({"a": [2]}),
                    cat.metadata_path,
                    "t",
                    data_path=cat.data_path,
                    mode="append",
                )
        finally:
            os.chmod(cat.metadata_path, stat.S_IRUSR | stat.S_IWUSR)

    def test_read_only_time_travel_works(self, ducklake_catalog_sqlite):
        """Time travel should work on a read-only catalog."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1)")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.execute("INSERT INTO ducklake.t VALUES (2)")
        cat.close()

        os.chmod(cat.metadata_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        try:
            result = read_ducklake(
                cat.metadata_path,
                "t",
                snapshot_version=v1,
                data_path=cat.data_path,
            )
            assert result.shape[0] == 1
            assert result["a"].to_list() == [1]
        finally:
            os.chmod(cat.metadata_path, stat.S_IRUSR | stat.S_IWUSR)


class TestMissingParquetErrorHandling:
    """Graceful error when referenced Parquet files are missing."""

    def test_missing_parquet_raises(self, ducklake_catalog_sqlite):
        """Reading a table whose data files were deleted should raise."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1), (2), (3)")
        cat.close()

        # Verify data exists first
        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.shape[0] == 3

        # Delete parquet files
        data_dir = cat.data_path
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".parquet"):
                    os.remove(os.path.join(root, f))

        # Read should now fail with a clear error
        with pytest.raises((FileNotFoundError, OSError, Exception)):
            read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)

    def test_missing_parquet_after_multiple_inserts(self, ducklake_catalog_sqlite):
        """Multiple data files — deleting one should cause an error."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1)")
        cat.execute("INSERT INTO ducklake.t VALUES (2)")
        cat.execute("INSERT INTO ducklake.t VALUES (3)")
        cat.close()

        # Find and delete just one parquet file
        data_dir = cat.data_path
        parquet_files = []
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".parquet"):
                    parquet_files.append(os.path.join(root, f))

        assert len(parquet_files) >= 1
        os.remove(parquet_files[0])

        with pytest.raises((FileNotFoundError, OSError, Exception)):
            read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)

    def test_empty_table_no_parquet_needed(self, ducklake_catalog_sqlite):
        """An empty table (no data files) should read fine with no parquet."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER, b VARCHAR)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.shape[0] == 0
        assert set(result.columns) == {"a", "b"}


class TestDatabaseSizeReporting:
    """catalog_info reports snapshot_count, table_count, version, etc."""

    def test_catalog_info_basic(self, ducklake_catalog_sqlite):
        """catalog_info should return version, counts, data_path."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1)")
        cat.execute("CREATE TABLE ducklake.t2 (b VARCHAR)")
        cat.close()

        info = catalog_info(cat.metadata_path, data_path=cat.data_path)
        assert "version" in info
        assert info["table_count"] == 2
        assert info["snapshot_count"] > 0
        assert "data_path" in info
        assert "current_snapshot_id" in info

    def test_catalog_info_after_drop(self, ducklake_catalog_sqlite):
        """Dropping a table should reduce table_count."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("CREATE TABLE ducklake.t2 (b VARCHAR)")
        cat.close()

        info1 = catalog_info(cat.metadata_path, data_path=cat.data_path)
        assert info1["table_count"] == 2

        drop_ducklake_table(cat.metadata_path, "t1", data_path=cat.data_path)

        info2 = catalog_info(cat.metadata_path, data_path=cat.data_path)
        assert info2["table_count"] == 1
        assert info2["snapshot_count"] > info1["snapshot_count"]

    def test_catalog_info_snapshot_count_increments(self, ducklake_catalog_sqlite):
        """Each write operation should increment snapshot_count."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.close()

        info1 = catalog_info(cat.metadata_path, data_path=cat.data_path)
        snap1 = info1["snapshot_count"]

        write_ducklake(
            pl.DataFrame({"a": [1]}),
            cat.metadata_path,
            "t",
            data_path=cat.data_path,
            mode="append",
        )

        info2 = catalog_info(cat.metadata_path, data_path=cat.data_path)
        assert info2["snapshot_count"] == snap1 + 1

    def test_catalog_info_empty_catalog(self, tmp_path):
        """catalog_info on an empty catalog (no user tables)."""
        metadata_path = str(tmp_path / "empty.ducklake")
        data_path = str(tmp_path / "data")

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.install_extension("sqlite_scanner")
        con.load_extension("sqlite_scanner")
        con.execute(
            f"ATTACH 'ducklake:sqlite:{metadata_path}' AS dl "
            f"(DATA_PATH '{data_path}')"
        )
        con.close()

        info = catalog_info(metadata_path, data_path=data_path)
        assert info["table_count"] == 0


# =====================================================================
# 3. COMMENTS/TAGS EDGE CASES — schema versioning, rename, drop/recreate
# =====================================================================


class TestCommentSchemaVersioning:
    """Comments survive and track across schema changes (ALTER TABLE)."""

    def test_column_comment_survives_add_column(self, ducklake_catalog_sqlite):
        """Adding a new column doesn't disturb existing column comments."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER, b VARCHAR)")
        cat.execute("COMMENT ON COLUMN ducklake.t.a IS 'original a'")
        cat.execute("ALTER TABLE ducklake.t ADD COLUMN c DOUBLE")
        cat.close()

        catalog = DuckLakeCatalog(cat.metadata_path, data_path=cat.data_path)
        a_tags = catalog.column_tags("t", "a")
        assert a_tags.shape[0] == 1
        assert a_tags["value"][0] == "original a"

    def test_table_comment_survives_add_column(self, ducklake_catalog_sqlite):
        """Table comment survives ALTER TABLE ADD COLUMN."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("COMMENT ON TABLE ducklake.t IS 'my table desc'")
        cat.execute("ALTER TABLE ducklake.t ADD COLUMN b VARCHAR")
        cat.close()

        catalog = DuckLakeCatalog(cat.metadata_path, data_path=cat.data_path)
        tags = catalog.table_tags("t")
        assert tags.shape[0] == 1
        assert tags["value"][0] == "my table desc"

    def test_column_comment_survives_drop_other_column(self, ducklake_catalog_sqlite):
        """Dropping a different column doesn't disturb existing comments."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER, b VARCHAR, c DOUBLE)")
        cat.execute("COMMENT ON COLUMN ducklake.t.a IS 'col a desc'")
        cat.execute("COMMENT ON COLUMN ducklake.t.c IS 'col c desc'")
        cat.execute("ALTER TABLE ducklake.t DROP COLUMN b")
        cat.close()

        catalog = DuckLakeCatalog(cat.metadata_path, data_path=cat.data_path)
        a_tags = catalog.column_tags("t", "a")
        assert a_tags.shape[0] == 1
        assert a_tags["value"][0] == "col a desc"

        c_tags = catalog.column_tags("t", "c")
        assert c_tags.shape[0] == 1
        assert c_tags["value"][0] == "col c desc"

    def test_comment_on_newly_added_column(self, ducklake_catalog_sqlite):
        """Comment on a column that was added via ALTER TABLE."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("ALTER TABLE ducklake.t ADD COLUMN b VARCHAR")
        cat.execute("COMMENT ON COLUMN ducklake.t.b IS 'added later'")
        cat.close()

        catalog = DuckLakeCatalog(cat.metadata_path, data_path=cat.data_path)
        b_tags = catalog.column_tags("t", "b")
        assert b_tags.shape[0] == 1
        assert b_tags["value"][0] == "added later"

    def test_comment_via_writer_after_alter(self, ducklake_catalog_sqlite):
        """Set column comment via writer API after schema has been altered."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("ALTER TABLE ducklake.t ADD COLUMN b VARCHAR")
        cat.close()

        set_ducklake_column_tag(
            cat.metadata_path,
            "t",
            "a",
            "comment",
            "col a after alter",
            data_path=cat.data_path,
        )
        set_ducklake_column_tag(
            cat.metadata_path,
            "t",
            "b",
            "comment",
            "col b after alter",
            data_path=cat.data_path,
        )

        catalog = DuckLakeCatalog(cat.metadata_path, data_path=cat.data_path)
        assert catalog.column_tags("t", "a")["value"][0] == "col a after alter"
        assert catalog.column_tags("t", "b")["value"][0] == "col b after alter"

    def test_comment_time_travel_across_schema_versions(self, ducklake_catalog_sqlite):
        """Comments at different schema versions should be correctly versioned."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("COMMENT ON TABLE ducklake.t IS 'v1 schema'")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("ALTER TABLE ducklake.t ADD COLUMN b VARCHAR")
        cat.execute("COMMENT ON TABLE ducklake.t IS 'v2 schema'")
        v2 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        # Time travel verification using the reader
        with DuckLakeCatalogReader(
            cat.metadata_path, data_path_override=cat.data_path
        ) as reader:
            table_info = reader.get_table("t", "main", v1)
            tags_v1 = reader.get_table_tags(table_info.table_id, v1)
            assert tags_v1.get("comment") == "v1 schema"

            tags_v2 = reader.get_table_tags(table_info.table_id, v2)
            assert tags_v2.get("comment") == "v2 schema"

    def test_column_comment_after_rename(self, ducklake_catalog_sqlite):
        """Renaming a column preserves its comment (same column_id)."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER, b VARCHAR)")
        cat.execute("COMMENT ON COLUMN ducklake.t.a IS 'the a column'")
        cat.execute("ALTER TABLE ducklake.t RENAME COLUMN a TO x")
        cat.close()

        catalog = DuckLakeCatalog(cat.metadata_path, data_path=cat.data_path)
        # After rename, the comment should be accessible via the new name
        x_tags = catalog.column_tags("t", "x")
        assert x_tags.shape[0] == 1
        assert x_tags["value"][0] == "the a column"

    def test_comment_on_renamed_column_via_writer(self, ducklake_catalog_sqlite):
        """Set comment on a renamed column via the writer API."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.close()

        alter_ducklake_rename_column(
            cat.metadata_path, "t", "a", "x", data_path=cat.data_path
        )
        set_ducklake_column_tag(
            cat.metadata_path,
            "t",
            "x",
            "comment",
            "renamed column",
            data_path=cat.data_path,
        )

        catalog = DuckLakeCatalog(cat.metadata_path, data_path=cat.data_path)
        tags = catalog.column_tags("t", "x")
        assert tags.shape[0] == 1
        assert tags["value"][0] == "renamed column"

    def test_comment_survives_drop_recreate(self, tmp_path):
        """Drop and recreate table — old comments should NOT carry over."""
        metadata_path = str(tmp_path / "test.ducklake")
        data_path = str(tmp_path / "data")

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.install_extension("sqlite_scanner")
        con.load_extension("sqlite_scanner")
        con.execute(
            f"ATTACH 'ducklake:sqlite:{metadata_path}' AS dl "
            f"(DATA_PATH '{data_path}')"
        )
        con.execute("CREATE TABLE dl.t (a INTEGER)")
        con.execute("COMMENT ON TABLE dl.t IS 'original'")
        con.execute("DROP TABLE dl.t")
        con.execute("CREATE TABLE dl.t (b VARCHAR)")
        con.close()

        catalog = DuckLakeCatalog(metadata_path, data_path=data_path)
        tags = catalog.table_tags("t")
        # Recreated table is a new entity — should have no tags
        assert tags.shape[0] == 0

    def test_multiple_comments_across_schema_ops(self, ducklake_catalog_sqlite):
        """Interleave comments and schema changes in one DuckDB session."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("COMMENT ON TABLE ducklake.t IS 'initial'")
        cat.execute("ALTER TABLE ducklake.t ADD COLUMN b VARCHAR")
        cat.execute("COMMENT ON COLUMN ducklake.t.b IS 'added col'")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'x')")
        cat.execute("ALTER TABLE ducklake.t ADD COLUMN c DOUBLE")
        cat.execute("COMMENT ON COLUMN ducklake.t.c IS 'second added'")
        cat.execute("COMMENT ON TABLE ducklake.t IS 'final'")
        cat.close()

        catalog = DuckLakeCatalog(cat.metadata_path, data_path=cat.data_path)

        # Table comment should be the last one set
        table_tags = catalog.table_tags("t")
        assert table_tags.shape[0] == 1
        assert table_tags["value"][0] == "final"

        # Column comments should all be present
        b_tags = catalog.column_tags("t", "b")
        assert b_tags["value"][0] == "added col"

        c_tags = catalog.column_tags("t", "c")
        assert c_tags["value"][0] == "second added"

        # Data should be intact
        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.shape[0] == 1
