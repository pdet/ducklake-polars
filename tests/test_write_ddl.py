"""Tests for ducklake-polars DDL operations (rename column, drop table,
create/drop schema, rename table)."""

from __future__ import annotations

import os
import sqlite3

import duckdb
import polars as pl
import pytest

from ducklake_polars import (
    alter_ducklake_rename_column,
    create_ducklake_schema,
    create_ducklake_table,
    drop_ducklake_schema,
    drop_ducklake_table,
    read_ducklake,
    rename_ducklake_table,
    write_ducklake,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_catalog(tmp_path):
    """Create a DuckLake catalog via DuckDB and return (metadata_path, data_path)."""
    metadata_path = str(tmp_path / "ddl_test.ducklake")
    data_path = str(tmp_path / "data")
    os.makedirs(data_path, exist_ok=True)

    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:sqlite:{metadata_path}' AS ducklake "
        f"(DATA_PATH '{data_path}', DATA_INLINING_ROW_LIMIT 0)"
    )
    con.close()
    return metadata_path, data_path


def _read_with_duckdb(metadata_path, data_path, table_name, schema_name="main"):
    """Read a table with DuckDB's DuckLake extension."""
    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:sqlite:{metadata_path}' AS ducklake "
        f"(DATA_PATH '{data_path}', DATA_INLINING_ROW_LIMIT 0)"
    )
    safe_schema = schema_name.replace('"', '""')
    safe_table = table_name.replace('"', '""')
    cursor = con.execute(f'SELECT * FROM ducklake."{safe_schema}"."{safe_table}"')
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    con.close()
    if not rows:
        return pl.DataFrame({c: [] for c in columns})
    data = {c: [r[i] for r in rows] for i, c in enumerate(columns)}
    return pl.DataFrame(data)


# ===========================================================================
# RENAME COLUMN
# ===========================================================================


class TestRenameColumn:
    """Test ALTER TABLE RENAME COLUMN."""

    def test_rename_column_basic(self, tmp_path):
        """Rename a column and read back."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        write_ducklake(df, metadata_path, "test", mode="error")

        alter_ducklake_rename_column(metadata_path, "test", "b", "c")

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result.columns == ["a", "c"]
        assert result["a"].to_list() == [1, 2, 3]
        assert result["c"].to_list() == ["x", "y", "z"]

    def test_rename_column_metadata_correct(self, tmp_path):
        """Verify metadata: old row ended, new row inserted with same column_id."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1], "b": ["hello"]})
        write_ducklake(df, metadata_path, "test", mode="error")

        con = sqlite3.connect(metadata_path)
        sv_before = con.execute(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()[0]
        # Get column_id of 'b'
        col_b_row = con.execute(
            "SELECT column_id FROM ducklake_column "
            "WHERE column_name = 'b' AND end_snapshot IS NULL"
        ).fetchone()
        col_b_id = col_b_row[0]
        con.close()

        alter_ducklake_rename_column(metadata_path, "test", "b", "c")

        con = sqlite3.connect(metadata_path)
        # Schema version incremented
        sv_after = con.execute(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()[0]
        assert sv_after == sv_before + 1

        # Old column row has end_snapshot set
        old_col = con.execute(
            "SELECT end_snapshot FROM ducklake_column "
            "WHERE column_name = 'b'"
        ).fetchone()
        assert old_col is not None
        assert old_col[0] is not None

        # New column row uses the same column_id
        new_col = con.execute(
            "SELECT column_id, column_name FROM ducklake_column "
            "WHERE column_name = 'c' AND end_snapshot IS NULL"
        ).fetchone()
        assert new_col is not None
        assert new_col[0] == col_b_id  # same column_id

        # Changes recorded
        change = con.execute(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()
        assert "altered_table" in change[0]

        con.close()

    def test_rename_column_then_insert(self, tmp_path):
        """Insert data after renaming a column."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        write_ducklake(df, metadata_path, "test", mode="error")

        alter_ducklake_rename_column(metadata_path, "test", "b", "c")

        new_row = pl.DataFrame({"a": [3], "c": ["z"]})
        write_ducklake(new_row, metadata_path, "test", mode="append")

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result.columns == ["a", "c"]
        assert result["a"].to_list() == [1, 2, 3]
        assert result["c"].to_list() == ["x", "y", "z"]

    def test_rename_column_time_travel(self, tmp_path):
        """Time travel sees the old column name."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        write_ducklake(df, metadata_path, "test", mode="error")

        con = sqlite3.connect(metadata_path)
        snap_before = con.execute(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        ).fetchone()[0]
        con.close()

        alter_ducklake_rename_column(metadata_path, "test", "b", "c")

        # Latest: column is 'c'
        result = read_ducklake(metadata_path, "test")
        assert "c" in result.columns
        assert "b" not in result.columns

        # Before rename: column is 'b'
        result_old = read_ducklake(
            metadata_path, "test", snapshot_version=snap_before
        )
        assert "b" in result_old.columns
        assert "c" not in result_old.columns

    def test_rename_column_nonexistent_raises(self, tmp_path):
        """Renaming a column that doesn't exist raises."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1]})
        write_ducklake(df, metadata_path, "test", mode="error")

        with pytest.raises(ValueError, match="not found"):
            alter_ducklake_rename_column(
                metadata_path, "test", "missing", "new"
            )

    def test_rename_column_duplicate_raises(self, tmp_path):
        """Renaming to an existing column name raises."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1], "b": [2]})
        write_ducklake(df, metadata_path, "test", mode="error")

        with pytest.raises(ValueError, match="already exists"):
            alter_ducklake_rename_column(metadata_path, "test", "b", "a")

    def test_rename_column_nonexistent_table_raises(self, tmp_path):
        """Renaming on a nonexistent table raises."""
        metadata_path, data_path = _make_catalog(tmp_path)

        with pytest.raises(ValueError, match="not found"):
            alter_ducklake_rename_column(
                metadata_path, "missing", "a", "b"
            )

    def test_rename_column_duckdb_interop(self, tmp_path):
        """DuckDB can read after polars renames a column."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        write_ducklake(df, metadata_path, "test", mode="error")

        alter_ducklake_rename_column(metadata_path, "test", "b", "c")

        pdf = _read_with_duckdb(metadata_path, data_path, "test").sort("a")
        assert "c" in pdf.columns
        assert "b" not in pdf.columns
        assert pdf["a"].to_list() == [1, 2]
        assert pdf["c"].to_list() == ["x", "y"]

    def test_duckdb_rename_column_polars_reads(self, tmp_path):
        """DuckDB renames a column, polars reads correctly."""
        metadata_path = str(tmp_path / "test.ducklake")
        data_path = str(tmp_path / "data")
        os.makedirs(data_path, exist_ok=True)

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:sqlite:{metadata_path}' AS ducklake "
            f"(DATA_PATH '{data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        con.execute("INSERT INTO ducklake.test VALUES (1, 'hello'), (2, 'world')")
        con.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO c")
        con.close()

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result.columns == ["a", "c"]
        assert result["a"].to_list() == [1, 2]
        assert result["c"].to_list() == ["hello", "world"]

    def test_rename_column_multiple(self, tmp_path):
        """Rename multiple columns sequentially."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
        write_ducklake(df, metadata_path, "test", mode="error")

        alter_ducklake_rename_column(metadata_path, "test", "b", "bb")
        alter_ducklake_rename_column(metadata_path, "test", "c", "cc")

        result = read_ducklake(metadata_path, "test")
        assert result.columns == ["a", "bb", "cc"]
        assert result["a"].to_list() == [1]
        assert result["bb"].to_list() == [2]
        assert result["cc"].to_list() == [3]


# ===========================================================================
# DROP TABLE
# ===========================================================================


class TestDropTable:
    """Test DROP TABLE."""

    def test_drop_table_basic(self, tmp_path):
        """Drop a table and verify it's no longer readable."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, metadata_path, "test", mode="error")

        drop_ducklake_table(metadata_path, "test")

        with pytest.raises(ValueError, match="not found"):
            read_ducklake(metadata_path, "test")

    def test_drop_table_metadata_correct(self, tmp_path):
        """Verify metadata: end_snapshot on table, columns, data files."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1], "b": ["hello"]})
        write_ducklake(df, metadata_path, "test", mode="error")

        con = sqlite3.connect(metadata_path)
        sv_before = con.execute(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()[0]
        con.close()

        drop_ducklake_table(metadata_path, "test")

        con = sqlite3.connect(metadata_path)
        # Schema version incremented
        sv_after = con.execute(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()[0]
        assert sv_after == sv_before + 1

        # Table has end_snapshot set
        table_row = con.execute(
            "SELECT end_snapshot FROM ducklake_table "
            "WHERE table_name = 'test'"
        ).fetchone()
        assert table_row[0] is not None

        # All columns have end_snapshot set
        active_cols = con.execute(
            "SELECT COUNT(*) FROM ducklake_column WHERE end_snapshot IS NULL"
        ).fetchone()[0]
        assert active_cols == 0

        # All data files have end_snapshot set
        active_files = con.execute(
            "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL"
        ).fetchone()[0]
        assert active_files == 0

        # Changes recorded
        change = con.execute(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()
        assert "dropped_table" in change[0]

        con.close()

    def test_drop_table_with_delete_files(self, tmp_path):
        """Drop a table that has delete files."""
        from ducklake_polars import delete_ducklake

        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, metadata_path, "test", mode="error")
        delete_ducklake(metadata_path, "test", pl.col("a") == 1)

        drop_ducklake_table(metadata_path, "test")

        con = sqlite3.connect(metadata_path)
        # Delete files should also be ended
        active_deletes = con.execute(
            "SELECT COUNT(*) FROM ducklake_delete_file WHERE end_snapshot IS NULL"
        ).fetchone()[0]
        assert active_deletes == 0
        con.close()

    def test_drop_table_time_travel(self, tmp_path):
        """Time travel can read the table before it was dropped."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, metadata_path, "test", mode="error")

        con = sqlite3.connect(metadata_path)
        snap_before = con.execute(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        ).fetchone()[0]
        con.close()

        drop_ducklake_table(metadata_path, "test")

        # Latest: table is gone
        with pytest.raises(ValueError, match="not found"):
            read_ducklake(metadata_path, "test")

        # Before drop: table exists
        result = read_ducklake(
            metadata_path, "test", snapshot_version=snap_before
        )
        assert result["a"].to_list() == [1, 2, 3]

    def test_drop_table_nonexistent_raises(self, tmp_path):
        """Dropping a nonexistent table raises."""
        metadata_path, data_path = _make_catalog(tmp_path)

        with pytest.raises(ValueError, match="not found"):
            drop_ducklake_table(metadata_path, "missing")

    def test_drop_table_duckdb_interop(self, tmp_path):
        """DuckDB confirms table is dropped."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df, metadata_path, "test", mode="error")

        drop_ducklake_table(metadata_path, "test")

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:sqlite:{metadata_path}' AS ducklake "
            f"(DATA_PATH '{data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        with pytest.raises(duckdb.CatalogException):
            con.execute("SELECT * FROM ducklake.test")
        con.close()

    def test_duckdb_drop_table_polars_reads(self, tmp_path):
        """DuckDB drops a table, polars confirms it's gone."""
        metadata_path = str(tmp_path / "test.ducklake")
        data_path = str(tmp_path / "data")
        os.makedirs(data_path, exist_ok=True)

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:sqlite:{metadata_path}' AS ducklake "
            f"(DATA_PATH '{data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE TABLE ducklake.test (a INTEGER)")
        con.execute("INSERT INTO ducklake.test VALUES (1)")
        con.execute("DROP TABLE ducklake.test")
        con.close()

        with pytest.raises(ValueError, match="not found"):
            read_ducklake(metadata_path, "test")

    def test_drop_and_recreate_table(self, tmp_path):
        """Drop a table and create a new one with the same name."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df1, metadata_path, "test", mode="error")

        drop_ducklake_table(metadata_path, "test")

        df2 = pl.DataFrame({"x": ["hello", "world"]})
        write_ducklake(df2, metadata_path, "test", mode="error")

        result = read_ducklake(metadata_path, "test").sort("x")
        assert result.columns == ["x"]
        assert result["x"].to_list() == ["hello", "world"]


# ===========================================================================
# CREATE SCHEMA
# ===========================================================================


class TestCreateSchema:
    """Test CREATE SCHEMA."""

    def test_create_schema_basic(self, tmp_path):
        """Create a new schema."""
        metadata_path, data_path = _make_catalog(tmp_path)

        create_ducklake_schema(metadata_path, "myschema")

        con = sqlite3.connect(metadata_path)
        row = con.execute(
            "SELECT schema_name, path FROM ducklake_schema "
            "WHERE schema_name = 'myschema' AND end_snapshot IS NULL"
        ).fetchone()
        assert row is not None
        assert row[0] == "myschema"
        assert row[1] == "myschema/"
        con.close()

    def test_create_schema_metadata_correct(self, tmp_path):
        """Verify metadata: snapshot, schema_versions, changes."""
        metadata_path, data_path = _make_catalog(tmp_path)

        con = sqlite3.connect(metadata_path)
        sv_before = con.execute(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()[0]
        con.close()

        create_ducklake_schema(metadata_path, "myschema")

        con = sqlite3.connect(metadata_path)
        sv_after = con.execute(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()[0]
        assert sv_after == sv_before + 1

        # Schema version recorded
        sv_row = con.execute(
            "SELECT schema_version FROM ducklake_schema_versions "
            "ORDER BY begin_snapshot DESC LIMIT 1"
        ).fetchone()
        assert sv_row[0] == sv_after

        # Changes recorded
        change = con.execute(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()
        assert 'created_schema:"myschema"' in change[0]

        con.close()

    def test_create_schema_duplicate_raises(self, tmp_path):
        """Creating a schema that already exists raises."""
        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_schema(metadata_path, "myschema")

        with pytest.raises(ValueError, match="already exists"):
            create_ducklake_schema(metadata_path, "myschema")

    def test_create_schema_then_create_table(self, tmp_path):
        """Create a schema, then a table in it."""
        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_schema(metadata_path, "myschema")

        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, metadata_path, "test", schema="myschema", mode="error")

        result = read_ducklake(metadata_path, "test", schema="myschema").sort("a")
        assert result["a"].to_list() == [1, 2, 3]

    def test_create_schema_duckdb_interop(self, tmp_path):
        """DuckDB can see a schema created by polars."""
        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_schema(metadata_path, "myschema")

        # Create a table in the schema and write data
        df = pl.DataFrame({"a": [42]})
        write_ducklake(df, metadata_path, "test", schema="myschema", mode="error")

        pdf = _read_with_duckdb(metadata_path, data_path, "test", "myschema")
        assert pdf["a"].to_list() == [42]

    def test_duckdb_create_schema_polars_uses(self, tmp_path):
        """DuckDB creates a schema, polars creates a table in it."""
        metadata_path = str(tmp_path / "test.ducklake")
        data_path = str(tmp_path / "data")
        os.makedirs(data_path, exist_ok=True)

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:sqlite:{metadata_path}' AS ducklake "
            f"(DATA_PATH '{data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE SCHEMA ducklake.myschema")
        con.close()

        df = pl.DataFrame({"a": [10, 20]})
        write_ducklake(df, metadata_path, "test", schema="myschema", mode="error")

        result = read_ducklake(metadata_path, "test", schema="myschema").sort("a")
        assert result["a"].to_list() == [10, 20]

    def test_create_multiple_schemas(self, tmp_path):
        """Create multiple schemas."""
        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_schema(metadata_path, "s1")
        create_ducklake_schema(metadata_path, "s2")
        create_ducklake_schema(metadata_path, "s3")

        con = sqlite3.connect(metadata_path)
        count = con.execute(
            "SELECT COUNT(*) FROM ducklake_schema WHERE end_snapshot IS NULL"
        ).fetchone()[0]
        # main + s1 + s2 + s3
        assert count == 4

        names = [
            r[0]
            for r in con.execute(
                "SELECT schema_name FROM ducklake_schema "
                "WHERE end_snapshot IS NULL ORDER BY schema_id"
            ).fetchall()
        ]
        assert names == ["main", "s1", "s2", "s3"]
        con.close()


# ===========================================================================
# DROP SCHEMA
# ===========================================================================


class TestDropSchema:
    """Test DROP SCHEMA."""

    def test_drop_schema_empty(self, tmp_path):
        """Drop an empty schema."""
        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_schema(metadata_path, "myschema")

        drop_ducklake_schema(metadata_path, "myschema")

        con = sqlite3.connect(metadata_path)
        row = con.execute(
            "SELECT end_snapshot FROM ducklake_schema "
            "WHERE schema_name = 'myschema'"
        ).fetchone()
        assert row is not None
        assert row[0] is not None  # end_snapshot is set
        con.close()

    def test_drop_schema_metadata_correct(self, tmp_path):
        """Verify metadata: snapshot, schema_versions, changes."""
        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_schema(metadata_path, "myschema")

        con = sqlite3.connect(metadata_path)
        sv_before = con.execute(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()[0]
        con.close()

        drop_ducklake_schema(metadata_path, "myschema")

        con = sqlite3.connect(metadata_path)
        sv_after = con.execute(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()[0]
        assert sv_after == sv_before + 1

        change = con.execute(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()
        assert "dropped_schema" in change[0]
        con.close()

    def test_drop_schema_with_tables_no_cascade_raises(self, tmp_path):
        """Dropping a schema with tables without cascade raises."""
        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_schema(metadata_path, "myschema")

        df = pl.DataFrame({"a": [1]})
        write_ducklake(df, metadata_path, "test", schema="myschema", mode="error")

        with pytest.raises(ValueError, match="entries that depend on it"):
            drop_ducklake_schema(metadata_path, "myschema")

    def test_drop_schema_cascade(self, tmp_path):
        """Drop a schema with tables using cascade."""
        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_schema(metadata_path, "myschema")

        df = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df, metadata_path, "test", schema="myschema", mode="error")

        drop_ducklake_schema(metadata_path, "myschema", cascade=True)

        # Schema is gone
        con = sqlite3.connect(metadata_path)
        row = con.execute(
            "SELECT end_snapshot FROM ducklake_schema "
            "WHERE schema_name = 'myschema'"
        ).fetchone()
        assert row[0] is not None

        # Table is gone
        active_tables = con.execute(
            "SELECT COUNT(*) FROM ducklake_table WHERE end_snapshot IS NULL"
        ).fetchone()[0]
        assert active_tables == 0

        # Changes recorded
        change = con.execute(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()
        assert "dropped_schema" in change[0]
        assert "dropped_table" in change[0]
        con.close()

    def test_drop_schema_nonexistent_raises(self, tmp_path):
        """Dropping a nonexistent schema raises."""
        metadata_path, data_path = _make_catalog(tmp_path)

        with pytest.raises(ValueError, match="not found"):
            drop_ducklake_schema(metadata_path, "missing")

    def test_drop_schema_duckdb_interop(self, tmp_path):
        """DuckDB confirms dropped schema."""
        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_schema(metadata_path, "myschema")

        drop_ducklake_schema(metadata_path, "myschema")

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:sqlite:{metadata_path}' AS ducklake "
            f"(DATA_PATH '{data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        with pytest.raises((duckdb.CatalogException, duckdb.BinderException)):
            con.execute("CREATE TABLE ducklake.myschema.test (a INTEGER)")
        con.close()

    def test_duckdb_drop_schema_polars_confirms(self, tmp_path):
        """DuckDB drops a schema, polars can no longer use it."""
        metadata_path = str(tmp_path / "test.ducklake")
        data_path = str(tmp_path / "data")
        os.makedirs(data_path, exist_ok=True)

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:sqlite:{metadata_path}' AS ducklake "
            f"(DATA_PATH '{data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE SCHEMA ducklake.myschema")
        con.execute("DROP SCHEMA ducklake.myschema")
        con.close()

        # Can't create a table in a dropped schema
        with pytest.raises(ValueError, match="not found"):
            create_ducklake_table(
                metadata_path, "test",
                {"a": pl.Int64()},
                schema="myschema",
            )

    def test_drop_schema_cascade_duckdb_interop(self, tmp_path):
        """DuckDB can read table data from time travel after cascade drop."""
        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_schema(metadata_path, "myschema")
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, metadata_path, "test", schema="myschema", mode="error")

        con = sqlite3.connect(metadata_path)
        snap_before = con.execute(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        ).fetchone()[0]
        con.close()

        drop_ducklake_schema(metadata_path, "myschema", cascade=True)

        # Time travel works
        result = read_ducklake(
            metadata_path, "test",
            schema="myschema", snapshot_version=snap_before,
        ).sort("a")
        assert result["a"].to_list() == [1, 2, 3]


# ===========================================================================
# RENAME TABLE
# ===========================================================================


class TestRenameTable:
    """Test RENAME TABLE."""

    def test_rename_table_basic(self, tmp_path):
        """Rename a table and read back by new name."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, metadata_path, "t1", mode="error")

        rename_ducklake_table(metadata_path, "t1", "t2")

        result = read_ducklake(metadata_path, "t2").sort("a")
        assert result["a"].to_list() == [1, 2, 3]

        # Old name should be gone
        with pytest.raises(ValueError, match="not found"):
            read_ducklake(metadata_path, "t1")

    def test_rename_table_metadata_correct(self, tmp_path):
        """Verify metadata: old row ended, new row with same table_id."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1]})
        write_ducklake(df, metadata_path, "t1", mode="error")

        con = sqlite3.connect(metadata_path)
        sv_before = con.execute(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()[0]
        old_table = con.execute(
            "SELECT table_id, table_uuid, path FROM ducklake_table "
            "WHERE table_name = 't1' AND end_snapshot IS NULL"
        ).fetchone()
        old_table_id = old_table[0]
        old_table_uuid = old_table[1]
        old_path = old_table[2]
        con.close()

        rename_ducklake_table(metadata_path, "t1", "t2")

        con = sqlite3.connect(metadata_path)
        # Schema version incremented
        sv_after = con.execute(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()[0]
        assert sv_after == sv_before + 1

        # Old table row ended
        old_end = con.execute(
            "SELECT end_snapshot FROM ducklake_table "
            "WHERE table_name = 't1'"
        ).fetchone()
        assert old_end[0] is not None

        # New row has same table_id, table_uuid, and path
        new_table = con.execute(
            "SELECT table_id, table_uuid, path FROM ducklake_table "
            "WHERE table_name = 't2' AND end_snapshot IS NULL"
        ).fetchone()
        assert new_table[0] == old_table_id
        assert new_table[1] == old_table_uuid
        assert new_table[2] == old_path  # path doesn't change

        # Columns unchanged
        cols = con.execute(
            "SELECT COUNT(*) FROM ducklake_column "
            "WHERE table_id = ? AND end_snapshot IS NULL",
            [old_table_id],
        ).fetchone()[0]
        assert cols > 0

        # Data files unchanged
        files = con.execute(
            "SELECT COUNT(*) FROM ducklake_data_file "
            "WHERE table_id = ? AND end_snapshot IS NULL",
            [old_table_id],
        ).fetchone()[0]
        assert files > 0

        con.close()

    def test_rename_table_then_insert(self, tmp_path):
        """Insert data after renaming a table."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df, metadata_path, "t1", mode="error")

        rename_ducklake_table(metadata_path, "t1", "t2")

        new_row = pl.DataFrame({"a": [3]})
        write_ducklake(new_row, metadata_path, "t2", mode="append")

        result = read_ducklake(metadata_path, "t2").sort("a")
        assert result["a"].to_list() == [1, 2, 3]

    def test_rename_table_time_travel(self, tmp_path):
        """Time travel sees the old table name."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df, metadata_path, "t1", mode="error")

        con = sqlite3.connect(metadata_path)
        snap_before = con.execute(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        ).fetchone()[0]
        con.close()

        rename_ducklake_table(metadata_path, "t1", "t2")

        # Before rename: old name works
        result_old = read_ducklake(
            metadata_path, "t1", snapshot_version=snap_before
        ).sort("a")
        assert result_old["a"].to_list() == [1, 2]

    def test_rename_table_nonexistent_raises(self, tmp_path):
        """Renaming a nonexistent table raises."""
        metadata_path, data_path = _make_catalog(tmp_path)

        with pytest.raises(ValueError, match="not found"):
            rename_ducklake_table(metadata_path, "missing", "new")

    def test_rename_table_duplicate_raises(self, tmp_path):
        """Renaming to an existing table name raises."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1]})
        write_ducklake(df, metadata_path, "t1", mode="error")
        write_ducklake(df, metadata_path, "t2", mode="error")

        with pytest.raises(ValueError, match="already exists"):
            rename_ducklake_table(metadata_path, "t1", "t2")

    def test_rename_table_duckdb_interop(self, tmp_path):
        """DuckDB can read a table renamed by polars."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df, metadata_path, "t1", mode="error")

        rename_ducklake_table(metadata_path, "t1", "t2")

        pdf = _read_with_duckdb(metadata_path, data_path, "t2").sort("a")
        assert pdf["a"].to_list() == [1, 2]

        # Old name should fail
        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:sqlite:{metadata_path}' AS ducklake "
            f"(DATA_PATH '{data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        with pytest.raises(duckdb.CatalogException):
            con.execute("SELECT * FROM ducklake.t1")
        con.close()

    def test_duckdb_rename_table_polars_reads(self, tmp_path):
        """DuckDB renames a table, polars reads by new name."""
        metadata_path = str(tmp_path / "test.ducklake")
        data_path = str(tmp_path / "data")
        os.makedirs(data_path, exist_ok=True)

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:sqlite:{metadata_path}' AS ducklake "
            f"(DATA_PATH '{data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        con.execute("INSERT INTO ducklake.t1 VALUES (1), (2)")
        con.execute("ALTER TABLE ducklake.t1 RENAME TO t2")
        con.close()

        result = read_ducklake(metadata_path, "t2").sort("a")
        assert result["a"].to_list() == [1, 2]

        with pytest.raises(ValueError, match="not found"):
            read_ducklake(metadata_path, "t1")

    def test_rename_table_then_create_with_old_name(self, tmp_path):
        """Rename a table, then create a new table with the old name."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df1 = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df1, metadata_path, "t1", mode="error")

        rename_ducklake_table(metadata_path, "t1", "t2")

        df2 = pl.DataFrame({"x": ["hello"]})
        write_ducklake(df2, metadata_path, "t1", mode="error")

        # Both tables exist
        r1 = read_ducklake(metadata_path, "t1")
        assert r1.columns == ["x"]
        assert r1["x"].to_list() == ["hello"]

        r2 = read_ducklake(metadata_path, "t2").sort("a")
        assert r2["a"].to_list() == [1, 2]
