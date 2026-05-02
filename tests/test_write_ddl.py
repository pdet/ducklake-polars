"""Tests for ducklake-dataframe DDL operations (rename column, drop table,
create/drop schema, rename table)."""

from __future__ import annotations

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


# ===========================================================================
# RENAME COLUMN
# ===========================================================================


class TestRenameColumn:
    """Test ALTER TABLE RENAME COLUMN."""

    def test_rename_column_basic(self, make_write_catalog):
        """Rename a column and read back."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_rename_column(cat.metadata_path, "test", "b", "c")

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result.columns == ["a", "c"]
        assert result["a"].to_list() == [1, 2, 3]
        assert result["c"].to_list() == ["x", "y", "z"]

    def test_rename_column_metadata_correct(self, make_write_catalog):
        """Verify metadata: old row ended, new row inserted with same column_id."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1], "b": ["hello"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        sv_before = cat.query_one(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )[0]
        # Get column_id of 'b'
        col_b_row = cat.query_one(
            "SELECT column_id FROM ducklake_column "
            "WHERE column_name = 'b' AND end_snapshot IS NULL"
        )
        col_b_id = col_b_row[0]

        alter_ducklake_rename_column(cat.metadata_path, "test", "b", "c")

        # Schema version incremented
        sv_after = cat.query_one(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )[0]
        assert sv_after == sv_before + 1

        # Old column row has end_snapshot set
        old_col = cat.query_one(
            "SELECT end_snapshot FROM ducklake_column "
            "WHERE column_name = 'b'"
        )
        assert old_col is not None
        assert old_col[0] is not None

        # New column row uses the same column_id
        new_col = cat.query_one(
            "SELECT column_id, column_name FROM ducklake_column "
            "WHERE column_name = 'c' AND end_snapshot IS NULL"
        )
        assert new_col is not None
        assert new_col[0] == col_b_id  # same column_id

        # Changes recorded
        change = cat.query_one(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )
        assert "altered_table" in change[0]

    def test_rename_column_then_insert(self, make_write_catalog):
        """Insert data after renaming a column."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_rename_column(cat.metadata_path, "test", "b", "c")

        new_row = pl.DataFrame({"a": [3], "c": ["z"]})
        write_ducklake(new_row, cat.metadata_path, "test", mode="append")

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result.columns == ["a", "c"]
        assert result["a"].to_list() == [1, 2, 3]
        assert result["c"].to_list() == ["x", "y", "z"]

    def test_rename_column_time_travel(self, make_write_catalog):
        """Time travel sees the old column name."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        snap_before = cat.query_one(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        )[0]

        alter_ducklake_rename_column(cat.metadata_path, "test", "b", "c")

        # Latest: column is 'c'
        result = read_ducklake(cat.metadata_path, "test")
        assert "c" in result.columns
        assert "b" not in result.columns

        # Before rename: column is 'b'
        result_old = read_ducklake(
            cat.metadata_path, "test", snapshot_version=snap_before
        )
        assert "b" in result_old.columns
        assert "c" not in result_old.columns

    def test_rename_column_nonexistent_raises(self, make_write_catalog):
        """Renaming a column that doesn't exist raises."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        with pytest.raises(ValueError, match="not found"):
            alter_ducklake_rename_column(
                cat.metadata_path, "test", "missing", "new"
            )

    def test_rename_column_duplicate_raises(self, make_write_catalog):
        """Renaming to an existing column name raises."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1], "b": [2]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        with pytest.raises(ValueError, match="already exists"):
            alter_ducklake_rename_column(cat.metadata_path, "test", "b", "a")

    def test_rename_column_nonexistent_table_raises(self, make_write_catalog):
        """Renaming on a nonexistent table raises."""
        cat = make_write_catalog()

        with pytest.raises(ValueError, match="not found"):
            alter_ducklake_rename_column(
                cat.metadata_path, "missing", "a", "b"
            )

    def test_rename_column_duckdb_interop(self, make_write_catalog):
        """DuckDB can read after polars renames a column."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_rename_column(cat.metadata_path, "test", "b", "c")

        pdf = cat.read_with_duckdb("test").sort("a")
        assert "c" in pdf.columns
        assert "b" not in pdf.columns
        assert pdf["a"].to_list() == [1, 2]
        assert pdf["c"].to_list() == ["x", "y"]

    def test_duckdb_rename_column_polars_reads(self, make_write_catalog):
        """DuckDB renames a column, polars reads correctly."""
        cat = make_write_catalog()

        source = cat.attach_source()

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH '{source}' AS ducklake "
            f"(DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        con.execute("INSERT INTO ducklake.test VALUES (1, 'hello'), (2, 'world')")
        con.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO c")
        con.close()

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result.columns == ["a", "c"]
        assert result["a"].to_list() == [1, 2]
        assert result["c"].to_list() == ["hello", "world"]

    def test_rename_column_multiple(self, make_write_catalog):
        """Rename multiple columns sequentially."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        alter_ducklake_rename_column(cat.metadata_path, "test", "b", "bb")
        alter_ducklake_rename_column(cat.metadata_path, "test", "c", "cc")

        result = read_ducklake(cat.metadata_path, "test")
        assert result.columns == ["a", "bb", "cc"]
        assert result["a"].to_list() == [1]
        assert result["bb"].to_list() == [2]
        assert result["cc"].to_list() == [3]


# ===========================================================================
# DROP TABLE
# ===========================================================================


class TestDropTable:
    """Test DROP TABLE."""

    def test_drop_table_basic(self, make_write_catalog):
        """Drop a table and verify it's no longer readable."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        drop_ducklake_table(cat.metadata_path, "test")

        with pytest.raises(ValueError, match="not found"):
            read_ducklake(cat.metadata_path, "test")

    def test_drop_table_metadata_correct(self, make_write_catalog):
        """Verify metadata: end_snapshot on table, columns, data files."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1], "b": ["hello"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        sv_before = cat.query_one(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )[0]

        drop_ducklake_table(cat.metadata_path, "test")

        # Schema version incremented
        sv_after = cat.query_one(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )[0]
        assert sv_after == sv_before + 1

        # Table has end_snapshot set
        table_row = cat.query_one(
            "SELECT end_snapshot FROM ducklake_table "
            "WHERE table_name = 'test'"
        )
        assert table_row[0] is not None

        # All columns have end_snapshot set
        active_cols = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_column WHERE end_snapshot IS NULL"
        )[0]
        assert active_cols == 0

        # All data files have end_snapshot set
        active_files = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL"
        )[0]
        assert active_files == 0

        # Changes recorded
        change = cat.query_one(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )
        assert "dropped_table" in change[0]

    def test_drop_table_with_delete_files(self, make_write_catalog):
        """Drop a table that has delete files."""
        from ducklake_polars import delete_ducklake

        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")
        delete_ducklake(cat.metadata_path, "test", pl.col("a") == 1)

        drop_ducklake_table(cat.metadata_path, "test")

        # Delete files should also be ended
        active_deletes = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_delete_file WHERE end_snapshot IS NULL"
        )[0]
        assert active_deletes == 0

    def test_drop_table_time_travel(self, make_write_catalog):
        """Time travel can read the table before it was dropped."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        snap_before = cat.query_one(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        )[0]

        drop_ducklake_table(cat.metadata_path, "test")

        # Latest: table is gone
        with pytest.raises(ValueError, match="not found"):
            read_ducklake(cat.metadata_path, "test")

        # Before drop: table exists
        result = read_ducklake(
            cat.metadata_path, "test", snapshot_version=snap_before
        )
        assert result["a"].to_list() == [1, 2, 3]

    def test_drop_table_nonexistent_raises(self, make_write_catalog):
        """Dropping a nonexistent table raises."""
        cat = make_write_catalog()

        with pytest.raises(ValueError, match="not found"):
            drop_ducklake_table(cat.metadata_path, "missing")

    def test_drop_table_duckdb_interop(self, make_write_catalog):
        """DuckDB confirms table is dropped."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        drop_ducklake_table(cat.metadata_path, "test")

        source = cat.attach_source()

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH '{source}' AS ducklake "
            f"(DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        with pytest.raises(duckdb.CatalogException):
            con.execute("SELECT * FROM ducklake.test")
        con.close()

    def test_duckdb_drop_table_polars_reads(self, make_write_catalog):
        """DuckDB drops a table, polars confirms it's gone."""
        cat = make_write_catalog()

        source = cat.attach_source()

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH '{source}' AS ducklake "
            f"(DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE TABLE ducklake.test (a INTEGER)")
        con.execute("INSERT INTO ducklake.test VALUES (1)")
        con.execute("DROP TABLE ducklake.test")
        con.close()

        with pytest.raises(ValueError, match="not found"):
            read_ducklake(cat.metadata_path, "test")

    def test_drop_and_recreate_table(self, make_write_catalog):
        """Drop a table and create a new one with the same name."""
        cat = make_write_catalog()
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df1, cat.metadata_path, "test", mode="error")

        drop_ducklake_table(cat.metadata_path, "test")

        df2 = pl.DataFrame({"x": ["hello", "world"]})
        write_ducklake(df2, cat.metadata_path, "test", mode="error")

        result = read_ducklake(cat.metadata_path, "test").sort("x")
        assert result.columns == ["x"]
        assert result["x"].to_list() == ["hello", "world"]


# ===========================================================================
# CREATE SCHEMA
# ===========================================================================


class TestCreateSchema:
    """Test CREATE SCHEMA."""

    def test_create_schema_basic(self, make_write_catalog):
        """Create a new schema."""
        cat = make_write_catalog()

        create_ducklake_schema(cat.metadata_path, "myschema")

        row = cat.query_one(
            "SELECT schema_name, path FROM ducklake_schema "
            "WHERE schema_name = 'myschema' AND end_snapshot IS NULL"
        )
        assert row is not None
        assert row[0] == "myschema"
        assert row[1] == "myschema/"

    def test_create_schema_metadata_correct(self, make_write_catalog):
        """Verify metadata: snapshot, schema_versions, changes."""
        cat = make_write_catalog()

        sv_before = cat.query_one(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )[0]

        create_ducklake_schema(cat.metadata_path, "myschema")

        sv_after = cat.query_one(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )[0]
        assert sv_after == sv_before + 1

        # Schema version recorded
        sv_row = cat.query_one(
            "SELECT schema_version FROM ducklake_schema_versions "
            "ORDER BY begin_snapshot DESC LIMIT 1"
        )
        assert sv_row[0] == sv_after

        # Changes recorded
        change = cat.query_one(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )
        assert 'created_schema:"myschema"' in change[0]

    def test_create_schema_duplicate_raises(self, make_write_catalog):
        """Creating a schema that already exists raises."""
        cat = make_write_catalog()
        create_ducklake_schema(cat.metadata_path, "myschema")

        with pytest.raises(ValueError, match="already exists"):
            create_ducklake_schema(cat.metadata_path, "myschema")

    def test_create_schema_then_create_table(self, make_write_catalog):
        """Create a schema, then a table in it."""
        cat = make_write_catalog()
        create_ducklake_schema(cat.metadata_path, "myschema")

        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "test", schema="myschema", mode="error")

        result = read_ducklake(cat.metadata_path, "test", schema="myschema").sort("a")
        assert result["a"].to_list() == [1, 2, 3]

    def test_create_schema_duckdb_interop(self, make_write_catalog):
        """DuckDB can see a schema created by polars."""
        cat = make_write_catalog()
        create_ducklake_schema(cat.metadata_path, "myschema")

        # Create a table in the schema and write data
        df = pl.DataFrame({"a": [42]})
        write_ducklake(df, cat.metadata_path, "test", schema="myschema", mode="error")

        pdf = cat.read_with_duckdb_schema("test", "myschema")
        assert pdf["a"].to_list() == [42]

    def test_duckdb_create_schema_polars_uses(self, make_write_catalog):
        """DuckDB creates a schema, polars creates a table in it."""
        cat = make_write_catalog()

        source = cat.attach_source()

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH '{source}' AS ducklake "
            f"(DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE SCHEMA ducklake.myschema")
        con.close()

        df = pl.DataFrame({"a": [10, 20]})
        write_ducklake(df, cat.metadata_path, "test", schema="myschema", mode="error")

        result = read_ducklake(cat.metadata_path, "test", schema="myschema").sort("a")
        assert result["a"].to_list() == [10, 20]

    def test_create_multiple_schemas(self, make_write_catalog):
        """Create multiple schemas."""
        cat = make_write_catalog()
        create_ducklake_schema(cat.metadata_path, "s1")
        create_ducklake_schema(cat.metadata_path, "s2")
        create_ducklake_schema(cat.metadata_path, "s3")

        count = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_schema WHERE end_snapshot IS NULL"
        )[0]
        # main + s1 + s2 + s3
        assert count == 4

        rows = cat.query_all(
            "SELECT schema_name FROM ducklake_schema "
            "WHERE end_snapshot IS NULL ORDER BY schema_id"
        )
        names = [r[0] for r in rows]
        assert names == ["main", "s1", "s2", "s3"]


# ===========================================================================
# DROP SCHEMA
# ===========================================================================


class TestDropSchema:
    """Test DROP SCHEMA."""

    def test_drop_schema_empty(self, make_write_catalog):
        """Drop an empty schema."""
        cat = make_write_catalog()
        create_ducklake_schema(cat.metadata_path, "myschema")

        drop_ducklake_schema(cat.metadata_path, "myschema")

        row = cat.query_one(
            "SELECT end_snapshot FROM ducklake_schema "
            "WHERE schema_name = 'myschema'"
        )
        assert row is not None
        assert row[0] is not None  # end_snapshot is set

    def test_drop_schema_metadata_correct(self, make_write_catalog):
        """Verify metadata: snapshot, schema_versions, changes."""
        cat = make_write_catalog()
        create_ducklake_schema(cat.metadata_path, "myschema")

        sv_before = cat.query_one(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )[0]

        drop_ducklake_schema(cat.metadata_path, "myschema")

        sv_after = cat.query_one(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )[0]
        assert sv_after == sv_before + 1

        change = cat.query_one(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )
        assert "dropped_schema" in change[0]

    def test_drop_schema_with_tables_no_cascade_raises(self, make_write_catalog):
        """Dropping a schema with tables without cascade raises."""
        cat = make_write_catalog()
        create_ducklake_schema(cat.metadata_path, "myschema")

        df = pl.DataFrame({"a": [1]})
        write_ducklake(df, cat.metadata_path, "test", schema="myschema", mode="error")

        with pytest.raises(ValueError, match="entries that depend on it"):
            drop_ducklake_schema(cat.metadata_path, "myschema")

    def test_drop_schema_cascade(self, make_write_catalog):
        """Drop a schema with tables using cascade."""
        cat = make_write_catalog()
        create_ducklake_schema(cat.metadata_path, "myschema")

        df = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df, cat.metadata_path, "test", schema="myschema", mode="error")

        drop_ducklake_schema(cat.metadata_path, "myschema", cascade=True)

        # Schema is gone
        row = cat.query_one(
            "SELECT end_snapshot FROM ducklake_schema "
            "WHERE schema_name = 'myschema'"
        )
        assert row[0] is not None

        # Table is gone
        active_tables = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_table WHERE end_snapshot IS NULL"
        )[0]
        assert active_tables == 0

        # Changes recorded
        change = cat.query_one(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )
        assert "dropped_schema" in change[0]
        assert "dropped_table" in change[0]

    def test_drop_schema_nonexistent_raises(self, make_write_catalog):
        """Dropping a nonexistent schema raises."""
        cat = make_write_catalog()

        with pytest.raises(ValueError, match="not found"):
            drop_ducklake_schema(cat.metadata_path, "missing")

    def test_drop_schema_duckdb_interop(self, make_write_catalog):
        """DuckDB confirms dropped schema."""
        cat = make_write_catalog()
        create_ducklake_schema(cat.metadata_path, "myschema")

        drop_ducklake_schema(cat.metadata_path, "myschema")

        source = cat.attach_source()

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH '{source}' AS ducklake "
            f"(DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        with pytest.raises((duckdb.CatalogException, duckdb.BinderException)):
            con.execute("CREATE TABLE ducklake.myschema.test (a INTEGER)")
        con.close()

    def test_duckdb_drop_schema_polars_confirms(self, make_write_catalog):
        """DuckDB drops a schema, polars can no longer use it."""
        cat = make_write_catalog()

        source = cat.attach_source()

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH '{source}' AS ducklake "
            f"(DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE SCHEMA ducklake.myschema")
        con.execute("DROP SCHEMA ducklake.myschema")
        con.close()

        # Can't create a table in a dropped schema
        with pytest.raises(ValueError, match="not found"):
            create_ducklake_table(
                cat.metadata_path, "test",
                {"a": pl.Int64()},
                schema="myschema",
            )

    def test_drop_schema_cascade_duckdb_interop(self, make_write_catalog):
        """DuckDB can read table data from time travel after cascade drop."""
        cat = make_write_catalog()
        create_ducklake_schema(cat.metadata_path, "myschema")
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "test", schema="myschema", mode="error")

        snap_before = cat.query_one(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        )[0]

        drop_ducklake_schema(cat.metadata_path, "myschema", cascade=True)

        # Time travel works
        result = read_ducklake(
            cat.metadata_path, "test",
            schema="myschema", snapshot_version=snap_before,
        ).sort("a")
        assert result["a"].to_list() == [1, 2, 3]


# ===========================================================================
# RENAME TABLE
# ===========================================================================


class TestRenameTable:
    """Test RENAME TABLE."""

    def test_rename_table_basic(self, make_write_catalog):
        """Rename a table and read back by new name."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "t1", mode="error")

        rename_ducklake_table(cat.metadata_path, "t1", "t2")

        result = read_ducklake(cat.metadata_path, "t2").sort("a")
        assert result["a"].to_list() == [1, 2, 3]

        # Old name should be gone
        with pytest.raises(ValueError, match="not found"):
            read_ducklake(cat.metadata_path, "t1")

    def test_rename_table_metadata_correct(self, make_write_catalog):
        """Verify metadata: old row ended, new row with same table_id."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1]})
        write_ducklake(df, cat.metadata_path, "t1", mode="error")

        sv_before = cat.query_one(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )[0]
        old_table = cat.query_one(
            "SELECT table_id, table_uuid, path FROM ducklake_table "
            "WHERE table_name = 't1' AND end_snapshot IS NULL"
        )
        old_table_id = old_table[0]
        old_table_uuid = old_table[1]
        old_path = old_table[2]

        rename_ducklake_table(cat.metadata_path, "t1", "t2")

        # Schema version incremented
        sv_after = cat.query_one(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )[0]
        assert sv_after == sv_before + 1

        # Old table row ended
        old_end = cat.query_one(
            "SELECT end_snapshot FROM ducklake_table "
            "WHERE table_name = 't1'"
        )
        assert old_end[0] is not None

        # New row has same table_id, table_uuid, and path
        new_table = cat.query_one(
            "SELECT table_id, table_uuid, path FROM ducklake_table "
            "WHERE table_name = 't2' AND end_snapshot IS NULL"
        )
        assert new_table[0] == old_table_id
        assert new_table[1] == old_table_uuid
        assert new_table[2] == old_path  # path doesn't change

        # Columns unchanged
        cols = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_column "
            "WHERE table_id = ? AND end_snapshot IS NULL",
            [old_table_id],
        )[0]
        assert cols > 0

        # Data files unchanged
        files = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_data_file "
            "WHERE table_id = ? AND end_snapshot IS NULL",
            [old_table_id],
        )[0]
        assert files > 0

    def test_rename_table_then_insert(self, make_write_catalog):
        """Insert data after renaming a table."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df, cat.metadata_path, "t1", mode="error")

        rename_ducklake_table(cat.metadata_path, "t1", "t2")

        new_row = pl.DataFrame({"a": [3]})
        write_ducklake(new_row, cat.metadata_path, "t2", mode="append")

        result = read_ducklake(cat.metadata_path, "t2").sort("a")
        assert result["a"].to_list() == [1, 2, 3]

    def test_rename_table_time_travel(self, make_write_catalog):
        """Time travel sees the old table name."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df, cat.metadata_path, "t1", mode="error")

        snap_before = cat.query_one(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        )[0]

        rename_ducklake_table(cat.metadata_path, "t1", "t2")

        # Before rename: old name works
        result_old = read_ducklake(
            cat.metadata_path, "t1", snapshot_version=snap_before
        ).sort("a")
        assert result_old["a"].to_list() == [1, 2]

    def test_rename_table_nonexistent_raises(self, make_write_catalog):
        """Renaming a nonexistent table raises."""
        cat = make_write_catalog()

        with pytest.raises(ValueError, match="not found"):
            rename_ducklake_table(cat.metadata_path, "missing", "new")

    def test_rename_table_duplicate_raises(self, make_write_catalog):
        """Renaming to an existing table name raises."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1]})
        write_ducklake(df, cat.metadata_path, "t1", mode="error")
        write_ducklake(df, cat.metadata_path, "t2", mode="error")

        with pytest.raises(ValueError, match="already exists"):
            rename_ducklake_table(cat.metadata_path, "t1", "t2")

    def test_rename_table_duckdb_interop(self, make_write_catalog):
        """DuckDB can read a table renamed by polars."""
        cat = make_write_catalog()
        df = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df, cat.metadata_path, "t1", mode="error")

        rename_ducklake_table(cat.metadata_path, "t1", "t2")

        pdf = cat.read_with_duckdb("t2").sort("a")
        assert pdf["a"].to_list() == [1, 2]

        # Old name should fail
        source = cat.attach_source()

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH '{source}' AS ducklake "
            f"(DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        with pytest.raises(duckdb.CatalogException):
            con.execute("SELECT * FROM ducklake.t1")
        con.close()

    def test_duckdb_rename_table_polars_reads(self, make_write_catalog):
        """DuckDB renames a table, polars reads by new name."""
        cat = make_write_catalog()

        source = cat.attach_source()

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH '{source}' AS ducklake "
            f"(DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        con.execute("INSERT INTO ducklake.t1 VALUES (1), (2)")
        con.execute("ALTER TABLE ducklake.t1 RENAME TO t2")
        con.close()

        result = read_ducklake(cat.metadata_path, "t2").sort("a")
        assert result["a"].to_list() == [1, 2]

        with pytest.raises(ValueError, match="not found"):
            read_ducklake(cat.metadata_path, "t1")

    def test_rename_table_then_create_with_old_name(self, make_write_catalog):
        """Rename a table, then create a new table with the old name."""
        cat = make_write_catalog()
        df1 = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df1, cat.metadata_path, "t1", mode="error")

        rename_ducklake_table(cat.metadata_path, "t1", "t2")

        df2 = pl.DataFrame({"x": ["hello"]})
        write_ducklake(df2, cat.metadata_path, "t1", mode="error")

        # Both tables exist
        r1 = read_ducklake(cat.metadata_path, "t1")
        assert r1.columns == ["x"]
        assert r1["x"].to_list() == ["hello"]

        r2 = read_ducklake(cat.metadata_path, "t2").sort("a")
        assert r2["a"].to_list() == [1, 2]
