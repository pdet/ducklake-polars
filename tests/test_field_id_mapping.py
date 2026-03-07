"""Tests for field_id-based column mapping across schema changes.

Verifies that:
- Writer stamps PARQUET:field_id metadata on each column
- Reader uses field_id (not name/position) for column mapping
- Column renames are handled correctly via field_id
- Added columns get default values for old files
- Dropped columns in old files are ignored
- field_id is preserved through add_files
- DuckDB can read files with field_id metadata correctly
"""
from __future__ import annotations

import os

import polars as pl
import pyarrow.parquet as pq
import pytest

from ducklake_polars import add_files_ducklake, create_ducklake_table, read_ducklake, write_ducklake


class TestFieldIdInParquetMetadata:
    """Verify that the writer stamps PARQUET:field_id on Parquet columns."""

    def test_field_id_written_to_parquet(self, ducklake_catalog_sqlite):
        """Writer must set PARQUET:field_id in Parquet schema metadata."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'x'), (2, 'y')")
        cat.close()

        # Find the Parquet file
        data_dir = cat.data_path
        parquet_files = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".parquet"):
                    parquet_files.append(os.path.join(root, f))
        assert len(parquet_files) >= 1

        pf = pq.ParquetFile(parquet_files[0])
        schema = pf.schema_arrow
        field_ids = {}
        for field in schema:
            if field.metadata and b"PARQUET:field_id" in field.metadata:
                field_ids[field.name] = int(field.metadata[b"PARQUET:field_id"])

        # Both columns should have field_id metadata
        assert "a" in field_ids
        assert "b" in field_ids
        # field_ids should be distinct
        assert field_ids["a"] != field_ids["b"]

    def test_field_id_written_by_polars_writer(self, tmp_path):
        """write_ducklake must also stamp PARQUET:field_id."""
        metadata_path = str(tmp_path / "test.ducklake")
        data_path = str(tmp_path / "data")

        df = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        write_ducklake(df, metadata_path, "t", data_path=data_path)

        # Find parquet file
        parquet_files = []
        for root, _, files in os.walk(data_path):
            for f in files:
                if f.endswith(".parquet"):
                    parquet_files.append(os.path.join(root, f))
        assert len(parquet_files) >= 1

        pf = pq.ParquetFile(parquet_files[0])
        schema = pf.schema_arrow
        field_ids = {}
        for field in schema:
            if field.metadata and b"PARQUET:field_id" in field.metadata:
                field_ids[field.name] = int(field.metadata[b"PARQUET:field_id"])

        assert "x" in field_ids
        assert "y" in field_ids

    def test_field_id_consistent_across_inserts(self, ducklake_catalog_sqlite):
        """Same column should have same field_id across multiple inserts."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'x')")
        cat.execute("INSERT INTO ducklake.t VALUES (2, 'y')")
        cat.close()

        data_dir = cat.data_path
        parquet_files = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".parquet"):
                    parquet_files.append(os.path.join(root, f))
        assert len(parquet_files) >= 2

        # Read field_ids from both files
        all_fids = []
        for pf_path in parquet_files:
            pf = pq.ParquetFile(pf_path)
            schema = pf.schema_arrow
            fids = {}
            for field in schema:
                if field.metadata and b"PARQUET:field_id" in field.metadata:
                    fids[field.name] = int(field.metadata[b"PARQUET:field_id"])
            all_fids.append(fids)

        # field_ids should be the same across files for same column
        assert all_fids[0]["a"] == all_fids[1]["a"]
        assert all_fids[0]["b"] == all_fids[1]["b"]


class TestFieldIdColumnRename:
    """Test that field_id-based mapping handles column renames."""

    def test_write_rename_read(self, ducklake_catalog_sqlite):
        """Write, rename column, read -- field_id mapping works."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (old_name INTEGER, value VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'a'), (2, 'b')")
        cat.execute("ALTER TABLE ducklake.t RENAME COLUMN old_name TO new_name")
        cat.execute("INSERT INTO ducklake.t VALUES (3, 'c')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert "new_name" in result.columns
        assert "old_name" not in result.columns
        assert result.shape[0] == 3
        assert sorted(result["new_name"].to_list()) == [1, 2, 3]

    def test_rename_preserves_field_id_in_parquet(self, ducklake_catalog_sqlite):
        """After rename, the old file still has old name but same field_id."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (col_a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1)")

        # Get field_id before rename
        cat.close()
        data_dir = cat.data_path
        parquet_files = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".parquet"):
                    parquet_files.append(os.path.join(root, f))

        pf = pq.ParquetFile(parquet_files[0])
        old_fid = None
        for field in pf.schema_arrow:
            if field.name == "col_a" and field.metadata:
                old_fid = int(field.metadata[b"PARQUET:field_id"])
        assert old_fid is not None

        # Reopen, rename, insert new data
        cat._closed = False
        cat.__post_init__()
        cat.execute("ALTER TABLE ducklake.t RENAME COLUMN col_a TO col_b")
        cat.execute("INSERT INTO ducklake.t VALUES (2)")
        cat.close()

        # New file should have same field_id but new name
        new_parquet_files = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                fp = os.path.join(root, f)
                if f.endswith(".parquet") and fp not in parquet_files:
                    new_parquet_files.append(fp)

        assert len(new_parquet_files) >= 1
        pf2 = pq.ParquetFile(new_parquet_files[0])
        new_fid = None
        for field in pf2.schema_arrow:
            if field.name == "col_b" and field.metadata:
                new_fid = int(field.metadata[b"PARQUET:field_id"])
        assert new_fid is not None
        assert old_fid == new_fid  # Same field_id, different name

    def test_multiple_renames_then_read(self, ducklake_catalog_sqlite):
        """Column renamed twice -- all files readable with final name."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (x INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1)")
        cat.execute("ALTER TABLE ducklake.t RENAME COLUMN x TO y")
        cat.execute("INSERT INTO ducklake.t VALUES (2)")
        cat.execute("ALTER TABLE ducklake.t RENAME COLUMN y TO z")
        cat.execute("INSERT INTO ducklake.t VALUES (3)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert "z" in result.columns
        assert "x" not in result.columns
        assert "y" not in result.columns
        assert result.shape[0] == 3
        assert sorted(result["z"].to_list()) == [1, 2, 3]


class TestFieldIdAddColumn:
    """Test that new columns added after file was written get defaults."""

    def test_add_column_with_default_read_old_file(self, ducklake_catalog_sqlite):
        """Write, add column with default, read old file -- default applied."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1), (2)")
        cat.execute("ALTER TABLE ducklake.t ADD COLUMN b INTEGER DEFAULT 99")
        cat.execute("INSERT INTO ducklake.t VALUES (3, 30)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert "a" in result.columns
        assert "b" in result.columns
        assert result.shape[0] == 3

        sorted_result = result.sort("a")
        assert sorted_result["a"].to_list() == [1, 2, 3]
        # The new row should have its explicit value
        assert sorted_result.filter(pl.col("a") == 3)["b"].to_list() == [30]

    def test_add_column_no_default(self, ducklake_catalog_sqlite):
        """Write, add column without default, read -- old rows get NULL."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1)")
        cat.execute("ALTER TABLE ducklake.t ADD COLUMN b VARCHAR")
        cat.execute("INSERT INTO ducklake.t VALUES (2, 'hello')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.shape[0] == 2
        sorted_result = result.sort("a")
        assert sorted_result["b"].to_list()[0] is None
        assert sorted_result["b"].to_list()[1] == "hello"


class TestFieldIdDropColumn:
    """Test that dropped columns in old files are ignored."""

    def test_drop_column_read_old_file(self, ducklake_catalog_sqlite):
        """Write, drop column, read old file -- dropped column excluded."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER, b VARCHAR, c DOUBLE)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'x', 1.0), (2, 'y', 2.0)")
        cat.execute("ALTER TABLE ducklake.t DROP COLUMN b")
        cat.execute("INSERT INTO ducklake.t VALUES (3, 3.0)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert "a" in result.columns
        assert "c" in result.columns
        assert "b" not in result.columns
        assert result.shape[0] == 3
        assert sorted(result["a"].to_list()) == [1, 2, 3]

    def test_drop_and_add_same_name(self, ducklake_catalog_sqlite):
        """Drop column, add new column with same name -- field_id distinguishes them."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER, b INTEGER)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 10)")
        cat.execute("ALTER TABLE ducklake.t DROP COLUMN b")
        cat.execute("ALTER TABLE ducklake.t ADD COLUMN b VARCHAR")
        cat.execute("INSERT INTO ducklake.t VALUES (2, 'new')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.shape[0] == 2
        sorted_result = result.sort("a")
        # Old row's 'b' was an INTEGER that was dropped; new 'b' is VARCHAR
        # Old row should have NULL for the new 'b' column
        assert sorted_result["b"].to_list()[0] is None
        assert sorted_result["b"].to_list()[1] == "new"


class TestFieldIdMultipleSchemaChanges:
    """Test complex multi-step schema evolution scenarios."""

    def test_rename_add_drop_then_read_all(self, ducklake_catalog_sqlite):
        """Multiple schema changes then read all files."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER, b VARCHAR, c DOUBLE)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'x', 1.0)")

        # Rename b -> b_renamed
        cat.execute("ALTER TABLE ducklake.t RENAME COLUMN b TO b_renamed")
        cat.execute("INSERT INTO ducklake.t VALUES (2, 'y', 2.0)")

        # Add column d
        cat.execute("ALTER TABLE ducklake.t ADD COLUMN d INTEGER")
        cat.execute("INSERT INTO ducklake.t VALUES (3, 'z', 3.0, 30)")

        # Drop column c
        cat.execute("ALTER TABLE ducklake.t DROP COLUMN c")
        cat.execute("INSERT INTO ducklake.t VALUES (4, 'w', 40)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert "a" in result.columns
        assert "b_renamed" in result.columns
        assert "d" in result.columns
        assert "b" not in result.columns
        assert "c" not in result.columns
        assert result.shape[0] == 4

        sorted_result = result.sort("a")
        assert sorted_result["a"].to_list() == [1, 2, 3, 4]
        assert sorted_result["b_renamed"].to_list() == ["x", "y", "z", "w"]

    def test_many_inserts_with_renames_between(self, ducklake_catalog_sqlite):
        """Multiple files written with renames between -- all map correctly."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (v1 INTEGER, data VARCHAR)")
        for i in range(3):
            cat.execute(f"INSERT INTO ducklake.t VALUES ({i}, 'batch_{i}')")

        cat.execute("ALTER TABLE ducklake.t RENAME COLUMN v1 TO v2")
        for i in range(3, 6):
            cat.execute(f"INSERT INTO ducklake.t VALUES ({i}, 'batch_{i}')")

        cat.execute("ALTER TABLE ducklake.t RENAME COLUMN v2 TO v3")
        for i in range(6, 9):
            cat.execute(f"INSERT INTO ducklake.t VALUES ({i}, 'batch_{i}')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert "v3" in result.columns
        assert result.shape[0] == 9
        assert sorted(result["v3"].to_list()) == list(range(9))


class TestFieldIdAddFiles:
    """Test that field_id is preserved through add_files."""

    def test_add_files_preserves_field_id_mapping(self, tmp_path):
        """Add external Parquet files -- field_id mapping works for reads."""
        import pyarrow as pa

        metadata_path = str(tmp_path / "test.ducklake")
        data_path = str(tmp_path / "data")

        # Write initial data
        df = pl.DataFrame({"a": [1], "b": ["x"]})
        write_ducklake(df, metadata_path, "t", data_path=data_path)

        # Create an external Parquet file with matching schema
        external_path = str(tmp_path / "external.parquet")
        ext_table = pa.table({"a": [2, 3], "b": ["y", "z"]})
        pq.write_table(ext_table, external_path)

        add_files_ducklake(metadata_path, "t", [external_path], data_path=data_path)

        result = read_ducklake(metadata_path, "t", data_path=data_path)
        assert result.shape[0] == 3
        assert sorted(result["a"].to_list()) == [1, 2, 3]

    def test_add_files_then_rename_then_read(self, ducklake_catalog_sqlite, tmp_path):
        """Add files, rename column, read -- field_id mapping resolves correctly."""
        import pyarrow as pa

        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (col_a INTEGER, col_b VARCHAR)")

        # Create and add external file
        external_path = str(tmp_path / "external.parquet")
        ext_table = pa.table(
            {"col_a": pa.array([1, 2], type=pa.int32()), "col_b": ["x", "y"]}
        )
        pq.write_table(ext_table, external_path)

        add_files_ducklake(
            cat.metadata_path, "t", [external_path], data_path=cat.data_path,
        )

        # Rename via DuckDB
        cat.execute("ALTER TABLE ducklake.t RENAME COLUMN col_a TO col_renamed")
        cat.execute("INSERT INTO ducklake.t VALUES (3, 'z')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert "col_renamed" in result.columns
        assert "col_a" not in result.columns
        assert result.shape[0] == 3
        assert sorted(result["col_renamed"].to_list()) == [1, 2, 3]


class TestFieldIdDuckDBInterop:
    """Test that DuckDB reads files with field_id metadata correctly."""

    def test_duckdb_reads_field_id_files(self, ducklake_catalog_sqlite):
        """DuckDB reads files with PARQUET:field_id metadata correctly."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (x INTEGER, y VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'a'), (2, 'b')")
        cat.execute("ALTER TABLE ducklake.t RENAME COLUMN x TO z")
        cat.execute("INSERT INTO ducklake.t VALUES (3, 'c')")

        # DuckDB should see data correctly
        result = cat.execute("SELECT * FROM ducklake.t ORDER BY z").fetchall()
        assert result == [(1, "a"), (2, "b"), (3, "c")]
        cat.close()

        # Polars should also see data correctly
        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert "z" in result.columns
        assert result.shape[0] == 3

    def test_duckdb_reads_files_after_add_drop(self, ducklake_catalog_sqlite):
        """DuckDB handles add/drop column with field_id metadata."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'x')")
        cat.execute("ALTER TABLE ducklake.t ADD COLUMN c DOUBLE")
        cat.execute("INSERT INTO ducklake.t VALUES (2, 'y', 2.0)")

        duckdb_result = cat.execute("SELECT * FROM ducklake.t ORDER BY a").fetchall()
        assert len(duckdb_result) == 2
        assert duckdb_result[0][0] == 1
        assert duckdb_result[1][0] == 2
        assert duckdb_result[1][2] == 2.0
        cat.close()

    def test_polars_writer_field_id_duckdb_readable(self, tmp_path):
        """Files written by ducklake-polars with field_id are DuckDB-readable."""
        import duckdb

        metadata_path = str(tmp_path / "test.ducklake")
        data_path = str(tmp_path / "data")

        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        write_ducklake(df, metadata_path, "t", data_path=data_path)

        # Read the Parquet file directly with DuckDB
        parquet_files = []
        for root, _, files in os.walk(data_path):
            for f in files:
                if f.endswith(".parquet"):
                    parquet_files.append(os.path.join(root, f))
        assert len(parquet_files) >= 1

        con = duckdb.connect()
        result = con.execute(
            f"SELECT * FROM '{parquet_files[0]}' ORDER BY a"
        ).fetchall()
        assert result == [(1, "x"), (2, "y")]
        con.close()


class TestFieldIdPolarsWriterSchemaEvolution:
    """Test field_id with the Polars writer's schema evolution."""

    def test_polars_writer_append_preserves_field_ids(self, tmp_path):
        """Multiple appends via write_ducklake preserve field_ids."""
        metadata_path = str(tmp_path / "test.ducklake")
        data_path = str(tmp_path / "data")

        df1 = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        write_ducklake(df1, metadata_path, "t", data_path=data_path)

        df2 = pl.DataFrame({"a": [3, 4], "b": ["z", "w"]})
        write_ducklake(df2, metadata_path, "t", mode="append", data_path=data_path)

        # Read all parquet files and verify field_ids match
        parquet_files = []
        for root, _, files in os.walk(data_path):
            for f in files:
                if f.endswith(".parquet"):
                    parquet_files.append(os.path.join(root, f))
        assert len(parquet_files) >= 2

        all_fids = []
        for pf_path in parquet_files:
            pf = pq.ParquetFile(pf_path)
            fids = {}
            for field in pf.schema_arrow:
                if field.metadata and b"PARQUET:field_id" in field.metadata:
                    fids[field.name] = int(field.metadata[b"PARQUET:field_id"])
            all_fids.append(fids)

        for fids in all_fids:
            assert "a" in fids
            assert "b" in fids

        # All files should use the same field_ids
        for fids in all_fids[1:]:
            assert fids["a"] == all_fids[0]["a"]
            assert fids["b"] == all_fids[0]["b"]

        # Data should be complete
        result = read_ducklake(metadata_path, "t", data_path=data_path)
        assert result.shape[0] == 4
        assert sorted(result["a"].to_list()) == [1, 2, 3, 4]
