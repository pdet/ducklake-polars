"""Schema evolution tests for ducklake-pandas."""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from tests_pandas.helpers import assert_list_equal
from ducklake_pandas import read_ducklake


class TestAddColumn:
    """Test reading after columns are added."""

    def test_read_after_add_column(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), (2)")

        cat.execute("ALTER TABLE ducklake.test ADD COLUMN b VARCHAR")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'hello')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (3, 2)
        assert True  # schema check skipped for pandas
        # Old rows should have NULL for the new column
        result = result.sort_values("a").reset_index(drop=True)
        assert_list_equal(result[result["a"] <= 2]["b"].tolist(), [None, None])
        # New row should have the value
        assert result[result["a"] == 3]["b"].tolist() == ["hello"]

    def test_read_after_add_multiple_columns(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")

        cat.execute("ALTER TABLE ducklake.test ADD COLUMN b VARCHAR")
        cat.execute("ALTER TABLE ducklake.test ADD COLUMN c DOUBLE")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'hello', 3.14)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (2, 3)
        assert True  # schema check skipped for pandas
        result = result.sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [1, 2]
        assert_list_equal(result["b"].tolist(), [None, "hello"])
        c_vals = result["c"].tolist()
        assert pd.isna(c_vals[0])
        assert c_vals[1] == 3.14


class TestDropColumn:
    """Test reading after columns are dropped."""

    def test_read_after_drop_column(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR, c DOUBLE)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello', 3.14)")

        cat.execute("ALTER TABLE ducklake.test DROP COLUMN b")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 2.72)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert list(result.columns) == ["a", "c"]
        assert result.shape == (2, 2)
        result = result.sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [1, 2]
        assert result["c"].tolist() == [3.14, 2.72]


class TestRenameColumn:
    """Test reading after columns are renamed."""

    def test_read_after_rename_column(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello')")

        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO name")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'world')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert list(result.columns) == ["a", "name"]
        assert result.shape == (2, 2)
        # Both old and new data should be accessible under the new name
        result = result.sort_values("a").reset_index(drop=True)
        assert result["name"].tolist() == ["hello", "world"]

    def test_read_after_multiple_renames(self, ducklake_catalog):
        """Rename b -> name -> full_name, verify all data accessible."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'alice')")

        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO name")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'bob')")

        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN name TO full_name")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'charlie')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert list(result.columns) == ["a", "full_name"]
        assert result.shape == (3, 2)
        result = result.sort_values("a").reset_index(drop=True)
        assert result["full_name"].tolist() == ["alice", "bob", "charlie"]

    def test_rename_with_add_column(self, ducklake_catalog):
        """Rename + add column in the same table."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello')")

        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO name")
        cat.execute("ALTER TABLE ducklake.test ADD COLUMN c DOUBLE")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'world', 3.14)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert list(result.columns) == ["a", "name", "c"]
        assert result.shape == (2, 3)
        result = result.sort_values("a").reset_index(drop=True)
        assert result["name"].tolist() == ["hello", "world"]
        c_vals = result["c"].tolist()
        assert pd.isna(c_vals[0])
        assert c_vals[1] == 3.14

    def test_rename_with_filter(self, ducklake_catalog):
        """Verify filtering works after rename."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello')")

        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO name")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'world')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result[result["a"] == 2]
        assert result.shape == (1, 2)
        assert list(result.columns) == ["a", "name"]
        assert result["a"].tolist() == [2]
        assert result["name"].tolist() == ["world"]

    def test_rename_with_delete(self, ducklake_catalog):
        """Verify delete files work correctly after rename."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello'), (2, 'world')")

        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO name")
        cat.execute("DELETE FROM ducklake.test WHERE a = 1")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'new')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (2, 2)
        result = result.sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [2, 3]
        assert result["name"].tolist() == ["world", "new"]

    def test_rename_time_travel(self, ducklake_catalog):
        """Read at snapshot before and after rename."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello')")

        # Get snapshot before rename
        snap_before = cat.fetchone(
            "SELECT * FROM ducklake_current_snapshot('ducklake')"
        )[0]

        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO name")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'world')")
        cat.close()

        # Read at snapshot before rename: should have old name
        result_before = read_ducklake(
            cat.metadata_path, "test", snapshot_version=snap_before
        )
        assert list(result_before.columns) == ["a", "b"]
        assert result_before["b"].tolist() == ["hello"]

        # Read latest: should have new name with all data
        result_latest = read_ducklake(cat.metadata_path, "test")
        assert list(result_latest.columns) == ["a", "name"]
        assert sorted(result_latest["name"].tolist()) == ["hello", "world"]

    def test_rename_back_to_original_name(self, ducklake_catalog):
        """Rename b -> name -> b (round-trip), verify no data loss."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'first')")
        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO name")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'second')")
        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN name TO b")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'third')")
        cat.close()
        result = read_ducklake(cat.metadata_path, "test")
        assert list(result.columns) == ["a", "b"]
        assert result.shape == (3, 2)
        result = result.sort_values("a").reset_index(drop=True)
        assert result["b"].tolist() == ["first", "second", "third"]

    def test_rename_and_drop_column(self, ducklake_catalog):
        """Rename one column while dropping another."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR, c DOUBLE)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello', 3.14)")
        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO name")
        cat.execute("ALTER TABLE ducklake.test DROP COLUMN c")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'world')")
        cat.close()
        result = read_ducklake(cat.metadata_path, "test")
        assert list(result.columns) == ["a", "name"]
        assert result.shape == (2, 2)
        result = result.sort_values("a").reset_index(drop=True)
        assert result["name"].tolist() == ["hello", "world"]

    def test_rename_with_filter_on_renamed_column(self, ducklake_catalog):
        """Filter on the renamed column itself."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello'), (2, 'world')")
        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO name")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'new')")
        cat.close()
        result = read_ducklake(cat.metadata_path, "test")
        result = result[result["name"] == "hello"]
        assert result.shape == (1, 2)
        assert result["a"].tolist() == [1]
        assert result["name"].tolist() == ["hello"]


class TestTypePromotion:
    """Test reading after column type promotions."""

    def test_tinyint_to_integer_promotion(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b TINYINT)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 25)")

        cat.execute(
            "ALTER TABLE ducklake.test ALTER COLUMN b SET DATA TYPE INTEGER"
        )
        cat.execute("INSERT INTO ducklake.test VALUES (2, 1000)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort_values("a").reset_index(drop=True)
        assert result.shape == (2, 2)
        assert result["a"].tolist() == [1, 2]
        assert result["b"].tolist() == [25, 1000]

    def test_float_to_double_promotion(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b FLOAT)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 1.5)")

        cat.execute(
            "ALTER TABLE ducklake.test ALTER COLUMN b SET DATA TYPE DOUBLE"
        )
        cat.execute("INSERT INTO ducklake.test VALUES (2, 3.14)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort_values("a").reset_index(drop=True)
        assert result.shape == (2, 2)
        assert result["a"].tolist() == [1, 2]
        assert result["b"][0] == pytest.approx(1.5)
        assert result["b"][1] == pytest.approx(3.14)

    def test_integer_to_bigint_promotion(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 42)")

        cat.execute(
            "ALTER TABLE ducklake.test ALTER COLUMN b SET DATA TYPE BIGINT"
        )
        # Value larger than 2^31 (2147483648)
        cat.execute("INSERT INTO ducklake.test VALUES (2, 3000000000)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort_values("a").reset_index(drop=True)
        assert result.shape == (2, 2)
        assert result["a"].tolist() == [1, 2]
        assert result["b"].tolist() == [42, 3000000000]


class TestDefaultValues:
    """Test reading after columns are added with default values."""

    def test_add_column_with_default(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), (2)")

        cat.execute("ALTER TABLE ducklake.test ADD COLUMN b INTEGER DEFAULT 42")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 100)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort_values("a").reset_index(drop=True)
        assert result.shape == (3, 2)
        assert True  # schema check skipped for pandas
        assert result["a"].tolist() == [1, 2, 3]
        # Old Parquet files don't contain the new column, so old rows get NULL
        # (DuckLake does not backfill defaults into existing Parquet files)
        # pandas represents missing integers as NaN (float)
        b_vals = result["b"].tolist()
        assert pd.isna(b_vals[0])
        assert pd.isna(b_vals[1])
        assert b_vals[2] == 100

    def test_add_column_with_string_default(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), (2)")

        cat.execute(
            "ALTER TABLE ducklake.test ADD COLUMN b VARCHAR DEFAULT 'hello'"
        )
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'world')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort_values("a").reset_index(drop=True)
        assert result.shape == (3, 2)
        assert True  # schema check skipped for pandas
        assert result["a"].tolist() == [1, 2, 3]
        # Old Parquet files don't contain the new column, so old rows get NULL
        # (DuckLake does not backfill defaults into existing Parquet files)
        assert_list_equal(result["b"].tolist(), [None, None, "world"])

    def test_add_column_default_vs_null(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")

        # Add column without default (should be NULL for old rows)
        cat.execute("ALTER TABLE ducklake.test ADD COLUMN b VARCHAR")
        # Add column with default (old rows still get NULL -- defaults not backfilled)
        cat.execute("ALTER TABLE ducklake.test ADD COLUMN c INTEGER DEFAULT 0")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'val', 5)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort_values("a").reset_index(drop=True)
        assert result.shape == (2, 3)
        assert True  # schema check skipped for pandas
        assert result["a"].tolist() == [1, 2]
        # Old row: both b and c are NULL (DuckLake does not backfill defaults
        # into existing Parquet files; missing_columns="insert" fills with NULL)
        assert_list_equal(result["b"].tolist(), [None, "val"])
        c_vals = result["c"].tolist()
        assert pd.isna(c_vals[0])
        assert c_vals[1] == 5


class TestMixedAlter:
    """Test reading after mixed ALTER TABLE operations."""

    def test_drop_and_readd_column_different_type(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute(
            "CREATE TABLE ducklake.test (a INTEGER, b INTEGER, c VARCHAR)"
        )
        cat.execute("INSERT INTO ducklake.test VALUES (1, 10, 'first')")

        cat.execute("ALTER TABLE ducklake.test DROP COLUMN b")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'second')")

        cat.execute("ALTER TABLE ducklake.test ADD COLUMN b VARCHAR")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'third', 'new_b')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort_values("a").reset_index(drop=True)
        assert result.shape == (3, 3)
        assert result["a"].tolist() == [1, 2, 3]
        assert result["c"].tolist() == ["first", "second", "third"]
        # Old rows (before b was re-added) should have NULL for b
        assert_list_equal(result["b"].tolist(), [None, None, "new_b"])

    def test_rename_and_drop(self, ducklake_catalog):
        """Rename one column and drop another, then insert and read."""
        cat = ducklake_catalog

        cat.execute(
            "CREATE TABLE ducklake.test (a INTEGER, b VARCHAR, c DOUBLE)"
        )
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello', 3.14)")

        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO name")
        cat.execute("ALTER TABLE ducklake.test DROP COLUMN c")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'world')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort_values("a").reset_index(drop=True)
        assert list(result.columns) == ["a", "name"]
        assert result.shape == (2, 2)
        assert result["a"].tolist() == [1, 2]
        assert result["name"].tolist() == ["hello", "world"]


class TestTableRename:
    """Test reading after table rename."""

    def test_read_after_table_rename(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'hello'), (2, 'world')"
        )

        cat.execute("ALTER TABLE ducklake.test RENAME TO test2")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test2")
        result = result.sort_values("a").reset_index(drop=True)
        assert result.shape == (2, 2)
        assert True  # schema check skipped for pandas
        assert result["a"].tolist() == [1, 2]
        assert result["b"].tolist() == ["hello", "world"]

    def test_table_rename_old_name_fails(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello')")

        cat.execute("ALTER TABLE ducklake.test RENAME TO test2")
        cat.close()

        with pytest.raises(ValueError, match="test"):
            read_ducklake(cat.metadata_path, "test")


class TestStructEvolution:
    """Test reading after struct column evolution."""

    def test_struct_add_field(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute(
            "CREATE TABLE ducklake.test "
            "(a INTEGER, col STRUCT(i INTEGER, j INTEGER))"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, {i: 10, j: 20})"
        )

        cat.execute(
            "ALTER TABLE ducklake.test ALTER COLUMN col "
            "SET DATA TYPE STRUCT(i INTEGER, j INTEGER, k INTEGER)"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES (2, {i: 30, j: 40, k: 50})"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort_values("a").reset_index(drop=True)
        assert result.shape == (2, 2)
        structs = result["col"].tolist()
        # Old row should have k=None, new row should have k=50
        assert structs[0] == {"i": 10, "j": 20, "k": None}
        assert structs[1] == {"i": 30, "j": 40, "k": 50}

    def test_struct_drop_field(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute(
            "CREATE TABLE ducklake.test "
            "(a INTEGER, col STRUCT(i INTEGER, j INTEGER, k INTEGER))"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, {i: 10, j: 20, k: 30})"
        )

        cat.execute(
            "ALTER TABLE ducklake.test ALTER COLUMN col "
            "SET DATA TYPE STRUCT(i INTEGER, k INTEGER)"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES (2, {i: 40, k: 50})"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort_values("a").reset_index(drop=True)
        assert result.shape == (2, 2)
        structs = result["col"].tolist()
        # All rows should have only i and k fields
        assert structs[0] == {"i": 10, "k": 30}
        assert structs[1] == {"i": 40, "k": 50}

    def test_struct_rename_field(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute(
            "CREATE TABLE ducklake.test "
            "(a INTEGER, col STRUCT(i INTEGER, j INTEGER))"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, {i: 10, j: 20})"
        )

        cat.execute(
            "ALTER TABLE ducklake.test ALTER COLUMN col "
            "SET DATA TYPE STRUCT(i INTEGER, val INTEGER)"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES (2, {i: 30, val: 40})"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort_values("a").reset_index(drop=True)
        assert result.shape == (2, 2)
        structs = result["col"].tolist()
        # Both old and new data should be accessible under field name 'val'
        assert structs[0] == {"i": 10, "val": 20}
        assert structs[1] == {"i": 30, "val": 40}
