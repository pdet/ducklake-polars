"""Tests for reading data after compaction/maintenance operations.

Verifies that the reader correctly handles multi-file scenarios arising
from multiple inserts, deletes, updates, and truncation via DuckDB.
"""

from __future__ import annotations

from ducklake_polars import read_ducklake


class TestMultipleSmallInserts:
    """Multiple small inserts then read: all rows visible."""

    def test_five_batches_of_ten(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        for batch in range(5):
            values = ", ".join(
                f"({batch * 10 + i}, 'row_{batch * 10 + i}')" for i in range(10)
            )
            cat.execute(f"INSERT INTO ducklake.test VALUES {values}")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 50
        assert sorted(result["a"].to_list()) == list(range(50))


class TestInsertDeleteRead:
    """Insert rows, delete half, verify only remaining rows visible."""

    def test_delete_half(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute(
            "CREATE TABLE ducklake.test AS "
            "SELECT i AS a FROM range(100) t(i)"
        )
        cat.execute("DELETE FROM ducklake.test WHERE a >= 50")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 50
        assert sorted(result["a"].to_list()) == list(range(50))

    def test_delete_even_rows(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute(
            "CREATE TABLE ducklake.test AS "
            "SELECT i AS a FROM range(100) t(i)"
        )
        cat.execute("DELETE FROM ducklake.test WHERE a % 2 = 0")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 50
        assert sorted(result["a"].to_list()) == list(range(1, 100, 2))


class TestMultipleDataFiles:
    """Inserts in separate transactions create multiple Parquet files; read all."""

    def test_separate_transactions(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        # Each INSERT is its own transaction → separate Parquet files
        for i in range(5):
            cat.execute(f"INSERT INTO ducklake.test VALUES ({i})")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 5
        assert sorted(result["a"].to_list()) == list(range(5))

    def test_larger_separate_batches(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        for batch in range(3):
            values = ", ".join(
                f"({batch * 100 + i}, 'b{batch}')" for i in range(100)
            )
            cat.execute(f"INSERT INTO ducklake.test VALUES {values}")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 300
        for batch in range(3):
            batch_rows = result.filter(
                result["b"] == f"b{batch}"
            )
            assert batch_rows.shape[0] == 100


class TestInsertUpdateRead:
    """Insert + update creates new files; read back correct values."""

    def test_update_subset(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 'old'), (2, 'old'), (3, 'old'), (4, 'old'), (5, 'old')"
        )
        cat.execute("UPDATE ducklake.test SET b = 'new' WHERE a <= 3")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 5
        result = result.sort("a")
        assert result["b"].to_list() == ["new", "new", "new", "old", "old"]

    def test_update_all_rows(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b INTEGER)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 10), (2, 20), (3, 30)"
        )
        cat.execute("UPDATE ducklake.test SET b = b * 2")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result["b"].to_list() == [20, 40, 60]


class TestLargeBatchSmallUpdates:
    """Insert 1000 rows, update 10 rows, verify correct data."""

    def test_large_insert_small_update(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute(
            "CREATE TABLE ducklake.test AS "
            "SELECT i AS a, i * 10 AS b FROM range(1000) t(i)"
        )
        # Update rows 0-9
        cat.execute(
            "UPDATE ducklake.test SET b = -1 WHERE a < 10"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 1000

        result = result.sort("a")
        # First 10 rows should have b = -1
        updated = result.filter(result["a"] < 10)
        assert all(v == -1 for v in updated["b"].to_list())

        # Remaining rows unchanged
        unchanged = result.filter(result["a"] >= 10)
        expected_b = [i * 10 for i in range(10, 1000)]
        assert unchanged.sort("a")["b"].to_list() == expected_b


class TestTruncateReinsert:
    """Truncate (DELETE WHERE true) + reinsert: verify fresh data only."""

    def test_truncate_and_reinsert(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'old'), (2, 'old'), (3, 'old')"
        )
        # Truncate
        cat.execute("DELETE FROM ducklake.test WHERE true")
        # Reinsert fresh data
        cat.execute(
            "INSERT INTO ducklake.test VALUES (10, 'fresh'), (20, 'fresh')"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 2
        result = result.sort("a")
        assert result["a"].to_list() == [10, 20]
        assert result["b"].to_list() == ["fresh", "fresh"]

    def test_truncate_leaves_empty_table(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), (2), (3)")
        cat.execute("DELETE FROM ducklake.test WHERE true")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 0
        assert result.columns == ["a"]


class TestMultipleTablesIndependent:
    """Operations on table A don't affect reads from table B."""

    def test_independent_tables(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite

        # Create two tables
        cat.execute("CREATE TABLE ducklake.table_a (a INTEGER, b VARCHAR)")
        cat.execute("CREATE TABLE ducklake.table_b (x INTEGER, y INTEGER)")

        # Insert into both
        cat.execute(
            "INSERT INTO ducklake.table_a VALUES (1, 'a1'), (2, 'a2'), (3, 'a3')"
        )
        cat.execute(
            "INSERT INTO ducklake.table_b VALUES (10, 100), (20, 200), (30, 300)"
        )

        # Mutate table_a: delete and update
        cat.execute("DELETE FROM ducklake.table_a WHERE a = 2")
        cat.execute("UPDATE ducklake.table_a SET b = 'modified' WHERE a = 1")

        cat.close()

        # table_a reflects mutations
        result_a = read_ducklake(cat.metadata_path, "table_a")
        assert result_a.shape[0] == 2
        result_a = result_a.sort("a")
        assert result_a["a"].to_list() == [1, 3]
        assert result_a["b"].to_list() == ["modified", "a3"]

        # table_b is untouched
        result_b = read_ducklake(cat.metadata_path, "table_b")
        assert result_b.shape[0] == 3
        result_b = result_b.sort("x")
        assert result_b["x"].to_list() == [10, 20, 30]
        assert result_b["y"].to_list() == [100, 200, 300]

    def test_truncate_one_table_preserves_other(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite

        cat.execute("CREATE TABLE ducklake.alpha (val INTEGER)")
        cat.execute("CREATE TABLE ducklake.beta (val INTEGER)")

        cat.execute("INSERT INTO ducklake.alpha VALUES (1), (2), (3)")
        cat.execute("INSERT INTO ducklake.beta VALUES (10), (20), (30)")

        # Truncate alpha
        cat.execute("DELETE FROM ducklake.alpha WHERE true")

        cat.close()

        result_alpha = read_ducklake(cat.metadata_path, "alpha")
        assert result_alpha.shape[0] == 0

        result_beta = read_ducklake(cat.metadata_path, "beta")
        assert result_beta.shape[0] == 3
        assert sorted(result_beta["val"].to_list()) == [10, 20, 30]
