"""Delete handling tests for ducklake-pandas."""

from __future__ import annotations

from ducklake_pandas import read_ducklake


class TestDeleteFiles:
    """Test reading tables that have had rows deleted."""

    def test_read_after_delete(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test AS SELECT i AS a FROM range(10) t(i)"
        )
        cat.execute("DELETE FROM ducklake.test WHERE a >= 5")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].tolist()) == [0, 1, 2, 3, 4]

    def test_read_after_partial_delete(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')"
        )
        cat.execute("DELETE FROM ducklake.test WHERE a = 2")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 3
        assert sorted(result["a"].tolist()) == [1, 3, 4]

    def test_read_after_multiple_deletes(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test AS SELECT i AS a FROM range(20) t(i)"
        )
        cat.execute("DELETE FROM ducklake.test WHERE a < 5")
        cat.execute("DELETE FROM ducklake.test WHERE a >= 15")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].tolist()) == list(range(5, 15))

    def test_read_after_delete_all(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), (2), (3)")
        cat.execute("DELETE FROM ducklake.test")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 0

    def test_time_travel_before_delete(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), (2), (3)")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("DELETE FROM ducklake.test WHERE a = 2")
        cat.close()

        result_current = read_ducklake(cat.metadata_path, "test")
        assert sorted(result_current["a"].tolist()) == [1, 3]

        result_v1 = read_ducklake(cat.metadata_path, "test", snapshot_version=v1)
        assert sorted(result_v1["a"].tolist()) == [1, 2, 3]


class TestMultiFileDelete:
    """Test deletes that span multiple Parquet files."""

    def test_delete_spanning_multiple_files(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (i INTEGER, label VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test SELECT i, 'batch1' FROM range(0, 100) t(i)"
        )
        cat.execute(
            "INSERT INTO ducklake.test SELECT i + 1000, 'batch2' FROM range(0, 100) t(i)"
        )
        cat.execute("DELETE FROM ducklake.test WHERE i % 2 = 0")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 100
        i_values = sorted(result["i"].tolist())
        expected_batch1 = [x for x in range(0, 100) if x % 2 != 0]
        expected_batch2 = [x + 1000 for x in range(0, 100) if (x + 1000) % 2 != 0]
        assert i_values == sorted(expected_batch1 + expected_batch2)

    def test_delete_from_one_of_multiple_files(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (i INTEGER, label VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test SELECT i, 'batch1' FROM range(0, 100) t(i)"
        )
        cat.execute(
            "INSERT INTO ducklake.test SELECT i + 1000, 'batch2' FROM range(0, 100) t(i)"
        )
        cat.execute("DELETE FROM ducklake.test WHERE i < 50")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        i_values = sorted(result["i"].tolist())
        expected = list(range(50, 100)) + list(range(1000, 1100))
        assert i_values == sorted(expected)
        assert result.shape[0] == 150

    def test_delete_all_from_one_file(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (i INTEGER, label VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test SELECT i, 'batch1' FROM range(0, 100) t(i)"
        )
        cat.execute(
            "INSERT INTO ducklake.test SELECT i + 1000, 'batch2' FROM range(0, 100) t(i)"
        )
        cat.execute("DELETE FROM ducklake.test WHERE i < 100")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        i_values = sorted(result["i"].tolist())
        expected = list(range(1000, 1100))
        assert i_values == expected
        assert result.shape[0] == 100
        assert sorted(result["label"].tolist()) == ["batch2"] * 100


class TestDeleteTimeTravelAdvanced:
    """Advanced time travel tests with deletes."""

    def test_time_travel_between_deletes(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test AS SELECT i FROM range(0, 100) t(i)"
        )
        v1 = cat.fetchone(
            "SELECT * FROM ducklake_current_snapshot('ducklake')"
        )[0]

        cat.execute("DELETE FROM ducklake.test WHERE i < 25")
        v2 = cat.fetchone(
            "SELECT * FROM ducklake_current_snapshot('ducklake')"
        )[0]

        cat.execute("DELETE FROM ducklake.test WHERE i >= 75")
        cat.close()

        result_latest = read_ducklake(cat.metadata_path, "test")
        assert sorted(result_latest["i"].tolist()) == list(range(25, 75))
        assert result_latest.shape[0] == 50

        result_v2 = read_ducklake(
            cat.metadata_path, "test", snapshot_version=v2
        )
        assert sorted(result_v2["i"].tolist()) == list(range(25, 100))
        assert result_v2.shape[0] == 75

        result_v1 = read_ducklake(
            cat.metadata_path, "test", snapshot_version=v1
        )
        assert sorted(result_v1["i"].tolist()) == list(range(0, 100))
        assert result_v1.shape[0] == 100

    def test_time_travel_multi_file_delete(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (i INTEGER, label VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test SELECT i, 'batch1' FROM range(0, 50) t(i)"
        )
        cat.execute(
            "INSERT INTO ducklake.test SELECT i + 1000, 'batch2' FROM range(0, 50) t(i)"
        )
        snap_before = cat.fetchone(
            "SELECT * FROM ducklake_current_snapshot('ducklake')"
        )[0]

        cat.execute("DELETE FROM ducklake.test WHERE i % 2 = 0")
        cat.close()

        result_old = read_ducklake(
            cat.metadata_path, "test", snapshot_version=snap_before
        )
        assert result_old.shape[0] == 100

        result_latest = read_ducklake(cat.metadata_path, "test")
        expected_batch1 = [x for x in range(0, 50) if x % 2 != 0]
        expected_batch2 = [x + 1000 for x in range(0, 50) if (x + 1000) % 2 != 0]
        expected = sorted(expected_batch1 + expected_batch2)
        assert sorted(result_latest["i"].tolist()) == expected
        assert result_latest.shape[0] < 100


class TestDeleteEdgeCases:
    """Edge case tests for delete operations."""

    def test_noop_delete(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test AS SELECT i FROM range(0, 10) t(i)"
        )
        cat.execute("DELETE FROM ducklake.test WHERE i > 100")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 10
        assert sorted(result["i"].tolist()) == list(range(0, 10))

    def test_delete_empty_table(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (i INTEGER, label VARCHAR)")
        cat.execute("DELETE FROM ducklake.test")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 0
        assert list(result.columns) == ["i", "label"]

    def test_truncate_via_delete_all(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (i INTEGER, label VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test SELECT i, 'val' FROM range(0, 100) t(i)"
        )
        cat.execute("DELETE FROM ducklake.test")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 0
        assert list(result.columns) == ["i", "label"]

    def test_delete_then_insert(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test AS SELECT i FROM range(0, 10) t(i)"
        )
        cat.execute("DELETE FROM ducklake.test WHERE i < 5")
        cat.execute(
            "INSERT INTO ducklake.test SELECT i + 100 AS i FROM range(0, 5) t(i)"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 10
        i_values = sorted(result["i"].tolist())
        expected = list(range(5, 10)) + list(range(100, 105))
        assert i_values == sorted(expected)

    def test_multiple_sequential_deletes(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test AS SELECT i FROM range(0, 100) t(i)"
        )
        cat.execute("DELETE FROM ducklake.test WHERE i < 25")
        cat.execute("DELETE FROM ducklake.test WHERE i >= 75")
        cat.execute("DELETE FROM ducklake.test WHERE i = 50")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        expected = [x for x in range(25, 75) if x != 50]
        assert result.shape[0] == 49
        assert sorted(result["i"].tolist()) == expected
