"""Tests for ducklake-pandas view operations (CREATE VIEW, DROP VIEW)."""

from __future__ import annotations

import duckdb
import pandas as pd
import numpy as np
import pytest

from ducklake_pandas import (
    create_ducklake_view,
    create_ducklake_table,
    drop_ducklake_view,
    read_ducklake,
    write_ducklake,
)


# ===========================================================================
# CREATE VIEW
# ===========================================================================


class TestCreateView:
    """Test CREATE VIEW."""

    def test_create_view_basic(self, make_write_catalog):
        """Create a view and verify metadata is written correctly."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        create_ducklake_view(
            cat.metadata_path,
            "test_view",
            "SELECT id, name FROM test WHERE id > 1",
        )

        # View row exists in ducklake_view
        row = cat.query_one(
            "SELECT view_name, dialect, sql, column_aliases "
            "FROM ducklake_view WHERE view_name = 'test_view' "
            "AND end_snapshot IS NULL"
        )
        assert row is not None
        assert row[0] == "test_view"
        assert row[1] == "duckdb"
        assert "SELECT" in row[2]
        assert row[3] == ""

    def test_create_view_metadata_correct(self, make_write_catalog):
        """Verify metadata: snapshot, schema_version, changes."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        sv_before = cat.query_one(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )[0]

        create_ducklake_view(
            cat.metadata_path,
            "test_view",
            "SELECT id FROM test",
        )

        # Schema version incremented
        sv_after = cat.query_one(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )[0]
        assert sv_after == sv_before + 1

        # Changes recorded
        change = cat.query_one(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )
        assert "created_view" in change[0]
        assert "test_view" in change[0]

    def test_create_view_uses_next_catalog_id(self, make_write_catalog):
        """View IDs share the next_catalog_id counter with tables."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        # Get the next_catalog_id before creating the view
        nci_before = cat.query_one(
            "SELECT next_catalog_id FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )[0]

        create_ducklake_view(
            cat.metadata_path,
            "test_view",
            "SELECT id FROM test",
        )

        # View got the expected ID
        view_row = cat.query_one(
            "SELECT view_id FROM ducklake_view "
            "WHERE view_name = 'test_view' AND end_snapshot IS NULL"
        )
        assert view_row[0] == nci_before

        # next_catalog_id was incremented
        nci_after = cat.query_one(
            "SELECT next_catalog_id FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )[0]
        assert nci_after == nci_before + 1

    def test_create_view_duplicate_raises(self, make_write_catalog):
        """Creating a view that already exists raises."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        create_ducklake_view(
            cat.metadata_path,
            "test_view",
            "SELECT id FROM test",
        )

        with pytest.raises(ValueError, match="already exists"):
            create_ducklake_view(
                cat.metadata_path,
                "test_view",
                "SELECT id FROM test",
            )

    def test_create_view_or_replace(self, make_write_catalog):
        """CREATE OR REPLACE VIEW replaces an existing view."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        create_ducklake_view(
            cat.metadata_path,
            "test_view",
            "SELECT id FROM test",
        )

        old_view = cat.query_one(
            "SELECT view_id FROM ducklake_view "
            "WHERE view_name = 'test_view' AND end_snapshot IS NULL"
        )
        old_view_id = old_view[0]

        create_ducklake_view(
            cat.metadata_path,
            "test_view",
            "SELECT id, name FROM test",
            or_replace=True,
        )

        # Old view ended
        old_end = cat.query_one(
            "SELECT end_snapshot FROM ducklake_view "
            "WHERE view_id = ?",
            [old_view_id],
        )
        assert old_end[0] is not None

        # New view is active with updated SQL
        new_view = cat.query_one(
            "SELECT view_id, sql FROM ducklake_view "
            "WHERE view_name = 'test_view' AND end_snapshot IS NULL"
        )
        assert new_view is not None
        assert new_view[0] != old_view_id
        assert "name" in new_view[1]

        # Changes recorded both drop and create
        change = cat.query_one(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )
        assert "dropped_view" in change[0]
        assert "created_view" in change[0]

    def test_create_view_in_nondefault_schema(self, make_write_catalog):
        """Create a view in a non-default schema."""
        from ducklake_pandas import create_ducklake_schema

        cat = make_write_catalog()
        create_ducklake_schema(cat.metadata_path, "myschema")

        df = pd.DataFrame({"id": [1, 2]})
        write_ducklake(
            df, cat.metadata_path, "test", schema="myschema", mode="error"
        )

        create_ducklake_view(
            cat.metadata_path,
            "test_view",
            "SELECT id FROM test",
            schema="myschema",
        )

        row = cat.query_one(
            "SELECT view_name FROM ducklake_view "
            "WHERE view_name = 'test_view' AND end_snapshot IS NULL"
        )
        assert row is not None

    def test_create_view_nonexistent_schema_raises(self, make_write_catalog):
        """Creating a view in a nonexistent schema raises."""
        cat = make_write_catalog()

        with pytest.raises(ValueError, match="not found"):
            create_ducklake_view(
                cat.metadata_path,
                "test_view",
                "SELECT 1",
                schema="missing",
            )

    def test_create_view_duckdb_interop(self, make_write_catalog):
        """DuckDB can see and query a view created by polars."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        create_ducklake_view(
            cat.metadata_path,
            "test_view",
            "SELECT id, name FROM main.test WHERE id > 1",
        )

        if cat.backend == "sqlite":
            source = f"ducklake:sqlite:{cat.metadata_path}"
        else:
            source = f"ducklake:postgres:{cat.metadata_path}"

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH '{source}' AS ducklake "
            f"(DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        result = con.execute(
            "SELECT * FROM ducklake.test_view ORDER BY id"
        ).fetchall()
        con.close()

        assert len(result) == 2
        assert result[0] == (2, "b")
        assert result[1] == (3, "c")

    def test_duckdb_create_view_polars_sees_metadata(self, make_write_catalog):
        """DuckDB creates a view, polars can see it in metadata."""
        cat = make_write_catalog()

        if cat.backend == "sqlite":
            source = f"ducklake:sqlite:{cat.metadata_path}"
        else:
            source = f"ducklake:postgres:{cat.metadata_path}"

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH '{source}' AS ducklake "
            f"(DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE TABLE ducklake.test (id INTEGER, name VARCHAR)")
        con.execute("INSERT INTO ducklake.test VALUES (1, 'hello'), (2, 'world')")
        con.execute(
            "CREATE VIEW ducklake.test_view AS "
            "SELECT id, name FROM ducklake.main.test WHERE id > 0"
        )
        con.close()

        # Verify view exists in metadata
        row = cat.query_one(
            "SELECT view_name, dialect, sql FROM ducklake_view "
            "WHERE view_name = 'test_view' AND end_snapshot IS NULL"
        )
        assert row is not None
        assert row[0] == "test_view"
        assert row[1] == "duckdb"
        assert "SELECT" in row[2]

    def test_create_view_with_author_and_message(self, make_write_catalog):
        """Author and commit message are recorded in snapshot changes."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        create_ducklake_view(
            cat.metadata_path,
            "test_view",
            "SELECT id FROM test",
            author="test-user",
            commit_message="adding a view",
        )

        row = cat.query_one(
            "SELECT author, commit_message FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )
        assert row[0] == "test-user"
        assert row[1] == "adding a view"

    def test_create_multiple_views(self, make_write_catalog):
        """Create multiple views on the same table."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10.0, 20.0, 30.0]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        create_ducklake_view(
            cat.metadata_path,
            "view1",
            "SELECT id FROM test",
        )
        create_ducklake_view(
            cat.metadata_path,
            "view2",
            "SELECT id, value FROM test WHERE value > 10",
        )

        count = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_view WHERE end_snapshot IS NULL"
        )[0]
        assert count == 2


# ===========================================================================
# DROP VIEW
# ===========================================================================


class TestDropView:
    """Test DROP VIEW."""

    def test_drop_view_basic(self, make_write_catalog):
        """Drop a view and verify metadata."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        create_ducklake_view(
            cat.metadata_path,
            "test_view",
            "SELECT id FROM test",
        )

        drop_ducklake_view(cat.metadata_path, "test_view")

        # View has end_snapshot set
        row = cat.query_one(
            "SELECT end_snapshot FROM ducklake_view "
            "WHERE view_name = 'test_view'"
        )
        assert row is not None
        assert row[0] is not None

        # No active views remain
        active = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_view WHERE end_snapshot IS NULL"
        )[0]
        assert active == 0

    def test_drop_view_metadata_correct(self, make_write_catalog):
        """Verify metadata: snapshot, schema_version, changes."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        create_ducklake_view(
            cat.metadata_path,
            "test_view",
            "SELECT id FROM test",
        )

        sv_before = cat.query_one(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )[0]

        view_row = cat.query_one(
            "SELECT view_id FROM ducklake_view "
            "WHERE view_name = 'test_view' AND end_snapshot IS NULL"
        )
        view_id = view_row[0]

        drop_ducklake_view(cat.metadata_path, "test_view")

        # Schema version incremented
        sv_after = cat.query_one(
            "SELECT schema_version FROM ducklake_snapshot "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )[0]
        assert sv_after == sv_before + 1

        # Changes recorded with view_id
        change = cat.query_one(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        )
        assert f"dropped_view:{view_id}" == change[0]

    def test_drop_view_nonexistent_raises(self, make_write_catalog):
        """Dropping a nonexistent view raises."""
        cat = make_write_catalog()

        with pytest.raises(ValueError, match="not found"):
            drop_ducklake_view(cat.metadata_path, "missing")

    def test_drop_view_duckdb_interop(self, make_write_catalog):
        """DuckDB confirms view is dropped after polars drops it."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        create_ducklake_view(
            cat.metadata_path,
            "test_view",
            "SELECT id FROM main.test",
        )

        drop_ducklake_view(cat.metadata_path, "test_view")

        if cat.backend == "sqlite":
            source = f"ducklake:sqlite:{cat.metadata_path}"
        else:
            source = f"ducklake:postgres:{cat.metadata_path}"

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH '{source}' AS ducklake "
            f"(DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        with pytest.raises(duckdb.CatalogException):
            con.execute("SELECT * FROM ducklake.test_view")
        con.close()

    def test_duckdb_drop_view_polars_confirms(self, make_write_catalog):
        """DuckDB drops a view, polars confirms it in metadata."""
        cat = make_write_catalog()

        if cat.backend == "sqlite":
            source = f"ducklake:sqlite:{cat.metadata_path}"
        else:
            source = f"ducklake:postgres:{cat.metadata_path}"

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH '{source}' AS ducklake "
            f"(DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE TABLE ducklake.test (id INTEGER)")
        con.execute("INSERT INTO ducklake.test VALUES (1)")
        con.execute(
            "CREATE VIEW ducklake.test_view AS SELECT id FROM ducklake.main.test"
        )
        con.execute("DROP VIEW ducklake.test_view")
        con.close()

        # All views should have end_snapshot set
        active = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_view WHERE end_snapshot IS NULL"
        )[0]
        assert active == 0

    def test_drop_view_does_not_affect_underlying_table(self, make_write_catalog):
        """Dropping a view does not affect the underlying table."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1, 2, 3]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        create_ducklake_view(
            cat.metadata_path,
            "test_view",
            "SELECT id FROM test",
        )

        drop_ducklake_view(cat.metadata_path, "test_view")

        # Table is still readable
        result = read_ducklake(cat.metadata_path, "test").sort_values("id").reset_index(drop=True)
        assert result["id"].tolist() == [1, 2, 3]

    def test_drop_and_recreate_view(self, make_write_catalog):
        """Drop a view and create a new one with the same name."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        create_ducklake_view(
            cat.metadata_path,
            "test_view",
            "SELECT id FROM test",
        )

        drop_ducklake_view(cat.metadata_path, "test_view")

        create_ducklake_view(
            cat.metadata_path,
            "test_view",
            "SELECT id, name FROM test",
        )

        # New view is active
        row = cat.query_one(
            "SELECT sql FROM ducklake_view "
            "WHERE view_name = 'test_view' AND end_snapshot IS NULL"
        )
        assert row is not None
        assert "name" in row[0]

        # Total view rows (2 ended + 1 active)
        total = cat.query_one(
            "SELECT COUNT(*) FROM ducklake_view WHERE view_name = 'test_view'"
        )[0]
        assert total == 2  # original (ended) + recreated (active)

    def test_create_and_drop_view_duckdb_roundtrip(self, make_write_catalog):
        """Full roundtrip: polars creates view, DuckDB queries it, polars drops it."""
        cat = make_write_catalog()
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10.0, 20.0, 30.0]})
        write_ducklake(df, cat.metadata_path, "test", mode="error")

        create_ducklake_view(
            cat.metadata_path,
            "high_values",
            "SELECT id, value FROM main.test WHERE value > 15.0",
        )

        # DuckDB can query the view
        if cat.backend == "sqlite":
            source = f"ducklake:sqlite:{cat.metadata_path}"
        else:
            source = f"ducklake:postgres:{cat.metadata_path}"

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH '{source}' AS ducklake "
            f"(DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        result = con.execute(
            "SELECT * FROM ducklake.high_values ORDER BY id"
        ).fetchall()
        con.close()

        assert len(result) == 2
        assert result[0] == (2, 20.0)
        assert result[1] == (3, 30.0)

        # Now drop the view
        drop_ducklake_view(cat.metadata_path, "high_values")

        # DuckDB can no longer query it
        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH '{source}' AS ducklake "
            f"(DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        with pytest.raises(duckdb.CatalogException):
            con.execute("SELECT * FROM ducklake.high_values")
        con.close()
