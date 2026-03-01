"""Tests for table and column tags (metadata comments)."""

from __future__ import annotations

import duckdb
import polars as pl
import pytest

from ducklake_polars import (
    DuckLakeCatalog,
    create_ducklake_table,
    set_ducklake_table_tag,
    set_ducklake_column_tag,
    delete_ducklake_table_tag,
    delete_ducklake_column_tag,
    write_ducklake,
    read_ducklake,
)


# ------------------------------------------------------------------
# Set and read table tags
# ------------------------------------------------------------------


def test_set_and_read_table_tag(make_write_catalog):
    """Set a tag on a table and read it back via DuckLakeCatalog."""
    cat = make_write_catalog()
    path = cat.metadata_path
    dp = cat.data_path

    create_ducklake_table(
        path, "test", {"a": pl.Int64, "b": pl.Utf8}, data_path=dp
    )

    set_ducklake_table_tag(path, "test", "comment", "my table description", data_path=dp)

    catalog = DuckLakeCatalog(path, data_path=dp)
    tags = catalog.table_tags("test")
    assert tags.shape[0] == 1
    assert tags["key"][0] == "comment"
    assert tags["value"][0] == "my table description"


def test_set_and_read_column_tag(make_write_catalog):
    """Set a tag on a column and read it back via DuckLakeCatalog."""
    cat = make_write_catalog()
    path = cat.metadata_path
    dp = cat.data_path

    create_ducklake_table(
        path, "test", {"a": pl.Int64, "b": pl.Utf8}, data_path=dp
    )

    set_ducklake_column_tag(path, "test", "a", "comment", "primary key", data_path=dp)

    catalog = DuckLakeCatalog(path, data_path=dp)
    tags = catalog.column_tags("test", "a")
    assert tags.shape[0] == 1
    assert tags["key"][0] == "comment"
    assert tags["value"][0] == "primary key"


# ------------------------------------------------------------------
# Delete tags
# ------------------------------------------------------------------


def test_delete_table_tag(make_write_catalog):
    """Set and then delete a table tag."""
    cat = make_write_catalog()
    path = cat.metadata_path
    dp = cat.data_path

    create_ducklake_table(
        path, "test", {"a": pl.Int64}, data_path=dp
    )

    set_ducklake_table_tag(path, "test", "comment", "hello", data_path=dp)
    catalog = DuckLakeCatalog(path, data_path=dp)
    assert catalog.table_tags("test").shape[0] == 1

    delete_ducklake_table_tag(path, "test", "comment", data_path=dp)
    tags = catalog.table_tags("test")
    assert tags.shape[0] == 0


def test_delete_column_tag(make_write_catalog):
    """Set and then delete a column tag."""
    cat = make_write_catalog()
    path = cat.metadata_path
    dp = cat.data_path

    create_ducklake_table(
        path, "test", {"a": pl.Int64}, data_path=dp
    )

    set_ducklake_column_tag(path, "test", "a", "comment", "pk", data_path=dp)
    catalog = DuckLakeCatalog(path, data_path=dp)
    assert catalog.column_tags("test", "a").shape[0] == 1

    delete_ducklake_column_tag(path, "test", "a", "comment", data_path=dp)
    tags = catalog.column_tags("test", "a")
    assert tags.shape[0] == 0


def test_delete_nonexistent_table_tag_raises(make_write_catalog):
    """Deleting a tag that doesn't exist should raise ValueError."""
    cat = make_write_catalog()
    path = cat.metadata_path
    dp = cat.data_path

    create_ducklake_table(
        path, "test", {"a": pl.Int64}, data_path=dp
    )

    with pytest.raises(ValueError, match="Tag 'comment' not found"):
        delete_ducklake_table_tag(path, "test", "comment", data_path=dp)


# ------------------------------------------------------------------
# Multiple tags on the same table
# ------------------------------------------------------------------


def test_multiple_tags_on_table(make_write_catalog):
    """Multiple tags can be set on the same table."""
    cat = make_write_catalog()
    path = cat.metadata_path
    dp = cat.data_path

    create_ducklake_table(
        path, "test", {"a": pl.Int64}, data_path=dp
    )

    set_ducklake_table_tag(path, "test", "comment", "desc", data_path=dp)
    set_ducklake_table_tag(path, "test", "owner", "pdet", data_path=dp)
    set_ducklake_table_tag(path, "test", "version", "1.0", data_path=dp)

    catalog = DuckLakeCatalog(path, data_path=dp)
    tags = catalog.table_tags("test")
    assert tags.shape[0] == 3
    tag_dict = dict(zip(tags["key"].to_list(), tags["value"].to_list()))
    assert tag_dict["comment"] == "desc"
    assert tag_dict["owner"] == "pdet"
    assert tag_dict["version"] == "1.0"


def test_overwrite_table_tag(make_write_catalog):
    """Setting a tag with the same key overwrites the previous value."""
    cat = make_write_catalog()
    path = cat.metadata_path
    dp = cat.data_path

    create_ducklake_table(
        path, "test", {"a": pl.Int64}, data_path=dp
    )

    set_ducklake_table_tag(path, "test", "comment", "first", data_path=dp)
    set_ducklake_table_tag(path, "test", "comment", "second", data_path=dp)

    catalog = DuckLakeCatalog(path, data_path=dp)
    tags = catalog.table_tags("test")
    assert tags.shape[0] == 1
    assert tags["value"][0] == "second"


# ------------------------------------------------------------------
# DuckDB interop
# ------------------------------------------------------------------


def test_duckdb_sets_comment_we_read_tag(ducklake_catalog):
    """DuckDB sets COMMENT ON TABLE, we read it as a tag."""
    catalog = ducklake_catalog
    catalog.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
    catalog.execute("COMMENT ON TABLE ducklake.test IS 'duckdb description'")
    catalog.execute("COMMENT ON COLUMN ducklake.test.a IS 'col a desc'")
    catalog.close()

    cat = DuckLakeCatalog(catalog.metadata_path, data_path=catalog.data_path)

    table_tags = cat.table_tags("test")
    assert table_tags.shape[0] == 1
    assert table_tags["key"][0] == "comment"
    assert table_tags["value"][0] == "duckdb description"

    col_tags = cat.column_tags("test", "a")
    assert col_tags.shape[0] == 1
    assert col_tags["key"][0] == "comment"
    assert col_tags["value"][0] == "col a desc"


def test_we_set_tag_duckdb_reads_comment(make_write_catalog):
    """We set a 'comment' tag, DuckDB reads it via COMMENT."""
    cat = make_write_catalog()
    path = cat.metadata_path
    dp = cat.data_path

    create_ducklake_table(
        path, "test", {"a": pl.Int64, "b": pl.Utf8}, data_path=dp
    )

    set_ducklake_table_tag(path, "test", "comment", "our table desc", data_path=dp)
    set_ducklake_column_tag(path, "test", "a", "comment", "our col desc", data_path=dp)

    # Read back with DuckDB
    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")

    if cat.backend == "sqlite":
        source = f"ducklake:sqlite:{path}"
    else:
        source = f"ducklake:postgres:{path}"

    con.execute(
        f"ATTACH '{source}' AS ducklake (DATA_PATH '{dp}')"
    )

    table_comment = con.execute(
        "SELECT comment FROM duckdb_tables() "
        "WHERE table_name = 'test' AND schema_name = 'main'"
    ).fetchone()
    assert table_comment is not None
    assert table_comment[0] == "our table desc"

    col_comment = con.execute(
        "SELECT comment FROM duckdb_columns() "
        "WHERE table_name = 'test' AND column_name = 'a'"
    ).fetchone()
    assert col_comment is not None
    assert col_comment[0] == "our col desc"

    con.close()


# ------------------------------------------------------------------
# Tags survive time travel
# ------------------------------------------------------------------


def test_tags_time_travel(make_write_catalog):
    """Tags at an earlier snapshot should still be visible via time travel."""
    cat = make_write_catalog()
    path = cat.metadata_path
    dp = cat.data_path

    create_ducklake_table(
        path, "test", {"a": pl.Int64}, data_path=dp
    )

    # Record snapshot before setting tag
    catalog = DuckLakeCatalog(path, data_path=dp)
    snap_before = catalog.current_snapshot()

    set_ducklake_table_tag(path, "test", "comment", "v1", data_path=dp)
    snap_with_v1 = catalog.current_snapshot()

    set_ducklake_table_tag(path, "test", "comment", "v2", data_path=dp)
    snap_with_v2 = catalog.current_snapshot()

    # Current snapshot should show v2
    tags = catalog.table_tags("test")
    assert tags["value"][0] == "v2"

    # Use the reader directly for time-travel verification
    from ducklake_core._catalog import DuckLakeCatalogReader

    with DuckLakeCatalogReader(path, data_path_override=dp) as reader:
        # Before any tags were set
        table_info = reader.get_table("test", "main", snap_before)
        tags_before = reader.get_table_tags(table_info.table_id, snap_before)
        assert len(tags_before) == 0

        # At v1
        tags_v1 = reader.get_table_tags(table_info.table_id, snap_with_v1)
        assert tags_v1["comment"] == "v1"

        # At v2
        tags_v2 = reader.get_table_tags(table_info.table_id, snap_with_v2)
        assert tags_v2["comment"] == "v2"


def test_column_tags_time_travel(make_write_catalog):
    """Column tags at earlier snapshots should be correctly versioned."""
    cat = make_write_catalog()
    path = cat.metadata_path
    dp = cat.data_path

    create_ducklake_table(
        path, "test", {"a": pl.Int64}, data_path=dp
    )

    set_ducklake_column_tag(path, "test", "a", "comment", "first", data_path=dp)
    catalog = DuckLakeCatalog(path, data_path=dp)
    snap_v1 = catalog.current_snapshot()

    set_ducklake_column_tag(path, "test", "a", "comment", "second", data_path=dp)
    snap_v2 = catalog.current_snapshot()

    delete_ducklake_column_tag(path, "test", "a", "comment", data_path=dp)
    snap_deleted = catalog.current_snapshot()

    from ducklake_core._catalog import DuckLakeCatalogReader

    with DuckLakeCatalogReader(path, data_path_override=dp) as reader:
        table_info = reader.get_table("test", "main", snap_v1)

        # Get column_id for 'a'
        cols = reader.get_columns(table_info.table_id, snap_v1)
        col_a = [c for c in cols if c.column_name == "a"][0]

        tags_v1 = reader.get_column_tags(table_info.table_id, col_a.column_id, snap_v1)
        assert tags_v1["comment"] == "first"

        tags_v2 = reader.get_column_tags(table_info.table_id, col_a.column_id, snap_v2)
        assert tags_v2["comment"] == "second"

        tags_del = reader.get_column_tags(table_info.table_id, col_a.column_id, snap_deleted)
        assert len(tags_del) == 0


# ------------------------------------------------------------------
# Empty / no tags
# ------------------------------------------------------------------


def test_no_tags_returns_empty(make_write_catalog):
    """Reading tags from a table with no tags returns an empty DataFrame."""
    cat = make_write_catalog()
    path = cat.metadata_path
    dp = cat.data_path

    create_ducklake_table(
        path, "test", {"a": pl.Int64}, data_path=dp
    )

    catalog = DuckLakeCatalog(path, data_path=dp)
    tags = catalog.table_tags("test")
    assert tags.shape[0] == 0
    assert "key" in tags.columns
    assert "value" in tags.columns
