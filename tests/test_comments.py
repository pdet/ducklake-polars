"""Comment / tag edge-case tests for DuckLake catalog.

Covers:
- Setting comment on a column directly via the writer API
- Setting comments mixed with other operations in the same DuckDB session
- Setting comment on table + column in sequence
- Uses ducklake_catalog_sqlite fixture to avoid Postgres deadlocks
"""

from __future__ import annotations

import polars as pl
import pytest

from ducklake_polars import (
    DuckLakeCatalog,
    create_ducklake_table,
    set_ducklake_column_tag,
    set_ducklake_table_tag,
    write_ducklake,
    read_ducklake,
)


# ------------------------------------------------------------------
# Column comment via writer API
# ------------------------------------------------------------------


def test_set_column_comment_via_writer(ducklake_catalog_sqlite):
    """Set a 'comment' tag on a column through the writer API and read back."""
    cat = ducklake_catalog_sqlite
    cat.execute("CREATE TABLE ducklake.items (id INTEGER, name VARCHAR, price DOUBLE)")
    cat.close()

    path = cat.metadata_path
    dp = cat.data_path

    set_ducklake_column_tag(path, "items", "id", "comment", "primary key", data_path=dp)
    set_ducklake_column_tag(path, "items", "name", "comment", "item display name", data_path=dp)
    set_ducklake_column_tag(path, "items", "price", "comment", "unit price in USD", data_path=dp)

    catalog = DuckLakeCatalog(path, data_path=dp)

    id_tags = catalog.column_tags("items", "id")
    assert id_tags.shape[0] == 1
    assert id_tags["key"][0] == "comment"
    assert id_tags["value"][0] == "primary key"

    name_tags = catalog.column_tags("items", "name")
    assert name_tags["value"][0] == "item display name"

    price_tags = catalog.column_tags("items", "price")
    assert price_tags["value"][0] == "unit price in USD"


def test_set_column_comment_overwrite(ducklake_catalog_sqlite):
    """Overwriting a column comment replaces the old value."""
    cat = ducklake_catalog_sqlite
    cat.execute("CREATE TABLE ducklake.t (a INTEGER)")
    cat.close()

    path = cat.metadata_path
    dp = cat.data_path

    set_ducklake_column_tag(path, "t", "a", "comment", "old", data_path=dp)
    set_ducklake_column_tag(path, "t", "a", "comment", "new", data_path=dp)

    catalog = DuckLakeCatalog(path, data_path=dp)
    tags = catalog.column_tags("t", "a")
    assert tags.shape[0] == 1
    assert tags["value"][0] == "new"


# ------------------------------------------------------------------
# Comments mixed with other DuckDB operations in one session
# ------------------------------------------------------------------


def test_comment_mixed_with_insert(ducklake_catalog_sqlite):
    """COMMENT ON mixed with INSERT in the same DuckDB session."""
    cat = ducklake_catalog_sqlite
    cat.execute("CREATE TABLE ducklake.mixed (id INTEGER, val VARCHAR)")
    cat.execute("INSERT INTO ducklake.mixed VALUES (1, 'first')")
    cat.execute("COMMENT ON TABLE ducklake.mixed IS 'table with data'")
    cat.execute("INSERT INTO ducklake.mixed VALUES (2, 'second')")
    cat.execute("COMMENT ON COLUMN ducklake.mixed.val IS 'the value column'")
    cat.close()

    # Verify data
    result = read_ducklake(cat.metadata_path, "mixed", data_path=cat.data_path)
    result = result.sort("id")
    assert result.shape[0] == 2
    assert result["id"].to_list() == [1, 2]

    # Verify comments
    catalog = DuckLakeCatalog(cat.metadata_path, data_path=cat.data_path)
    table_tags = catalog.table_tags("mixed")
    assert table_tags.shape[0] == 1
    assert table_tags["value"][0] == "table with data"

    col_tags = catalog.column_tags("mixed", "val")
    assert col_tags.shape[0] == 1
    assert col_tags["value"][0] == "the value column"


def test_comment_mixed_with_alter(ducklake_catalog_sqlite):
    """COMMENT ON mixed with ALTER TABLE in the same DuckDB session."""
    cat = ducklake_catalog_sqlite
    cat.execute("CREATE TABLE ducklake.evolve (a INTEGER)")
    cat.execute("COMMENT ON TABLE ducklake.evolve IS 'v1 schema'")
    cat.execute("ALTER TABLE ducklake.evolve ADD COLUMN b VARCHAR")
    cat.execute("COMMENT ON COLUMN ducklake.evolve.b IS 'added later'")
    cat.close()

    catalog = DuckLakeCatalog(cat.metadata_path, data_path=cat.data_path)

    table_tags = catalog.table_tags("evolve")
    assert table_tags.shape[0] == 1
    assert table_tags["value"][0] == "v1 schema"

    col_tags = catalog.column_tags("evolve", "b")
    assert col_tags.shape[0] == 1
    assert col_tags["value"][0] == "added later"


# ------------------------------------------------------------------
# Table + column comments in sequence
# ------------------------------------------------------------------


def test_table_then_column_comment_sequence(ducklake_catalog_sqlite):
    """Set table comment first, then column comment — both should survive."""
    cat = ducklake_catalog_sqlite
    cat.execute("CREATE TABLE ducklake.seq (x INTEGER, y VARCHAR)")
    cat.close()

    path = cat.metadata_path
    dp = cat.data_path

    # Table tag first
    set_ducklake_table_tag(path, "seq", "comment", "sequence table", data_path=dp)
    # Then column tags
    set_ducklake_column_tag(path, "seq", "x", "comment", "x axis", data_path=dp)
    set_ducklake_column_tag(path, "seq", "y", "comment", "y label", data_path=dp)

    catalog = DuckLakeCatalog(path, data_path=dp)

    table_tags = catalog.table_tags("seq")
    assert table_tags.shape[0] == 1
    assert table_tags["value"][0] == "sequence table"

    x_tags = catalog.column_tags("seq", "x")
    assert x_tags["value"][0] == "x axis"

    y_tags = catalog.column_tags("seq", "y")
    assert y_tags["value"][0] == "y label"


def test_column_then_table_comment_sequence(ducklake_catalog_sqlite):
    """Set column comment first, then table comment — both should survive."""
    cat = ducklake_catalog_sqlite
    cat.execute("CREATE TABLE ducklake.rev (a INTEGER, b VARCHAR)")
    cat.close()

    path = cat.metadata_path
    dp = cat.data_path

    # Column tags first
    set_ducklake_column_tag(path, "rev", "a", "comment", "col a", data_path=dp)
    set_ducklake_column_tag(path, "rev", "b", "comment", "col b", data_path=dp)
    # Then table tag
    set_ducklake_table_tag(path, "rev", "comment", "reverse order", data_path=dp)

    catalog = DuckLakeCatalog(path, data_path=dp)

    table_tags = catalog.table_tags("rev")
    assert table_tags.shape[0] == 1
    assert table_tags["value"][0] == "reverse order"

    a_tags = catalog.column_tags("rev", "a")
    assert a_tags["value"][0] == "col a"

    b_tags = catalog.column_tags("rev", "b")
    assert b_tags["value"][0] == "col b"


def test_multiple_custom_tags_on_columns(ducklake_catalog_sqlite):
    """Multiple non-comment tags on the same column."""
    cat = ducklake_catalog_sqlite
    cat.execute("CREATE TABLE ducklake.multi (col1 INTEGER)")
    cat.close()

    path = cat.metadata_path
    dp = cat.data_path

    set_ducklake_column_tag(path, "multi", "col1", "comment", "main column", data_path=dp)
    set_ducklake_column_tag(path, "multi", "col1", "pii", "false", data_path=dp)
    set_ducklake_column_tag(path, "multi", "col1", "unit", "count", data_path=dp)

    catalog = DuckLakeCatalog(path, data_path=dp)
    tags = catalog.column_tags("multi", "col1")
    assert tags.shape[0] == 3
    tag_dict = dict(zip(tags["key"].to_list(), tags["value"].to_list()))
    assert tag_dict["comment"] == "main column"
    assert tag_dict["pii"] == "false"
    assert tag_dict["unit"] == "count"


def test_comment_on_table_with_data_via_duckdb(ducklake_catalog_sqlite):
    """DuckDB COMMENT after inserting data — comment and data both readable."""
    cat = ducklake_catalog_sqlite
    cat.execute("CREATE TABLE ducklake.withdata (a INTEGER, b VARCHAR)")
    cat.execute("INSERT INTO ducklake.withdata VALUES (1, 'hello'), (2, 'world')")
    cat.execute("COMMENT ON TABLE ducklake.withdata IS 'has data'")
    cat.execute("COMMENT ON COLUMN ducklake.withdata.a IS 'integer col'")
    cat.close()

    # Data intact
    result = read_ducklake(cat.metadata_path, "withdata", data_path=cat.data_path)
    result = result.sort("a")
    assert result["a"].to_list() == [1, 2]
    assert result["b"].to_list() == ["hello", "world"]

    # Comments intact
    catalog = DuckLakeCatalog(cat.metadata_path, data_path=cat.data_path)
    assert catalog.table_tags("withdata")["value"][0] == "has data"
    assert catalog.column_tags("withdata", "a")["value"][0] == "integer col"
