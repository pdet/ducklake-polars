"""Default value tests for ducklake-dataframe.

Tests covering CREATE TABLE with default values, default value interactions
with schema evolution, and struct field default-like behavior — matching
ducklake-ref test parity for default_values and struct_field_default.

Note: DuckLake currently only supports literal defaults (numbers and strings).
Boolean, date, timestamp, and expression-based defaults are NOT supported
by the DuckLake extension ("Only literals (e.g. 42 or 'hello world') are
supported as default values").
"""

from __future__ import annotations

import os

import duckdb
import polars as pl
import pytest

from ducklake_polars import read_ducklake, scan_ducklake


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_duckdb_catalog(tmp_path, *, inline: bool = False):
    """Create a DuckDB connection attached to a fresh ducklake catalog."""
    metadata_path = str(tmp_path / "defaults.ducklake")
    data_path = str(tmp_path / "data")
    os.makedirs(data_path, exist_ok=True)

    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")

    inline_opt = "" if inline else ", DATA_INLINING_ROW_LIMIT 0"
    con.execute(
        f"ATTACH 'ducklake:sqlite:{metadata_path}' AS ducklake "
        f"(DATA_PATH '{data_path}'{inline_opt})"
    )
    return con, metadata_path, data_path


# ===================================================================
# CREATE TABLE with default values (literal defaults only)
# ===================================================================


class TestCreateTableWithDefaults:
    """Test creating tables with DEFAULT on columns, then reading via polars."""

    def test_integer_default(self, ducklake_catalog):
        """CREATE TABLE with INTEGER column DEFAULT, insert without specifying it."""
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test (a INTEGER, b INTEGER DEFAULT 42)"
        )
        cat.execute("INSERT INTO ducklake.test (a) VALUES (1)")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 100)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result.shape == (2, 2)
        assert result.schema == {"a": pl.Int32, "b": pl.Int32}
        assert result["a"].to_list() == [1, 2]
        assert result["b"].to_list() == [42, 100]

    def test_string_default(self, ducklake_catalog):
        """CREATE TABLE with VARCHAR column DEFAULT."""
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test (id INTEGER, name VARCHAR DEFAULT 'unknown')"
        )
        cat.execute("INSERT INTO ducklake.test (id) VALUES (1)")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'alice')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        assert result["name"].to_list() == ["unknown", "alice"]

    def test_float_default(self, ducklake_catalog):
        """CREATE TABLE with DOUBLE column DEFAULT."""
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test (id INTEGER, score DOUBLE DEFAULT 0.0)"
        )
        cat.execute("INSERT INTO ducklake.test (id) VALUES (1)")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 3.14)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        assert result.schema["score"] == pl.Float64
        assert result["score"].to_list() == [0.0, 3.14]

    def test_negative_default(self, ducklake_catalog):
        """CREATE TABLE with negative numeric default."""
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test (id INTEGER, val INTEGER DEFAULT -1)"
        )
        cat.execute("INSERT INTO ducklake.test (id) VALUES (1)")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 50)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        assert result["val"].to_list() == [-1, 50]

    def test_multiple_columns_with_defaults(self, ducklake_catalog):
        """CREATE TABLE with multiple columns having literal defaults."""
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test ("
            "  id INTEGER,"
            "  status VARCHAR DEFAULT 'pending',"
            "  priority INTEGER DEFAULT 0,"
            "  score DOUBLE DEFAULT 1.0"
            ")"
        )
        cat.execute("INSERT INTO ducklake.test (id) VALUES (1)")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'done', 5, 9.9)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        assert result["status"].to_list() == ["pending", "done"]
        assert result["priority"].to_list() == [0, 5]
        assert result["score"].to_list() == [1.0, 9.9]

    def test_default_with_explicit_null(self, ducklake_catalog):
        """Inserting explicit NULL should override the default."""
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test (id INTEGER, val INTEGER DEFAULT 99)"
        )
        cat.execute("INSERT INTO ducklake.test VALUES (1, NULL)")
        cat.execute("INSERT INTO ducklake.test (id) VALUES (2)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        assert result["val"].to_list() == [None, 99]

    def test_zero_default(self, ducklake_catalog):
        """Zero as a default value (edge case: falsy but not NULL)."""
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test (id INTEGER, counter INTEGER DEFAULT 0)"
        )
        cat.execute("INSERT INTO ducklake.test (id) VALUES (1)")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 10)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        assert result["counter"].to_list() == [0, 10]

    def test_empty_string_default(self, ducklake_catalog):
        """Empty string as a default (edge case: falsy but not NULL)."""
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test (id INTEGER, tag VARCHAR DEFAULT '')"
        )
        cat.execute("INSERT INTO ducklake.test (id) VALUES (1)")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'important')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        assert result["tag"].to_list() == ["", "important"]


# ===================================================================
# Default values with schema evolution (ADD COLUMN with default)
# ===================================================================


class TestDefaultsSchemaEvolution:
    """Test default value interaction with schema evolution."""

    def test_add_column_with_default_then_insert(self, ducklake_catalog):
        """Add column with default, then insert rows."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), (2)")

        cat.execute("ALTER TABLE ducklake.test ADD COLUMN b INTEGER DEFAULT 42")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 100)")
        cat.execute("INSERT INTO ducklake.test (a) VALUES (4)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result.shape == (4, 2)
        # Old rows: column b not in Parquet -> NULL
        # Row 3: explicit 100
        # Row 4: used default 42
        assert result["b"].to_list() == [None, None, 100, 42]

    def test_add_multiple_columns_with_defaults(self, ducklake_catalog):
        """Add multiple columns with defaults sequentially."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")

        cat.execute(
            "ALTER TABLE ducklake.test ADD COLUMN status VARCHAR DEFAULT 'active'"
        )
        cat.execute(
            "ALTER TABLE ducklake.test ADD COLUMN count INTEGER DEFAULT 0"
        )
        cat.execute("INSERT INTO ducklake.test (id) VALUES (2)")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'closed', 5)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        # Row 1: pre-alter -> NULLs
        assert result["status"][0] is None
        assert result["count"][0] is None
        # Row 2: omitted -> defaults
        assert result["status"][1] == "active"
        assert result["count"][1] == 0
        # Row 3: explicit
        assert result["status"][2] == "closed"
        assert result["count"][2] == 5

    def test_add_column_default_preserves_old_data(self, ducklake_catalog):
        """Verify old data is intact after adding column with default."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'hello'), (2, 'world')"
        )

        cat.execute(
            "ALTER TABLE ducklake.test ADD COLUMN c DOUBLE DEFAULT 1.0"
        )
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'new', 2.5)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == ["hello", "world", "new"]
        assert result["c"].to_list() == [None, None, 2.5]

    def test_add_column_default_then_type_change(self, ducklake_catalog):
        """Add column with default, then promote its type."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")

        cat.execute(
            "ALTER TABLE ducklake.test ADD COLUMN val INTEGER DEFAULT 10"
        )
        cat.execute("INSERT INTO ducklake.test (id) VALUES (2)")

        cat.execute(
            "ALTER TABLE ducklake.test ALTER COLUMN val SET DATA TYPE BIGINT"
        )
        cat.execute("INSERT INTO ducklake.test VALUES (3, 9999999999)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        assert result.schema["val"] == pl.Int64
        # Row 1: pre-alter -> NULL
        assert result["val"][0] is None
        # Row 2: default 10
        assert result["val"][1] == 10
        # Row 3: explicit large value
        assert result["val"][2] == 9999999999

    def test_add_column_default_then_drop_then_readd(self, ducklake_catalog):
        """Add column with default, drop it, re-add with different default."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")

        cat.execute(
            "ALTER TABLE ducklake.test ADD COLUMN x INTEGER DEFAULT 10"
        )
        cat.execute("INSERT INTO ducklake.test (id) VALUES (2)")

        cat.execute("ALTER TABLE ducklake.test DROP COLUMN x")
        cat.execute("INSERT INTO ducklake.test VALUES (3)")

        cat.execute(
            "ALTER TABLE ducklake.test ADD COLUMN x VARCHAR DEFAULT 'new'"
        )
        cat.execute("INSERT INTO ducklake.test (id) VALUES (4)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        assert result.columns == ["id", "x"]
        assert result.schema["x"] == pl.String
        # Rows before re-add get NULL; row 4 gets new default
        assert result["x"].to_list() == [None, None, None, "new"]


# ===================================================================
# Default values preserved after rename column
# ===================================================================


class TestDefaultsAfterRename:
    """Test that default values survive column renames."""

    def test_default_preserved_after_rename(self, ducklake_catalog):
        """Rename a column that has a default — default should still work."""
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test (id INTEGER, val INTEGER DEFAULT 99)"
        )
        cat.execute("INSERT INTO ducklake.test (id) VALUES (1)")

        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN val TO value")
        cat.execute("INSERT INTO ducklake.test (id) VALUES (2)")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 50)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        assert result.columns == ["id", "value"]
        assert result["value"][0] == 99
        assert result["value"][1] == 99
        assert result["value"][2] == 50

    def test_added_column_default_preserved_after_rename(
        self, ducklake_catalog
    ):
        """Add column with default, rename it, then insert — default still works."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")

        cat.execute(
            "ALTER TABLE ducklake.test ADD COLUMN x INTEGER DEFAULT 10"
        )
        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN x TO y")
        cat.execute("INSERT INTO ducklake.test (id) VALUES (2)")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 77)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        assert result.columns == ["id", "y"]
        assert result["y"][0] is None  # pre-alter
        assert result["y"][1] == 10    # default after rename
        assert result["y"][2] == 77    # explicit

    def test_string_default_preserved_after_rename(self, ducklake_catalog):
        """String default survives rename."""
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test (id INTEGER, label VARCHAR DEFAULT 'todo')"
        )
        cat.execute("INSERT INTO ducklake.test (id) VALUES (1)")

        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN label TO tag")
        cat.execute("INSERT INTO ducklake.test (id) VALUES (2)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        assert result.columns == ["id", "tag"]
        assert result["tag"].to_list() == ["todo", "todo"]


# ===================================================================
# Default values with inlined data
# ===================================================================


class TestDefaultsInlined:
    """Test default values when data is inlined (stored in metadata DB, not Parquet)."""

    def test_create_table_defaults_with_inlining(self, ducklake_catalog_inline):
        """Defaults work correctly when data inlining is enabled."""
        cat = ducklake_catalog_inline
        cat.execute(
            "CREATE TABLE ducklake.test (id INTEGER, val INTEGER DEFAULT 5)"
        )
        cat.execute("INSERT INTO ducklake.test (id) VALUES (1)")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 20)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        assert result["val"].to_list() == [5, 20]

    def test_add_column_default_with_inlining(self, ducklake_catalog_inline):
        """ADD COLUMN with default works with inlined data."""
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (id INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")

        cat.execute(
            "ALTER TABLE ducklake.test ADD COLUMN tag VARCHAR DEFAULT 'none'"
        )
        cat.execute("INSERT INTO ducklake.test (id) VALUES (2)")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'special')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        assert result["tag"][0] is None     # pre-alter
        assert result["tag"][1] == "none"   # default
        assert result["tag"][2] == "special"  # explicit

    def test_string_default_inlined(self, ducklake_catalog_inline):
        """String default with inlining."""
        cat = ducklake_catalog_inline
        cat.execute(
            "CREATE TABLE ducklake.test (id INTEGER, name VARCHAR DEFAULT 'anon')"
        )
        cat.execute("INSERT INTO ducklake.test (id) VALUES (1)")
        cat.execute("INSERT INTO ducklake.test (id) VALUES (2)")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'alice')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        assert result["name"].to_list() == ["anon", "anon", "alice"]


# ===================================================================
# Interop: DuckDB writes with defaults, polars reads
# ===================================================================


class TestDefaultsInterop:
    """DuckDB creates tables with defaults -> ducklake-dataframe reads them."""

    def test_duckdb_create_with_int_default_polars_reads(self, tmp_path):
        """Create table with integer default via DuckDB, read via polars."""
        con, metadata_path, data_path = _make_duckdb_catalog(tmp_path)

        con.execute(
            "CREATE TABLE ducklake.test ("
            "  id INTEGER,"
            "  name VARCHAR DEFAULT 'anon',"
            "  score DOUBLE DEFAULT 0.0"
            ")"
        )
        con.execute("INSERT INTO ducklake.test (id) VALUES (1)")
        con.execute(
            "INSERT INTO ducklake.test VALUES (2, 'alice', 95.5)"
        )
        con.close()

        result = read_ducklake(metadata_path, "test")
        result = result.sort("id")
        assert result["name"].to_list() == ["anon", "alice"]
        assert result["score"].to_list() == [0.0, 95.5]

    def test_duckdb_add_column_default_polars_reads(self, tmp_path):
        """DuckDB adds column with default, polars reads correctly."""
        con, metadata_path, data_path = _make_duckdb_catalog(tmp_path)

        con.execute("CREATE TABLE ducklake.test (id INTEGER)")
        con.execute("INSERT INTO ducklake.test VALUES (1), (2)")
        con.execute(
            "ALTER TABLE ducklake.test ADD COLUMN val INTEGER DEFAULT 99"
        )
        con.execute("INSERT INTO ducklake.test (id) VALUES (3)")
        con.execute("INSERT INTO ducklake.test VALUES (4, 200)")
        con.close()

        result = read_ducklake(metadata_path, "test")
        result = result.sort("id")
        assert result["val"].to_list() == [None, None, 99, 200]

    def test_duckdb_defaults_with_inlining_polars_reads(self, tmp_path):
        """DuckDB creates table with defaults and inlined data, polars reads."""
        con, metadata_path, data_path = _make_duckdb_catalog(
            tmp_path, inline=True
        )

        con.execute(
            "CREATE TABLE ducklake.test (id INTEGER, val VARCHAR DEFAULT 'hi')"
        )
        con.execute("INSERT INTO ducklake.test (id) VALUES (1)")
        con.execute("INSERT INTO ducklake.test VALUES (2, 'bye')")
        con.close()

        result = read_ducklake(metadata_path, "test")
        result = result.sort("id")
        assert result["val"].to_list() == ["hi", "bye"]

    def test_duckdb_rename_default_column_polars_reads(self, tmp_path):
        """DuckDB renames a column with default, polars reads correctly."""
        con, metadata_path, data_path = _make_duckdb_catalog(tmp_path)

        con.execute(
            "CREATE TABLE ducklake.test (id INTEGER, val INTEGER DEFAULT 7)"
        )
        con.execute("INSERT INTO ducklake.test (id) VALUES (1)")
        con.execute("ALTER TABLE ducklake.test RENAME COLUMN val TO value")
        con.execute("INSERT INTO ducklake.test (id) VALUES (2)")
        con.close()

        result = read_ducklake(metadata_path, "test")
        result = result.sort("id")
        assert result.columns == ["id", "value"]
        assert result["value"].to_list() == [7, 7]


# ===================================================================
# Default values and time travel
# ===================================================================


class TestDefaultsTimeTravel:
    """Test that default values interact correctly with time travel."""

    def test_defaults_at_different_snapshots(self, ducklake_catalog):
        """Read at snapshot before and after add-column-with-default."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (id INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        snap_v1 = cat.fetchone(
            "SELECT * FROM ducklake_current_snapshot('ducklake')"
        )[0]

        cat.execute(
            "ALTER TABLE ducklake.test ADD COLUMN val INTEGER DEFAULT 42"
        )
        cat.execute("INSERT INTO ducklake.test (id) VALUES (2)")
        cat.close()

        # At v1: only id column
        r1 = read_ducklake(cat.metadata_path, "test", snapshot_version=snap_v1)
        assert r1.columns == ["id"]
        assert r1["id"].to_list() == [1]

        # At latest: id + val
        r2 = read_ducklake(cat.metadata_path, "test")
        r2 = r2.sort("id")
        assert r2.columns == ["id", "val"]
        assert r2["val"].to_list() == [None, 42]

    def test_create_table_default_time_travel(self, ducklake_catalog):
        """Time travel with defaults set at CREATE TABLE time."""
        cat = ducklake_catalog

        cat.execute(
            "CREATE TABLE ducklake.test (id INTEGER, val INTEGER DEFAULT 5)"
        )
        cat.execute("INSERT INTO ducklake.test (id) VALUES (1)")
        snap_v1 = cat.fetchone(
            "SELECT * FROM ducklake_current_snapshot('ducklake')"
        )[0]

        cat.execute("INSERT INTO ducklake.test VALUES (2, 20)")
        cat.close()

        r1 = read_ducklake(cat.metadata_path, "test", snapshot_version=snap_v1)
        assert r1.shape == (1, 2)
        assert r1["val"].to_list() == [5]

        r2 = read_ducklake(cat.metadata_path, "test")
        r2 = r2.sort("id")
        assert r2["val"].to_list() == [5, 20]


# ===================================================================
# Default values with delete and filter
# ===================================================================


class TestDefaultsWithOperations:
    """Test defaults interacting with delete and filter pushdown."""

    def test_default_column_with_delete(self, ducklake_catalog):
        """Delete from table with default column."""
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test (id INTEGER, val INTEGER DEFAULT 10)"
        )
        cat.execute("INSERT INTO ducklake.test (id) VALUES (1)")
        cat.execute("INSERT INTO ducklake.test (id) VALUES (2)")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 30)")

        cat.execute("DELETE FROM ducklake.test WHERE id = 2")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        assert result["id"].to_list() == [1, 3]
        assert result["val"].to_list() == [10, 30]

    def test_default_column_filter_pushdown(self, ducklake_catalog):
        """Filter pushdown works on columns with default values."""
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test (id INTEGER, val INTEGER DEFAULT 10)"
        )
        cat.execute("INSERT INTO ducklake.test (id) VALUES (1)")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 20)")
        cat.execute("INSERT INTO ducklake.test (id) VALUES (3)")
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.filter(pl.col("val") == 10).collect().sort("id")
        assert result["id"].to_list() == [1, 3]
        assert result["val"].to_list() == [10, 10]

    def test_default_column_with_update(self, ducklake_catalog):
        """Update rows in table with default column."""
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test (id INTEGER, val INTEGER DEFAULT 10)"
        )
        cat.execute("INSERT INTO ducklake.test (id) VALUES (1)")
        cat.execute("INSERT INTO ducklake.test (id) VALUES (2)")

        cat.execute("UPDATE ducklake.test SET val = 99 WHERE id = 1")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        assert result["val"].to_list() == [99, 10]


# ===================================================================
# Struct field defaults (struct with nested default values)
# ===================================================================


class TestStructFieldDefault:
    """Test struct columns where fields exhibit default-like behavior."""

    def test_struct_column_with_null_fields(self, ducklake_catalog):
        """Struct column where some fields are NULL (partial struct)."""
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test ("
            "  id INTEGER,"
            "  info STRUCT(name VARCHAR, age INTEGER)"
            ")"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, {name: 'alice', age: 30})"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES (2, {name: 'bob', age: NULL})"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        structs = result["info"].to_list()
        assert structs[0] == {"name": "alice", "age": 30}
        assert structs[1] == {"name": "bob", "age": None}

    def test_struct_field_added_via_evolution(self, ducklake_catalog):
        """Add a new field to a struct — old rows have NULL for the new field."""
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test ("
            "  id INTEGER,"
            "  info STRUCT(x INTEGER, y INTEGER)"
            ")"
        )
        cat.execute("INSERT INTO ducklake.test VALUES (1, {x: 10, y: 20})")

        cat.execute(
            "ALTER TABLE ducklake.test ALTER COLUMN info "
            "SET DATA TYPE STRUCT(x INTEGER, y INTEGER, z INTEGER)"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES (2, {x: 30, y: 40, z: 50})"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        structs = result["info"].to_list()
        assert structs[0] == {"x": 10, "y": 20, "z": None}
        assert structs[1] == {"x": 30, "y": 40, "z": 50}

    def test_null_struct(self, ducklake_catalog):
        """Struct column that is entirely NULL."""
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test ("
            "  id INTEGER,"
            "  info STRUCT(a INTEGER, b VARCHAR)"
            ")"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, {a: 1, b: 'hello'})"
        )
        cat.execute("INSERT INTO ducklake.test VALUES (2, NULL)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        structs = result["info"].to_list()
        assert structs[0] == {"a": 1, "b": "hello"}
        # A NULL struct is None, not {"a": None, "b": None}
        assert structs[1] is None

    def test_struct_field_evolution_with_rename(self, ducklake_catalog):
        """Struct field rename — old data accessible under new field name."""
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test ("
            "  id INTEGER,"
            "  info STRUCT(i INTEGER, j INTEGER)"
            ")"
        )
        cat.execute("INSERT INTO ducklake.test VALUES (1, {i: 10, j: 20})")

        cat.execute(
            "ALTER TABLE ducklake.test ALTER COLUMN info "
            "SET DATA TYPE STRUCT(i INTEGER, val INTEGER)"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES (2, {i: 30, val: 40})"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        structs = result["info"].to_list()
        assert structs[0] == {"i": 10, "val": 20}
        assert structs[1] == {"i": 30, "val": 40}
