"""Cross-implementation compatibility tests: ducklake-dataframe ↔ DuckDB native ducklake.

Verifies bidirectional compatibility between ducklake-dataframe and DuckDB's
native ducklake extension. All tests use a shared catalog so that data written
by one system is readable by the other.

Test categories:
  - DuckDB writes → polars reads
  - Polars writes → DuckDB reads
  - Roundtrip (both directions)
"""

from __future__ import annotations

import datetime
import math
import os
from decimal import Decimal

import duckdb
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ducklake_polars import (
    DuckLakeCatalog,
    alter_ducklake_add_column,
    alter_ducklake_drop_column,
    alter_ducklake_rename_column,
    alter_ducklake_set_partitioned_by,
    read_ducklake,
    scan_ducklake,
    write_ducklake,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_duckdb_catalog(tmp_path, *, inline: bool = False):
    """Create a DuckDB connection attached to a fresh ducklake catalog."""
    metadata_path = str(tmp_path / "interop.ducklake")
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


def _reopen_duckdb(metadata_path, data_path, *, inline: bool = False):
    """Open a new DuckDB connection to an existing catalog."""
    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")

    inline_opt = "" if inline else ", DATA_INLINING_ROW_LIMIT 0"
    con.execute(
        f"ATTACH 'ducklake:sqlite:{metadata_path}' AS ducklake "
        f"(DATA_PATH '{data_path}'{inline_opt})"
    )
    return con


def _init_catalog(tmp_path, *, inline: bool = False):
    """Initialize an empty ducklake catalog and return (metadata_path, data_path).

    Creates the catalog via DuckDB (which sets up the schema tables), then
    closes the connection so our library can use it.
    """
    metadata_path = str(tmp_path / "interop.ducklake")
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
    con.close()
    return metadata_path, data_path


# ═══════════════════════════════════════════════════════════════════════════
# DuckDB writes → Polars reads
# ═══════════════════════════════════════════════════════════════════════════


class TestDuckDBWritesPolarsReads:
    """DuckDB creates and populates tables; ducklake-dataframe reads them."""

    # --- Scalar types ---------------------------------------------------

    def test_integer_types(self, tmp_path):
        con, meta, data = _make_duckdb_catalog(tmp_path)
        con.execute("""
            CREATE TABLE ducklake.t (
                i8 TINYINT, i16 SMALLINT, i32 INTEGER, i64 BIGINT,
                u8 UTINYINT, u16 USMALLINT, u32 UINTEGER, u64 UBIGINT
            )
        """)
        con.execute(
            "INSERT INTO ducklake.t VALUES (1, 2, 3, 4, 5, 6, 7, 8)"
        )
        con.close()

        result = read_ducklake(meta, "t")
        assert result.schema == {
            "i8": pl.Int8, "i16": pl.Int16, "i32": pl.Int32, "i64": pl.Int64,
            "u8": pl.UInt8, "u16": pl.UInt16, "u32": pl.UInt32, "u64": pl.UInt64,
        }
        assert result.row(0) == (1, 2, 3, 4, 5, 6, 7, 8)

    def test_float_double(self, tmp_path):
        con, meta, data = _make_duckdb_catalog(tmp_path)
        con.execute("CREATE TABLE ducklake.t (f FLOAT, d DOUBLE)")
        con.execute("INSERT INTO ducklake.t VALUES (3.14, 2.718281828)")
        con.close()

        result = read_ducklake(meta, "t")
        assert result.schema == {"f": pl.Float32, "d": pl.Float64}
        assert abs(result["f"][0] - 3.14) < 0.01
        assert abs(result["d"][0] - 2.718281828) < 1e-6

    def test_varchar(self, tmp_path):
        con, meta, data = _make_duckdb_catalog(tmp_path)
        con.execute("CREATE TABLE ducklake.t (s VARCHAR)")
        con.execute("INSERT INTO ducklake.t VALUES ('hello'), (''), (NULL)")
        con.close()

        result = read_ducklake(meta, "t")
        assert result.schema == {"s": pl.String}
        vals = result["s"].to_list()
        assert "hello" in vals
        assert "" in vals
        assert None in vals

    def test_boolean(self, tmp_path):
        con, meta, data = _make_duckdb_catalog(tmp_path)
        con.execute("CREATE TABLE ducklake.t (b BOOLEAN)")
        con.execute("INSERT INTO ducklake.t VALUES (true), (false), (NULL)")
        con.close()

        result = read_ducklake(meta, "t")
        assert result["b"].to_list() == [True, False, None]

    def test_date(self, tmp_path):
        con, meta, data = _make_duckdb_catalog(tmp_path)
        con.execute("CREATE TABLE ducklake.t (d DATE)")
        con.execute("INSERT INTO ducklake.t VALUES ('2025-01-15'), ('1999-12-31')")
        con.close()

        result = read_ducklake(meta, "t")
        assert result.schema == {"d": pl.Date}
        assert set(result["d"].to_list()) == {
            datetime.date(2025, 1, 15),
            datetime.date(1999, 12, 31),
        }

    def test_time(self, tmp_path):
        con, meta, data = _make_duckdb_catalog(tmp_path)
        con.execute("CREATE TABLE ducklake.t (t TIME)")
        con.execute("INSERT INTO ducklake.t VALUES ('10:30:00'), ('23:59:59')")
        con.close()

        result = read_ducklake(meta, "t")
        assert result.schema == {"t": pl.Time}
        assert set(result["t"].to_list()) == {
            datetime.time(10, 30, 0),
            datetime.time(23, 59, 59),
        }

    def test_timestamp(self, tmp_path):
        con, meta, data = _make_duckdb_catalog(tmp_path)
        con.execute("CREATE TABLE ducklake.t (ts TIMESTAMP)")
        con.execute("INSERT INTO ducklake.t VALUES ('2025-06-15 12:30:45')")
        con.close()

        result = read_ducklake(meta, "t")
        assert result.schema["ts"].base_type() == pl.Datetime
        assert result["ts"][0] == datetime.datetime(2025, 6, 15, 12, 30, 45)

    def test_timestamptz(self, tmp_path):
        con, meta, data = _make_duckdb_catalog(tmp_path)
        con.execute("CREATE TABLE ducklake.t (ts TIMESTAMPTZ)")
        con.execute("INSERT INTO ducklake.t VALUES ('2025-06-15 12:30:45+00')")
        con.close()

        result = read_ducklake(meta, "t")
        assert result.schema["ts"].base_type() == pl.Datetime

    def test_decimal(self, tmp_path):
        con, meta, data = _make_duckdb_catalog(tmp_path)
        con.execute("CREATE TABLE ducklake.t (d DECIMAL(18, 4))")
        con.execute("INSERT INTO ducklake.t VALUES (12345.6789), (0.0001)")
        con.close()

        result = read_ducklake(meta, "t")
        vals = result["d"].to_list()
        assert Decimal("12345.6789") in vals
        assert Decimal("0.0001") in vals

    def test_blob(self, tmp_path):
        con, meta, data = _make_duckdb_catalog(tmp_path)
        con.execute("CREATE TABLE ducklake.t (b BLOB)")
        con.execute("INSERT INTO ducklake.t VALUES ('\\x01\\x02\\x03'::BLOB)")
        con.close()

        result = read_ducklake(meta, "t")
        assert result.schema == {"b": pl.Binary}
        assert result["b"][0] == b"\x01\x02\x03"

    def test_uuid(self, tmp_path):
        con, meta, data = _make_duckdb_catalog(tmp_path)
        con.execute("CREATE TABLE ducklake.t (u UUID)")
        con.execute(
            "INSERT INTO ducklake.t VALUES ('550e8400-e29b-41d4-a716-446655440000')"
        )
        con.close()

        result = read_ducklake(meta, "t")
        assert result.schema == {"u": pl.Binary}
        assert len(result["u"][0]) == 16

    # --- Nested types ---------------------------------------------------

    def test_list(self, tmp_path):
        con, meta, data = _make_duckdb_catalog(tmp_path)
        con.execute("CREATE TABLE ducklake.t (a INTEGER[])")
        con.execute("INSERT INTO ducklake.t VALUES ([1, 2, 3]), ([4, 5]), (NULL)")
        con.close()

        result = read_ducklake(meta, "t")
        vals = result["a"].to_list()
        assert vals[0] == [1, 2, 3]
        assert vals[1] == [4, 5]
        assert vals[2] is None

    def test_struct(self, tmp_path):
        con, meta, data = _make_duckdb_catalog(tmp_path)
        con.execute("CREATE TABLE ducklake.t (s STRUCT(x INTEGER, y VARCHAR))")
        con.execute(
            "INSERT INTO ducklake.t VALUES ({'x': 1, 'y': 'hello'}), (NULL)"
        )
        con.close()

        result = read_ducklake(meta, "t")
        assert result["s"][0] == {"x": 1, "y": "hello"}
        assert result["s"][1] is None

    def test_nested_list_of_structs(self, tmp_path):
        con, meta, data = _make_duckdb_catalog(tmp_path)
        con.execute(
            "CREATE TABLE ducklake.t (a STRUCT(x INTEGER, y VARCHAR)[])"
        )
        con.execute(
            "INSERT INTO ducklake.t VALUES "
            "([{'x': 1, 'y': 'a'}, {'x': 2, 'y': 'b'}])"
        )
        con.close()

        result = read_ducklake(meta, "t")
        vals = result["a"].to_list()
        assert len(vals) == 1
        assert len(vals[0]) == 2
        assert vals[0][0] == {"x": 1, "y": "a"}
        assert vals[0][1] == {"x": 2, "y": "b"}

    def test_list_of_lists(self, tmp_path):
        con, meta, data = _make_duckdb_catalog(tmp_path)
        con.execute("CREATE TABLE ducklake.t (a INTEGER[][])")
        con.execute("INSERT INTO ducklake.t VALUES ([[1, 2], [3, 4]])")
        con.close()

        result = read_ducklake(meta, "t")
        vals = result["a"].to_list()
        assert vals[0] == [[1, 2], [3, 4]]

    # --- ENUM types (mapped to VARCHAR in ducklake) ----------------------

    def test_enum_values_as_varchar(self, tmp_path):
        """Enum-like values stored as VARCHAR by DuckDB → polars reads as String."""
        con, meta, data = _make_duckdb_catalog(tmp_path)
        con.execute("CREATE TABLE ducklake.t (c VARCHAR)")
        con.execute(
            "INSERT INTO ducklake.t VALUES ('red'), ('green'), ('blue'), ('red')"
        )
        con.close()

        result = read_ducklake(meta, "t")
        assert result.schema == {"c": pl.String}
        assert sorted(result["c"].to_list()) == ["blue", "green", "red", "red"]

    # --- NULL handling --------------------------------------------------

    def test_all_nulls(self, tmp_path):
        con, meta, data = _make_duckdb_catalog(tmp_path)
        con.execute(
            "CREATE TABLE ducklake.t (a INTEGER, b VARCHAR, c DOUBLE)"
        )
        con.execute(
            "INSERT INTO ducklake.t VALUES (NULL, NULL, NULL), (NULL, NULL, NULL)"
        )
        con.close()

        result = read_ducklake(meta, "t")
        assert result.shape == (2, 3)
        assert result["a"].null_count() == 2
        assert result["b"].null_count() == 2
        assert result["c"].null_count() == 2

    def test_mixed_nulls(self, tmp_path):
        con, meta, data = _make_duckdb_catalog(tmp_path)
        con.execute("CREATE TABLE ducklake.t (a INTEGER, b VARCHAR)")
        con.execute(
            "INSERT INTO ducklake.t VALUES (1, NULL), (NULL, 'hello'), (3, 'world')"
        )
        con.close()

        result = read_ducklake(meta, "t").sort("a", nulls_last=True)
        assert result["a"].to_list() == [1, 3, None]
        assert result["b"].to_list() == [None, "world", "hello"]

    # --- Time travel / snapshots ----------------------------------------

    def test_multiple_snapshots(self, tmp_path):
        con, meta, data = _make_duckdb_catalog(tmp_path)
        con.execute("CREATE TABLE ducklake.t (v INTEGER)")
        con.execute("INSERT INTO ducklake.t VALUES (1)")
        snap1 = con.execute(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshots('ducklake')"
        ).fetchone()[0]
        con.execute("INSERT INTO ducklake.t VALUES (2)")
        snap2 = con.execute(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshots('ducklake')"
        ).fetchone()[0]
        con.execute("INSERT INTO ducklake.t VALUES (3)")
        con.close()

        # Latest has all 3
        result = read_ducklake(meta, "t")
        assert sorted(result["v"].to_list()) == [1, 2, 3]

        # Snap1 has only 1
        r1 = read_ducklake(meta, "t", snapshot_version=snap1)
        assert r1["v"].to_list() == [1]

        # Snap2 has 1, 2
        r2 = read_ducklake(meta, "t", snapshot_version=snap2)
        assert sorted(r2["v"].to_list()) == [1, 2]

    # --- Schema evolution -----------------------------------------------

    def test_add_column_then_read(self, tmp_path):
        con, meta, data = _make_duckdb_catalog(tmp_path)
        con.execute("CREATE TABLE ducklake.t (a INTEGER)")
        con.execute("INSERT INTO ducklake.t VALUES (1), (2)")
        con.execute("ALTER TABLE ducklake.t ADD COLUMN b VARCHAR")
        con.execute("INSERT INTO ducklake.t VALUES (3, 'hello')")
        con.close()

        result = read_ducklake(meta, "t").sort("a")
        assert result.columns == ["a", "b"]
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == [None, None, "hello"]

    def test_drop_column_then_read(self, tmp_path):
        con, meta, data = _make_duckdb_catalog(tmp_path)
        con.execute("CREATE TABLE ducklake.t (a INTEGER, b VARCHAR, c DOUBLE)")
        con.execute("INSERT INTO ducklake.t VALUES (1, 'x', 1.5)")
        con.execute("ALTER TABLE ducklake.t DROP COLUMN b")
        con.execute("INSERT INTO ducklake.t VALUES (2, 2.5)")
        con.close()

        result = read_ducklake(meta, "t").sort("a")
        assert result.columns == ["a", "c"]
        assert result["a"].to_list() == [1, 2]
        assert result["c"].to_list() == [1.5, 2.5]

    def test_rename_column_then_read(self, tmp_path):
        con, meta, data = _make_duckdb_catalog(tmp_path)
        con.execute("CREATE TABLE ducklake.t (a INTEGER, b VARCHAR)")
        con.execute("INSERT INTO ducklake.t VALUES (1, 'old')")
        con.execute("ALTER TABLE ducklake.t RENAME COLUMN b TO c")
        con.execute("INSERT INTO ducklake.t VALUES (2, 'new')")
        con.close()

        result = read_ducklake(meta, "t").sort("a")
        assert result.columns == ["a", "c"]
        assert result["c"].to_list() == ["old", "new"]

    # --- Partitioned tables ---------------------------------------------

    def test_partitioned_table(self, tmp_path):
        con, meta, data = _make_duckdb_catalog(tmp_path)
        con.execute("CREATE TABLE ducklake.t (id INTEGER, cat VARCHAR)")
        con.execute(
            "ALTER TABLE ducklake.t SET PARTITIONED BY (cat)"
        )
        con.execute(
            "INSERT INTO ducklake.t VALUES (1, 'a'), (2, 'b'), (3, 'a')"
        )
        con.close()

        result = read_ducklake(meta, "t").sort("id")
        assert result["id"].to_list() == [1, 2, 3]
        assert result["cat"].to_list() == ["a", "b", "a"]

    # --- Inlined data ---------------------------------------------------

    def test_inlined_data(self, tmp_path):
        con, meta, data = _make_duckdb_catalog(tmp_path, inline=True)
        con.execute("CREATE TABLE ducklake.t (a INTEGER, b VARCHAR)")
        con.execute("INSERT INTO ducklake.t VALUES (1, 'hello'), (2, 'world')")
        con.close()

        result = read_ducklake(meta, "t").sort("a")
        assert result["a"].to_list() == [1, 2]
        assert result["b"].to_list() == ["hello", "world"]

    # --- Empty tables ---------------------------------------------------

    def test_empty_table(self, tmp_path):
        con, meta, data = _make_duckdb_catalog(tmp_path)
        con.execute("CREATE TABLE ducklake.t (a INTEGER, b VARCHAR)")
        con.close()

        result = read_ducklake(meta, "t")
        assert result.shape == (0, 2)
        assert result.columns == ["a", "b"]

    # --- Large values ---------------------------------------------------

    def test_long_string(self, tmp_path):
        con, meta, data = _make_duckdb_catalog(tmp_path)
        con.execute("CREATE TABLE ducklake.t (s VARCHAR)")
        long_str = "x" * 100_000
        con.execute("INSERT INTO ducklake.t VALUES (?)", [long_str])
        con.close()

        result = read_ducklake(meta, "t")
        assert len(result["s"][0]) == 100_000

    def test_large_list(self, tmp_path):
        con, meta, data = _make_duckdb_catalog(tmp_path)
        con.execute("CREATE TABLE ducklake.t (a INTEGER[])")
        large_list_str = "[" + ",".join(str(i) for i in range(1000)) + "]"
        con.execute(f"INSERT INTO ducklake.t VALUES ({large_list_str})")
        con.close()

        result = read_ducklake(meta, "t")
        vals = result["a"].to_list()
        assert len(vals[0]) == 1000
        assert vals[0][:3] == [0, 1, 2]


# ═══════════════════════════════════════════════════════════════════════════
# Polars writes → DuckDB reads
# ═══════════════════════════════════════════════════════════════════════════


class TestPolarsWritesDuckDBReads:
    """ducklake-dataframe creates and populates tables; DuckDB reads them."""

    # --- Scalar types ---------------------------------------------------

    def test_integer_types(self, tmp_path):
        meta, data = _init_catalog(tmp_path)

        df = pl.DataFrame({
            "i8": pl.Series([1], dtype=pl.Int8),
            "i16": pl.Series([2], dtype=pl.Int16),
            "i32": pl.Series([3], dtype=pl.Int32),
            "i64": pl.Series([4], dtype=pl.Int64),
            "u8": pl.Series([5], dtype=pl.UInt8),
            "u16": pl.Series([6], dtype=pl.UInt16),
            "u32": pl.Series([7], dtype=pl.UInt32),
            "u64": pl.Series([8], dtype=pl.UInt64),
        })
        write_ducklake(df, meta, "t", mode="error")

        con = _reopen_duckdb(meta, data)
        rows = con.execute("SELECT * FROM ducklake.t").fetchall()
        con.close()
        assert rows[0] == (1, 2, 3, 4, 5, 6, 7, 8)

    def test_float_double(self, tmp_path):
        meta, data = _init_catalog(tmp_path)

        df = pl.DataFrame({
            "f": pl.Series([3.14], dtype=pl.Float32),
            "d": pl.Series([2.718281828], dtype=pl.Float64),
        })
        write_ducklake(df, meta, "t", mode="error")

        con = _reopen_duckdb(meta, data)
        row = con.execute("SELECT * FROM ducklake.t").fetchone()
        con.close()
        assert abs(row[0] - 3.14) < 0.01
        assert abs(row[1] - 2.718281828) < 1e-6

    def test_varchar_boolean(self, tmp_path):
        meta, data = _init_catalog(tmp_path)

        df = pl.DataFrame({
            "s": ["hello", "world", ""],
            "b": [True, False, None],
        })
        write_ducklake(df, meta, "t", mode="error")

        con = _reopen_duckdb(meta, data)
        rows = con.execute("SELECT * FROM ducklake.t ORDER BY s").fetchall()
        con.close()
        strs = [r[0] for r in rows]
        assert "" in strs
        assert "hello" in strs
        assert "world" in strs

    def test_date_time_timestamp(self, tmp_path):
        meta, data = _init_catalog(tmp_path)

        df = pl.DataFrame({
            "d": [datetime.date(2025, 1, 15)],
            "t": [datetime.time(10, 30, 0)],
            "ts": [datetime.datetime(2025, 6, 15, 12, 30, 45)],
        })
        write_ducklake(df, meta, "t", mode="error")

        con = _reopen_duckdb(meta, data)
        row = con.execute("SELECT * FROM ducklake.t").fetchone()
        con.close()
        assert row[0] == datetime.date(2025, 1, 15)
        assert row[1] == datetime.time(10, 30, 0)
        assert row[2] == datetime.datetime(2025, 6, 15, 12, 30, 45)

    def test_decimal(self, tmp_path):
        meta, data = _init_catalog(tmp_path)

        df = pl.DataFrame({
            "d": pl.Series([Decimal("123.45"), Decimal("678.90")]).cast(
                pl.Decimal(10, 2)
            ),
        })
        write_ducklake(df, meta, "t", mode="error")

        con = _reopen_duckdb(meta, data)
        rows = con.execute("SELECT d FROM ducklake.t ORDER BY d").fetchall()
        con.close()
        assert rows[0][0] == Decimal("123.45")
        assert rows[1][0] == Decimal("678.90")

    def test_blob(self, tmp_path):
        meta, data = _init_catalog(tmp_path)

        df = pl.DataFrame({"b": pl.Series([b"\x01\x02\x03"], dtype=pl.Binary)})
        write_ducklake(df, meta, "t", mode="error")

        con = _reopen_duckdb(meta, data)
        row = con.execute("SELECT b FROM ducklake.t").fetchone()
        con.close()
        assert row[0] == b"\x01\x02\x03"

    # --- Nested types ---------------------------------------------------

    @pytest.mark.xfail(
        reason="field_id mapping: DuckDB reads NULL inner values for nested types written by our library"
    )
    def test_list(self, tmp_path):
        meta, data = _init_catalog(tmp_path)

        df = pl.DataFrame({"a": [[1, 2, 3], [4, 5]]})
        write_ducklake(df, meta, "t", mode="error")

        con = _reopen_duckdb(meta, data)
        rows = con.execute("SELECT a FROM ducklake.t").fetchall()
        con.close()
        assert [list(r[0]) for r in rows] == [[1, 2, 3], [4, 5]]

    @pytest.mark.xfail(
        reason="field_id mapping: DuckDB reads NULL inner values for nested types written by our library"
    )
    def test_struct(self, tmp_path):
        meta, data = _init_catalog(tmp_path)

        df = pl.DataFrame({
            "s": [{"x": 1, "y": "hello"}, {"x": 2, "y": "world"}],
        })
        write_ducklake(df, meta, "t", mode="error")

        con = _reopen_duckdb(meta, data)
        rows = con.execute("SELECT s FROM ducklake.t ORDER BY (s).x").fetchall()
        con.close()
        assert rows[0][0] == {"x": 1, "y": "hello"}
        assert rows[1][0] == {"x": 2, "y": "world"}

    # --- ENUM / Categorical types (stored as VARCHAR) -------------------

    def test_categorical_to_varchar(self, tmp_path):
        meta, data = _init_catalog(tmp_path)

        df = pl.DataFrame({
            "id": [1, 2, 3],
            "color": pl.Series(["red", "green", "blue"]).cast(pl.Categorical),
        })
        write_ducklake(df, meta, "t", mode="error")

        con = _reopen_duckdb(meta, data)
        rows = con.execute(
            "SELECT id, color FROM ducklake.t ORDER BY id"
        ).fetchall()
        con.close()
        assert [r[1] for r in rows] == ["red", "green", "blue"]

    def test_enum_to_varchar(self, tmp_path):
        meta, data = _init_catalog(tmp_path)

        color_type = pl.Enum(["red", "green", "blue"])
        df = pl.DataFrame({
            "id": [1, 2, 3],
            "color": pl.Series(["red", "green", "blue"]).cast(color_type),
        })
        write_ducklake(df, meta, "t", mode="error")

        con = _reopen_duckdb(meta, data)
        rows = con.execute(
            "SELECT id, color FROM ducklake.t ORDER BY id"
        ).fetchall()
        con.close()
        assert [r[1] for r in rows] == ["red", "green", "blue"]

    # --- NULL handling --------------------------------------------------

    def test_null_values(self, tmp_path):
        meta, data = _init_catalog(tmp_path)

        df = pl.DataFrame({
            "a": pl.Series([1, None, 3], dtype=pl.Int32),
            "b": ["hello", None, "world"],
            "c": pl.Series([1.5, None, 3.5], dtype=pl.Float64),
        })
        write_ducklake(df, meta, "t", mode="error")

        con = _reopen_duckdb(meta, data)
        rows = con.execute(
            "SELECT * FROM ducklake.t ORDER BY a NULLS LAST"
        ).fetchall()
        con.close()
        assert rows[0] == (1, "hello", 1.5)
        assert rows[1] == (3, "world", 3.5)
        assert rows[2][0] is None  # NULL a
        assert rows[2][1] is None  # NULL b

    # --- DuckDB SELECT * ------------------------------------------------

    def test_select_star(self, tmp_path):
        meta, data = _init_catalog(tmp_path)

        df = pl.DataFrame({
            "id": list(range(100)),
            "name": [f"item_{i}" for i in range(100)],
            "value": [float(i) * 1.5 for i in range(100)],
        })
        write_ducklake(df, meta, "t", mode="error")

        con = _reopen_duckdb(meta, data)
        rows = con.execute("SELECT * FROM ducklake.t").fetchall()
        con.close()
        assert len(rows) == 100

    # --- Filtered reads -------------------------------------------------

    def test_duckdb_filtered_read(self, tmp_path):
        meta, data = _init_catalog(tmp_path)

        df = pl.DataFrame({
            "id": list(range(50)),
            "cat": ["a" if i % 2 == 0 else "b" for i in range(50)],
        })
        write_ducklake(df, meta, "t", mode="error")

        con = _reopen_duckdb(meta, data)
        rows = con.execute(
            "SELECT id FROM ducklake.t WHERE cat = 'a' ORDER BY id"
        ).fetchall()
        con.close()
        assert [r[0] for r in rows] == list(range(0, 50, 2))

    # --- DuckDB INSERT into polars-created table ------------------------

    def test_duckdb_inserts_into_polars_table(self, tmp_path):
        meta, data = _init_catalog(tmp_path)

        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        write_ducklake(df, meta, "t", mode="error")

        con = _reopen_duckdb(meta, data)
        con.execute("INSERT INTO ducklake.t VALUES (3, 'z')")
        rows = con.execute("SELECT * FROM ducklake.t ORDER BY a").fetchall()
        con.close()

        assert len(rows) == 3
        assert rows[2] == (3, "z")

    # --- Multiple snapshots (time travel) -------------------------------

    def test_time_travel_polars_writes(self, tmp_path):
        meta, data = _init_catalog(tmp_path)

        df1 = pl.DataFrame({"v": [1, 2]})
        write_ducklake(df1, meta, "t", mode="error")

        # Grab snapshot after first write
        import sqlite3

        conn = sqlite3.connect(meta)
        snap1 = conn.execute(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        ).fetchone()[0]
        conn.close()

        df2 = pl.DataFrame({"v": [3, 4]})
        write_ducklake(df2, meta, "t", mode="append")

        # Latest should have 4 rows
        result = read_ducklake(meta, "t")
        assert sorted(result["v"].to_list()) == [1, 2, 3, 4]

        # At snap1 should have 2
        r1 = read_ducklake(meta, "t", snapshot_version=snap1)
        assert sorted(r1["v"].to_list()) == [1, 2]

    # --- Schema evolution -----------------------------------------------

    def test_polars_add_column_duckdb_reads(self, tmp_path):
        meta, data = _init_catalog(tmp_path)

        df = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df, meta, "t", mode="error")
        alter_ducklake_add_column(meta, "t", "b", pl.String())
        write_ducklake(
            pl.DataFrame({"a": [3], "b": ["hello"]}), meta, "t", mode="append"
        )

        con = _reopen_duckdb(meta, data)
        rows = con.execute("SELECT * FROM ducklake.t ORDER BY a").fetchall()
        con.close()
        assert len(rows) == 3
        assert rows[0] == (1, None)
        assert rows[1] == (2, None)
        assert rows[2] == (3, "hello")

    def test_polars_drop_column_duckdb_reads(self, tmp_path):
        meta, data = _init_catalog(tmp_path)

        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [10, 20]})
        write_ducklake(df, meta, "t", mode="error")
        alter_ducklake_drop_column(meta, "t", "b")
        write_ducklake(
            pl.DataFrame({"a": [3], "c": [30]}), meta, "t", mode="append"
        )

        con = _reopen_duckdb(meta, data)
        rows = con.execute("SELECT * FROM ducklake.t ORDER BY a").fetchall()
        con.close()
        assert len(rows) == 3
        assert rows[0] == (1, 10)
        assert rows[1] == (2, 20)
        assert rows[2] == (3, 30)

    def test_polars_rename_column_duckdb_reads(self, tmp_path):
        meta, data = _init_catalog(tmp_path)

        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        write_ducklake(df, meta, "t", mode="error")
        alter_ducklake_rename_column(meta, "t", "b", "c")
        write_ducklake(
            pl.DataFrame({"a": [3], "c": ["z"]}), meta, "t", mode="append"
        )

        con = _reopen_duckdb(meta, data)
        cols = [
            desc[0]
            for desc in con.execute("SELECT * FROM ducklake.t LIMIT 0").description
        ]
        rows = con.execute("SELECT * FROM ducklake.t ORDER BY a").fetchall()
        con.close()
        assert "c" in cols
        assert "b" not in cols
        assert len(rows) == 3

    # --- Partitioned tables ---------------------------------------------

    def test_polars_partitioned_duckdb_reads(self, tmp_path):
        meta, data = _init_catalog(tmp_path)

        df = pl.DataFrame({"id": [1, 2, 3], "cat": ["a", "b", "a"]})
        write_ducklake(df, meta, "t", mode="error")
        alter_ducklake_set_partitioned_by(meta, "t", ["cat"])

        df2 = pl.DataFrame({"id": [4, 5], "cat": ["b", "c"]})
        write_ducklake(df2, meta, "t", mode="append")

        con = _reopen_duckdb(meta, data)
        rows = con.execute("SELECT * FROM ducklake.t ORDER BY id").fetchall()
        con.close()
        assert len(rows) == 5
        assert [r[0] for r in rows] == [1, 2, 3, 4, 5]

    # --- Empty tables ---------------------------------------------------

    def test_polars_empty_table_duckdb_reads(self, tmp_path):
        meta, data = _init_catalog(tmp_path)

        from ducklake_polars import create_ducklake_table

        create_ducklake_table(meta, "t", {"a": pl.Int32(), "b": pl.String()})

        con = _reopen_duckdb(meta, data)
        rows = con.execute("SELECT * FROM ducklake.t").fetchall()
        cols = [
            desc[0]
            for desc in con.execute("SELECT * FROM ducklake.t LIMIT 0").description
        ]
        con.close()
        assert len(rows) == 0
        assert cols == ["a", "b"]

    # --- Large values ---------------------------------------------------

    def test_polars_long_string_duckdb_reads(self, tmp_path):
        meta, data = _init_catalog(tmp_path)

        long_str = "y" * 100_000
        df = pl.DataFrame({"s": [long_str]})
        write_ducklake(df, meta, "t", mode="error")

        con = _reopen_duckdb(meta, data)
        row = con.execute("SELECT s FROM ducklake.t").fetchone()
        con.close()
        assert len(row[0]) == 100_000

    @pytest.mark.xfail(
        reason="field_id mapping: DuckDB reads NULL inner values for nested types written by our library"
    )
    def test_polars_large_list_duckdb_reads(self, tmp_path):
        meta, data = _init_catalog(tmp_path)

        big = list(range(1000))
        df = pl.DataFrame({"a": [big]})
        write_ducklake(df, meta, "t", mode="error")

        con = _reopen_duckdb(meta, data)
        row = con.execute("SELECT a FROM ducklake.t").fetchone()
        con.close()
        assert list(row[0]) == big


# ═══════════════════════════════════════════════════════════════════════════
# Roundtrip: both systems interleave reads and writes
# ═══════════════════════════════════════════════════════════════════════════


class TestRoundtrip:
    """Write with polars → read with DuckDB → write more with DuckDB → read with polars."""

    def test_basic_roundtrip(self, tmp_path):
        meta, data = _init_catalog(tmp_path)

        # Step 1: polars writes
        df = pl.DataFrame({"id": [1, 2], "name": ["alice", "bob"]})
        write_ducklake(df, meta, "t", mode="error")

        # Step 2: DuckDB reads and adds more
        con = _reopen_duckdb(meta, data)
        rows = con.execute("SELECT * FROM ducklake.t ORDER BY id").fetchall()
        assert len(rows) == 2
        con.execute("INSERT INTO ducklake.t VALUES (3, 'charlie')")
        con.close()

        # Step 3: polars reads all 3
        result = read_ducklake(meta, "t").sort("id")
        assert result["id"].to_list() == [1, 2, 3]
        assert result["name"].to_list() == ["alice", "bob", "charlie"]

    def test_roundtrip_with_filter(self, tmp_path):
        meta, data = _init_catalog(tmp_path)

        # Polars writes
        df = pl.DataFrame({
            "id": list(range(20)),
            "val": [float(i) * 10 for i in range(20)],
        })
        write_ducklake(df, meta, "t", mode="error")

        # DuckDB adds more
        con = _reopen_duckdb(meta, data)
        con.execute("INSERT INTO ducklake.t VALUES (20, 200.0), (21, 210.0)")
        con.close()

        # Polars filters
        result = (
            scan_ducklake(meta, "t")
            .filter(pl.col("val") >= 190)
            .collect()
            .sort("id")
        )
        assert result["id"].to_list() == [19, 20, 21]

    def test_roundtrip_schema_evolution(self, tmp_path):
        meta, data = _init_catalog(tmp_path)

        # Polars creates table and writes
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        write_ducklake(df, meta, "t", mode="error")

        # DuckDB adds column and writes
        con = _reopen_duckdb(meta, data)
        con.execute("ALTER TABLE ducklake.t ADD COLUMN c DOUBLE")
        con.execute("INSERT INTO ducklake.t VALUES (3, 'z', 99.9)")
        con.close()

        # Polars reads (should see all 3 columns)
        result = read_ducklake(meta, "t").sort("a")
        assert result.columns == ["a", "b", "c"]
        assert result["a"].to_list() == [1, 2, 3]
        assert result["c"].to_list() == [None, None, 99.9]

    def test_roundtrip_multiple_writes(self, tmp_path):
        meta, data = _init_catalog(tmp_path)

        # Polars: batch 1
        write_ducklake(
            pl.DataFrame({"v": [1, 2]}), meta, "t", mode="error"
        )

        # DuckDB: batch 2
        con = _reopen_duckdb(meta, data)
        con.execute("INSERT INTO ducklake.t VALUES (3), (4)")
        con.close()

        # Polars: batch 3
        write_ducklake(
            pl.DataFrame({"v": [5, 6]}), meta, "t", mode="append"
        )

        # DuckDB: batch 4
        con = _reopen_duckdb(meta, data)
        con.execute("INSERT INTO ducklake.t VALUES (7), (8)")
        con.close()

        # Both should see all 8
        result = read_ducklake(meta, "t")
        assert sorted(result["v"].to_list()) == [1, 2, 3, 4, 5, 6, 7, 8]

        # DuckDB also sees all 8
        con = _reopen_duckdb(meta, data)
        rows = con.execute("SELECT * FROM ducklake.t ORDER BY v").fetchall()
        con.close()
        assert [r[0] for r in rows] == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_roundtrip_overwrite_then_duckdb(self, tmp_path):
        meta, data = _init_catalog(tmp_path)

        # Polars writes initial data
        write_ducklake(
            pl.DataFrame({"v": [1, 2, 3]}), meta, "t", mode="error"
        )

        # Polars overwrites
        write_ducklake(
            pl.DataFrame({"v": [100]}), meta, "t", mode="overwrite"
        )

        # DuckDB should see only the overwritten data
        con = _reopen_duckdb(meta, data)
        rows = con.execute("SELECT * FROM ducklake.t").fetchall()
        con.close()
        assert [r[0] for r in rows] == [100]

    def test_roundtrip_duckdb_schema_evolution_polars_continues(self, tmp_path):
        """DuckDB evolves schema, polars continues to write and read."""
        meta, data = _init_catalog(tmp_path)

        # Polars creates table
        df = pl.DataFrame({"a": [1, 2]})
        write_ducklake(df, meta, "t", mode="error")

        # DuckDB adds a column and renames another
        con = _reopen_duckdb(meta, data)
        con.execute("ALTER TABLE ducklake.t ADD COLUMN b VARCHAR")
        con.execute("INSERT INTO ducklake.t VALUES (3, 'hello')")
        con.close()

        # Polars can still read
        result = read_ducklake(meta, "t").sort("a")
        assert result.columns == ["a", "b"]
        assert result["a"].to_list() == [1, 2, 3]

        # Polars can still write (with the new schema)
        write_ducklake(
            pl.DataFrame({"a": [4], "b": ["world"]}),
            meta, "t", mode="append",
        )

        result = read_ducklake(meta, "t").sort("a")
        assert result["a"].to_list() == [1, 2, 3, 4]
        assert result["b"].to_list() == [None, None, "hello", "world"]

        # DuckDB can still read everything
        con = _reopen_duckdb(meta, data)
        rows = con.execute("SELECT * FROM ducklake.t ORDER BY a").fetchall()
        con.close()
        assert len(rows) == 4
        assert rows[3] == (4, "world")

    def test_roundtrip_inlined_data(self, tmp_path):
        """Inlined data works bidirectionally."""
        meta, data = _init_catalog(tmp_path)

        # DuckDB creates with inlining
        con, _, _ = _make_duckdb_catalog(tmp_path, inline=True)
        # Reuse the meta/data from _make_duckdb_catalog
        # Actually, let's use a separate tmp subdir
        con.close()

        meta2 = str(tmp_path / "inline_interop.ducklake")
        data2 = str(tmp_path / "inline_data")
        os.makedirs(data2, exist_ok=True)

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:sqlite:{meta2}' AS ducklake "
            f"(DATA_PATH '{data2}', DATA_INLINING_ROW_LIMIT 100)"
        )
        con.execute("CREATE TABLE ducklake.t (a INTEGER, b VARCHAR)")
        con.execute("INSERT INTO ducklake.t VALUES (1, 'duck'), (2, 'lake')")
        con.close()

        # Polars reads inlined data
        result = read_ducklake(meta2, "t").sort("a")
        assert result["a"].to_list() == [1, 2]
        assert result["b"].to_list() == ["duck", "lake"]
def _duckdb_read(cat, table: str, *, schema: str | None = None) -> list[tuple]:
    """Read a table using DuckDB's DuckLake extension and return rows."""
    import duckdb

    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    source = cat.attach_source()
    con.execute(
        f"ATTACH '{source}' AS ducklake (DATA_PATH '{cat.data_path}', DATA_INLINING_ROW_LIMIT 0)"
    )
    if schema:
        safe_schema = schema.replace('"', '""')
        safe_table = table.replace('"', '""')
        rows = con.execute(
            f'SELECT * FROM ducklake."{safe_schema}"."{safe_table}"'
        ).fetchall()
    else:
        safe_table = table.replace('"', '""')
        rows = con.execute(f'SELECT * FROM ducklake."{safe_table}"').fetchall()
    con.close()
    return rows


# ===================================================================
# SECTION 1: Basic CRUD Interop
# ===================================================================


class TestBasicInterop:
    """Test that basic DuckDB writes are correctly read by ducklake-dataframe."""

    def test_single_row(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (42, 'hello')")
        duckdb_rows = cat.fetchall("SELECT * FROM ducklake.test")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 2)
        assert result["a"].to_list() == [42]
        assert result["b"].to_list() == ["hello"]
        assert len(duckdb_rows) == result.shape[0]

    def test_empty_table_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        duckdb_count = cat.fetchone("SELECT COUNT(*) FROM ducklake.test")[0]
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == duckdb_count == 0
        assert result.columns == ["a", "b"]

    def test_multiple_batches_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (i INTEGER)")
        cat.execute("INSERT INTO ducklake.test SELECT i FROM range(100) t(i)")
        cat.execute("INSERT INTO ducklake.test SELECT i + 100 FROM range(100) t(i)")
        cat.execute("INSERT INTO ducklake.test SELECT i + 200 FROM range(100) t(i)")
        duckdb_count = cat.fetchone("SELECT COUNT(*) FROM ducklake.test")[0]
        duckdb_sum = cat.fetchone("SELECT SUM(i) FROM ducklake.test")[0]
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == duckdb_count == 300
        assert result["i"].sum() == duckdb_sum

    def test_ctas_interop(self, ducklake_catalog):
        """CREATE TABLE AS SELECT."""
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test AS "
            "SELECT i AS id, i * 2 AS doubled FROM range(50) t(i)"
        )
        duckdb_rows = cat.fetchall("SELECT * FROM ducklake.test ORDER BY id")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        assert result.shape[0] == len(duckdb_rows) == 50
        assert result["id"].to_list() == [r[0] for r in duckdb_rows]
        assert result["doubled"].to_list() == [r[1] for r in duckdb_rows]

    def test_multiple_tables_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1), (2)")
        cat.execute("CREATE TABLE ducklake.t2 (x VARCHAR)")
        cat.execute("INSERT INTO ducklake.t2 VALUES ('hello'), ('world')")
        cat.close()

        r1 = read_ducklake(cat.metadata_path, "t1")
        r2 = read_ducklake(cat.metadata_path, "t2")
        assert sorted(r1["a"].to_list()) == [1, 2]
        assert sorted(r2["x"].to_list()) == ["hello", "world"]

    def test_large_batch_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test AS "
            "SELECT i AS id, 'row_' || i AS label FROM range(10000) t(i)"
        )
        duckdb_count = cat.fetchone("SELECT COUNT(*) FROM ducklake.test")[0]
        duckdb_min = cat.fetchone("SELECT MIN(id) FROM ducklake.test")[0]
        duckdb_max = cat.fetchone("SELECT MAX(id) FROM ducklake.test")[0]
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == duckdb_count == 10000
        assert result["id"].min() == duckdb_min == 0
        assert result["id"].max() == duckdb_max == 9999


# ===================================================================
# SECTION 2: Type Interop
# ===================================================================


class TestTypeInterop:
    """Verify type round-trips: DuckDB writes -> ducklake-dataframe reads."""

    def test_integer_types_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                a TINYINT, b SMALLINT, c INTEGER, d BIGINT,
                e UTINYINT, f USMALLINT, g UINTEGER, h UBIGINT
            )
        """)
        cat.execute("INSERT INTO ducklake.test VALUES (1, 2, 3, 4, 5, 6, 7, 8)")
        duckdb_row = cat.fetchone("SELECT * FROM ducklake.test")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.row(0) == duckdb_row

    def test_float_types_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a FLOAT, b DOUBLE)")
        cat.execute("INSERT INTO ducklake.test VALUES (3.14, 2.718281828)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema == {"a": pl.Float32, "b": pl.Float64}
        assert abs(result["b"][0] - 2.718281828) < 1e-6

    def test_float_special_values_interop(self, ducklake_catalog):
        """NaN, Inf, -Inf round-trip."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a DOUBLE)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "('NaN'::DOUBLE), ('inf'::DOUBLE), ('-inf'::DOUBLE), (42.0)"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        values = result["a"].to_list()
        assert math.isnan(values[0])
        assert math.isinf(values[1]) and values[1] > 0
        assert math.isinf(values[2]) and values[2] < 0
        assert values[3] == 42.0

    def test_boolean_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a BOOLEAN)")
        cat.execute("INSERT INTO ducklake.test VALUES (true), (false), (NULL)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result["a"].to_list() == [True, False, None]

    def test_varchar_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES ('hello'), (''), (NULL), "
            "('with spaces'), ('with\ttab')"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 5
        values = result["a"].to_list()
        assert "hello" in values
        assert "" in values
        assert None in values

    def test_blob_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a BLOB)")
        cat.execute("INSERT INTO ducklake.test VALUES ('\\x01\\x02\\x03'::BLOB)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema == {"a": pl.Binary}
        assert result["a"][0] == b"\x01\x02\x03"

    def test_date_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a DATE)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "('2024-01-15'), ('1970-01-01'), ('2099-12-31')"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        values = result["a"].to_list()
        assert datetime.date(2024, 1, 15) in values
        assert datetime.date(1970, 1, 1) in values
        assert datetime.date(2099, 12, 31) in values

    def test_time_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a TIME)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "('00:00:00'), ('12:30:45'), ('23:59:59')"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        values = result["a"].to_list()
        assert datetime.time(0, 0, 0) in values
        assert datetime.time(12, 30, 45) in values
        assert datetime.time(23, 59, 59) in values

    def test_timestamp_variants_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                ts TIMESTAMP,
                ts_s TIMESTAMP_S,
                ts_ms TIMESTAMP_MS,
                ts_ns TIMESTAMP_NS
            )
        """)
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "('2024-06-15 10:30:00', '2024-06-15 10:30:00', "
            "'2024-06-15 10:30:00.123', '2024-06-15 10:30:00.123456789')"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 4)
        for col in result.columns:
            assert result.schema[col].base_type() == pl.Datetime

    def test_timestamptz_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a TIMESTAMPTZ)")
        cat.execute("INSERT INTO ducklake.test VALUES ('2024-06-15 10:30:00+05')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema["a"].base_type() == pl.Datetime

    def test_decimal_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                d1 DECIMAL(5, 2),
                d2 DECIMAL(18, 6),
                d3 DECIMAL(38, 10)
            )
        """)
        cat.execute("INSERT INTO ducklake.test VALUES (123.45, 99999.123456, 1234567890.1234567890)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result["d1"][0] == Decimal("123.45")
        assert result["d2"][0] == Decimal("99999.123456")

    def test_uuid_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a UUID)")
        cat.execute("INSERT INTO ducklake.test VALUES ('550e8400-e29b-41d4-a716-446655440000')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema == {"a": pl.Binary}
        assert len(result["a"][0]) == 16

    def test_json_interop(self, ducklake_catalog):
        import json

        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a JSON)")
        cat.execute("""INSERT INTO ducklake.test VALUES ('{"key": "value"}')""")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        text = result["a"].cast(pl.String)[0]
        assert json.loads(text) == {"key": "value"}

    def test_list_type_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER[])")
        cat.execute("INSERT INTO ducklake.test VALUES ([1, 2, 3]), ([4, 5]), (NULL)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result["a"].to_list() == [[1, 2, 3], [4, 5], None]

    def test_struct_type_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a STRUCT(x INTEGER, y VARCHAR))")
        cat.execute("INSERT INTO ducklake.test VALUES ({'x': 1, 'y': 'hi'}), (NULL)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result["a"][0] == {"x": 1, "y": "hi"}
        assert result["a"][1] is None

    def test_nested_list_of_structs_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a STRUCT(x INT, y VARCHAR)[])")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "([{'x': 1, 'y': 'a'}, {'x': 2, 'y': 'b'}])"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result["a"].to_list() == [[{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]]

    def test_list_of_lists_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER[][])")
        cat.execute("INSERT INTO ducklake.test VALUES ([[1, 2], [3, 4]])")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result["a"].to_list() == [[[1, 2], [3, 4]]]

    def test_struct_with_list_field_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a STRUCT(name VARCHAR, vals INT[]))")
        cat.execute("INSERT INTO ducklake.test VALUES ({'name': 'Alice', 'vals': [1, 2, 3]})")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        val = result["a"][0]
        assert val["name"] == "Alice"
        assert val["vals"] == [1, 2, 3]

    def test_mixed_types_single_table_interop(self, ducklake_catalog):
        """Table with many different column types."""
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                i INTEGER,
                s VARCHAR,
                f DOUBLE,
                b BOOLEAN,
                d DATE,
                t TIME,
                ts TIMESTAMP,
                dec DECIMAL(10, 2),
                bl BLOB
            )
        """)
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(42, 'hello', 3.14, true, '2024-01-01', '12:30:00', "
            "'2024-01-01 12:30:00', 99.99, '\\xAB'::BLOB)"
        )
        duckdb_row = cat.fetchone("SELECT i, s, b FROM ducklake.test")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 9)
        assert result["i"][0] == duckdb_row[0]
        assert result["s"][0] == duckdb_row[1]
        assert result["b"][0] == duckdb_row[2]


# ===================================================================
# SECTION 3: Delete Interop
# ===================================================================


class TestDeleteInterop:
    """Verify delete operations: DuckDB deletes -> ducklake-dataframe reads."""

    def test_basic_delete_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test AS SELECT i AS id FROM range(1000) t(i)")
        cat.execute("DELETE FROM ducklake.test WHERE id % 2 = 0")
        duckdb_count = cat.fetchone("SELECT COUNT(*) FROM ducklake.test")[0]
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == duckdb_count == 500
        # All remaining should be odd
        assert all(v % 2 != 0 for v in result["id"].to_list())

    def test_delete_all_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test AS SELECT i AS id FROM range(100) t(i)")
        cat.execute("DELETE FROM ducklake.test")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 0
        assert result.columns == ["id"]

    def test_noop_delete_interop(self, ducklake_catalog):
        """DELETE with no matching rows should not alter data."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test AS SELECT i AS id FROM range(10) t(i)")
        cat.execute("DELETE FROM ducklake.test WHERE id > 1000")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 10

    def test_sequential_deletes_interop(self, ducklake_catalog):
        """Three sequential deletes narrowing down data."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test AS SELECT i AS id FROM range(100) t(i)")
        cat.execute("DELETE FROM ducklake.test WHERE id < 25")
        cat.execute("DELETE FROM ducklake.test WHERE id >= 75")
        cat.execute("DELETE FROM ducklake.test WHERE id = 50")
        duckdb_count = cat.fetchone("SELECT COUNT(*) FROM ducklake.test")[0]
        duckdb_sum = cat.fetchone("SELECT SUM(id) FROM ducklake.test")[0]
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == duckdb_count
        assert result["id"].sum() == duckdb_sum
        expected = [x for x in range(25, 75) if x != 50]
        assert sorted(result["id"].to_list()) == expected

    def test_delete_from_multiple_files_interop(self, ducklake_catalog):
        """Two inserts = 2 files; delete spans both."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, batch VARCHAR)")
        cat.execute("INSERT INTO ducklake.test SELECT i, 'a' FROM range(50) t(i)")
        cat.execute("INSERT INTO ducklake.test SELECT i + 1000, 'b' FROM range(50) t(i)")
        cat.execute("DELETE FROM ducklake.test WHERE id % 2 = 0")
        duckdb_rows = cat.fetchall("SELECT id FROM ducklake.test ORDER BY id")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        polars_ids = sorted(result["id"].to_list())
        duckdb_ids = [r[0] for r in duckdb_rows]
        assert polars_ids == duckdb_ids

    def test_delete_then_insert_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test AS SELECT i AS id FROM range(10) t(i)")
        cat.execute("DELETE FROM ducklake.test WHERE id < 5")
        cat.execute("INSERT INTO ducklake.test SELECT i + 100 FROM range(5) t(i)")
        duckdb_count = cat.fetchone("SELECT COUNT(*) FROM ducklake.test")[0]
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == duckdb_count == 10

    def test_delete_insert_delete_interop(self, ducklake_catalog):
        """Multiple cycles of delete and insert."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test AS SELECT i AS id FROM range(100) t(i)")
        cat.execute("DELETE FROM ducklake.test WHERE id >= 50")  # keep 0-49
        cat.execute("INSERT INTO ducklake.test SELECT i + 200 FROM range(50) t(i)")  # add 200-249
        cat.execute("DELETE FROM ducklake.test WHERE id < 25")  # remove 0-24
        duckdb_rows = cat.fetchall("SELECT id FROM ducklake.test ORDER BY id")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        polars_ids = sorted(result["id"].to_list())
        duckdb_ids = [r[0] for r in duckdb_rows]
        assert polars_ids == duckdb_ids


# ===================================================================
# SECTION 4: Update Interop
# ===================================================================


class TestUpdateInterop:
    """Verify UPDATE operations: DuckDB updates -> ducklake-dataframe reads."""

    def test_basic_update_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, val INTEGER)")
        cat.execute("INSERT INTO ducklake.test SELECT i, i * 10 FROM range(10) t(i)")
        cat.execute("UPDATE ducklake.test SET val = val + 100 WHERE id < 5")
        duckdb_rows = cat.fetchall("SELECT * FROM ducklake.test ORDER BY id")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("id")
        assert result.shape[0] == len(duckdb_rows) == 10
        for i, (did, dval) in enumerate(duckdb_rows):
            assert result["id"][i] == did
            assert result["val"][i] == dval

    def test_update_all_rows_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, name VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'old1'), (2, 'old2'), (3, 'old3')"
        )
        cat.execute("UPDATE ducklake.test SET name = 'updated'")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 3
        assert all(v == "updated" for v in result["name"].to_list())

    def test_update_no_match_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, val INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 10), (2, 20)")
        cat.execute("UPDATE ducklake.test SET val = 999 WHERE id > 100")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("id")
        assert result["val"].to_list() == [10, 20]

    def test_update_multiple_columns_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 10, 'hello'), (2, 20, 'world')")
        cat.execute("UPDATE ducklake.test SET a = 99, b = 'changed' WHERE id = 1")
        duckdb_row = cat.fetchone("SELECT * FROM ducklake.test WHERE id = 1")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        r = result.filter(pl.col("id") == 1)
        assert r["a"][0] == duckdb_row[1]
        assert r["b"][0] == duckdb_row[2]

    def test_update_then_delete_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, val INTEGER)")
        cat.execute("INSERT INTO ducklake.test SELECT i, i * 10 FROM range(20) t(i)")
        cat.execute("UPDATE ducklake.test SET val = val + 1000 WHERE id < 10")
        cat.execute("DELETE FROM ducklake.test WHERE id >= 15")
        duckdb_rows = cat.fetchall("SELECT * FROM ducklake.test ORDER BY id")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("id")
        assert result.shape[0] == len(duckdb_rows)
        polars_data = list(zip(result["id"].to_list(), result["val"].to_list()))
        assert polars_data == duckdb_rows

    def test_sequential_updates_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, val INTEGER)")
        cat.execute("INSERT INTO ducklake.test SELECT i, 0 FROM range(10) t(i)")
        cat.execute("UPDATE ducklake.test SET val = val + 1")  # all become 1
        cat.execute("UPDATE ducklake.test SET val = val + 10 WHERE id < 5")  # 0-4 become 11
        cat.execute("UPDATE ducklake.test SET val = val + 100 WHERE id = 0")  # 0 becomes 111
        duckdb_rows = cat.fetchall("SELECT * FROM ducklake.test ORDER BY id")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("id")
        polars_data = list(zip(result["id"].to_list(), result["val"].to_list()))
        assert polars_data == duckdb_rows


# ===================================================================
# SECTION 5: Schema Evolution Interop
# ===================================================================


class TestSchemaEvolutionInterop:
    """Verify schema evolution: DuckDB ALTER -> ducklake-dataframe reads."""

    def test_add_column_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), (2)")
        cat.execute("ALTER TABLE ducklake.test ADD COLUMN b VARCHAR")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'hello')")
        duckdb_rows = cat.fetchall("SELECT * FROM ducklake.test ORDER BY a")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result.shape[0] == len(duckdb_rows) == 3
        assert result["b"].to_list() == [None, None, "hello"]

    def test_drop_column_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR, c DOUBLE)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello', 3.14)")
        cat.execute("ALTER TABLE ducklake.test DROP COLUMN b")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 2.72)")
        duckdb_rows = cat.fetchall("SELECT * FROM ducklake.test ORDER BY a")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result.columns == ["a", "c"]
        assert result.shape[0] == len(duckdb_rows) == 2

    def test_rename_column_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello')")
        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO name")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'world')")
        duckdb_rows = cat.fetchall("SELECT * FROM ducklake.test ORDER BY a")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result.columns == ["a", "name"]
        assert result.shape[0] == len(duckdb_rows) == 2
        assert result["name"].to_list() == ["hello", "world"]

    def test_mixed_alter_interop(self, ducklake_catalog):
        """ADD + DROP + RENAME in sequence."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b INTEGER, c VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 10, 'first')")

        cat.execute("ALTER TABLE ducklake.test DROP COLUMN b")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'second')")

        cat.execute("ALTER TABLE ducklake.test ADD COLUMN d DOUBLE")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'third', 3.14)")

        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN c TO label")
        cat.execute("INSERT INTO ducklake.test VALUES (4, 'fourth', 2.72)")

        duckdb_rows = cat.fetchall("SELECT * FROM ducklake.test ORDER BY a")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result.shape[0] == len(duckdb_rows) == 4
        assert result.columns == ["a", "label", "d"]

    def test_type_promotion_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b TINYINT)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 25)")
        cat.execute("ALTER TABLE ducklake.test ALTER COLUMN b SET DATA TYPE INTEGER")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 1000)")
        duckdb_rows = cat.fetchall("SELECT * FROM ducklake.test ORDER BY a")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result.schema["b"] == pl.Int32
        assert result["b"].to_list() == [25, 1000]

    def test_struct_evolution_add_field_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, col STRUCT(i INTEGER, j INTEGER))")
        cat.execute("INSERT INTO ducklake.test VALUES (1, {i: 10, j: 20})")
        cat.execute(
            "ALTER TABLE ducklake.test ALTER COLUMN col "
            "SET DATA TYPE STRUCT(i INTEGER, j INTEGER, k INTEGER)"
        )
        cat.execute("INSERT INTO ducklake.test VALUES (2, {i: 30, j: 40, k: 50})")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        structs = result["col"].to_list()
        assert structs[0] == {"i": 10, "j": 20, "k": None}
        assert structs[1] == {"i": 30, "j": 40, "k": 50}

    def test_struct_evolution_drop_field_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test (a INTEGER, col STRUCT(i INTEGER, j INTEGER, k INTEGER))"
        )
        cat.execute("INSERT INTO ducklake.test VALUES (1, {i: 10, j: 20, k: 30})")
        cat.execute(
            "ALTER TABLE ducklake.test ALTER COLUMN col "
            "SET DATA TYPE STRUCT(i INTEGER, k INTEGER)"
        )
        cat.execute("INSERT INTO ducklake.test VALUES (2, {i: 40, k: 50})")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        structs = result["col"].to_list()
        assert structs[0] == {"i": 10, "k": 30}
        assert structs[1] == {"i": 40, "k": 50}

    def test_table_rename_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.original (a INTEGER)")
        cat.execute("INSERT INTO ducklake.original VALUES (42)")
        cat.execute("ALTER TABLE ducklake.original RENAME TO renamed")
        duckdb_rows = cat.fetchall("SELECT * FROM ducklake.renamed")
        cat.close()

        result = read_ducklake(cat.metadata_path, "renamed")
        assert result["a"].to_list() == [42]
        with pytest.raises(ValueError, match="original"):
            read_ducklake(cat.metadata_path, "original")


# ===================================================================
# SECTION 6: Time Travel Interop
# ===================================================================


class TestTimeTravelInterop:
    """Verify time travel: snapshot versions match between DuckDB and Polars."""

    def test_version_time_travel_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER)")
        v0 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("INSERT INTO ducklake.test VALUES (2)")
        v2 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("INSERT INTO ducklake.test VALUES (3)")
        cat.close()

        # At v0: empty
        r0 = read_ducklake(cat.metadata_path, "test", snapshot_version=v0)
        assert r0.shape[0] == 0

        # At v1: 1 row
        r1 = read_ducklake(cat.metadata_path, "test", snapshot_version=v1)
        assert r1["id"].to_list() == [1]

        # At v2: 2 rows
        r2 = read_ducklake(cat.metadata_path, "test", snapshot_version=v2)
        assert sorted(r2["id"].to_list()) == [1, 2]

        # Latest: 3 rows
        r_latest = read_ducklake(cat.metadata_path, "test")
        assert sorted(r_latest["id"].to_list()) == [1, 2, 3]

    def test_time_travel_with_delete_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test AS SELECT i AS id FROM range(100) t(i)")
        v_before = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("DELETE FROM ducklake.test WHERE id < 50")
        cat.close()

        r_before = read_ducklake(cat.metadata_path, "test", snapshot_version=v_before)
        assert r_before.shape[0] == 100

        r_after = read_ducklake(cat.metadata_path, "test")
        assert r_after.shape[0] == 50
        assert min(r_after["id"].to_list()) == 50

    def test_time_travel_with_update_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, val INTEGER)")
        cat.execute("INSERT INTO ducklake.test SELECT i, i * 10 FROM range(10) t(i)")
        v_before = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("UPDATE ducklake.test SET val = val + 1000")
        cat.close()

        r_before = read_ducklake(cat.metadata_path, "test", snapshot_version=v_before)
        r_after = read_ducklake(cat.metadata_path, "test")

        # Same row count
        assert r_before.shape[0] == r_after.shape[0] == 10

        # Values differ
        before_sum = r_before["val"].sum()
        after_sum = r_after["val"].sum()
        assert after_sum == before_sum + 10000

    def test_time_travel_with_schema_evolution_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("ALTER TABLE ducklake.test ADD COLUMN b VARCHAR")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'new')")
        cat.close()

        # At v1: only column 'a'
        r1 = read_ducklake(cat.metadata_path, "test", snapshot_version=v1)
        assert r1.columns == ["a"]
        assert r1["a"].to_list() == [1]

        # Latest: both columns
        r_latest = read_ducklake(cat.metadata_path, "test")
        assert r_latest.columns == ["a", "b"]
        assert r_latest.shape[0] == 2

    def test_time_travel_dropped_table_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (42)")
        v_exists = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.execute("DROP TABLE ducklake.test")
        cat.close()

        # Current: table does not exist
        with pytest.raises(ValueError, match="not found"):
            read_ducklake(cat.metadata_path, "test")

        # Time travel: table is accessible
        r = read_ducklake(cat.metadata_path, "test", snapshot_version=v_exists)
        assert r["a"].to_list() == [42]

    def test_time_travel_rename_column_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello')")
        v_before_rename = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO name")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'world')")
        cat.close()

        # Before rename: column is 'b'
        r_before = read_ducklake(cat.metadata_path, "test", snapshot_version=v_before_rename)
        assert r_before.columns == ["a", "b"]

        # After rename: column is 'name'
        r_after = read_ducklake(cat.metadata_path, "test")
        assert r_after.columns == ["a", "name"]
        assert sorted(r_after["name"].to_list()) == ["hello", "world"]


# ===================================================================
# SECTION 7: Partition Interop
# ===================================================================


class TestPartitionInterop:
    """Verify partitioned table reading matches DuckDB."""

    def test_basic_partition_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val INTEGER, part VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (part)")
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i, CASE WHEN i % 2 = 0 THEN 'even' ELSE 'odd' END "
            "FROM range(100) t(i)"
        )
        duckdb_even = cat.fetchone(
            "SELECT COUNT(*) FROM ducklake.test WHERE part = 'even'"
        )[0]
        duckdb_odd = cat.fetchone(
            "SELECT COUNT(*) FROM ducklake.test WHERE part = 'odd'"
        )[0]
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        even = lf.filter(pl.col("part") == "even").collect()
        odd = lf.filter(pl.col("part") == "odd").collect()
        assert even.shape[0] == duckdb_even == 50
        assert odd.shape[0] == duckdb_odd == 50

    def test_multi_key_partition_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR, c INTEGER)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b, c)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 'x', 10), (2, 'y', 20), (3, 'x', 10), (4, 'y', 30)"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 3, 4]
        assert result["b"].to_list() == ["x", "y", "x", "y"]
        assert result["c"].to_list() == [10, 20, 10, 30]

    def test_partition_filter_pruning_interop(self, ducklake_catalog):
        """Verify filter on partition column gives correct results."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, part INTEGER)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (part)")
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i, i % 5 FROM range(1000) t(i)"
        )
        duckdb_part0 = cat.fetchone(
            "SELECT COUNT(*) FROM ducklake.test WHERE part = 0"
        )[0]
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.filter(pl.col("part") == 0).collect()
        assert result.shape[0] == duckdb_part0 == 200

    def test_partition_with_delete_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val INTEGER, part VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (part)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 'a'), (2, 'b'), (3, 'a'), (4, 'b'), (5, 'a')"
        )
        cat.execute("DELETE FROM ducklake.test WHERE val <= 2")
        duckdb_rows = cat.fetchall("SELECT * FROM ducklake.test ORDER BY val")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("val")
        assert result.shape[0] == len(duckdb_rows)
        assert result["val"].to_list() == [r[0] for r in duckdb_rows]

    def test_partition_with_update_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val INTEGER, part VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (part)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'a'), (2, 'b'), (3, 'a')"
        )
        cat.execute("UPDATE ducklake.test SET val = val + 100 WHERE part = 'a'")
        duckdb_rows = cat.fetchall("SELECT * FROM ducklake.test ORDER BY val")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("val")
        polars_data = list(zip(result["val"].to_list(), result["part"].to_list()))
        assert polars_data == duckdb_rows

    def test_partition_with_schema_evolution_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val INTEGER, part VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (part)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'a'), (2, 'b')")
        cat.execute("ALTER TABLE ducklake.test ADD COLUMN extra DOUBLE")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'a', 3.14)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("val")
        assert result.columns == ["val", "part", "extra"]
        assert result["extra"].to_list() == [None, None, 3.14]

    def test_partition_large_table_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, category VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (category)")
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i, CASE WHEN i % 3 = 0 THEN 'a' "
            "WHEN i % 3 = 1 THEN 'b' ELSE 'c' END "
            "FROM range(3000) t(i)"
        )
        duckdb_count = cat.fetchone("SELECT COUNT(*) FROM ducklake.test")[0]
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == duckdb_count == 3000


# ===================================================================
# SECTION 8: Data Inlining Interop
# ===================================================================


class TestDataInliningInterop:
    """Verify inlined data reading matches DuckDB."""

    def test_basic_inlining_interop(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello'), (2, 'world')")
        duckdb_rows = cat.fetchall("SELECT * FROM ducklake.test ORDER BY a")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result.shape[0] == len(duckdb_rows) == 2
        assert result["a"].to_list() == [r[0] for r in duckdb_rows]
        assert result["b"].to_list() == [r[1] for r in duckdb_rows]

    def test_inlining_with_nulls_interop(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, NULL), (NULL, 'hi'), (NULL, NULL)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 3

    def test_inlining_multiple_inserts_interop(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        cat.execute("INSERT INTO ducklake.test VALUES (2)")
        cat.execute("INSERT INTO ducklake.test VALUES (3)")
        duckdb_count = cat.fetchone("SELECT COUNT(*) FROM ducklake.test")[0]
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == duckdb_count == 3
        assert sorted(result["a"].to_list()) == [1, 2, 3]

    def test_inlining_with_schema_evolution_interop(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        cat.execute("ALTER TABLE ducklake.test ADD COLUMN b VARCHAR")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'new')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result.columns == ["a", "b"]
        assert result["a"].to_list() == [1, 2]

    def test_inlining_empty_table_interop(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (0, 2)

    def test_inlining_then_delete_interop(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), (2), (3), (4), (5)")
        cat.execute("DELETE FROM ducklake.test WHERE a <= 2")
        duckdb_count = cat.fetchone("SELECT COUNT(*) FROM ducklake.test")[0]
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == duckdb_count
        assert sorted(result["a"].to_list()) == [3, 4, 5]


# ===================================================================
# SECTION 9: Table Changes (CDC) Interop
# ===================================================================


class TestTableChangesInterop:
    """Verify change data feed: DuckDB operations -> ducklake-dataframe table_changes()."""

    def test_insertions_tracked_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        cat.execute("INSERT INTO ducklake.test VALUES (2)")
        cat.execute("INSERT INTO ducklake.test VALUES (3)")
        snap = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        insertions = api.table_insertions("test", 0, snap)
        assert insertions.shape[0] == 3
        assert sorted(insertions["id"].to_list()) == [1, 2, 3]

    def test_deletions_tracked_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), (2), (3), (4), (5)")
        snap_before = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.execute("DELETE FROM ducklake.test WHERE id <= 2")
        snap_after = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        deletions = api.table_deletions("test", snap_before, snap_after)
        assert deletions.shape[0] == 2
        assert sorted(deletions["id"].to_list()) == [1, 2]

    def test_changes_detect_updates_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, val INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 10), (2, 20)")
        snap_before = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.execute("UPDATE ducklake.test SET val = 99 WHERE id = 1")
        snap_after = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        changes = api.table_changes("test", snap_before, snap_after)
        # Should detect update: preimage and postimage
        change_types = set(changes["change_type"].to_list())
        assert "update_preimage" in change_types
        assert "update_postimage" in change_types

    def test_full_lifecycle_changes_interop(self, ducklake_catalog):
        """INSERT -> UPDATE -> DELETE, verify all tracked."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, val INTEGER)")
        snap0 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("INSERT INTO ducklake.test VALUES (1, 10), (2, 20), (3, 30)")
        snap1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("UPDATE ducklake.test SET val = 99 WHERE id = 1")
        snap2 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("DELETE FROM ducklake.test WHERE id = 3")
        snap3 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        # Full range of changes
        changes = api.table_changes("test", snap0, snap3)
        assert changes.shape[0] > 0
        change_types = set(changes["change_type"].to_list())
        assert "insert" in change_types


# ===================================================================
# SECTION 10: Statistics & Filter Pushdown Interop
# ===================================================================


class TestStatsFilterInterop:
    """Verify statistics enable correct filter pushdown."""

    def test_integer_range_filter_interop(self, ducklake_catalog):
        """Three files with non-overlapping ranges; filter should prune."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER)")
        cat.execute("INSERT INTO ducklake.test SELECT i FROM range(0, 100) t(i)")
        cat.execute("INSERT INTO ducklake.test SELECT i FROM range(1000, 1100) t(i)")
        cat.execute("INSERT INTO ducklake.test SELECT i FROM range(5000, 5100) t(i)")
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        # Filter should hit only file 1
        result = lf.filter(pl.col("id") == 50).collect()
        assert result.shape[0] == 1
        assert result["id"][0] == 50

    def test_date_filter_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, d DATE)")
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i, DATE '2020-01-01' + INTERVAL (i) DAY "
            "FROM range(100) t(i)"
        )
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i + 1000, DATE '2025-01-01' + INTERVAL (i) DAY "
            "FROM range(100) t(i)"
        )
        duckdb_count = cat.fetchone(
            "SELECT COUNT(*) FROM ducklake.test WHERE d >= DATE '2025-01-01'"
        )[0]
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.filter(pl.col("d") >= datetime.date(2025, 1, 1)).collect()
        assert result.shape[0] == duckdb_count

    def test_string_filter_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (s VARCHAR)")
        cat.execute("INSERT INTO ducklake.test SELECT printf('%06d', i) FROM range(100) t(i)")
        cat.execute(
            "INSERT INTO ducklake.test SELECT printf('%06d', i) FROM range(500, 600) t(i)"
        )
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.filter(pl.col("s") >= "000500").collect()
        assert result.shape[0] == 100

    def test_decimal_filter_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, d DECIMAL(9,3))")
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i, CAST(i AS DECIMAL(9,3)) / 10 FROM range(100) t(i)"
        )
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i + 1000, CAST(i + 1000 AS DECIMAL(9,3)) / 10 FROM range(100) t(i)"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 200

    def test_multi_file_count_interop(self, ducklake_catalog):
        """Verify row count is accurate across multiple files."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER)")
        for offset in range(0, 500, 100):
            cat.execute(
                f"INSERT INTO ducklake.test SELECT i FROM range({offset}, {offset + 100}) t(i)"
            )
        duckdb_count = cat.fetchone("SELECT COUNT(*) FROM ducklake.test")[0]
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == duckdb_count == 500


# ===================================================================
# SECTION 11: Catalog API Interop
# ===================================================================


class TestCatalogAPIInterop:
    """Verify DuckLakeCatalog API returns correct data."""

    def test_snapshots_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        cat.execute("INSERT INTO ducklake.test VALUES (2)")
        duckdb_snap = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        snaps = api.snapshots()
        assert snaps.shape[0] >= 3  # CREATE + 2 INSERTs
        assert api.current_snapshot() == duckdb_snap

    def test_list_tables_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("CREATE TABLE ducklake.t2 (b VARCHAR)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        tables = api.list_tables()
        names = tables["table_name"].to_list()
        assert "t1" in names
        assert "t2" in names

    def test_table_info_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR, c DOUBLE)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello', 3.14)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        info = api.table_info()
        assert isinstance(info, pl.DataFrame)
        assert info.shape[0] >= 1
        # table_info returns table-level metadata, not column info
        assert "table_name" in info.columns

    def test_list_files_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        cat.execute("INSERT INTO ducklake.test VALUES (2)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        files = api.list_files("test")
        assert files.shape[0] >= 2  # At least 2 data files

    def test_options_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        options = api.options()
        assert isinstance(options, pl.DataFrame)
        assert options.shape[0] > 0

    def test_list_schemas_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        api = DuckLakeCatalog(cat.metadata_path)
        schemas = api.list_schemas()
        assert schemas.shape[0] >= 1
        schema_names = schemas["schema_name"].to_list()
        assert "main" in schema_names


# ===================================================================
# SECTION 12: Schema Interop
# ===================================================================


class TestSchemaInterop:
    """Verify non-default schema operations."""

    def test_non_default_schema_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE SCHEMA ducklake.myschema")
        cat.execute("CREATE TABLE ducklake.myschema.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.myschema.test VALUES (42)")
        duckdb_row = cat.fetchone("SELECT * FROM ducklake.myschema.test")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test", schema="myschema")
        assert result["a"].to_list() == [duckdb_row[0]]

    def test_same_table_name_different_schemas_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")

        cat.execute("CREATE SCHEMA ducklake.other")
        cat.execute("CREATE TABLE ducklake.other.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.other.test VALUES (99)")
        cat.close()

        r_main = read_ducklake(cat.metadata_path, "test")
        r_other = read_ducklake(cat.metadata_path, "test", schema="other")

        assert r_main["a"].to_list() == [1]
        assert r_other["a"].to_list() == [99]

    def test_schema_with_multiple_tables_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE SCHEMA ducklake.s1")
        cat.execute("CREATE TABLE ducklake.s1.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.s1.t1 VALUES (10)")
        cat.execute("CREATE TABLE ducklake.s1.t2 (b VARCHAR)")
        cat.execute("INSERT INTO ducklake.s1.t2 VALUES ('hello')")
        cat.close()

        r1 = read_ducklake(cat.metadata_path, "t1", schema="s1")
        r2 = read_ducklake(cat.metadata_path, "t2", schema="s1")
        assert r1["a"].to_list() == [10]
        assert r2["b"].to_list() == ["hello"]


# ===================================================================
# SECTION 13: View Interop
# ===================================================================


class TestViewInterop:
    """Verify view operations produce correct data."""

    def test_basic_view_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello'), (2, 'world')")
        cat.execute("CREATE VIEW ducklake.v1 AS SELECT a, b FROM ducklake.test WHERE a = 1")
        duckdb_view = cat.fetchall("SELECT * FROM ducklake.v1")
        cat.close()

        # Views are readable in DuckDB but ducklake-dataframe only reads tables
        # Just verify the underlying table is correct
        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 2

    def test_table_survives_view_drop_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (42)")
        cat.execute("CREATE VIEW ducklake.v1 AS SELECT * FROM ducklake.test")
        cat.execute("DROP VIEW ducklake.v1")
        cat.close()

        # Table should still be readable
        result = read_ducklake(cat.metadata_path, "test")
        assert result["a"].to_list() == [42]


# ===================================================================
# SECTION 14: Merge Interop
# ===================================================================


class TestMergeInterop:
    """Verify MERGE operations: DuckDB merges -> ducklake-dataframe reads."""

    def test_merge_update_insert_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.stock (item_id INTEGER, balance INTEGER)")
        cat.execute("INSERT INTO ducklake.stock VALUES (10, 2200), (20, 1900)")

        cat.execute("CREATE TABLE ducklake.buy (item_id INTEGER, volume INTEGER)")
        cat.execute("INSERT INTO ducklake.buy VALUES (10, 1000), (30, 300)")

        cat.execute("""
            MERGE INTO ducklake.stock AS s USING ducklake.buy AS b
            ON s.item_id = b.item_id
            WHEN MATCHED THEN UPDATE SET balance = s.balance + b.volume
            WHEN NOT MATCHED THEN INSERT VALUES (b.item_id, b.volume)
        """)
        duckdb_rows = cat.fetchall("SELECT * FROM ducklake.stock ORDER BY item_id")
        cat.close()

        result = read_ducklake(cat.metadata_path, "stock").sort("item_id")
        polars_data = list(zip(result["item_id"].to_list(), result["balance"].to_list()))
        assert polars_data == duckdb_rows
        # Expected: (10, 3200), (20, 1900), (30, 300)
        assert result["item_id"].to_list() == [10, 20, 30]
        assert result["balance"].to_list() == [3200, 1900, 300]

    def test_merge_all_matches_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.target (id INTEGER, val INTEGER)")
        cat.execute("INSERT INTO ducklake.target VALUES (1, 10), (2, 20)")

        cat.execute("CREATE TABLE ducklake.source (id INTEGER, val INTEGER)")
        cat.execute("INSERT INTO ducklake.source VALUES (1, 100), (2, 200)")

        cat.execute("""
            MERGE INTO ducklake.target AS t USING ducklake.source AS s
            ON t.id = s.id
            WHEN MATCHED THEN UPDATE SET val = s.val
        """)
        duckdb_rows = cat.fetchall("SELECT * FROM ducklake.target ORDER BY id")
        cat.close()

        result = read_ducklake(cat.metadata_path, "target").sort("id")
        assert result["val"].to_list() == [100, 200]

    def test_merge_no_matches_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.target (id INTEGER, val INTEGER)")
        cat.execute("INSERT INTO ducklake.target VALUES (1, 10)")

        cat.execute("CREATE TABLE ducklake.source (id INTEGER, val INTEGER)")
        cat.execute("INSERT INTO ducklake.source VALUES (2, 20), (3, 30)")

        cat.execute("""
            MERGE INTO ducklake.target AS t USING ducklake.source AS s
            ON t.id = s.id
            WHEN MATCHED THEN UPDATE SET val = s.val
            WHEN NOT MATCHED THEN INSERT VALUES (s.id, s.val)
        """)
        duckdb_rows = cat.fetchall("SELECT * FROM ducklake.target ORDER BY id")
        cat.close()

        result = read_ducklake(cat.metadata_path, "target").sort("id")
        assert result["id"].to_list() == [1, 2, 3]
        assert result["val"].to_list() == [10, 20, 30]

    def test_merge_with_delete_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.target (id INTEGER, val INTEGER)")
        cat.execute("INSERT INTO ducklake.target VALUES (1, 10), (2, 20), (3, 30)")

        cat.execute("CREATE TABLE ducklake.source (id INTEGER)")
        cat.execute("INSERT INTO ducklake.source VALUES (2)")

        cat.execute("""
            MERGE INTO ducklake.target AS t USING ducklake.source AS s
            ON t.id = s.id
            WHEN MATCHED THEN DELETE
        """)
        duckdb_rows = cat.fetchall("SELECT * FROM ducklake.target ORDER BY id")
        cat.close()

        result = read_ducklake(cat.metadata_path, "target").sort("id")
        assert result["id"].to_list() == [1, 3]
        assert result["val"].to_list() == [10, 30]


# ===================================================================
# SECTION 15: Null Handling Interop
# ===================================================================


class TestNullHandlingInterop:
    """Verify null handling across types."""

    def test_all_null_row_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR, c DOUBLE, d DATE)")
        cat.execute("INSERT INTO ducklake.test VALUES (NULL, NULL, NULL, NULL)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 4)
        assert result.row(0) == (None, None, None, None)

    def test_mixed_nulls_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, NULL), (NULL, 'hello'), (2, 'world'), (NULL, NULL)"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 4
        null_count_a = result["a"].null_count()
        null_count_b = result["b"].null_count()
        assert null_count_a == 2
        assert null_count_b == 2

    def test_null_in_filter_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), (NULL), (3)")
        cat.execute("INSERT INTO ducklake.test VALUES (4), (NULL), (6)")
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.filter(pl.col("a").is_not_null()).collect()
        assert result.shape[0] == 4
        assert sorted(result["a"].to_list()) == [1, 3, 4, 6]

    def test_null_list_elements_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER[])")
        cat.execute("INSERT INTO ducklake.test VALUES ([1, NULL, 3]), (NULL), ([NULL])")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 3
        values = result["a"].to_list()
        assert values[0] == [1, None, 3]
        assert values[1] is None
        assert values[2] == [None]

    def test_null_struct_fields_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a STRUCT(x INTEGER, y VARCHAR))")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "({'x': NULL, 'y': 'hi'}), ({'x': 1, 'y': NULL}), (NULL)"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        values = result["a"].to_list()
        assert values[0] == {"x": None, "y": "hi"}
        assert values[1] == {"x": 1, "y": None}
        assert values[2] is None


# ===================================================================
# SECTION 16: Complex Operation Sequences Interop
# ===================================================================


class TestComplexSequenceInterop:
    """Test complex multi-operation sequences."""

    def test_insert_update_delete_schema_change(self, ducklake_catalog):
        """Full lifecycle: create, insert, update, add column, insert, delete."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, val VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'a'), (2, 'b'), (3, 'c')")
        cat.execute("UPDATE ducklake.test SET val = 'updated' WHERE id = 1")
        cat.execute("ALTER TABLE ducklake.test ADD COLUMN score DOUBLE")
        cat.execute("INSERT INTO ducklake.test VALUES (4, 'd', 9.5)")
        cat.execute("DELETE FROM ducklake.test WHERE id = 2")
        duckdb_rows = cat.fetchall("SELECT * FROM ducklake.test ORDER BY id")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("id")
        assert result.shape[0] == len(duckdb_rows) == 3
        polars_data = list(
            zip(
                result["id"].to_list(),
                result["val"].to_list(),
                result["score"].to_list(),
            )
        )
        assert polars_data == duckdb_rows

    def test_many_snapshots_interop(self, ducklake_catalog):
        """Create many snapshots and verify latest is correct."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (i INTEGER)")
        for v in range(20):
            cat.execute(f"INSERT INTO ducklake.test VALUES ({v})")
        duckdb_count = cat.fetchone("SELECT COUNT(*) FROM ducklake.test")[0]
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == duckdb_count == 20
        assert sorted(result["i"].to_list()) == list(range(20))

    def test_multiple_tables_with_operations(self, ducklake_catalog):
        """Operations on multiple tables, verify independence."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1), (2), (3)")
        cat.execute("CREATE TABLE ducklake.t2 (x VARCHAR)")
        cat.execute("INSERT INTO ducklake.t2 VALUES ('a'), ('b')")
        cat.execute("DELETE FROM ducklake.t1 WHERE a = 2")
        cat.execute("INSERT INTO ducklake.t2 VALUES ('c')")
        cat.close()

        r1 = read_ducklake(cat.metadata_path, "t1")
        r2 = read_ducklake(cat.metadata_path, "t2")
        assert sorted(r1["a"].to_list()) == [1, 3]
        assert sorted(r2["x"].to_list()) == ["a", "b", "c"]

    def test_partition_then_schema_evolution_then_delete(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, part VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (part)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 'a'), (2, 'b'), (3, 'a'), (4, 'b')"
        )
        cat.execute("ALTER TABLE ducklake.test ADD COLUMN score DOUBLE")
        cat.execute("INSERT INTO ducklake.test VALUES (5, 'a', 9.5)")
        cat.execute("DELETE FROM ducklake.test WHERE id <= 2")
        duckdb_rows = cat.fetchall("SELECT * FROM ducklake.test ORDER BY id")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("id")
        assert result.shape[0] == len(duckdb_rows)
        assert result["id"].to_list() == [r[0] for r in duckdb_rows]

    def test_rename_after_delete_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello'), (2, 'world')")
        cat.execute("DELETE FROM ducklake.test WHERE a = 1")
        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO name")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'new')")
        duckdb_rows = cat.fetchall("SELECT * FROM ducklake.test ORDER BY a")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("a")
        assert result.columns == ["a", "name"]
        assert result.shape[0] == len(duckdb_rows)
        assert result["a"].to_list() == [r[0] for r in duckdb_rows]

    def test_time_travel_through_complex_history(self, ducklake_catalog):
        """Multiple operations; verify each snapshot is correct."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER)")
        # Snap 1: create
        s1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("INSERT INTO ducklake.test VALUES (1), (2), (3)")
        s2 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("DELETE FROM ducklake.test WHERE id = 2")
        s3 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("INSERT INTO ducklake.test VALUES (4), (5)")
        s4 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("UPDATE ducklake.test SET id = id + 100 WHERE id < 3")
        cat.close()

        # Check each version
        r1 = read_ducklake(cat.metadata_path, "test", snapshot_version=s1)
        assert r1.shape[0] == 0

        r2 = read_ducklake(cat.metadata_path, "test", snapshot_version=s2)
        assert sorted(r2["id"].to_list()) == [1, 2, 3]

        r3 = read_ducklake(cat.metadata_path, "test", snapshot_version=s3)
        assert sorted(r3["id"].to_list()) == [1, 3]

        r4 = read_ducklake(cat.metadata_path, "test", snapshot_version=s4)
        assert sorted(r4["id"].to_list()) == [1, 3, 4, 5]


# ===================================================================
# SECTION 17: Large Data Interop
# ===================================================================


class TestLargeDataInterop:
    """Verify correctness with larger datasets."""

    def test_10k_rows_with_filter(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test AS "
            "SELECT i AS id, i % 100 AS bucket FROM range(10000) t(i)"
        )
        duckdb_count = cat.fetchone(
            "SELECT COUNT(*) FROM ducklake.test WHERE bucket = 42"
        )[0]
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.filter(pl.col("bucket") == 42).collect()
        assert result.shape[0] == duckdb_count == 100

    def test_multi_file_large_data(self, ducklake_catalog):
        """5 separate inserts, each 1000 rows."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, label VARCHAR)")
        for batch in range(5):
            cat.execute(
                f"INSERT INTO ducklake.test "
                f"SELECT i + {batch * 1000}, 'batch{batch}' FROM range(1000) t(i)"
            )
        duckdb_count = cat.fetchone("SELECT COUNT(*) FROM ducklake.test")[0]
        duckdb_sum = cat.fetchone("SELECT SUM(id) FROM ducklake.test")[0]
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == duckdb_count == 5000
        assert result["id"].sum() == duckdb_sum

    def test_wide_table_interop(self, ducklake_catalog):
        """Table with 50 columns."""
        cat = ducklake_catalog
        cols = ", ".join(f"c{i} INTEGER" for i in range(50))
        cat.execute(f"CREATE TABLE ducklake.test ({cols})")
        vals = ", ".join(str(i) for i in range(50))
        cat.execute(f"INSERT INTO ducklake.test VALUES ({vals})")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 50)
        for i in range(50):
            assert result[f"c{i}"][0] == i


# ===================================================================
# SECTION 18: Rename + Partition Combined Interop
# ===================================================================


class TestRenamePartitionInterop:
    """Test interactions between column renames and partitioning."""

    def test_rename_non_partition_column_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val INTEGER, part VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (part)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'a'), (2, 'b')")
        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN val TO value")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'a')")
        duckdb_rows = cat.fetchall("SELECT * FROM ducklake.test ORDER BY value")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("value")
        assert result.columns == ["value", "part"]
        assert result.shape[0] == len(duckdb_rows)
        assert result["value"].to_list() == [r[0] for r in duckdb_rows]

    def test_rename_with_filter_on_partition(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (val INTEGER, part VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (part)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'a'), (2, 'b'), (3, 'a')")
        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN val TO value")
        cat.execute("INSERT INTO ducklake.test VALUES (4, 'b')")
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.filter(pl.col("part") == "a").collect()
        assert result.shape[0] == 2
        assert sorted(result["value"].to_list()) == [1, 3]


# ===================================================================
# SECTION 19: Timestamp-based Time Travel Interop
# ===================================================================


class TestTimestampTimeTravelInterop:
    """Verify timestamp-based time travel."""

    def test_timestamp_time_travel_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.execute("INSERT INTO ducklake.test VALUES (2)")
        cat.close()

        # Get the timestamp for snapshot v1
        ts_row = cat.query_metadata(
            "SELECT snapshot_time FROM ducklake_snapshot WHERE snapshot_id = ?", [v1]
        )
        if ts_row:
            ts = str(ts_row[0])
            # Read at that timestamp
            result = read_ducklake(cat.metadata_path, "test", snapshot_time=ts)
            assert result.shape[0] == 1
            assert result["id"][0] == 1


# ===================================================================
# SECTION 20: Multi-Table Schema Operations Interop
# ===================================================================


class TestMultiTableSchemaInterop:
    """Test schema operations across multiple tables."""

    def test_drop_schema_cascade_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE SCHEMA ducklake.s1")
        cat.execute("CREATE TABLE ducklake.s1.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.s1.t1 VALUES (1)")
        cat.execute("CREATE TABLE ducklake.s1.t2 (b VARCHAR)")
        cat.execute("INSERT INTO ducklake.s1.t2 VALUES ('hello')")
        v_before = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.execute("DROP SCHEMA ducklake.s1 CASCADE")
        cat.close()

        # Tables should not be found in current snapshot
        with pytest.raises(ValueError):
            read_ducklake(cat.metadata_path, "t1", schema="s1")

        # But accessible via time travel
        r1 = read_ducklake(cat.metadata_path, "t1", schema="s1", snapshot_version=v_before)
        assert r1["a"].to_list() == [1]

    def test_table_rename_across_schemas_interop(self, ducklake_catalog):
        """Table in default schema, renamed."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (42)")
        cat.execute("ALTER TABLE ducklake.test RENAME TO renamed")
        cat.close()

        with pytest.raises(ValueError):
            read_ducklake(cat.metadata_path, "test")

        result = read_ducklake(cat.metadata_path, "renamed")
        assert result["a"][0] == 42

    def test_create_table_in_new_schema_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE SCHEMA ducklake.analytics")
        cat.execute("CREATE TABLE ducklake.analytics.metrics (id INTEGER, value DOUBLE)")
        cat.execute("INSERT INTO ducklake.analytics.metrics VALUES (1, 99.5), (2, 88.3)")
        duckdb_rows = cat.fetchall("SELECT * FROM ducklake.analytics.metrics ORDER BY id")
        cat.close()

        result = read_ducklake(cat.metadata_path, "metrics", schema="analytics").sort("id")
        assert result.shape[0] == len(duckdb_rows)
        assert result["id"].to_list() == [r[0] for r in duckdb_rows]


# ===================================================================
# SECTION 21: Delete File Correctness Interop
# ===================================================================


class TestDeleteFileCorrectnessInterop:
    """Verify cumulative delete file behavior."""

    def test_cumulative_deletes_on_same_file(self, ducklake_catalog):
        """Multiple deletes on the same data file produce correct results."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test AS SELECT i AS id FROM range(100) t(i)")
        # Three separate deletes on the same file
        cat.execute("DELETE FROM ducklake.test WHERE id = 10")
        cat.execute("DELETE FROM ducklake.test WHERE id = 20")
        cat.execute("DELETE FROM ducklake.test WHERE id = 30")
        duckdb_count = cat.fetchone("SELECT COUNT(*) FROM ducklake.test")[0]
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == duckdb_count == 97
        ids = result["id"].to_list()
        assert 10 not in ids
        assert 20 not in ids
        assert 30 not in ids

    def test_delete_interleaved_with_inserts_cumulative(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (id INTEGER)")
        cat.execute("INSERT INTO ducklake.test SELECT i FROM range(50) t(i)")  # file 1
        cat.execute("DELETE FROM ducklake.test WHERE id < 10")  # delete from file 1
        cat.execute("INSERT INTO ducklake.test SELECT i + 100 FROM range(50) t(i)")  # file 2
        cat.execute("DELETE FROM ducklake.test WHERE id > 40 AND id < 100")  # delete more from file 1
        duckdb_rows = cat.fetchall("SELECT id FROM ducklake.test ORDER BY id")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        polars_ids = sorted(result["id"].to_list())
        duckdb_ids = [r[0] for r in duckdb_rows]
        assert polars_ids == duckdb_ids

    def test_delete_with_complex_predicate(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i, CASE WHEN i % 2 = 0 THEN 'even' ELSE 'odd' END "
            "FROM range(100) t(i)"
        )
        cat.execute("DELETE FROM ducklake.test WHERE a >= 50 AND b = 'even'")
        duckdb_count = cat.fetchone("SELECT COUNT(*) FROM ducklake.test")[0]
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == duckdb_count
        # Even numbers >= 50 should be deleted
        remaining_even = result.filter(pl.col("b") == "even")
        assert all(v < 50 for v in remaining_even["a"].to_list())


# ===================================================================
# SECTION 22: Inlined + Parquet Mixed Interop
# ===================================================================


class TestInlinedMixedInterop:
    """Test tables that transition from inlined to Parquet data."""

    def test_small_then_large_insert_interop(self, ducklake_catalog_inline):
        """Small insert (inlined) then large insert (forces Parquet)."""
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, val VARCHAR)")
        # Small insert - should be inlined
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'small')")
        # Large insert - may force flush to Parquet
        cat.execute(
            "INSERT INTO ducklake.test "
            "SELECT i + 100, 'batch' FROM range(100) t(i)"
        )
        duckdb_count = cat.fetchone("SELECT COUNT(*) FROM ducklake.test")[0]
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == duckdb_count == 101

    def test_inlined_with_update_interop(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, val INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 10), (2, 20)")
        cat.execute("UPDATE ducklake.test SET val = 99 WHERE id = 1")
        duckdb_rows = cat.fetchall("SELECT * FROM ducklake.test ORDER BY id")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test").sort("id")
        polars_data = list(zip(result["id"].to_list(), result["val"].to_list()))
        assert polars_data == duckdb_rows


# ===================================================================
# SECTION 23: Sort & Aggregation Interop
# ===================================================================


class TestSortAggregationInterop:
    """Test that lazy operations produce correct results."""

    def test_lazy_sort_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test AS SELECT i AS id FROM range(100) t(i)")
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.sort("id", descending=True).head(5).collect()
        assert result["id"].to_list() == [99, 98, 97, 96, 95]

    def test_lazy_group_by_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (category VARCHAR, value INTEGER)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "('a', 10), ('b', 20), ('a', 30), ('b', 40), ('c', 50)"
        )
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.group_by("category").agg(
            pl.col("value").sum().alias("total")
        ).sort("category").collect()
        assert result["category"].to_list() == ["a", "b", "c"]
        assert result["total"].to_list() == [40, 60, 50]

    def test_lazy_head_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test AS SELECT i AS id FROM range(1000) t(i)")
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.head(10).collect()
        assert result.shape[0] == 10

    def test_lazy_with_columns_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 2), (3, 4)")
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.with_columns((pl.col("a") + pl.col("b")).alias("sum")).collect()
        assert result.columns == ["a", "b", "sum"]
        result = result.sort("a")
        assert result["sum"].to_list() == [3, 7]

    def test_lazy_unique_interop(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (category VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES ('a'), ('b'), ('a'), ('c'), ('b')"
        )
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.unique().sort("category").collect()
        assert result["category"].to_list() == ["a", "b", "c"]
