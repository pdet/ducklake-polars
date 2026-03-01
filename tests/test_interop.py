"""Cross-implementation compatibility tests: ducklake-polars ↔ DuckDB native ducklake.

Verifies bidirectional compatibility between ducklake-polars and DuckDB's
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
    """DuckDB creates and populates tables; ducklake-polars reads them."""

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
    """ducklake-polars creates and populates tables; DuckDB reads them."""

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
