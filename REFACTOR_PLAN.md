# Arrow Unified API Refactor Plan

## Goal
Refactor ducklake-dataframe to use PyArrow as the internal data representation,
then add pandas support as a thin wrapper. The result is a single repo with
three packages: `ducklake_core` (Arrow-based internals), `ducklake_polars`
(thin Polars wrapper), and `ducklake_pandas` (thin pandas wrapper).

## Environment
- Python venv at `.venv/` — activate with `source .venv/bin/activate`
- `export PATH="$HOME/.local/bin:$PATH"` for uv
- Run tests: `python -m pytest tests/ -x -q --tb=short`
- All 587 tests must pass after each commit
- Branch: `feature/arrow-unified-api`

## Architecture

### Core package: `src/ducklake_core/`
Contains all catalog reading, writing, schema mapping, statistics, and backend
logic. Uses PyArrow internally for all DataFrame operations.

### Polars wrapper: `src/ducklake_polars/`
Thin wrapper that converts `pl.DataFrame ↔ pa.Table` at boundaries.
Contains the Polars-specific `_dataset.py` (PythonDatasetProvider for lazy scan).
Public API (`__init__.py`) stays identical.

### Pandas wrapper: `src/ducklake_pandas/`
Thin wrapper that converts `pd.DataFrame ↔ pa.Table` at boundaries.
Contains `read_ducklake` (eager only, no lazy scan equivalent).
Public API mirrors ducklake_polars where possible.

## Commit Plan

### Commit 1: Extract _backend.py and _catalog.py into ducklake_core

1. Create `src/ducklake_core/__init__.py` (empty)
2. Copy `src/ducklake_polars/_backend.py` → `src/ducklake_core/_backend.py` (no changes needed — already framework-agnostic)
3. Copy `src/ducklake_polars/_catalog.py` → `src/ducklake_core/_catalog.py`
   - Change the one function that returns `pl.DataFrame` (`get_inlined_data`) to return `pyarrow.Table` instead
   - Replace `import polars as pl` with `import pyarrow as pa`
   - Replace `pl.DataFrame(data)` with `pa.table(data)`
   - Replace `pl.concat(frames)` with `pa.concat_tables(frames)`
4. Update `src/ducklake_polars/_catalog.py` to import from `ducklake_core._catalog` and re-export, wrapping Arrow→Polars where needed
5. Update `src/ducklake_polars/_backend.py` to just re-export from `ducklake_core._backend`
6. Run tests — all 587 must pass

### Commit 2: Move _schema.py and _stats.py to core

1. Create `src/ducklake_core/_schema.py`:
   - Replace all `pl.DataType` mappings with `pa.DataType` equivalents
   - `duckdb_type_to_arrow(type_str) -> pa.DataType` (was `duckdb_type_to_polars`)
   - `arrow_type_to_duckdb(dtype) -> str` (was `polars_type_to_duckdb`)
   - `resolve_column_type()` returns `pa.DataType` instead of `pl.DataType`
   - Keep compound type parsing (_parse_struct_fields, _split_top_level_args, etc.)
2. Create `src/ducklake_core/_stats.py`:
   - `_parse_stat_value(value, arrow_type)` — same logic but checks `pa.types.is_integer()`, etc.
   - `build_table_statistics()` returns `pa.Table` instead of `pl.DataFrame`
3. Update `src/ducklake_polars/_schema.py` to be a thin wrapper:
   - Import from core, provide `duckdb_type_to_polars()` by converting `pa.DataType → pl.DataType`
   - Provide `polars_type_to_duckdb()` by converting `pl.DataType → pa.DataType → str`
   - Use `pl.from_arrow()` / `.to_arrow()` for conversions
4. Update `src/ducklake_polars/_stats.py` to wrap core stats, converting Arrow→Polars
5. Run tests — all 587 must pass

### Commit 3: Move _writer.py to core

1. Create `src/ducklake_core/_writer.py`:
   - All DataFrame params become `pa.Table`
   - Replace `df.write_parquet(path)` with `pq.write_table(df, path)`
   - Replace `pl.read_parquet(path)` with `pq.read_table(path)`
   - Replace `df.iter_rows()` with iterating Arrow table rows
   - Replace `df.is_empty()` with `len(df) == 0`
   - Replace `df.schema` with `df.schema` (Arrow tables have schema too)
   - Replace `df.null_count()` with `col.null_count`
   - Replace `pl.concat()` with `pa.concat_tables()`
   - Predicates: accept `Callable[[pa.Table], pa.Array]` (returns boolean array)
   - Stats: use Arrow compute (`pc.min`, `pc.max`, etc.)
   - Replace `df.group_by()` with Arrow-based grouping
2. Update `src/ducklake_polars/_writer.py` to be a thin wrapper:
   - Import from core
   - Convert `pl.DataFrame → pa.Table` on input
   - Convert `pl.Expr` predicates to Arrow-compatible callables
   - The `polars_type_to_duckdb` calls route through the polars schema wrapper
3. Run tests — all 587 must pass

### Commit 4: Move _catalog_api.py to core

1. Create `src/ducklake_core/_catalog_api.py`:
   - All methods that return DataFrames now return `pa.Table`
   - Change data feed methods use Arrow tables internally
2. Update `src/ducklake_polars/_catalog_api.py` to wrap core, converting Arrow→Polars
3. Run tests — all 587 must pass

### Commit 5: Rebuild ducklake_polars __init__.py as thin wrapper

1. Update `src/ducklake_polars/__init__.py`:
   - Top-level functions (`read_ducklake`, `write_ducklake`, etc.) call core functions
   - Convert `pl.DataFrame ↔ pa.Table` at boundaries
   - `scan_ducklake` stays Polars-specific (uses PythonDatasetProvider)
2. `_dataset.py` stays mostly Polars-specific but uses core for catalog reading
3. Run tests — all 587 must pass

### Commit 6: Add ducklake_pandas package

1. Create `src/ducklake_pandas/__init__.py`:
   - `read_ducklake(path, table, ...) -> pd.DataFrame`
   - `write_ducklake(df, path, table, ...) -> None`
   - `delete_ducklake(path, table, predicate, ...) -> int`
   - `update_ducklake(path, table, updates, predicate, ...) -> int`
   - `merge_ducklake(path, table, source_df, ...) -> tuple`
   - All DDL functions (create/drop table, schema, view, alter, etc.)
   - `DuckLakeCatalog` wrapper returning pd.DataFrames
   - Predicates: `Callable[[pd.DataFrame], pd.Series[bool]]`
   - Convert at boundaries: `pd.DataFrame ↔ pa.Table`
2. Create `src/ducklake_pandas/_dataset.py`:
   - Simple `read()` method (eager, no lazy scan)
   - Reads via core, converts Arrow→pandas
3. Update `pyproject.toml` to declare both packages

### Commit 7: Add pandas test suite

1. Create `tests_pandas/` directory
2. Create `tests_pandas/conftest.py` — same fixtures but returns pandas DataFrames
3. Port all test files from `tests/` to `tests_pandas/`:
   - Replace `pl.DataFrame` with `pd.DataFrame`
   - Replace `pl.col("x") > 5` with `lambda df: df["x"] > 5`
   - Replace `polars.testing.assert_frame_equal` with `pandas.testing.assert_frame_equal`
   - Replace `scan_ducklake` tests with `read_ducklake` (no lazy scan in pandas)
   - Replace `pl.Series` with `pd.Series`
   - Adjust type assertions (pl.Int32 → np.int32, etc.)
4. Run both test suites — all must pass

## Key Type Mappings (Arrow ↔ Polars ↔ Pandas)

| DuckDB       | Arrow              | Polars            | Pandas          |
|--------------|--------------------|-------------------|-----------------|
| boolean      | pa.bool_()         | pl.Boolean        | "boolean"       |
| int8         | pa.int8()          | pl.Int8           | "Int8"          |
| int32        | pa.int32()         | pl.Int32          | "Int32"         |
| int64        | pa.int64()         | pl.Int64          | "Int64"         |
| float32      | pa.float32()       | pl.Float32        | "Float32"       |
| float64      | pa.float64()       | pl.Float64        | "Float64"       |
| varchar      | pa.string()        | pl.String         | "object"        |
| date         | pa.date32()        | pl.Date           | "object"        |
| timestamp    | pa.timestamp("us") | pl.Datetime("us") | "datetime64[us]"|
| blob         | pa.binary()        | pl.Binary         | "object"        |
| list         | pa.list_(...)      | pl.List(...)      | "object"        |
| struct       | pa.struct(...)     | pl.Struct(...)    | "object"        |

## Conversion Functions

```python
# Polars ↔ Arrow (zero-copy)
arrow_table = polars_df.to_arrow()
polars_df = pl.from_arrow(arrow_table)

# Pandas ↔ Arrow (near zero-copy)
arrow_table = pa.Table.from_pandas(pandas_df)
pandas_df = arrow_table.to_pandas()

# Arrow type ↔ Polars type
# Use mapping dicts, not runtime conversion
```

## Important Notes

- The Polars PythonDatasetProvider (`_dataset.py`) is inherently Polars-specific
  and uses `scan_parquet` with Polars-specific kwargs (`_deletion_files`,
  `_table_statistics`, `ScanCastOptions`). This CANNOT be shared with pandas.
  Keep it in `ducklake_polars`.

- Polars predicates (`pl.Expr`) need to be converted to Arrow-compatible
  predicates at the wrapper boundary. The core should accept
  `Callable[[pa.Table], pa.ChunkedArray]` where the result is a boolean array.

- For the writer's `_compute_file_column_stats`, use `pyarrow.compute` functions
  (`pc.min`, `pc.max`, `pc.sum`, `pc.is_nan`).

- The `_cast_inlined_to_schema` function needs Arrow-based type casting.

- `pa.Table` supports `.to_pydict()` for iterating rows.
