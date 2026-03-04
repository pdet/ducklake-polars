"""DuckLake-Pandas: Pandas integration for DuckLake catalogs."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable

try:
    import pandas  # noqa: F401
except ImportError as _e:
    raise ImportError(
        "Pandas is required for ducklake_pandas. "
        "Install with: pip install ducklake-dataframe[pandas]"
    ) from _e

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    import pandas as pd

from ducklake_pandas._catalog_api import DuckLakeCatalog
from ducklake_core._catalog import DuckLakeCatalogReader
from ducklake_core._writer import TransactionConflictError

__all__ = [
    "read_ducklake",
    "write_ducklake",
    "TransactionConflictError",
    "create_ducklake_table",
    "delete_ducklake",
    "update_ducklake",
    "merge_ducklake",
    "create_table_as_ducklake",
    "alter_ducklake_add_column",
    "alter_ducklake_drop_column",
    "alter_ducklake_rename_column",
    "alter_ducklake_set_type",
    "alter_ducklake_set_partitioned_by",
    "alter_ducklake_set_sort_keys",
    "alter_ducklake_reset_sort_keys",
    "drop_ducklake_table",
    "create_ducklake_schema",
    "drop_ducklake_schema",
    "rename_ducklake_table",
    "expire_snapshots",
    "vacuum_ducklake",
    "create_ducklake_view",
    "drop_ducklake_view",
    "set_ducklake_table_tag",
    "set_ducklake_column_tag",
    "delete_ducklake_table_tag",
    "delete_ducklake_column_tag",
    "list_tables",
    "list_snapshots",
    "catalog_info",
    "DuckLakeCatalog",
]


def _can_skip_file_by_partition(
    file_id: int,
    partition_col_map: dict[int, str],
    file_partition_values: dict[int, dict[int, str | None]],
    predicate: Callable,
    column_names: list[str],
) -> bool:
    """Check if a file can be skipped based on partition values.

    Builds a single-row DataFrame from the file's partition values and
    evaluates the predicate on it.  If the predicate returns False for
    that row, the file can be safely skipped.

    Parameters
    ----------
    file_id
        The data_file_id to check.
    partition_col_map
        Mapping from partition_key_index to column_name.
    file_partition_values
        Mapping from data_file_id -> {partition_key_index -> value}.
    predicate
        User-supplied predicate callable.
    column_names
        All column names in the table schema.

    Returns
    -------
    bool
        True if the file can be skipped.
    """
    import pandas as pd

    if not partition_col_map or file_id not in file_partition_values:
        return False

    pv = file_partition_values[file_id]  # {key_index -> value}
    row_data: dict[str, list] = {}
    for key_idx, col_name in partition_col_map.items():
        val = pv.get(key_idx)
        row_data[col_name] = [val]

    # Do NOT fill non-partition columns. If the predicate references them
    # it will raise a KeyError, which we catch below — meaning the file
    # cannot be pruned by partition values alone. This avoids the problem
    # where None/NaN comparisons silently return False and cause incorrect
    # skipping.

    try:
        probe = pd.DataFrame(row_data)
        # Attempt to cast partition columns to numeric where possible so
        # that predicates using numeric comparisons work correctly.
        for col_name in partition_col_map.values():
            try:
                probe[col_name] = pd.to_numeric(probe[col_name])
            except (ValueError, TypeError):
                pass
        mask = predicate(probe)
        # If the predicate returns all False, this file can be skipped.
        if hasattr(mask, "__iter__"):
            return not mask.any()
        return not bool(mask)
    except Exception:
        # If we can't evaluate the predicate, don't skip the file.
        return False


def _can_skip_file_by_stats(
    file_id: int,
    file_stats: dict[int, dict[int, tuple[str | None, str | None]]],
    col_id_to_name: dict[int, str],
    predicate: Callable,
    column_names: list[str],
) -> bool:
    """Check if a file can be skipped based on column min/max statistics.

    For each column with stats, we build a two-row DataFrame containing
    the min and max values.  If the predicate returns False for both rows,
    no row in the file can match and it can be skipped.

    This is best-effort: if evaluation fails, we don't skip.
    """
    import pandas as pd

    if file_id not in file_stats:
        return False

    stats = file_stats[file_id]  # {column_id -> (min_val, max_val)}
    if not stats:
        return False

    # Build a two-row DataFrame: row 0 = all mins, row 1 = all maxes
    row_data: dict[str, list] = {}
    has_useful_stats = False
    for col_id, (min_val, max_val) in stats.items():
        col_name = col_id_to_name.get(col_id)
        if col_name is None:
            continue
        if min_val is not None and max_val is not None:
            has_useful_stats = True
            row_data[col_name] = [min_val, max_val]
        else:
            row_data[col_name] = [None, None]

    if not has_useful_stats:
        return False

    # Fill missing columns with None
    for col in column_names:
        if col not in row_data:
            row_data[col] = [None, None]

    try:
        probe = pd.DataFrame(row_data)
        # Cast to numeric where possible
        for col in probe.columns:
            try:
                probe[col] = pd.to_numeric(probe[col])
            except (ValueError, TypeError):
                pass
        mask = predicate(probe)
        if hasattr(mask, "__iter__"):
            return not mask.any()
        return not bool(mask)
    except Exception:
        return False


def read_ducklake(
    path: str | Path,
    table: str,
    *,
    schema: str = "main",
    columns: list[str] | None = None,
    predicate: Callable[[pd.DataFrame], pd.Series] | None = None,
    snapshot_version: int | None = None,
    snapshot_time: datetime | str | None = None,
    data_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Read a DuckLake table into a Pandas DataFrame.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db),
        or a PostgreSQL connection string (e.g., "postgresql://user:pass@host/dbname").
    table
        Name of the table to read.
    schema
        Schema name (default: "main").
    columns
        Columns to select. If None, reads all columns.
    predicate
        An optional callable that accepts a ``pd.DataFrame`` and returns a
        boolean ``pd.Series``. When provided, partition pruning and
        file-level statistics pruning are applied to skip files that
        cannot match the predicate. The predicate is also applied to
        the final result to filter rows.
    snapshot_version
        Read the table at a specific snapshot version.
    snapshot_time
        Read the table at a specific timestamp.
    data_path
        Override the data path stored in the catalog.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If both snapshot_version and snapshot_time are specified, or if the
        table or snapshot is not found.
    """
    if snapshot_version is not None and snapshot_time is not None:
        msg = "Cannot specify both snapshot_version and snapshot_time"
        raise ValueError(msg)

    import pyarrow as pa
    import pyarrow.parquet as pq

    import ducklake_core._storage as storage
    from ducklake_core._catalog import DuckLakeCatalogReader, ColumnHistoryEntry
    from ducklake_core._schema import resolve_column_type

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    def _is_active_at(entry: ColumnHistoryEntry, snapshot: int) -> bool:
        """Check if a column history entry was active at a given snapshot."""
        return entry.begin_snapshot <= snapshot and (
            entry.end_snapshot is None or entry.end_snapshot > snapshot
        )

    def _get_physical_name(
        column_id: int,
        file_begin_snapshot: int,
        history: list[ColumnHistoryEntry],
    ) -> str | None:
        """Get the physical column name that was active when the file was written."""
        for entry in history:
            if entry.column_id != column_id:
                continue
            if _is_active_at(entry, file_begin_snapshot):
                return entry.column_name
        return None

    with DuckLakeCatalogReader(metadata_path, data_path_override=dp) as reader:
        # Resolve snapshot
        if snapshot_version is not None:
            snap = reader.get_snapshot_at_version(snapshot_version)
        elif snapshot_time is not None:
            snap = reader.get_snapshot_at_time(snapshot_time)
        else:
            snap = reader.get_current_snapshot()

        table_info = reader.get_table(table, schema, snap.snapshot_id)
        all_columns = reader.get_all_columns(table_info.table_id, snap.snapshot_id)
        top_columns = [c for c in all_columns if c.parent_column is None]
        column_names = [c.column_name for c in top_columns]
        # Map column_id -> column_name for current schema (used for field_id matching)
        col_id_to_name: dict[int, str] = {c.column_id: c.column_name for c in top_columns}

        # Get column history for rename detection
        column_history = reader.get_column_history(table_info.table_id)
        top_history = [e for e in column_history if e.parent_column is None]

        # Build Arrow schema for empty table creation
        arrow_fields = []
        for col in top_columns:
            col_type = resolve_column_type(col.column_id, col.column_type, all_columns)
            arrow_fields.append(pa.field(col.column_name, col_type))
        arrow_schema = pa.schema(arrow_fields)

        # Read data files
        data_files = reader.get_data_files(table_info.table_id, snap.snapshot_id)
        delete_files = reader.get_delete_files(table_info.table_id, snap.snapshot_id)

        # --- Partition & stats pruning ---
        # Build partition metadata if a predicate is provided
        partition_col_map: dict[int, str] = {}  # key_index -> column_name
        file_partition_values: dict[int, dict[int, str | None]] = {}  # file_id -> {key_index -> value}
        file_stats_map: dict[int, dict[int, tuple[str | None, str | None]]] = {}  # file_id -> {col_id -> (min, max)}

        if predicate is not None and data_files:
            # Get partition info
            part_info = reader.get_partition_info(table_info.table_id, snap.snapshot_id)
            if part_info is not None:
                part_cols = reader.get_partition_columns(part_info.partition_id, table_info.table_id)
                # Only use identity-transform partitions for pruning
                for pc in part_cols:
                    if pc.transform == "identity":
                        col_name = col_id_to_name.get(pc.column_id)
                        if col_name is not None:
                            partition_col_map[pc.partition_key_index] = col_name

                if partition_col_map:
                    file_ids = [f.data_file_id for f in data_files]
                    pv_list = reader.get_file_partition_values(table_info.table_id, file_ids)
                    for pv in pv_list:
                        file_partition_values.setdefault(pv.data_file_id, {})[pv.partition_key_index] = pv.partition_value

            # Get column stats for all files
            file_ids = [f.data_file_id for f in data_files]
            col_ids = [c.column_id for c in top_columns]
            stats_list = reader.get_column_stats(table_info.table_id, file_ids, col_ids)
            for s in stats_list:
                file_stats_map.setdefault(s.data_file_id, {})[s.column_id] = (s.min_value, s.max_value)

        # Build delete file mapping: data_file_id -> list of (path, partial_max)
        del_map: dict[int, list[tuple[str, int | None]]] = {}
        for df in delete_files:
            del_path = reader.resolve_data_file_path(df.path, df.path_is_relative, table_info)
            del_map.setdefault(df.data_file_id, []).append((del_path, df.partial_max))

        frames: list[pa.Table] = []

        for f in data_files:
            # Partition pruning: skip files whose partition values don't match
            if predicate is not None and _can_skip_file_by_partition(
                f.data_file_id, partition_col_map, file_partition_values,
                predicate, column_names,
            ):
                continue

            # Stats pruning: skip files where min/max proves no rows match
            if predicate is not None and _can_skip_file_by_stats(
                f.data_file_id, file_stats_map, col_id_to_name,
                predicate, column_names,
            ):
                continue
            file_path = reader.resolve_data_file_path(f.path, f.path_is_relative, table_info)
            tbl = storage.read_parquet(file_path)

            # Apply delete files (Iceberg position-delete format)
            if f.data_file_id in del_map:
                all_positions: list[int] = []
                for del_path, partial_max in del_map[f.data_file_id]:
                    del_tbl = storage.read_parquet(del_path)
                    # Filter cumulative delete files when time-traveling.
                    # v0.4 catalogs tag each position with the snapshot that
                    # deleted it via _ducklake_internal_snapshot_id.
                    if (
                        partial_max is not None
                        and partial_max > snap.snapshot_id
                        and "_ducklake_internal_snapshot_id" in del_tbl.column_names
                    ):
                        import pyarrow.compute as pc

                        mask = pc.less_equal(
                            del_tbl.column("_ducklake_internal_snapshot_id"),
                            snap.snapshot_id,
                        )
                        del_tbl = del_tbl.filter(mask)
                    if "pos" in del_tbl.column_names:
                        positions = del_tbl.column("pos").to_pylist()
                        all_positions.extend(
                            p for p in positions if 0 <= p < tbl.num_rows
                        )
                if all_positions:
                    # Create mask: True = keep, then filter
                    keep = [True] * tbl.num_rows
                    for p in all_positions:
                        keep[p] = False
                    tbl = tbl.filter(pa.array(keep, type=pa.bool_()))

            # Build column mapping: try field_id first, fall back to name
            file_field_map: dict[int, int] = {}
            file_schema = tbl.schema
            for i in range(len(file_schema)):
                field = file_schema.field(i)
                if field.metadata and b"PARQUET:field_id" in field.metadata:
                    fid = int(field.metadata[b"PARQUET:field_id"])
                    file_field_map[fid] = i

            # Also build name → index map for fallback (map_by_name)
            file_name_map: dict[str, int] = {
                file_schema.field(i).name: i for i in range(len(file_schema))
            }

            # Build rename map: {physical_name -> current_name} using column history
            # This handles the case where a column was renamed after the file was written
            current_ids = {c.column_id for c in top_columns}
            current_name_set = {c.column_name for c in top_columns}
            rename_map: dict[str, str] = {}
            for col in top_columns:
                physical = _get_physical_name(col.column_id, f.begin_snapshot, top_history)
                if physical is not None and physical != col.column_name:
                    rename_map[physical] = col.column_name
            # Handle dropped column name conflicts: a dropped column's physical name
            # collides with a new column's name (different column_id, same name)
            for entry in top_history:
                if entry.column_id in current_ids:
                    continue  # Not dropped
                if _is_active_at(entry, f.begin_snapshot):
                    if entry.column_name in current_name_set and entry.column_name not in rename_map:
                        rename_map[entry.column_name] = f"__ducklake_dropped_{entry.column_id}__"

            # Handle struct field renames
            struct_field_renames: dict[str, list[str]] = {}
            child_history = [e for e in column_history if e.parent_column is not None]
            if child_history:
                struct_cols = {
                    c.column_id: c
                    for c in all_columns
                    if c.column_type == "struct" and c.parent_column is None
                }
                for struct_id, struct_col in struct_cols.items():
                    current_children = sorted(
                        [c for c in all_columns if c.parent_column == struct_id],
                        key=lambda c: c.column_order,
                    )
                    physical_children = sorted(
                        [e for e in child_history
                         if e.parent_column == struct_id
                         and _is_active_at(e, f.begin_snapshot)],
                        key=lambda e: e.column_order,
                    )
                    if len(current_children) != len(physical_children):
                        continue
                    physical_names = [e.column_name for e in physical_children]
                    current_names = [c.column_name for c in current_children]
                    if physical_names != current_names:
                        struct_field_renames[struct_col.column_name] = current_names

            # Select columns; prefer field_id, fall back to rename map, then name match
            selected_cols = []
            for col_info in top_columns:
                file_idx = file_field_map.get(col_info.column_id)
                if file_idx is None:
                    # Try current name first
                    file_idx = file_name_map.get(col_info.column_name)
                if file_idx is None:
                    # Try physical (old) name via rename map
                    for old_name, new_name in rename_map.items():
                        if new_name == col_info.column_name:
                            file_idx = file_name_map.get(old_name)
                            if file_idx is not None:
                                break
                # Check for dropped column conflict: if this file_idx points to a
                # column that was dropped and re-added with a different column_id,
                # we should fill with nulls instead
                if file_idx is not None:
                    file_col_name = file_schema.field(file_idx).name
                    mapped_target = rename_map.get(file_col_name)
                    if mapped_target is not None and mapped_target.startswith("__ducklake_dropped_"):
                        # This column in the file belongs to a dropped column
                        if file_col_name != col_info.column_name or mapped_target != col_info.column_name:
                            file_idx = None
                if file_idx is not None:
                    col_data = tbl.column(file_idx)
                    # Apply struct field renames if needed
                    if col_info.column_name in struct_field_renames:
                        new_field_names = struct_field_renames[col_info.column_name]
                        # Rebuild the struct array with renamed fields
                        if pa.types.is_struct(col_data.type):
                            old_type = col_data.type
                            new_fields = []
                            for idx, new_name in enumerate(new_field_names):
                                if idx < old_type.num_fields:
                                    old_field = old_type.field(idx)
                                    new_fields.append(pa.field(new_name, old_field.type, old_field.nullable))
                            if len(new_fields) == old_type.num_fields:
                                chunks = []
                                for chunk in col_data.chunks:
                                    arrays = [chunk.field(i) for i in range(old_type.num_fields)]
                                    new_struct = pa.StructArray.from_arrays(
                                        arrays, fields=new_fields, mask=chunk.is_null()
                                    )
                                    chunks.append(new_struct)
                                col_data = pa.chunked_array(chunks)
                    target_type = arrow_schema.field(col_info.column_name).type
                    if col_data.type != target_type:
                        try:
                            col_data = col_data.cast(target_type, safe=False)
                        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                            pass
                    selected_cols.append(col_data)
                else:
                    # Column doesn't exist in this file — fill with nulls
                    col_type = arrow_schema.field(col_info.column_name).type
                    null_arr = pa.nulls(tbl.num_rows, type=col_type)
                    selected_cols.append(null_arr)
            tbl = pa.table(
                {name: col for name, col in zip(column_names, selected_cols)}
            )
            frames.append(tbl)

        # Read inlined data
        inlined = reader.read_inlined_data(
            table_info.table_id,
            snap.snapshot_id,
            column_names,
        )
        if inlined is not None and inlined.num_rows > 0:
            # Cast inlined data to target schema (inlined values stored as strings)
            cast_cols = []
            for col_name in inlined.column_names:
                col = inlined.column(col_name)
                if col_name in arrow_schema.names:
                    target_type = arrow_schema.field(col_name).type
                    if col.type != target_type:
                        try:
                            col = col.cast(target_type, safe=False)
                        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                            pass
                cast_cols.append(col)
            inlined = pa.table(
                {name: col for name, col in zip(inlined.column_names, cast_cols)}
            )
            frames.append(inlined)

    if not frames:
        # Return empty DataFrame with correct schema
        result = arrow_schema.empty_table().to_pandas()
    else:
        result = pa.concat_tables(frames, promote_options="permissive").to_pandas()

    # Apply predicate to final result for precise row-level filtering
    if predicate is not None and len(result) > 0:
        mask = predicate(result)
        result = result.loc[mask].reset_index(drop=True)

    if columns is not None:
        result = result[columns]

    return result


def write_ducklake(
    df: pd.DataFrame,
    path: str | Path,
    table: str,
    *,
    schema: str = "main",
    mode: str = "error",
    data_path: str | Path | None = None,
    data_inlining_row_limit: int = 0,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Write a Pandas DataFrame to a DuckLake table.

    Parameters
    ----------
    df
        DataFrame to write.
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
    table
        Name of the table to write to.
    schema
        Schema name (default: "main").
    mode
        Write mode:
        - ``"error"`` (default): Fail if the table already exists.
        - ``"append"``: Append data to an existing table. Creates the
          table if it does not exist.
        - ``"overwrite"``: Replace all data in the table. Creates the
          table if it does not exist.
    data_path
        Override the data path stored in the catalog.
    data_inlining_row_limit
        Maximum number of rows to store inline in the metadata catalog
        instead of writing Parquet files.

    Raises
    ------
    ValueError
        If mode is ``"error"`` and the table already exists, or if the
        mode is not recognized.
    """
    if mode not in ("error", "append", "overwrite"):
        msg = f"Invalid write mode '{mode}'. Must be 'error', 'append', or 'overwrite'."
        raise ValueError(msg)

    import pyarrow as pa

    from ducklake_pandas._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None
    arrow_table = pa.Table.from_pandas(df)

    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=dp,
        data_inlining_row_limit=data_inlining_row_limit,
        author=author,
        commit_message=commit_message,
    ) as writer:
        snapshot_info = writer._get_latest_snapshot()
        if snapshot_info is None:
            snap_id = -1
        else:
            snap_id, _sv, _nci, _nfi = snapshot_info
        table_id = writer._table_exists(table, schema, snap_id)

        if mode == "error":
            if table_id is not None:
                msg = f"Table '{schema}.{table}' already exists (mode='error')"
                raise ValueError(msg)
            # Infer Arrow schema from the DataFrame
            arrow_schema = {field.name: field.type for field in arrow_table.schema}
            writer._core.create_table(table, arrow_schema, schema_name=schema)
            if len(df) > 0:
                writer._core.insert_data(arrow_table, table, schema_name=schema)

        elif mode == "append":
            if table_id is None:
                arrow_schema = {field.name: field.type for field in arrow_table.schema}
                writer._core.create_table(table, arrow_schema, schema_name=schema)
            if len(df) > 0:
                writer._core.insert_data(arrow_table, table, schema_name=schema)

        elif mode == "overwrite":
            if table_id is None:
                arrow_schema = {field.name: field.type for field in arrow_table.schema}
                writer._core.create_table(table, arrow_schema, schema_name=schema)
                if len(df) > 0:
                    writer._core.insert_data(arrow_table, table, schema_name=schema)
            else:
                writer._core.overwrite_data(arrow_table, table, schema_name=schema)


def create_ducklake_table(
    path: str | Path,
    table: str,
    schema_dict: dict[str, str],
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Create a new table in a DuckLake catalog.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
    table
        Name of the table to create.
    schema_dict
        Schema for the new table as a dict mapping column names to
        DuckDB type strings (e.g., ``{"id": "int64", "name": "varchar"}``).
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.

    Raises
    ------
    ValueError
        If the table already exists.
    """
    from ducklake_pandas._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.create_table(table, schema_dict, schema_name=schema)


def delete_ducklake(
    path: str | Path,
    table: str,
    predicate: Callable[[pd.DataFrame], pd.Series] | bool,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    data_inlining_row_limit: int = 0,
    author: str | None = None,
    commit_message: str | None = None,
) -> int:
    """
    Delete rows matching a predicate from a DuckLake table.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
    table
        Name of the table to delete from.
    predicate
        A callable that accepts a ``pd.DataFrame`` and returns a boolean
        ``pd.Series``. Rows where the series is ``True`` will be deleted.
        Pass ``True`` to delete all rows.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.
    data_inlining_row_limit
        Maximum number of inlined rows.

    Returns
    -------
    int
        The number of rows deleted.
    """
    from ducklake_pandas._writer import DuckLakeCatalogWriter

    # Handle boolean predicate: True means delete all rows
    if predicate is True:
        import pandas as _pd
        predicate = lambda df: _pd.Series([True] * len(df), index=df.index)
    elif predicate is False:
        import pandas as _pd
        predicate = lambda df: _pd.Series([False] * len(df), index=df.index)

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=dp,
        data_inlining_row_limit=data_inlining_row_limit,
        author=author,
        commit_message=commit_message,
    ) as writer:
        return writer.delete_data(predicate, table, schema_name=schema)


def update_ducklake(
    path: str | Path,
    table: str,
    updates: dict[str, Any],
    predicate: Callable[[pd.DataFrame], pd.Series],
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    data_inlining_row_limit: int = 0,
    author: str | None = None,
    commit_message: str | None = None,
) -> int:
    """
    Update rows matching a predicate in a DuckLake table.

    Parameters
    ----------
    updates
        Dictionary mapping column names to new values. Values can be
        literals (int, str, float, ...) or callables that accept a
        ``pd.DataFrame`` and return a ``pd.Series``.
    predicate
        A callable that accepts a ``pd.DataFrame`` and returns a boolean
        ``pd.Series``. Rows where the series is ``True`` will be updated.
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
    table
        Name of the table to update.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.
    data_inlining_row_limit
        Maximum number of inlined rows.

    Returns
    -------
    int
        The number of rows updated.
    """
    from ducklake_pandas._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=dp,
        data_inlining_row_limit=data_inlining_row_limit,
        author=author,
        commit_message=commit_message,
    ) as writer:
        return writer.update_data(updates, predicate, table, schema_name=schema)


def merge_ducklake(
    path: str | Path,
    table: str,
    source_df: pd.DataFrame,
    on: str | list[str],
    *,
    when_matched_update: dict[str, Any] | bool | None = None,
    when_not_matched_insert: bool = True,
    schema: str = "main",
    data_path: str | Path | None = None,
    data_inlining_row_limit: int = 0,
    author: str | None = None,
    commit_message: str | None = None,
) -> tuple[int, int]:
    """
    Merge a source DataFrame into an existing DuckLake table.

    Parameters
    ----------
    source_df
        Source DataFrame to merge from.
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
    table
        Name of the target table.
    on
        Column name(s) to match on.
    when_matched_update
        - ``None``: matched target rows are left untouched.
        - ``True``: replace matched target rows with source rows.
        - ``dict``: update matched target rows with these values
          (literal or callable).
    when_not_matched_insert
        If True (default), source rows that have no match in the
        target are inserted.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.
    data_inlining_row_limit
        Maximum number of inlined rows.

    Returns
    -------
    tuple[int, int]
        ``(rows_updated, rows_inserted)``.
    """
    from ducklake_pandas._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=dp,
        data_inlining_row_limit=data_inlining_row_limit,
        author=author,
        commit_message=commit_message,
    ) as writer:
        return writer.merge_data(
            source_df,
            table,
            on,
            when_matched_update=when_matched_update,
            when_not_matched_insert=when_not_matched_insert,
            schema_name=schema,
        )


def create_table_as_ducklake(
    df: pd.DataFrame,
    path: str | Path,
    table: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    data_inlining_row_limit: int = 0,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Create a new table and insert data in a single snapshot.

    Equivalent to ``CREATE TABLE ... AS SELECT ...`` — the table schema
    is inferred from the DataFrame and the data is written atomically.

    Parameters
    ----------
    df
        DataFrame whose schema defines the new table and whose rows
        are the initial data.
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
    table
        Name of the table to create.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.
    data_inlining_row_limit
        Maximum number of rows to store inline in the metadata catalog.

    Raises
    ------
    ValueError
        If the table already exists.
    """
    from ducklake_pandas._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path,
        data_path_override=dp,
        data_inlining_row_limit=data_inlining_row_limit,
        author=author,
        commit_message=commit_message,
    ) as writer:
        writer.create_table_with_data(table, df, schema_name=schema)


def alter_ducklake_add_column(
    path: str | Path,
    table: str,
    column_name: str,
    dtype_str: str,
    *,
    default: object = None,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Add a column to a DuckLake table.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
    table
        Name of the table to alter.
    column_name
        Name of the new column.
    dtype_str
        DuckDB type string for the new column (e.g., ``"int64"``, ``"varchar"``).
    default
        Default value for the new column.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.

    Raises
    ------
    ValueError
        If the table does not exist or the column already exists.
    """
    from ducklake_pandas._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.add_column(table, column_name, dtype_str, default=default, schema_name=schema)


def alter_ducklake_drop_column(
    path: str | Path,
    table: str,
    column_name: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Drop a column from a DuckLake table.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
    table
        Name of the table to alter.
    column_name
        Name of the column to drop.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.

    Raises
    ------
    ValueError
        If the table or column does not exist.
    """
    from ducklake_pandas._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.drop_column(table, column_name, schema_name=schema)


def alter_ducklake_rename_column(
    path: str | Path,
    table: str,
    old_name: str,
    new_name: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Rename a column in a DuckLake table.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
    table
        Name of the table to alter.
    old_name
        Current name of the column.
    new_name
        New name for the column.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.

    Raises
    ------
    ValueError
        If the table or column does not exist, or if the new name
        already exists.
    """
    from ducklake_pandas._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.rename_column(table, old_name, new_name, schema_name=schema)


def alter_ducklake_set_partitioned_by(
    path: str | Path,
    table: str,
    column_names: list[str],
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Set identity-transform partitioning on a DuckLake table.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
    table
        Name of the table to partition.
    column_names
        Column names to partition by (identity transform).
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.

    Raises
    ------
    ValueError
        If the table or any column does not exist.
    """
    from ducklake_pandas._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.set_partitioned_by(table, column_names, schema_name=schema)


def alter_ducklake_set_sort_keys(
    path: str | Path,
    table: str,
    sort_keys: list[str | tuple[str, str] | tuple[str, str, str]],
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Set sort keys on a DuckLake table.

    Equivalent to ``ALTER TABLE t SET SORTED BY (col1, col2 DESC, ...)``.
    Future writes will sort data by these columns before writing Parquet
    files, improving filter pushdown via Parquet row group statistics.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
    table
        Name of the table.
    sort_keys
        Sort key specifications. Each element can be:

        - ``"col"`` — ascending, nulls last
        - ``("col", "DESC")`` — descending, nulls last
        - ``("col", "ASC", "NULLS_FIRST")`` — ascending, nulls first
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.

    Raises
    ------
    ValueError
        If the table or any column does not exist.
    """
    from ducklake_pandas._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.set_sort_keys(table, sort_keys, schema_name=schema)


def alter_ducklake_reset_sort_keys(
    path: str | Path,
    table: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Remove sort keys from a DuckLake table.

    Equivalent to ``ALTER TABLE t RESET SORTED BY``.
    """
    from ducklake_pandas._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.reset_sort_keys(table, schema_name=schema)


def alter_ducklake_set_type(
    path: str | Path,
    table: str,
    column_name: str,
    new_type: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Change the type of a column in a DuckLake table.

    Existing Parquet files keep their original types; the reader casts
    values from the old type when reading files written before the change.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
    table
        Name of the table to alter.
    column_name
        Name of the column whose type to change.
    new_type
        DuckDB type string for the new column type (e.g., ``"BIGINT"``,
        ``"VARCHAR"``, ``"DOUBLE"``).
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.

    Raises
    ------
    ValueError
        If the table or column does not exist.
    """
    from ducklake_pandas._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.set_column_type(
            table, column_name, new_type, schema_name=schema
        )


def drop_ducklake_table(
    path: str | Path,
    table: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Drop a table from a DuckLake catalog.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
    table
        Name of the table to drop.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.

    Raises
    ------
    ValueError
        If the table does not exist.
    """
    from ducklake_pandas._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.drop_table(table, schema_name=schema)


def rename_ducklake_table(
    path: str | Path,
    old_table: str,
    new_table: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Rename a table in a DuckLake catalog.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
    old_table
        Current name of the table.
    new_table
        New name for the table.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.

    Raises
    ------
    ValueError
        If the old table does not exist or if the new name already exists.
    """
    from ducklake_pandas._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.rename_table(old_table, new_table, schema_name=schema)


def create_ducklake_schema(
    path: str | Path,
    schema_name: str,
    *,
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Create a new schema in a DuckLake catalog.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
    schema_name
        Name of the schema to create.
    data_path
        Override the data path stored in the catalog.

    Raises
    ------
    ValueError
        If the schema already exists.
    """
    from ducklake_pandas._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.create_schema(schema_name)


def drop_ducklake_schema(
    path: str | Path,
    schema_name: str,
    *,
    cascade: bool = False,
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Drop a schema from a DuckLake catalog.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
    schema_name
        Name of the schema to drop.
    cascade
        If True, drop all tables in the schema first.
    data_path
        Override the data path stored in the catalog.

    Raises
    ------
    ValueError
        If the schema does not exist or contains tables (when cascade=False).
    """
    from ducklake_pandas._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.drop_schema(schema_name, cascade=cascade)


def create_ducklake_view(
    path: str | Path,
    view_name: str,
    sql: str,
    *,
    schema: str = "main",
    or_replace: bool = False,
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Create a view in a DuckLake catalog.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db),
        or a PostgreSQL connection string.
    view_name
        Name of the view to create.
    sql
        The SQL definition of the view.
    schema
        Schema name (default: "main").
    or_replace
        If True, replace an existing view with the same name.
    data_path
        Override the data path stored in the catalog.

    Raises
    ------
    ValueError
        If the view already exists and or_replace is False.
    """
    from ducklake_pandas._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.create_view(
            view_name, sql, schema_name=schema, or_replace=or_replace,
        )


def drop_ducklake_view(
    path: str | Path,
    view_name: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Drop a view from a DuckLake catalog.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db),
        or a PostgreSQL connection string.
    view_name
        Name of the view to drop.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.

    Raises
    ------
    ValueError
        If the view does not exist.
    """
    from ducklake_pandas._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.drop_view(view_name, schema_name=schema)


def expire_snapshots(
    path: str | Path,
    *,
    older_than_snapshot: int | None = None,
    keep_last_n: int | None = None,
    data_path: str | Path | None = None,
) -> int:
    """
    Expire old snapshots and clean up associated metadata.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
    older_than_snapshot
        Expire all snapshots with ``snapshot_id < older_than_snapshot``.
    keep_last_n
        Keep the most recent *n* snapshots, expire the rest.
    data_path
        Override the data path stored in the catalog.

    Returns
    -------
    int
        The number of snapshots expired.
    """
    from ducklake_pandas._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(metadata_path, data_path_override=dp) as writer:
        return writer.expire_snapshots(
            older_than_snapshot=older_than_snapshot,
            keep_last_n=keep_last_n,
        )


def vacuum_ducklake(
    path: str | Path,
    *,
    data_path: str | Path | None = None,
) -> int:
    """
    Delete orphaned Parquet files not referenced by any catalog entry.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
        Supports SQLite and PostgreSQL backends.
    data_path
        Override the data path stored in the catalog.

    Returns
    -------
    int
        The number of Parquet files deleted.
    """
    from ducklake_pandas._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(metadata_path, data_path_override=dp) as writer:
        return writer.vacuum()


def set_ducklake_table_tag(
    path: str | Path,
    table: str,
    key: str,
    value: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Set a tag on a DuckLake table.

    Tags are key-value metadata pairs. Setting a tag with an existing key
    overwrites the previous value. The ``comment`` key is interoperable
    with DuckDB's ``COMMENT ON TABLE`` statement.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
    table
        Name of the table.
    key
        Tag key (e.g., ``"comment"``).
    value
        Tag value.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.
    """
    from ducklake_pandas._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.set_table_tag(table, key, value, schema_name=schema)


def set_ducklake_column_tag(
    path: str | Path,
    table: str,
    column: str,
    key: str,
    value: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Set a tag on a column in a DuckLake table.

    Tags are key-value metadata pairs. Setting a tag with an existing key
    overwrites the previous value. The ``comment`` key is interoperable
    with DuckDB's ``COMMENT ON COLUMN`` statement.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
    table
        Name of the table.
    column
        Name of the column.
    key
        Tag key (e.g., ``"comment"``).
    value
        Tag value.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.
    """
    from ducklake_pandas._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.set_column_tag(table, column, key, value, schema_name=schema)


def delete_ducklake_table_tag(
    path: str | Path,
    table: str,
    key: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Remove a tag from a DuckLake table.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
    table
        Name of the table.
    key
        Tag key to remove.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.
    """
    from ducklake_pandas._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.delete_table_tag(table, key, schema_name=schema)


def delete_ducklake_column_tag(
    path: str | Path,
    table: str,
    column: str,
    key: str,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
    author: str | None = None,
    commit_message: str | None = None,
) -> None:
    """
    Remove a tag from a column in a DuckLake table.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (.ducklake or .db).
    table
        Name of the table.
    column
        Name of the column.
    key
        Tag key to remove.
    schema
        Schema name (default: "main").
    data_path
        Override the data path stored in the catalog.
    """
    from ducklake_pandas._writer import DuckLakeCatalogWriter

    metadata_path = os.fspath(path)
    dp = os.fspath(data_path) if data_path is not None else None

    with DuckLakeCatalogWriter(
        metadata_path, data_path_override=dp,
        author=author, commit_message=commit_message,
    ) as writer:
        writer.delete_column_tag(table, column, key, schema_name=schema)


def list_tables(
    path: str | Path,
    *,
    schema: str = "main",
    data_path: str | Path | None = None,
) -> list[str]:
    """List all table names in a DuckLake catalog schema."""
    dp = os.fspath(data_path) if data_path is not None else None
    with DuckLakeCatalogReader(os.fspath(path), data_path_override=dp) as reader:
        return reader.list_tables(schema)


def list_snapshots(
    path: str | Path,
    *,
    limit: int = 20,
    data_path: str | Path | None = None,
) -> list[dict]:
    """List recent snapshots from a DuckLake catalog."""
    dp = os.fspath(data_path) if data_path is not None else None
    with DuckLakeCatalogReader(os.fspath(path), data_path_override=dp) as reader:
        return reader.list_snapshots(limit=limit)


def catalog_info(
    path: str | Path,
    *,
    data_path: str | Path | None = None,
) -> dict:
    """Get catalog summary: version, data_path, table_count, snapshot_count."""
    dp = os.fspath(data_path) if data_path is not None else None
    with DuckLakeCatalogReader(os.fspath(path), data_path_override=dp) as reader:
        return reader.catalog_info()
