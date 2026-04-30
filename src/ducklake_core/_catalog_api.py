"""DuckLake catalog utility functions — Arrow-based internals."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc
import ducklake_core._storage as storage
from ducklake_core._catalog import DuckLakeCatalogReader
from ducklake_core._schema import resolve_column_type

if TYPE_CHECKING:
    from pathlib import Path


class DuckLakeCatalog:
    """
    High-level interface for inspecting DuckLake catalog metadata.

    Provides Python equivalents of DuckLake's DuckDB extension utility
    functions (``ducklake_snapshots``, ``ducklake_table_info``, etc.).
    All methods return ``pa.Table``.

    Parameters
    ----------
    path
        Path to the DuckLake metadata catalog file (``.ducklake`` or ``.db``),
        or a PostgreSQL connection string.
    data_path
        Override the data path stored in the catalog.
    """

    def __init__(self, path: str | Path, *, data_path: str | Path | None = None) -> None:
        self._metadata_path = os.fspath(path)
        self._data_path_override = os.fspath(data_path) if data_path is not None else None

    @staticmethod
    def _build_cdf_schema(
        all_columns: list,
        extra_prefix: list[tuple[str, pa.DataType]] | None = None,
    ) -> pa.Schema:
        """Build an Arrow schema for change data feed tables from catalog columns."""
        fields: list[pa.Field] = []
        if extra_prefix:
            for name, dtype in extra_prefix:
                fields.append(pa.field(name, dtype))
        top_level = [c for c in all_columns if c.parent_column is None]
        for col in top_level:
            try:
                col_type = resolve_column_type(
                    col.column_id, col.column_type, all_columns
                )
            except ValueError:
                col_type = pa.string()
            fields.append(pa.field(col.column_name, col_type))
        return pa.schema(fields)

    def _reader(self) -> DuckLakeCatalogReader:
        return DuckLakeCatalogReader(self._metadata_path, data_path_override=self._data_path_override)

    def __enter__(self) -> DuckLakeCatalog:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    # ------------------------------------------------------------------
    # Snapshot functions
    # ------------------------------------------------------------------

    def snapshots(self) -> pa.Table:
        """
        List all snapshots in the catalog.

        Equivalent to ``ducklake_snapshots(catalog)``.

        Returns a table with columns:
        - ``snapshot_id`` (int64)
        - ``snapshot_time`` (string)
        - ``schema_version`` (int64)
        """
        with self._reader() as reader:
            rows = reader.get_all_snapshots()

        schema = pa.schema([
            pa.field("snapshot_id", pa.int64()),
            pa.field("snapshot_time", pa.string()),
            pa.field("schema_version", pa.int64()),
        ])

        if not rows:
            return schema.empty_table()

        return pa.table(
            {
                "snapshot_id": pa.array([r[0] for r in rows], type=pa.int64()),
                "snapshot_time": pa.array([str(r[2]) for r in rows], type=pa.string()),
                "schema_version": pa.array([r[1] for r in rows], type=pa.int64()),
            },
            schema=schema,
        )

    def current_snapshot(self) -> int:
        """
        Get the current (latest) snapshot ID.

        Equivalent to ``ducklake_current_snapshot(catalog)``
        and ``ducklake_last_committed_snapshot(catalog)``.
        """
        with self._reader() as reader:
            snap = reader.get_current_snapshot()
        return snap.snapshot_id

    # ------------------------------------------------------------------
    # Table metadata
    # ------------------------------------------------------------------

    def table_info(self, *, schema: str = "main") -> pa.Table:
        """
        Get per-table storage metadata.

        Equivalent to ``ducklake_table_info(catalog)``.

        Returns a table with columns:
        - ``table_name`` (string)
        - ``table_id`` (int64)
        - ``file_count`` (int64)
        - ``file_size_bytes`` (int64)
        - ``delete_file_count`` (int64)
        - ``delete_row_count`` (int64)
        """
        ti_schema = pa.schema([
            pa.field("table_name", pa.string()),
            pa.field("table_id", pa.int64()),
            pa.field("file_count", pa.int64()),
            pa.field("file_size_bytes", pa.int64()),
            pa.field("delete_file_count", pa.int64()),
            pa.field("delete_row_count", pa.int64()),
        ])

        with self._reader() as reader:
            snap = reader.get_current_snapshot()
            schemas = reader.get_all_schemas(snap.snapshot_id)
            schema_row = None
            for s in schemas:
                if s[1] == schema:
                    schema_row = s
                    break
            if schema_row is None:
                return ti_schema.empty_table()

            tables = reader.get_all_tables(schema_row[0], snap.snapshot_id)
            result_rows = []
            for table_id, table_name in tables:
                data_files = reader.get_data_files(table_id, snap.snapshot_id)
                delete_files = reader.get_delete_files(table_id, snap.snapshot_id)
                file_count = len(data_files)
                file_size = sum(f.file_size_bytes for f in data_files)
                del_count = len(delete_files)
                del_size = sum(f.delete_count for f in delete_files)
                result_rows.append((table_name, table_id, file_count, file_size, del_count, del_size))

        if not result_rows:
            return ti_schema.empty_table()

        return pa.table(
            {
                "table_name": pa.array([r[0] for r in result_rows], type=pa.string()),
                "table_id": pa.array([r[1] for r in result_rows], type=pa.int64()),
                "file_count": pa.array([r[2] for r in result_rows], type=pa.int64()),
                "file_size_bytes": pa.array([r[3] for r in result_rows], type=pa.int64()),
                "delete_file_count": pa.array([r[4] for r in result_rows], type=pa.int64()),
                "delete_row_count": pa.array([r[5] for r in result_rows], type=pa.int64()),
            },
            schema=ti_schema,
        )

    def list_files(
        self,
        table: str,
        *,
        schema: str = "main",
        snapshot_version: int | None = None,
    ) -> pa.Table:
        """
        List data files and delete files for a table.

        Equivalent to ``ducklake_list_files(catalog, table)``.

        Returns a table with columns:
        - ``data_file`` (string) — resolved path to the data file
        - ``data_file_size_bytes`` (int64)
        - ``delete_file`` (string or null) — resolved path to delete file
        - ``delete_row_count`` (int64 or null)
        """
        lf_schema = pa.schema([
            pa.field("data_file", pa.string()),
            pa.field("data_file_size_bytes", pa.int64()),
            pa.field("delete_file", pa.string()),
            pa.field("delete_row_count", pa.int64()),
        ])

        with self._reader() as reader:
            if snapshot_version is not None:
                snap = reader.get_snapshot_at_version(snapshot_version)
            else:
                snap = reader.get_current_snapshot()

            table_info = reader.get_table(table, schema, snap.snapshot_id)
            data_files = reader.get_data_files(table_info.table_id, snap.snapshot_id)
            delete_files = reader.get_delete_files(table_info.table_id, snap.snapshot_id)

            # Build delete file mapping: data_file_id -> list of delete file paths
            del_map: dict[int, list[tuple[str, int]]] = {}
            for df in delete_files:
                path = reader.resolve_data_file_path(df.path, df.path_is_relative, table_info)
                del_map.setdefault(df.data_file_id, []).append((path, df.delete_count))

            rows: list[tuple[str, int, str | None, int | None]] = []
            for f in data_files:
                data_path = reader.resolve_data_file_path(f.path, f.path_is_relative, table_info)
                del_entries = del_map.get(f.data_file_id, [])
                if del_entries:
                    for del_path, del_count in del_entries:
                        rows.append((data_path, f.file_size_bytes, del_path, del_count))
                else:
                    rows.append((data_path, f.file_size_bytes, None, None))

        if not rows:
            return lf_schema.empty_table()

        return pa.table(
            {
                "data_file": pa.array([r[0] for r in rows], type=pa.string()),
                "data_file_size_bytes": pa.array([r[1] for r in rows], type=pa.int64()),
                "delete_file": pa.array([r[2] for r in rows], type=pa.string()),
                "delete_row_count": pa.array([r[3] for r in rows], type=pa.int64()),
            },
            schema=lf_schema,
        )

    # ------------------------------------------------------------------
    # Schema / table listing
    # ------------------------------------------------------------------

    def list_schemas(self, *, snapshot_version: int | None = None) -> pa.Table:
        """
        List all schemas in the catalog.

        Returns a table with columns:
        - ``schema_id`` (int64)
        - ``schema_name`` (string)
        """
        ls_schema = pa.schema([
            pa.field("schema_id", pa.int64()),
            pa.field("schema_name", pa.string()),
        ])

        with self._reader() as reader:
            if snapshot_version is not None:
                snap = reader.get_snapshot_at_version(snapshot_version)
            else:
                snap = reader.get_current_snapshot()
            rows = reader.get_all_schemas(snap.snapshot_id)

        if not rows:
            return ls_schema.empty_table()

        return pa.table(
            {
                "schema_id": pa.array([r[0] for r in rows], type=pa.int64()),
                "schema_name": pa.array([r[1] for r in rows], type=pa.string()),
            },
            schema=ls_schema,
        )

    def list_tables(
        self,
        *,
        schema: str = "main",
        snapshot_version: int | None = None,
    ) -> pa.Table:
        """
        List all tables in a schema.

        Returns a table with columns:
        - ``table_id`` (int64)
        - ``table_name`` (string)
        """
        lt_schema = pa.schema([
            pa.field("table_id", pa.int64()),
            pa.field("table_name", pa.string()),
        ])

        with self._reader() as reader:
            if snapshot_version is not None:
                snap = reader.get_snapshot_at_version(snapshot_version)
            else:
                snap = reader.get_current_snapshot()

            schemas = reader.get_all_schemas(snap.snapshot_id)
            schema_row = None
            for s in schemas:
                if s[1] == schema:
                    schema_row = s
                    break
            if schema_row is None:
                return lt_schema.empty_table()

            tables = reader.get_all_tables(schema_row[0], snap.snapshot_id)

        if not tables:
            return lt_schema.empty_table()

        return pa.table(
            {
                "table_id": pa.array([r[0] for r in tables], type=pa.int64()),
                "table_name": pa.array([r[1] for r in tables], type=pa.string()),
            },
            schema=lt_schema,
        )

    # ------------------------------------------------------------------
    # Catalog options
    # ------------------------------------------------------------------

    def options(self) -> pa.Table:
        """
        Get catalog key-value options from ducklake_metadata.

        Equivalent to ``ducklake_options(catalog)``.

        Returns a table with columns:
        - ``key`` (string)
        - ``value`` (string)
        - ``scope`` (string or null) — ``"schema"`` / ``"table"`` for
          scoped options, ``null`` for catalog-wide rows.
        - ``scope_id`` (int64 or null) — schema_id / table_id for the
          scoped row; ``null`` otherwise.
        """
        opt_schema = pa.schema([
            pa.field("key", pa.string()),
            pa.field("value", pa.string()),
            pa.field("scope", pa.string()),
            pa.field("scope_id", pa.int64()),
        ])

        with self._reader() as reader:
            rows = reader.get_all_metadata()

        if not rows:
            return opt_schema.empty_table()

        # Older readers may yield 2-tuples; normalise.
        norm = [r if len(r) == 4 else (r[0], r[1], None, None) for r in rows]

        return pa.table(
            {
                "key": pa.array([r[0] for r in norm], type=pa.string()),
                "value": pa.array([r[1] for r in norm], type=pa.string()),
                "scope": pa.array([r[2] for r in norm], type=pa.string()),
                "scope_id": pa.array(
                    [r[3] if r[3] is None else int(r[3]) for r in norm],
                    type=pa.int64(),
                ),
            },
            schema=opt_schema,
        )

    def settings(self) -> pa.Table:
        """
        Get catalog type and data path.

        Equivalent to ``ducklake_settings(catalog)``.

        Returns a table with columns:
        - ``catalog_type`` (string) — ``"sqlite"`` or ``"postgresql"``
        - ``data_path`` (string)
        """
        with self._reader() as reader:
            catalog_type = reader._backend.__class__.__name__
            if "sqlite" in catalog_type.lower():
                cat_type = "sqlite"
            else:
                cat_type = "postgresql"
            data_path = reader.data_path

        return pa.table(
            {
                "catalog_type": pa.array([cat_type], type=pa.string()),
                "data_path": pa.array([data_path], type=pa.string()),
            },
            schema=pa.schema([
                pa.field("catalog_type", pa.string()),
                pa.field("data_path", pa.string()),
            ]),
        )

    # ------------------------------------------------------------------
    # Tags
    # ------------------------------------------------------------------

    def table_tags(self, table: str, *, schema: str = "main") -> pa.Table:
        """
        Get tags for a table.

        Returns a table with columns:
        - ``key`` (string)
        - ``value`` (string)
        """
        tag_schema = pa.schema([
            pa.field("key", pa.string()),
            pa.field("value", pa.string()),
        ])

        with self._reader() as reader:
            snap = reader.get_current_snapshot()
            table_info = reader.get_table(table, schema, snap.snapshot_id)
            tags = reader.get_table_tags(table_info.table_id, snap.snapshot_id)

        if not tags:
            return tag_schema.empty_table()

        return pa.table(
            {
                "key": pa.array(list(tags.keys()), type=pa.string()),
                "value": pa.array(list(tags.values()), type=pa.string()),
            },
            schema=tag_schema,
        )

    def column_tags(
        self, table: str, column: str, *, schema: str = "main"
    ) -> pa.Table:
        """
        Get tags for a column.

        Returns a table with columns:
        - ``key`` (string)
        - ``value`` (string)
        """
        tag_schema = pa.schema([
            pa.field("key", pa.string()),
            pa.field("value", pa.string()),
        ])

        with self._reader() as reader:
            snap = reader.get_current_snapshot()
            table_info = reader.get_table(table, schema, snap.snapshot_id)
            columns = reader.get_columns(table_info.table_id, snap.snapshot_id)
            col_info = None
            for c in columns:
                if c.column_name == column:
                    col_info = c
                    break
            if col_info is None:
                msg = f"Column '{column}' not found in table '{schema}.{table}'"
                raise ValueError(msg)
            tags = reader.get_column_tags(
                table_info.table_id, col_info.column_id, snap.snapshot_id
            )

        if not tags:
            return tag_schema.empty_table()

        return pa.table(
            {
                "key": pa.array(list(tags.keys()), type=pa.string()),
                "value": pa.array(list(tags.values()), type=pa.string()),
            },
            schema=tag_schema,
        )

    # ------------------------------------------------------------------
    # Change data feed
    # ------------------------------------------------------------------

    def table_insertions(
        self,
        table: str,
        start_version: int,
        end_version: int,
        *,
        schema: str = "main",
    ) -> pa.Table:
        """
        Get rows inserted between two snapshots.

        Equivalent to ``ducklake_table_insertions(catalog, schema, table, start, end)``.

        Reads the Parquet data files that were added in the snapshot range
        ``(start_version, end_version]``.

        Returns a table with ``snapshot_id`` column plus all table columns.
        """
        with self._reader() as reader:
            snap = reader.get_snapshot_at_version(end_version)
            table_info = reader.get_table(table, schema, snap.snapshot_id)
            all_columns = reader.get_all_columns(table_info.table_id, snap.snapshot_id)
            column_names = [c.column_name for c in all_columns if c.parent_column is None]

            files_with_snap = reader.get_data_files_in_range_with_snapshot(
                table_info.table_id, start_version, end_version
            )

            if not files_with_snap:
                return self._build_cdf_schema(
                    all_columns, [("snapshot_id", pa.int64())]
                ).empty_table()

            frames: list[pa.Table] = []
            for file_info, begin_snap in files_with_snap:
                path = reader.resolve_data_file_path(
                    file_info.path, file_info.path_is_relative, table_info
                )
                df = storage.read_parquet(path)
                # Only keep top-level columns that exist in the current schema
                available = [c for c in column_names if c in df.column_names]
                df = df.select(available)
                df = df.append_column(
                    "snapshot_id",
                    pa.array([begin_snap] * df.num_rows, type=pa.int64()),
                )
                frames.append(df)

        result = pa.concat_tables(frames, promote_options="permissive")
        # Reorder: snapshot_id first
        cols = ["snapshot_id"] + [c for c in result.column_names if c != "snapshot_id"]
        return result.select(cols)

    def table_deletions(
        self,
        table: str,
        start_version: int,
        end_version: int,
        *,
        schema: str = "main",
    ) -> pa.Table:
        """
        Get rows deleted between two snapshots.

        Equivalent to ``ducklake_table_deletions(catalog, schema, table, start, end)``.

        Reads delete files added in the snapshot range ``(start_version, end_version]``,
        then reads the corresponding rows from their associated data files.

        Returns a table with ``snapshot_id`` column plus all table columns.
        """
        with self._reader() as reader:
            snap = reader.get_snapshot_at_version(end_version)
            table_info = reader.get_table(table, schema, snap.snapshot_id)
            all_columns = reader.get_all_columns(table_info.table_id, snap.snapshot_id)
            column_names = [c.column_name for c in all_columns if c.parent_column is None]
            empty_schema = self._build_cdf_schema(
                all_columns, [("snapshot_id", pa.int64())]
            )

            del_files = reader.get_delete_files_in_range(
                table_info.table_id, start_version, end_version
            )

            if not del_files:
                return empty_schema.empty_table()

            frames: list[pa.Table] = []
            for del_info, begin_snap in del_files:
                # Read the delete file (position-delete format: file_path, pos columns)
                del_path = reader.resolve_data_file_path(
                    del_info.path, del_info.path_is_relative, table_info
                )
                del_df = storage.read_parquet(del_path)

                # Get the corresponding data file
                data_file = reader.get_data_file_by_id(del_info.data_file_id)
                if data_file is None:
                    continue

                data_path = reader.resolve_data_file_path(
                    data_file.path, data_file.path_is_relative, table_info
                )
                data_df = storage.read_parquet(data_path)

                # Extract deleted row positions
                # DuckLake delete files use Iceberg position-delete format
                # with row_id_start offset and "pos" column
                if "pos" in del_df.column_names:
                    positions = del_df.column("pos").to_pylist()
                    row_id_start = data_file.row_id_start
                    local_positions = [p - row_id_start for p in positions]
                    # Filter to valid positions
                    local_positions = [p for p in local_positions if 0 <= p < data_df.num_rows]
                    if local_positions:
                        deleted_rows = data_df.take(pa.array(local_positions))
                        available = [c for c in column_names if c in deleted_rows.column_names]
                        deleted_rows = deleted_rows.select(available)
                        deleted_rows = deleted_rows.append_column(
                            "snapshot_id",
                            pa.array([begin_snap] * deleted_rows.num_rows, type=pa.int64()),
                        )
                        frames.append(deleted_rows)

        if not frames:
            return empty_schema.empty_table()

        result = pa.concat_tables(frames, promote_options="permissive")
        cols = ["snapshot_id"] + [c for c in result.column_names if c != "snapshot_id"]
        return result.select(cols)

    def table_changes(
        self,
        table: str,
        start_version: int,
        end_version: int,
        *,
        schema: str = "main",
    ) -> pa.Table:
        """
        Get combined change data feed between two snapshots.

        Equivalent to ``ducklake_table_changes(catalog, schema, table, start, end)``.

        Returns a table with columns:
        - ``snapshot_id`` (int64)
        - ``change_type`` (string) — one of ``'insert'``, ``'delete'``,
          ``'update_preimage'``, ``'update_postimage'``
        - All table columns

        Updates are detected when both an insertion and a deletion occur in the
        same snapshot. Deletions in that snapshot become ``update_preimage`` and
        insertions become ``update_postimage``.
        """
        # Use explicit base-class calls to ensure we get pa.Table even
        # when called from a subclass that overrides these methods.
        insertions = DuckLakeCatalog.table_insertions(self, table, start_version, end_version, schema=schema)
        deletions = DuckLakeCatalog.table_deletions(self, table, start_version, end_version, schema=schema)

        has_ins = insertions.num_rows > 0
        has_del = deletions.num_rows > 0

        if not has_ins and not has_del:
            # Derive schema from the (empty) insertions/deletions frames
            source = insertions if len(insertions.column_names) >= len(deletions.column_names) else deletions
            fields = [pa.field("snapshot_id", pa.int64()), pa.field("change_type", pa.string())]
            for i in range(len(source.schema)):
                f = source.schema.field(i)
                if f.name != "snapshot_id":
                    fields.append(f)
            return pa.schema(fields).empty_table()

        # Find snapshots that have both insertions and deletions (potential updates)
        ins_snapshots = set(insertions.column("snapshot_id").to_pylist()) if has_ins else set()
        del_snapshots = set(deletions.column("snapshot_id").to_pylist()) if has_del else set()
        update_snapshots = ins_snapshots & del_snapshots

        frames: list[pa.Table] = []

        if has_ins:
            if update_snapshots:
                update_set = pa.array(list(update_snapshots), type=pa.int64())
                mask = pc.is_in(insertions.column("snapshot_id"), value_set=update_set)
                inv_mask = pc.invert(mask)

                pure_ins = insertions.filter(inv_mask)
                update_post = insertions.filter(mask)

                if pure_ins.num_rows > 0:
                    pure_ins = pure_ins.append_column(
                        "change_type",
                        pa.array(["insert"] * pure_ins.num_rows, type=pa.string()),
                    )
                    frames.append(pure_ins)
                if update_post.num_rows > 0:
                    update_post = update_post.append_column(
                        "change_type",
                        pa.array(["update_postimage"] * update_post.num_rows, type=pa.string()),
                    )
                    frames.append(update_post)
            else:
                ins_with_type = insertions.append_column(
                    "change_type",
                    pa.array(["insert"] * insertions.num_rows, type=pa.string()),
                )
                frames.append(ins_with_type)

        if has_del:
            if update_snapshots:
                update_set = pa.array(list(update_snapshots), type=pa.int64())
                mask = pc.is_in(deletions.column("snapshot_id"), value_set=update_set)
                inv_mask = pc.invert(mask)

                pure_del = deletions.filter(inv_mask)
                update_pre = deletions.filter(mask)

                if pure_del.num_rows > 0:
                    pure_del = pure_del.append_column(
                        "change_type",
                        pa.array(["delete"] * pure_del.num_rows, type=pa.string()),
                    )
                    frames.append(pure_del)
                if update_pre.num_rows > 0:
                    update_pre = update_pre.append_column(
                        "change_type",
                        pa.array(["update_preimage"] * update_pre.num_rows, type=pa.string()),
                    )
                    frames.append(update_pre)
            else:
                del_with_type = deletions.append_column(
                    "change_type",
                    pa.array(["delete"] * deletions.num_rows, type=pa.string()),
                )
                frames.append(del_with_type)

        if not frames:
            fields = [pa.field("snapshot_id", pa.int64()), pa.field("change_type", pa.string())]
            return pa.schema(fields).empty_table()

        result = pa.concat_tables(frames, promote_options="permissive")
        # Reorder: snapshot_id, change_type, then rest
        other_cols = [c for c in result.column_names if c not in ("snapshot_id", "change_type")]
        return result.select(["snapshot_id", "change_type"] + other_cols)

    # ------------------------------------------------------------------
    # Macros
    # ------------------------------------------------------------------

    def list_macros(
        self,
        *,
        schema: str = "main",
        snapshot_version: int | None = None,
    ) -> pa.Table:
        """
        List all macros in a schema.

        Returns a table with columns:
        - ``macro_id`` (int64)
        - ``macro_name`` (string)
        - ``macro_type`` (string) — ``"scalar"`` or ``"table"``
        """
        lm_schema = pa.schema([
            pa.field("macro_id", pa.int64()),
            pa.field("macro_name", pa.string()),
            pa.field("macro_type", pa.string()),
        ])

        with self._reader() as reader:
            if snapshot_version is not None:
                snap = reader.get_snapshot_at_version(snapshot_version)
            else:
                snap = reader.get_current_snapshot()

            schemas = reader.get_all_schemas(snap.snapshot_id)
            schema_row = None
            for s in schemas:
                if s[1] == schema:
                    schema_row = s
                    break
            if schema_row is None:
                return lm_schema.empty_table()

            macros = reader.get_macros(schema_row[0], snap.snapshot_id)
            if not macros:
                return lm_schema.empty_table()

            # Determine macro type from the first implementation
            macro_ids = []
            macro_names = []
            macro_types = []
            for m in macros:
                impls = reader.get_macro_implementations(m.macro_id)
                macro_type = impls[0].macro_type if impls else "scalar"
                macro_ids.append(m.macro_id)
                macro_names.append(m.macro_name)
                macro_types.append(macro_type)

        return pa.table(
            {
                "macro_id": pa.array(macro_ids, type=pa.int64()),
                "macro_name": pa.array(macro_names, type=pa.string()),
                "macro_type": pa.array(macro_types, type=pa.string()),
            },
            schema=lm_schema,
        )

    def get_macro(
        self,
        name: str,
        *,
        schema: str = "main",
        dialect: str | None = None,
        snapshot_version: int | None = None,
    ) -> pa.Table:
        """
        Get macro definition(s) by name.

        Parameters
        ----------
        name
            Macro name.
        schema
            Schema name (default: "main").
        dialect
            If provided, return only the implementation for this dialect
            (e.g., ``"duckdb"``). If None, return all implementations.
        snapshot_version
            Snapshot version to query. If None, use the latest snapshot.

        Returns a table with columns:
        - ``macro_name`` (string)
        - ``macro_type`` (string) — ``"scalar"`` or ``"table"``
        - ``dialect`` (string)
        - ``sql`` (string) — the SQL definition
        - ``parameters`` (string) — comma-separated parameter list
          (e.g., ``"x INTEGER, y VARCHAR DEFAULT 'hello'"``).
        """
        gm_schema = pa.schema([
            pa.field("macro_name", pa.string()),
            pa.field("macro_type", pa.string()),
            pa.field("dialect", pa.string()),
            pa.field("sql", pa.string()),
            pa.field("parameters", pa.string()),
        ])

        with self._reader() as reader:
            if snapshot_version is not None:
                snap = reader.get_snapshot_at_version(snapshot_version)
            else:
                snap = reader.get_current_snapshot()

            schemas = reader.get_all_schemas(snap.snapshot_id)
            schema_row = None
            for s in schemas:
                if s[1] == schema:
                    schema_row = s
                    break
            if schema_row is None:
                msg = f"Schema '{schema}' not found"
                raise ValueError(msg)

            macros = reader.get_macros(schema_row[0], snap.snapshot_id)
            macro = None
            for m in macros:
                if m.macro_name.lower() == name.lower():
                    macro = m
                    break
            if macro is None:
                msg = f"Macro '{name}' not found in schema '{schema}'"
                raise ValueError(msg)

            impls = reader.get_macro_implementations(macro.macro_id)
            if dialect is not None:
                impls = [i for i in impls if i.dialect.lower() == dialect.lower()]

            names = []
            types = []
            dialects = []
            sqls = []
            params_strs = []

            for impl in impls:
                params = reader.get_macro_parameters(macro.macro_id, impl.impl_id)
                param_parts = []
                for p in params:
                    part = f"{p.parameter_name} {p.parameter_type}"
                    if p.default_value is not None and p.default_value != "":
                        part += f" DEFAULT {p.default_value}"
                    param_parts.append(part)
                params_str = ", ".join(param_parts)

                names.append(macro.macro_name)
                types.append(impl.macro_type)
                dialects.append(impl.dialect)
                sqls.append(impl.sql)
                params_strs.append(params_str)

        if not names:
            return gm_schema.empty_table()

        return pa.table(
            {
                "macro_name": pa.array(names, type=pa.string()),
                "macro_type": pa.array(types, type=pa.string()),
                "dialect": pa.array(dialects, type=pa.string()),
                "sql": pa.array(sqls, type=pa.string()),
                "parameters": pa.array(params_strs, type=pa.string()),
            },
            schema=gm_schema,
        )

    # ------------------------------------------------------------------
    # Sort keys
    # ------------------------------------------------------------------

    def sort_keys(
        self,
        table: str,
        *,
        schema: str = "main",
        snapshot_version: int | None = None,
    ) -> pa.Table:
        """Return the active sort keys for *table*.

        Returns a table with columns:
        - ``sort_key_index`` (int64) — position in the sort order
        - ``expression`` (string) — column name
        - ``sort_direction`` (string) — ``ASC`` or ``DESC``
        - ``null_order`` (string) — ``NULLS_FIRST`` or ``NULLS_LAST``

        Returns an empty table if no sort keys are set.
        """
        sk_schema = pa.schema([
            pa.field("sort_key_index", pa.int64()),
            pa.field("expression", pa.string()),
            pa.field("sort_direction", pa.string()),
            pa.field("null_order", pa.string()),
        ])

        with self._reader() as reader:
            if snapshot_version is not None:
                snap = reader.get_snapshot_at_version(snapshot_version)
            else:
                snap = reader.get_current_snapshot()
            table_info = reader.get_table(table, schema, snap.snapshot_id)
            con = reader._connect()
            try:
                row = con.execute(
                    reader._sql(
                        "SELECT sort_id FROM ducklake_sort_info "
                        "WHERE table_id = ? AND begin_snapshot <= ? "
                        "AND (end_snapshot IS NULL OR end_snapshot > ?)"
                    ),
                    [table_info.table_id, snap.snapshot_id, snap.snapshot_id],
                ).fetchone()
            except Exception as exc:
                if reader._backend.is_table_not_found(exc):
                    return sk_schema.empty_table()
                raise
            if row is None:
                return sk_schema.empty_table()
            sort_id = row[0]
            rows = con.execute(
                reader._sql(
                    "SELECT sort_key_index, expression, sort_direction, "
                    "null_order FROM ducklake_sort_expression "
                    "WHERE sort_id = ? AND table_id = ? "
                    "ORDER BY sort_key_index"
                ),
                [sort_id, table_info.table_id],
            ).fetchall()

        if not rows:
            return sk_schema.empty_table()

        return pa.table(
            {
                "sort_key_index": pa.array(
                    [r[0] for r in rows], type=pa.int64(),
                ),
                "expression": pa.array(
                    [r[1] for r in rows], type=pa.string(),
                ),
                "sort_direction": pa.array(
                    [(r[2] or "ASC").upper() for r in rows],
                    type=pa.string(),
                ),
                "null_order": pa.array(
                    [(r[3] or "NULLS_LAST").upper() for r in rows],
                    type=pa.string(),
                ),
            },
            schema=sk_schema,
        )
