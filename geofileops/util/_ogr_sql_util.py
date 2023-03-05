# -*- coding: utf-8 -*-
"""
Module containing utilities regarding the formatting of sql statements meant for use
with ogr.
"""

from typing import Iterable, List, Optional


class ColumnFormatter:
    """
    Format strings with the columns for use in sql statements.

    There are some specific hacks that are related to specific behaviours of ogr, mainly
    regarding the handling of the special "fid" column.
    """

    _aliases_cache = None

    def __init__(
        self,
        columns_asked: Optional[List[str]],
        columns_in_layer: Iterable[str],
        ogr_and_fid_no_column: bool,
        table_alias: Optional[str] = None,
        columnname_prefix: str = "",
    ):
        # First prepare the actual column list to use
        if columns_asked is not None:
            # Add special column "fid" to available columns so it can be specified
            columns_in_layer = list(columns_in_layer) + ["fid"]
            # Case-insensitive check if input1_columns contains columns not in layer...
            columns_in_layer_upper = {
                column.upper(): column for column in columns_in_layer
            }
            missing_columns = [
                col
                for col in columns_asked
                if (col.upper() not in columns_in_layer_upper)
            ]
            if len(missing_columns) > 0:
                raise ValueError(
                    "columns_asked contains columns not in columns_in_layer: "
                    f"{missing_columns}. Available: {columns_in_layer}"
                )

            # Create column list to keep in the casing of the original columns
            columns = [columns_in_layer_upper[col.upper()] for col in columns_asked]
        else:
            columns = list(columns_in_layer)

        self._columns = columns
        self._columns_in_layer = columns_in_layer
        self._columnname_prefix = columnname_prefix
        self._table_alias = table_alias
        self.ogr_and_fid_no_column = ogr_and_fid_no_column
        self._table_prefix = ""
        if table_alias is not None and table_alias != "":
            self._table_prefix = f"{table_alias}."

    def _columns_prefixed(self) -> List[str]:
        columns_prefixed = [
            f'{self._table_prefix}"{column}"' for column in self._columns
        ]
        columns_prefixed = self._fix_fid_columns(columns_prefixed)
        return columns_prefixed

    def _fix_fid_columns(self, columns: List[str]) -> List[str]:
        """
        The "fid" column needs some special treatment:
            - If ogr is used on a file format that doesn't actually save the fid
              (eg. shapefile), "rowid" should be used (because sql_dialect = SQLITE).
            - If ogr is not used or it is used with a file format that saves fid as a
              column (eg. in GPKG), ogr will treat the column as "fid" (with the alias
              specified) anyway.
              To stop ogr from doing this, the fid column needs to be CASTed so ogr
              doesn't recognize it anymore.
              Remark: the ogr2ogr -unsetFID switch didn't help.
        """
        columns = list(columns)
        fid_column_indexes = [
            idx for idx, col in enumerate(self._columns) if col.upper() == "FID"
        ]
        if self.ogr_and_fid_no_column:
            for fid_column_index in fid_column_indexes:
                columns[fid_column_index] = "rowid"
        else:
            for fid_column_index in fid_column_indexes:
                columns[fid_column_index] = f"CAST({columns[fid_column_index]} AS INT)"

        return columns

    def _aliases(self) -> List[str]:
        if self._aliases_cache is not None:
            return self._aliases_cache

        aliases = [f"{self._columnname_prefix}{column}" for column in self._columns]

        # If no prefix, create a unique alias for fid column(s)
        if self._columnname_prefix == "":
            for alias_idx, alias in enumerate(aliases):
                if alias.lower() == "fid":
                    # If alias "fid", change it + make sure the alias isn't in use yet
                    for idx in range(1, 100):
                        alias_with_id = f"{alias}_{idx}"
                        if alias_with_id not in aliases:
                            aliases[alias_idx] = alias_with_id
                            break

        self._aliases_cache = aliases
        return self._aliases_cache

    def quoted(self) -> str:
        if len(self._columns) == 0:
            return ""

        columns_quoted = [f'"{column}"' for column in self._columns]
        columns_quoted = self._fix_fid_columns(columns_quoted)
        return f",{', '.join(columns_quoted)}"

    def prefixed(self) -> str:
        if len(self._columns) == 0:
            return ""
        return f",{', '.join(self._columns_prefixed())}"

    def prefixed_aliased(self):
        if len(self._columns) == 0:
            return ""

        columns_prefixed_aliased = [
            f'{column_prefixed} "{column_alias}"'
            for column_prefixed, column_alias in zip(
                self._columns_prefixed(), self._aliases()
            )
        ]
        return f",{', '.join(columns_prefixed_aliased)}"

    def null_aliased(self):
        if len(self._columns) == 0:
            return ""

        columns_null_aliased = [f'NULL "{alias}"' for alias in self._aliases()]
        return f",{', '.join(columns_null_aliased)}"

    def from_subselect(self, subselect_alias: str = "sub"):
        if len(self._columns) == 0:
            return ""

        prefix = "" if subselect_alias == "" else f"{subselect_alias}."
        columns_from_subselect = [f'{prefix}"{alias}"' for alias in self._aliases()]
        return f",{', '.join(columns_from_subselect)}"
