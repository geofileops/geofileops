# -*- coding: utf-8 -*-
"""
Tests for functionalities in sql_util.
"""

from pathlib import Path
import sys
from typing import Iterable, List, Optional

import pytest

# Add path so the local geofileops packages are found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geofileops.util import _ogr_sql_util


@pytest.mark.parametrize(
    "columns_specified, columns_available, table_alias, columnname_prefix, "
    "ogr_and_fid_no_column, "
    "exp_quoted, exp_prefixed, exp_prefixed_aliased, "
    "exp_null_aliased, exp_from_subselect",
    [
        ([], ["test1"], "", "", False, "", "", "", "", ""),
        (
            ["test1", "test2"],
            ["test1", "test3", "test2"],
            "",
            "",
            False,
            ',"test1", "test2"',
            ',"test1", "test2"',
            ',"test1" "test1", "test2" "test2"',
            ',NULL "test1", NULL "test2"',
            ',sub."test1", sub."test2"',
        ),
        (
            ["fid", "test1", "test2"],
            ["test2", "test3", "test1"],
            "",
            "",
            False,
            ',CAST("fid" AS INT), "test1", "test2"',
            ',CAST("fid" AS INT), "test1", "test2"',
            ',CAST("fid" AS INT) "fid_1", "test1" "test1", "test2" "test2"',
            ',NULL "fid_1", NULL "test1", NULL "test2"',
            ',sub."fid_1", sub."test1", sub."test2"',
        ),
        (
            ["fid", "test1"],
            ["test2", "test3", "test1"],
            "table_alias1",
            "",
            False,
            ',CAST("fid" AS INT), "test1"',
            ',CAST(table_alias1."fid" AS INT), table_alias1."test1"',
            ',CAST(table_alias1."fid" AS INT) "fid_1", table_alias1."test1" "test1"',
            ',NULL "fid_1", NULL "test1"',
            ',sub."fid_1", sub."test1"',
        ),
        (
            ["fid", "test1"],
            ["test2", "test3", "test1"],
            "table_alias1",
            "l1_",
            True,
            ',rowid, "test1"',
            ',rowid, table_alias1."test1"',
            ',rowid "l1_fid", table_alias1."test1" "l1_test1"',
            ',NULL "l1_fid", NULL "l1_test1"',
            ',sub."l1_fid", sub."l1_test1"',
        ),
    ],
)
def test_ColumnFormatter(
    columns_specified: Optional[List[str]],
    columns_available: Iterable[str],
    table_alias: str,
    columnname_prefix: str,
    ogr_and_fid_no_column: bool,
    exp_quoted: str,
    exp_prefixed: str,
    exp_prefixed_aliased: str,
    exp_null_aliased: str,
    exp_from_subselect: str,
):
    column_frmt = _ogr_sql_util.ColumnFormatter(
        columns_asked=columns_specified,
        columns_in_layer=columns_available,
        table_alias=table_alias,
        columnname_prefix=columnname_prefix,
        ogr_and_fid_no_column=ogr_and_fid_no_column,
    )
    assert column_frmt.quoted() == exp_quoted
    assert column_frmt.prefixed() == exp_prefixed
    assert column_frmt.prefixed_aliased() == exp_prefixed_aliased
    assert column_frmt.null_aliased() == exp_null_aliased
    assert column_frmt.from_subselect() == exp_from_subselect


def test_ColumnFormatter_invalidcolumn():
    with pytest.raises(ValueError, match="columns_asked contains columns not in"):
        _ogr_sql_util.ColumnFormatter(
            columns_asked=["test"],
            columns_in_layer=["test1"],
            ogr_and_fid_no_column=True,
        )
