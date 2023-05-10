# -*- coding: utf-8 -*-
"""
Tests for operations that are executed using a sql statement on one layer.
"""

import math

import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, Polygon

import geofileops as gfo
from geofileops import GeometryType
from geofileops.util import _geoops_sql
from tests import test_helper
from tests.test_helper import DEFAULT_EPSGS, DEFAULT_SUFFIXES
from tests.test_helper import assert_geodataframe_equal


def test_delete_duplicate_geometries(tmp_path):
    # Prepare test data
    test_gdf = gpd.GeoDataFrame(
        geometry=[  # type: ignore
            test_helper.TestData.polygon_with_island,
            test_helper.TestData.polygon_with_island,
            test_helper.TestData.polygon_no_islands,
            test_helper.TestData.polygon_no_islands,
            test_helper.TestData.polygon_with_island2,
        ],
        crs=test_helper.TestData.crs_epsg,  # type: ignore
    )
    suffix = ".gpkg"
    input_path = tmp_path / f"input_test_data{suffix}"
    gfo.to_file(test_gdf, input_path)
    input_info = gfo.get_layerinfo(input_path)

    # Run test
    output_path = tmp_path / f"{input_path.stem}-output{suffix}"
    print(f"Run test for suffix {suffix}")
    # delete_duplicate_geometries isn't multiprocess, so no batchsize needed
    gfo.delete_duplicate_geometries(input_path=input_path, output_path=output_path)

    # Check result, 2 duplicates should be removed
    result_info = gfo.get_layerinfo(output_path)
    assert result_info.featurecount == input_info.featurecount - 2


def test_dissolve_singlethread_output_exists(tmp_path):
    # Prepare test data
    input_path = test_helper.get_testfile("polygon-parcel")
    output_path = tmp_path / f"{input_path.stem}-output{input_path.suffix}"
    output_path.touch()

    # Run test without force
    _geoops_sql.dissolve_singlethread(
        input_path=input_path,
        output_path=output_path,
    )
    assert output_path.exists()
    assert output_path.stat().st_size == 0

    # Run test with force
    _geoops_sql.dissolve_singlethread(
        input_path=input_path,
        output_path=output_path,
        force=True,
    )
    assert output_path.exists()
    assert output_path.stat().st_size != 0


@pytest.mark.parametrize("suffix", DEFAULT_SUFFIXES)
@pytest.mark.parametrize("epsg", DEFAULT_EPSGS)
def test_isvalid(tmp_path, suffix, epsg):
    # Prepare test data
    input_path = test_helper.get_testfile(
        "polygon-invalid", dst_dir=tmp_path, suffix=suffix, epsg=epsg
    )

    # Now run test
    output_path = tmp_path / f"{input_path.stem}-output{suffix}"
    input_layerinfo = gfo.get_layerinfo(input_path)
    batchsize = math.ceil(input_layerinfo.featurecount / 2)
    gfo.isvalid(input_path=input_path, output_path=output_path, batchsize=batchsize)

    # Now check if the tmp file is correctly created
    assert output_path.exists() is True
    result_layerinfo = gfo.get_layerinfo(output_path)
    assert input_layerinfo.featurecount == result_layerinfo.featurecount
    assert len(input_layerinfo.columns) == len(result_layerinfo.columns) - 2

    output_gdf = gfo.read_file(output_path)
    assert output_gdf["geometry"][0] is not None
    assert output_gdf["isvalid"][0] == 0

    # Do operation, without specifying output path
    gfo.isvalid(
        input_path=input_path, batchsize=batchsize, validate_attribute_data=True
    )

    # Now check if the tmp file is correctly created
    output_auto_path = (
        output_path.parent / f"{input_path.stem}_isvalid{output_path.suffix}"
    )
    assert output_auto_path.exists()
    result_auto_layerinfo = gfo.get_layerinfo(output_auto_path)
    assert input_layerinfo.featurecount == result_auto_layerinfo.featurecount
    assert len(input_layerinfo.columns) == len(result_auto_layerinfo.columns) - 2

    output_auto_gdf = gfo.read_file(output_auto_path)
    assert output_auto_gdf["geometry"][0] is not None
    assert output_auto_gdf["isvalid"][0] == 0


@pytest.mark.parametrize("suffix", DEFAULT_SUFFIXES)
@pytest.mark.parametrize("input_empty", [True, False])
def test_makevalid(tmp_path, suffix, input_empty):
    # Prepare test data
    input_path = test_helper.get_testfile(
        "polygon-invalid", suffix=suffix, empty=input_empty
    )

    # If the input file is not empty, it should have invalid geoms
    if not input_empty:
        input_isvalid_path = tmp_path / f"{input_path.stem}_is-valid{suffix}"
        isvalid = gfo.isvalid(input_path=input_path, output_path=input_isvalid_path)
        assert isvalid is False, "Input file should contain invalid features"

    # Make sure the input file is not valid
    if not input_empty:
        output_isvalid_path = (
            tmp_path / f"{input_path.stem}_is-valid{input_path.suffix}"
        )
        isvalid = gfo.isvalid(input_path=input_path, output_path=output_isvalid_path)
        assert isvalid is False, "Input file should contain invalid features"

    # Do operation
    output_path = tmp_path / f"{input_path.stem}-output{suffix}"
    gfo.makevalid(
        input_path=input_path,
        output_path=output_path,
        nb_parallel=2,
        force_output_geometrytype=gfo.GeometryType.MULTIPOLYGON,
        validate_attribute_data=True,
    )

    # Now check if the output file is correctly created
    assert output_path.exists()
    layerinfo_orig = gfo.get_layerinfo(input_path)
    layerinfo_output = gfo.get_layerinfo(output_path)
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)
    assert layerinfo_output.geometrytype in [
        GeometryType.POLYGON,
        GeometryType.MULTIPOLYGON,
    ]

    if not input_empty:
        assert layerinfo_orig.featurecount == layerinfo_output.featurecount

    # Check if the result file is valid
    output_new_isvalid_path = (
        tmp_path / f"{output_path.stem}_new_is-valid{output_path.suffix}"
    )
    isvalid = gfo.isvalid(input_path=output_path, output_path=output_new_isvalid_path)
    assert isvalid is True, "Output file shouldn't contain invalid features"

    # Run makevalid with existing output file and force=False (=default)
    gfo.makevalid(input_path=input_path, output_path=output_path)


@pytest.mark.parametrize(
    "descr, geometry, expected_geometry",
    [
        ("sliver", Polygon([(0, 0), (10, 0), (10, 0.5), (0, 0)]), Polygon()),
        (
            "poly + sliver",
            MultiPolygon(
                [
                    Polygon([(0, 5), (5, 5), (5, 10), (0, 10), (0, 5)]),
                    Polygon([(0, 0), (10, 0), (10, 0.5), (0, 0)]),
                ]
            ),
            Polygon([(0, 5), (5, 5), (5, 10), (0, 10), (0, 5)]),
        ),
    ],
)
def test_makevalid_gridsize(tmp_path, descr: str, geometry, expected_geometry):
    # Prepare test data
    # -----------------
    input_gdf = gpd.GeoDataFrame(
        {"descr": [descr]}, geometry=[geometry], crs=31370
    )  # type: ignore
    input_path = tmp_path / "test.gpkg"
    gfo.to_file(input_gdf, input_path)
    gridsize = 1

    # Now we are ready to test
    # ------------------------
    result_path = tmp_path / "test_makevalid.gpkg"
    gfo.makevalid(
        input_path=input_path,
        output_path=result_path,
        gridsize=gridsize,
        force=True,
    )
    result_gdf = gfo.read_file(result_path)

    # Compare with expected result
    expected_gdf = gpd.GeoDataFrame(
        {"descr": [descr]}, geometry=[expected_geometry], crs=31370
    )  # type: ignore
    expected_gdf = expected_gdf[~expected_gdf.geometry.is_empty]
    if len(expected_gdf) == 0:
        assert len(result_gdf) == 0
    else:
        assert_geodataframe_equal(result_gdf, expected_gdf)


def test_makevalid_invalidparams():
    expected_error = (
        "the precision parameter is deprecated and cannot be combined with gridsize"
    )
    with pytest.raises(ValueError, match=expected_error):
        gfo.makevalid(
            input_path="abc",
            output_path="def",
            gridsize=1,
            precision=1,
        )


@pytest.mark.parametrize("input_suffix", DEFAULT_SUFFIXES)
@pytest.mark.parametrize("output_suffix", DEFAULT_SUFFIXES)
def test_select(tmp_path, input_suffix, output_suffix):
    # Prepare test data
    input_path = test_helper.get_testfile("polygon-parcel", suffix=input_suffix)

    # Now run test
    output_path = (
        tmp_path
        / f"{input_path.stem}-{input_suffix.replace('.', '')}-output{output_suffix}"
    )
    layerinfo_input = gfo.get_layerinfo(input_path)
    sql_stmt = 'SELECT {geometrycolumn}, oidn, uidn FROM "{input_layer}"'
    gfo.select(input_path=input_path, output_path=output_path, sql_stmt=sql_stmt)

    # Now check if the tmp file is correctly created
    layerinfo_output = gfo.get_layerinfo(output_path)
    assert layerinfo_input.featurecount == layerinfo_output.featurecount
    assert "OIDN" in layerinfo_output.columns
    assert "UIDN" in layerinfo_output.columns
    assert len(layerinfo_output.columns) == 2
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON

    # Now check the contents of the result file
    output_gdf = gfo.read_file(output_path)
    assert output_gdf["geometry"][0] is not None


@pytest.mark.parametrize("suffix", DEFAULT_SUFFIXES)
def test_select_column_casing(tmp_path, suffix):
    # Prepare test data
    input_path = test_helper.get_testfile("polygon-parcel", tmp_path, suffix)

    # Check if columns parameter works (case insensitive)
    output_path = tmp_path / f"{input_path.stem}-output{suffix}"
    columns = ["OIDN", "uidn", "HFDTLT", "lblhfdtlt", "GEWASGROEP", "lengte", "OPPERVL"]
    layerinfo_input = gfo.get_layerinfo(input_path)
    sql_stmt = '''SELECT {geometrycolumn}
                        {columns_to_select_str}
                    FROM "{input_layer}"'''
    gfo.select(
        input_path=input_path,
        output_path=output_path,
        columns=columns,
        sql_stmt=sql_stmt,
    )

    # Now check if the tmp file is correctly created
    layerinfo_select = gfo.get_layerinfo(output_path)
    assert layerinfo_input.featurecount == layerinfo_select.featurecount
    assert "OIDN" in layerinfo_select.columns
    assert "uidn" in layerinfo_select.columns
    assert len(layerinfo_select.columns) == len(columns)

    output_gdf = gfo.read_file(output_path)
    assert output_gdf["geometry"][0] is not None


@pytest.mark.parametrize("input_suffix", DEFAULT_SUFFIXES)
@pytest.mark.parametrize("output_suffix", DEFAULT_SUFFIXES)
def test_select_emptyinput(tmp_path, input_suffix, output_suffix):
    # Prepare test data
    input_path = test_helper.get_testfile(
        "polygon-parcel", suffix=input_suffix, dst_dir=tmp_path, empty=True
    )
    input_layerinfo = gfo.get_layerinfo(input_path)
    assert input_layerinfo.featurecount == 0

    # Test with simple select
    # -----------------------
    output_stem = f"{input_path.stem}-{input_suffix.replace('.', '')}-simple"
    output_path = tmp_path / f"{output_stem}{output_suffix}"
    sql_stmt = 'SELECT {geometrycolumn}, oidn, uidn FROM "{input_layer}"'

    gfo.select(input_path=input_path, output_path=output_path, sql_stmt=sql_stmt)

    # Now check if the tmp file is correctly created
    layerinfo_output = gfo.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 0
    assert "OIDN" in layerinfo_output.columns
    assert "UIDN" in layerinfo_output.columns
    assert len(layerinfo_output.columns) == 2
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON


@pytest.mark.parametrize(
    "input_suffix, output_suffix",
    [
        (".gpkg", ".gpkg"),
        # (".gpkg", ".shp"),
        (".shp", ".gpkg"),
        (".shp", ".shp"),
    ],
)
def test_select_emptyinput_operation(tmp_path, input_suffix, output_suffix):
    """
    A select with a geometry operation (eg. buffer,...) on an empty input file should
    result in an empty output file.
    """
    # Prepare test data
    input_path = test_helper.get_testfile(
        "polygon-parcel", suffix=input_suffix, dst_dir=tmp_path, empty=True
    )

    input_layerinfo = gfo.get_layerinfo(input_path)
    assert input_layerinfo.featurecount == 0

    # Test with complex select: with geometry operation
    # -------------------------------------------------
    output_stem = f"{input_path.stem}-{input_suffix.replace('.', '')}-complex"
    output_path = tmp_path / f"{output_stem}{output_suffix}"
    sql_stmt = """
        SELECT st_buffer({geometrycolumn}, 1, 5) as geom, oidn, uidn
          FROM "{input_layer}"
    """
    gfo.select(input_path=input_path, output_path=output_path, sql_stmt=sql_stmt)

    assert output_path.exists()
    output_layerinfo = gfo.get_layerinfo(output_path)
    assert output_layerinfo.featurecount == 0


@pytest.mark.parametrize("input_suffix", DEFAULT_SUFFIXES)
@pytest.mark.parametrize("output_suffix", DEFAULT_SUFFIXES)
def test_select_emptyresult(tmp_path, input_suffix, output_suffix):
    # Prepare test data
    input_path = test_helper.get_testfile("polygon-parcel", suffix=input_suffix)

    # Now run test
    output_stem = f"{input_path.stem}-{input_suffix.replace('.', '')}-output"
    output_path = tmp_path / f"{output_stem}{output_suffix}"
    sql_stmt = 'SELECT {geometrycolumn}, oidn, uidn FROM "{input_layer}" WHERE 1=0'

    gfo.select(input_path=input_path, output_path=output_path, sql_stmt=sql_stmt)

    # Now check if the tmp file is correctly created
    layerinfo_output = gfo.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 0
    assert "OIDN" in layerinfo_output.columns
    assert "UIDN" in layerinfo_output.columns
    assert len(layerinfo_output.columns) == 2
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON


@pytest.mark.parametrize("suffix", DEFAULT_SUFFIXES)
def test_select_invalid_sql(tmp_path, suffix):
    # Prepare test data
    input_path = test_helper.get_testfile("polygon-parcel", suffix=suffix)

    # Now run test
    output_path = tmp_path / f"{input_path.stem}-output{suffix}"
    sql_stmt = 'SELECT {geometrycolumn}, not_existing_column FROM "{input_layer}"'

    with pytest.raises(Exception, match="Error <"):
        gfo.select(input_path=input_path, output_path=output_path, sql_stmt=sql_stmt)


def test_select_output_exists(tmp_path):
    # Prepare test data
    input_path = test_helper.get_testfile("polygon-parcel")
    output_path = tmp_path / f"{input_path.stem}-output{input_path.suffix}"
    output_path.touch()
    sql_stmt = 'SELECT {geometrycolumn}, oidn, uidn FROM "{input_layer}"'

    # Now run test
    gfo.select(input_path=input_path, output_path=output_path, sql_stmt=sql_stmt)
    assert output_path.stat().st_size == 0


@pytest.mark.parametrize("suffix", DEFAULT_SUFFIXES)
@pytest.mark.parametrize(
    "nb_parallel, has_batch_filter, exp_raise",
    [(1, False, False), (2, True, False), (2, False, True)],
)
def test_select_batch_filter(
    tmp_path, suffix, nb_parallel, has_batch_filter, exp_raise
):
    """
    Test if batch_filter checks are OK.
    """
    input_path = test_helper.get_testfile("polygon-parcel", tmp_path, suffix)
    output_path = tmp_path / f"{input_path.stem}-output{suffix}"
    sql_stmt = """
        SELECT {geometrycolumn}
              {columns_to_select_str}
          FROM "{input_layer}" layer
         WHERE 1=1
    """
    if has_batch_filter:
        sql_stmt += "{batch_filter}"

    if exp_raise:
        with pytest.raises(
            ValueError,
            match="Number batches > 1 requires a batch_filter placeholder in ",
        ):
            gfo.select(input_path, output_path, sql_stmt, nb_parallel=nb_parallel)
    else:
        gfo.select(input_path, output_path, sql_stmt, nb_parallel=nb_parallel)
