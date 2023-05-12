# -*- coding: utf-8 -*-
"""
Tests for operations that are executed using a sql statement on one layer.
"""

from importlib import import_module
import logging
import math
from typing import List

import geopandas._compat as gpd_compat
import pytest

if gpd_compat.USE_PYGEOS:
    import pygeos as shapely2_or_pygeos
else:
    import shapely as shapely2_or_pygeos

from geofileops import geoops
from geofileops import fileops
from geofileops import GeometryType
from geofileops.util import _io_util as io_util
from tests import test_helper
from tests.test_helper import (
    EPSGS,
    GRIDSIZE_DEFAULT,
    SUFFIXES,
    TESTFILES,
)
from tests.test_helper import assert_geodataframe_equal

# Init gfo module
current_geoops_module = "unknown"
GEOOPS_MODULES = ["geofileops.geoops", "geofileops.util._geoops_gpd"]


def set_geoops_module(geoops_module: str):
    global current_geoops_module
    if current_geoops_module == geoops_module:
        # The right module is already loaded, so don't do anything
        return
    else:
        # Load the desired module as fileops
        global geoops
        geoops = import_module(geoops_module, __package__)
        current_geoops_module = geoops_module
        print(f"gfo module switched to: {current_geoops_module}")


def get_combinations_to_test(
    geoops_modules: List[str] = GEOOPS_MODULES,
    testfiles: List[str] = TESTFILES,
    epsgs: List[int] = EPSGS,
    suffixes: List[str] = SUFFIXES,
) -> list:
    """
    Return sensible combinations of parameters to be used in tests for following params:
        suffix, epsg, geoops_module, testfile, empty_input, gridsize
    """
    result = []

    # On .gpkg test:
    #   - all combinations of geoops_modules, testfiles and epsgs
    #   - fixed empty_input, suffix
    for epsg in epsgs:
        for geoops_module in geoops_modules:
            for testfile in testfiles:
                gridsize = 0.001 if epsg == 31370 else GRIDSIZE_DEFAULT
                result.append((".gpkg", epsg, geoops_module, testfile, False, gridsize))

    # On other suffixes test:
    #   - all combinations of geoops_modules, testfiles
    #   - fixed epsg and empty_input
    other_suffixes = list(suffixes)
    other_suffixes.remove(".gpkg")
    for suffix in other_suffixes:
        for geoops_module in geoops_modules:
            for testfile in testfiles:
                gridsize = 0.001 if testfile == "polygon-parcel" else 0.0
                result.append((suffix, 31370, geoops_module, testfile, False, 0.0))

    # Test empty_input=True on
    #   - all combinations of geoops_modules and DEFAULT_SUFFIXES
    #   - fixed epsg, testfile and empty_input
    for geoops_module in geoops_modules:
        for suffix in suffixes:
            gridsize = 0.001 if suffix == ".gpkg" else GRIDSIZE_DEFAULT
            result.append(
                (suffix, 31370, geoops_module, "polygon-parcel", True, gridsize)
            )

    return result


@pytest.mark.parametrize(
    "suffix, epsg, geoops_module, testfile, empty_input, gridsize",
    get_combinations_to_test(),
)
def test_buffer(tmp_path, suffix, epsg, geoops_module, testfile, empty_input, gridsize):
    """Buffer basics are available both in the gpd and sql implementations."""
    # Prepare test data
    input_path = test_helper.get_testfile(
        testfile, suffix=suffix, epsg=epsg, empty=empty_input
    )

    # Now run test
    output_path = tmp_path / f"{input_path.stem}-{geoops_module}{suffix}"
    set_geoops_module(geoops_module)
    input_layerinfo = fileops.get_layerinfo(input_path)
    batchsize = math.ceil(input_layerinfo.featurecount / 2)
    assert input_layerinfo.crs is not None
    distance = 1
    if input_layerinfo.crs.is_projected is False:
        # 1 degree = 111 km or 111000 m
        distance /= 111000

    # Test positive buffer
    geoops.buffer(
        input_path=input_path,
        output_path=output_path,
        distance=distance,
        gridsize=gridsize,
        nb_parallel=2,
        batchsize=batchsize,
    )

    # Now check if the output file is correctly created
    assert output_path.exists()
    assert fileops.has_spatial_index(output_path)
    output_layerinfo = fileops.get_layerinfo(output_path)
    assert len(output_layerinfo.columns) == len(input_layerinfo.columns)

    if not empty_input:
        output_gdf = fileops.read_file(output_path)
        assert output_gdf["geometry"][0] is not None

        # Prepare expected data
        expected_gdf = fileops.read_file(input_path)
        expected_gdf.geometry = expected_gdf.geometry.buffer(
            distance=distance, resolution=5
        )
        check_less_precise = (
            True if input_layerinfo.crs.is_projected is False else False
        )
        if gridsize != 0.0:
            expected_gdf.geometry = shapely2_or_pygeos.set_precision(
                expected_gdf.geometry.array.data, grid_size=gridsize
            )
        assert_geodataframe_equal(
            output_gdf,
            expected_gdf,
            promote_to_multi=True,
            check_less_precise=check_less_precise,
            sort_values=True,
        )


@pytest.mark.parametrize("suffix", SUFFIXES)
@pytest.mark.parametrize("testfile", ["polygon-parcel"])
@pytest.mark.parametrize("geoops_module", GEOOPS_MODULES)
def test_buffer_columns_fid(tmp_path, suffix, geoops_module, testfile):
    """Buffer basics are available both in the gpd and sql implementations."""
    # Prepare test data
    input_path = test_helper.get_testfile(testfile, suffix=suffix)

    # Now run test
    output_path = tmp_path / f"{input_path.stem}-{geoops_module}{suffix}"
    set_geoops_module(geoops_module)
    input_layerinfo = fileops.get_layerinfo(input_path)
    batchsize = math.ceil(input_layerinfo.featurecount / 2)

    # Test positive buffer
    geoops.buffer(
        input_path=input_path,
        output_path=output_path,
        distance=1,
        columns=["LblHfdTlt", "fid"],
        explodecollections=True,
        nb_parallel=2,
        batchsize=batchsize,
    )

    # Read input file and extract some info
    input_gdf = fileops.read_file(input_path, fid_as_index=True)
    if fileops.GeofileType(input_path).is_fid_zerobased:
        assert input_gdf.index[0] == 0
    else:
        assert input_gdf.index[0] == 1
    input_multi_gdf = input_gdf[
        input_gdf.geometry.buffer(0).geom_type == "MultiPolygon"
    ]
    assert len(input_multi_gdf) == 2
    multi_fid = input_multi_gdf.index[0]

    # Now check if the output file is correctly created
    assert output_path.exists()
    assert fileops.has_spatial_index(output_path)
    output_layerinfo = fileops.get_layerinfo(output_path)
    output_gdf = fileops.read_file(output_path)
    assert output_gdf["geometry"][0] is not None
    assert list(output_layerinfo.columns) == ["LblHfdTlt", "fid_1"]
    assert len(output_gdf[output_gdf.fid_1 == multi_fid]) == 2


@pytest.mark.parametrize("geoops_module", GEOOPS_MODULES)
def test_buffer_force(tmp_path, geoops_module):
    input_path = test_helper.get_testfile("polygon-parcel")
    input_layerinfo = fileops.get_layerinfo(input_path)
    batchsize = math.ceil(input_layerinfo.featurecount / 2)
    distance = 1
    set_geoops_module(geoops_module)

    # Run buffer
    output_path = tmp_path / f"{input_path.stem}-output{input_path.suffix}"
    assert output_path.exists() is False

    geoops.buffer(
        input_path=input_path,
        output_path=output_path,
        distance=distance,
        nb_parallel=2,
        batchsize=batchsize,
    )

    # Test buffer to existing output path
    assert output_path.exists()
    assert fileops.has_spatial_index(output_path)
    mtime_orig = output_path.stat().st_mtime
    geoops.buffer(
        input_path=input_path,
        output_path=output_path,
        distance=distance,
        nb_parallel=2,
        batchsize=batchsize,
    )
    assert output_path.stat().st_mtime == mtime_orig

    # With force=True
    geoops.buffer(
        input_path=input_path,
        output_path=output_path,
        distance=distance,
        nb_parallel=2,
        batchsize=batchsize,
        force=True,
    )
    assert output_path.stat().st_mtime != mtime_orig


@pytest.mark.parametrize(
    "expected_error, input_path, output_path",
    [
        (
            "buffer: output_path must not equal input_path",
            test_helper.get_testfile("polygon-parcel"),
            test_helper.get_testfile("polygon-parcel"),
        ),
        (
            "buffer: input_path doesn't exist:",
            "not_existing_path",
            "output.gpkg",
        ),
    ],
)
@pytest.mark.parametrize("geoops_module", GEOOPS_MODULES)
def test_buffer_invalid_params(
    tmp_path, input_path, output_path, expected_error, geoops_module
):
    """
    Invalid params for single layer operations.
    """
    # Internal functions are directly called, so need to be Path objects
    if isinstance(output_path, str):
        output_path = tmp_path / output_path
    if isinstance(input_path, str):
        input_path = tmp_path / input_path

    # Now run test
    set_geoops_module(geoops_module)
    with pytest.raises(ValueError, match=expected_error):
        geoops.buffer(input_path=input_path, output_path=output_path, distance=1)


@pytest.mark.parametrize("suffix", SUFFIXES)
@pytest.mark.parametrize("geoops_module", GEOOPS_MODULES)
@pytest.mark.parametrize("suffix", TESTFILES)
def test_buffer_negative(tmp_path, suffix, geoops_module, testfile):
    """Buffer basics are available both in the gpd and sql implementations."""
    input_path = test_helper.get_testfile(testfile, suffix=suffix)

    # Now run test
    output_path = tmp_path / f"{input_path.stem}-{geoops_module}{suffix}"
    set_geoops_module(geoops_module)
    input_layerinfo = fileops.get_layerinfo(input_path)
    batchsize = math.ceil(input_layerinfo.featurecount / 2)

    # Test negative buffer
    distance = -10
    output_path = output_path.parent / f"{output_path.stem}_m10m{output_path.suffix}"
    geoops.buffer(
        input_path=input_path,
        output_path=output_path,
        distance=distance,
        nb_parallel=2,
        batchsize=batchsize,
    )

    # Now check if the output file is correctly created
    assert output_path.exists()
    assert fileops.has_spatial_index(output_path)
    output_layerinfo = fileops.get_layerinfo(output_path)
    assert len(output_layerinfo.columns) == len(input_layerinfo.columns)

    if input_layerinfo.geometrytype in [
        GeometryType.MULTIPOINT,
        GeometryType.MULTILINESTRING,
    ]:
        # A Negative buffer of points or linestrings doesn't give a result.
        if output_path.exists():
            assert fileops.get_layerinfo(output_path).featurecount == 0
    else:
        # A Negative buffer of polygons gives a result for large polygons.
        # 7 polygons disappear because of the negative buffer
        assert output_layerinfo.featurecount == input_layerinfo.featurecount - 7
        assert output_layerinfo.geometrytype == GeometryType.MULTIPOLYGON

        # Read result for some more detailed checks
        output_gdf = fileops.read_file(output_path)
        expected_gdf = fileops.read_file(input_path)
        expected_gdf.geometry = expected_gdf.geometry.buffer(
            distance=distance, resolution=5
        )
        # Remove rows where geom is empty
        expected_gdf = expected_gdf[~expected_gdf.geometry.is_empty]
        expected_gdf = expected_gdf[~expected_gdf.geometry.isna()]
        assert_geodataframe_equal(output_gdf, expected_gdf, sort_values=True)


@pytest.mark.parametrize("geoops_module", GEOOPS_MODULES)
def test_buffer_negative_explode(tmp_path, geoops_module):
    """Buffer basics are available both in the gpd and sql implementations."""
    input_path = test_helper.get_testfile("polygon-parcel")

    # Now run test
    output_path = tmp_path / f"{input_path.stem}-output{input_path.suffix}"
    set_geoops_module(geoops_module)
    input_layerinfo = fileops.get_layerinfo(input_path)
    batchsize = math.ceil(input_layerinfo.featurecount / 2)

    # Test negative buffer with explodecollections
    output_path = (
        output_path.parent / f"{output_path.stem}_m10m_explode{output_path.suffix}"
    )
    distance = -10
    geoops.buffer(
        input_path=input_path,
        output_path=output_path,
        distance=distance,
        explodecollections=True,
        nb_parallel=2,
        batchsize=batchsize,
    )

    # Now check if the output file is correctly created
    assert output_path.exists()
    assert fileops.has_spatial_index(output_path)
    layerinfo_output = fileops.get_layerinfo(output_path)
    assert len(layerinfo_output.columns) == len(input_layerinfo.columns)

    # 6 polygons disappear because of the negative buffer, 3 polygons are
    # split in 2 because of the negative buffer and/or explodecollections=True.
    assert layerinfo_output.featurecount == input_layerinfo.featurecount - 7 + 3
    assert layerinfo_output.geometrytype == GeometryType.POLYGON

    # Read result for some more detailed checks
    output_gdf = fileops.read_file(output_path)
    expected_gdf = fileops.read_file(input_path)
    expected_gdf.geometry = expected_gdf.geometry.buffer(
        distance=distance, resolution=5
    )
    # Remove rows where geom is empty
    expected_gdf = expected_gdf[~expected_gdf.geometry.is_empty]
    expected_gdf = expected_gdf[~expected_gdf.geometry.isna()]
    expected_gdf = expected_gdf.explode(ignore_index=True)  # type: ignore
    assert_geodataframe_equal(
        output_gdf, expected_gdf, promote_to_multi=True, sort_values=True
    )


@pytest.mark.parametrize("suffix", SUFFIXES)
@pytest.mark.parametrize("geoops_module", GEOOPS_MODULES)
@pytest.mark.parametrize("empty_input, gridsize", [(True, 0.0), (False, 0.001)])
def test_convexhull(tmp_path, geoops_module, suffix, empty_input, gridsize):
    logging.basicConfig(level=logging.DEBUG)
    input_path = test_helper.get_testfile(
        "polygon-parcel", suffix=suffix, empty=empty_input
    )
    output_path = tmp_path / f"{input_path.stem}-output{suffix}"
    set_geoops_module(geoops_module)
    input_layerinfo = fileops.get_layerinfo(input_path)
    batchsize = math.ceil(input_layerinfo.featurecount / 2)

    # Also check if columns parameter works (case insensitive)
    columns = ["OIDN", "uidn", "HFDTLT", "lblhfdtlt", "GEWASGROEP", "lengte", "OPPERVL"]
    geoops.convexhull(
        input_path=input_path,
        columns=columns,
        output_path=output_path,
        gridsize=gridsize,
        nb_parallel=2,
        batchsize=batchsize,
    )

    # Now check if the output file is correctly created
    assert output_path.exists()
    assert fileops.has_spatial_index(output_path)
    layerinfo_output = fileops.get_layerinfo(output_path)
    assert "OIDN" in layerinfo_output.columns
    assert "uidn" in layerinfo_output.columns
    assert len(layerinfo_output.columns) == len(columns)
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON

    if not empty_input:
        assert input_layerinfo.featurecount == layerinfo_output.featurecount

        # Read result for some more detailed checks
        output_gdf = fileops.read_file(output_path)
        assert output_gdf["geometry"][0] is not None
        expected_gdf = fileops.read_file(input_path, columns=columns)
        expected_gdf.geometry = expected_gdf.geometry.convex_hull
        if gridsize != 0.0:
            expected_gdf.geometry = shapely2_or_pygeos.set_precision(
                expected_gdf.geometry.array.data, grid_size=gridsize
            )
        assert_geodataframe_equal(output_gdf, expected_gdf, sort_values=True)


@pytest.mark.parametrize(
    "suffix, epsg, geoops_module, testfile, empty_input, gridsize",
    get_combinations_to_test(
        geoops_modules=GEOOPS_MODULES,
        testfiles=["polygon-parcel", "linestring-row-trees"],
    ),
)
def test_simplify(
    tmp_path, suffix, epsg, geoops_module, testfile, empty_input, gridsize
):
    # Prepare test data
    tmp_dir = tmp_path / f"{geoops_module}_{epsg}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    input_path = test_helper.get_testfile(
        testfile, dst_dir=tmp_dir, suffix=suffix, epsg=epsg, empty=empty_input
    )
    output_path = tmp_dir / f"{input_path.stem}-output{suffix}"
    set_geoops_module(geoops_module)
    input_layerinfo = fileops.get_layerinfo(input_path)
    batchsize = math.ceil(input_layerinfo.featurecount / 2)
    assert input_layerinfo.crs is not None
    if input_layerinfo.crs.is_projected:
        tolerance = 5
    else:
        # 1 degree = 111 km or 111000 m
        tolerance = 5 / 111000

    # Test default algorithm (rdp)
    output_path = io_util.with_stem(input_path, output_path)
    geoops.simplify(
        input_path=input_path,
        output_path=output_path,
        tolerance=tolerance,
        gridsize=gridsize,
        nb_parallel=2,
        batchsize=batchsize,
    )

    # Now check if the tmp file is correctly created
    assert output_path.exists()
    assert fileops.has_spatial_index(output_path)
    output_layerinfo = fileops.get_layerinfo(output_path)
    assert len(output_layerinfo.columns) == len(input_layerinfo.columns)

    if not empty_input:
        output_gdf = fileops.read_file(output_path)
        expected_gdf = fileops.read_file(input_path)
        expected_gdf.geometry = expected_gdf.geometry.simplify(
            tolerance=tolerance, preserve_topology=True
        )
        if gridsize != 0.0:
            expected_gdf.geometry = shapely2_or_pygeos.set_precision(
                expected_gdf.geometry.array.data, grid_size=gridsize
            )
        assert_geodataframe_equal(output_gdf, expected_gdf, sort_values=True)
