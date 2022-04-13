# -*- coding: utf-8 -*-
"""
Tests for operations that are executed using a sql statement on one layer.
"""

from importlib import import_module
from pathlib import Path
import sys

import pytest

# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geofileops import geoops
from geofileops import fileops 
from geofileops import GeometryType
from geofileops.util import _io_util
from tests import test_helper
from tests.test_helper import TestFiles

# Init gfo module
current_fileops_module = "geofileops.fileops"

def set_fileops_module(fileops_module: str):
    global current_fileops_module
    if current_fileops_module == fileops_module:
        # The right module is already loaded, so don't do anything
        return
    else:
        # Load the desired module as fileops
        global geoops
        geoops = import_module(fileops_module, __package__)
        current_fileops_module = fileops_module
        print(f"gfo module switched to: {current_fileops_module}")

def get_nb_parallel() -> int:
    # The number of parallel processes to use for these tests.
    return 2

def get_batchsize() -> int:
    return 5

def get_combinations_to_test() -> list:
    result = []
    fileops_modules = ["geofileops.geoops", "geofileops.util._geoops_gpd", "geofileops.util._geoops_sql"]
    testfiles = [   
            (TestFiles.polygons_parcels_gpkg, GeometryType.MULTIPOLYGON),
            (TestFiles.points_gpkg, GeometryType.MULTIPOINT),
            (TestFiles.linestrings_rows_of_trees_gpkg, GeometryType.MULTILINESTRING) ]

    # Test all combination of fileops_modules, testfiles, crs_epsg 31370 on .shp 
    for fileops_module in fileops_modules:
        for testfile_path, testfile_geometrytype in testfiles:
            result.append((".shp", 31370, fileops_module, testfile_path, testfile_geometrytype))
    
    # Test all combination of fileops_modules, testfiles, crs_epsgs on .gpkg 
    for crs_epsg in test_helper.get_test_crs_epsg_list():
        for fileops_module in fileops_modules:
            for testfile_path, testfile_geometrytype in testfiles:
                result.append((".gpkg", crs_epsg, fileops_module, testfile_path, testfile_geometrytype))
        
    return result

@pytest.mark.parametrize(
        "suffix, crs_epsg, fileops_module, input_path, expected_geometrytype", 
        get_combinations_to_test())
def test_buffer_basic(tmpdir, suffix, crs_epsg, fileops_module, input_path, expected_geometrytype):
    """ 
    Buffer basics are available both in the gpd and sql implementations. 
    """
    # Prepare test data
    tmp_basedir = Path(tmpdir)
    tmp_dir = tmp_basedir / fileops_module
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # If test input file is in wrong format, convert it
    input_path = test_helper.prepare_test_file(
            input_path=input_path,
            output_dir=tmp_dir,
            suffix=suffix,
            crs_epsg=crs_epsg)

    # Now run test
    output_path = tmp_dir / f"{input_path.stem}-output{suffix}"
    set_fileops_module(fileops_module)
    layerinfo_input = fileops.get_layerinfo(input_path)
    assert layerinfo_input.crs is not None
    distance = 1
    if layerinfo_input.crs.is_projected is False:
        # 1 degree = 111 km or 111000 m
        distance /= 111000

    ### Test positive buffer ###
    geoops.buffer(
            input_path=input_path,
            output_path=output_path,
            distance=distance,
            nb_parallel=get_nb_parallel(),
            batchsize=get_batchsize())

    # Now check if the output file is correctly created
    assert output_path.exists() == True
    layerinfo_output = fileops.get_layerinfo(output_path)
    assert layerinfo_input.featurecount == layerinfo_output.featurecount
    assert len(layerinfo_output.columns) == len(layerinfo_input.columns)
    
    # Check geometry type
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON 

    # Read result for some more detailed checks
    output_gdf = fileops.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

    ### Test buffer to existing output path ###
    assert output_path.exists() is True
    mtime_orig = output_path.stat().st_mtime
    geoops.buffer(
            input_path=input_path,
            output_path=output_path,
            distance=distance,
            nb_parallel=get_nb_parallel())
    assert output_path.stat().st_mtime == mtime_orig

    # With force=True
    geoops.buffer(
            input_path=input_path,
            output_path=output_path,
            distance=distance,
            nb_parallel=get_nb_parallel(),
            force=True)
    assert output_path.stat().st_mtime != mtime_orig

    ### Test negative buffer ###
    distance = -10
    if layerinfo_input.crs.is_projected is False:
        # 1 degree = 111 km or 111000 m
        distance /= 111000

    output_path = output_path.parent / f"{output_path.stem}_m10m{output_path.suffix}"
    geoops.buffer(
            input_path=input_path,
            output_path=output_path,
            distance=distance,
            nb_parallel=get_nb_parallel())

    # Now check if the output file is correctly created
    if expected_geometrytype in [GeometryType.MULTIPOINT, GeometryType.MULTILINESTRING]:
        # A Negative buffer of points or linestrings doesn't give a result.
        assert output_path.exists() == False
    else:    
        # A Negative buffer of polygons gives a result for large polygons.
        assert output_path.exists() == True
        layerinfo_output = fileops.get_layerinfo(output_path)
        assert len(layerinfo_output.columns) == len(layerinfo_input.columns) 
        if layerinfo_input.crs.is_projected is True:
            # 7 polygons disappear because of the negative buffer
            assert layerinfo_output.featurecount == layerinfo_input.featurecount - 7
        else:
            assert layerinfo_output.featurecount == layerinfo_input.featurecount - 4
        
        # Check geometry type
        assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON

        # Read result for some more detailed checks
        output_gdf = fileops.read_file(output_path)
        assert output_gdf['geometry'][0] is not None
    
    ### Test negative buffer with explodecollections ###
    output_path = output_path.parent / f"{output_path.stem}_m10m_explode{output_path.suffix}"
    geoops.buffer(
            input_path=input_path,
            output_path=output_path,
            distance=distance,
            explodecollections=True,
            nb_parallel=get_nb_parallel())

    # Now check if the output file is correctly created
    if expected_geometrytype in [GeometryType.MULTIPOINT, GeometryType.MULTILINESTRING]:
        # A Negative buffer of points or linestrings doesn't give a result.
        assert output_path.exists() == False
    else:    
        # A Negative buffer of polygons gives a result for large polygons
        assert output_path.exists() == True
        layerinfo_output = fileops.get_layerinfo(output_path)
        assert len(layerinfo_output.columns) == len(layerinfo_input.columns) 

        if layerinfo_input.crs.is_projected is True:
            # 6 polygons disappear because of the negative buffer, 3 polygons are 
            # split in 2 because of the negative buffer and/or explodecollections=True.
            assert layerinfo_output.featurecount == layerinfo_input.featurecount - 7 + 3
        else:
            assert layerinfo_output.featurecount == layerinfo_input.featurecount - 3 + 3

        # Check geometry type
        assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON

        # Read result for some more detailed checks
        output_gdf = fileops.read_file(output_path)
        assert output_gdf['geometry'][0] is not None

def test_convexhull(tmpdir):
    # Prepare test data + run tests
    tmp_basedir = Path(tmpdir)

    for fileops_module in ["geofileops.geoops", "geofileops.util._geoops_gpd"]:
        tmp_dir = tmp_basedir / fileops_module
        tmp_dir.mkdir(parents=True, exist_ok=True)

        for suffix in test_helper.get_test_suffix_list():
            for crs_epsg in test_helper.get_test_crs_epsg_list():
                # If test input file is in wrong format, convert it
                input_path = test_helper.prepare_test_file(
                        input_path=test_helper.TestFiles.polygons_parcels_gpkg,
                        output_dir=tmp_dir,
                        suffix=suffix,
                        crs_epsg=crs_epsg)

                # Now run test
                output_path = tmp_dir / f"{input_path.stem}-output{suffix}"
                print(f"Run test for suffix {suffix}, crs_epsg {crs_epsg}")
                basetest_convexhull(input_path, output_path, fileops_module)

def basetest_convexhull(
        input_path: Path, 
        output_path: Path, 
        fileops_module: str):
    
    set_fileops_module(fileops_module)
    layerinfo_orig = fileops.get_layerinfo(input_path)
    
    # Also check if columns parameter works (case insensitive)
    columns = ['OIDN', 'uidn', 'HFDTLT', 'lblhfdtlt', 'GEWASGROEP', 'lengte', 'OPPERVL']
    geoops.convexhull(
            input_path=input_path,
            columns=columns,
            output_path=output_path,
            nb_parallel=get_nb_parallel())

    # Now check if the output file is correctly created
    assert output_path.exists() == True
    layerinfo_output = fileops.get_layerinfo(output_path)
    assert layerinfo_orig.featurecount == layerinfo_output.featurecount
    assert 'OIDN' in layerinfo_output.columns
    assert 'UIDN' in layerinfo_output.columns
    assert len(layerinfo_output.columns) == len(columns)
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON 

    # Read result for some more detailed checks
    output_gdf = fileops.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

@pytest.mark.parametrize(
        "suffix, crs_epsg, fileops_module, input_path, expected_geometrytype", 
        get_combinations_to_test())
def test_simplify_basic(tmpdir, suffix, crs_epsg, fileops_module, input_path, expected_geometrytype):
    # Prepare test data
    tmp_basedir = Path(tmpdir)
    tmp_dir = tmp_basedir / f"{fileops_module}_{crs_epsg}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    input_path = test_helper.prepare_test_file(
            input_path=input_path,
            output_dir=tmp_dir,
            suffix=suffix,
            crs_epsg=crs_epsg)
    output_path = tmp_dir / f"{input_path.stem}-output{suffix}"
    set_fileops_module(fileops_module)
    layerinfo_orig = fileops.get_layerinfo(input_path)
    assert layerinfo_orig.crs is not None
    if layerinfo_orig.crs.is_projected:
        tolerance = 5
    else:
        # 1 degree = 111 km or 111000 m
        tolerance = 5/111000

    ### Test default algorithm (rdp) ###
    output_path = _io_util.with_stem(input_path, output_path)
    geoops.simplify(
            input_path=input_path,
            output_path=output_path,
            tolerance=tolerance,
            nb_parallel=get_nb_parallel())

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_output = fileops.get_layerinfo(output_path)
    assert layerinfo_orig.featurecount == layerinfo_output.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)
    assert layerinfo_output.geometrytype == expected_geometrytype

    # Now check the contents of the result file
    input_gdf = fileops.read_file(input_path)
    output_gdf = fileops.read_file(output_path)
    assert input_gdf.crs == output_gdf.crs
    assert len(output_gdf) == layerinfo_output.featurecount
    assert output_gdf['geometry'][0] is not None
    #expected_gdf = fileops.read_file(input_path)
    #expected_gdf.geometry = expected_gdf.geometry.simplify(tolerance=tolerance)
    #assert_geodataframe_equal(output_gdf, expected_gdf)
