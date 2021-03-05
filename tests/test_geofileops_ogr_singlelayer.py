# -*- coding: utf-8 -*-
"""
Tests for operations using ogr on one layer.
"""

from pathlib import Path
import sys

# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geofileops import geofile
from geofileops.geofile import GeometryType
from geofileops.util import geofileops_ogr
from geofileops.util import ogr_util
from tests import test_helper

def test_buffer_gpkg(tmpdir):
    # Buffer to test dir, and try with and without gdal_bin set
    input_path = test_helper.get_testdata_dir() / 'polygons_parcels.gpkg'
    output_path = Path(tmpdir) / 'polygons_parcels_output.gpkg'
    basetest_buffer(input_path, output_path, gdal_installation='gdal_default')
    basetest_buffer(input_path, output_path, gdal_installation='gdal_bin')

    # Buffer point source to test dir
    input_path = test_helper.get_testdata_dir() / 'points.gpkg'
    output_path = Path(tmpdir) / 'points_output.gpkg'
    basetest_buffer(input_path, output_path, gdal_installation='gdal_default')
    basetest_buffer(input_path, output_path, gdal_installation='gdal_bin')

    # Buffer line source to test dir
    input_path = test_helper.get_testdata_dir() / 'linestrings_rows_of_trees.gpkg'
    output_path = Path(tmpdir) / 'linestrings_rows_of_trees_output.gpkg'
    basetest_buffer(input_path, output_path, gdal_installation='gdal_default')
    basetest_buffer(input_path, output_path, gdal_installation='gdal_bin')

def test_buffer_shp(tmpdir):
    # Buffer to test dir
    input_path = test_helper.get_testdata_dir() / 'polygons_parcels.shp'
    output_path = Path(tmpdir) / 'polygons_parcels_output.shp'

    # Try both with and without gdal_bin set
    basetest_buffer(input_path, output_path, gdal_installation='gdal_default')
    basetest_buffer(input_path, output_path, gdal_installation='gdal_bin')
    
def basetest_buffer(
        input_path: Path, 
        output_basepath: Path, 
        gdal_installation: str):
    
    # Do operation
    output_path = output_basepath.parent / f"{output_basepath.stem}_{gdal_installation}{output_basepath.suffix}"
    with test_helper.GdalBin(gdal_installation):
        ok_expected = test_helper.is_gdal_ok('buffer', gdal_installation)
        try:
            geofileops_ogr.buffer(input_path=input_path, output_path=output_path, distance=1)
            test_ok = True
        except ogr_util.SQLNotSupportedException:
            test_ok = False
    assert test_ok is ok_expected, f"Error: for {gdal_installation}, test_ok: {test_ok}, expected: {ok_expected}"

    # If it is expected not to be OK, don't do other checks
    if ok_expected is False:
        return

    # Now check if the tmp file is correctly created
    layerinfo_orig = geofile.get_layerinfo(input_path)
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_orig.featurecount == layerinfo_output.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)

    # Check geometry type
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON 

    # Now check the contents of the result file
    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None
    geofile.remove(output_path)

def test_convexhull_gpkg(tmpdir):
    # Execute to test dir
    input_path = test_helper.get_testdata_dir() / 'polygons_parcels.gpkg'
    output_path = Path(tmpdir) / 'polygons_parcels_output.gpkg'
    
    # Try both with and without gdal_bin set
    basetest_convexhull(input_path, output_path, gdal_installation='gdal_bin')
    basetest_convexhull(input_path, output_path, gdal_installation='gdal_default')

def test_convexhull_shp(tmpdir):
    # Select some data from src to tmp file
    input_path = test_helper.get_testdata_dir() / 'polygons_parcels.shp'
    output_path = Path(tmpdir) / 'polygons_parcels_output.shp'

    # Try both with and without gdal_bin set
    basetest_convexhull(input_path, output_path, gdal_installation='gdal_bin')
    basetest_convexhull(input_path, output_path, gdal_installation='gdal_default')

def basetest_convexhull(
        input_path: Path, 
        output_basepath: Path, 
        gdal_installation: str):
    
    # Do operation  
    output_path = output_basepath.parent / f"{output_basepath.stem}_{gdal_installation}{output_basepath.suffix}"
    with test_helper.GdalBin(gdal_installation):
        ok_expected = test_helper.is_gdal_ok('', gdal_installation)
        try:
            geofileops_ogr.convexhull(input_path=input_path, output_path=output_path)
            test_ok = True
        except ogr_util.SQLNotSupportedException:
            test_ok = False
    assert test_ok is ok_expected, f"Error: for {gdal_installation}, test_ok: {test_ok}, expected: {ok_expected}"

    # If it is expected not to be OK, don't do other checks
    if ok_expected is False:
        return

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_orig = geofile.get_layerinfo(input_path)
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_orig.featurecount == layerinfo_output.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)

    # Check geometry type
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON 

    # Now check the contents of the result file
    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_isvalid_gpkg(tmpdir):
    # Buffer to test dir
    input_path = test_helper.get_testdata_dir() / 'polygons_parcels.gpkg'
    output_path = Path(tmpdir) / 'polygons_parcels_output.gpkg'

    # Try both with and without gdal_bin set
    basetest_isvalid(input_path, output_path, gdal_installation='gdal_bin')
    basetest_isvalid(input_path, output_path, gdal_installation='gdal_default')

def test_isvalid_shp(tmpdir):
    # Select some data from src to tmp file
    input_path = test_helper.get_testdata_dir() / 'polygons_parcels.shp'
    output_path = Path(tmpdir) / 'polygons_parcels_output.gpkg'
    
    # Try both with and without gdal_bin set
    basetest_isvalid(input_path, output_path, gdal_installation='gdal_bin')
    basetest_isvalid(input_path, output_path, gdal_installation='gdal_default')
    
def basetest_isvalid(
        input_path: Path, 
        output_basepath: Path, 
        gdal_installation: str,
        ok_expected: bool = None):
    
    # Do operation
    output_path = output_basepath.parent / f"{output_basepath.stem}_{gdal_installation}{output_basepath.suffix}"
    with test_helper.GdalBin(gdal_installation):
        if ok_expected is None:
            ok_expected = test_helper.is_gdal_ok('isvalid', gdal_installation)
        try:
            geofileops_ogr.isvalid(input_path=input_path, output_path=output_path, nb_parallel=2)
            test_ok = True
        except ogr_util.SQLNotSupportedException:
            test_ok = False
    assert test_ok is ok_expected, f"Error: for {gdal_installation}, test_ok: {test_ok}, expected: {ok_expected}"

    # If it is expected not to be OK, don't do other checks
    if ok_expected is False:
        return

    '''
    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_select = geofile.getlayerinfo(output_path)
    assert layerinfo_orig.featurecount == layerinfo_select.featurecount
    assert (len(layerinfo_orig.columns)+3) == len(layerinfo_select.columns)

    output_gdf = geofile.read_file(output_path)
    print(output_gdf)
    assert output_gdf['geom'][0] is None
    assert output_gdf['isvalid'][0] == 1
    assert output_gdf['isvalidreason'][0] == 'Valid Geometry'
    '''

def test_makevalid_gpkg(tmpdir):
    # makevalid to test dir
    input_path = test_helper.get_testdata_dir() / 'polygons_invalid_geometries.gpkg'
    output_path = Path(tmpdir) / f"{input_path.stem}_valid.gpkg"
    
    # Try both with and without gdal_bin set
    basetest_makevalid(input_path, output_path, gdal_installation='gdal_bin')
    basetest_makevalid(input_path, output_path, gdal_installation='gdal_default')

def test_makevalid_shp(tmpdir):
    # makevalid to test dir
    input_path = test_helper.get_testdata_dir() / 'polygons_invalid_geometries.shp'
    output_path = Path(tmpdir) / f"{input_path.stem}_valid.shp"
    
    # Try both with and without gdal_bin set
    basetest_makevalid(input_path, output_path, gdal_installation='gdal_bin')
    basetest_makevalid(input_path, output_path, gdal_installation='gdal_default')
           
def basetest_makevalid(
        input_path: Path, 
        output_basepath: Path, 
        gdal_installation: str,
        ok_expected: bool = None):

    # Do operation
    output_path = output_basepath.parent / f"{output_basepath.stem}_{gdal_installation}{output_basepath.suffix}"
    with test_helper.GdalBin(gdal_installation):
        if ok_expected is None:
            ok_expected = test_helper.is_gdal_ok('makevalid', gdal_installation)
        try: 
            geofileops_ogr.makevalid(input_path=input_path, output_path=output_path, nb_parallel=2)
            test_ok = True
        except ogr_util.SQLNotSupportedException:
            test_ok = False
        assert test_ok is ok_expected, f"Error: for {gdal_installation}, test_ok: {test_ok}, expected: {ok_expected}"

        # If it is expected not to be OK, don't do other checks
        if ok_expected is False:
            return

        # Now check if the output file is correctly created
        assert output_path.exists() == True
        layerinfo_orig = geofile.get_layerinfo(input_path)
        layerinfo_output = geofile.get_layerinfo(output_path)
        assert layerinfo_orig.featurecount == layerinfo_output.featurecount
        assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)
    
        # Check geometry type
        assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON 

        # Now check the contents of the result file
        output_gdf = geofile.read_file(output_path)
        assert output_gdf['geometry'][0] is not None

        # Make sure the input file was not valid
        output_isvalid_path = output_path.parent / f"{output_path.stem}_isvalid{output_path.suffix}"
        isvalid = geofileops_ogr.isvalid(input_path=input_path, output_path=output_isvalid_path)
        assert isvalid is False, "Input file should contain invalid features"

        # Check if the result file is valid
        output_new_isvalid_path = output_path.parent / f"{output_path.stem}_new_isvalid{output_path.suffix}"
        isvalid = geofileops_ogr.isvalid(input_path=output_path, output_path=output_new_isvalid_path)
        assert isvalid == True, "Output file shouldn't contain invalid features"

def test_select_gpkg(tmpdir):
    # Select some data from src to tmp file
    input_path = test_helper.get_testdata_dir() / 'polygons_parcels.gpkg'
    output_path = Path(tmpdir) / 'polygons_parcels_output.gpkg'

    basetest_select(input_path, output_path)

def test_select_gpkg_to_shp(tmpdir):
    # Select some data from src to tmp file
    input_path = test_helper.get_testdata_dir() / 'polygons_parcels.gpkg'
    output_path = Path(tmpdir) / 'polygons_parcels_output.shp'

    basetest_select(input_path, output_path)

def test_select_shp(tmpdir):
    # Select some data from src to tmp file
    input_path = test_helper.get_testdata_dir() / 'polygons_parcels.shp'
    output_path = Path(tmpdir) / 'polygons_parcels_output.shp'

    basetest_select(input_path, output_path)

def test_select_shp_to_gpkg(tmpdir):
    # Select some data from src to tmp file
    input_path = test_helper.get_testdata_dir() / 'polygons_parcels.shp'
    output_path = Path(tmpdir) / 'polygons_parcels_output.gpkg'

    basetest_select(input_path, output_path)

def basetest_select(
        input_path: Path, 
        output_path: Path):

    layerinfo_input = geofile.get_layerinfo(input_path)
    sql_stmt = 'SELECT {geometrycolumn}, oidn, uidn FROM {input_layer}'
    geofileops_ogr.select(
            input_path=input_path,
            output_path=output_path,
            sql_stmt=sql_stmt)

    # Now check if the tmp file is correctly created
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_input.featurecount == layerinfo_output.featurecount
    assert 'OIDN' in layerinfo_output.columns
    assert 'UIDN' in layerinfo_output.columns
    assert len(layerinfo_output.columns) == 2

    # Check geometry type
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON 

    # Now check the contents of the result file
    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_select_various_options_shp(tmpdir):
    # Select some data from src to tmp file
    input_path = test_helper.get_testdata_dir() / 'polygons_parcels.shp'
    output_path = Path(tmpdir) / 'polygons_parcels_output.shp'

    basetest_select_various_options(input_path, output_path)

def test_select_various_options_gpkg(tmpdir):
    # Select some data from src to tmp file
    input_path = test_helper.get_testdata_dir() / 'polygons_parcels.gpkg'
    output_path = Path(tmpdir) / 'polygons_parcels_output.gpkg'

    basetest_select_various_options(input_path, output_path)
    
def basetest_select_various_options(
        input_path: Path, 
        output_path: Path):

    ### Check if columns parameter works (case insensitive) ###
    columns = ['OIDN', 'uidn', 'HFDTLT', 'lblhfdtlt', 'GEWASGROEP', 'lengte', 'OPPERVL']
    layerinfo_input = geofile.get_layerinfo(input_path)
    sql_stmt = '''SELECT {geometrycolumn}
                        {columns_to_select_str} 
                    FROM {input_layer} '''
    geofileops_ogr.select(
            input_path=input_path,
            output_path=output_path,
            columns=columns,
            sql_stmt=sql_stmt)

    # Now check if the tmp file is correctly created
    layerinfo_select = geofile.get_layerinfo(output_path)
    assert layerinfo_input.featurecount == layerinfo_select.featurecount
    assert 'OIDN' in layerinfo_select.columns
    assert 'UIDN' in layerinfo_select.columns
    assert len(layerinfo_select.columns) == len(columns)

    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

    ### Check if ... parameter works ###
    # TODO: increase test coverage of other options...

def test_simplify_gpkg(tmpdir):
    # Simplify polygon source to test dir, with and without gdal_bin set
    input_path = test_helper.get_testdata_dir() / 'polygons_parcels.gpkg'
    output_path = Path(tmpdir) / input_path.name
    basetest_simplify(input_path, output_path, 
            expected_output_geometrytype=GeometryType.MULTIPOLYGON,
            gdal_installation='gdal_default')
    basetest_simplify(input_path, output_path, 
            expected_output_geometrytype=GeometryType.MULTIPOLYGON,
            gdal_installation='gdal_bin')

    # Simplify point source to test dir
    input_path = test_helper.get_testdata_dir() / 'points.gpkg'
    output_path = Path(tmpdir) / 'points_output.gpkg'
    basetest_simplify(input_path, output_path, 
            expected_output_geometrytype=GeometryType.MULTIPOINT,
            gdal_installation='gdal_default')
    basetest_simplify(input_path, output_path, 
            expected_output_geometrytype=GeometryType.MULTIPOINT,
            gdal_installation='gdal_bin')

    # Simplify line source to test dir
    input_path = test_helper.get_testdata_dir() / 'linestrings_rows_of_trees.gpkg'
    output_path = Path(tmpdir) / 'linestrings_rows_of_trees_output.gpkg'
    basetest_simplify(input_path, output_path, 
            expected_output_geometrytype=GeometryType.MULTILINESTRING,
            gdal_installation='gdal_default')
    basetest_simplify(input_path, output_path, 
            expected_output_geometrytype=GeometryType.MULTILINESTRING,
            gdal_installation='gdal_bin')

def test_simplify_shp(tmpdir):
    # Simplify to test dir
    input_path = test_helper.get_testdata_dir() / 'polygons_parcels.shp'
    output_path = Path(tmpdir) / 'polygons_parcels.shp'

    # Try both with and without gdal_bin set
    basetest_simplify(input_path, output_path, 
            expected_output_geometrytype=GeometryType.MULTIPOLYGON,
            gdal_installation='gdal_default')
    basetest_simplify(input_path, output_path, 
            expected_output_geometrytype=GeometryType.MULTIPOLYGON,
            gdal_installation='gdal_bin')
    
def basetest_simplify(
        input_path: Path, 
        output_basepath: Path,
        expected_output_geometrytype: GeometryType,
        gdal_installation: str):

    # Do operation
    output_path = output_basepath.parent / f"{output_basepath.stem}_{gdal_installation}{output_basepath.suffix}"
    with test_helper.GdalBin(gdal_installation):
        ok_expected = test_helper.is_gdal_ok('', gdal_installation)
        try:
            geofileops_ogr.simplify(
                    input_path=input_path, output_path=output_path,
                    tolerance=5)
            test_ok = True
        except ogr_util.SQLNotSupportedException:
            test_ok = False
    assert test_ok is ok_expected, f"Error: for {gdal_installation}, test_ok: {test_ok}, expected: {ok_expected}"

    # If it is expected not to be OK, don't do other checks
    if ok_expected is False:
        return

    # Now check if the output file is correctly created
    assert output_path.exists() == True
    layerinfo_orig = geofile.get_layerinfo(input_path)
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_orig.featurecount == layerinfo_output.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)

    # Check geometry type
    assert layerinfo_output.geometrytype == expected_output_geometrytype 

    # Now check the contents of the result file
    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

if __name__ == '__main__':
    # Init
    tmpdir = test_helper.init_test_for_debug(Path(__file__).stem)

    # Single layer operations
    test_buffer_gpkg(tmpdir)
    #test_makevalid_shp(tmpdir)
    #test_makevalid_gpkg(tmpdir)
    #test_isvalid_shp(tmpdir)
    #test_isvalid_gpkg(tmpdir)
    #test_convexhull_shp(tmpdir)
    #test_convexhull_gpkg(tmpdir)
    #test_select_geos_version(tmpdir)
