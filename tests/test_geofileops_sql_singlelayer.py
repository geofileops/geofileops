# -*- coding: utf-8 -*-
"""
Tests for operations that are executed using a sql statement on one layer.
"""

from pathlib import Path
import sys

# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geofileops import geofile
from geofileops.geofile import GeometryType
from geofileops.util.general_util import MissingRuntimeDependencyError 
from geofileops.util import geofileops_sql
from tests import test_helper

def test_buffer_gpkg(tmpdir):
    # Buffer to test dir
    input_path = test_helper.TestFiles.polygons_parcels_gpkg
    output_path = Path(tmpdir) / 'polygons_parcels-output.gpkg'
    basetest_buffer(input_path, output_path)

    # Buffer point source to test dir
    input_path = test_helper.TestFiles.points_gpkg
    output_path = Path(tmpdir) / 'points-output.gpkg'
    basetest_buffer(input_path, output_path)

    # Buffer line source to test dir
    input_path = test_helper.TestFiles.linestrings_rows_of_trees_gpkg
    output_path = Path(tmpdir) / 'linestrings_rows_of_trees-output.gpkg'
    basetest_buffer(input_path, output_path)

def test_buffer_shp(tmpdir):
    # Buffer to test dir
    input_path = test_helper.TestFiles.polygons_parcels_shp
    output_path = Path(tmpdir) / 'polygons_parcels-output.shp'
    basetest_buffer(input_path, output_path)
    
def basetest_buffer(
        input_path: Path, 
        output_path: Path):
    
    # Do operation
    geofileops_sql.buffer(input_path=input_path, output_path=output_path, distance=1)

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
    input_path = test_helper.TestFiles.polygons_parcels_gpkg
    output_path = Path(tmpdir) / 'polygons_parcels-output.gpkg'
    basetest_convexhull(input_path, output_path)

def test_convexhull_shp(tmpdir):
    # Select some data from src to tmp file
    input_path = test_helper.TestFiles.polygons_parcels_shp
    output_path = Path(tmpdir) / 'polygons_parcels-output.shp'
    basetest_convexhull(input_path, output_path)

def basetest_convexhull(
        input_path: Path, 
        output_path: Path):
    
    # Do operation  
    geofileops_sql.convexhull(input_path=input_path, output_path=output_path)

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
    input_path = test_helper.TestFiles.polygons_parcels_gpkg
    output_path = Path(tmpdir) / 'polygons_parcels-output.gpkg'
    basetest_isvalid(input_path, output_path)

def test_isvalid_shp(tmpdir):
    # Select some data from src to tmp file
    input_path = test_helper.TestFiles.polygons_parcels_shp
    output_path = Path(tmpdir) / 'polygons_parcels-output.gpkg'
    basetest_isvalid(input_path, output_path)
    
def basetest_isvalid(
        input_path: Path, 
        output_path: Path):
    
    # Do operation
    geofileops_sql.isvalid(input_path=input_path, output_path=output_path, nb_parallel=2)

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
    input_path = test_helper.TestFiles.polygons_invalid_geometries_gpkg
    output_path = Path(tmpdir) / f"{input_path.stem}_valid-output.gpkg"
    basetest_makevalid(input_path, output_path)

def test_makevalid_shp(tmpdir):
    # makevalid to test dir
    input_path = test_helper.TestFiles.polygons_invalid_geometries_shp
    output_path = Path(tmpdir) / f"{input_path.stem}_valid-output.shp"
    basetest_makevalid(input_path, output_path)
           
def basetest_makevalid(
        input_path: Path, 
        output_path: Path):

    # Do operation
    geofileops_sql.makevalid(input_path=input_path, output_path=output_path, nb_parallel=2)

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
    output_isvalid_path = output_path.parent / f"{output_path.stem}_is-valid{output_path.suffix}"
    isvalid = geofileops_sql.isvalid(input_path=input_path, output_path=output_isvalid_path)
    assert isvalid is False, "Input file should contain invalid features"

    # Check if the result file is valid
    output_new_isvalid_path = output_path.parent / f"{output_path.stem}_new_is-valid{output_path.suffix}"
    isvalid = geofileops_sql.isvalid(input_path=output_path, output_path=output_new_isvalid_path)
    assert isvalid == True, "Output file shouldn't contain invalid features"

def test_select_gpkg(tmpdir):
    # Select some data from src to tmp file
    input_path = test_helper.TestFiles.polygons_parcels_gpkg
    output_path = Path(tmpdir) / 'polygons_parcels-output.gpkg'

    basetest_select(input_path, output_path)

def test_select_gpkg_to_shp(tmpdir):
    # Select some data from src to tmp file
    input_path = test_helper.TestFiles.polygons_parcels_gpkg
    output_path = Path(tmpdir) / 'polygons_parcels-output.shp'

    basetest_select(input_path, output_path)

def test_select_shp(tmpdir):
    # Select some data from src to tmp file
    input_path = test_helper.TestFiles.polygons_parcels_shp
    output_path = Path(tmpdir) / 'polygons_parcels-output.shp'

    basetest_select(input_path, output_path)

def test_select_shp_to_gpkg(tmpdir):
    # Select some data from src to tmp file
    input_path = test_helper.TestFiles.polygons_parcels_shp
    output_path = Path(tmpdir) / 'polygons_parcels-output.gpkg'

    basetest_select(input_path, output_path)

def basetest_select(
        input_path: Path, 
        output_path: Path):

    layerinfo_input = geofile.get_layerinfo(input_path)
    sql_stmt = 'SELECT {geometrycolumn}, oidn, uidn FROM "{input_layer}"'
    geofileops_sql.select(
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
    input_path = test_helper.TestFiles.polygons_parcels_shp
    output_path = Path(tmpdir) / 'polygons_parcels-output.shp'

    basetest_select_various_options(input_path, output_path)

def test_select_various_options_gpkg(tmpdir):
    # Select some data from src to tmp file
    input_path = test_helper.TestFiles.polygons_parcels_gpkg
    output_path = Path(tmpdir) / 'polygons_parcels-output.gpkg'

    basetest_select_various_options(input_path, output_path)
    
def basetest_select_various_options(
        input_path: Path, 
        output_path: Path):

    ### Check if columns parameter works (case insensitive) ###
    columns = ['OIDN', 'uidn', 'HFDTLT', 'lblhfdtlt', 'GEWASGROEP', 'lengte', 'OPPERVL']
    layerinfo_input = geofile.get_layerinfo(input_path)
    sql_stmt = '''SELECT {geometrycolumn}
                        {columns_to_select_str} 
                    FROM "{input_layer}"'''
    geofileops_sql.select(
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
    # Simplify polygon source to test dir
    input_path = test_helper.TestFiles.polygons_parcels_gpkg
    output_path = Path(tmpdir) / input_path.name
    basetest_simplify(input_path, output_path, 
            expected_output_geometrytype=GeometryType.MULTIPOLYGON)

    # Simplify point source to test dir
    input_path = test_helper.TestFiles.points_gpkg
    output_path = Path(tmpdir) / 'points-output.gpkg'
    basetest_simplify(input_path, output_path, 
            expected_output_geometrytype=GeometryType.MULTIPOINT)

    # Simplify line source to test dir
    input_path = test_helper.TestFiles.linestrings_rows_of_trees_gpkg
    output_path = Path(tmpdir) / 'linestrings_rows_of_trees-output.gpkg'
    basetest_simplify(input_path, output_path, 
            expected_output_geometrytype=GeometryType.MULTILINESTRING)

def test_simplify_shp(tmpdir):
    # Simplify to test dir
    input_path = test_helper.TestFiles.polygons_parcels_shp
    output_path = Path(tmpdir) / 'polygons_parcels-output.shp'
    basetest_simplify(input_path, output_path, 
            expected_output_geometrytype=GeometryType.MULTIPOLYGON)
    
def basetest_simplify(
        input_path: Path, 
        output_path: Path,
        expected_output_geometrytype: GeometryType):

    # Do operation
    geofileops_sql.simplify(
            input_path=input_path, output_path=output_path,
            tolerance=5)

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
    #test_convexhull_gpkg(tmpdir)
    #test_convexhull_shp(tmpdir)
    #test_select_geos_version(tmpdir)
    #test_simplify_gpkg(tmpdir)
    