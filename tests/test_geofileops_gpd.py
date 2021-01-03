# -*- coding: utf-8 -*-
"""
Tests for operations using GeoPandas.
"""

from pathlib import Path
import sys

# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))
from geofileops import gfo_general
from geofileops.util import geofileops_gpd

def get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / 'data'

def get_nb_parallel() -> int:
    # The number of parallel processes to use for these tests.
    return 2

def test_buffer_gpkg(tmpdir):
    # Buffer polygon source to test dir
    input_path = get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels_output.gpkg'
    basetest_buffer(input_path, output_path, 'MULTIPOLYGON')

    # Buffer point source to test dir
    input_path = get_testdata_dir() / 'points.gpkg'
    output_path = Path(tmpdir) / 'points_output.gpkg'
    basetest_buffer(input_path, output_path, 'MULTIPOINT')

    # Buffer line source to test dir
    input_path = get_testdata_dir() / 'rows_of_trees.gpkg'
    output_path = Path(tmpdir) / 'rows_of_trees_output.gpkg'
    basetest_buffer(input_path, output_path, 'MULTILINESTRING')

def test_buffer_shp(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels_output.shp'
    basetest_buffer(input_path, output_path, 'MULTIPOLYGON')

def basetest_buffer(input_path, output_path, input_geometry_type):
    layerinfo_orig = gfo_general.get_layerinfo(input_path)
    
    ### Test positive buffer ###
    geofileops_gpd.buffer(
            input_path=input_path,
            output_path=output_path,
            distance=1,
            nb_parallel=get_nb_parallel())

    # Now check if the output file is correctly created
    assert output_path.exists() == True
    layerinfo_output = gfo_general.get_layerinfo(output_path)
    assert layerinfo_orig.featurecount == layerinfo_output.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)
    
    # Check geometry type
    assert layerinfo_output.geometrytypename == 'MULTIPOLYGON' 

    # Read result for some more detailed checks
    output_gdf = gfo_general.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

    ### Test negative buffer ###
    output_path = output_path.parent / f"{output_path.stem}_m10m{output_path.suffix}"
    geofileops_gpd.buffer(
            input_path=input_path,
            output_path=output_path,
            distance=-10,
            nb_parallel=get_nb_parallel())

    # Now check if the output file is correctly created
    if input_geometry_type in ['MULTIPOINT', 'MULTILINESTRING']:
        # A Negative buffer of points or linestrings doesn't give a result.
        assert output_path.exists() == False
    else:    
        # A Negative buffer of polygons  gives a result for large polygons.
        assert output_path.exists() == True
        layerinfo_output = gfo_general.get_layerinfo(output_path)
        assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)
        assert layerinfo_output.featurecount == 39
        
        # Check geometry type
        assert layerinfo_output.geometrytypename == 'MULTIPOLYGON' 

        # Read result for some more detailed checks
        output_gdf = gfo_general.read_file(output_path)
        assert output_gdf['geometry'][0] is not None

def test_buffer_various_options_gpkg(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels_output.gpkg'
    basetest_buffer_various_options(input_path, output_path)

def test_buffer_various_options_shp(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels_output.shp'
    basetest_buffer_various_options(input_path, output_path)

def basetest_buffer_various_options(input_path, output_path):

    ### Check if columns parameter works (case insensitive) ###
    columns = ['OIDN', 'uidn', 'HFDTLT', 'lblhfdtlt', 'GEWASGROEP', 'lengte', 'OPPERVL']
    geofileops_gpd.buffer(
            input_path=input_path,
            columns=columns,
            output_path=output_path,
            distance=1,
            nb_parallel=get_nb_parallel())

    # Now check if the tmp file is correctly created
    layerinfo_orig = gfo_general.get_layerinfo(input_path)
    layerinfo_output = gfo_general.get_layerinfo(output_path)
    assert layerinfo_orig.featurecount == layerinfo_output.featurecount
    assert 'OIDN' in layerinfo_output.columns
    assert 'UIDN' in layerinfo_output.columns
    assert len(layerinfo_output.columns) == len(columns)

    # Read result for some more detailed checks
    output_gdf = gfo_general.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

    ### Check if ... parameter works ###
    # TODO: increase test coverage of other options...

def test_convexhull_gpkg(tmpdir):
    # Select some data from input to output file
    input_path = get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels_output.gpkg'
    basetest_convexhull(input_path, output_path)

def test_convexhull_shp(tmpdir):
    # Select some data from input to output file
    input_path = get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels_output.shp'
    basetest_convexhull(input_path, output_path)

def basetest_convexhull(input_path, output_path):
    layerinfo_orig = gfo_general.get_layerinfo(input_path)
    geofileops_gpd.convexhull(
            input_path=input_path,
            output_path=output_path,
            nb_parallel=get_nb_parallel())

    # Now check if the output file is correctly created
    assert output_path.exists() == True
    layerinfo_output = gfo_general.get_layerinfo(output_path)
    assert layerinfo_orig.featurecount == layerinfo_output.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)

    # Check geometry type
    assert layerinfo_output.geometrytypename == 'MULTIPOLYGON' 

    # Read result for some more detailed checks
    output_gdf = gfo_general.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_dissolve_groupby_gpkg(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels_output.gpkg'
    basetest_dissolve_groupby(input_path, output_path)

def test_dissolve_groupby_shp(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels_output.shp'
    basetest_dissolve_groupby(input_path, output_path)

def basetest_dissolve_groupby(input_path, output_path):
    layerinfo_orig = gfo_general.get_layerinfo(input_path)

    # Test dissolve without explodecollections
    geofileops_gpd.dissolve(
            input_path=input_path,
            output_path=output_path,
            groupby_columns=['GEWASGROEP'],
            explodecollections=False,
            nb_parallel=get_nb_parallel())

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_output = gfo_general.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 3
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)

    # Check geometry type
    assert layerinfo_output.geometrytypename == 'MULTIPOLYGON' 

    # Now check the contents of the result file
    input_gdf = gfo_general.read_file(input_path)
    output_gdf = gfo_general.read_file(output_path)
    assert input_gdf.crs == output_gdf.crs
    assert len(output_gdf) == layerinfo_output.featurecount
    assert output_gdf['geometry'][0] is not None

    # Test dissolve with explodecollections
    geofileops_gpd.dissolve(
            input_path=input_path,
            output_path=output_path,
            groupby_columns=['GEWASGROEP'],
            explodecollections=True,
            nb_parallel=get_nb_parallel())

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_output = gfo_general.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 3
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)

    # Check geometry type
    assert layerinfo_output.geometrytypename == 'MULTIPOLYGON' 

    # Now check the contents of the result file
    input_gdf = gfo_general.read_file(input_path)
    output_gdf = gfo_general.read_file(output_path)
    assert input_gdf.crs == output_gdf.crs
    assert len(output_gdf) == layerinfo_output.featurecount
    assert output_gdf['geometry'][0] is not None

def test_dissolve_nogroupby_gpkg(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels_output.gpkg'
    basetest_dissolve_nogroupby(input_path, output_path)

def test_dissolve_nogroupby_shp(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels_output.shp'
    basetest_dissolve_nogroupby(input_path, output_path)

def basetest_dissolve_nogroupby(input_path, output_path):
    geofileops_gpd.dissolve(
            input_path=input_path,
            output_path=output_path,
            explodecollections=True,
            nb_parallel=get_nb_parallel())

    # Now check if the result file is correctly created
    assert output_path.exists() == True
    layerinfo_orig = gfo_general.get_layerinfo(input_path)
    layerinfo_output = gfo_general.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 21
    assert len(layerinfo_output.columns) >= 0

    # Check geometry type
    assert layerinfo_output.geometrytypename == 'MULTIPOLYGON' 

    # Now check the contents of the result file
    input_gdf = gfo_general.read_file(input_path)
    output_gdf = gfo_general.read_file(output_path)
    assert input_gdf.crs == output_gdf.crs
    assert len(output_gdf) == layerinfo_output.featurecount
    assert output_gdf['geometry'][0] is not None

def test_simplify_gpkg(tmpdir):
    # Simplify polygon source to test dir
    input_path = get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels_output.gpkg'
    basetest_simplify(input_path, output_path, 'MULTIPOLYGON')

    # Simplify point source to test dir
    input_path = get_testdata_dir() / 'points.gpkg'
    output_path = Path(tmpdir) / 'points_output.gpkg'
    basetest_simplify(input_path, output_path, 'MULTIPOINT')

    # Simplify line source to test dir
    input_path = get_testdata_dir() / 'rows_of_trees.gpkg'
    output_path = Path(tmpdir) / 'rows_of_trees_output.gpkg'
    basetest_simplify(input_path, output_path, 'MULTILINESTRING')

def test_simplify_shp(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels_output.shp'
    basetest_simplify(input_path, output_path, 'MULTIPOLYGON')

def basetest_simplify(
        input_path: Path, 
        output_path: Path, 
        expected_output_geometrytype: str):
    layerinfo_orig = gfo_general.get_layerinfo(input_path)
    geofileops_gpd.simplify(
            input_path=input_path,
            output_path=output_path,
            tolerance=5,
            nb_parallel=get_nb_parallel())

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_output = gfo_general.get_layerinfo(output_path)
    assert layerinfo_orig.featurecount == layerinfo_output.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)

    # Check geometry type
    assert layerinfo_output.geometrytypename == expected_output_geometrytype

    # Now check the contents of the result file
    input_gdf = gfo_general.read_file(input_path)
    output_gdf = gfo_general.read_file(output_path)
    assert input_gdf.crs == output_gdf.crs
    assert len(output_gdf) == layerinfo_output.featurecount
    assert output_gdf['geometry'][0] is not None

if __name__ == '__main__':
    #Prepare tempdir
    import tempfile
    import shutil
    tmpdir = Path(tempfile.gettempdir()) / 'test_geofileops_gpd'
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    if not tmpdir.exists():
        tmpdir.mkdir()

    # Run
    test_buffer_gpkg(tmpdir)
    #test_buffer_various_options_gpkg(tmpdir)
    #test_dissolve_nogroupby_shp(tmpdir)
    #test_dissolve_nogroupby_gpkg(tmpdir)
