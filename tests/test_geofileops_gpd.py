# -*- coding: utf-8 -*-
"""
Tests for operations using GeoPandas.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))

from geofileops import geofile
from geofileops.util import geofileops_gpd

def get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / 'data'

def test_buffer_gpkg(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels_output.gpkg'
    basetest_buffer(input_path, output_path)

def test_buffer_shp(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels_output.shp'
    basetest_buffer(input_path, output_path)

def basetest_buffer(input_path, output_path):
    layerinfo_orig = geofile.get_layerinfo(input_path)
    geofileops_gpd.buffer(
            input_path=input_path,
            output_path=output_path,
            distance=1)

    # Now check if the output file is correctly created
    assert output_path.exists() == True
    layerinfo_output = geofile.get_layerinfo(input_path)
    assert layerinfo_orig.featurecount == layerinfo_output.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)
    
    # Check geometry type
    assert layerinfo_output.geometrytypename == 'MULTIPOLYGON' 

    # Read result for some more detailed checks
    output_gdf = geofile.read_file(output_path)
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
            distance=1)

    # Now check if the tmp file is correctly created
    layerinfo_orig = geofile.get_layerinfo(input_path)
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_orig.featurecount == layerinfo_output.featurecount
    assert 'OIDN' in layerinfo_output.columns
    assert 'UIDN' in layerinfo_output.columns
    assert len(layerinfo_output.columns) == len(columns)

    # Read result for some more detailed checks
    output_gdf = geofile.read_file(output_path)
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
    layerinfo_orig = geofile.get_layerinfo(input_path)
    geofileops_gpd.convexhull(
            input_path=input_path,
            output_path=output_path)

    # Now check if the output file is correctly created
    assert output_path.exists() == True
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_orig.featurecount == layerinfo_output.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)

    # Check geometry type
    assert layerinfo_output.geometrytypename == 'MULTIPOLYGON' 

    # Read result for some more detailed checks
    output_gdf = geofile.read_file(output_path)
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
    layerinfo_orig = geofile.get_layerinfo(input_path)
    geofileops_gpd.dissolve(
            input_path=input_path,
            output_path=output_path,
            groupby_columns=['GEWASGROEP'],
            explodecollections=False)

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 3
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)

    # Check geometry type
    assert layerinfo_output.geometrytypename == 'MULTIPOLYGON' 

    # Now check the contents of the result file
    input_gdf = geofile.read_file(input_path)
    output_gdf = geofile.read_file(output_path)
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
            explodecollections=True)

    # Now check if the result file is correctly created
    assert output_path.exists() == True
    layerinfo_orig = geofile.get_layerinfo(input_path)
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 21
    assert len(layerinfo_output.columns) >= 0

    # Check geometry type
    assert layerinfo_output.geometrytypename == 'MULTIPOLYGON' 

    # Now check the contents of the result file
    input_gdf = geofile.read_file(input_path)
    output_gdf = geofile.read_file(output_path)
    assert input_gdf.crs == output_gdf.crs
    assert len(output_gdf) == layerinfo_output.featurecount
    assert output_gdf['geometry'][0] is not None

def test_simplify_gpkg(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels_output.gpkg'
    basetest_simplify(input_path, output_path)

def test_simplify_shp(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels_output.shp'
    basetest_simplify(input_path, output_path)

def basetest_simplify(input_path, output_path):
    layerinfo_orig = geofile.get_layerinfo(input_path)
    geofileops_gpd.simplify(
            input_path=input_path,
            output_path=output_path,
            tolerance=5)

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_output = geofile.get_layerinfo(input_path)
    assert layerinfo_orig.featurecount == layerinfo_output.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)

    # Check geometry type
    assert layerinfo_output.geometrytypename == 'MULTIPOLYGON' 

    # Now check the contents of the result file
    input_gdf = geofile.read_file(input_path)
    output_gdf = geofile.read_file(output_path)
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
    test_buffer_various_options_gpkg(tmpdir)
    #test_dissolve_nogroupby_shp(tmpdir)
    #test_dissolve_nogroupby_gpkg(tmpdir)
