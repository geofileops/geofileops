# -*- coding: utf-8 -*-
"""
Tests for functionalities in geofileops.general.
"""

import os
from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd

# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geofileops import geofile

def _get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / 'data'

def test_add_column(tmpdir):
    # First copy test file to tmpdir
    # Now add area column
    src = _get_testdata_dir() / 'polygons_parcels.gpkg'
    tmppath = Path(tmpdir) / 'polygons_parcels.gpkg'
    geofile.copy(src, tmppath)

    # The area column shouldn't be in the test file yet
    layerinfo = geofile.get_layerinfo(path=tmppath, layer='parcels')
    assert 'AREA' not in layerinfo.columns
        
    ### Add area column ###
    try: 
        os.environ['GDAL_BIN'] = r"X:\GIS\Software\_Progs\OSGeo4W64_2020-05-29\bin"
        geofile.add_column(tmppath, layer='parcels', name='AREA', type='real', expression='ST_area(geom)')
    finally:
        del os.environ['GDAL_BIN']
        
    layerinfo = geofile.get_layerinfo(path=tmppath, layer='parcels')
    assert 'AREA' in layerinfo.columns
    
    gdf = geofile.read_file(tmppath)
    assert round(gdf['AREA'].astype('float')[0], 1) == round(gdf['OPPERVL'].astype('float')[0], 1)

    ### Add invalid column type -> should raise an exception
    test_ok = False
    try: 
        geofile.add_column(tmppath, layer='parcels', name='joske', type='joske', expression='ST_area(geom)')
        test_ok = False
    except:
        test_ok = True
    assert test_ok is True

def test_cmp(tmpdir):
    # Copy test file to tmpdir
    src = _get_testdata_dir() / 'polygons_parcels.shp'
    dst = Path(tmpdir) / 'polygons_parcels_output.shp'
    geofile.copy(src, dst)

    # Now compare source and dst file
    assert geofile.cmp(src, dst) == True

def test_convert(tmpdir):
    # Convert test file to tmpdir
    src = _get_testdata_dir() / 'polygons_parcels.shp'
    dst = Path(tmpdir) / 'polygons_parcels_output.gpkg'
    geofile.convert(src, dst)

    # Now compare source and dst file 
    src_layerinfo = geofile.get_layerinfo(src)
    dst_layerinfo = geofile.get_layerinfo(dst)
    assert src_layerinfo.featurecount == dst_layerinfo.featurecount
    assert len(src_layerinfo.columns) == len(dst_layerinfo.columns)

def test_copy(tmpdir):
    # Copy test file to tmpdir
    src = _get_testdata_dir() / 'polygons_parcels.shp'
    dst = Path(tmpdir) / 'polygons_parcels_output.shp'
    geofile.copy(src, dst)
    assert dst.exists() == True

def test_get_crs():
    # Test shapefile
    srcpath = _get_testdata_dir() / 'polygons_parcels.shp'
    crs = geofile.get_crs(srcpath)
    assert crs.to_epsg() == 31370

    # Test geopackage
    srcpath = _get_testdata_dir() / 'polygons_parcels.gpkg'
    crs = geofile.get_crs(srcpath)
    assert crs.to_epsg() == 31370

def test_get_default_layer():
    srcpath = _get_testdata_dir() / 'polygons_parcels.shp'
    layer = geofile.get_default_layer(srcpath)
    assert layer == 'polygons_parcels'

def test_get_driver():
    # Test shapefile
    src = _get_testdata_dir() / 'polygons_parcels.shp'
    assert geofile.get_driver(src) == "ESRI Shapefile"

    # Test geopackage
    src = _get_testdata_dir() / 'polygons_parcels.gpkg'
    assert geofile.get_driver(src) == "GPKG"

def test_getlayerinfo():
    # Test shapefile
    srcpath = _get_testdata_dir() / 'polygons_parcels.shp'
    layerinfo = geofile.get_layerinfo(srcpath)
    assert layerinfo.featurecount == 46
    assert layerinfo.geometrycolumn == 'geometry' 

    # Test geopackage
    srcpath = _get_testdata_dir() / 'polygons_parcels.gpkg'
    layerinfo = geofile.get_layerinfo(srcpath, 'parcels')
    assert layerinfo.featurecount == 46
    assert layerinfo.geometrycolumn == 'geom' 

def test_get_only_layer():
    srcpath = _get_testdata_dir() / 'polygons_parcels.shp'
    layer = geofile.get_only_layer(srcpath)
    assert layer == 'polygons_parcels'

    srcpath = _get_testdata_dir() / 'polygons_parcels.gpkg'
    layer = geofile.get_only_layer(srcpath)
    assert layer == 'parcels'

def test_is_geofile():
    # Test shapefile
    srcpath = _get_testdata_dir() / 'polygons_parcels.shp'
    assert geofile.is_geofile(srcpath) == True

    # Test geopackage
    srcpath = _get_testdata_dir() / 'polygons_parcels.gpkg'
    assert geofile.is_geofile(srcpath) == True
    
def test_listlayers():
    # Test shapefile
    srcpath = _get_testdata_dir() / 'polygons_parcels.shp'
    layers = geofile.listlayers(srcpath)
    assert layers[0] == 'polygons_parcels'
    
    # Test geopackage
    srcpath = _get_testdata_dir() / 'polygons_parcels.gpkg'
    layers = geofile.listlayers(srcpath)
    assert layers[0] == 'parcels'

def test_move(tmpdir):
    # Copy test file to tmpdir
    src = _get_testdata_dir() / 'polygons_parcels.shp'
    tmp1path = Path(tmpdir) / 'polygons_parcels_tmp.shp'
    geofile.copy(src, tmp1path)
    assert tmp1path.exists() == True

    # Move (rename actually) and check result
    tmp2path = Path(tmpdir) / 'polygons_parcels_tmp2.shp'
    geofile.move(tmp1path, tmp2path)
    assert tmp1path.exists() == False
    assert tmp2path.exists() == True

def test_update_column(tmpdir):
    # First copy test file to tmpdir
    # Now add area column
    src = _get_testdata_dir() / 'polygons_parcels.gpkg'
    tmppath = Path(tmpdir) / 'polygons_parcels.gpkg'
    geofile.copy(src, tmppath)

    # The area column shouldn't be in the test file yet
    layerinfo = geofile.get_layerinfo(path=tmppath, layer='parcels')
    assert 'area' not in layerinfo.columns
        
    ### Add area column ###
    try: 
        os.environ['GDAL_BIN'] = r"X:\GIS\Software\_Progs\OSGeo4W64_2020-05-29\bin"
        geofile.add_column(tmppath, layer='parcels', name='AREA', type='real', expression='ST_area(geom)')
        geofile.update_column(tmppath, name='AreA', expression='ST_area(geom)')
    finally:
        del os.environ['GDAL_BIN']
        
    layerinfo = geofile.get_layerinfo(path=tmppath, layer='parcels')
    assert 'AREA' in layerinfo.columns
    gdf = geofile.read_file(tmppath)
    assert round(gdf['AREA'].astype('float')[0], 1) == round(gdf['OPPERVL'].astype('float')[0], 1)

def test_read_file():
    # Test shapefile
    srcpath = _get_testdata_dir() / 'polygons_parcels.shp'
    basetest_read_file(srcpath=srcpath)

    # Test geopackage
    srcpath = _get_testdata_dir() / 'polygons_parcels.gpkg'
    basetest_read_file(srcpath=srcpath)

def basetest_read_file(srcpath: Path):
    # Test with defaults
    read_gdf = geofile.read_file(srcpath)
    assert isinstance(read_gdf, gpd.GeoDataFrame)
    assert len(read_gdf) == 46

    # Test no columns
    read_gdf = geofile.read_file(srcpath, columns=[])
    assert isinstance(read_gdf, gpd.GeoDataFrame)
    assert len(read_gdf) == 46

    # Test specific columns (+ test case insensitivity)
    columns = ['OIDN', 'uidn', 'HFDTLT', 'lblhfdtlt', 'GEWASGROEP', 'lengte', 'OPPERVL']
    read_gdf = geofile.read_file(srcpath, columns=columns)
    assert len(read_gdf) == 46
    assert len(read_gdf.columns) == (len(columns) + 1)

    # Test no geom
    read_gdf = geofile.read_file_nogeom(srcpath)
    assert isinstance(read_gdf, pd.DataFrame)
    assert len(read_gdf) == 46

    # Test ignore_geometry, no columns
    read_gdf = geofile.read_file_nogeom(srcpath, columns=[])
    assert isinstance(read_gdf, pd.DataFrame)
    assert len(read_gdf) == 46

def test_rename_layer(tmpdir):
    # First copy test file to tmpdir
    src = _get_testdata_dir() / 'polygons_parcels.gpkg'
    tmppath = Path(tmpdir) / 'polygons_parcels_tmp.gpkg'
    geofile.copy(src, tmppath)

    # Now test rename layer
    geofile.rename_layer(tmppath, layer='parcels', new_layer='parcels_renamed')
    layernames_renamed = geofile.listlayers(path=tmppath)
    assert layernames_renamed[0] == 'parcels_renamed'

def test_spatial_index_gpkg(tmpdir):
    # First copy test file to tmpdir
    src = _get_testdata_dir() / 'polygons_parcels.gpkg'
    tmppath = Path(tmpdir) / 'polygons_parcels.gpkg'
    geofile.copy(src, tmppath)

    # Check if spatial index present
    has_spatial_index = geofile.has_spatial_index(
        path=tmppath, layer='parcels')
    assert has_spatial_index is True

    # Remove spatial index
    geofile.remove_spatial_index(path=tmppath, layer='parcels')
    has_spatial_index = geofile.has_spatial_index(
        path=tmppath, layer='parcels')
    assert has_spatial_index is False

    # Create spatial index
    geofile.create_spatial_index(path=tmppath, layer='parcels')
    has_spatial_index = geofile.has_spatial_index(
        path=tmppath, layer='parcels')
    assert has_spatial_index is True

def test_spatial_index_shp(tmpdir):
    # First copy test file to tmpdir
    src = _get_testdata_dir() / 'polygons_parcels.shp'
    tmppath = Path(tmpdir) / 'polygons_parcels.shp'
    geofile.copy(src, tmppath)

    # Check if spatial index present
    has_spatial_index = geofile.has_spatial_index(path=tmppath)
    assert has_spatial_index is True

    # Remove spatial index
    geofile.remove_spatial_index(path=tmppath)
    has_spatial_index = geofile.has_spatial_index(path=tmppath)
    assert has_spatial_index is False

    # Create spatial index
    geofile.create_spatial_index(path=tmppath)
    has_spatial_index = geofile.has_spatial_index(path=tmppath)
    assert has_spatial_index is True

def test_to_file_shp(tmpdir):
    # Read test file and write to tmpdir
    srcpath = _get_testdata_dir() / 'polygons_parcels.shp'
    read_gdf = geofile.read_file(srcpath)
    tmppath = Path(tmpdir) / 'polygons_parcels_tmp.shp'
    geofile.to_file(read_gdf, tmppath)
    tmp_gdf = geofile.read_file(tmppath)
    
    assert len(read_gdf) == len(tmp_gdf)

def test_to_file_gpkg(tmpdir):
    # Read test file and write to tmpdir
    srcpath = _get_testdata_dir() / 'polygons_parcels.gpkg'
    read_gdf = geofile.read_file(srcpath)
    tmppath = Path(tmpdir) / 'polygons_parcels_tmp.gpkg'
    geofile.to_file(read_gdf, tmppath)
    tmp_gdf = geofile.read_file(tmppath)
    
    assert len(read_gdf) == len(tmp_gdf)

def test_remove(tmpdir):
    # Copy test file to tmpdir
    src = _get_testdata_dir() / 'polygons_parcels.shp'
    tmppath = Path(tmpdir) / 'polygons_parcels_tmp.shp'
    geofile.copy(src, tmppath)
    assert tmppath.exists() == True

    # Remove and check result
    geofile.remove(tmppath)
    assert tmppath.exists() == False

if __name__ == '__main__':
    import tempfile
    
    tmpdir = tempfile.gettempdir()
    test_listlayers()
    #test_add_column(tmpdir)
    #test_read_file()
