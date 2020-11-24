# -*- coding: utf-8 -*-
"""
Tests for functionalities in geofileops.general.
"""

from pathlib import Path
import sys

# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))
from geofileops import gfo_general

def _get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / 'data'

def test_add_column(tmpdir):
    # First copy test file to tmpdir
    # Now add area column
    src = _get_testdata_dir() / 'parcels.gpkg'
    tmppath = Path(tmpdir) / 'parcels.gpkg'
    gfo_general.copy(src, tmppath)

    # The area column shouldn't be in the test file yet
    layerinfo = gfo_general.get_layerinfo(path=tmppath, layer='parcels')
    assert 'AREA' not in layerinfo.columns
        
    ### Add area column ###
    try: 
        import os
        os.environ['GDAL_BIN'] = r"X:\GIS\Software\_Progs\OSGeo4W64_2020-05-29\bin"
        gfo_general.add_column(tmppath, layer='parcels', name='AREA', type='real', expression='ST_area(geom)')
    finally:
        del os.environ['GDAL_BIN']
        
    layerinfo = gfo_general.get_layerinfo(path=tmppath, layer='parcels')
    assert 'AREA' in layerinfo.columns
    
    gdf = gfo_general.read_file(tmppath)
    assert round(gdf['AREA'][0], 1) == round(gdf['OPPERVL'][0], 1)

    ### Add invalid column type -> should raise an exception
    test_ok = False
    try: 
        gfo_general.add_column(tmppath, layer='parcels', name='joske', type='joske', expression='ST_area(geom)')
        test_ok = False
    except:
        test_ok = True
    assert test_ok is True

def test_cmp(tmpdir):
    # Copy test file to tmpdir
    src = _get_testdata_dir() / 'parcels.shp'
    dst = Path(tmpdir) / 'parcels_output.shp'
    gfo_general.copy(src, dst)

    # Now compare source and dst file
    assert gfo_general.cmp(src, dst) == True

def test_convert(tmpdir):
    # Convert test file to tmpdir
    src = _get_testdata_dir() / 'parcels.shp'
    dst = Path(tmpdir) / 'parcels_output.gpkg'
    gfo_general.convert(src, dst)

    # Now compare source and dst file 
    src_layerinfo = gfo_general.get_layerinfo(src)
    dst_layerinfo = gfo_general.get_layerinfo(dst)
    assert src_layerinfo.featurecount == dst_layerinfo.featurecount
    assert len(src_layerinfo.columns) == len(dst_layerinfo.columns)

def test_copy(tmpdir):
    # Copy test file to tmpdir
    src = _get_testdata_dir() / 'parcels.shp'
    dst = Path(tmpdir) / 'parcels_output.shp'
    gfo_general.copy(src, dst)
    assert dst.exists() == True

def test_get_crs():
    # Test shapefile
    srcpath = _get_testdata_dir() / 'parcels.shp'
    crs = gfo_general.get_crs(srcpath)
    assert crs.to_epsg() == 31370

    # Test geopackage
    srcpath = _get_testdata_dir() / 'parcels.gpkg'
    crs = gfo_general.get_crs(srcpath)
    assert crs.to_epsg() == 31370

def test_get_default_layer():
    srcpath = _get_testdata_dir() / 'parcels.shp'
    layer = gfo_general.get_default_layer(srcpath)
    assert layer == 'parcels'

def test_get_driver():
    # Test shapefile
    src = _get_testdata_dir() / 'parcels.shp'
    assert gfo_general.get_driver(src) == "ESRI Shapefile"

    # Test geopackage
    src = _get_testdata_dir() / 'parcels.gpkg'
    assert gfo_general.get_driver(src) == "GPKG"

def test_getlayerinfo():
    # Test shapefile
    srcpath = _get_testdata_dir() / 'parcels.shp'
    layerinfo = gfo_general.get_layerinfo(srcpath)
    assert layerinfo.featurecount == 46
    assert layerinfo.geometrycolumn == 'geometry' 

    # Test geopackage
    srcpath = _get_testdata_dir() / 'parcels.gpkg'
    layerinfo = gfo_general.get_layerinfo(srcpath, 'parcels')
    assert layerinfo.featurecount == 46
    assert layerinfo.geometrycolumn == 'geom' 

def test_get_only_layer():
    srcpath = _get_testdata_dir() / 'parcels.shp'
    layer = gfo_general.get_only_layer(srcpath)
    assert layer == 'parcels'

def test_is_gfo_general():
    # Test shapefile
    srcpath = _get_testdata_dir() / 'parcels.shp'
    assert gfo_general.is_geofile(srcpath) == True

    # Test geopackage
    srcpath = _get_testdata_dir() / 'parcels.gpkg'
    assert gfo_general.is_geofile(srcpath) == True
    
def test_listlayers():
    # Test shapefile
    srcpath = _get_testdata_dir() / 'parcels.shp'
    layers = gfo_general.listlayers(srcpath)
    assert layers[0] == 'parcels'
    
    # Test geopackage
    srcpath = _get_testdata_dir() / 'parcels.gpkg'
    layers = gfo_general.listlayers(srcpath)
    assert layers[0] == 'parcels'

def test_move(tmpdir):
    # Copy test file to tmpdir
    src = _get_testdata_dir() / 'parcels.shp'
    tmp1path = Path(tmpdir) / 'parcels_tmp.shp'
    gfo_general.copy(src, tmp1path)
    assert tmp1path.exists() == True

    # Move (rename actually) and check result
    tmp2path = Path(tmpdir) / 'parcels_tmp2.shp'
    gfo_general.move(tmp1path, tmp2path)
    assert tmp1path.exists() == False
    assert tmp2path.exists() == True

def test_update_column(tmpdir):
    # First copy test file to tmpdir
    # Now add area column
    src = _get_testdata_dir() / 'parcels.gpkg'
    tmppath = Path(tmpdir) / 'parcels.gpkg'
    gfo_general.copy(src, tmppath)

    # The area column shouldn't be in the test file yet
    layerinfo = gfo_general.get_layerinfo(path=tmppath, layer='parcels')
    assert 'area' not in layerinfo.columns
        
    ### Add area column ###
    try: 
        import os
        os.environ['GDAL_BIN'] = r"X:\GIS\Software\_Progs\OSGeo4W64_2020-05-29\bin"
        gfo_general.add_column(tmppath, layer='parcels', name='AREA', type='real', expression='ST_area(geom)')
        gfo_general.update_column(tmppath, name='AreA', expression='ST_area(geom)')
    finally:
        del os.environ['GDAL_BIN']
        
    layerinfo = gfo_general.get_layerinfo(path=tmppath, layer='parcels')
    assert 'AREA' in layerinfo.columns
    
    gdf = gfo_general.read_file(tmppath)
    assert round(gdf['AREA'][0], 1) == round(gdf['OPPERVL'][0], 1)

def test_read_file():
    # Test shapefile, with defaults
    srcpath = _get_testdata_dir() / 'parcels.shp'
    read_gdf = gfo_general.read_file(srcpath)
    assert len(read_gdf) == 46

    # Test shapefile, specific columns (+ test case insensitivity)
    srcpath = _get_testdata_dir() / 'parcels.shp'
    columns = ['OIDN', 'uidn', 'HFDTLT', 'lblhfdtlt', 'GEWASGROEP', 'lengte', 'OPPERVL']
    read_gdf = gfo_general.read_file(srcpath, columns=columns)
    assert len(read_gdf) == 46
    assert len(read_gdf.columns) == (len(columns) + 1)

    # Test geopackage, with defaults
    srcpath = _get_testdata_dir() / 'parcels.gpkg'
    read_gdf = gfo_general.read_file(srcpath)
    assert len(read_gdf) == 46

    # Test shapefile, specific columns (+ test case insensitivity)
    srcpath = _get_testdata_dir() / 'parcels.gpkg'
    columns = ['OIDN', 'uidn', 'HFDTLT', 'lblhfdtlt', 'GEWASGROEP', 'lengte', 'OPPERVL']
    read_gdf = gfo_general.read_file(srcpath, columns=columns)
    assert len(read_gdf) == 46
    assert len(read_gdf.columns) == (len(columns) + 1)

def test_rename_layer(tmpdir):
    # First copy test file to tmpdir
    src = _get_testdata_dir() / 'parcels.gpkg'
    tmppath = Path(tmpdir) / 'parcels_tmp.gpkg'
    gfo_general.copy(src, tmppath)

    # Now test rename layer
    gfo_general.rename_layer(tmppath, layer='parcels', new_layer='parcels_renamed')
    layernames_renamed = gfo_general.listlayers(path=tmppath)
    assert layernames_renamed[0] == 'parcels_renamed'

def test_spatial_index_gpkg(tmpdir):
    # First copy test file to tmpdir
    src = _get_testdata_dir() / 'parcels.gpkg'
    tmppath = Path(tmpdir) / 'parcels.gpkg'
    gfo_general.copy(src, tmppath)

    # Check if spatial index present
    has_spatial_index = gfo_general.has_spatial_index(
        path=tmppath, layer='parcels')
    assert has_spatial_index is True

    # Remove spatial index
    gfo_general.remove_spatial_index(path=tmppath, layer='parcels')
    has_spatial_index = gfo_general.has_spatial_index(
        path=tmppath, layer='parcels')
    assert has_spatial_index is False

    # Create spatial index
    gfo_general.create_spatial_index(path=tmppath, layer='parcels')
    has_spatial_index = gfo_general.has_spatial_index(
        path=tmppath, layer='parcels')
    assert has_spatial_index is True

def test_spatial_index_shp(tmpdir):
    # First copy test file to tmpdir
    src = _get_testdata_dir() / 'parcels.shp'
    tmppath = Path(tmpdir) / 'parcels.shp'
    gfo_general.copy(src, tmppath)

    # Check if spatial index present
    has_spatial_index = gfo_general.has_spatial_index(path=tmppath)
    assert has_spatial_index is True

    # Remove spatial index
    gfo_general.remove_spatial_index(path=tmppath)
    has_spatial_index = gfo_general.has_spatial_index(path=tmppath)
    assert has_spatial_index is False

    # Create spatial index
    gfo_general.create_spatial_index(path=tmppath)
    has_spatial_index = gfo_general.has_spatial_index(path=tmppath)
    assert has_spatial_index is True

def test_to_file_shp(tmpdir):
    # Read test file and write to tmpdir
    srcpath = _get_testdata_dir() / 'parcels.shp'
    read_gdf = gfo_general.read_file(srcpath)
    tmppath = Path(tmpdir) / 'parcels_tmp.shp'
    gfo_general.to_file(read_gdf, tmppath)
    tmp_gdf = gfo_general.read_file(tmppath)
    
    assert len(read_gdf) == len(tmp_gdf)

def test_to_file_gpkg(tmpdir):
    # Read test file and write to tmpdir
    srcpath = _get_testdata_dir() / 'parcels.gpkg'
    read_gdf = gfo_general.read_file(srcpath)
    tmppath = Path(tmpdir) / 'parcels_tmp.gpkg'
    gfo_general.to_file(read_gdf, tmppath)
    tmp_gdf = gfo_general.read_file(tmppath)
    
    assert len(read_gdf) == len(tmp_gdf)

def test_remove(tmpdir):
    # Copy test file to tmpdir
    src = _get_testdata_dir() / 'parcels.shp'
    tmppath = Path(tmpdir) / 'parcels_tmp.shp'
    gfo_general.copy(src, tmppath)
    assert tmppath.exists() == True

    # Remove and check result
    gfo_general.remove(tmppath)
    assert tmppath.exists() == False

if __name__ == '__main__':
    import tempfile
    
    tmpdir = tempfile.gettempdir()
    test_add_column(tmpdir)
