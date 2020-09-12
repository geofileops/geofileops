
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))

from geofileops import geofile

def get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / 'data'

def test_listlayers():
    # Test shapefile
    srcpath = get_testdata_dir() / 'parcels.shp'
    layers = geofile.listlayers(srcpath)
    assert layers[0] == 'parcels'
    
    # Test geopackage
    srcpath = get_testdata_dir() / 'parcels.gpkg'
    layers = geofile.listlayers(srcpath)
    assert layers[0] == 'parcels'

def test_getlayerinfo():
    # Test shapefile
    srcpath = get_testdata_dir() / 'parcels.shp'
    layerinfo = geofile.getlayerinfo(srcpath)
    assert layerinfo.featurecount == 46
    assert layerinfo.geometrycolumn == 'geometry' 

    # Test geopackage
    srcpath = get_testdata_dir() / 'parcels.gpkg'
    layerinfo = geofile.getlayerinfo(srcpath, 'parcels')
    assert layerinfo.featurecount == 46
    assert layerinfo.geometrycolumn == 'geom' 

def test_get_only_layer():
    srcpath = get_testdata_dir() / 'parcels.shp'
    layer = geofile.get_only_layer(srcpath)
    assert layer == 'parcels'

def test_get_default_layer():
    srcpath = get_testdata_dir() / 'parcels.shp'
    layer = geofile.get_default_layer(srcpath)
    assert layer == 'parcels'

def test_spatial_index_gpkg(tmpdir):
    # First copy test file to tmpdir
    src = get_testdata_dir() / 'parcels.gpkg'
    tmppath = Path(tmpdir) / 'parcels.gpkg'
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
    src = get_testdata_dir() / 'parcels.shp'
    tmppath = Path(tmpdir) / 'parcels.shp'
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

def test_rename_layer(tmpdir):
    # First copy test file to tmpdir
    src = get_testdata_dir() / 'parcels.gpkg'
    tmppath = Path(tmpdir) / 'parcels.gpkg'
    geofile.copy(src, tmppath)

    # Now test rename layer
    geofile.rename_layer(tmppath, layer='parcels', new_layer='parcels_renamed')
    layernames_renamed = geofile.listlayers(path=tmppath)
    assert layernames_renamed[0] == 'parcels_renamed'

def test_add_column(tmpdir):
    # First copy test file to tmpdir
    # Now add area column
    src = get_testdata_dir() / 'parcels.gpkg'
    tmppath = Path(tmpdir) / 'parcels.gpkg'
    geofile.copy(src, tmppath)

    # The area column shouldn't be in the test file yet
    layerinfo = geofile.getlayerinfo(path=tmppath, layer='parcels')
    assert 'area' not in layerinfo.columns
        
    # Add area column
    try: 
        import os
        os.environ['GDAL_BIN'] = r"X:\GIS\Software\_Progs\OSGeo4W64_2020-05-29\bin"
        geofile.add_column(tmppath, layer='parcels', name='area', type='real', expression='ST_area(geom)')
    finally:
        del os.environ['GDAL_BIN']
        
    layerinfo = geofile.getlayerinfo(path=tmppath, layer='parcels')
    assert 'area' in layerinfo.columns
    
    gdf = geofile.read_file(tmppath)
    assert round(gdf['area'][0], 1) == round(gdf['OPPERVL'][0], 1)

def test_read_file():
    # Test shapefile
    srcpath = get_testdata_dir() / 'parcels.shp'
    read_gdf = geofile.read_file(srcpath)
    assert len(read_gdf) == 46

    # Test geopackage
    srcpath = get_testdata_dir() / 'parcels.gpkg'
    read_gdf = geofile.read_file(srcpath)
    assert len(read_gdf) == 46

def test_to_file_shp(tmpdir):
    # Read test file and write to tmpdir
    srcpath = get_testdata_dir() / 'parcels.shp'
    read_gdf = geofile.read_file(srcpath)
    tmppath = Path(tmpdir) / 'parcels.shp'
    geofile.to_file(read_gdf, tmppath)
    tmp_gdf = geofile.read_file(tmppath)
    
    assert len(read_gdf) == len(tmp_gdf)

def test_to_file_gpkg(tmpdir):
    # Read test file and write to tmpdir
    srcpath = get_testdata_dir() / 'parcels.gpkg'
    read_gdf = geofile.read_file(srcpath)
    tmppath = Path(tmpdir) / 'parcels.gpkg'
    geofile.to_file(read_gdf, tmppath)
    tmp_gdf = geofile.read_file(tmppath)
    
    assert len(read_gdf) == len(tmp_gdf)

def test_get_crs():
    # Test shapefile
    srcpath = get_testdata_dir() / 'parcels.shp'
    crs = geofile.get_crs(srcpath)
    assert crs.to_epsg() == 31370

    # Test geopackage
    srcpath = get_testdata_dir() / 'parcels.gpkg'
    crs = geofile.get_crs(srcpath)
    assert crs.to_epsg() == 31370

def test_is_geofile():
    # Test shapefile
    srcpath = get_testdata_dir() / 'parcels.shp'
    assert geofile.is_geofile(srcpath) == True

    # Test geopackage
    srcpath = get_testdata_dir() / 'parcels.gpkg'
    assert geofile.is_geofile(srcpath) == True

def test_copy(tmpdir):
    # Copy test file to tmpdir
    src = get_testdata_dir() / 'parcels.shp'
    dst = Path(tmpdir) / 'parcels.shp'
    geofile.copy(src, dst)
    assert dst.exists() == True

def test_cmp(tmpdir):
    # Copy test file to tmpdir
    src = get_testdata_dir() / 'parcels.shp'
    dst = Path(tmpdir) / 'parcels.shp'
    geofile.copy(src, dst)

    # Now compare source and dst file
    assert geofile.cmp(src, dst) == True

def test_move(tmpdir):
    # Copy test file to tmpdir
    src = get_testdata_dir() / 'parcels.shp'
    tmp1path = Path(tmpdir) / 'parcels.shp'
    geofile.copy(src, tmp1path)
    assert tmp1path.exists() == True

    # Move (rename actually) and check result
    tmp2path = Path(tmpdir) / 'parcels2.shp'
    geofile.move(tmp1path, tmp2path)
    assert tmp1path.exists() == False
    assert tmp2path.exists() == True

def test_remove(tmpdir):
    # Copy test file to tmpdir
    src = get_testdata_dir() / 'parcels.shp'
    tmppath = Path(tmpdir) / 'parcels.shp'
    geofile.copy(src, tmppath)
    assert tmppath.exists() == True

    # Remove and check result
    geofile.remove(tmppath)
    assert tmppath.exists() == False

def test_get_driver():
    # Test shapefile
    src = get_testdata_dir() / 'parcels.shp'
    assert geofile.get_driver(src) == "ESRI Shapefile"

    # Test geopackage
    src = get_testdata_dir() / 'parcels.gpkg'
    assert geofile.get_driver(src) == "GPKG"

if __name__ == '__main__':
    import tempfile
    
    tmpdir = tempfile.gettempdir()
    test_add_column(tmpdir)
