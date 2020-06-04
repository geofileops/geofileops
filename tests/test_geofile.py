
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))

from geofileops import geofile

def test_listlayers():
    script_dir = Path(__file__).resolve().parent

    layers = geofile.listlayers(script_dir / 'data' / 'parcels.shp')
    assert layers[0] == 'parcels'

    layers = geofile.listlayers(script_dir / 'data' / 'parcels.gpkg')
    assert layers[0] == 'parcels'

def test_getlayerinfo():
    script_dir = Path(__file__).resolve().parent
    
    layerinfo = geofile.getlayerinfo(script_dir / 'data' / 'parcels.shp', 'parcels')
    assert layerinfo.featurecount == 46
    assert layerinfo.geometrycolumn == 'geom' 

    layerinfo = geofile.getlayerinfo(script_dir / 'data' / 'parcels.gpkg', 'parcels')
    assert layerinfo.featurecount == 46
    assert layerinfo.geometrycolumn == 'geom' 

def test_copy(tmpdir):
    script_dir = Path(__file__).resolve().parent

    src = script_dir / 'data' / 'parcels.shp'
    dst = Path(tmpdir) / 'parcels.shp'
    geofile.copy(src, dst)

def test_spatial_index_gpkg(tmpdir):
    script_dir = Path(__file__).resolve().parent

    src = script_dir / 'data' / 'parcels.gpkg'
    tmppath = Path(tmpdir) / 'parcels.gpkg'

    # First copy test file to 
    geofile.copy(src, tmppath)

    # Check if spatial index present
    has_spatial_index = geofile.has_spatial_index(
        path=tmppath, layer='parcels')
    assert has_spatial_index == True

    # Remove spatial index
    geofile.remove_spatial_index(path=tmppath, layer='parcels')
    has_spatial_index = geofile.has_spatial_index(
        path=tmppath, layer='parcels')
    assert has_spatial_index == False

    # Create spatial index
    geofile.create_spatial_index(path=tmppath, layer='parcels')
    has_spatial_index = geofile.has_spatial_index(
        path=tmppath, layer='parcels')
    assert has_spatial_index == True

def test_spatial_index_shp(tmpdir):
    script_dir = Path(__file__).resolve().parent

    src = script_dir / 'data' / 'parcels.shp'

    # First copy test file to temp dir
    tmppath = Path(tmpdir) / 'parcels.shp'
    geofile.copy(src, tmppath)

    # Check if spatial index present
    has_spatial_index = geofile.has_spatial_index(path=tmppath)
    assert has_spatial_index == True

    # Remove spatial index
    geofile.remove_spatial_index(path=tmppath)
    has_spatial_index = geofile.has_spatial_index(path=tmppath)
    assert has_spatial_index == False

    # Create spatial index
    geofile.create_spatial_index(path=tmppath)
    has_spatial_index = geofile.has_spatial_index(path=tmppath)
    assert has_spatial_index == True

if __name__ == '__main__':
    import tempfile
    tmpdir = tempfile.gettempdir()
    test_spatial_index_shp(tmpdir)
